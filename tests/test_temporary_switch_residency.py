import copy
import unittest
from unittest import mock

import torch
from torch import nn

from e2e_common.temporary_switch_linear import TemporarySwitchLinear
from litebsq.autoencoder import Decoder
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from train_utils import cat_checkpoint_distill as ccd
from train_utils.cat_checkpoint_distill import (
    _CheckpointDistillResidency,
    _ensure_bank_param,
    _stash_vae_module_to_cpu,
)


class _FakeVAELinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.temporary = True
        self.original_weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device),
            requires_grad=False,
        )
        self.bias = None

    def clear_decoded_weight_cache(self) -> None:
        return None


def _make_test_vae_linear() -> VAELinear:
    decoder = Decoder(
        in_dim=4,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)
    with torch.no_grad():
        for index, parameter in enumerate(decoder.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).view_as(parameter)
            parameter.copy_(values / float(parameter.numel() + 1) + float(index + 1) * 0.01)

    vq_weight = torch.tensor(
        [
            [[True, False, True, False]],
            [[False, True, False, True]],
            [[True, True, False, False]],
            [[False, False, True, True]],
        ],
        dtype=torch.bool,
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=vq_weight,
        decoder=copy.deepcopy(decoder),
        codebook_dim=4,
        transpose=False,
    )


class _NestedProjectionModel(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.layer0 = nn.Module()
        self.layer0.q_proj = module


class TemporarySwitchLinearTests(unittest.TestCase):
    def test_set_temporary_switches_weights(self):
        student = torch.ones(2, 3)
        teacher = torch.full((2, 3), 2.0)
        module = TemporarySwitchLinear(3, 2, student, teacher)
        x = torch.ones(4, 3)
        module.set_temporary(True)
        out_student = module(x)
        module.set_temporary(False)
        out_teacher = module(x)
        self.assertTrue(torch.allclose(out_student, x @ student.T))
        self.assertTrue(torch.allclose(out_teacher, x @ teacher.T))


class ResidencyBankTests(unittest.TestCase):
    def test_bank_shared_and_vae_cpu_does_not_move_bank(self):
        device = torch.device("cpu")
        module = _FakeVAELinear(8, 8, device)
        residency = _CheckpointDistillResidency()
        name = "layer0.q_proj"
        bank = _ensure_bank_param(
            residency,
            name=name,
            source=module.original_weight,
            device=device,
        )
        module.original_weight = bank
        marker = bank.data_ptr()

        old_resolve = ccd._resolve_vae_base_layer
        ccd._resolve_vae_base_layer = lambda mod: mod
        try:
            _stash_vae_module_to_cpu(name=name, module=module, residency=residency)
        finally:
            ccd._resolve_vae_base_layer = old_resolve

        holder = nn.Module()
        holder.proj = TemporarySwitchLinear(
            8,
            8,
            student_weight=torch.zeros(8, 8),
            teacher_weight=bank,
        )
        self.assertIs(residency.original_weight_bank[name], bank)
        self.assertEqual(int(bank.data_ptr()), int(marker))
        self.assertIs(holder.proj.teacher_weight, bank)
        self.assertIsNone(residency.stashed_modules[name].original_weight)

    def test_frozen_linear_shares_bank_parameter_object(self):
        device = torch.device("cpu")
        residency = _CheckpointDistillResidency()
        bank = _ensure_bank_param(
            residency,
            name="layer0.q_proj",
            source=torch.randn(4, 4, device=device),
            device=device,
        )
        linear = ccd._make_frozen_linear_from_bank(
            name="layer0.q_proj",
            bank_weight=bank,
            bias=None,
            in_features=4,
            out_features=4,
        )
        self.assertIs(linear.weight, bank)


class FinalVAERestorationTests(unittest.TestCase):
    def test_restore_final_vae_representation_activates_compressed_path(self):
        name = "layer0.q_proj"
        vae_module = _make_test_vae_linear()
        decoded_before_stash = vae_module._decode_weight(dtype=torch.float32).detach().clone()
        teacher_weight = nn.Parameter(torch.full((4, 4), 7.0), requires_grad=False)
        residency = _CheckpointDistillResidency(
            stashed_modules={name: vae_module},
            original_weight_bank={name: teacher_weight},
            managed_shapes={name: (4, 4)},
        )
        switch = TemporarySwitchLinear(
            in_features=4,
            out_features=4,
            student_weight=decoded_before_stash,
            teacher_weight=teacher_weight,
        )
        model = _NestedProjectionModel(switch)
        logger = mock.Mock()

        restored_targets = ccd._restore_final_vae_representation(
            model=model,
            residency=residency,
            completed_categories=["q_proj"],
            logger=logger,
        )

        restored = model.layer0.q_proj
        self.assertIsInstance(restored, VAELinear)
        self.assertTrue(restored.temporary)
        self.assertIs(restored.original_weight, teacher_weight)
        self.assertEqual(len(residency.stashed_modules), 0)
        self.assertEqual([target.name for target in restored_targets], [name])
        self.assertTrue(
            torch.allclose(
                restored._decode_weight(dtype=torch.float32),
                decoded_before_stash,
                rtol=0.0,
                atol=1e-6,
            )
        )

        x = torch.tensor([[1.0, -1.0, 0.5, 2.0]], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(
                restored(x),
                x @ decoded_before_stash.T,
                rtol=0.0,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()

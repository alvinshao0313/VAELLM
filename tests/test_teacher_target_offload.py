import unittest
from unittest import mock

import torch
from torch import nn

from compressed_e2e_fintuning.offload import OffloadedCheckpointLayer
from compressed_e2e_fintuning.teacher_targets import (
    TeacherTargetBatch,
    copy_detached_tensor_to_cpu,
    copy_teacher_logit_chunk_to_device,
    extract_primary_hidden,
    iter_token_chunk_ranges,
    resolve_transformer_layers,
)


class _ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


class _BaseModelWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self._base = base

    def get_base_model(self) -> nn.Module:
        return self._base


class _FakeOffloadManager:
    def __init__(self) -> None:
        self.registered = []

    def register(self, wrapper) -> None:
        self.registered.append(wrapper)


class _TinyModel(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(4, 4, bias=False) for _ in range(num_layers)]
        )


class TeacherTargetOffloadUtilityTest(unittest.TestCase):
    def test_copy_detached_tensor_to_cpu_preserves_values_and_removes_grad(self):
        source = torch.randn(2, 3, requires_grad=True)
        if torch.cuda.is_available():
            source = source.cuda()

        copied = copy_detached_tensor_to_cpu(source, pin_memory=False)

        self.assertEqual(copied.device.type, "cpu")
        self.assertFalse(copied.requires_grad)
        self.assertEqual(copied.dtype, source.dtype)
        self.assertTrue(torch.equal(copied, source.detach().cpu()))

    def test_copy_detached_tensor_to_cpu_from_cpu_input(self):
        source = torch.randn(2, 3)
        copied = copy_detached_tensor_to_cpu(source, pin_memory=False)
        self.assertEqual(copied.device.type, "cpu")
        self.assertFalse(copied.requires_grad)
        self.assertTrue(torch.equal(copied, source))

    def test_iter_token_chunk_ranges_L11_chunk4(self):
        ranges = list(iter_token_chunk_ranges(11, 4))
        self.assertEqual(ranges, [(0, 4), (4, 8), (8, 11)])

    def test_iter_token_chunk_ranges_invalid_raises(self):
        with self.assertRaises(ValueError):
            list(iter_token_chunk_ranges(0, 4))
        with self.assertRaises(ValueError):
            list(iter_token_chunk_ranges(11, 0))

    def test_copy_teacher_logit_chunk_to_device_respects_range(self):
        logits_cpu = torch.randn(2, 11, 32)
        target_device = torch.device("cpu")
        if torch.cuda.is_available():
            target_device = torch.device("cuda")

        chunk = copy_teacher_logit_chunk_to_device(
            logits_cpu,
            start=4,
            end=8,
            target_device=target_device,
        )

        self.assertEqual(tuple(chunk.shape), (2, 4, 32))
        self.assertTrue(torch.equal(chunk.cpu(), logits_cpu[:, 4:8, :]))

    def test_copy_teacher_logit_chunk_to_device_invalid_range_raises(self):
        logits_cpu = torch.randn(2, 11, 32)
        with self.assertRaises(ValueError):
            copy_teacher_logit_chunk_to_device(
                logits_cpu,
                start=8,
                end=4,
                target_device=torch.device("cpu"),
            )

    def test_teacher_target_batch_clear(self):
        batch = TeacherTargetBatch(
            logits_cpu=torch.randn(2, 4, 8),
            eakld_gamma_cpu=torch.randn(2, 4),
            hidden_cpu_by_layer={0: torch.randn(2, 4, 16)},
            hidden_layer_indices=(0,),
            num_hidden_layers=1,
        )

        batch.clear()

        self.assertIsNone(batch.logits_cpu)
        self.assertIsNone(batch.eakld_gamma_cpu)
        self.assertEqual(batch.hidden_cpu_by_layer, {})
        self.assertEqual(batch.hidden_layer_indices, ())
        self.assertEqual(batch.num_hidden_layers, 0)

    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_resolve_transformer_layers_unwraps_wrappers(self, mock_get_layers):
        tiny_model = _TinyModel(num_layers=2)
        mock_get_layers.side_effect = lambda model: model.model.layers

        module_wrapped = _ModuleWrapper(tiny_model)
        resolved = resolve_transformer_layers(module_wrapped)
        self.assertEqual(len(resolved), 2)
        self.assertIs(resolved[0], tiny_model.model.layers[0])
        mock_get_layers.assert_called_once_with(tiny_model)

        mock_get_layers.reset_mock()
        base_wrapped = _BaseModelWrapper(tiny_model)
        resolved = resolve_transformer_layers(base_wrapped)
        self.assertEqual(len(resolved), 2)
        mock_get_layers.assert_called_once_with(tiny_model)

        mock_get_layers.reset_mock()
        double_wrapped = _ModuleWrapper(_BaseModelWrapper(tiny_model))
        resolved = resolve_transformer_layers(double_wrapped)
        self.assertEqual(len(resolved), 2)
        mock_get_layers.assert_called_once_with(tiny_model)

    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_resolve_transformer_layers_offloaded_layer(self, mock_get_layers):
        tiny_model = _TinyModel(num_layers=1)
        inner_layer = nn.Linear(4, 4, bias=False)
        manager = _FakeOffloadManager()
        wrapped_layer = OffloadedCheckpointLayer(
            layer=inner_layer,
            layer_idx=0,
            manager=manager,
        )
        tiny_model.model.layers = nn.ModuleList([wrapped_layer])
        mock_get_layers.side_effect = lambda model: model.model.layers

        resolved = resolve_transformer_layers(tiny_model)
        self.assertEqual(resolved, (inner_layer,))

    def test_extract_primary_hidden_tensor(self):
        tensor = torch.randn(2, 3)
        self.assertIs(extract_primary_hidden(tensor, context="test"), tensor)

    def test_extract_primary_hidden_tuple(self):
        tensor = torch.randn(2, 3)
        self.assertIs(extract_primary_hidden((tensor, torch.randn(2, 3)), context="test"), tensor)

    def test_extract_primary_hidden_invalid_raises(self):
        with self.assertRaises(TypeError):
            extract_primary_hidden(42, context="test")


if __name__ == "__main__":
    unittest.main()

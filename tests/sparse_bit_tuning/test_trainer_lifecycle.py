import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.checkpoint import sidecar_complete
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _TinyDataset(Dataset):
    def __init__(self, count=8):
        self.items = []
        for i in range(count):
            ids = torch.tensor([1 + (i % 3), 2, 3, 4], dtype=torch.long)
            self.items.append(
                {
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones_like(ids),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _decoder(latent_dim=9, codebook_dim=4, *, use_checkpoint=False):
    return Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=use_checkpoint,
        num_models=1,
    )


def _bits():
    return torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )


class _TinyCausalLM(nn.Module):
    def __init__(self, *, use_decoder_checkpoint=False):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layer = VAELinear(
            in_features=4,
            out_features=4,
            bias=None,
            original_weight=None,
            vq_weight=_bits(),
            decoder=_decoder(use_checkpoint=use_decoder_checkpoint),
            codebook_dim=4,
            transpose=False,
        )
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(use_cache=False, pad_token_id=0)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        del attention_mask, kwargs
        x = self.embed(input_ids)
        x = self.layer(x)
        logits = self.lm_head(x)
        hidden_states = (x,) if output_hidden_states else None
        return {"logits": logits, "hidden_states": hidden_states}


def _build_model_and_manager(*, use_decoder_checkpoint=False, round_steps=2):
    device = torch.device("cuda:0")
    model = _TinyCausalLM(use_decoder_checkpoint=use_decoder_checkpoint).to(device=device, dtype=torch.bfloat16)
    model.layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    manager = SparseBitTuningManager(
        root_model=model,
        targets=[("layer", model.layer)],
        target_devices={"layer": device},
        training_seed=123,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=0.2,
            round_steps=round_steps,
        ),
        streaming=False,
    )
    return model, manager


def _training_args(output_dir, *, max_steps, save_steps=1, fp16=False):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=1,
        learning_rate=1e-3,
        bf16=not fp16,
        fp16=fp16,
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_num_workers=0,
    )
    # This file validates one-process Trainer lifecycle on cuda:0. Multi-GPU DP is
    # covered separately by the torchrun/DDP smoke test.
    args._n_gpu = 1
    return args


def _trainer(model, manager, output_dir, *, max_steps, save_steps=1, fp16=False):
    return VAEDecoderE2ETrainer(
        model=model,
        args=_training_args(output_dir, max_steps=max_steps, save_steps=save_steps, fp16=fp16),
        train_dataset=_TinyDataset(),
        loss_type="sft",
        sparse_bit_manager=manager,
    )


def test_real_trainer_train_save_resume_bf16():
    with tempfile.TemporaryDirectory() as tmp:
        model0, manager0 = _build_model_and_manager(round_steps=2)
        before = model0.layer.get_stage_part_vq_storage(0, 0).detach().cpu().clone()
        trainer0 = _trainer(model0, manager0, tmp, max_steps=2, save_steps=1)
        out0 = trainer0.train()
        assert int(out0.global_step) == 2
        ckpt = os.path.join(tmp, "checkpoint-2")
        assert os.path.isdir(ckpt)
        assert sidecar_complete(ckpt)
        assert manager0.global_bit_round == 1
        assert manager0.bank_specs[0].canonical_key in manager0.checkpoint_packed_snapshot()

        model1, manager1 = _build_model_and_manager(round_steps=7)
        trainer1 = _trainer(model1, manager1, tmp, max_steps=3, save_steps=99)
        out1 = trainer1.train(resume_from_checkpoint=ckpt)
        assert int(out1.global_step) == 3
        assert manager1.global_bit_round >= 1
        assert manager1._initialized_scores
        assert all(score.grad is None for score in manager1.score_module.score_chunks)


def test_real_trainer_decoder_checkpoint_recompute_keeps_bit_state_stable():
    with tempfile.TemporaryDirectory() as tmp:
        model, manager = _build_model_and_manager(use_decoder_checkpoint=True, round_steps=4)
        trainer = _trainer(model, manager, tmp, max_steps=2, save_steps=99)
        out = trainer.train()
        assert int(out.global_step) == 2
        assert manager.global_bit_round == 0
        assert manager.bit_round_step == 2


def test_real_trainer_fp16_sparse_bit_scaler_runs_one_step():
    with tempfile.TemporaryDirectory() as tmp:
        model, manager = _build_model_and_manager(round_steps=4)
        # Production FP16 AMP keeps ordinary trainable weights in BF16/FP32; only the
        # Sparse Bit score Parameter is intentionally FP16.
        trainer = _trainer(model, manager, tmp, max_steps=1, save_steps=99, fp16=True)
        out = trainer.train()
        assert int(out.global_step) == 1
        assert manager.bit_round_step == 1
        assert getattr(trainer.accelerator.scaler, "_sparse_bit_grad_scaler", False)

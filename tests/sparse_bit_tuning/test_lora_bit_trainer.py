import os
import tempfile

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from compressed_e2e_fintuning.aux_trainables import (
    AUX_CHECKPOINT_FILE,
    enable_compressed_lora_auxiliary_trainables,
)
from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import LOW_RANK_SCOPE_FULL
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.checkpoint import sidecar_complete
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.full_lora_proxy import build_full_compressed_peft_model
from sparse_bit_tuning.manager import SparseBitTuningManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Dataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        ids = torch.tensor([1 + (idx % 3), 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


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


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layer = VAELinear(
            in_features=4,
            out_features=4,
            bias=nn.Parameter(torch.zeros(4)),
            original_weight=None,
            vq_weight=_bits(),
            decoder=Decoder(
                in_dim=9,
                out_dim=4,
                hidden_dim=8,
                num_res_blocks=0,
                norm_type="layer",
                decoder_type="linear",
                use_checkpoint=False,
                num_models=1,
            ),
            codebook_dim=4,
            transpose=False,
        )
        self.lm_head = nn.Linear(4, 8, bias=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        del attention_mask, kwargs
        hidden = self.layer(self.embed(input_ids))
        return {"logits": self.lm_head(hidden), "hidden_states": (hidden,) if output_hidden_states else None}


def _build():
    torch.manual_seed(606)
    base = _TinyLM()
    layer = base.layer
    peft_model = build_full_compressed_peft_model(
        base,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    aux = enable_compressed_lora_auxiliary_trainables(
        peft_model,
        selected_vae_modules=[("layer", layer)],
        low_rank_scope=LOW_RANK_SCOPE_FULL,
        sparse_bit_tuning=True,
        vae_tune_bias=True,
        tune_final_norm=False,
        use_post_norm_head_linear=False,
    )
    device = torch.device("cuda:0")
    peft_model.to(device=device, dtype=torch.bfloat16)
    manager = SparseBitTuningManager(
        root_model=peft_model,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=606,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=0.2,
            round_steps=3,
        ),
        streaming=False,
    )
    return peft_model, layer, aux, manager


def _args(output_dir, max_steps, save_steps):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=1,
        learning_rate=1e-3,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_num_workers=0,
    )
    args._n_gpu = 1
    return args


def _trainer(model, manager, aux, output_dir, max_steps, save_steps):
    return VAEDecoderE2ETrainer(
        model=model,
        args=_args(output_dir, max_steps=max_steps, save_steps=save_steps),
        train_dataset=_Dataset(),
        loss_type="sft",
        sparse_bit_manager=manager,
        aux_trainable_parameters=aux.parameters,
    )


def test_full_lora_bit_bias_real_trainer_adapter_sidecars_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        model0, layer0, aux0, manager0 = _build()
        initial_bias = layer0.bias.detach().cpu().clone()
        trainer0 = _trainer(model0, manager0, aux0, tmp, max_steps=1, save_steps=1)
        out0 = trainer0.train()
        assert int(out0.global_step) == 1
        ckpt = os.path.join(tmp, "checkpoint-1")
        assert os.path.isfile(os.path.join(ckpt, "adapter_model.bin"))
        assert os.path.isfile(os.path.join(ckpt, "adapter_config.json"))
        assert not os.path.isfile(os.path.join(ckpt, "pytorch_model.bin"))
        assert sidecar_complete(ckpt)
        assert os.path.isfile(os.path.join(ckpt, AUX_CHECKPOINT_FILE))
        assert not torch.equal(layer0.bias.detach().cpu(), initial_bias)

        model1, layer1, aux1, manager1 = _build()
        trainer1 = _trainer(model1, manager1, aux1, tmp, max_steps=2, save_steps=99)
        out1 = trainer1.train(resume_from_checkpoint=ckpt)
        assert int(out1.global_step) == 2
        assert manager1.bit_round_step >= 1
        assert layer1.bias.requires_grad
        assert any("lora_" in name and p.requires_grad for name, p in model1.named_parameters())

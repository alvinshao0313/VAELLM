import tempfile

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)

    def forward(self, input_ids=None, labels=None, **kwargs):
        del labels, kwargs
        x = torch.zeros((1, 4), dtype=self.decoder.weight.dtype)
        if input_ids is not None and torch.is_tensor(input_ids):
            x = x.to(input_ids.device)
        logits = self.head(self.decoder(x))
        return {"loss": logits.float().sum() * 0.0, "logits": logits.unsqueeze(0)}


def _group_for_param(optimizer, target):
    for group in optimizer.param_groups:
        if any(param is target for param in group["params"]):
            return group
    raise AssertionError("parameter not found in optimizer")


def test_decoder_lr_is_independent_from_global_learning_rate():
    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyModel()
        args = TrainingArguments(
            output_dir=tmp,
            use_cpu=True,
            learning_rate=1e-3,
            weight_decay=0.1,
            report_to=[],
        )
        decoder_ids = tuple(id(param) for param in model.decoder.parameters())
        trainer = VAEDecoderE2ETrainer(
            model=model,
            args=args,
            loss_type="sft",
            decoder_param_ids=decoder_ids,
            decoder_lr=2e-4,
        )

        optimizer = trainer.create_optimizer()

        assert _group_for_param(optimizer, model.decoder.weight)["lr"] == pytest.approx(2e-4)
        assert _group_for_param(optimizer, model.decoder.bias)["lr"] == pytest.approx(2e-4)
        assert _group_for_param(optimizer, model.head.weight)["lr"] == pytest.approx(1e-3)
        assert _group_for_param(optimizer, model.head.bias)["lr"] == pytest.approx(1e-3)
        assert _group_for_param(optimizer, model.decoder.weight)["weight_decay"] == pytest.approx(0.1)
        assert _group_for_param(optimizer, model.decoder.bias)["weight_decay"] == pytest.approx(0.0)

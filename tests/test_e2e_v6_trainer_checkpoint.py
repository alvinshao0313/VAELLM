from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from transformers import TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from train_utils import checkpoint_v6 as v6


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        del labels, kwargs
        x = input_ids.to(dtype=self.proj.weight.dtype)
        logits = self.proj(x)
        loss = logits.square().mean()
        return {"loss": loss, "logits": logits}


def _context(round_base_dir: str, round_base_checkpoint_id: str) -> dict:
    return {
        "round_base_dir": str(round_base_dir),
        "round_base_checkpoint_id": str(round_base_checkpoint_id),
        "train_mode": "none",
        "compressed_targets": (),
        "pending_dense_targets": (),
        "skip_targets": (),
        "legacy_original_only_sources": (),
        "norm_train_mode": "none",
        "lm_head_train_mode": "none",
        "lora_config": None,
        "resolved_learning_rates": {"learning_rate": 1e-3},
        "compression_categories": (),
        "target_layers": (),
        "target_modules": (),
        "immutable_resume_contract": {"test_contract": 1},
        "base_model_path": "tiny-test-model",
        "runtime_audit": {"test": True},
        "hf_artifact_refs": {},
    }


def test_v6_trainer_step_checkpoint_skips_full_model_and_restores_mutable_state(tmp_path: Path):
    model = _TinyTrainModel()
    round_base = tmp_path / "round_base"
    saved = v6.save_v6_full_checkpoint(
        model,
        str(round_base),
        checkpoint_kind="round_base",
        compressed_targets=(),
        train_mode="none",
        base_model_path="tiny-test-model",
        save_config=False,
    )

    trainer_dir = tmp_path / "trainer"
    args = TrainingArguments(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=1,
        max_steps=3,
        learning_rate=1e-3,
        save_strategy="steps",
        save_steps=1,
        save_safetensors=False,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = VAEDecoderE2ETrainer(model=model, args=args, loss_type="sft")
    trainer.configure_v6_step_checkpoint(
        context=_context(str(round_base), str(saved["checkpoint_id"])),
        selected_vae_modules=(),
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=3)
    trainer.state.global_step = 1

    expected_weight = model.proj.weight.detach().clone()
    trainer._save_checkpoint(trainer.model, trial=None)
    step_dir = trainer_dir / "checkpoint-1"

    assert (step_dir / v6.META_FILENAME).is_file()
    assert (step_dir / v6.TRAINING_MODEL_STATE_FILENAME).is_file()
    assert (step_dir / "optimizer.pt").is_file()
    assert (step_dir / "scheduler.pt").is_file()
    assert (step_dir / "trainer_state.json").is_file()
    assert any(path.name.startswith("rng_state") for path in step_dir.iterdir())

    forbidden = {
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
    assert forbidden.isdisjoint({path.name for path in step_dir.iterdir()})

    meta = v6.load_v6_training_step_meta(str(step_dir))
    assert meta["round_base_checkpoint_id"] == saved["checkpoint_id"]
    resolved_base, _ = v6.resolve_training_step_round_base_ref(str(step_dir), meta)
    assert os.path.samefile(resolved_base, round_base)

    with torch.no_grad():
        model.proj.weight.add_(123.0)
    assert not torch.equal(model.proj.weight.detach(), expected_weight)

    trainer._load_from_checkpoint(str(step_dir))
    assert trainer._v6_exact_resume_loaded is True
    assert torch.equal(model.proj.weight.detach(), expected_weight)


def test_v6_trainer_resume_rejects_immutable_contract_change(tmp_path: Path):
    model = _TinyTrainModel()
    round_base = tmp_path / "round_base"
    saved = v6.save_v6_full_checkpoint(
        model,
        str(round_base),
        checkpoint_kind="round_base",
        compressed_targets=(),
        train_mode="none",
        base_model_path="tiny-test-model",
        save_config=False,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=1,
        max_steps=2,
        learning_rate=1e-3,
        save_strategy="steps",
        save_steps=1,
        save_safetensors=False,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = VAEDecoderE2ETrainer(model=model, args=args, loss_type="sft")
    context = _context(str(round_base), str(saved["checkpoint_id"]))
    trainer.configure_v6_step_checkpoint(context=context, selected_vae_modules=())
    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    trainer.state.global_step = 1
    trainer._save_checkpoint(trainer.model, trial=None)

    changed = dict(context)
    changed["immutable_resume_contract"] = {"test_contract": 2}
    trainer.configure_v6_step_checkpoint(context=changed, selected_vae_modules=())
    try:
        trainer._load_from_checkpoint(str(Path(args.output_dir) / "checkpoint-1"))
    except ValueError as exc:
        assert "immutable contract mismatch" in str(exc)
    else:
        raise AssertionError("immutable contract change must reject exact resume")

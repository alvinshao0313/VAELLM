"""One-step real trainer/VAE smoke tests (no HF model download).

Exercises REAL VAEDecoderE2ETrainer.compute_loss / training_step,
CustomSFTTrainer.compute_loss (pre-MLP), and MultiLayerVAE train step
via tiny in-process nn.Module scaffolds — same spirit as unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import MultiLayerVAE
from litebsq.vae_args import apply_autoencoder_arch_defaults
from train_utils.cat_train_args import (
    process_cat_train_args,
    resolve_category_runtime_configs,
)
from train_utils.lora_training import (
    CustomSFTTrainer,
    compute_distill_pre_mlp_hidden_alignment_loss,
    parse_distill_hidden_alignment_layer_weighting,
)
from train_utils.train_args import TrainingArguments, create_optimizer


EAKLD_LOG_KEYS = {
    "eakld/teacher_entropy_mean",
    "eakld/gamma_reverse_mean",
    "eakld/lambda_forward_mean",
    "eakld/gamma_reverse_zero_fraction",
    "eakld/gamma_reverse_one_fraction",
    "eakld/forward_kl_mean",
    "eakld/reverse_kl_mean",
    "eakld/total_mean",
}

class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        output = hidden_states + 0.1 * torch.tanh(self.proj(hidden_states))
        return (output,)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TinyBlock(hidden_size) for _ in range(num_layers)]
        )


class _TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 31,
        hidden_size: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _TinyBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, vocab_size=vocab_size)
        self.output_hidden_states_calls: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_kwargs,
    ):
        del attention_mask
        self.output_hidden_states_calls.append(bool(output_hidden_states))
        hidden = self.embed_tokens(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
            if hidden_states is not None:
                hidden_states.append(hidden)
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(tuple(hidden_states) if hidden_states is not None else None),
        )


class _FakeOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class _TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporary = True
        self.scale = nn.Parameter(torch.tensor(1.5))

    def set_temporary(self, temporary: bool) -> None:
        self.temporary = bool(temporary)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.temporary:
            return hidden * self.scale
        return hidden


class _PreMlpLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _PreMlpBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_PreMlpLayer(hidden_size) for _ in range(num_layers)])


class _PreMlpFakeCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 11, hidden_size: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _PreMlpBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.temp_scale = _TempScale()
        self.output_hidden_states_calls: list[bool] = []
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_kwargs,
    ):
        del attention_mask
        self.output_hidden_states_calls.append(bool(output_hidden_states))
        hidden = self.temp_scale(self.embed_tokens(input_ids))
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.model.layers:
            hidden = layer(hidden)
            if hidden_states is not None:
                hidden_states.append(hidden)
        logits = self.lm_head(hidden)
        if labels is None:
            loss = logits.float().pow(2).mean()
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        packed = tuple(hidden_states) if hidden_states is not None else None
        return _FakeOutput(loss=loss, logits=logits, hidden_states=packed)


def _e2e_inputs(*, seq_len: int = 8) -> dict[str, torch.Tensor]:
    assert seq_len <= 128
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    # prompt prefix ignored by distill mask
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _build_e2e_trainer(
    tmp_path,
    *,
    teacher_output_offload: str,
    teacher_output_chunk_tokens: int = 8,
    prompt_kd_weight: float = 0.0,
):
    teacher = _TinyCausalLM()
    student = _TinyCausalLM()
    # mismatch student so loss is non-trivial
    with torch.no_grad():
        student.lm_head.weight.add_(0.05)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=True,
        logging_steps=1,
    )
    trainer = VAEDecoderE2ETrainer(
        model=student,
        args=args,
        loss_type="eakld",
        teacher_model=teacher,
        distill_temperature=1.0,
        hidden_loss_weight=0.0,
        prompt_kd_weight=float(prompt_kd_weight),
        eakld_confidence_k=16,
        hidden_layer_weighting="uniform",
        teacher_output_offload=teacher_output_offload,
        teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=int(teacher_output_chunk_tokens),
    )
    trainer._teacher_device = torch.device("cpu")
    return trainer, student, teacher


def _assert_eakld_telemetry_logs(logs: dict) -> None:
    assert set(EAKLD_LOG_KEYS).issubset(set(logs))
    gamma = float(logs["eakld/gamma_reverse_mean"])
    lam = float(logs["eakld/lambda_forward_mean"])
    assert abs(gamma + lam - 1.0) < 1e-5
    for key in (
        "eakld/teacher_entropy_mean",
        "eakld/forward_kl_mean",
        "eakld/reverse_kl_mean",
        "eakld/total_mean",
    ):
        assert abs(float(logs[key])) < float("inf")
        assert float(logs[key]) == float(logs[key])  # not NaN


@pytest.fixture(autouse=True)
def _patch_tiny_get_layers(monkeypatch):
    monkeypatch.setattr(
        "compressed_e2e_fintuning.teacher_targets.get_layers",
        lambda model: model.model.layers,
    )


def test_dense_eakld_one_step_trainer_smoke(tmp_path) -> None:
    """Step 1: dense EAKLD path through real VAEDecoderE2ETrainer."""
    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path,
        teacher_output_offload="none",
        prompt_kd_weight=0.1,
    )
    assert trainer.prompt_kd_weight == 0.1
    inputs = _e2e_inputs(seq_len=8)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-3,
    )
    before = student.lm_head.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    step_loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(step_loss)
    assert torch.isfinite(step_loss)
    optimizer.step()

    assert not torch.equal(student.lm_head.weight.detach(), before)
    assert trainer._active_teacher_targets is None

    # telemetry was recorded during compute_loss; flush via log()
    logs: dict = {}
    trainer.log(logs)
    _assert_eakld_telemetry_logs(logs)


def test_cpu_offload_eakld_one_step_trainer_smoke(tmp_path, monkeypatch) -> None:
    """Step 2: CPU teacher-output-offload EAKLD one-step."""
    entropy_calls = {"n": 0}
    original_entropy = (
        __import__(
            "compressed_e2e_fintuning.trainer",
            fromlist=["compute_teacher_entropy_mean_and_gamma"],
        ).compute_teacher_entropy_mean_and_gamma
    )

    def counting_entropy(*args, **kwargs):
        entropy_calls["n"] += 1
        return original_entropy(*args, **kwargs)

    monkeypatch.setattr(
        "compressed_e2e_fintuning.trainer.compute_teacher_entropy_mean_and_gamma",
        counting_entropy,
    )

    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path,
        teacher_output_offload="cpu",
        teacher_output_chunk_tokens=8,
        prompt_kd_weight=0.1,
    )
    assert trainer.prompt_kd_weight == 0.1
    inputs = _e2e_inputs(seq_len=8)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-3,
    )
    before = student.lm_head.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    step_loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(step_loss)
    assert torch.isfinite(step_loss)
    optimizer.step()

    assert not torch.equal(student.lm_head.weight.detach(), before)
    assert trainer._active_teacher_targets is None
    # Entropy/gamma computed once when building CPU targets — not recomputed
    # via a second full-vocab softmax during the offloaded loss path.
    assert entropy_calls["n"] == 1

    logs: dict = {}
    trainer.log(logs)
    _assert_eakld_telemetry_logs(logs)


def _build_pre_mlp_trainer(
    *,
    hidden_alignment_layer_weighting: str,
) -> CustomSFTTrainer:
    trainer = CustomSFTTrainer.__new__(CustomSFTTrainer)
    trainer.args = SimpleNamespace(bf16=False, fp16=False)
    trainer.loss_type = "sft"
    trainer.temperature = 1.0
    trainer.loss_alpha = 0.5
    trainer.hidden_loss_weight = 0.0
    trainer.pre_mlp_hidden_loss_weight = 0.25
    trainer.prompt_kd_weight = 0.0
    trainer.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
        hidden_alignment_layer_weighting
    )
    trainer.eakld_confidence_k = 16
    trainer.teacher_logits_cpu_staging = False
    trainer.distill_hif4_act_controller = None
    trainer.teacher_param_snapshots = []
    trainer.accelerator = None
    return trainer


def _pre_mlp_inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


@pytest.mark.parametrize(
    "weighting,expected_hidden_calls",
    (
        ("uniform", [False, False]),
        ("adaptive_top_3", [True, False]),
    ),
)
def test_pre_mlp_only_one_step_smoke(weighting: str, expected_hidden_calls: list[bool]) -> None:
    """Step 3: category distill pre-MLP-only (uniform + adaptive_top_3)."""
    model = _PreMlpFakeCausalLM()
    trainer = _build_pre_mlp_trainer(hidden_alignment_layer_weighting=weighting)
    inputs = _pre_mlp_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.temp_scale.scale.detach().clone()

    with mock.patch(
        "train_utils.lora_training.compute_distill_pre_mlp_hidden_alignment_loss",
        wraps=compute_distill_pre_mlp_hidden_alignment_loss,
    ) as pre_mlp_mock:
        loss = trainer.compute_loss(model, inputs)
        assert pre_mlp_mock.call_count == 1
        pre_kwargs = pre_mlp_mock.call_args.kwargs
        if weighting == "uniform":
            assert pre_kwargs["teacher_reference_hidden"] is None
        else:
            assert pre_kwargs["teacher_reference_hidden"] is not None

    assert model.output_hidden_states_calls == expected_hidden_calls
    assert torch.isfinite(loss).item()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert model.temp_scale.scale.grad is not None or not torch.equal(
        model.temp_scale.scale.detach(), before
    )
    assert not torch.equal(model.temp_scale.scale.detach(), before)


def _tiny_vae_args() -> SimpleNamespace:
    args = SimpleNamespace(
        codebook_bits=4,
        codebook_dim=4,
        residual_stages=1,
        base_ch=8,
        num_res_blocks=0,
        decoder_base_ch=None,
        decoder_num_res_blocks=None,
        decoder_type="linear",
        norm_type="no",
        activation_type="relu",
        quantizer_type="BSQ",
        recon_loss_type="mse",
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        inv_temperature=100.0,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        l1_weight=1.0,
        lfq_weight=1.0,
        new_quant=False,
        vae_weight_dtype="fp32",
        vae_autocast_dtype="fp32",
        parallel_layers=1,
        normalize_weight=False,
        optimizer="adamw",
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        use_checkpoint=False,
    )
    apply_autoencoder_arch_defaults(args)
    return args


def test_vae_mse_one_step_smoke() -> None:
    """Step 4: VAE one-step with recon_loss_type=mse and BSQ."""
    cat_args, _hf, _training, vae_args = process_cat_train_args(
        [
            "--recon_loss_type",
            "default=mse",
            "--quantizer_type",
            "BSQ",
            "--steps_per_category",
            "default=1",
        ]
    )
    resolved = resolve_category_runtime_configs(
        cat_args,
        vae_args,
        active_categories=["q_proj"],
    )
    cfg = resolved["q_proj"]
    assert cfg.recon_loss_type == "mse"

    args = _tiny_vae_args()
    assert args.recon_loss_type == "mse"
    assert args.quantizer_type == "BSQ"
    vae = MultiLayerVAE(args)
    optimizer = create_optimizer(vae.parameters(), args, lr=1e-3)

    torch.manual_seed(0)
    # [B, 1, codebook_dim] weight-block chunks
    x = torch.randn(8, 1, int(args.codebook_dim), dtype=torch.float32)
    before = next(vae.parameters()).detach().clone()

    vae.train()
    optimizer.zero_grad(set_to_none=True)
    _x_recon, loss_dict = vae(x, is_train=True)
    loss = loss_dict["loss"]
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["train/recon_loss"])
    assert torch.isfinite(loss_dict["train/commitment_loss"])
    loss.backward()
    optimizer.step()

    after = next(vae.parameters()).detach()
    assert not torch.equal(before, after)

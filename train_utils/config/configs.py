from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, normalize_dataset_mix_spec
from litebsq.protected_channel_quant import (
    PROTECTED_CHANNEL_QUANT_CHOICES,
    PROTECTED_CHANNEL_QUANT_NONE,
    normalize_protected_channel_quant_format,
)

from train_utils.config.overrides import parse_float_text


LOSS_TYPES = ("sft", "kl", "kl_top", "kd", "kd_top")
LOSS_TYPES_NEED_TEACHER = frozenset({"kl", "kl_top", "kd", "kd_top"})
HIDDEN_LAYER_WEIGHTING_STATIC = ("uniform", "linear_depth", "adaptive")
TRAIN_MODES = (
    "none",
    "decoder",
    "lora",
    "sparse_bit",
    "decoder_lora",
    "decoder_sparse_bit",
    "lora_sparse_bit",
    "decoder_lora_sparse_bit",
)
AFTER_CATEGORY_MODES = (
    "none",
    "current_lora",
    "current_decoder",
    "current_lora_decoder",
    "remaining_lora",
    "remaining_lora_current_decoder",
    "remaining_lora_prefix_decoder",
)
DATASET_TASKS = ("lm", "sft")
NORM_TRAIN_MODES = ("none", "final", "all")
LM_HEAD_TRAIN_MODES = ("none", "linear", "lora", "full")
TEACHER_OFFLOAD_MODES = ("none", "cpu")
PARALLEL_MODES = ("dp", "layer_mp")
OFFLOAD_MODES = ("none", "saved_tensors", "streaming")
CHANNEL_PROTECT_MODES = ("none", "channel")
CHANNEL_SCOPES = ("layer", "category", "global")
CHANNEL_AXES = ("input", "output")
CHANNEL_RANK_METRICS = (
    "channel_weight_abs",
    "channel_weight_actmax_abs",
    "channel_weight_actmean_abs",
)
CHANNEL_MLP_RANK_METRICS = (
    "none",
    "mlp_intermediate_aligned_actrms",
    "mlp_intermediate_aligned_actmean_abs",
    "mlp_intermediate_aligned_actrms_abs",
)
QUANTIZER_TYPES = ("bsq", "lfq")
RECON_LOSS_TYPES = (
    "mse",
    "l1",
    "huber",
    "relative_l1",
    "w_mse",
    "w2_mse",
    "wa_mse",
    "amse",
)
VAE_NORM_TYPES = ("group", "batch", "layer", "rms", "no")
VAE_ACTIVATION_TYPES = ("swish", "relu", "none", "sigmoid", "gelu", "hard_swish")
VAE_DECODER_TYPES = ("linear", "symmetric", "asymmetric")
VAE_OPTIMS = ("adam", "adamw", "sgd", "rmsprop")
VAE_LR_SCHEDULERS = ("constant", "linear", "cosine")


def _require_finite(value: float, *, arg_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{arg_name} must be finite, got {value}.")
    return out


def parse_hidden_layer_weighting(raw: object) -> str:
    mode = str(raw).strip().lower()
    if mode in HIDDEN_LAYER_WEIGHTING_STATIC:
        return mode
    if mode == "adaptive_top_k":
        raise argparse.ArgumentTypeError(
            "Invalid --hidden_layer_weighting 'adaptive_top_K'. "
            "K must be a positive integer, for example adaptive_top_3."
        )
    if mode.startswith("adaptive_top_"):
        suffix = mode[len("adaptive_top_") :]
        if suffix.isdigit() and int(suffix) >= 1:
            return mode
    raise argparse.ArgumentTypeError(
        f"Invalid --hidden_layer_weighting: {raw!r}. "
        "Supported: uniform, linear_depth, adaptive, adaptive_top_<K>."
    )


def parse_loss_type(raw: object) -> str:
    value = str(raw).strip().lower()
    if value in LOSS_TYPES:
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid --loss_type {raw!r}. Supported: {', '.join(LOSS_TYPES)}. "
        "Do not encode top-k in the type string; use --top_k."
    )


def parse_train_mode(raw: object) -> str:
    value = str(raw).strip().lower()
    if value in TRAIN_MODES:
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid --train_mode {raw!r}. Supported: {', '.join(TRAIN_MODES)}."
    )


def parse_after_category_mode(raw: object) -> str:
    value = str(raw).strip().lower()
    if value in AFTER_CATEGORY_MODES:
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid --after_category_mode {raw!r}. Supported: {', '.join(AFTER_CATEGORY_MODES)}."
    )


def parse_dataset_mix_spec(raw: object) -> Tuple[Tuple[str, ...], Tuple[float, ...], str]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("--dataset_mix cannot be empty.")
    expanded = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            alias = token.strip().lower()
            if alias not in DATASET_MIX_SOURCE_PRESETS:
                raise ValueError(
                    f"Unsupported --dataset_mix alias '{alias}'. "
                    f"Supported: {sorted(DATASET_MIX_SOURCE_PRESETS)}."
                )
            expanded.append(f"{alias}=1.0")
        else:
            expanded.append(token)
    if not expanded:
        raise ValueError("--dataset_mix cannot be empty.")
    sources, weights, spec = normalize_dataset_mix_spec(",".join(expanded))
    return tuple(sources), tuple(float(weight) for weight in weights), str(spec)


def vae_num_warmup_steps(ratio: float, vae_steps: int) -> int:
    ratio_value = _require_finite(ratio, arg_name="vae_warmup_ratio")
    if ratio_value < 0.0 or ratio_value > 1.0:
        raise ValueError(f"vae_warmup_ratio must be in [0, 1], got {ratio}.")
    steps = int(vae_steps)
    if steps < 0:
        raise ValueError(f"vae_steps must be >= 0, got {vae_steps}.")
    return int(ratio_value * steps)


def parse_optional_positive_float(raw: object, *, arg_name: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    value = parse_float_text(text, arg_name=arg_name, min_value=0.0, inclusive_min=False)
    return float(value)


@dataclass
class DistillDataConfig:
    dataset_mix: Optional[str] = None
    dataset_task: str = "sft"
    train_file: Optional[str] = None
    text_field: str = "text"
    model_max_length: int = 1024
    dynamic_padding: bool = True
    seed: int = 42
    data_seed: int = 42
    group_by_length: bool = True
    dataset_mix_sources: Optional[Tuple[str, ...]] = None
    dataset_mix_weights: Optional[Tuple[float, ...]] = None

    def validate(self) -> None:
        task = str(self.dataset_task or "").strip().lower()
        if task not in DATASET_TASKS:
            raise ValueError(f"dataset_task must be one of {DATASET_TASKS}, got {self.dataset_task!r}.")
        self.dataset_task = task
        if int(self.model_max_length) < 2:
            raise ValueError(
                f"model_max_length must be >= 2 to form a causal next-token target, got {self.model_max_length}."
            )
        mix = None if self.dataset_mix is None else str(self.dataset_mix).strip()
        train_file = None if self.train_file is None else str(self.train_file).strip()
        if mix and train_file:
            raise ValueError("dataset_mix and train_file are mutually exclusive.")
        if mix:
            sources, weights, spec = parse_dataset_mix_spec(mix)
            self.dataset_mix = spec
            self.dataset_mix_sources = sources
            self.dataset_mix_weights = weights
            self.train_file = None
        elif train_file:
            self.train_file = train_file
            self.dataset_mix = None
            self.dataset_mix_sources = None
            self.dataset_mix_weights = None
        else:
            self.dataset_mix = None
            self.dataset_mix_sources = None
            self.dataset_mix_weights = None


@dataclass
class DistillLossConfig:
    loss_type: str = "sft"
    top_k: int = 100
    temperature: float = 1.0
    alpha: float = 0.5
    prompt_loss_weight: float = 0.0
    hidden_loss_weight: float = 0.0
    pre_mlp_hidden_loss_weight: float = 0.0
    hidden_layer_weighting: str = "uniform"
    selective_student_topk: bool = False
    selective_student_topk_chunk_rows: int = 32

    def validate(self) -> None:
        self.loss_type = parse_loss_type(self.loss_type)
        if int(self.top_k) <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}.")
        self.temperature = _require_finite(self.temperature, arg_name="temperature")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}.")
        self.alpha = _require_finite(self.alpha, arg_name="alpha")
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")
        for name in ("prompt_loss_weight", "hidden_loss_weight", "pre_mlp_hidden_loss_weight"):
            value = _require_finite(getattr(self, name), arg_name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
            setattr(self, name, value)
        self.hidden_layer_weighting = parse_hidden_layer_weighting(self.hidden_layer_weighting)
        if int(self.selective_student_topk_chunk_rows) < 1:
            raise ValueError("selective_student_topk_chunk_rows must be >= 1.")
        if bool(self.selective_student_topk) and self.loss_type != "kl_top":
            raise ValueError("selective_student_topk=true is only allowed when loss_type=kl_top.")


def teacher_required(loss: DistillLossConfig) -> bool:
    """Thin adapter over the single teacher-required formula."""
    from train_utils.distill_teacher import resolve_distill_teacher_required

    return bool(
        resolve_distill_teacher_required(
            loss_type=str(loss.loss_type),
            hidden_loss_weight=float(loss.hidden_loss_weight),
            pre_mlp_hidden_loss_weight=float(loss.pre_mlp_hidden_loss_weight),
        )
    )


@dataclass
class LoRAConfig:
    rank: int = 12
    alpha: float = 24.0
    dropout: float = 0.03
    rank_explicit: bool = False
    alpha_explicit: bool = False
    dropout_explicit: bool = False

    def validate(self) -> None:
        if int(self.rank) < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.rank}.")
        self.alpha = _require_finite(self.alpha, arg_name="lora_alpha")
        if self.alpha <= 0.0:
            raise ValueError(f"lora_alpha must be > 0, got {self.alpha}.")
        self.dropout = _require_finite(self.dropout, arg_name="lora_dropout")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"lora_dropout must satisfy 0 <= dropout < 1, got {self.dropout}.")


def validate_lora_against_checkpoint(user: LoRAConfig, checkpoint: LoRAConfig) -> LoRAConfig:
    user.validate()
    checkpoint.validate()

    def _resolve(name: str, user_value, explicit: bool, checkpoint_value):
        if not explicit:
            return checkpoint_value
        if user_value == checkpoint_value:
            return user_value
        raise ValueError(
            f"{name} explicit value {user_value!r} conflicts with checkpoint {checkpoint_value!r}."
        )

    return LoRAConfig(
        rank=int(_resolve("lora_rank", int(user.rank), user.rank_explicit, int(checkpoint.rank))),
        alpha=float(_resolve("lora_alpha", float(user.alpha), user.alpha_explicit, float(checkpoint.alpha))),
        dropout=float(
            _resolve("lora_dropout", float(user.dropout), user.dropout_explicit, float(checkpoint.dropout))
        ),
        rank_explicit=bool(user.rank_explicit),
        alpha_explicit=bool(user.alpha_explicit),
        dropout_explicit=bool(user.dropout_explicit),
    )


@dataclass
class AuxTrainableConfig:
    norm_train_mode: str = "none"
    norm_lr: Optional[float] = None
    lm_head_train_mode: str = "none"
    lm_head_lr: Optional[float] = None

    def validate(self) -> None:
        norm_mode = str(self.norm_train_mode or "none").strip().lower()
        if norm_mode not in NORM_TRAIN_MODES:
            raise ValueError(f"norm_train_mode must be one of {NORM_TRAIN_MODES}, got {self.norm_train_mode!r}.")
        self.norm_train_mode = norm_mode
        head_mode = str(self.lm_head_train_mode or "none").strip().lower()
        if head_mode not in LM_HEAD_TRAIN_MODES:
            raise ValueError(
                f"lm_head_train_mode must be one of {LM_HEAD_TRAIN_MODES}, got {self.lm_head_train_mode!r}."
            )
        self.lm_head_train_mode = head_mode
        if self.norm_lr is not None:
            self.norm_lr = _require_finite(self.norm_lr, arg_name="norm_lr")
            if self.norm_lr <= 0.0:
                raise ValueError("norm_lr must be > 0 when set.")
        if self.lm_head_lr is not None:
            self.lm_head_lr = _require_finite(self.lm_head_lr, arg_name="lm_head_lr")
            if self.lm_head_lr <= 0.0:
                raise ValueError("lm_head_lr must be > 0 when set.")


@dataclass
class DistillOptimizationConfig:
    steps: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    decoder_lr: Optional[float] = None
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, object] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    logging_steps: int = 1

    def validate(self) -> None:
        if int(self.steps) < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}.")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        self.learning_rate = _require_finite(self.learning_rate, arg_name="learning_rate")
        if self.decoder_lr is not None:
            self.decoder_lr = _require_finite(self.decoder_lr, arg_name="decoder_lr")
        self.weight_decay = _require_finite(self.weight_decay, arg_name="weight_decay")
        if int(self.gradient_accumulation_steps) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        self.max_grad_norm = _require_finite(self.max_grad_norm, arg_name="max_grad_norm")
        self.warmup_ratio = _require_finite(self.warmup_ratio, arg_name="warmup_ratio")
        if self.warmup_ratio < 0.0 or self.warmup_ratio > 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}.")
        if int(self.logging_steps) < 1:
            raise ValueError("logging_steps must be >= 1.")
        if not str(self.optim or "").strip():
            raise ValueError("optim cannot be empty.")
        if not str(self.lr_scheduler_type or "").strip():
            raise ValueError("lr_scheduler_type cannot be empty.")

    def resolved_decoder_lr(self) -> float:
        if self.decoder_lr is None:
            return float(self.learning_rate)
        return float(self.decoder_lr)


@dataclass
class EvaluationRuntimeConfig:
    eval_tasks: Optional[str] = None
    eval_num_fewshot: int = 0
    eval_batch_size: str = "auto"
    eval_limit: Optional[int] = None
    eval_device: str = "cuda"
    eval_after_save: bool = False
    skip_ppl_eval: bool = False
    ppl_seqlen: int = 2048
    ppl_limit: int = -1
    eval_hif4_act: bool = False
    eval_prewarm_group_size: int = 8

    def validate(self) -> None:
        tasks = None if self.eval_tasks is None else str(self.eval_tasks).strip()
        self.eval_tasks = tasks or None
        if int(self.eval_num_fewshot) < 0:
            raise ValueError("eval_num_fewshot must be >= 0.")
        if not str(self.eval_batch_size or "").strip():
            raise ValueError("eval_batch_size cannot be empty.")
        if self.eval_limit is not None and int(self.eval_limit) < 1:
            raise ValueError("eval_limit must be >= 1 when provided.")
        if not str(self.eval_device or "").strip():
            raise ValueError("eval_device cannot be empty.")
        if int(self.ppl_seqlen) < 1:
            raise ValueError("ppl_seqlen must be >= 1.")
        if int(self.ppl_limit) == 0 or int(self.ppl_limit) < -1:
            raise ValueError("ppl_limit must be -1 or >= 1.")
        if int(self.eval_prewarm_group_size) < 1:
            raise ValueError("eval_prewarm_group_size must be >= 1.")


@dataclass
class DistillRuntimeConfig:
    teacher_output_offload: str = "none"
    teacher_model_offload: str = "none"
    teacher_output_pin_memory: bool = True
    teacher_output_chunk_tokens: int = 8
    vae_decoder_checkpoint: bool = True
    parallel_mode: str = "dp"
    layer_device_map: str = "auto"
    offload_mode: str = "none"
    offload_checkpoint: bool = True
    offload_prefetch_distance: int = 1
    offload_min_tensor_bytes: int = 1048576
    offload_pin_memory: bool = True
    distill_hif4_act: bool = False
    evaluation: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)
    inactive_fields: Tuple[str, ...] = ()

    def validate(self) -> None:
        output_offload = str(self.teacher_output_offload or "none").strip().lower()
        model_offload = str(self.teacher_model_offload or "none").strip().lower()
        if output_offload not in TEACHER_OFFLOAD_MODES:
            raise ValueError("teacher_output_offload must be one of: none | cpu.")
        if model_offload not in TEACHER_OFFLOAD_MODES:
            raise ValueError("teacher_model_offload must be one of: none | cpu.")
        if model_offload == "cpu" and output_offload != "cpu":
            raise ValueError("teacher_model_offload=cpu requires teacher_output_offload=cpu.")
        self.teacher_output_offload = output_offload
        self.teacher_model_offload = model_offload
        if int(self.teacher_output_chunk_tokens) < 1:
            raise ValueError("teacher_output_chunk_tokens must be >= 1.")
        parallel_mode = str(self.parallel_mode or "dp").strip().lower()
        if parallel_mode not in PARALLEL_MODES:
            raise ValueError("parallel_mode must be one of: dp | layer_mp.")
        self.parallel_mode = parallel_mode
        offload_mode = str(self.offload_mode or "none").strip().lower()
        if offload_mode not in OFFLOAD_MODES:
            raise ValueError("offload_mode must be one of: none | saved_tensors | streaming.")
        self.offload_mode = offload_mode
        if parallel_mode == "dp" and offload_mode == "streaming":
            raise ValueError("parallel_mode=dp does not support offload_mode=streaming.")
        import os

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if offload_mode == "streaming" and world_size != 1:
            raise ValueError(
                "offload_mode=streaming only supports single-process (WORLD_SIZE=1)."
            )
        if parallel_mode == "layer_mp" and world_size != 1:
            raise ValueError("parallel_mode=layer_mp requires WORLD_SIZE=1.")
        if int(self.offload_prefetch_distance) < 0:
            raise ValueError("offload_prefetch_distance must be >= 0.")
        if int(self.offload_min_tensor_bytes) < 0:
            raise ValueError("offload_min_tensor_bytes must be >= 0.")
        self.layer_device_map = str(self.layer_device_map or "auto").strip().lower()
        if parallel_mode == "dp" and self.layer_device_map not in {"", "auto"}:
            raise ValueError(
                "parallel_mode=dp requires layer_device_map=auto "
                f"(got {self.layer_device_map!r}). Explicit layer maps are only valid for layer_mp."
            )
        self.evaluation.validate()
        inactive = []
        if output_offload != "cpu":
            inactive.append("teacher_output_pin_memory")
        if offload_mode == "none":
            inactive.extend(
                [
                    "offload_checkpoint",
                    "offload_prefetch_distance",
                    "offload_min_tensor_bytes",
                    "offload_pin_memory",
                ]
            )
        elif offload_mode == "saved_tensors":
            inactive.extend(["offload_checkpoint", "offload_prefetch_distance"])
        self.inactive_fields = tuple(inactive)


@dataclass
class VAECoreConfig:
    codebook_bits: int = 16
    codebook_dim: int = 8
    residual_stages: int = 1
    base_ch: int = 128
    num_res_blocks: int = 1
    quantizer_type: str = "BSQ"
    gamma0: float = 1.0
    gamma: float = 1.0
    zeta: float = 1.0
    inv_temperature: float = 100.0
    normalize_weight: bool = False
    new_quant: bool = False
    transpose_modules: str = "v_proj,o_proj,gate_proj,up_proj,down_proj"
    intra_parallel: Tuple[int, int] = (1, 1)
    linear_group_size: int = 32
    allow_tail_group: bool = True

    def validate(self) -> None:
        if int(self.codebook_bits) < 1:
            raise ValueError("codebook_bits must be >= 1.")
        if int(self.codebook_dim) < 1:
            raise ValueError("codebook_dim must be >= 1.")
        if int(self.residual_stages) < 1:
            raise ValueError("residual_stages must be >= 1.")
        if int(self.base_ch) < 1:
            raise ValueError("base_ch must be >= 1.")
        if int(self.num_res_blocks) < 0:
            raise ValueError("num_res_blocks must be >= 0.")
        quantizer = str(self.quantizer_type or "").strip().upper()
        if quantizer.lower() not in QUANTIZER_TYPES:
            raise ValueError(f"quantizer_type must be BSQ or LFQ, got {self.quantizer_type!r}.")
        self.quantizer_type = quantizer
        for name in ("gamma0", "gamma", "zeta", "inv_temperature"):
            setattr(self, name, _require_finite(getattr(self, name), arg_name=name))
        parts = tuple(int(v) for v in self.intra_parallel)
        if len(parts) != 2 or parts[0] < 1 or parts[1] < 1:
            raise ValueError(
                f"intra_parallel must be a pair of integers >= 1, got {self.intra_parallel!r}."
            )
        self.intra_parallel = (int(parts[0]), int(parts[1]))
        if int(self.linear_group_size) < 1:
            raise ValueError("linear_group_size must be >= 1.")


@dataclass
class VAEDecoderConfig:
    decoder_base_ch: Optional[int] = None
    decoder_num_res_blocks: Optional[int] = None
    norm_type: str = "group"
    activation_type: str = "swish"
    decoder_type: str = "linear"

    def validate(self) -> None:
        if self.decoder_base_ch is not None and int(self.decoder_base_ch) < 1:
            raise ValueError("decoder_base_ch must be >= 1 when set.")
        if self.decoder_num_res_blocks is not None and int(self.decoder_num_res_blocks) < 0:
            raise ValueError("decoder_num_res_blocks must be >= 0 when set.")
        norm = str(self.norm_type or "").strip().lower()
        if norm not in VAE_NORM_TYPES:
            raise ValueError(f"norm_type must be one of {VAE_NORM_TYPES}, got {self.norm_type!r}.")
        self.norm_type = norm
        activation = str(self.activation_type or "").strip().lower()
        if activation not in VAE_ACTIVATION_TYPES:
            raise ValueError(
                f"activation_type must be one of {VAE_ACTIVATION_TYPES}, got {self.activation_type!r}."
            )
        self.activation_type = activation
        decoder_type = str(self.decoder_type or "").strip().lower()
        if decoder_type not in VAE_DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {VAE_DECODER_TYPES}, got {self.decoder_type!r}.")
        self.decoder_type = decoder_type


@dataclass
class ChannelProtectionConfig:
    channel_protect_mode: str = "channel"
    channel_rank_metric: str = "channel_weight_abs"
    channel_mlp_rank_metric: str = "none"
    channel_mlp_fuse_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_scope: str = "layer"
    channel_min_per_layer: int = 0
    channel_quant: str = PROTECTED_CHANNEL_QUANT_NONE
    channel_axis: str = "input"
    channel_protect_count: Union[int, float] = 0

    def validate(self) -> None:
        mode = str(self.channel_protect_mode or "none").strip().lower()
        if mode not in CHANNEL_PROTECT_MODES:
            raise ValueError(
                f"channel_protect_mode must be one of {CHANNEL_PROTECT_MODES}, got {self.channel_protect_mode!r}."
            )
        self.channel_protect_mode = mode
        scope = str(self.channel_scope or "layer").strip().lower()
        if scope not in CHANNEL_SCOPES:
            raise ValueError(f"channel_scope must be one of {CHANNEL_SCOPES}, got {self.channel_scope!r}.")
        self.channel_scope = scope
        axis = str(self.channel_axis or "input").strip().lower()
        if axis not in CHANNEL_AXES:
            raise ValueError(f"channel_axis must be one of {CHANNEL_AXES}, got {self.channel_axis!r}.")
        self.channel_axis = axis
        metric = str(self.channel_rank_metric or "").strip().lower()
        if metric not in CHANNEL_RANK_METRICS:
            raise ValueError(
                f"channel_rank_metric must be one of {CHANNEL_RANK_METRICS}, got {self.channel_rank_metric!r}."
            )
        self.channel_rank_metric = metric
        mlp_metric = str(self.channel_mlp_rank_metric or "none").strip().lower()
        if mlp_metric not in CHANNEL_MLP_RANK_METRICS:
            raise ValueError(
                f"channel_mlp_rank_metric must be one of {CHANNEL_MLP_RANK_METRICS}, "
                f"got {self.channel_mlp_rank_metric!r}."
            )
        self.channel_mlp_rank_metric = mlp_metric
        weights = tuple(float(v) for v in self.channel_mlp_fuse_weights)
        if len(weights) != 3 or any(weight <= 0.0 or not math.isfinite(weight) for weight in weights):
            raise ValueError("channel_mlp_fuse_weights must be three finite floats > 0.")
        self.channel_mlp_fuse_weights = weights  # type: ignore[assignment]
        if int(self.channel_min_per_layer) < 0:
            raise ValueError("channel_min_per_layer must be >= 0.")
        self.channel_quant = normalize_protected_channel_quant_format(
            self.channel_quant,
            arg_name="channel_quant",
        )
        if self.channel_quant not in PROTECTED_CHANNEL_QUANT_CHOICES:
            raise ValueError(f"Unsupported channel_quant={self.channel_quant!r}.")
        if scope == "global":
            ratio = _require_finite(float(self.channel_protect_count), arg_name="channel_protect_count")
            if ratio < 0.0 or ratio >= 1.0:
                raise ValueError(
                    "channel_scope=global requires 0 <= channel_protect_count < 1, "
                    f"got {self.channel_protect_count}."
                )
            self.channel_protect_count = ratio
            if mode == "channel" and axis != "input" and ratio > 0.0:
                raise ValueError("channel_scope=global only acts on mode=channel + axis=input.")
        else:
            count = int(self.channel_protect_count)
            if count < 0:
                raise ValueError("channel_protect_count must be >= 0.")
            self.channel_protect_count = count
        if mlp_metric != "none" and scope != "layer":
            raise ValueError("channel_mlp_rank_metric is only valid when channel_scope=layer.")
        if mlp_metric != "none" and mode != "channel":
            raise ValueError("channel_mlp_rank_metric is only valid when channel_protect_mode=channel.")


@dataclass
class VAECompressionConfig:
    core: VAECoreConfig = field(default_factory=VAECoreConfig)
    decoder: VAEDecoderConfig = field(default_factory=VAEDecoderConfig)
    channel: ChannelProtectionConfig = field(default_factory=ChannelProtectionConfig)
    recon_loss_type: str = "mse"

    def validate(self) -> None:
        self.core.validate()
        self.decoder.validate()
        self.channel.validate()
        recon = str(self.recon_loss_type or "").strip().lower()
        if recon not in RECON_LOSS_TYPES:
            raise ValueError(f"recon_loss_type must be one of {RECON_LOSS_TYPES}, got {self.recon_loss_type!r}.")
        self.recon_loss_type = recon


@dataclass
class VAEOptimizationConfig:
    vae_steps: int = 2000
    vae_batch_size: int = 256
    vae_learning_rate: float = 1e-4
    vae_weight_decay: float = 1e-2
    vae_gradient_accumulation_steps: int = 1
    vae_max_grad_norm: Optional[float] = None
    vae_warmup_ratio: float = 0.0
    vae_lr_scheduler_type: str = "constant"
    vae_optim: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    l1_weight: float = 1.0
    lfq_weight: float = 1.0
    commitment_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    gpu_resident_data: bool = True
    log_every: int = 50
    eval_every: int = 0
    eval_blocks: int = 256

    def validate(self) -> None:
        if int(self.vae_steps) < 0:
            raise ValueError("vae_steps must be >= 0.")
        if int(self.vae_batch_size) < 1:
            raise ValueError("vae_batch_size must be >= 1.")
        self.vae_learning_rate = _require_finite(self.vae_learning_rate, arg_name="vae_learning_rate")
        self.vae_weight_decay = _require_finite(self.vae_weight_decay, arg_name="vae_weight_decay")
        if int(self.vae_gradient_accumulation_steps) < 1:
            raise ValueError("vae_gradient_accumulation_steps must be >= 1.")
        if self.vae_max_grad_norm is not None:
            self.vae_max_grad_norm = _require_finite(self.vae_max_grad_norm, arg_name="vae_max_grad_norm")
            if self.vae_max_grad_norm <= 0.0:
                raise ValueError("vae_max_grad_norm must be finite and > 0 when set; use None to disable.")
        self.vae_warmup_ratio = _require_finite(self.vae_warmup_ratio, arg_name="vae_warmup_ratio")
        if self.vae_warmup_ratio < 0.0 or self.vae_warmup_ratio > 1.0:
            raise ValueError("vae_warmup_ratio must be in [0, 1].")
        scheduler = str(self.vae_lr_scheduler_type or "").strip().lower()
        if scheduler not in VAE_LR_SCHEDULERS:
            raise ValueError(
                f"vae_lr_scheduler_type must be one of {VAE_LR_SCHEDULERS}, got {self.vae_lr_scheduler_type!r}."
            )
        self.vae_lr_scheduler_type = scheduler
        optim = str(self.vae_optim or "").strip().lower()
        if optim not in VAE_OPTIMS:
            raise ValueError(f"vae_optim must be one of {VAE_OPTIMS}, got {self.vae_optim!r}.")
        self.vae_optim = optim
        for name in (
            "beta1",
            "beta2",
            "l1_weight",
            "lfq_weight",
            "commitment_loss_weight",
            "entropy_loss_weight",
        ):
            setattr(self, name, _require_finite(getattr(self, name), arg_name=name))
        if int(self.log_every) < 0:
            raise ValueError("log_every must be >= 0.")
        if int(self.eval_every) < 0:
            raise ValueError("eval_every must be >= 0.")
        if int(self.eval_blocks) < 1:
            raise ValueError("eval_blocks must be >= 1.")

    def num_warmup_steps(self) -> int:
        return vae_num_warmup_steps(self.vae_warmup_ratio, self.vae_steps)


@dataclass
class ActivationCalibrationConfig:
    activation_calib_dataset: str = ""
    activation_calib_nsamples: int = 512
    activation_calib_seqlen: int = 512
    activation_calib_seed: int = 0
    activation_calib_device: str = ""
    activation_calib_log_every: int = 0

    def validate(self) -> None:
        dataset = str(self.activation_calib_dataset or "").strip()
        self.activation_calib_dataset = dataset
        if dataset:
            parse_dataset_mix_spec(dataset)
        if int(self.activation_calib_nsamples) < 1:
            raise ValueError("activation_calib_nsamples must be >= 1.")
        if int(self.activation_calib_seqlen) < 1:
            raise ValueError("activation_calib_seqlen must be >= 1.")
        if int(self.activation_calib_log_every) < 0:
            raise ValueError("activation_calib_log_every must be >= 0.")


@dataclass
class CandidateArtifactConfig:
    save_candidate_artifact: bool = False
    candidate_artifact_spec: Optional[str] = None
    candidate_artifact_output_dir: Optional[str] = None
    save_model: bool = False
    convert: bool = False

    def validate(self) -> None:
        if self.save_candidate_artifact and self.save_model:
            raise ValueError("save_candidate_artifact and save_model are mutually exclusive.")
        if self.save_candidate_artifact:
            if not self.convert:
                raise ValueError("save_candidate_artifact requires convert.")
            if not self.candidate_artifact_spec:
                raise ValueError("save_candidate_artifact requires candidate_artifact_spec.")
            if not self.candidate_artifact_output_dir:
                raise ValueError("save_candidate_artifact requires candidate_artifact_output_dir.")
        elif self.candidate_artifact_spec is not None or self.candidate_artifact_output_dir is not None:
            raise ValueError(
                "candidate_artifact_spec/candidate_artifact_output_dir require save_candidate_artifact."
            )


@dataclass
class CompressionTrainConfig:
    rot_llm: bool = False
    convert: bool = False
    convert_device: str = "cuda"
    save_model: bool = False
    output_dir: str = "./output_linear_by_category"
    resume_from_checkpoint: Optional[str] = None
    train_device: str = "cuda"
    deterministic: bool = False
    activation_calibration: ActivationCalibrationConfig = field(default_factory=ActivationCalibrationConfig)
    candidate_artifact: CandidateArtifactConfig = field(default_factory=CandidateArtifactConfig)

    def validate(self) -> None:
        self.activation_calibration.validate()
        self.candidate_artifact.save_model = bool(self.save_model)
        self.candidate_artifact.validate()
        if self.rot_llm and self.resume_from_checkpoint:
            raise ValueError("rot_llm is mutually exclusive with resume_from_checkpoint.")


def validate_train_mode_aux(train_mode: str, aux: AuxTrainableConfig) -> None:
    mode = parse_train_mode(train_mode)
    aux.validate()
    if mode == "none" and aux.norm_train_mode == "none" and aux.lm_head_train_mode == "none":
        raise ValueError(
            "train_mode=none requires norm_train_mode != none or lm_head_train_mode != none."
        )


@dataclass
class AfterCategoryResolvedConfig:
    data: DistillDataConfig
    loss: DistillLossConfig
    opt: DistillOptimizationConfig
    lora: LoRAConfig
    aux: AuxTrainableConfig
    runtime: DistillRuntimeConfig

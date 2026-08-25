import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import transformers

from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_FULL,
    VALID_LOW_RANK_SCOPES,
    normalize_low_rank_scope,
)
from litebsq.protected_channel_quant import (
    PROTECTED_CHANNEL_QUANT_CHOICES,
    PROTECTED_CHANNEL_QUANT_NONE,
    normalize_protected_channel_quant_format,
)
from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_CHOICES,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    SPARSE_RESIDUAL_INDEX_BITS_CHOICES,
    SPARSE_RESIDUAL_VALUE_BITS_CHOICES,
    get_default_block_shape_for_index_bits,
    validate_sparse_residual_block_shape,
)
from train_utils.lora_training import parse_distill_hidden_alignment_layer_weighting
from train_utils.cat_data_prep import normalize_intra_part_sort_mode
from train_utils.cat_arg_overrides import (
    OverrideSpec,
    OverrideTable,
    make_choice_parser,
    parse_bool_text,
    parse_float_text,
    parse_intra_part_sort_mode_text,
    parse_int_text,
    parse_optional_int_text,
    parse_override_table,
    resolve_after_category_value,
    resolve_category_value,
    validate_category_keys,
)
from train_utils.train_args import (
    HFArguments,
    _parse_bool_like,
    _parse_distill_loss_type,
)
from train_utils.mlp_channel_selection import is_mlp_aligned_rank_metric
from train_utils.utils import split_csv


@dataclass
class NormalizedCatArgs:
    target_categories: str
    transpose_modules: str
    include_all_linears: bool
    steps_per_category: OverrideTable[int]
    # 联合优化代码，已关闭。旧字段保留如下：
    # joint_decoder_steps: OverrideTable[Optional[int]]
    # joint_decoder_lr: OverrideTable[Optional[float]]
    # joint_decoder_group_size: OverrideTable[Optional[int]]
    # joint_decoder_batch_size: OverrideTable[Optional[int]]
    skip_layers: str
    linear_group_size: int
    intra_part_sort_mode: OverrideTable[str]
    batch_size: int
    gpu_resident_data: bool
    log_every: int
    eval_every: int
    eval_blocks: int
    # 排序代码，已关闭。旧字段保留如下：
    # sort_prep_workers: int
    outlier_protect_count: OverrideTable[int]
    outlier_protect_min_per_layer: int
    outlier_protect_mode: str
    outlier_channel_scope: str
    outlier_residual_top_p: OverrideTable[float]
    outlier_rank_metric: str
    outlier_mlp_rank_metric: str
    outlier_mlp_fuse_weights: Tuple[float, float, float]
    outlier_residual_min_abs: float
    outlier_residual_codec: str
    outlier_residual_index_bits: int
    outlier_residual_value_bits: int
    outlier_residual_block_shape: Tuple[int, int]
    outlier_protect_axis: str
    outlier_protect_channel_quant: str
    outlier_residual_vae_stages: OverrideTable[int]
    outlier_residual_vae_decoder_share_scope: str
    outlier_residual_vae_batch_multiplier: int
    outlier_residual_vae_steps: int
    outlier_residual_vae_lr: float
    outlier_residual_vae_codebook_bits: OverrideTable[int]
    outlier_residual_vae_codebook_dim: OverrideTable[int]
    activation_calib_dataset: str
    activation_calib_nsamples: int
    activation_calib_seqlen: int
    activation_calib_seed: int
    activation_calib_device: str
    activation_calib_log_every: int
    eval_ppl: bool
    eval_tasks: str
    ppl_limit: int
    eval_hif4_act: bool
    distill_after_category: str
    compressed_lora_scope: str
    distill_dataset: str
    lora_rank: OverrideTable[int]
    lora_alpha: OverrideTable[float]
    lora_dropout: OverrideTable[float]
    distill_steps: OverrideTable[int]
    distill_batch_size: OverrideTable[int]
    distill_lr: OverrideTable[float]
    distill_decoder_lr: OverrideTable[Optional[float]]
    distill_weight_decay: OverrideTable[float]
    distill_log_every: OverrideTable[int]
    distill_temperature: OverrideTable[float]
    distill_loss_alpha: OverrideTable[float]
    distill_loss_type: OverrideTable[str]
    distill_hidden_loss_weight: OverrideTable[float]
    distill_pre_mlp_hidden_loss_weight: OverrideTable[float]
    distill_prompt_kd_weight: OverrideTable[float]
    distill_hidden_alignment_layer_weighting: str
    distill_eakld_confidence_k: int
    lora_use_dora: OverrideTable[bool]
    distill_tune_final_norm: bool
    distill_use_post_norm_head_linear: bool
    seed: int
    deterministic: bool
    train_device: str
    rot_llm: bool
    resume_from_checkpoint: Optional[str]
    convert: bool
    convert_device: str
    save_model: bool
    save_candidate_artifact: bool
    candidate_artifact_spec: Optional[str]
    candidate_artifact_output_dir: Optional[str]
    distill_reset_completed: bool
    distill_independent_categories: bool
    output_dir: str
    allow_tail_group: bool


@dataclass
class CatTrainHFTrainingArguments:
    distill_model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length used by the after-category distill trainer."},
    )
    distill_dynamic_padding: bool = field(
        default=False,
        metadata={
            "help": (
                "Use per-micro-batch longest-sequence padding rounded up to a multiple "
                "of 8 for category distillation. distill_model_max_length remains the "
                "per-sample truncation ceiling."
            )
        },
    )
    distill_gradient_accumulation_steps: int = field(default=1)
    distill_optim: str = field(default="paged_adamw_8bit")
    distill_max_grad_norm: float = field(default=0.3)
    distill_warmup_ratio: float = field(default=0.3)
    distill_group_by_length: bool = field(default=True)
    distill_lr_scheduler_type: str = field(default="linear")
    distill_gradient_checkpointing: bool = field(default=False)
    distill_gradient_checkpointing_kwargs: Optional[str] = field(default=None)
    distill_hif4_act: bool = field(
        default=False,
        metadata={"help": "Enable HiFloat4 activation pseudo-quantization for student linear inputs during the after-category distill stage."},
    )
    distill_teacher_logits_cpu_staging: bool = field(
        default=True,
        metadata={
            "help": "After teacher forward, move teacher logits to CPU (bf16/fp16) until loss computation to reduce GPU peak memory."
        },
    )
    distill_teacher_model_offload: str = field(
        default="none",
        metadata={"help": "Teacher model residency: none or cpu."},
    )
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


@dataclass(frozen=True)
class ResolvedCategoryRuntimeConfig:
    category: str
    residual_stages: int
    steps: int
    # 联合优化代码，已关闭。旧字段保留如下：
    # joint_decoder_steps: int
    # joint_decoder_lr: float
    # joint_decoder_group_size: int
    # joint_decoder_batch_size: Optional[int]
    intra_part_sort_mode: str
    codebook_bits: int
    codebook_dim: int
    outlier_protect_count: int
    outlier_residual_top_p: float
    outlier_residual_vae_stages: int
    outlier_residual_vae_codebook_bits: int
    outlier_residual_vae_codebook_dim: int
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    activation_type: str
    decoder_type: str


@dataclass(frozen=True)
class ResolvedDistillRuntimeConfig:
    rank: int
    alpha: float
    dropout: float
    steps: int
    batch_size: int
    lr: float
    decoder_lr: Optional[float]
    weight_decay: float
    log_every: int
    temperature: float
    loss_alpha: float
    loss_type: str
    hidden_loss_weight: float
    pre_mlp_hidden_loss_weight: float
    prompt_kd_weight: float
    hidden_alignment_layer_weighting: str
    eakld_confidence_k: int
    use_dora: bool


_SKIP_LAYER_PATTERN = re.compile(r"^(\d+)\.([A-Za-z0-9_]+)$")
_CATEGORY_OVERRIDE_SELECTORS = ("default", "cat")
_AFTER_CATEGORY_OVERRIDE_SELECTORS = ("default", "after")
_CAT_RECON_LOSS_CHOICES = (
    "mse",
    "l1",
    "huber",
    "relative_l1",
    "w_mse",
    "w2_mse",
    "wa_mse",
    "amse",
)
_CAT_NORM_TYPE_CHOICES = ("group", "batch", "layer", "rms", "no")
_CAT_ACTIVATION_TYPE_CHOICES = ("swish", "relu", "none", "sigmoid", "gelu", "hard_swish")
_CAT_DECODER_TYPE_CHOICES = ("linear", "symmetric", "asymmetric")
_OUTLIER_PROTECT_MODE_CHOICES = ("none", "channel", "channel_residual_vae", "residual_sparse")
_OUTLIER_CHANNEL_SCOPE_CHOICES = ("layer", "category")
_OUTLIER_RESIDUAL_VAE_DECODER_SHARE_SCOPE_CHOICES = ("none", "category")
_DISTILL_HIDDEN_ALIGNMENT_LAYER_WEIGHTING_HELP = (
    "LoRA hidden alignment 层权重模式：uniform | linear_depth | adaptive | adaptive_top_<K>。"
    " adaptive 默认 K=3，仅对 teacher 相邻层 cosine 最低的 K 层计算 hidden 对齐损失。"
)
_DISTILL_AFTER_CATEGORY_CHOICES = (
    "none",
    "remaining_lora",
    "remaining_lora_decoder",
    "remaining_lora_all_decoder",
    "compressed_lora",
    "decoder",
    "both",
)
_DISTILL_AFTER_CATEGORY_REMAINING_MODES = {
    "remaining_lora",
    "remaining_lora_decoder",
    "remaining_lora_all_decoder",
}
_DISTILL_AFTER_CATEGORY_COMPRESSED_LORA_MODES = {"compressed_lora", "both"}
_OUTLIER_RANK_METRIC_CHOICES = (
    "sparse_residual_abs",
    "sparse_residual_actmax_abs",
    "sparse_residual_actmean_abs",
    "sparse_weight_abs",
    "sparse_weight_actmax_abs",
    "sparse_weight_actmean_abs",
    "channel_weight_abs",
    "channel_weight_actmax_abs",
    "channel_weight_actmean_abs",
    "channel_residual_abs",
    "channel_residual_actmax_abs",
    "channel_residual_actmean_abs",
    "channel_residual_actrms_abs",
)
_SPARSE_OUTLIER_RANK_METRICS = (
    "sparse_residual_abs",
    "sparse_residual_actmax_abs",
    "sparse_residual_actmean_abs",
    "sparse_weight_abs",
    "sparse_weight_actmax_abs",
    "sparse_weight_actmean_abs",
)
_CHANNEL_OUTLIER_RANK_METRICS = (
    "channel_weight_abs",
    "channel_weight_actmax_abs",
    "channel_weight_actmean_abs",
    "channel_residual_abs",
    "channel_residual_actmax_abs",
    "channel_residual_actmean_abs",
    "channel_residual_actrms_abs",
)
_CHANNEL_PRE_BASE_RANK_METRICS = (
    "channel_weight_abs",
    "channel_weight_actmax_abs",
    "channel_weight_actmean_abs",
)
_OUTLIER_RANK_METRICS_NEED_ACTMAX = (
    "sparse_residual_actmax_abs",
    "sparse_weight_actmax_abs",
    "channel_weight_actmax_abs",
    "channel_residual_actmax_abs",
)
_OUTLIER_RANK_METRICS_NEED_ACTMEAN = (
    "sparse_residual_actmean_abs",
    "sparse_weight_actmean_abs",
    "channel_weight_actmean_abs",
    "channel_residual_actmean_abs",
)
_OUTLIER_RANK_METRICS_NEED_ACT_SQ_MEAN = (
    "channel_residual_actrms_abs",
)
_OUTLIER_CHANNEL_MODES = ("channel", "channel_residual_vae")
_OUTLIER_MLP_RANK_METRIC_CHOICES = (
    "none",
    "mlp_intermediate_aligned_actrms",
    "mlp_intermediate_aligned_actmean_abs",
    "mlp_intermediate_aligned_actrms_abs",
)
_MLP_PROTECT_CATEGORIES = ("gate_proj", "up_proj", "down_proj")


def _normalize_target_categories(value: Optional[str]) -> str:
    categories = split_csv(None if value is None else str(value))
    if not categories:
        raise ValueError("--target_categories must not be empty.")
    reserved = [category for category in categories if category.strip().lower() in {"auto", "others"}]
    if reserved:
        raise ValueError(
            "--target_categories only accepts explicit categories; "
            f"unsupported values: {','.join(reserved)}"
        )
    seen: Set[str] = set()
    duplicates: List[str] = []
    for category in categories:
        if category in seen and category not in duplicates:
            duplicates.append(category)
        seen.add(category)
    if duplicates:
        raise ValueError(
            "--target_categories contains duplicate categories: "
            + ",".join(duplicates)
        )
    return ",".join(categories)


def parse_skip_layers(value: Optional[str]) -> Set[Tuple[int, str]]:
    entries = split_csv(None if value is None else str(value))
    out: Set[Tuple[int, str]] = set()
    for item in entries:
        match = _SKIP_LAYER_PATTERN.match(item)
        if not match:
            raise ValueError(
                f"Invalid --skip_layers entry '{item}'. Expected format: <layer_idx>.<category>, "
                "for example 0.down_proj or 30.q_proj."
            )
        out.add((int(match.group(1)), match.group(2)))
    return out


def resolve_skip_layer_matches(
    skip_layers: Optional[str],
    discovered_keys: Sequence[Tuple[int, str]],
) -> Tuple[Set[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    requested = parse_skip_layers(skip_layers)
    discovered_set = {(int(layer_idx), str(category)) for layer_idx, category in discovered_keys}
    matched = sorted(requested & discovered_set)
    missing = sorted(requested - discovered_set)
    return requested, matched, missing


def _parse_distill_loss_alpha_text(raw: str, *, arg_name: str) -> float:
    value = parse_float_text(raw, arg_name=arg_name, min_value=0.0, inclusive_min=True)
    if value > 1.0:
        raise argparse.ArgumentTypeError(f"{arg_name} must be <= 1.0, got {value}.")
    return float(value)


def _parse_nonnegative_float_text(raw: str, *, arg_name: str) -> float:
    return float(parse_float_text(raw, arg_name=arg_name, min_value=0.0, inclusive_min=True))


def _parse_optional_distill_decoder_lr_text(raw: object) -> Optional[float]:
    text = str(raw).strip().lower()
    if text == "none":
        return None
    return float(parse_float_text(raw, arg_name="--distill_decoder_lr"))


def _parse_distill_dataset_mix_text(raw: str, *, arg_name: str) -> str:
    try:
        from e2e_common.data import normalize_dataset_mix_spec

        _sources, _weights, normalized_spec = normalize_dataset_mix_spec(raw)
        return str(normalized_spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _normalize_distill_after_category(raw: str) -> str:
    mode = str(raw or "none").strip().lower()
    if mode not in _DISTILL_AFTER_CATEGORY_CHOICES:
        raise ValueError(
            "--distill_after_category must be one of: "
            f"{', '.join(_DISTILL_AFTER_CATEGORY_CHOICES)}."
        )
    return mode


def _normalize_distill_dataset_arg(raw: str, *, distill_after_category: str) -> str:
    value = str(raw or "").strip()
    mode = _normalize_distill_after_category(distill_after_category)
    if mode == "none":
        return value
    if not value:
        raise ValueError("--distill_dataset must be set when --distill_after_category is enabled.")
    if "=" not in value:
        raise ValueError(
            "--distill_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )
    return _parse_distill_dataset_mix_text(value, arg_name="--distill_dataset")


def _parse_activation_calib_dataset_text(raw: str, *, arg_name: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"{arg_name} only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )
    return _parse_distill_dataset_mix_text(value, arg_name=arg_name)


def _make_override_spec(
    *,
    arg_name: str,
    parse_value,
    allowed_selectors: Sequence[str],
    example: str,
) -> OverrideSpec:
    return OverrideSpec(
        arg_name=arg_name,
        parse_value=parse_value,
        allowed_selectors=tuple(str(selector) for selector in allowed_selectors),
        example=example,
    )


def _parse_cat_override(raw: str, *, spec: OverrideSpec):
    return parse_override_table(raw, spec)


def _make_positive_int_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    min_value: int = 1,
) -> OverrideSpec:
    return _make_override_spec(
        arg_name=arg_name,
        parse_value=lambda raw: parse_int_text(raw, arg_name=arg_name, min_value=min_value),
        allowed_selectors=allowed_selectors,
        example=example,
    )


def _make_optional_int_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    min_value: int,
) -> OverrideSpec:
    return _make_override_spec(
        arg_name=arg_name,
        parse_value=lambda raw: parse_optional_int_text(raw, arg_name=arg_name, min_value=min_value),
        allowed_selectors=allowed_selectors,
        example=example,
    )


def _make_choice_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    choices: Sequence[str],
) -> OverrideSpec:
    return _make_override_spec(
        arg_name=arg_name,
        parse_value=make_choice_parser(arg_name=arg_name, choices=choices),
        allowed_selectors=allowed_selectors,
        example=example,
    )


_STEPS_PER_CATEGORY_SPEC = _make_positive_int_override_spec(
    arg_name="--steps_per_category",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=2000,cat:down_proj=1000",
)
# 联合优化代码，已关闭。旧 joint decoder override spec 保留如下：
# _JOINT_DECODER_STEPS_SPEC = _make_optional_int_override_spec(
#     arg_name="--joint_decoder_steps",
#     allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
#     example="default=none,cat:down_proj=500",
#     min_value=0,
# )
# _JOINT_DECODER_LR_SPEC = _make_override_spec(
#     arg_name="--joint_decoder_lr",
#     parse_value=lambda raw: (
#         None if str(raw).strip().lower() == "none"
#         else parse_float_text(raw, arg_name="--joint_decoder_lr", min_value=0.0, inclusive_min=False)
#     ),
#     allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
#     example="default=none,cat:down_proj=5e-5",
# )
# _JOINT_DECODER_GROUP_SIZE_SPEC = _make_optional_int_override_spec(
#     arg_name="--joint_decoder_group_size",
#     allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
#     example="default=none,cat:down_proj=2",
#     min_value=1,
# )
# _JOINT_DECODER_BATCH_SIZE_SPEC = _make_optional_int_override_spec(
#     arg_name="--joint_decoder_batch_size",
#     allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
#     example="default=none,cat:down_proj=1024",
#     min_value=1,
# )
# 排序代码，已关闭：保留 spec 仅用于内部固定 default=none 的结构化配置。
_INTRA_PART_SORT_MODE_SPEC = _make_override_spec(
    arg_name="--intra_part_sort_mode",
    parse_value=lambda raw: parse_intra_part_sort_mode_text(raw, arg_name="--intra_part_sort_mode"),
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=none",
)
_OUTLIER_PROTECT_COUNT_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_protect_count",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0,cat:down_proj=64",
    min_value=0,
)
_OUTLIER_RESIDUAL_TOP_P_SPEC = _make_override_spec(
    arg_name="--outlier_residual_top_p",
    parse_value=lambda raw: parse_float_text(
        raw,
        arg_name="--outlier_residual_top_p",
        min_value=0.0,
        inclusive_min=True,
    ),
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.01,cat:down_proj=0.02",
)
_OUTLIER_RESIDUAL_VAE_STAGES_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_residual_vae_stages",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1,cat:q_proj=2",
)
_OUTLIER_RESIDUAL_VAE_CODEBOOK_BITS_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_residual_vae_codebook_bits",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0,cat:q_proj=4",
    min_value=0,
)
_OUTLIER_RESIDUAL_VAE_CODEBOOK_DIM_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_residual_vae_codebook_dim",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0,cat:q_proj=8",
    min_value=0,
)
_CODEBOOK_BITS_SPEC = _make_positive_int_override_spec(
    arg_name="--codebook_bits",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=16,cat:q_proj=24",
)
_CODEBOOK_DIM_SPEC = _make_positive_int_override_spec(
    arg_name="--codebook_dim",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=8,cat:down_proj=16",
)
_RESIDUAL_STAGES_SPEC = _make_positive_int_override_spec(
    arg_name="--residual_stages",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1,cat:q_proj=2",
)
_BASE_CH_SPEC = _make_positive_int_override_spec(
    arg_name="--base_ch",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=128,cat:q_proj=192",
)
_NUM_RES_BLOCKS_SPEC = _make_positive_int_override_spec(
    arg_name="--num_res_blocks",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1,cat:down_proj=2",
    min_value=0,
)
_DECODER_BASE_CH_SPEC = _make_optional_int_override_spec(
    arg_name="--decoder_base_ch",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=none,cat:q_proj=256",
    min_value=1,
)
_DECODER_NUM_RES_BLOCKS_SPEC = _make_optional_int_override_spec(
    arg_name="--decoder_num_res_blocks",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=none,cat:q_proj=0",
    min_value=0,
)
_RECON_LOSS_TYPE_SPEC = _make_choice_override_spec(
    arg_name="--recon_loss_type",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=mse,cat:q_proj=wa_mse,cat:down_proj=amse",
    choices=_CAT_RECON_LOSS_CHOICES,
)
_NORM_TYPE_SPEC = _make_choice_override_spec(
    arg_name="--norm_type",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=group,cat:q_proj=layer",
    choices=_CAT_NORM_TYPE_CHOICES,
)
_ACTIVATION_TYPE_SPEC = _make_choice_override_spec(
    arg_name="--activation_type",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=swish,cat:q_proj=relu",
    choices=_CAT_ACTIVATION_TYPE_CHOICES,
)
_DECODER_TYPE_SPEC = _make_choice_override_spec(
    arg_name="--decoder_type",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=linear,cat:q_proj=asymmetric",
    choices=_CAT_DECODER_TYPE_CHOICES,
)
_LORA_RANK_SPEC = _make_positive_int_override_spec(
    arg_name="--lora_rank",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=8,after:q_proj=16",
)
_LORA_ALPHA_SPEC = _make_override_spec(
    arg_name="--lora_alpha",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_alpha", min_value=0.0, inclusive_min=False),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=16.0,after:q_proj=32.0",
)
_LORA_DROPOUT_SPEC = _make_override_spec(
    arg_name="--lora_dropout",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_dropout", min_value=0.0, inclusive_min=True),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:q_proj=0.1",
)
_DISTILL_STEPS_SPEC = _make_positive_int_override_spec(
    arg_name="--distill_steps",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=50,after:q_proj=200",
    min_value=0,
)
_DISTILL_BATCH_SIZE_SPEC = _make_positive_int_override_spec(
    arg_name="--distill_batch_size",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=2,after:q_proj=4",
)
_DISTILL_LR_SPEC = _make_override_spec(
    arg_name="--distill_lr",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--distill_lr"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1e-4,after:q_proj=5e-5",
)
_DISTILL_DECODER_LR_SPEC = _make_override_spec(
    arg_name="--distill_decoder_lr",
    parse_value=_parse_optional_distill_decoder_lr_text,
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=5e-5,after:gate_proj=3e-5",
)
_DISTILL_WEIGHT_DECAY_SPEC = _make_override_spec(
    arg_name="--distill_weight_decay",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--distill_weight_decay"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:q_proj=0.01",
)
_DISTILL_LOG_EVERY_SPEC = _make_positive_int_override_spec(
    arg_name="--distill_log_every",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1,after:q_proj=10",
)
_DISTILL_TEMPERATURE_SPEC = _make_override_spec(
    arg_name="--distill_temperature",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--distill_temperature", min_value=0.0, inclusive_min=False),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1.0,after:q_proj=2.0",
)
_DISTILL_LOSS_ALPHA_SPEC = _make_override_spec(
    arg_name="--distill_loss_alpha",
    parse_value=lambda raw: _parse_distill_loss_alpha_text(raw, arg_name="--distill_loss_alpha"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.5,after:q_proj=0.3",
)
_DISTILL_LOSS_TYPE_SPEC = _make_override_spec(
    arg_name="--distill_loss_type",
    parse_value=lambda raw: _parse_distill_loss_type(str(raw)),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=sft,after:q_proj=dual_kl_top_1000",
)
_DISTILL_HIDDEN_LOSS_WEIGHT_SPEC = _make_override_spec(
    arg_name="--distill_hidden_loss_weight",
    parse_value=lambda raw: parse_float_text(
        raw,
        arg_name="--distill_hidden_loss_weight",
        min_value=0.0,
        inclusive_min=True,
    ),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:q_proj=0.01",
)
_DISTILL_PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC = _make_override_spec(
    arg_name="--distill_pre_mlp_hidden_loss_weight",
    parse_value=lambda raw: parse_float_text(
        raw,
        arg_name="--distill_pre_mlp_hidden_loss_weight",
        min_value=0.0,
        inclusive_min=True,
    ),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:o_proj=0.01",
)
_DISTILL_PROMPT_KD_WEIGHT_SPEC = _make_override_spec(
    arg_name="--distill_prompt_kd_weight",
    parse_value=lambda raw: _parse_nonnegative_float_text(raw, arg_name="--distill_prompt_kd_weight"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:q_proj=0.05",
)
_LORA_USE_DORA_SPEC = _make_override_spec(
    arg_name="--lora_use_dora",
    parse_value=lambda raw: parse_bool_text(raw, arg_name="--lora_use_dora"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=true,after:q_proj=false",
)


def _build_cat_train_vae_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--codebook_bits", type=str, default="default=16", help=f"Category overrides. Example: {_CODEBOOK_BITS_SPEC.example}")
    parser.add_argument("--codebook_dim", type=str, default="default=8", help=f"Category overrides. Example: {_CODEBOOK_DIM_SPEC.example}")
    parser.add_argument("--residual_stages", type=str, default="default=1", help=f"Category overrides. Example: {_RESIDUAL_STAGES_SPEC.example}")
    parser.add_argument("--base_ch", type=str, default="default=128", help=f"Category overrides. Example: {_BASE_CH_SPEC.example}")
    parser.add_argument("--num_res_blocks", type=str, default="default=1", help=f"Category overrides. Example: {_NUM_RES_BLOCKS_SPEC.example}")
    parser.add_argument(
        "--decoder_base_ch",
        "--decoder_hidden_dim",
        dest="decoder_base_ch",
        type=str,
        default="default=none",
        help=f"Category overrides. Example: {_DECODER_BASE_CH_SPEC.example}",
    )
    parser.add_argument(
        "--decoder_num_res_blocks",
        type=str,
        default="default=none",
        help=f"Category overrides. Example: {_DECODER_NUM_RES_BLOCKS_SPEC.example}",
    )
    parser.add_argument("--quantizer_type", type=str, default="BSQ")
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)
    parser.add_argument("--norm_type", type=str, default="default=group", help=f"Category overrides. Example: {_NORM_TYPE_SPEC.example}")
    parser.add_argument("--activation_type", type=str, default="default=swish", help=f"Category overrides. Example: {_ACTIVATION_TYPE_SPEC.example}")
    parser.add_argument("--decoder_type", type=str, default="default=linear", help=f"Category overrides. Example: {_DECODER_TYPE_SPEC.example}")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "linear", "cosine"], help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup steps for scheduler")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path or HuggingFace ID of the LLM")
    parser.add_argument("--normalize_weight", action="store_true", help="Normalize weight (z-score) before training")
    parser.add_argument("--recon_loss_type", type=str, default="default=mse", help=f"Category overrides. Example: {_RECON_LOSS_TYPE_SPEC.example}")
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lfq_weight", type=float, default=1.0)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--vae_decoder_checkpoint",
        type=lambda v: _parse_bool_like(v, arg_name="--vae_decoder_checkpoint"),
        default=None,
        help="Override VAE decoder activation checkpointing.",
    )
    parser.add_argument("--new_quant", action="store_true")
    return parser


def _normalize_cat_train_script_args(raw_args) -> NormalizedCatArgs:
    resolved_index_bits = int(raw_args.outlier_residual_index_bits)
    resolved_block_shape = get_default_block_shape_for_index_bits(resolved_index_bits)
    resolved_block_shape = validate_sparse_residual_block_shape(
        block_rows=int(resolved_block_shape[0]),
        block_cols=int(resolved_block_shape[1]),
        index_bits=resolved_index_bits,
        arg_name="derived sparse residual block shape",
    )
    return NormalizedCatArgs(
        target_categories=_normalize_target_categories(raw_args.target_categories),
        transpose_modules=str(raw_args.transpose_modules),
        include_all_linears=bool(raw_args.include_all_linears),
        steps_per_category=_parse_cat_override(raw_args.steps_per_category, spec=_STEPS_PER_CATEGORY_SPEC),
        # 联合优化代码，已关闭。原 joint CLI 解析保留如下：
        # joint_decoder_steps=_parse_cat_override(raw_args.joint_decoder_steps, spec=_JOINT_DECODER_STEPS_SPEC),
        # joint_decoder_lr=_parse_cat_override(raw_args.joint_decoder_lr, spec=_JOINT_DECODER_LR_SPEC),
        # joint_decoder_group_size=_parse_cat_override(raw_args.joint_decoder_group_size, spec=_JOINT_DECODER_GROUP_SIZE_SPEC),
        # joint_decoder_batch_size=_parse_cat_override(raw_args.joint_decoder_batch_size, spec=_JOINT_DECODER_BATCH_SIZE_SPEC),
        skip_layers=str(raw_args.skip_layers),
        linear_group_size=int(raw_args.linear_group_size),
        intra_part_sort_mode=_parse_cat_override("default=none", spec=_INTRA_PART_SORT_MODE_SPEC),
        batch_size=int(raw_args.batch_size),
        gpu_resident_data=bool(raw_args.gpu_resident_data),
        log_every=int(raw_args.log_every),
        eval_every=int(raw_args.eval_every),
        eval_blocks=int(raw_args.eval_blocks),
        # 排序代码，已关闭。旧字段赋值保留如下：
        # sort_prep_workers=1,
        outlier_protect_count=_parse_cat_override(raw_args.outlier_protect_count, spec=_OUTLIER_PROTECT_COUNT_SPEC),
        outlier_protect_min_per_layer=int(raw_args.outlier_protect_min_per_layer),
        outlier_protect_mode=str(raw_args.outlier_protect_mode).strip().lower(),
        outlier_channel_scope=str(raw_args.outlier_channel_scope).strip().lower(),
        outlier_residual_top_p=_parse_cat_override(raw_args.outlier_residual_top_p, spec=_OUTLIER_RESIDUAL_TOP_P_SPEC),
        outlier_rank_metric=str(raw_args.outlier_rank_metric).strip().lower(),
        outlier_mlp_rank_metric=str(raw_args.outlier_mlp_rank_metric).strip().lower(),
        outlier_mlp_fuse_weights=_parse_outlier_mlp_fuse_weights_text(
            raw_args.outlier_mlp_fuse_weights,
            arg_name="--outlier_mlp_fuse_weights",
        ),
        outlier_residual_min_abs=float(raw_args.outlier_residual_min_abs),
        outlier_residual_codec=str(raw_args.outlier_residual_codec).strip().lower(),
        outlier_residual_index_bits=resolved_index_bits,
        outlier_residual_value_bits=int(raw_args.outlier_residual_value_bits),
        outlier_residual_block_shape=tuple(int(v) for v in resolved_block_shape),
        outlier_protect_axis=str(raw_args.outlier_protect_axis).strip().lower(),
        outlier_protect_channel_quant=normalize_protected_channel_quant_format(
            raw_args.outlier_protect_channel_quant,
            arg_name="--outlier_protect_channel_quant",
        ),
        outlier_residual_vae_stages=_parse_cat_override(raw_args.outlier_residual_vae_stages, spec=_OUTLIER_RESIDUAL_VAE_STAGES_SPEC),
        outlier_residual_vae_codebook_bits=_parse_cat_override(
            raw_args.outlier_residual_vae_codebook_bits,
            spec=_OUTLIER_RESIDUAL_VAE_CODEBOOK_BITS_SPEC,
        ),
        outlier_residual_vae_codebook_dim=_parse_cat_override(
            raw_args.outlier_residual_vae_codebook_dim,
            spec=_OUTLIER_RESIDUAL_VAE_CODEBOOK_DIM_SPEC,
        ),
        outlier_residual_vae_decoder_share_scope=str(raw_args.outlier_residual_vae_decoder_share_scope).strip().lower(),
        outlier_residual_vae_batch_multiplier=int(raw_args.outlier_residual_vae_batch_multiplier),
        outlier_residual_vae_steps=int(raw_args.outlier_residual_vae_steps),
        outlier_residual_vae_lr=float(raw_args.outlier_residual_vae_lr),
        activation_calib_dataset=_parse_activation_calib_dataset_text(
            raw_args.activation_calib_dataset,
            arg_name="--activation_calib_dataset",
        ),
        activation_calib_nsamples=int(raw_args.activation_calib_nsamples),
        activation_calib_seqlen=int(raw_args.activation_calib_seqlen),
        activation_calib_seed=int(raw_args.activation_calib_seed),
        activation_calib_device=str(raw_args.activation_calib_device),
        activation_calib_log_every=int(raw_args.activation_calib_log_every),
        eval_ppl=bool(raw_args.eval_ppl),
        eval_tasks=str(raw_args.eval_tasks),
        ppl_limit=int(raw_args.ppl_limit),
        eval_hif4_act=bool(raw_args.eval_hif4_act),
        distill_after_category=_normalize_distill_after_category(raw_args.distill_after_category),
        compressed_lora_scope=normalize_low_rank_scope(raw_args.compressed_lora_scope),
        distill_dataset=_normalize_distill_dataset_arg(
            raw_args.distill_dataset,
            distill_after_category=str(raw_args.distill_after_category),
        ),
        lora_rank=_parse_cat_override(raw_args.lora_rank, spec=_LORA_RANK_SPEC),
        lora_alpha=_parse_cat_override(raw_args.lora_alpha, spec=_LORA_ALPHA_SPEC),
        lora_dropout=_parse_cat_override(raw_args.lora_dropout, spec=_LORA_DROPOUT_SPEC),
        distill_steps=_parse_cat_override(raw_args.distill_steps, spec=_DISTILL_STEPS_SPEC),
        distill_batch_size=_parse_cat_override(raw_args.distill_batch_size, spec=_DISTILL_BATCH_SIZE_SPEC),
        distill_lr=_parse_cat_override(raw_args.distill_lr, spec=_DISTILL_LR_SPEC),
        distill_decoder_lr=_parse_cat_override(raw_args.distill_decoder_lr, spec=_DISTILL_DECODER_LR_SPEC),
        distill_weight_decay=_parse_cat_override(raw_args.distill_weight_decay, spec=_DISTILL_WEIGHT_DECAY_SPEC),
        distill_log_every=_parse_cat_override(raw_args.distill_log_every, spec=_DISTILL_LOG_EVERY_SPEC),
        distill_temperature=_parse_cat_override(raw_args.distill_temperature, spec=_DISTILL_TEMPERATURE_SPEC),
        distill_loss_alpha=_parse_cat_override(raw_args.distill_loss_alpha, spec=_DISTILL_LOSS_ALPHA_SPEC),
        distill_loss_type=_parse_cat_override(raw_args.distill_loss_type, spec=_DISTILL_LOSS_TYPE_SPEC),
        distill_hidden_loss_weight=_parse_cat_override(raw_args.distill_hidden_loss_weight, spec=_DISTILL_HIDDEN_LOSS_WEIGHT_SPEC),
        distill_pre_mlp_hidden_loss_weight=_parse_cat_override(
            raw_args.distill_pre_mlp_hidden_loss_weight,
            spec=_DISTILL_PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC,
        ),
        distill_prompt_kd_weight=_parse_cat_override(
            raw_args.distill_prompt_kd_weight,
            spec=_DISTILL_PROMPT_KD_WEIGHT_SPEC,
        ),
        distill_hidden_alignment_layer_weighting=parse_distill_hidden_alignment_layer_weighting(
            raw_args.distill_hidden_alignment_layer_weighting
        ),
        distill_eakld_confidence_k=int(raw_args.distill_eakld_confidence_k),
        lora_use_dora=_parse_cat_override(raw_args.lora_use_dora, spec=_LORA_USE_DORA_SPEC),
        distill_tune_final_norm=bool(raw_args.distill_tune_final_norm),
        distill_use_post_norm_head_linear=bool(raw_args.distill_use_post_norm_head_linear),
        seed=int(raw_args.seed),
        deterministic=bool(raw_args.deterministic),
        train_device=str(raw_args.train_device),
        rot_llm=bool(raw_args.rot_llm),
        resume_from_checkpoint=None if raw_args.resume_from_checkpoint is None else str(raw_args.resume_from_checkpoint),
        convert=bool(raw_args.convert),
        convert_device=str(raw_args.convert_device),
        save_model=bool(raw_args.save_model),
        save_candidate_artifact=bool(raw_args.save_candidate_artifact),
        candidate_artifact_spec=(
            None
            if raw_args.candidate_artifact_spec is None
            else str(raw_args.candidate_artifact_spec)
        ),
        candidate_artifact_output_dir=(
            None
            if raw_args.candidate_artifact_output_dir is None
            else str(raw_args.candidate_artifact_output_dir)
        ),
        distill_reset_completed=bool(raw_args.distill_reset_completed),
        distill_independent_categories=bool(raw_args.distill_independent_categories),
        output_dir=str(raw_args.output_dir),
        allow_tail_group=bool(raw_args.allow_tail_group),
    )


def _iter_override_entries(table: OverrideTable[object]):
    if bool(getattr(table, "has_default", False)):
        yield "default", getattr(table, "default")
    for category, value in sorted(getattr(table, "by_category", {}).items()):
        yield f"cat:{category}", value
    for category, value in sorted(getattr(table, "by_after_category", {}).items()):
        yield f"after:{category}", value


def _override_table_contains(table: OverrideTable[object], predicate) -> bool:
    return any(bool(predicate(value)) for _selector, value in _iter_override_entries(table))


def _parse_outlier_mlp_fuse_weights_text(value: str, *, arg_name: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{arg_name} must contain exactly 3 comma-separated floats, got {value!r}.")
    parsed = tuple(float(part) for part in parts)
    if any(weight <= 0.0 for weight in parsed):
        raise ValueError(f"{arg_name} entries must be > 0, got {value!r}.")
    return parsed  # type: ignore[return-value]


def _mlp_aligned_rank_metric_enabled(cat_args: NormalizedCatArgs) -> bool:
    return is_mlp_aligned_rank_metric(cat_args.outlier_mlp_rank_metric)


def _validate_outlier_mlp_args(cat_args: NormalizedCatArgs) -> None:
    metric = str(cat_args.outlier_mlp_rank_metric).strip().lower()
    if metric not in _OUTLIER_MLP_RANK_METRIC_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_mlp_rank_metric={cat_args.outlier_mlp_rank_metric!r}. "
            f"Expected one of: {', '.join(_OUTLIER_MLP_RANK_METRIC_CHOICES)}."
        )
    if metric == "none":
        return

    mode = str(cat_args.outlier_protect_mode).strip().lower()
    if mode != "channel":
        raise ValueError(
            f"--outlier_mlp_rank_metric={metric!r} is only valid when --outlier_protect_mode=channel."
        )
    if str(cat_args.outlier_channel_scope).strip().lower() != "layer":
        raise ValueError(
            f"--outlier_mlp_rank_metric={metric!r} is only valid when --outlier_channel_scope=layer."
        )
    if not str(cat_args.activation_calib_dataset).strip():
        raise ValueError(
            f"--outlier_mlp_rank_metric={metric!r} requires --activation_calib_dataset."
        )

    categories = split_csv(cat_args.target_categories)
    missing = [cat for cat in _MLP_PROTECT_CATEGORIES if cat not in categories]
    if missing:
        raise ValueError(
            f"--outlier_mlp_rank_metric={metric!r} requires target_categories to include "
            + ",".join(_MLP_PROTECT_CATEGORIES)
            + f", missing: {','.join(missing)}."
        )

    counts = {
        cat: int(resolve_category_value(cat_args.outlier_protect_count, cat))
        for cat in _MLP_PROTECT_CATEGORIES
    }
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        raise ValueError(
            "--outlier_mlp_rank_metric requires equal --outlier_protect_count for "
            "gate_proj, up_proj, and down_proj. Got: "
            + ",".join(f"{cat}={counts[cat]}" for cat in _MLP_PROTECT_CATEGORIES)
        )


def _validate_dynamic_calib_dataset_args(cat_args: NormalizedCatArgs, vae_args) -> None:
    channel_needs_activation = (
        str(cat_args.outlier_protect_mode).strip().lower() in _OUTLIER_CHANNEL_MODES
        and (
            str(cat_args.outlier_rank_metric).strip().lower() in _OUTLIER_RANK_METRICS_NEED_ACTMAX
            or str(cat_args.outlier_rank_metric).strip().lower() in _OUTLIER_RANK_METRICS_NEED_ACTMEAN
            or str(cat_args.outlier_rank_metric).strip().lower() in _OUTLIER_RANK_METRICS_NEED_ACT_SQ_MEAN
        )
        and _override_table_contains(cat_args.outlier_protect_count, lambda value: int(value) > 0)
    )
    mlp_channel_needs_activation = (
        _mlp_aligned_rank_metric_enabled(cat_args)
        and str(cat_args.outlier_protect_mode).strip().lower() == "channel"
        and any(
            int(resolve_category_value(cat_args.outlier_protect_count, cat)) > 0
            for cat in _MLP_PROTECT_CATEGORIES
            if cat in split_csv(cat_args.target_categories)
        )
    )
    dynamic_calib_enabled = (
        _override_table_contains(
            vae_args.recon_loss_type,
            lambda value: str(value).strip().lower() == "wa_mse",
        )
        or channel_needs_activation
        or mlp_channel_needs_activation
        or (
            str(cat_args.outlier_protect_mode).strip().lower() == "residual_sparse"
            and (
                str(cat_args.outlier_rank_metric).strip().lower() in _OUTLIER_RANK_METRICS_NEED_ACTMAX
                or str(cat_args.outlier_rank_metric).strip().lower() in _OUTLIER_RANK_METRICS_NEED_ACTMEAN
            )
        )
    )
    if dynamic_calib_enabled and not str(cat_args.activation_calib_dataset).strip():
        raise ValueError(
            "--activation_calib_dataset must be set when dynamic activation calibration is enabled. "
            "Use ratio-style dataset specs such as 'wiki=1.0', 'openorca=1.0' or "
            "'openorca=0.5,fineweb_edu=0.5'."
        )


def _validate_distill_after_category_args(cat_args: NormalizedCatArgs) -> None:
    mode = _normalize_distill_after_category(cat_args.distill_after_category)
    enabled = []
    if bool(cat_args.distill_tune_final_norm):
        enabled.append("--distill_tune_final_norm")
    if bool(cat_args.distill_use_post_norm_head_linear):
        enabled.append("--distill_use_post_norm_head_linear")
    if enabled and mode not in _DISTILL_AFTER_CATEGORY_REMAINING_MODES:
        raise ValueError(
            f"{', '.join(enabled)} is only supported with remaining-family --distill_after_category modes."
        )
    if mode in _DISTILL_AFTER_CATEGORY_COMPRESSED_LORA_MODES:
        for _selector, use_dora in _iter_override_entries(cat_args.lora_use_dora):
            if bool(use_dora):
                raise ValueError(
                    f"--distill_after_category={mode} does not support --lora_use_dora=true."
                )


def _validate_distill_lr_scheduler_args(training_args) -> None:
    scheduler = str(getattr(training_args, "distill_lr_scheduler_type", "linear") or "linear").strip().lower()
    warmup_ratio = float(getattr(training_args, "distill_warmup_ratio", 0.0) or 0.0)
    if scheduler == "constant" and warmup_ratio > 0.0:
        raise ValueError(
            "--distill_lr_scheduler_type=constant ignores warmup. "
            "Use --distill_lr_scheduler_type=constant_with_warmup when "
            f"--distill_warmup_ratio={warmup_ratio} > 0, or set --distill_warmup_ratio 0."
        )


def _validate_distill_teacher_model_offload_args(training_args) -> None:
    mode = str(getattr(training_args, "distill_teacher_model_offload", "none")).strip().lower()
    if mode not in {"none", "cpu"}:
        raise ValueError("--distill_teacher_model_offload must be one of: none, cpu.")
    training_args.distill_teacher_model_offload = mode


def validate_outlier_rank_metric(
    outlier_mode: str,
    outlier_rank_metric: str,
    *,
    channel_mode_uses_metric: bool = True,
) -> None:
    mode = str(outlier_mode).strip().lower()
    metric = str(outlier_rank_metric).strip().lower()
    if metric not in _OUTLIER_RANK_METRIC_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_rank_metric={outlier_rank_metric!r}. "
            f"Expected one of: {', '.join(_OUTLIER_RANK_METRIC_CHOICES)}."
        )
    if mode == "residual_sparse":
        if metric not in _SPARSE_OUTLIER_RANK_METRICS:
            raise ValueError(
                f"outlier_rank_metric={metric!r} is only valid for "
                "outlier_mode='channel_residual_vae', but got outlier_mode='residual_sparse'."
            )
        return
    if mode == "channel_residual_vae":
        if metric not in _CHANNEL_OUTLIER_RANK_METRICS:
            raise ValueError(
                f"outlier_rank_metric={metric!r} is only valid for "
                "outlier_mode='residual_sparse', but got outlier_mode='channel_residual_vae'."
            )
        return
    if mode == "channel" and bool(channel_mode_uses_metric) and metric not in _CHANNEL_PRE_BASE_RANK_METRICS:
        raise ValueError(
            f"outlier_rank_metric={metric!r} is not valid for outlier_mode='channel'. "
            f"Expected one of: {', '.join(_CHANNEL_PRE_BASE_RANK_METRICS)}."
        )


def _validate_outlier_protect_mode_args(cat_args: NormalizedCatArgs) -> None:
    mode = str(cat_args.outlier_protect_mode).strip().lower()
    codec = str(cat_args.outlier_residual_codec).strip().lower()
    index_bits = int(cat_args.outlier_residual_index_bits)
    value_bits = int(cat_args.outlier_residual_value_bits)
    block_rows, block_cols = tuple(int(v) for v in cat_args.outlier_residual_block_shape)
    top_p_table = cat_args.outlier_residual_top_p
    protect_table = cat_args.outlier_protect_count
    min_per_layer = int(cat_args.outlier_protect_min_per_layer)
    residual_vae_batch_multiplier = int(cat_args.outlier_residual_vae_batch_multiplier)
    residual_vae_steps = int(cat_args.outlier_residual_vae_steps)
    residual_vae_lr = float(cat_args.outlier_residual_vae_lr)
    if mode not in _OUTLIER_PROTECT_MODE_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_protect_mode={cat_args.outlier_protect_mode!r}. "
            f"Expected one of: {', '.join(_OUTLIER_PROTECT_MODE_CHOICES)}."
        )
    if str(cat_args.outlier_channel_scope).strip().lower() not in _OUTLIER_CHANNEL_SCOPE_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_channel_scope={cat_args.outlier_channel_scope!r}. "
            f"Expected one of: {', '.join(_OUTLIER_CHANNEL_SCOPE_CHOICES)}."
        )
    if min_per_layer < 0:
        raise ValueError(f"--outlier_protect_min_per_layer must be >= 0, got {min_per_layer}.")
    channel_quant = normalize_protected_channel_quant_format(cat_args.outlier_protect_channel_quant)
    if channel_quant != PROTECTED_CHANNEL_QUANT_NONE:
        if mode != "channel":
            raise ValueError(
                f"--outlier_protect_channel_quant={cat_args.outlier_protect_channel_quant!r} "
                f"is only valid when --outlier_protect_mode=channel."
            )
        if not _override_table_contains(protect_table, lambda value: int(value) > 0):
            raise ValueError(
                f"--outlier_protect_channel_quant={cat_args.outlier_protect_channel_quant!r} "
                "requires --outlier_protect_count > 0 for at least one selector."
            )
    invalid_min_entries = [
        f"{selector}={int(value)}"
        for selector, value in _iter_override_entries(protect_table)
        if int(value) < min_per_layer
    ]
    if invalid_min_entries:
        raise ValueError(
            "--outlier_protect_min_per_layer must be <= --outlier_protect_count for every selector. "
            f"min_per_layer={min_per_layer}, counts: " + ",".join(invalid_min_entries)
        )
    if (
        str(cat_args.outlier_residual_vae_decoder_share_scope).strip().lower()
        not in _OUTLIER_RESIDUAL_VAE_DECODER_SHARE_SCOPE_CHOICES
    ):
        raise ValueError(
            f"Unsupported --outlier_residual_vae_decoder_share_scope="
            f"{cat_args.outlier_residual_vae_decoder_share_scope!r}. "
            f"Expected one of: {', '.join(_OUTLIER_RESIDUAL_VAE_DECODER_SHARE_SCOPE_CHOICES)}."
        )
    if codec not in SPARSE_RESIDUAL_FORMAT_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_residual_codec={cat_args.outlier_residual_codec!r}. "
            f"Expected one of: {', '.join(SPARSE_RESIDUAL_FORMAT_CHOICES)}."
        )
    if index_bits not in SPARSE_RESIDUAL_INDEX_BITS_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_residual_index_bits={index_bits}. "
            f"Expected one of: {SPARSE_RESIDUAL_INDEX_BITS_CHOICES}."
        )
    if value_bits not in SPARSE_RESIDUAL_VALUE_BITS_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_residual_value_bits={value_bits}. "
            f"Expected one of: {SPARSE_RESIDUAL_VALUE_BITS_CHOICES}."
        )
    validate_sparse_residual_block_shape(
        block_rows=block_rows,
        block_cols=block_cols,
        index_bits=index_bits,
        arg_name="derived sparse residual block shape",
    )
    channel_mode_uses_metric = (
        mode == "channel"
        and _override_table_contains(protect_table, lambda value: int(value) > 0)
    )
    validate_outlier_rank_metric(
        mode,
        cat_args.outlier_rank_metric,
        channel_mode_uses_metric=channel_mode_uses_metric,
    )
    if residual_vae_batch_multiplier < 1:
        raise ValueError(
            f"--outlier_residual_vae_batch_multiplier must be >= 1, got {residual_vae_batch_multiplier}."
        )
    if residual_vae_steps < 0:
        raise ValueError("--outlier_residual_vae_steps must be >= 0.")
    if residual_vae_lr < 0.0:
        raise ValueError("--outlier_residual_vae_lr must be >= 0.")
    if float(cat_args.outlier_residual_min_abs) < 0.0:
        raise ValueError(
            f"--outlier_residual_min_abs must be >= 0, got {float(cat_args.outlier_residual_min_abs)}."
        )
    invalid_top_p_entries = [
        f"{selector}={float(value)}"
        for selector, value in _iter_override_entries(top_p_table)
        if not (0.0 <= float(value) <= 1.0)
    ]
    if invalid_top_p_entries:
        raise ValueError(
            "--outlier_residual_top_p must satisfy 0 <= p <= 1 for every selector. Got: "
            + ",".join(invalid_top_p_entries)
        )
    if mode in {"none", "channel", "channel_residual_vae"}:
        nonzero_top_p_entries = [
            f"{selector}={float(value)}"
            for selector, value in _iter_override_entries(top_p_table)
            if float(value) != 0.0
        ]
        if mode in _OUTLIER_CHANNEL_MODES and nonzero_top_p_entries:
            raise ValueError(
                "--outlier_residual_top_p must be 0 for every selector when "
                f"--outlier_protect_mode={mode}. Got: " + ",".join(nonzero_top_p_entries)
            )
        return
    nonzero_entries = []
    if bool(getattr(protect_table, "has_default", False)) and int(getattr(protect_table, "default", 0)) != 0:
        nonzero_entries.append(f"default={int(getattr(protect_table, 'default', 0))}")
    nonzero_entries.extend(
        f"cat:{category}={int(value)}"
        for category, value in sorted(getattr(protect_table, "by_category", {}).items())
        if int(value) != 0
    )
    if nonzero_entries:
        raise ValueError(
            "--outlier_protect_count must be 0 for every selector when "
            "--outlier_protect_mode=residual_sparse. Got: " + ",".join(nonzero_entries)
        )


def _normalize_cat_train_vae_args(raw_args):
    args = argparse.Namespace(**vars(raw_args))
    args.codebook_bits = _parse_cat_override(raw_args.codebook_bits, spec=_CODEBOOK_BITS_SPEC)
    args.codebook_dim = _parse_cat_override(raw_args.codebook_dim, spec=_CODEBOOK_DIM_SPEC)
    args.residual_stages = _parse_cat_override(raw_args.residual_stages, spec=_RESIDUAL_STAGES_SPEC)
    args.base_ch = _parse_cat_override(raw_args.base_ch, spec=_BASE_CH_SPEC)
    args.num_res_blocks = _parse_cat_override(raw_args.num_res_blocks, spec=_NUM_RES_BLOCKS_SPEC)
    args.decoder_base_ch = _parse_cat_override(raw_args.decoder_base_ch, spec=_DECODER_BASE_CH_SPEC)
    args.decoder_num_res_blocks = _parse_cat_override(raw_args.decoder_num_res_blocks, spec=_DECODER_NUM_RES_BLOCKS_SPEC)
    args.recon_loss_type = _parse_cat_override(raw_args.recon_loss_type, spec=_RECON_LOSS_TYPE_SPEC)
    args.norm_type = _parse_cat_override(raw_args.norm_type, spec=_NORM_TYPE_SPEC)
    args.activation_type = _parse_cat_override(raw_args.activation_type, spec=_ACTIVATION_TYPE_SPEC)
    args.decoder_type = _parse_cat_override(raw_args.decoder_type, spec=_DECODER_TYPE_SPEC)
    return args


def resolve_category_runtime_configs(cat_args: NormalizedCatArgs, vae_args, active_categories: Sequence[str]) -> Dict[str, ResolvedCategoryRuntimeConfig]:
    resolved_outlier_mode = str(cat_args.outlier_protect_mode).strip().lower()
    tables = (
        (cat_args.steps_per_category, "--steps_per_category"),
        # 联合优化代码，已关闭。旧 joint decoder category table 校验保留如下：
        # (cat_args.joint_decoder_steps, "--joint_decoder_steps"),
        # (cat_args.joint_decoder_lr, "--joint_decoder_lr"),
        # (cat_args.joint_decoder_group_size, "--joint_decoder_group_size"),
        # (cat_args.joint_decoder_batch_size, "--joint_decoder_batch_size"),
        (cat_args.intra_part_sort_mode, "--intra_part_sort_mode"),
        (cat_args.outlier_protect_count, "--outlier_protect_count"),
        (cat_args.outlier_residual_top_p, "--outlier_residual_top_p"),
        (cat_args.outlier_residual_vae_stages, "--outlier_residual_vae_stages"),
        (cat_args.outlier_residual_vae_codebook_bits, "--outlier_residual_vae_codebook_bits"),
        (cat_args.outlier_residual_vae_codebook_dim, "--outlier_residual_vae_codebook_dim"),
        (vae_args.codebook_bits, "--codebook_bits"),
        (vae_args.codebook_dim, "--codebook_dim"),
        (vae_args.residual_stages, "--residual_stages"),
        (vae_args.base_ch, "--base_ch"),
        (vae_args.num_res_blocks, "--num_res_blocks"),
        (vae_args.decoder_base_ch, "--decoder_base_ch"),
        (vae_args.decoder_num_res_blocks, "--decoder_num_res_blocks"),
        (vae_args.recon_loss_type, "--recon_loss_type"),
        (vae_args.norm_type, "--norm_type"),
        (vae_args.activation_type, "--activation_type"),
        (vae_args.decoder_type, "--decoder_type"),
    )
    for table, arg_name in tables:
        validate_category_keys(table, active_categories, arg_name)

    resolved: Dict[str, ResolvedCategoryRuntimeConfig] = {}
    for category in active_categories:
        steps_per_category = resolve_category_value(cat_args.steps_per_category, category)
        # 联合优化代码，已关闭。旧 joint decoder runtime 解析保留如下：
        # joint_decoder_steps = resolve_category_value(cat_args.joint_decoder_steps, category)
        # joint_decoder_lr = resolve_category_value(cat_args.joint_decoder_lr, category)
        # joint_decoder_group_size = resolve_category_value(cat_args.joint_decoder_group_size, category)
        # joint_decoder_batch_size = resolve_category_value(cat_args.joint_decoder_batch_size, category)
        # resolved_joint_decoder_steps = int(steps_per_category) if joint_decoder_steps is None else int(joint_decoder_steps)
        # resolved_joint_decoder_lr = float(vae_args.lr) if joint_decoder_lr is None else float(joint_decoder_lr)
        # resolved_joint_decoder_group_size = (
        #     int(cat_args.linear_group_size)
        #     if joint_decoder_group_size is None
        #     else int(joint_decoder_group_size)
        # )
        # resolved_joint_decoder_batch_size = (
        #     None if joint_decoder_batch_size is None else int(joint_decoder_batch_size)
        # )
        resolved_outlier_residual_top_p = float(resolve_category_value(cat_args.outlier_residual_top_p, category))
        if resolved_outlier_mode in _OUTLIER_CHANNEL_MODES and resolved_outlier_residual_top_p != 0.0:
            raise ValueError(
                f"--outlier_residual_top_p resolved to {resolved_outlier_residual_top_p} for category "
                f"'{category}', but --outlier_protect_mode={resolved_outlier_mode} requires 0."
            )
        if resolved_outlier_mode == "residual_sparse" and not (0.0 < resolved_outlier_residual_top_p <= 1.0):
            raise ValueError(
                f"--outlier_residual_top_p resolved to {resolved_outlier_residual_top_p} for category "
                f"'{category}', but --outlier_protect_mode=residual_sparse requires 0 < p <= 1."
            )
        resolved_codebook_bits = int(resolve_category_value(vae_args.codebook_bits, category))
        resolved_codebook_dim = int(resolve_category_value(vae_args.codebook_dim, category))
        requested_residual_codebook_bits = int(
            resolve_category_value(cat_args.outlier_residual_vae_codebook_bits, category)
        )
        requested_residual_codebook_dim = int(
            resolve_category_value(cat_args.outlier_residual_vae_codebook_dim, category)
        )
        resolved_residual_codebook_bits = (
            int(requested_residual_codebook_bits)
            if int(requested_residual_codebook_bits) > 0
            else int(resolved_codebook_bits)
        )
        resolved_residual_codebook_dim = (
            int(requested_residual_codebook_dim)
            if int(requested_residual_codebook_dim) > 0
            else int(resolved_codebook_dim)
        )
        resolved[category] = ResolvedCategoryRuntimeConfig(
            category=str(category),
            residual_stages=int(resolve_category_value(vae_args.residual_stages, category)),
            steps=int(steps_per_category),
            # 联合优化代码，已关闭。旧字段赋值保留如下：
            # joint_decoder_steps=int(resolved_joint_decoder_steps),
            # joint_decoder_lr=float(resolved_joint_decoder_lr),
            # joint_decoder_group_size=int(resolved_joint_decoder_group_size),
            # joint_decoder_batch_size=resolved_joint_decoder_batch_size,
            intra_part_sort_mode=normalize_intra_part_sort_mode(
                resolve_category_value(cat_args.intra_part_sort_mode, category),
                arg_name="--intra_part_sort_mode",
            ),
            codebook_bits=int(resolved_codebook_bits),
            codebook_dim=int(resolved_codebook_dim),
            outlier_protect_count=int(resolve_category_value(cat_args.outlier_protect_count, category)),
            outlier_residual_top_p=resolved_outlier_residual_top_p,
            outlier_residual_vae_stages=int(resolve_category_value(cat_args.outlier_residual_vae_stages, category)),
            outlier_residual_vae_codebook_bits=int(resolved_residual_codebook_bits),
            outlier_residual_vae_codebook_dim=int(resolved_residual_codebook_dim),
            recon_loss_type=str(resolve_category_value(vae_args.recon_loss_type, category)).strip().lower(),
            base_ch=int(resolve_category_value(vae_args.base_ch, category)),
            num_res_blocks=int(resolve_category_value(vae_args.num_res_blocks, category)),
            decoder_base_ch=resolve_category_value(vae_args.decoder_base_ch, category),
            decoder_num_res_blocks=resolve_category_value(vae_args.decoder_num_res_blocks, category),
            norm_type=str(resolve_category_value(vae_args.norm_type, category)).strip().lower(),
            activation_type=str(resolve_category_value(vae_args.activation_type, category)).strip().lower(),
            decoder_type=str(resolve_category_value(vae_args.decoder_type, category)).strip().lower(),
        )
    return resolved


def resolve_distill_runtime_config(cat_args: NormalizedCatArgs, after_category: Optional[str]) -> ResolvedDistillRuntimeConfig:
    return ResolvedDistillRuntimeConfig(
        rank=int(resolve_after_category_value(cat_args.lora_rank, after_category)),
        alpha=float(resolve_after_category_value(cat_args.lora_alpha, after_category)),
        dropout=float(resolve_after_category_value(cat_args.lora_dropout, after_category)),
        steps=int(resolve_after_category_value(cat_args.distill_steps, after_category)),
        batch_size=int(resolve_after_category_value(cat_args.distill_batch_size, after_category)),
        lr=float(resolve_after_category_value(cat_args.distill_lr, after_category)),
        decoder_lr=resolve_after_category_value(cat_args.distill_decoder_lr, after_category),
        weight_decay=float(resolve_after_category_value(cat_args.distill_weight_decay, after_category)),
        log_every=int(resolve_after_category_value(cat_args.distill_log_every, after_category)),
        temperature=float(resolve_after_category_value(cat_args.distill_temperature, after_category)),
        loss_alpha=float(resolve_after_category_value(cat_args.distill_loss_alpha, after_category)),
        loss_type=str(resolve_after_category_value(cat_args.distill_loss_type, after_category)),
        hidden_loss_weight=float(resolve_after_category_value(cat_args.distill_hidden_loss_weight, after_category)),
        pre_mlp_hidden_loss_weight=float(
            resolve_after_category_value(cat_args.distill_pre_mlp_hidden_loss_weight, after_category)
        ),
        prompt_kd_weight=float(resolve_after_category_value(cat_args.distill_prompt_kd_weight, after_category)),
        hidden_alignment_layer_weighting=str(cat_args.distill_hidden_alignment_layer_weighting),
        eakld_confidence_k=int(cat_args.distill_eakld_confidence_k),
        use_dora=bool(resolve_after_category_value(cat_args.lora_use_dora, after_category)),
    )


def build_cat_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--target_categories",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="要压缩的类别及顺序，必须是显式逗号分隔列表。",
    )
    parser.add_argument("--transpose_modules", type=str, default="v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--include_all_linears",
        action="store_true",
        default=False,
        help="关闭默认的 decoder projection 路径限制，但仍只收集 target_categories 指定的类别。",
    )
    parser.add_argument("--steps_per_category", type=str, default="default=2000", help=f"类别覆盖参数。示例：{_STEPS_PER_CATEGORY_SPEC.example}")
    # 联合优化代码，已关闭：不再注册 joint decoder CLI。
    # parser.add_argument("--joint_decoder_steps", type=str, default="default=none", help=f"类别覆盖参数。示例：{_JOINT_DECODER_STEPS_SPEC.example}")
    # parser.add_argument("--joint_decoder_lr", type=str, default="default=none", help=f"类别覆盖参数。示例：{_JOINT_DECODER_LR_SPEC.example}")
    # parser.add_argument("--joint_decoder_group_size", type=str, default="default=none", help=f"类别覆盖参数。示例：{_JOINT_DECODER_GROUP_SIZE_SPEC.example}")
    # parser.add_argument("--joint_decoder_batch_size", type=str, default="default=none", help=f"类别覆盖参数。示例：{_JOINT_DECODER_BATCH_SIZE_SPEC.example}")
    parser.add_argument("--skip_layers", type=str, default="", help="指定在 LLM 前向中始终使用原始线性权重的层，格式: layer_idx.category，例如 0.down_proj,30.q_proj。")
    parser.add_argument("--linear_group_size", type=int, default=32, help="跨层分组大小：每组同时训练多少个同类 Linear。")
    # 排序代码，已关闭：不再注册 --intra_part_sort_mode CLI。
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--gpu_resident_data",
        type=lambda raw: parse_bool_text(raw, arg_name="--gpu_resident_data"),
        default=False,
        help="是否把当前 VAE residual stage 的训练数据常驻 GPU。只影响搬运方式，不改变 batch size。",
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_blocks", type=int, default=256)
    # 排序代码，已关闭：不再注册 --sort_prep_workers CLI。
    parser.add_argument("--outlier_protect_count", type=str, default="default=0", help=f"类别覆盖参数。示例：{_OUTLIER_PROTECT_COUNT_SPEC.example}")
    parser.add_argument("--outlier_protect_min_per_layer", type=int, default=0)
    parser.add_argument(
        "--outlier_protect_mode",
        type=str,
        choices=list(_OUTLIER_PROTECT_MODE_CHOICES),
        default="channel",
        help="离群值保护模式：none 为关闭，channel 为压缩前保护通道，channel_residual_vae 为训练后 VAE 压缩通道残差，residual_sparse 为训练后保存残差补丁。",
    )
    parser.add_argument(
        "--outlier_channel_scope",
        type=str,
        choices=list(_OUTLIER_CHANNEL_SCOPE_CHOICES),
        default="layer",
        help="channel/channel_residual_vae 模式下的通道预算范围：layer 为每层独立，category 为同类全局排序。",
    )
    parser.add_argument(
        "--outlier_residual_top_p",
        type=str,
        default="default=0.0",
        help=f"类别覆盖参数。仅 residual_sparse 模式生效。示例：{_OUTLIER_RESIDUAL_TOP_P_SPEC.example}",
    )
    parser.add_argument(
        "--outlier_rank_metric",
        type=str,
        choices=list(_OUTLIER_RANK_METRIC_CHOICES),
        default="sparse_residual_abs",
        help=(
            "outlier/protected target 排序指标。residual_sparse 使用 sparse_*；"
            "channel_residual_vae 使用 channel_*。"
        ),
    )
    parser.add_argument(
        "--outlier_mlp_rank_metric",
        type=str,
        choices=list(_OUTLIER_MLP_RANK_METRIC_CHOICES),
        default="none",
        help=(
            "MLP gate/up/down 专用选道指标。none 表示 MLP 仍走 --outlier_rank_metric 的 per-linear 逻辑；"
            "mlp_intermediate_aligned_actrms / actmean_abs / actrms_abs 表示按 SwiGLU intermediate path 共享保护通道。"
        ),
    )
    parser.add_argument(
        "--outlier_mlp_fuse_weights",
        type=str,
        default="1,1,1",
        help="MLP aligned 选道融合权重。格式：alpha_up,alpha_gate,alpha_down。",
    )
    parser.add_argument(
        "--outlier_residual_min_abs",
        type=lambda v: _parse_nonnegative_float_text(v, arg_name="--outlier_residual_min_abs"),
        default=1e-6,
        help="仅 residual_sparse 模式生效。若 |original-reconstructed| < 该阈值，则该位置不允许进入 sparse residual。",
    )
    parser.add_argument(
        "--outlier_residual_codec",
        type=str,
        choices=list(SPARSE_RESIDUAL_FORMAT_CHOICES),
        default=SPARSE_RESIDUAL_FORMAT_COO_FP16,
        help="仅 residual_sparse 模式生效。残差存储格式：coo_fp16 或 blocked_quantized。",
    )
    parser.add_argument(
        "--outlier_residual_index_bits",
        type=int,
        choices=list(SPARSE_RESIDUAL_INDEX_BITS_CHOICES),
        default=8,
        help="仅 blocked_quantized 生效。块内索引位宽：4 或 8。",
    )
    parser.add_argument(
        "--outlier_residual_value_bits",
        type=int,
        choices=list(SPARSE_RESIDUAL_VALUE_BITS_CHOICES),
        default=8,
        help="仅 blocked_quantized 生效。残差 value 量化位宽：4 或 8。",
    )
    parser.add_argument("--outlier_protect_axis", type=str, choices=["input", "output"], default="input", help="Choose whether outlier protection preserves input channels or output channels.")
    parser.add_argument(
        "--outlier_protect_channel_quant",
        type=str,
        choices=list(PROTECTED_CHANNEL_QUANT_CHOICES),
        default=PROTECTED_CHANNEL_QUANT_NONE,
        help="仅 channel 模式生效。protected channel 权重存储格式：none / fp8_e4m3 / fp8_e5m2 / int8。",
    )
    parser.add_argument("--outlier_residual_vae_stages", type=str, default="default=1", help=f"channel_residual_vae 模式下 protected channel residual VAE 阶数。示例：{_OUTLIER_RESIDUAL_VAE_STAGES_SPEC.example}")
    parser.add_argument(
        "--outlier_residual_vae_codebook_bits",
        type=str,
        default="default=0",
        help=(
            "channel_residual_vae 模式下 protected residual VAE 的 codebook bits。"
            "0 表示继承 base VAE --codebook_bits。"
            f"示例：{_OUTLIER_RESIDUAL_VAE_CODEBOOK_BITS_SPEC.example}"
        ),
    )
    parser.add_argument(
        "--outlier_residual_vae_codebook_dim",
        type=str,
        default="default=0",
        help=(
            "channel_residual_vae 模式下 protected residual VAE 的 codebook dim。"
            "0 表示继承 base VAE --codebook_dim。"
            f"示例：{_OUTLIER_RESIDUAL_VAE_CODEBOOK_DIM_SPEC.example}"
        ),
    )
    parser.add_argument(
        "--outlier_residual_vae_decoder_share_scope",
        type=str,
        choices=list(_OUTLIER_RESIDUAL_VAE_DECODER_SHARE_SCOPE_CHOICES),
        default="none",
        help="channel_residual_vae 模式下 protected residual VAE decoder 共享范围：none 为每个 linear 独立，category 为同类别共享。",
    )
    parser.add_argument(
        "--outlier_residual_vae_batch_multiplier",
        type=int,
        default=1,
        help="channel_residual_vae 模式下 protected residual VAE 的 batch 放大倍数；只影响 residual VAE，不影响 base VAE。category share scope 下建议设为 32。",
    )
    parser.add_argument(
        "--outlier_residual_vae_steps",
        type=int,
        default=0,
        help="Number of optimization steps for protected residual VAE. If 0, reuse the base VAE residual stage steps.",
    )
    parser.add_argument(
        "--outlier_residual_vae_lr",
        type=float,
        default=0.0,
        help="Learning rate for protected residual VAE. If 0, reuse the base VAE learning rate.",
    )
    parser.add_argument(
        "--activation_calib_dataset",
        type=lambda raw: _parse_activation_calib_dataset_text(raw, arg_name="--activation_calib_dataset"),
        default="",
        help="Calibration dataset for dynamic activation stats collection. "
        "Required when dynamic calibration is enabled. Format: alias=weight,alias=weight. "
        "For example: wiki=1.0, openorca=1.0, or openorca=0.5,fineweb_edu=0.5.",
    )
    parser.add_argument("--activation_calib_nsamples", type=int, default=512, help="Calibration sample count for dynamic activation stats collection.")
    parser.add_argument("--activation_calib_seqlen", type=int, default=512, help="Calibration sequence length for dynamic activation stats collection.")
    parser.add_argument("--activation_calib_seed", type=int, default=0, help="Calibration sampling seed for dynamic activation stats collection.")
    parser.add_argument("--activation_calib_device", type=str, default="", help="Device for dynamic activation stats collection. Empty means use --train_device.")
    parser.add_argument("--activation_calib_log_every", type=int, default=0, help="Log interval for dynamic activation stats collection progress (0 to disable).")
    parser.add_argument(
        "--eval_ppl",
        type=lambda v: _parse_bool_like(v, arg_name="--eval_ppl"),
        default=True,
        help="是否在 cat_train 内部的类别后评估阶段运行 PPL。",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="类别后评估的 lm_eval 任务列表，逗号分隔；空串表示不跑下游任务。",
    )
    parser.add_argument("--ppl_limit", type=int, default=-1, help="每类训练后 PPL 评估样本上限，-1 为全量。")
    parser.add_argument(
        "--eval_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--eval_hif4_act"),
        default=False,
        help="是否在 cat_train 内部的 PPL 评估阶段启用 HiFloat4 激活伪量化。",
    )
    parser.add_argument(
        "--distill_after_category",
        type=str,
        choices=list(_DISTILL_AFTER_CATEGORY_CHOICES),
        default="none",
        help=(
            "每个类别 VAE 训练后的蒸馏模式：none 不蒸馏；remaining_lora 保留旧的剩余 dense Linear LoRA；"
            "compressed_lora 只给刚压缩类别挂 proxy LoRA；decoder 只微调刚压缩类别 decoder；both 同时训练两者。"
        ),
    )
    parser.add_argument(
        "--compressed_lora_scope",
        type=str,
        choices=sorted(VALID_LOW_RANK_SCOPES),
        default=LOW_RANK_SCOPE_FULL,
        help=(
            "compressed_lora / both 模式下最终写入 VAELinear 的 LoRA 作用域："
            "full 作用于完整权重；compressed_subspace 只作用于 VAE 压缩子空间。"
        ),
    )
    parser.add_argument(
        "--distill_dataset",
        type=str,
        default="",
        help=(
            "每类后蒸馏训练数据集比例串。开启 --distill_after_category 非 none 时必填；"
            "格式: alias=weight,alias=weight，例如 wiki=1.0、openorca=1.0 或 "
            "openorca=0.5,fineweb_edu=0.5。支持 dense_e2e 的 dataset_mix alias。"
        ),
    )
    parser.add_argument("--lora_rank", type=str, default="default=8", help=f"after_category 覆盖参数。示例：{_LORA_RANK_SPEC.example}")
    parser.add_argument("--lora_alpha", type=str, default="default=16.0", help=f"after_category 覆盖参数。示例：{_LORA_ALPHA_SPEC.example}")
    parser.add_argument("--lora_dropout", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_LORA_DROPOUT_SPEC.example}")
    parser.add_argument("--distill_steps", type=str, default="default=50", help=f"after_category 覆盖参数。示例：{_DISTILL_STEPS_SPEC.example}")
    parser.add_argument("--distill_batch_size", type=str, default="default=2", help=f"after_category 覆盖参数。示例：{_DISTILL_BATCH_SIZE_SPEC.example}")
    parser.add_argument("--distill_lr", type=str, default="default=1e-4", help=f"after_category 覆盖参数。示例：{_DISTILL_LR_SPEC.example}")
    parser.add_argument("--distill_decoder_lr", type=str, default="default=none", help=f"after_category 覆盖参数。示例：{_DISTILL_DECODER_LR_SPEC.example}")
    parser.add_argument("--distill_weight_decay", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_DISTILL_WEIGHT_DECAY_SPEC.example}")
    parser.add_argument("--distill_log_every", type=str, default="default=1", help=f"after_category 覆盖参数。示例：{_DISTILL_LOG_EVERY_SPEC.example}")
    parser.add_argument("--distill_temperature", type=str, default="default=1.0", help=f"after_category 覆盖参数。示例：{_DISTILL_TEMPERATURE_SPEC.example}")
    parser.add_argument("--distill_loss_alpha", type=str, default="default=0.5", help=f"after_category 覆盖参数。示例：{_DISTILL_LOSS_ALPHA_SPEC.example}")
    parser.add_argument("--distill_loss_type", type=str, default="default=eakld", help=f"after_category 覆盖参数。示例：{_DISTILL_LOSS_TYPE_SPEC.example}")
    parser.add_argument("--distill_hidden_loss_weight", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_DISTILL_HIDDEN_LOSS_WEIGHT_SPEC.example}")
    parser.add_argument(
        "--distill_pre_mlp_hidden_loss_weight",
        type=str,
        default="default=0.0",
        help=f"after_category 覆盖参数。示例：{_DISTILL_PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC.example}",
    )
    parser.add_argument(
        "--distill_prompt_kd_weight",
        type=str,
        default="default=0.0",
        help=f"after_category 覆盖参数。示例：{_DISTILL_PROMPT_KD_WEIGHT_SPEC.example}",
    )
    parser.add_argument(
        "--distill_hidden_alignment_layer_weighting",
        type=parse_distill_hidden_alignment_layer_weighting,
        default="uniform",
        help=_DISTILL_HIDDEN_ALIGNMENT_LAYER_WEIGHTING_HELP,
    )
    parser.add_argument(
        "--distill_eakld_confidence_k",
        type=int,
        default=16,
        help="EAKLD 熵归一化常数 K（非 vocab top-k）。用于 eakld / eakld_kd。",
    )
    parser.add_argument("--lora_use_dora", type=str, default="default=true", help=f"after_category 覆盖参数。示例：{_LORA_USE_DORA_SPEC.example}")
    parser.add_argument(
        "--distill_tune_final_norm",
        type=lambda v: _parse_bool_like(v, arg_name="--distill_tune_final_norm"),
        default=False,
        help="每类后蒸馏阶段是否同时微调模型最终 norm。",
    )
    parser.add_argument(
        "--distill_use_post_norm_head_linear",
        type=lambda v: _parse_bool_like(v, arg_name="--distill_use_post_norm_head_linear"),
        default=False,
        help="每类后蒸馏阶段是否训练 post-norm head linear；最终保存前会融合回 lm_head。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        type=lambda v: _parse_bool_like(v, arg_name="--deterministic"),
        default=False,
        help="启用严格确定性模式；遇到非确定性 CUDA 算子会直接报错。",
    )
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument(
        "--rot_llm",
        type=lambda v: _parse_bool_like(v, arg_name="--rot_llm"),
        nargs="?",
        const=True,
        default=False,
        help="在 VAE 压缩前先对基座 LLM 执行一次离线旋转融合。",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从已有 cat_train checkpoint 继续训练。可传 run 目录、final_model 目录，或 checkpoint_meta.json。",
    )
    parser.add_argument("--convert", action="store_true", help="每个类别训练完成后，将 Linear 替换为压缩后的线性层。")
    parser.add_argument("--convert_device", type=str, default="cuda")
    parser.add_argument("--save_model", action="store_true", help="保存最终模型 state_dict/config/tokenizer（需要 --convert）。")
    parser.add_argument(
        "--save_candidate_artifact",
        action="store_true",
        help="仅导出当前目标类别的压缩 VAELinear candidate artifact（与 --save_model 互斥）。",
    )
    parser.add_argument(
        "--candidate_artifact_spec",
        type=str,
        default=None,
        help="candidate-only 导出所需的 trial_spec.json 路径。",
    )
    parser.add_argument(
        "--candidate_artifact_output_dir",
        type=str,
        default=None,
        help="candidate-only 导出目录。",
    )
    parser.add_argument(
        "--distill_reset_completed",
        type=lambda v: _parse_bool_like(v, arg_name="--distill_reset_completed"),
        default=False,
        help=(
            "checkpoint distill 时忽略 resume ckpt 中的 completed_categories，对 target_categories "
            "全量再跑一轮类别蒸馏。对已有 low_rank_a/b 的类：用其初始化 proxy LoRA 继续训并覆盖写回；"
            "decoder 在已有权重上继续。默认 false：按 completed_categories / 已有 low_rank 跳过，"
            "用于从 after_<category> 续跑未完成类。"
        ),
    )
    parser.add_argument(
        "--distill_independent_categories",
        type=lambda v: _parse_bool_like(v, arg_name="--distill_independent_categories"),
        default=False,
        help=(
            "checkpoint distill 时每类独立蒸馏：训当前类前把已完成类恢复为未压缩 Linear，"
            "不累积前缀压缩状态；全部类结束后再一次性激活已完成类做最终评估/保存。"
            "默认 false（前缀累积）。仅对 cat checkpoint distill 生效。"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./output_linear_by_category")
    parser.add_argument("--allow_tail_group", type=lambda v: _parse_bool_like(v, arg_name="--allow_tail_group"), default=True, help="是否允许处理最后一个不足分组大小的尾部分组（true/false）。")
    return parser


def _validate_candidate_artifact_args(cat_args: NormalizedCatArgs) -> None:
    save_candidate = bool(cat_args.save_candidate_artifact)
    save_model = bool(cat_args.save_model)
    spec = cat_args.candidate_artifact_spec
    out_dir = cat_args.candidate_artifact_output_dir
    if save_candidate and save_model:
        raise ValueError("--save_candidate_artifact and --save_model are mutually exclusive")
    if save_candidate:
        if not cat_args.convert:
            raise ValueError("--save_candidate_artifact requires --convert")
        if not spec or not str(spec).strip():
            raise ValueError("--save_candidate_artifact requires --candidate_artifact_spec")
        if not out_dir or not str(out_dir).strip():
            raise ValueError("--save_candidate_artifact requires --candidate_artifact_output_dir")
        return
    if spec is not None or out_dir is not None:
        raise ValueError(
            "--candidate_artifact_spec/--candidate_artifact_output_dir require --save_candidate_artifact"
        )


def process_cat_train_args(argv: Optional[Sequence[str]]):
    if argv is None:
        import sys

        argv = sys.argv[1:]
    script_parser = build_cat_train_parser()
    raw_script_args, remaining = script_parser.parse_known_args(list(argv))
    cat_args = _normalize_cat_train_script_args(raw_script_args)
    _validate_candidate_artifact_args(cat_args)
    _validate_outlier_protect_mode_args(cat_args)
    _validate_outlier_mlp_args(cat_args)
    _validate_distill_after_category_args(cat_args)

    vae_parser = _build_cat_train_vae_parser()
    raw_vae_args, unknown_args = vae_parser.parse_known_args(remaining)
    vae_args = _normalize_cat_train_vae_args(raw_vae_args)
    _validate_dynamic_calib_dataset_args(cat_args, vae_args)

    hf_parser = transformers.HfArgumentParser((HFArguments, CatTrainHFTrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=unknown_args)
    _validate_distill_lr_scheduler_args(training_args)
    _validate_distill_teacher_model_offload_args(training_args)
    use_bf16 = bool(training_args.bf16)
    vae_args.vae_weight_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.vae_autocast_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.access_token = hf_args.access_token
    return cat_args, hf_args, training_args, vae_args

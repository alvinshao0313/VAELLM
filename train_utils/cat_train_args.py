import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import transformers

from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_CHOICES,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    SPARSE_RESIDUAL_INDEX_BITS_CHOICES,
    SPARSE_RESIDUAL_VALUE_BITS_CHOICES,
    get_default_block_shape_for_index_bits,
    validate_sparse_residual_block_shape,
)
from train_utils.cat_data_prep import normalize_intra_part_sort_mode
from train_utils.cat_arg_overrides import (
    OverrideSpec,
    OverrideTable,
    make_choice_parser,
    parse_bool_text,
    parse_float_text,
    parse_intra_parallel_text,
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
    intra_parallel: OverrideTable[Tuple[int, int]]
    intra_part_sort_mode: OverrideTable[str]
    batch_size: int
    gpu_resident_data: bool
    log_every: int
    eval_every: int
    eval_blocks: int
    # 排序代码，已关闭。旧字段保留如下：
    # sort_prep_workers: int
    outlier_protect_count: OverrideTable[int]
    outlier_protect_mode: str
    outlier_low_rank: OverrideTable[int]
    outlier_residual_top_p: OverrideTable[float]
    outlier_residual_score: str
    outlier_residual_min_abs: float
    outlier_residual_codec: str
    outlier_residual_index_bits: int
    outlier_residual_value_bits: int
    outlier_residual_block_shape: Tuple[int, int]
    outlier_protect_axis: str
    wa_mse_calib_dataset: str
    wa_mse_calib_nsamples: int
    wa_mse_calib_seqlen: int
    wa_mse_calib_seed: int
    wa_mse_calib_device: str
    wa_mse_calib_log_every: int
    eval_ppl: bool
    eval_tasks: str
    ppl_limit: int
    eval_hif4_act: bool
    distill_after_category: str
    distill_dataset: str
    lora_rank: OverrideTable[int]
    lora_alpha: OverrideTable[float]
    lora_dropout: OverrideTable[float]
    distill_steps: OverrideTable[int]
    distill_batch_size: OverrideTable[int]
    distill_nsamples: OverrideTable[int]
    distill_lr: OverrideTable[float]
    distill_weight_decay: OverrideTable[float]
    distill_log_every: OverrideTable[int]
    distill_temperature: OverrideTable[float]
    distill_loss_alpha: OverrideTable[float]
    distill_loss_type: OverrideTable[str]
    distill_hidden_loss_weight: OverrideTable[float]
    distill_pre_mlp_hidden_loss_weight: OverrideTable[float]
    distill_hidden_alignment_layer_weighting: str
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
    unload_vae_original_weights_on_final_save: bool
    output_dir: str
    allow_tail_group: bool


@dataclass
class CatTrainHFTrainingArguments:
    distill_model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length used by the after-category distill trainer."},
    )
    distill_gradient_accumulation_steps: int = field(default=1)
    distill_optim: str = field(default="paged_adamw_8bit")
    distill_max_grad_norm: float = field(default=0.3)
    distill_warmup_ratio: float = field(default=0.3)
    distill_group_by_length: bool = field(default=True)
    distill_lr_scheduler_type: str = field(default="linear")
    distill_gradient_checkpointing: bool = field(default=False)
    distill_gradient_checkpointing_kwargs: Optional[str] = field(default=None)
    distill_post_attn: bool = field(
        default=False,
        metadata={"help": "For *_top distillation losses, compute KL on gathered full-vocab probabilities instead of renormalizing within the top-k subset."},
    )
    distill_hif4_act: bool = field(
        default=False,
        metadata={"help": "Enable HiFloat4 activation pseudo-quantization for student linear inputs during the after-category distill stage."},
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
    intra_parallel: Tuple[int, int]
    intra_part_sort_mode: str
    codebook_bits: int
    codebook_dim: int
    outlier_protect_count: int
    outlier_low_rank: int
    outlier_residual_top_p: float
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    decoder_type: str


@dataclass(frozen=True)
class ResolvedDistillRuntimeConfig:
    rank: int
    alpha: float
    dropout: float
    steps: int
    batch_size: int
    nsamples: int
    lr: float
    weight_decay: float
    log_every: int
    temperature: float
    loss_alpha: float
    loss_type: str
    hidden_loss_weight: float
    pre_mlp_hidden_loss_weight: float
    hidden_alignment_layer_weighting: str
    use_dora: bool


_SKIP_LAYER_PATTERN = re.compile(r"^(\d+)\.([A-Za-z0-9_]+)$")
_CATEGORY_OVERRIDE_SELECTORS = ("default", "cat")
_AFTER_CATEGORY_OVERRIDE_SELECTORS = ("default", "after")
_CAT_RECON_LOSS_CHOICES = ("mse", "l1", "huber", "relative_l1", "top_k_mse", "cosine", "w_mse", "w2_mse", "wa_mse")
_CAT_NORM_TYPE_CHOICES = ("group", "batch", "layer", "no")
_CAT_DECODER_TYPE_CHOICES = ("linear", "symmetric", "asymmetric")
_OUTLIER_PROTECT_MODE_CHOICES = ("none", "channel", "residual_sparse", "per_vae_low_rank", "post_vae_low_rank")
_DISTILL_HIDDEN_ALIGNMENT_LAYER_WEIGHTING_CHOICES = ("uniform", "linear_depth")
_DISTILL_AFTER_CATEGORY_CHOICES = ("none", "remaining_lora", "compressed_lora", "decoder", "both")
_DISTILL_AFTER_CATEGORY_COMPRESSED_LORA_MODES = {"compressed_lora", "both"}
_OUTLIER_RESIDUAL_SCORE_CHOICES = (
    "abs",
    "input_act_weighted_abs",
    "original_weight_abs",
    "input_act_weighted_original_weight_abs",
)
_OUTLIER_RESIDUAL_SCORE_MODES_NEED_ACT = (
    "input_act_weighted_abs",
    "input_act_weighted_original_weight_abs",
)


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


def _parse_wa_mse_calib_dataset_text(raw: str, *, arg_name: str) -> str:
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
_INTRA_PARALLEL_SPEC = _make_override_spec(
    arg_name="--intra_parallel",
    parse_value=lambda raw: parse_intra_parallel_text(raw, arg_name="--intra_parallel"),
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1x1,cat:q_proj=4x1",
)
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
_OUTLIER_LOW_RANK_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_low_rank",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=16,cat:down_proj=32",
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
    example="default=mse,cat:q_proj=wa_mse",
    choices=_CAT_RECON_LOSS_CHOICES,
)
_NORM_TYPE_SPEC = _make_choice_override_spec(
    arg_name="--norm_type",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=group,cat:q_proj=layer",
    choices=_CAT_NORM_TYPE_CHOICES,
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
_DISTILL_NSAMPLES_SPEC = _make_positive_int_override_spec(
    arg_name="--distill_nsamples",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=128,after:q_proj=256",
)
_DISTILL_LR_SPEC = _make_override_spec(
    arg_name="--distill_lr",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--distill_lr"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1e-4,after:q_proj=5e-5",
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
    parser.add_argument("--diversity_gamma", type=float, default=1.0)
    parser.add_argument("--use_checkpoint", action="store_true")
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
        intra_parallel=_parse_cat_override(raw_args.intra_parallel, spec=_INTRA_PARALLEL_SPEC),
        intra_part_sort_mode=_parse_cat_override("default=none", spec=_INTRA_PART_SORT_MODE_SPEC),
        batch_size=int(raw_args.batch_size),
        gpu_resident_data=bool(raw_args.gpu_resident_data),
        log_every=int(raw_args.log_every),
        eval_every=int(raw_args.eval_every),
        eval_blocks=int(raw_args.eval_blocks),
        # 排序代码，已关闭。旧字段赋值保留如下：
        # sort_prep_workers=1,
        outlier_protect_count=_parse_cat_override(raw_args.outlier_protect_count, spec=_OUTLIER_PROTECT_COUNT_SPEC),
        outlier_protect_mode=str(raw_args.outlier_protect_mode).strip().lower(),
        outlier_low_rank=_parse_cat_override(raw_args.outlier_low_rank, spec=_OUTLIER_LOW_RANK_SPEC),
        outlier_residual_top_p=_parse_cat_override(raw_args.outlier_residual_top_p, spec=_OUTLIER_RESIDUAL_TOP_P_SPEC),
        outlier_residual_score=str(raw_args.outlier_residual_score).strip().lower(),
        outlier_residual_min_abs=float(raw_args.outlier_residual_min_abs),
        outlier_residual_codec=str(raw_args.outlier_residual_codec).strip().lower(),
        outlier_residual_index_bits=resolved_index_bits,
        outlier_residual_value_bits=int(raw_args.outlier_residual_value_bits),
        outlier_residual_block_shape=tuple(int(v) for v in resolved_block_shape),
        outlier_protect_axis=str(raw_args.outlier_protect_axis).strip().lower(),
        wa_mse_calib_dataset=_parse_wa_mse_calib_dataset_text(
            raw_args.wa_mse_calib_dataset,
            arg_name="--wa_mse_calib_dataset",
        ),
        wa_mse_calib_nsamples=int(raw_args.wa_mse_calib_nsamples),
        wa_mse_calib_seqlen=int(raw_args.wa_mse_calib_seqlen),
        wa_mse_calib_seed=int(raw_args.wa_mse_calib_seed),
        wa_mse_calib_device=str(raw_args.wa_mse_calib_device),
        wa_mse_calib_log_every=int(raw_args.wa_mse_calib_log_every),
        eval_ppl=bool(raw_args.eval_ppl),
        eval_tasks=str(raw_args.eval_tasks),
        ppl_limit=int(raw_args.ppl_limit),
        eval_hif4_act=bool(raw_args.eval_hif4_act),
        distill_after_category=_normalize_distill_after_category(raw_args.distill_after_category),
        distill_dataset=_normalize_distill_dataset_arg(
            raw_args.distill_dataset,
            distill_after_category=str(raw_args.distill_after_category),
        ),
        lora_rank=_parse_cat_override(raw_args.lora_rank, spec=_LORA_RANK_SPEC),
        lora_alpha=_parse_cat_override(raw_args.lora_alpha, spec=_LORA_ALPHA_SPEC),
        lora_dropout=_parse_cat_override(raw_args.lora_dropout, spec=_LORA_DROPOUT_SPEC),
        distill_steps=_parse_cat_override(raw_args.distill_steps, spec=_DISTILL_STEPS_SPEC),
        distill_batch_size=_parse_cat_override(raw_args.distill_batch_size, spec=_DISTILL_BATCH_SIZE_SPEC),
        distill_nsamples=_parse_cat_override(raw_args.distill_nsamples, spec=_DISTILL_NSAMPLES_SPEC),
        distill_lr=_parse_cat_override(raw_args.distill_lr, spec=_DISTILL_LR_SPEC),
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
        distill_hidden_alignment_layer_weighting=make_choice_parser(
            arg_name="--distill_hidden_alignment_layer_weighting",
            choices=_DISTILL_HIDDEN_ALIGNMENT_LAYER_WEIGHTING_CHOICES,
        )(raw_args.distill_hidden_alignment_layer_weighting),
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
        unload_vae_original_weights_on_final_save=bool(raw_args.unload_vae_original_weights_on_final_save),
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


def _validate_dynamic_calib_dataset_args(cat_args: NormalizedCatArgs, vae_args) -> None:
    dynamic_calib_enabled = (
        _override_table_contains(
            vae_args.recon_loss_type,
            lambda value: str(value).strip().lower() == "wa_mse",
        )
        or _override_table_contains(
            cat_args.outlier_protect_count,
            lambda value: int(value) > 0,
        )
        or (
            str(cat_args.outlier_protect_mode).strip().lower() == "residual_sparse"
            and str(cat_args.outlier_residual_score).strip().lower() in _OUTLIER_RESIDUAL_SCORE_MODES_NEED_ACT
        )
    )
    if dynamic_calib_enabled and not str(cat_args.wa_mse_calib_dataset).strip():
        raise ValueError(
            "--wa_mse_calib_dataset must be set when dynamic activation calibration is enabled. "
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
    if enabled and mode != "remaining_lora":
        raise ValueError(
            f"{', '.join(enabled)} is only supported with --distill_after_category=remaining_lora."
        )
    if mode in _DISTILL_AFTER_CATEGORY_COMPRESSED_LORA_MODES:
        for _selector, use_dora in _iter_override_entries(cat_args.lora_use_dora):
            if bool(use_dora):
                raise ValueError(
                    f"--distill_after_category={mode} does not support --lora_use_dora=true."
                )


def _validate_outlier_protect_mode_args(cat_args: NormalizedCatArgs) -> None:
    mode = str(cat_args.outlier_protect_mode).strip().lower()
    codec = str(cat_args.outlier_residual_codec).strip().lower()
    index_bits = int(cat_args.outlier_residual_index_bits)
    value_bits = int(cat_args.outlier_residual_value_bits)
    block_rows, block_cols = tuple(int(v) for v in cat_args.outlier_residual_block_shape)
    top_p_table = cat_args.outlier_residual_top_p
    protect_table = cat_args.outlier_protect_count
    low_rank_table = cat_args.outlier_low_rank
    if mode not in _OUTLIER_PROTECT_MODE_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_protect_mode={cat_args.outlier_protect_mode!r}. "
            f"Expected one of: {', '.join(_OUTLIER_PROTECT_MODE_CHOICES)}."
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
    if str(cat_args.outlier_residual_score).strip().lower() not in _OUTLIER_RESIDUAL_SCORE_CHOICES:
        raise ValueError(
            f"Unsupported --outlier_residual_score={cat_args.outlier_residual_score!r}. "
            f"Expected one of: {', '.join(_OUTLIER_RESIDUAL_SCORE_CHOICES)}."
        )
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
    if mode in {"none", "channel"}:
        nonzero_top_p_entries = [
            f"{selector}={float(value)}"
            for selector, value in _iter_override_entries(top_p_table)
            if float(value) != 0.0
        ]
        if mode == "channel" and nonzero_top_p_entries:
            raise ValueError(
                "--outlier_residual_top_p must be 0 for every selector when "
                "--outlier_protect_mode=channel. Got: " + ",".join(nonzero_top_p_entries)
            )
        return
    if mode in {"per_vae_low_rank", "post_vae_low_rank"}:
        nonzero_protect_entries = []
        if bool(getattr(protect_table, "has_default", False)) and int(getattr(protect_table, "default", 0)) != 0:
            nonzero_protect_entries.append(f"default={int(getattr(protect_table, 'default', 0))}")
        nonzero_protect_entries.extend(
            f"cat:{category}={int(value)}"
            for category, value in sorted(getattr(protect_table, "by_category", {}).items())
            if int(value) != 0
        )
        if nonzero_protect_entries:
            raise ValueError(
                "--outlier_protect_count must be 0 for every selector when "
                f"--outlier_protect_mode={mode}. Got: " + ",".join(nonzero_protect_entries)
            )
        nonzero_top_p_entries = [
            f"{selector}={float(value)}"
            for selector, value in _iter_override_entries(top_p_table)
            if float(value) != 0.0
        ]
        if nonzero_top_p_entries:
            raise ValueError(
                "--outlier_residual_top_p must be 0 for every selector when "
                f"--outlier_protect_mode={mode}. Got: " + ",".join(nonzero_top_p_entries)
            )
        invalid_rank_entries = [
            f"{selector}={int(value)}"
            for selector, value in _iter_override_entries(low_rank_table)
            if int(value) <= 0
        ]
        if invalid_rank_entries:
            raise ValueError(
                "--outlier_low_rank must be > 0 for every selector when "
                f"--outlier_protect_mode={mode}. Got: " + ",".join(invalid_rank_entries)
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
        (cat_args.intra_parallel, "--intra_parallel"),
        (cat_args.intra_part_sort_mode, "--intra_part_sort_mode"),
        (cat_args.outlier_protect_count, "--outlier_protect_count"),
        (cat_args.outlier_low_rank, "--outlier_low_rank"),
        (cat_args.outlier_residual_top_p, "--outlier_residual_top_p"),
        (vae_args.codebook_bits, "--codebook_bits"),
        (vae_args.codebook_dim, "--codebook_dim"),
        (vae_args.residual_stages, "--residual_stages"),
        (vae_args.base_ch, "--base_ch"),
        (vae_args.num_res_blocks, "--num_res_blocks"),
        (vae_args.decoder_base_ch, "--decoder_base_ch"),
        (vae_args.decoder_num_res_blocks, "--decoder_num_res_blocks"),
        (vae_args.recon_loss_type, "--recon_loss_type"),
        (vae_args.norm_type, "--norm_type"),
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
        if resolved_outlier_mode == "channel" and resolved_outlier_residual_top_p != 0.0:
            raise ValueError(
                f"--outlier_residual_top_p resolved to {resolved_outlier_residual_top_p} for category "
                f"'{category}', but --outlier_protect_mode=channel requires 0."
            )
        if resolved_outlier_mode == "residual_sparse" and not (0.0 < resolved_outlier_residual_top_p <= 1.0):
            raise ValueError(
                f"--outlier_residual_top_p resolved to {resolved_outlier_residual_top_p} for category "
                f"'{category}', but --outlier_protect_mode=residual_sparse requires 0 < p <= 1."
            )
        resolved_outlier_low_rank = int(resolve_category_value(cat_args.outlier_low_rank, category))
        if resolved_outlier_mode in {"per_vae_low_rank", "post_vae_low_rank"}:
            if resolved_outlier_low_rank <= 0:
                raise ValueError(
                    f"--outlier_low_rank resolved to {resolved_outlier_low_rank} for category "
                    f"'{category}', but --outlier_protect_mode={resolved_outlier_mode} requires rank > 0."
                )
            if resolved_outlier_residual_top_p != 0.0:
                raise ValueError(
                    f"--outlier_residual_top_p resolved to {resolved_outlier_residual_top_p} for category "
                    f"'{category}', but --outlier_protect_mode={resolved_outlier_mode} requires 0."
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
            intra_parallel=tuple(resolve_category_value(cat_args.intra_parallel, category)),
            intra_part_sort_mode=normalize_intra_part_sort_mode(
                resolve_category_value(cat_args.intra_part_sort_mode, category),
                arg_name="--intra_part_sort_mode",
            ),
            codebook_bits=int(resolve_category_value(vae_args.codebook_bits, category)),
            codebook_dim=int(resolve_category_value(vae_args.codebook_dim, category)),
            outlier_protect_count=int(resolve_category_value(cat_args.outlier_protect_count, category)),
            outlier_low_rank=resolved_outlier_low_rank,
            outlier_residual_top_p=resolved_outlier_residual_top_p,
            recon_loss_type=str(resolve_category_value(vae_args.recon_loss_type, category)).strip().lower(),
            base_ch=int(resolve_category_value(vae_args.base_ch, category)),
            num_res_blocks=int(resolve_category_value(vae_args.num_res_blocks, category)),
            decoder_base_ch=resolve_category_value(vae_args.decoder_base_ch, category),
            decoder_num_res_blocks=resolve_category_value(vae_args.decoder_num_res_blocks, category),
            norm_type=str(resolve_category_value(vae_args.norm_type, category)).strip().lower(),
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
        nsamples=int(resolve_after_category_value(cat_args.distill_nsamples, after_category)),
        lr=float(resolve_after_category_value(cat_args.distill_lr, after_category)),
        weight_decay=float(resolve_after_category_value(cat_args.distill_weight_decay, after_category)),
        log_every=int(resolve_after_category_value(cat_args.distill_log_every, after_category)),
        temperature=float(resolve_after_category_value(cat_args.distill_temperature, after_category)),
        loss_alpha=float(resolve_after_category_value(cat_args.distill_loss_alpha, after_category)),
        loss_type=str(resolve_after_category_value(cat_args.distill_loss_type, after_category)),
        hidden_loss_weight=float(resolve_after_category_value(cat_args.distill_hidden_loss_weight, after_category)),
        pre_mlp_hidden_loss_weight=float(
            resolve_after_category_value(cat_args.distill_pre_mlp_hidden_loss_weight, after_category)
        ),
        hidden_alignment_layer_weighting=str(cat_args.distill_hidden_alignment_layer_weighting),
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
    parser.add_argument("--intra_parallel", type=str, default="default=1x1", help=f"类别覆盖参数。示例：{_INTRA_PARALLEL_SPEC.example}")
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
    parser.add_argument("--outlier_low_rank", type=str, default="default=0", help=f"类别覆盖参数。示例：{_OUTLIER_LOW_RANK_SPEC.example}")
    parser.add_argument(
        "--outlier_protect_mode",
        type=str,
        choices=list(_OUTLIER_PROTECT_MODE_CHOICES),
        default="channel",
        help="离群值保护模式：none 为关闭，channel 为压缩前保护通道，residual_sparse 为训练后保存残差补丁，per_vae_low_rank/post_vae_low_rank 为低秩补丁。",
    )
    parser.add_argument(
        "--outlier_residual_top_p",
        type=str,
        default="default=0.0",
        help=f"类别覆盖参数。仅 residual_sparse 模式生效。示例：{_OUTLIER_RESIDUAL_TOP_P_SPEC.example}",
    )
    parser.add_argument(
        "--outlier_residual_score",
        type=str,
        choices=list(_OUTLIER_RESIDUAL_SCORE_CHOICES),
        default="abs",
        help=(
            "仅 residual_sparse 模式生效。选点打分方式："
            "abs / input_act_weighted_abs / original_weight_abs / "
            "input_act_weighted_original_weight_abs。"
        ),
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
        "--wa_mse_calib_dataset",
        type=lambda raw: _parse_wa_mse_calib_dataset_text(raw, arg_name="--wa_mse_calib_dataset"),
        default="",
        help="Calibration dataset used for wa_mse dynamic act-max recomputation. "
        "Required when dynamic calibration is enabled. Format: alias=weight,alias=weight. "
        "For example: wiki=1.0, openorca=1.0, or openorca=0.5,fineweb_edu=0.5.",
    )
    parser.add_argument("--wa_mse_calib_nsamples", type=int, default=512, help="Calibration sample count used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_seqlen", type=int, default=512, help="Calibration sequence length used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_seed", type=int, default=0, help="Calibration sampling seed used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_device", type=str, default="", help="Device for wa_mse dynamic act-max recomputation. Empty means use --train_device.")
    parser.add_argument("--wa_mse_calib_log_every", type=int, default=0, help="Log interval for wa_mse dynamic act-max recomputation progress (0 to disable).")
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
    parser.add_argument("--distill_nsamples", type=str, default="default=128", help=f"after_category 覆盖参数。示例：{_DISTILL_NSAMPLES_SPEC.example}")
    parser.add_argument("--distill_lr", type=str, default="default=1e-4", help=f"after_category 覆盖参数。示例：{_DISTILL_LR_SPEC.example}")
    parser.add_argument("--distill_weight_decay", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_DISTILL_WEIGHT_DECAY_SPEC.example}")
    parser.add_argument("--distill_log_every", type=str, default="default=1", help=f"after_category 覆盖参数。示例：{_DISTILL_LOG_EVERY_SPEC.example}")
    parser.add_argument("--distill_temperature", type=str, default="default=1.0", help=f"after_category 覆盖参数。示例：{_DISTILL_TEMPERATURE_SPEC.example}")
    parser.add_argument("--distill_loss_alpha", type=str, default="default=0.5", help=f"after_category 覆盖参数。示例：{_DISTILL_LOSS_ALPHA_SPEC.example}")
    parser.add_argument("--distill_loss_type", type=str, default="default=sft", help=f"after_category 覆盖参数。示例：{_DISTILL_LOSS_TYPE_SPEC.example}")
    parser.add_argument("--distill_hidden_loss_weight", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_DISTILL_HIDDEN_LOSS_WEIGHT_SPEC.example}")
    parser.add_argument(
        "--distill_pre_mlp_hidden_loss_weight",
        type=str,
        default="default=0.0",
        help=f"after_category 覆盖参数。示例：{_DISTILL_PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC.example}",
    )
    parser.add_argument(
        "--distill_hidden_alignment_layer_weighting",
        type=str,
        default="uniform",
        help="LoRA hidden alignment 辅助损失的层权重模式：uniform 或 linear_depth。",
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
    parser.add_argument("--rot_llm", action="store_true", default=False, help="在 VAE 压缩前先对基座 LLM 执行一次离线旋转融合。")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从已有 cat_train checkpoint 继续训练。可传 run 目录、final_model 目录，或 checkpoint_meta.json。",
    )
    parser.add_argument("--convert", action="store_true", help="每个类别训练完成后，将 Linear 替换为压缩后的线性层。")
    parser.add_argument("--convert_device", type=str, default="cuda")
    parser.add_argument("--save_model", action="store_true", help="保存最终模型 state_dict/config/tokenizer（需要 --convert）。")
    parser.add_argument("--unload_vae_original_weights_on_final_save", action="store_true", default=False, help="最终保存前卸载 VAELinear 中缓存的原始 Linear 权重，减小保存体积。")
    parser.add_argument("--output_dir", type=str, default="./output_linear_by_category")
    parser.add_argument("--allow_tail_group", type=lambda v: _parse_bool_like(v, arg_name="--allow_tail_group"), default=True, help="是否允许处理最后一个不足分组大小的尾部分组（true/false）。")
    return parser


def process_cat_train_args(argv: Optional[Sequence[str]]):
    if argv is None:
        import sys

        argv = sys.argv[1:]
    script_parser = build_cat_train_parser()
    raw_script_args, remaining = script_parser.parse_known_args(list(argv))
    cat_args = _normalize_cat_train_script_args(raw_script_args)
    _validate_outlier_protect_mode_args(cat_args)
    _validate_distill_after_category_args(cat_args)

    vae_parser = _build_cat_train_vae_parser()
    raw_vae_args, unknown_args = vae_parser.parse_known_args(remaining)
    vae_args = _normalize_cat_train_vae_args(raw_vae_args)
    _validate_dynamic_calib_dataset_args(cat_args, vae_args)

    hf_parser = transformers.HfArgumentParser((HFArguments, CatTrainHFTrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=unknown_args)
    use_bf16 = bool(training_args.bf16)
    vae_args.vae_weight_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.vae_autocast_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.access_token = hf_args.access_token
    return cat_args, hf_args, training_args, vae_args

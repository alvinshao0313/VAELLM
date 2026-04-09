import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import transformers

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
    _parse_lora_loss_type,
)
from train_utils.utils import split_csv


@dataclass
class NormalizedCatArgs:
    category_order: str
    transpose_modules: str
    projection_suffixes: str
    include_all_linears: bool
    steps_per_category: OverrideTable[int]
    skip_layers: str
    linear_group_size: int
    intra_parallel: OverrideTable[Tuple[int, int]]
    intra_part_sort_mode: OverrideTable[Union[str, Tuple[str, str]]]
    batch_size: int
    log_every: int
    eval_every: int
    eval_blocks: int
    outlier_protect_count: OverrideTable[int]
    outlier_protect_axis: str
    wa_mse_calib_dataset: str
    wa_mse_calib_nsamples: int
    wa_mse_calib_seqlen: int
    wa_mse_calib_seed: int
    wa_mse_calib_device: str
    wa_mse_calib_log_every: int
    ppl_limit: int
    lora_after_category: bool
    lora_dataset: str
    lora_rank: OverrideTable[int]
    lora_alpha: OverrideTable[float]
    lora_dropout: OverrideTable[float]
    lora_steps: OverrideTable[int]
    lora_batch_size: OverrideTable[int]
    lora_nsamples: OverrideTable[int]
    lora_lr: OverrideTable[float]
    lora_weight_decay: OverrideTable[float]
    lora_log_every: OverrideTable[int]
    lora_temperature: OverrideTable[float]
    lora_loss_alpha: OverrideTable[float]
    lora_loss_type: OverrideTable[str]
    lora_use_dora: OverrideTable[bool]
    seed: int
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
    lora_model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length used by the LoRA trainer."},
    )
    lora_gradient_accumulation_steps: int = field(default=1)
    lora_optim: str = field(default="paged_adamw_8bit")
    lora_max_grad_norm: float = field(default=0.3)
    lora_warmup_ratio: float = field(default=0.3)
    lora_group_by_length: bool = field(default=True)
    lora_lr_scheduler_type: str = field(default="linear")
    lora_post_attn: bool = field(
        default=False,
        metadata={"help": "For *_top LoRA distillation losses, compute KL on gathered full-vocab probabilities instead of renormalizing within the top-k subset."},
    )
    lora_hif4_act: bool = field(
        default=False,
        metadata={"help": "Enable HiFloat4 activation pseudo-quantization for student linear inputs during the LoRA stage."},
    )
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


@dataclass(frozen=True)
class ResolvedCategoryRuntimeConfig:
    category: str
    residual_stages: int
    steps: int
    intra_parallel: Tuple[int, int]
    intra_part_sort_mode: Tuple[str, str]
    codebook_bits: int
    codebook_dim: int
    outlier_protect_count: int
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    decoder_type: str


@dataclass(frozen=True)
class ResolvedLoraRuntimeConfig:
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
    use_dora: bool


_SKIP_LAYER_PATTERN = re.compile(r"^(\d+)\.([A-Za-z0-9_]+)$")
_CATEGORY_OVERRIDE_SELECTORS = ("default", "cat")
_AFTER_CATEGORY_OVERRIDE_SELECTORS = ("default", "after")
_CAT_RECON_LOSS_CHOICES = ("mse", "l1", "huber", "relative_l1", "top_k_mse", "cosine", "w_mse", "w2_mse", "wa_mse")
_CAT_NORM_TYPE_CHOICES = ("group", "batch", "layer", "no")
_CAT_DECODER_TYPE_CHOICES = ("linear", "symmetric", "asymmetric")
_LORA_DATASET_ALIASES = {
    "wiki": "wiki",
    "wikitext2": "wiki",
    "fineweb_edu": "fineweb_edu",
    "openorca": "openorca",
    "redpajama": "redpajama",
    "alpaca": "alpaca",
}

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


def _parse_lora_loss_alpha_text(raw: str, *, arg_name: str) -> float:
    value = parse_float_text(raw, arg_name=arg_name, min_value=0.0, inclusive_min=True)
    if value > 1.0:
        raise argparse.ArgumentTypeError(f"{arg_name} must be <= 1.0, got {value}.")
    return float(value)


def _parse_lora_dataset_text(raw: str, *, arg_name: str) -> str:
    value = str(raw).strip().lower()
    if value not in _LORA_DATASET_ALIASES:
        raise argparse.ArgumentTypeError(
            f"{arg_name} must be one of: {', '.join(_LORA_DATASET_ALIASES.keys())}. Got {raw!r}."
        )
    return str(_LORA_DATASET_ALIASES[value])


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
_INTRA_PARALLEL_SPEC = _make_override_spec(
    arg_name="--intra_parallel",
    parse_value=lambda raw: parse_intra_parallel_text(raw, arg_name="--intra_parallel"),
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1x1,cat:q_proj=4x1",
)
_INTRA_PART_SORT_MODE_SPEC = _make_override_spec(
    arg_name="--intra_part_sort_mode",
    parse_value=lambda raw: parse_intra_part_sort_mode_text(raw, arg_name="--intra_part_sort_mode"),
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=l2,cat:q_proj=row:l2|col:none",
)
_OUTLIER_PROTECT_COUNT_SPEC = _make_positive_int_override_spec(
    arg_name="--outlier_protect_count",
    allowed_selectors=_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0,cat:down_proj=64",
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
_LORA_STEPS_SPEC = _make_positive_int_override_spec(
    arg_name="--lora_steps",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=50,after:q_proj=200",
    min_value=0,
)
_LORA_BATCH_SIZE_SPEC = _make_positive_int_override_spec(
    arg_name="--lora_batch_size",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=2,after:q_proj=4",
)
_LORA_NSAMPLES_SPEC = _make_positive_int_override_spec(
    arg_name="--lora_nsamples",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=128,after:q_proj=256",
)
_LORA_LR_SPEC = _make_override_spec(
    arg_name="--lora_lr",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_lr"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1e-4,after:q_proj=5e-5",
)
_LORA_WEIGHT_DECAY_SPEC = _make_override_spec(
    arg_name="--lora_weight_decay",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_weight_decay"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.0,after:q_proj=0.01",
)
_LORA_LOG_EVERY_SPEC = _make_positive_int_override_spec(
    arg_name="--lora_log_every",
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1,after:q_proj=10",
)
_LORA_TEMPERATURE_SPEC = _make_override_spec(
    arg_name="--lora_temperature",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_temperature", min_value=0.0, inclusive_min=False),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=1.0,after:q_proj=2.0",
)
_LORA_LOSS_ALPHA_SPEC = _make_override_spec(
    arg_name="--lora_loss_alpha",
    parse_value=lambda raw: _parse_lora_loss_alpha_text(raw, arg_name="--lora_loss_alpha"),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=0.5,after:q_proj=0.3",
)
_LORA_LOSS_TYPE_SPEC = _make_override_spec(
    arg_name="--lora_loss_type",
    parse_value=lambda raw: _parse_lora_loss_type(str(raw)),
    allowed_selectors=_AFTER_CATEGORY_OVERRIDE_SELECTORS,
    example="default=sft,after:q_proj=dual_kl_top_1000",
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
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "linear", "cosine"], help="Learning rate scheduler")
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
    return NormalizedCatArgs(
        category_order=str(raw_args.category_order),
        transpose_modules=str(raw_args.transpose_modules),
        projection_suffixes=str(raw_args.projection_suffixes),
        include_all_linears=bool(raw_args.include_all_linears),
        steps_per_category=_parse_cat_override(raw_args.steps_per_category, spec=_STEPS_PER_CATEGORY_SPEC),
        skip_layers=str(raw_args.skip_layers),
        linear_group_size=int(raw_args.linear_group_size),
        intra_parallel=_parse_cat_override(raw_args.intra_parallel, spec=_INTRA_PARALLEL_SPEC),
        intra_part_sort_mode=_parse_cat_override(raw_args.intra_part_sort_mode, spec=_INTRA_PART_SORT_MODE_SPEC),
        batch_size=int(raw_args.batch_size),
        log_every=int(raw_args.log_every),
        eval_every=int(raw_args.eval_every),
        eval_blocks=int(raw_args.eval_blocks),
        outlier_protect_count=_parse_cat_override(raw_args.outlier_protect_count, spec=_OUTLIER_PROTECT_COUNT_SPEC),
        outlier_protect_axis=str(raw_args.outlier_protect_axis).strip().lower(),
        wa_mse_calib_dataset=str(raw_args.wa_mse_calib_dataset),
        wa_mse_calib_nsamples=int(raw_args.wa_mse_calib_nsamples),
        wa_mse_calib_seqlen=int(raw_args.wa_mse_calib_seqlen),
        wa_mse_calib_seed=int(raw_args.wa_mse_calib_seed),
        wa_mse_calib_device=str(raw_args.wa_mse_calib_device),
        wa_mse_calib_log_every=int(raw_args.wa_mse_calib_log_every),
        ppl_limit=int(raw_args.ppl_limit),
        lora_after_category=bool(raw_args.lora_after_category),
        lora_dataset=str(raw_args.lora_dataset),
        lora_rank=_parse_cat_override(raw_args.lora_rank, spec=_LORA_RANK_SPEC),
        lora_alpha=_parse_cat_override(raw_args.lora_alpha, spec=_LORA_ALPHA_SPEC),
        lora_dropout=_parse_cat_override(raw_args.lora_dropout, spec=_LORA_DROPOUT_SPEC),
        lora_steps=_parse_cat_override(raw_args.lora_steps, spec=_LORA_STEPS_SPEC),
        lora_batch_size=_parse_cat_override(raw_args.lora_batch_size, spec=_LORA_BATCH_SIZE_SPEC),
        lora_nsamples=_parse_cat_override(raw_args.lora_nsamples, spec=_LORA_NSAMPLES_SPEC),
        lora_lr=_parse_cat_override(raw_args.lora_lr, spec=_LORA_LR_SPEC),
        lora_weight_decay=_parse_cat_override(raw_args.lora_weight_decay, spec=_LORA_WEIGHT_DECAY_SPEC),
        lora_log_every=_parse_cat_override(raw_args.lora_log_every, spec=_LORA_LOG_EVERY_SPEC),
        lora_temperature=_parse_cat_override(raw_args.lora_temperature, spec=_LORA_TEMPERATURE_SPEC),
        lora_loss_alpha=_parse_cat_override(raw_args.lora_loss_alpha, spec=_LORA_LOSS_ALPHA_SPEC),
        lora_loss_type=_parse_cat_override(raw_args.lora_loss_type, spec=_LORA_LOSS_TYPE_SPEC),
        lora_use_dora=_parse_cat_override(raw_args.lora_use_dora, spec=_LORA_USE_DORA_SPEC),
        seed=int(raw_args.seed),
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
    tables = (
        (cat_args.steps_per_category, "--steps_per_category"),
        (cat_args.intra_parallel, "--intra_parallel"),
        (cat_args.intra_part_sort_mode, "--intra_part_sort_mode"),
        (cat_args.outlier_protect_count, "--outlier_protect_count"),
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
        resolved[category] = ResolvedCategoryRuntimeConfig(
            category=str(category),
            residual_stages=int(resolve_category_value(vae_args.residual_stages, category)),
            steps=int(steps_per_category),
            intra_parallel=tuple(resolve_category_value(cat_args.intra_parallel, category)),
            intra_part_sort_mode=normalize_intra_part_sort_mode(
                resolve_category_value(cat_args.intra_part_sort_mode, category),
                arg_name="--intra_part_sort_mode",
            ),
            codebook_bits=int(resolve_category_value(vae_args.codebook_bits, category)),
            codebook_dim=int(resolve_category_value(vae_args.codebook_dim, category)),
            outlier_protect_count=int(resolve_category_value(cat_args.outlier_protect_count, category)),
            recon_loss_type=str(resolve_category_value(vae_args.recon_loss_type, category)).strip().lower(),
            base_ch=int(resolve_category_value(vae_args.base_ch, category)),
            num_res_blocks=int(resolve_category_value(vae_args.num_res_blocks, category)),
            decoder_base_ch=resolve_category_value(vae_args.decoder_base_ch, category),
            decoder_num_res_blocks=resolve_category_value(vae_args.decoder_num_res_blocks, category),
            norm_type=str(resolve_category_value(vae_args.norm_type, category)).strip().lower(),
            decoder_type=str(resolve_category_value(vae_args.decoder_type, category)).strip().lower(),
        )
    return resolved


def resolve_lora_runtime_config(cat_args: NormalizedCatArgs, after_category: Optional[str]) -> ResolvedLoraRuntimeConfig:
    return ResolvedLoraRuntimeConfig(
        rank=int(resolve_after_category_value(cat_args.lora_rank, after_category)),
        alpha=float(resolve_after_category_value(cat_args.lora_alpha, after_category)),
        dropout=float(resolve_after_category_value(cat_args.lora_dropout, after_category)),
        steps=int(resolve_after_category_value(cat_args.lora_steps, after_category)),
        batch_size=int(resolve_after_category_value(cat_args.lora_batch_size, after_category)),
        nsamples=int(resolve_after_category_value(cat_args.lora_nsamples, after_category)),
        lr=float(resolve_after_category_value(cat_args.lora_lr, after_category)),
        weight_decay=float(resolve_after_category_value(cat_args.lora_weight_decay, after_category)),
        log_every=int(resolve_after_category_value(cat_args.lora_log_every, after_category)),
        temperature=float(resolve_after_category_value(cat_args.lora_temperature, after_category)),
        loss_alpha=float(resolve_after_category_value(cat_args.lora_loss_alpha, after_category)),
        loss_type=str(resolve_after_category_value(cat_args.lora_loss_type, after_category)),
        use_dora=bool(resolve_after_category_value(cat_args.lora_use_dora, after_category)),
    )


def build_cat_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--category_order", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--transpose_modules", type=str, default="v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--projection_suffixes",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="默认 projection-only 收集模式下，允许参与训练的投影层后缀列表。",
    )
    parser.add_argument(
        "--include_all_linears",
        action="store_true",
        default=False,
        help="关闭默认的 projection-only 过滤，改为包含模型中全部 nn.Linear。",
    )
    parser.add_argument("--steps_per_category", type=str, default="default=2000", help=f"类别覆盖参数。示例：{_STEPS_PER_CATEGORY_SPEC.example}")
    parser.add_argument("--skip_layers", type=str, default="", help="指定在 LLM 前向中始终使用原始线性权重的层，格式: layer_idx.category，例如 0.down_proj,30.q_proj。")
    parser.add_argument("--linear_group_size", type=int, default=32, help="跨层分组大小：每组同时训练多少个同类 Linear。")
    parser.add_argument("--intra_parallel", type=str, default="default=1x1", help=f"类别覆盖参数。示例：{_INTRA_PARALLEL_SPEC.example}")
    parser.add_argument("--intra_part_sort_mode", type=str, default="default=l2", help=f"类别覆盖参数。示例：{_INTRA_PART_SORT_MODE_SPEC.example}")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_blocks", type=int, default=256)
    parser.add_argument("--outlier_protect_count", type=str, default="default=0", help=f"类别覆盖参数。示例：{_OUTLIER_PROTECT_COUNT_SPEC.example}")
    parser.add_argument("--outlier_protect_axis", type=str, choices=["input", "output"], default="input", help="Choose whether outlier protection preserves input channels or output channels.")
    parser.add_argument("--wa_mse_calib_dataset", type=str, default="wikitext2", help="Calibration dataset used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_nsamples", type=int, default=512, help="Calibration sample count used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_seqlen", type=int, default=512, help="Calibration sequence length used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_seed", type=int, default=0, help="Calibration sampling seed used for wa_mse dynamic act-max recomputation.")
    parser.add_argument("--wa_mse_calib_device", type=str, default="", help="Device for wa_mse dynamic act-max recomputation. Empty means use --train_device.")
    parser.add_argument("--wa_mse_calib_log_every", type=int, default=0, help="Log interval for wa_mse dynamic act-max recomputation progress (0 to disable).")
    parser.add_argument("--ppl_limit", type=int, default=-1, help="每类训练后 PPL 评估样本上限，-1 为全量。")
    parser.add_argument("--lora_after_category", action="store_true", help="每个类别 VAE 训练后，对剩余类别做一次 LoRA 微调并融合。")
    parser.add_argument(
        "--lora_dataset",
        type=lambda raw: _parse_lora_dataset_text(raw, arg_name="--lora_dataset"),
        default="wiki",
        help="LoRA 补偿训练数据集。支持: wiki, fineweb_edu, openorca, redpajama, alpaca。",
    )
    parser.add_argument("--lora_rank", type=str, default="default=8", help=f"after_category 覆盖参数。示例：{_LORA_RANK_SPEC.example}")
    parser.add_argument("--lora_alpha", type=str, default="default=16.0", help=f"after_category 覆盖参数。示例：{_LORA_ALPHA_SPEC.example}")
    parser.add_argument("--lora_dropout", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_LORA_DROPOUT_SPEC.example}")
    parser.add_argument("--lora_steps", type=str, default="default=50", help=f"after_category 覆盖参数。示例：{_LORA_STEPS_SPEC.example}")
    parser.add_argument("--lora_batch_size", type=str, default="default=2", help=f"after_category 覆盖参数。示例：{_LORA_BATCH_SIZE_SPEC.example}")
    parser.add_argument("--lora_nsamples", type=str, default="default=128", help=f"after_category 覆盖参数。示例：{_LORA_NSAMPLES_SPEC.example}")
    parser.add_argument("--lora_lr", type=str, default="default=1e-4", help=f"after_category 覆盖参数。示例：{_LORA_LR_SPEC.example}")
    parser.add_argument("--lora_weight_decay", type=str, default="default=0.0", help=f"after_category 覆盖参数。示例：{_LORA_WEIGHT_DECAY_SPEC.example}")
    parser.add_argument("--lora_log_every", type=str, default="default=1", help=f"after_category 覆盖参数。示例：{_LORA_LOG_EVERY_SPEC.example}")
    parser.add_argument("--lora_temperature", type=str, default="default=1.0", help=f"after_category 覆盖参数。示例：{_LORA_TEMPERATURE_SPEC.example}")
    parser.add_argument("--lora_loss_alpha", type=str, default="default=0.5", help=f"after_category 覆盖参数。示例：{_LORA_LOSS_ALPHA_SPEC.example}")
    parser.add_argument("--lora_loss_type", type=str, default="default=sft", help=f"after_category 覆盖参数。示例：{_LORA_LOSS_TYPE_SPEC.example}")
    parser.add_argument("--lora_use_dora", type=str, default="default=true", help=f"after_category 覆盖参数。示例：{_LORA_USE_DORA_SPEC.example}")
    parser.add_argument("--seed", type=int, default=0)
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

    vae_parser = _build_cat_train_vae_parser()
    raw_vae_args, unknown_args = vae_parser.parse_known_args(remaining)
    vae_args = _normalize_cat_train_vae_args(raw_vae_args)

    hf_parser = transformers.HfArgumentParser((HFArguments, CatTrainHFTrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=unknown_args)
    use_bf16 = bool(training_args.bf16)
    vae_args.vae_weight_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.vae_autocast_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.access_token = hf_args.access_token
    return cat_args, hf_args, training_args, vae_args

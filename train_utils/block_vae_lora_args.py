import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from train_utils.cat_arg_overrides import (
    OverrideSpec,
    OverrideTable,
    parse_bool_text,
    parse_float_text,
    parse_int_text,
    parse_intra_parallel_text,
    parse_optional_int_text,
    parse_override_table,
    resolve_category_value,
    validate_category_keys,
)
from train_utils.cat_train_args import ResolvedCategoryRuntimeConfig, parse_skip_layers


_CATEGORY_SELECTORS = ("default", "cat")
_RECON_LOSS_TYPE_CHOICES = ("mse", "l1", "huber", "relative_l1", "top_k_mse", "cosine", "w_mse", "w2_mse")
_NORM_TYPE_CHOICES = ("group", "batch", "layer", "no")
_DECODER_TYPE_CHOICES = ("linear", "symmetric", "asymmetric")
_OPTIMIZER_CHOICES = ("adam", "adamw", "sgd", "rmsprop")
_LR_SCHEDULER_CHOICES = ("none", "linear", "cosine", "constant", "constant_with_warmup")
_QUANTIZER_CHOICES = ("LFQ", "BSQ")
_BLOCK_LORA_VARIANT_CHOICES = ("plain", "rslora", "dora", "adalora")
_BLOCK_LORA_BIAS_CHOICES = ("none", "lora_only")
_BLOCK_DISTILL_TRAIN_MODE_CHOICES = ("lora", "decoder", "both")
_BLOCK_VAE_PIPELINE_MODE_CHOICES = ("inline", "pretrain", "distill", "pretrain_distill")
_QWEN3_BLOCK_LINEAR_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_DEFAULT_BLOCK_VAE_CATEGORIES = ",".join(_QWEN3_BLOCK_LINEAR_CATEGORIES)
_DEFAULT_TRANSPOSE_MODULES = "q_proj,v_proj,o_proj,down_proj"


@dataclass(frozen=True)
class BlockHFArgs:
    access_token: Optional[str] = None


@dataclass(frozen=True)
class BlockTrainingArgs:
    fp16: bool = False
    bf16: bool = False


@dataclass(frozen=True)
class BlockVaeLoraArgs:
    model_path: str
    output_dir: str
    seed: int
    deterministic: bool
    train_device: str
    convert_device: str
    unload_vae_original_weights_on_final_save: bool
    block_vae_pipeline_mode: str
    vae_pretrained_checkpoint: Optional[str]
    block_vae_pretrain_devices: str
    block_vae_pretrain_workers: Optional[int]
    block_vae_linear_group_size: int
    block_vae_allow_tail_group: bool
    block_vae_categories: Tuple[str, ...]
    vae_steps: OverrideTable[int]
    vae_batch_size: str
    vae_gpu_resident_data: bool
    vae_log_every: int
    vae_eval_every: int
    intra_parallel: OverrideTable[Tuple[int, int]]
    codebook_bits: OverrideTable[int]
    codebook_dim: OverrideTable[int]
    residual_stages: OverrideTable[int]
    base_ch: OverrideTable[int]
    num_res_blocks: OverrideTable[int]
    decoder_base_ch: OverrideTable[Optional[int]]
    decoder_num_res_blocks: OverrideTable[Optional[int]]
    norm_type: OverrideTable[str]
    decoder_type: OverrideTable[str]
    recon_loss_type: OverrideTable[str]
    quantizer_type: str
    gamma0: float
    gamma: float
    zeta: float
    inv_temperature: float
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    optimizer: str
    lr_scheduler: str
    lr_warmup_steps: int
    l1_weight: float
    lfq_weight: float
    commitment_loss_weight: float
    entropy_loss_weight: float
    diversity_gamma: float
    normalize_weight: bool
    use_checkpoint: bool
    new_quant: bool
    block_distill_dataset: str
    block_distill_steps: int
    block_distill_nsamples: int
    block_distill_seqlen: int
    block_distill_train_mode: str
    block_lora_rank: int
    block_lora_lr: float
    block_lora_lr_scheduler: str
    block_lora_warmup_steps: int
    block_lora_variant: str
    block_lora_alpha: float
    block_lora_dropout: float
    block_lora_bias: str
    block_lora_hif4_act: bool
    block_adalora_init_rank: int
    block_adalora_tinit: int
    block_adalora_tfinal: int
    block_adalora_delta_t: int
    block_adalora_beta1: float
    block_adalora_beta2: float
    block_adalora_orth_reg_weight: float
    block_loss_alpha: float
    block_loss_beta: float
    block_attn_query_chunk_size: int
    block_distill_log_every: int
    block_decode_group_size: int
    transpose_modules: str
    skip_layers: str
    block_layers: str
    block_resume_from_checkpoint: Optional[str]
    block_keep_last_checkpoints: int
    block_eval_after_each_layer: bool
    block_eval_tasks: str
    block_eval_ppl: bool
    block_eval_ppl_limit: int
    block_eval_device: Optional[str]
    block_eval_hif4_act: bool


@dataclass(frozen=True)
class BlockVaeRuntimeConfig:
    category: str
    residual_stages: int
    steps: int
    intra_parallel: Tuple[int, int]
    intra_part_sort_mode: str
    codebook_bits: int
    codebook_dim: int
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    decoder_type: str
    joint_decoder_steps: int = 0
    joint_decoder_lr: float = 0.0
    joint_decoder_group_size: int = 1
    joint_decoder_batch_size: Optional[int] = None
    outlier_protect_count: int = 0
    outlier_low_rank: int = 0
    outlier_residual_top_p: float = 0.0


def _make_override_spec(arg_name, parse_value, example) -> OverrideSpec:
    return OverrideSpec(
        arg_name=arg_name,
        parse_value=parse_value,
        allowed_selectors=_CATEGORY_SELECTORS,
        example=example,
    )


def _make_int_override_spec(arg_name: str, *, min_value: int, example: str) -> OverrideSpec:
    return _make_override_spec(
        arg_name,
        lambda raw: parse_int_text(raw, arg_name=arg_name, min_value=min_value),
        example,
    )


def _make_optional_int_override_spec(arg_name: str, *, min_value: int, example: str) -> OverrideSpec:
    return _make_override_spec(
        arg_name,
        lambda raw: parse_optional_int_text(raw, arg_name=arg_name, min_value=min_value),
        example,
    )


def _make_choice_override_spec(arg_name: str, *, choices: Sequence[str], example: str) -> OverrideSpec:
    choice_set = {str(choice).lower() for choice in choices}

    def parse_choice(raw: str) -> str:
        value = str(raw).strip().lower()
        if value not in choice_set:
            raise argparse.ArgumentTypeError(
                f"Invalid {arg_name} value {raw!r}. Expected one of: {','.join(sorted(choice_set))}."
            )
        return value

    return _make_override_spec(arg_name, parse_choice, example)


_VAE_STEPS_SPEC = _make_int_override_spec("--vae_steps", min_value=1, example="default=20000,cat:k_proj=10000")
_INTRA_PARALLEL_SPEC = _make_override_spec(
    "--intra_parallel",
    lambda raw: parse_intra_parallel_text(raw, arg_name="--intra_parallel"),
    "default=1x1,cat:q_proj=4x1",
)
_CODEBOOK_BITS_SPEC = _make_int_override_spec("--codebook_bits", min_value=1, example="default=32,cat:k_proj=24")
_CODEBOOK_DIM_SPEC = _make_int_override_spec("--codebook_dim", min_value=1, example="default=32,cat:down_proj=16")
_RESIDUAL_STAGES_SPEC = _make_int_override_spec("--residual_stages", min_value=1, example="default=2,cat:q_proj=1")
_BASE_CH_SPEC = _make_int_override_spec("--base_ch", min_value=1, example="default=128,cat:q_proj=192")
_NUM_RES_BLOCKS_SPEC = _make_int_override_spec("--num_res_blocks", min_value=0, example="default=1,cat:down_proj=2")
_DECODER_BASE_CH_SPEC = _make_optional_int_override_spec(
    "--decoder_base_ch",
    min_value=1,
    example="default=128,cat:q_proj=none",
)
_DECODER_NUM_RES_BLOCKS_SPEC = _make_optional_int_override_spec(
    "--decoder_num_res_blocks",
    min_value=0,
    example="default=1,cat:q_proj=0",
)
_NORM_TYPE_SPEC = _make_choice_override_spec("--norm_type", choices=_NORM_TYPE_CHOICES, example="default=layer")
_DECODER_TYPE_SPEC = _make_choice_override_spec("--decoder_type", choices=_DECODER_TYPE_CHOICES, example="default=symmetric")
_RECON_LOSS_TYPE_SPEC = _make_choice_override_spec("--recon_loss_type", choices=_RECON_LOSS_TYPE_CHOICES, example="default=mse")


def _parse_positive_float(raw: str, *, arg_name: str) -> float:
    return parse_float_text(raw, arg_name=arg_name, min_value=0.0, inclusive_min=False)


def _parse_nonnegative_float(raw: str, *, arg_name: str) -> float:
    return parse_float_text(raw, arg_name=arg_name, min_value=0.0, inclusive_min=True)


def _parse_choice(raw: str, *, arg_name: str, choices: Sequence[str]) -> str:
    value = str(raw).strip()
    lookup = {str(choice).lower(): str(choice) for choice in choices}
    key = value.lower()
    if key not in lookup:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name}={raw!r}. Expected one of: {','.join(choices)}.")
    return lookup[key]


def parse_vae_batch_size(raw: str) -> str:
    text = str(raw).strip().lower()
    if text == "all":
        return "all"
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--vae_batch_size must be a positive integer or 'all'.") from exc
    if int(value) < 1:
        raise argparse.ArgumentTypeError("--vae_batch_size must be >= 1 or 'all'.")
    return str(int(value))


def parse_block_vae_categories(raw: str) -> Tuple[str, ...]:
    text = str(raw).strip()
    if not text:
        raise argparse.ArgumentTypeError("--block_vae_categories cannot be empty.")
    categories = []
    seen = set()
    for part in text.split(","):
        category = part.strip()
        if not category:
            raise argparse.ArgumentTypeError(f"Invalid --block_vae_categories={raw!r}: empty segment.")
        if category in seen:
            raise argparse.ArgumentTypeError(f"--block_vae_categories contains duplicate category {category!r}.")
        seen.add(category)
        categories.append(category)
    return tuple(categories)


def parse_block_layers(raw: str, *, num_layers: Optional[int] = None) -> Tuple[int, ...]:
    text = str(raw).strip().lower()
    if not text:
        raise ValueError("--block_layers cannot be empty.")
    if text == "all":
        if num_layers is None:
            return ()
        return tuple(range(int(num_layers)))
    layers = []
    seen = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"Invalid --block_layers={raw!r}: empty segment.")
        if "-" in item:
            pieces = item.split("-")
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise ValueError(f"Invalid --block_layers range segment: {item!r}.")
            if not pieces[0].isdigit() or not pieces[1].isdigit():
                raise ValueError(f"Invalid --block_layers range segment: {item!r}.")
            start, end = int(pieces[0]), int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid --block_layers range {item!r}: start must be <= end.")
            values = range(start, end + 1)
        else:
            if not item.isdigit():
                raise ValueError(f"Invalid --block_layers segment: {item!r}.")
            values = (int(item),)
        for layer_idx in values:
            if num_layers is not None and (layer_idx < 0 or layer_idx >= int(num_layers)):
                raise ValueError(
                    f"--block_layers contains layer {layer_idx}, but valid range is [0,{int(num_layers) - 1}]."
                )
            if layer_idx in seen:
                raise ValueError(f"--block_layers contains duplicate layer {layer_idx}.")
            seen.add(layer_idx)
            layers.append(layer_idx)
    return tuple(layers)


def parse_transpose_modules(raw: str) -> Tuple[str, ...]:
    text = str(raw).strip()
    if not text:
        return tuple()
    allowed = set(_QWEN3_BLOCK_LINEAR_CATEGORIES)
    modules = []
    seen = set()
    for item in text.split(","):
        name = item.strip()
        if not name:
            raise ValueError(f"Invalid --transpose_modules={raw!r}: empty segment.")
        if name not in allowed:
            raise ValueError(
                f"Invalid --transpose_modules entry {name!r}. "
                f"Allowed values: {','.join(_QWEN3_BLOCK_LINEAR_CATEGORIES)}."
            )
        if name in seen:
            raise ValueError(f"--transpose_modules contains duplicate module {name!r}.")
        seen.add(name)
        modules.append(name)
    return tuple(modules)


def format_skip_layers(skip_layer_keys: Sequence[Tuple[int, str]]) -> List[str]:
    return [f"{int(layer_idx)}.{category}" for layer_idx, category in sorted(skip_layer_keys)]


def validate_skip_layers_with_block_layers(
    *,
    skip_layer_keys: Sequence[Tuple[int, str]],
    selected_layers: Sequence[int],
) -> None:
    selected = {int(layer_idx) for layer_idx in selected_layers}
    outside_selected = sorted(
        (int(layer_idx), str(category))
        for layer_idx, category in skip_layer_keys
        if int(layer_idx) not in selected
    )
    if outside_selected:
        raise ValueError(
            "--skip_layers contains entries outside --block_layers: "
            + ",".join(format_skip_layers(outside_selected))
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Block-wise Qwen3 VAELinear compression with per-block LoRA distillation.",
        allow_abbrev=False,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=lambda raw: parse_bool_text(raw, arg_name="--deterministic"), default=False)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--convert_device", type=str, default="cuda")
    parser.add_argument(
        "--unload_vae_original_weights_on_final_save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--block_vae_pipeline_mode",
        type=lambda raw: _parse_choice(raw, arg_name="--block_vae_pipeline_mode", choices=_BLOCK_VAE_PIPELINE_MODE_CHOICES),
        default="inline",
    )
    parser.add_argument("--vae_pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--block_vae_pretrain_devices", type=str, default="")
    parser.add_argument(
        "--block_vae_pretrain_workers",
        type=lambda raw: None if str(raw).strip() == "" else int(raw),
        default=None,
    )
    parser.add_argument("--block_vae_linear_group_size", type=int, default=32)
    parser.add_argument(
        "--block_vae_allow_tail_group",
        type=lambda raw: parse_bool_text(raw, arg_name="--block_vae_allow_tail_group"),
        default=True,
    )
    parser.add_argument(
        "--block_vae_categories",
        type=parse_block_vae_categories,
        default=_QWEN3_BLOCK_LINEAR_CATEGORIES,
        help=f"Comma-separated block Linear categories to VAE-train/distill, in pretrain order. Default: {_DEFAULT_BLOCK_VAE_CATEGORIES}.",
    )
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--bf16", type=lambda raw: parse_bool_text(raw, arg_name="--bf16"), default=True)
    parser.add_argument("--fp16", type=lambda raw: parse_bool_text(raw, arg_name="--fp16"), default=False)

    parser.add_argument("--vae_steps", type=str, default="default=20000", help=f"Category override. Example: {_VAE_STEPS_SPEC.example}")
    parser.add_argument("--vae_batch_size", type=parse_vae_batch_size, default="8192")
    parser.add_argument(
        "--vae_gpu_resident_data",
        type=lambda raw: parse_bool_text(raw, arg_name="--vae_gpu_resident_data"),
        default=False,
        help="Whether to keep each VAE residual stage dataset on GPU during training.",
    )
    parser.add_argument("--vae_log_every", type=int, default=100)
    parser.add_argument("--vae_eval_every", type=int, default=0)
    parser.add_argument("--intra_parallel", type=str, default="default=1x1", help=f"Category override. Example: {_INTRA_PARALLEL_SPEC.example}")
    parser.add_argument("--codebook_bits", type=str, default="default=32", help=f"Category override. Example: {_CODEBOOK_BITS_SPEC.example}")
    parser.add_argument("--codebook_dim", type=str, default="default=32", help=f"Category override. Example: {_CODEBOOK_DIM_SPEC.example}")
    parser.add_argument("--residual_stages", type=str, default="default=2", help=f"Category override. Example: {_RESIDUAL_STAGES_SPEC.example}")
    parser.add_argument("--base_ch", type=str, default="default=128", help=f"Category override. Example: {_BASE_CH_SPEC.example}")
    parser.add_argument("--num_res_blocks", type=str, default="default=1", help=f"Category override. Example: {_NUM_RES_BLOCKS_SPEC.example}")
    parser.add_argument("--decoder_base_ch", type=str, default="default=128", help=f"Category override. Example: {_DECODER_BASE_CH_SPEC.example}")
    parser.add_argument("--decoder_num_res_blocks", type=str, default="default=1", help=f"Category override. Example: {_DECODER_NUM_RES_BLOCKS_SPEC.example}")
    parser.add_argument("--norm_type", type=str, default="default=layer", help=f"Category override. Example: {_NORM_TYPE_SPEC.example}")
    parser.add_argument("--decoder_type", type=str, default="default=symmetric", help=f"Category override. Example: {_DECODER_TYPE_SPEC.example}")
    parser.add_argument("--recon_loss_type", type=str, default="default=mse", help=f"Category override. Example: {_RECON_LOSS_TYPE_SPEC.example}")

    parser.add_argument("--quantizer_type", type=lambda raw: _parse_choice(raw, arg_name="--quantizer_type", choices=_QUANTIZER_CHOICES), default="BSQ")
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=200.0)
    parser.add_argument("--lr", type=lambda raw: _parse_positive_float(raw, arg_name="--lr"), default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=lambda raw: _parse_nonnegative_float(raw, arg_name="--weight_decay"), default=0.0)
    parser.add_argument("--optimizer", type=lambda raw: _parse_choice(raw, arg_name="--optimizer", choices=_OPTIMIZER_CHOICES), default="adamw")
    parser.add_argument("--lr_scheduler", type=lambda raw: _parse_choice(raw, arg_name="--lr_scheduler", choices=_LR_SCHEDULER_CHOICES), default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lfq_weight", type=float, default=5.0)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.1)
    parser.add_argument("--entropy_loss_weight", type=float, default=1e-4)
    parser.add_argument("--diversity_gamma", type=float, default=1.0)
    parser.add_argument("--normalize_weight", action="store_true", default=False)
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--new_quant", action="store_true", default=False)

    parser.add_argument("--block_distill_dataset", type=str, default="fineweb_edu=0.35,race=0.30,sciq=0.20,openorca=0.15")
    parser.add_argument("--block_distill_steps", type=int, default=100)
    parser.add_argument("--block_distill_nsamples", type=int, default=100)
    parser.add_argument("--block_distill_seqlen", type=int, default=4096)
    parser.add_argument(
        "--block_distill_train_mode",
        type=lambda raw: _parse_choice(raw, arg_name="--block_distill_train_mode", choices=_BLOCK_DISTILL_TRAIN_MODE_CHOICES),
        default="lora",
    )
    parser.add_argument("--block_lora_rank", type=int, default=32)
    parser.add_argument("--block_lora_lr", type=lambda raw: _parse_positive_float(raw, arg_name="--block_lora_lr"), default=1e-4)
    parser.add_argument(
        "--block_lora_lr_scheduler",
        type=lambda raw: _parse_choice(raw, arg_name="--block_lora_lr_scheduler", choices=_LR_SCHEDULER_CHOICES),
        default="none",
    )
    parser.add_argument("--block_lora_warmup_steps", type=int, default=0)
    parser.add_argument(
        "--block_lora_variant",
        type=lambda raw: _parse_choice(raw, arg_name="--block_lora_variant", choices=_BLOCK_LORA_VARIANT_CHOICES),
        default="plain",
    )
    parser.add_argument("--block_lora_alpha", type=lambda raw: _parse_positive_float(raw, arg_name="--block_lora_alpha"), default=None)
    parser.add_argument("--block_lora_dropout", type=lambda raw: _parse_nonnegative_float(raw, arg_name="--block_lora_dropout"), default=0.0)
    parser.add_argument(
        "--block_lora_bias",
        type=lambda raw: _parse_choice(raw, arg_name="--block_lora_bias", choices=_BLOCK_LORA_BIAS_CHOICES),
        default="none",
    )
    parser.add_argument(
        "--block_lora_hif4_act",
        type=lambda raw: parse_bool_text(raw, arg_name="--block_lora_hif4_act"),
        default=False,
        help="Enable HiFloat4 activation quantization on the student path during block LoRA distillation.",
    )
    parser.add_argument("--block_adalora_init_rank", type=int, default=None)
    parser.add_argument("--block_adalora_tinit", type=int, default=0)
    parser.add_argument("--block_adalora_tfinal", type=int, default=0)
    parser.add_argument("--block_adalora_delta_t", type=int, default=1)
    parser.add_argument("--block_adalora_beta1", type=lambda raw: _parse_positive_float(raw, arg_name="--block_adalora_beta1"), default=0.85)
    parser.add_argument("--block_adalora_beta2", type=lambda raw: _parse_positive_float(raw, arg_name="--block_adalora_beta2"), default=0.85)
    parser.add_argument("--block_adalora_orth_reg_weight", type=lambda raw: _parse_nonnegative_float(raw, arg_name="--block_adalora_orth_reg_weight"), default=0.5)
    parser.add_argument("--block_loss_alpha", type=float, default=0.1, help="Attention map KL loss weight.")
    parser.add_argument("--block_loss_beta", type=float, default=0.2, help="Linear output relative MSE loss weight.")
    parser.add_argument("--block_attn_query_chunk_size", type=int, default=128)
    parser.add_argument("--block_distill_log_every", type=int, default=10)
    parser.add_argument("--block_decode_group_size", type=int, default=8)
    parser.add_argument(
        "--transpose_modules",
        type=str,
        default=_DEFAULT_TRANSPOSE_MODULES,
        help="Comma-separated Qwen3 block Linear names to transpose before VAE splitting. Empty string disables transpose.",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default="",
        help="Keep selected Qwen3 block Linear modules uncompressed. Format: layer_idx.category, for example 0.down_proj,30.q_proj.",
    )
    parser.add_argument("--block_layers", type=str, default="all", help="Block layer selector: all or ranges like 0-3,8,12-15.")
    parser.add_argument(
        "--block_resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume block-wise compression from a block layer checkpoint directory or checkpoint_meta.json.",
    )
    parser.add_argument(
        "--block_keep_last_checkpoints",
        type=int,
        default=3,
        help="Keep at most N latest block layer checkpoints. Set 0 to disable intermediate block checkpoints.",
    )
    parser.add_argument(
        "--block_eval_after_each_layer",
        type=lambda raw: parse_bool_text(raw, arg_name="--block_eval_after_each_layer"),
        default=False,
    )
    parser.add_argument("--block_eval_tasks", type=str, default="")
    parser.add_argument(
        "--block_eval_ppl",
        type=lambda raw: parse_bool_text(raw, arg_name="--block_eval_ppl"),
        default=False,
    )
    parser.add_argument("--block_eval_ppl_limit", type=int, default=-1)
    parser.add_argument("--block_eval_device", type=str, default=None)
    parser.add_argument(
        "--block_eval_hif4_act",
        type=lambda raw: parse_bool_text(raw, arg_name="--block_eval_hif4_act"),
        default=False,
    )
    return parser


def _normalize_args(raw_args) -> BlockVaeLoraArgs:
    return BlockVaeLoraArgs(
        model_path=str(raw_args.model_path),
        output_dir=str(raw_args.output_dir),
        seed=int(raw_args.seed),
        deterministic=bool(raw_args.deterministic),
        train_device=str(raw_args.train_device),
        convert_device=str(raw_args.convert_device),
        unload_vae_original_weights_on_final_save=bool(raw_args.unload_vae_original_weights_on_final_save),
        block_vae_pipeline_mode=str(raw_args.block_vae_pipeline_mode).lower(),
        vae_pretrained_checkpoint=None
        if raw_args.vae_pretrained_checkpoint is None or str(raw_args.vae_pretrained_checkpoint).strip() == ""
        else str(raw_args.vae_pretrained_checkpoint),
        block_vae_pretrain_devices=str(raw_args.block_vae_pretrain_devices),
        block_vae_pretrain_workers=None
        if raw_args.block_vae_pretrain_workers is None
        else int(raw_args.block_vae_pretrain_workers),
        block_vae_linear_group_size=int(raw_args.block_vae_linear_group_size),
        block_vae_allow_tail_group=bool(raw_args.block_vae_allow_tail_group),
        block_vae_categories=tuple(str(category) for category in raw_args.block_vae_categories),
        vae_steps=parse_override_table(str(raw_args.vae_steps), spec=_VAE_STEPS_SPEC),
        vae_batch_size=str(raw_args.vae_batch_size),
        vae_gpu_resident_data=bool(raw_args.vae_gpu_resident_data),
        vae_log_every=int(raw_args.vae_log_every),
        vae_eval_every=int(raw_args.vae_eval_every),
        intra_parallel=parse_override_table(str(raw_args.intra_parallel), spec=_INTRA_PARALLEL_SPEC),
        codebook_bits=parse_override_table(str(raw_args.codebook_bits), spec=_CODEBOOK_BITS_SPEC),
        codebook_dim=parse_override_table(str(raw_args.codebook_dim), spec=_CODEBOOK_DIM_SPEC),
        residual_stages=parse_override_table(str(raw_args.residual_stages), spec=_RESIDUAL_STAGES_SPEC),
        base_ch=parse_override_table(str(raw_args.base_ch), spec=_BASE_CH_SPEC),
        num_res_blocks=parse_override_table(str(raw_args.num_res_blocks), spec=_NUM_RES_BLOCKS_SPEC),
        decoder_base_ch=parse_override_table(str(raw_args.decoder_base_ch), spec=_DECODER_BASE_CH_SPEC),
        decoder_num_res_blocks=parse_override_table(str(raw_args.decoder_num_res_blocks), spec=_DECODER_NUM_RES_BLOCKS_SPEC),
        norm_type=parse_override_table(str(raw_args.norm_type), spec=_NORM_TYPE_SPEC),
        decoder_type=parse_override_table(str(raw_args.decoder_type), spec=_DECODER_TYPE_SPEC),
        recon_loss_type=parse_override_table(str(raw_args.recon_loss_type), spec=_RECON_LOSS_TYPE_SPEC),
        quantizer_type=str(raw_args.quantizer_type),
        gamma0=float(raw_args.gamma0),
        gamma=float(raw_args.gamma),
        zeta=float(raw_args.zeta),
        inv_temperature=float(raw_args.inv_temperature),
        lr=float(raw_args.lr),
        beta1=float(raw_args.beta1),
        beta2=float(raw_args.beta2),
        weight_decay=float(raw_args.weight_decay),
        optimizer=str(raw_args.optimizer).lower(),
        lr_scheduler=str(raw_args.lr_scheduler).lower(),
        lr_warmup_steps=int(raw_args.lr_warmup_steps),
        l1_weight=float(raw_args.l1_weight),
        lfq_weight=float(raw_args.lfq_weight),
        commitment_loss_weight=float(raw_args.commitment_loss_weight),
        entropy_loss_weight=float(raw_args.entropy_loss_weight),
        diversity_gamma=float(raw_args.diversity_gamma),
        normalize_weight=bool(raw_args.normalize_weight),
        use_checkpoint=bool(raw_args.use_checkpoint),
        new_quant=bool(raw_args.new_quant),
        block_distill_dataset=str(raw_args.block_distill_dataset),
        block_distill_steps=int(raw_args.block_distill_steps),
        block_distill_nsamples=int(raw_args.block_distill_nsamples),
        block_distill_seqlen=int(raw_args.block_distill_seqlen),
        block_distill_train_mode=str(raw_args.block_distill_train_mode).lower(),
        block_lora_rank=int(raw_args.block_lora_rank),
        block_lora_lr=float(raw_args.block_lora_lr),
        block_lora_lr_scheduler=str(raw_args.block_lora_lr_scheduler).lower(),
        block_lora_warmup_steps=int(raw_args.block_lora_warmup_steps),
        block_lora_variant=str(raw_args.block_lora_variant).lower(),
        block_lora_alpha=float(raw_args.block_lora_rank if raw_args.block_lora_alpha is None else raw_args.block_lora_alpha),
        block_lora_dropout=float(raw_args.block_lora_dropout),
        block_lora_bias=str(raw_args.block_lora_bias).lower(),
        block_lora_hif4_act=bool(raw_args.block_lora_hif4_act),
        block_adalora_init_rank=int(raw_args.block_lora_rank if raw_args.block_adalora_init_rank is None else raw_args.block_adalora_init_rank),
        block_adalora_tinit=int(raw_args.block_adalora_tinit),
        block_adalora_tfinal=int(raw_args.block_adalora_tfinal),
        block_adalora_delta_t=int(raw_args.block_adalora_delta_t),
        block_adalora_beta1=float(raw_args.block_adalora_beta1),
        block_adalora_beta2=float(raw_args.block_adalora_beta2),
        block_adalora_orth_reg_weight=float(raw_args.block_adalora_orth_reg_weight),
        block_loss_alpha=float(raw_args.block_loss_alpha),
        block_loss_beta=float(raw_args.block_loss_beta),
        block_attn_query_chunk_size=int(raw_args.block_attn_query_chunk_size),
        block_distill_log_every=int(raw_args.block_distill_log_every),
        block_decode_group_size=int(raw_args.block_decode_group_size),
        transpose_modules=str(raw_args.transpose_modules),
        skip_layers=str(raw_args.skip_layers),
        block_layers=str(raw_args.block_layers),
        block_resume_from_checkpoint=None
        if raw_args.block_resume_from_checkpoint is None
        else str(raw_args.block_resume_from_checkpoint),
        block_keep_last_checkpoints=int(raw_args.block_keep_last_checkpoints),
        block_eval_after_each_layer=bool(raw_args.block_eval_after_each_layer),
        block_eval_tasks=str(raw_args.block_eval_tasks),
        block_eval_ppl=bool(raw_args.block_eval_ppl),
        block_eval_ppl_limit=int(raw_args.block_eval_ppl_limit),
        block_eval_device=None if raw_args.block_eval_device is None else str(raw_args.block_eval_device),
        block_eval_hif4_act=bool(raw_args.block_eval_hif4_act),
    )


def _validate_args(parser: argparse.ArgumentParser, args: BlockVaeLoraArgs, training_args: BlockTrainingArgs) -> None:
    if training_args.fp16 and training_args.bf16:
        parser.error("--fp16 and --bf16 cannot both be true.")
    try:
        parse_vae_batch_size(str(args.vae_batch_size))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if int(args.vae_log_every) < 1:
        parser.error("--vae_log_every must be >= 1.")
    if int(args.vae_eval_every) < 0:
        parser.error("--vae_eval_every must be >= 0.")
    if str(args.block_vae_pipeline_mode) not in _BLOCK_VAE_PIPELINE_MODE_CHOICES:
        parser.error(f"--block_vae_pipeline_mode must be one of: {','.join(_BLOCK_VAE_PIPELINE_MODE_CHOICES)}.")
    if str(args.block_vae_pipeline_mode) == "distill":
        if args.vae_pretrained_checkpoint is None:
            parser.error("--vae_pretrained_checkpoint is required when --block_vae_pipeline_mode is distill.")
    elif args.vae_pretrained_checkpoint is not None:
        parser.error("--vae_pretrained_checkpoint is only supported when --block_vae_pipeline_mode is distill.")
    if args.block_vae_pretrain_workers is not None and int(args.block_vae_pretrain_workers) < 1:
        parser.error("--block_vae_pretrain_workers must be >= 1.")
    if int(args.block_vae_linear_group_size) < 1:
        parser.error("--block_vae_linear_group_size must be >= 1.")
    allowed_block_categories = set(_QWEN3_BLOCK_LINEAR_CATEGORIES)
    invalid_block_categories = [category for category in args.block_vae_categories if category not in allowed_block_categories]
    if invalid_block_categories:
        parser.error(
            "--block_vae_categories contains invalid category values for the current Qwen3 block path: "
            f"{','.join(invalid_block_categories)}. "
            f"Allowed values: {','.join(_QWEN3_BLOCK_LINEAR_CATEGORIES)}."
        )
    if int(args.lr_warmup_steps) < 0:
        parser.error("--lr_warmup_steps must be >= 0.")
    if "=" not in str(args.block_distill_dataset):
        parser.error("--block_distill_dataset must use ratio syntax, for example fineweb_edu=1.0.")
    if int(args.block_distill_steps) <= 0:
        parser.error("--block_distill_steps must be > 0.")
    if int(args.block_distill_nsamples) <= 0:
        parser.error("--block_distill_nsamples must be > 0.")
    if int(args.block_distill_seqlen) <= 0:
        parser.error("--block_distill_seqlen must be > 0.")
    if str(args.block_distill_train_mode) not in _BLOCK_DISTILL_TRAIN_MODE_CHOICES:
        parser.error(f"--block_distill_train_mode must be one of: {','.join(_BLOCK_DISTILL_TRAIN_MODE_CHOICES)}.")
    if int(args.block_lora_rank) <= 0:
        parser.error("--block_lora_rank must be > 0.")
    if str(args.block_lora_lr_scheduler) not in _LR_SCHEDULER_CHOICES:
        parser.error(f"--block_lora_lr_scheduler must be one of: {','.join(_LR_SCHEDULER_CHOICES)}.")
    if int(args.block_lora_warmup_steps) < 0:
        parser.error("--block_lora_warmup_steps must be >= 0.")
    if str(args.block_lora_variant) not in _BLOCK_LORA_VARIANT_CHOICES:
        parser.error(f"--block_lora_variant must be one of: {','.join(_BLOCK_LORA_VARIANT_CHOICES)}.")
    if float(args.block_lora_alpha) <= 0.0:
        parser.error("--block_lora_alpha must be > 0.")
    if float(args.block_lora_dropout) < 0.0 or float(args.block_lora_dropout) > 1.0:
        parser.error("--block_lora_dropout must be between 0 and 1.")
    if str(args.block_lora_bias) not in _BLOCK_LORA_BIAS_CHOICES:
        parser.error(f"--block_lora_bias must be one of: {','.join(_BLOCK_LORA_BIAS_CHOICES)}.")
    if int(args.block_adalora_init_rank) <= 0:
        parser.error("--block_adalora_init_rank must be > 0.")
    if int(args.block_adalora_tinit) < 0:
        parser.error("--block_adalora_tinit must be >= 0.")
    if int(args.block_adalora_tfinal) < 0:
        parser.error("--block_adalora_tfinal must be >= 0.")
    if int(args.block_adalora_delta_t) <= 0:
        parser.error("--block_adalora_delta_t must be > 0.")
    if str(args.block_lora_variant) == "adalora" and int(args.block_adalora_init_rank) < int(args.block_lora_rank):
        parser.error("--block_adalora_init_rank must be >= --block_lora_rank for AdaLoRA.")
    if float(args.block_loss_alpha) < 0.0:
        parser.error("--block_loss_alpha must be >= 0.")
    if float(args.block_loss_beta) < 0.0:
        parser.error("--block_loss_beta must be >= 0.")
    if float(args.block_loss_alpha) + float(args.block_loss_beta) > 1.0:
        parser.error("--block_loss_alpha + --block_loss_beta must be <= 1.")
    if int(args.block_attn_query_chunk_size) <= 0:
        parser.error("--block_attn_query_chunk_size must be > 0.")
    if int(args.block_distill_log_every) <= 0:
        parser.error("--block_distill_log_every must be > 0.")
    if int(args.block_decode_group_size) <= 0:
        parser.error("--block_decode_group_size must be > 0.")
    try:
        parse_transpose_modules(str(args.transpose_modules))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        skip_layers = parse_skip_layers(str(args.skip_layers))
    except ValueError as exc:
        parser.error(str(exc))
    selected_categories = set(args.block_vae_categories)
    invalid_skip_categories = sorted({category for _layer_idx, category in skip_layers if category not in selected_categories})
    if invalid_skip_categories:
        parser.error(
            "--skip_layers contains invalid category values: "
            f"{','.join(invalid_skip_categories)}. "
            f"Allowed values from --block_vae_categories: {','.join(args.block_vae_categories)}."
        )
    try:
        parse_block_layers(str(args.block_layers), num_layers=None)
    except ValueError as exc:
        parser.error(str(exc))
    if int(args.block_keep_last_checkpoints) < 0:
        parser.error("--block_keep_last_checkpoints must be >= 0.")
    if bool(args.block_eval_after_each_layer) and not bool(args.block_eval_ppl) and not str(args.block_eval_tasks).strip():
        parser.error("--block_eval_after_each_layer=true requires --block_eval_ppl=true or non-empty --block_eval_tasks.")
    if int(args.block_eval_ppl_limit) == 0 or int(args.block_eval_ppl_limit) < -1:
        parser.error("--block_eval_ppl_limit must be -1 or >= 1.")


def parse_block_vae_lora_args(argv: Optional[Sequence[str]] = None) -> Tuple[BlockVaeLoraArgs, BlockHFArgs, BlockTrainingArgs]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    raw = parser.parse_args(raw_argv)
    args = _normalize_args(raw)
    hf_args = BlockHFArgs(access_token=None if raw.access_token is None else str(raw.access_token))
    training_args = BlockTrainingArgs(fp16=bool(raw.fp16), bf16=bool(raw.bf16))
    _validate_args(parser, args, training_args)
    return args, hf_args, training_args


def resolve_block_runtime_configs(
    args: BlockVaeLoraArgs,
    active_categories: Sequence[str],
) -> Dict[str, ResolvedCategoryRuntimeConfig]:
    tables = (
        (args.vae_steps, "--vae_steps"),
        (args.intra_parallel, "--intra_parallel"),
        (args.codebook_bits, "--codebook_bits"),
        (args.codebook_dim, "--codebook_dim"),
        (args.residual_stages, "--residual_stages"),
        (args.base_ch, "--base_ch"),
        (args.num_res_blocks, "--num_res_blocks"),
        (args.decoder_base_ch, "--decoder_base_ch"),
        (args.decoder_num_res_blocks, "--decoder_num_res_blocks"),
        (args.norm_type, "--norm_type"),
        (args.decoder_type, "--decoder_type"),
        (args.recon_loss_type, "--recon_loss_type"),
    )
    for table, arg_name in tables:
        validate_category_keys(table, active_categories, arg_name)

    resolved: Dict[str, ResolvedCategoryRuntimeConfig] = {}
    for category in active_categories:
        resolved[category] = ResolvedCategoryRuntimeConfig(
            category=str(category),
            residual_stages=int(resolve_category_value(args.residual_stages, category)),
            steps=int(resolve_category_value(args.vae_steps, category)),
            joint_decoder_steps=0,
            joint_decoder_lr=0.0,
            joint_decoder_group_size=1,
            joint_decoder_batch_size=None,
            intra_parallel=tuple(resolve_category_value(args.intra_parallel, category)),
            intra_part_sort_mode="none",
            codebook_bits=int(resolve_category_value(args.codebook_bits, category)),
            codebook_dim=int(resolve_category_value(args.codebook_dim, category)),
            outlier_protect_count=0,
            outlier_low_rank=0,
            outlier_residual_top_p=0.0,
            recon_loss_type=str(resolve_category_value(args.recon_loss_type, category)),
            base_ch=int(resolve_category_value(args.base_ch, category)),
            num_res_blocks=int(resolve_category_value(args.num_res_blocks, category)),
            decoder_base_ch=resolve_category_value(args.decoder_base_ch, category),
            decoder_num_res_blocks=resolve_category_value(args.decoder_num_res_blocks, category),
            norm_type=str(resolve_category_value(args.norm_type, category)),
            decoder_type=str(resolve_category_value(args.decoder_type, category)),
        )
    return resolved

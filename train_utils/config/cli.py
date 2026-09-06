from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from train_utils.config.configs import (
    AFTER_CATEGORY_MODES,
    CHANNEL_AXES,
    CHANNEL_MLP_RANK_METRICS,
    CHANNEL_PROTECT_MODES,
    CHANNEL_RANK_METRICS,
    CHANNEL_SCOPES,
    RECON_LOSS_TYPES,
    TRAIN_MODES,
    VAE_ACTIVATION_TYPES,
    VAE_DECODER_TYPES,
    VAE_LR_SCHEDULERS,
    VAE_NORM_TYPES,
    VAE_OPTIMS,
    AfterCategoryResolvedConfig,
    AuxTrainableConfig,
    CandidateArtifactConfig,
    ChannelProtectionConfig,
    DistillDataConfig,
    DistillLossConfig,
    DistillOptimizationConfig,
    DistillRuntimeConfig,
    EvaluationRuntimeConfig,
    LoRAConfig,
    VAECompressionConfig,
    VAECoreConfig,
    VAEDecoderConfig,
    VAEOptimizationConfig,
    parse_after_category_mode,
    parse_dataset_mix_spec,
    parse_hidden_layer_weighting,
    parse_loss_type,
    parse_optional_positive_float,
    parse_train_mode,
    validate_train_mode_aux,
)
from train_utils.config.overrides import (
    OverrideSpec,
    OverrideTable,
    looks_like_override_string,
    make_choice_override_spec,
    make_optional_int_override_spec,
    make_override_spec,
    make_positive_int_override_spec,
    parse_bool_text,
    parse_float_text,
    parse_int_text,
    parse_intra_parallel_text,
    parse_override_table,
    resolve_after_category_value,
    resolve_category_value,
    validate_category_keys,
)
from train_utils.config.targets import (
    TargetLayers,
    TargetModules,
    parse_compression_categories,
    parse_skip_layers,
    parse_target_layers,
    parse_target_modules,
    validate_skip_layers_scope,
)


DELETED_CLI_FLAGS = frozenset(
    {
        "--distill_temperature",
        "--distill_alpha",
        "--distill_loss_alpha",
        "--distill_loss_type",
        "--distill_hidden_loss_weight",
        "--distill_pre_mlp_hidden_loss_weight",
        "--distill_hidden_alignment_layer_weighting",
        "--prompt_kd_weight",
        "--distill_prompt_kd_weight",
        "--distill_eakld_confidence_k",
        "--eakld_confidence_k",
        "--distill_selective_student_topk",
        "--distill_selective_student_topk_chunk_rows",
        "--distill_teacher_logits_cpu_staging",
        "--distill_teacher_model_offload",
        "--distill_dataset",
        "--distill_model_max_length",
        "--distill_dynamic_padding",
        "--compressed_lora_scope",
        "--lora_use_dora",
        "--decoder_layers",
        "--sparse_bit_tuning",
        "--vae_tune_bias",
        "--tune_final_norm",
        "--distill_tune_final_norm",
        "--use_post_norm_head_linear",
        "--distill_use_post_norm_head_linear",
        "--distill_steps",
        "--distill_batch_size",
        "--distill_lr",
        "--distill_decoder_lr",
        "--distill_weight_decay",
        "--distill_gradient_accumulation_steps",
        "--distill_gradient_checkpointing",
        "--distill_gradient_checkpointing_kwargs",
        "--distill_optim",
        "--distill_max_grad_norm",
        "--distill_warmup_ratio",
        "--distill_lr_scheduler_type",
        "--distill_group_by_length",
        "--distill_log_every",
        "--target_categories",
        "--distill_after_category",
        "--include_all_linears",
        "--steps_per_category",
        "--outlier_protect_mode",
        "--outlier_channel_scope",
        "--outlier_protect_count",
        "--outlier_protect_min_per_layer",
        "--outlier_rank_metric",
        "--outlier_mlp_rank_metric",
        "--outlier_mlp_fuse_weights",
        "--outlier_protect_axis",
        "--outlier_protect_channel_quant",
        "--outlier_residual_top_p",
        "--outlier_residual_codec",
        "--outlier_residual_index_bits",
        "--outlier_residual_value_bits",
        "--outlier_residual_block_shape",
        "--outlier_residual_vae_stages",
        "--outlier_residual_vae_decoder_share_scope",
        "--outlier_residual_vae_batch_multiplier",
        "--outlier_residual_vae_steps",
        "--outlier_residual_vae_lr",
        "--outlier_residual_vae_codebook_bits",
        "--outlier_residual_vae_codebook_dim",
        "--lr_warmup_steps",
        "--eval_ppl",
        "--eval_lm_batch_size",
        "--eval_lm_limit",
        "--tasks",
        "--num_fewshot",
        "--lm_batch_size",
        "--lm_limit",
        "--max_train_samples",
        "--finetune_mode",
        "--parallel_stage_decode",
        "--packed_vq_decoder_linear",
        "--decode_device",
        "--decode_group_size",
        "--decoder_hidden_dim",
    }
)

_CATEGORY_SELECTORS = ("default", "cat")
_AFTER_SELECTORS = ("default", "after")

_VAE_STEPS_SPEC = make_positive_int_override_spec(
    arg_name="--vae_steps",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=10000,cat:down_proj=2000",
    min_value=0,
)
_CODEBOOK_BITS_SPEC = make_positive_int_override_spec(
    arg_name="--codebook_bits",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=16,cat:q_proj=24",
)
_CODEBOOK_DIM_SPEC = make_positive_int_override_spec(
    arg_name="--codebook_dim",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=8,cat:down_proj=16",
)
_RESIDUAL_STAGES_SPEC = make_positive_int_override_spec(
    arg_name="--residual_stages",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=1,cat:q_proj=2",
)
_BASE_CH_SPEC = make_positive_int_override_spec(
    arg_name="--base_ch",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=128,cat:q_proj=192",
)
_NUM_RES_BLOCKS_SPEC = make_positive_int_override_spec(
    arg_name="--num_res_blocks",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=1,cat:down_proj=2",
    min_value=0,
)
_DECODER_BASE_CH_SPEC = make_optional_int_override_spec(
    arg_name="--decoder_base_ch",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=none,cat:q_proj=256",
    min_value=1,
)
_DECODER_NUM_RES_BLOCKS_SPEC = make_optional_int_override_spec(
    arg_name="--decoder_num_res_blocks",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=none,cat:q_proj=0",
    min_value=0,
)
_RECON_LOSS_TYPE_SPEC = make_choice_override_spec(
    arg_name="--recon_loss_type",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=mse,cat:q_proj=wa_mse",
    choices=RECON_LOSS_TYPES,
)
_NORM_TYPE_SPEC = make_choice_override_spec(
    arg_name="--norm_type",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=group,cat:q_proj=layer",
    choices=VAE_NORM_TYPES,
)
_ACTIVATION_TYPE_SPEC = make_choice_override_spec(
    arg_name="--activation_type",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=swish,cat:q_proj=relu",
    choices=VAE_ACTIVATION_TYPES,
)
_DECODER_TYPE_SPEC = make_choice_override_spec(
    arg_name="--decoder_type",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=linear,cat:q_proj=asymmetric",
    choices=VAE_DECODER_TYPES,
)
_INTRA_PARALLEL_SPEC = make_override_spec(
    arg_name="--intra_parallel",
    parse_value=lambda raw: parse_intra_parallel_text(raw, arg_name="--intra_parallel"),
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=1x1,cat:down_proj=2x1",
)
_CHANNEL_PROTECT_COUNT_SPEC = make_positive_int_override_spec(
    arg_name="--channel_protect_count",
    allowed_selectors=_CATEGORY_SELECTORS,
    example="default=0,cat:down_proj=64",
    min_value=0,
)
_LORA_RANK_SPEC = make_positive_int_override_spec(
    arg_name="--lora_rank",
    allowed_selectors=_AFTER_SELECTORS,
    example="default=12,after:q_proj=16",
)
_LORA_ALPHA_SPEC = make_override_spec(
    arg_name="--lora_alpha",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--lora_alpha", min_value=0.0, inclusive_min=False),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=24.0,after:q_proj=32.0",
)
_LORA_DROPOUT_SPEC = make_override_spec(
    arg_name="--lora_dropout",
    parse_value=lambda raw: parse_float_text(
        raw,
        arg_name="--lora_dropout",
        min_value=0.0,
        max_value=1.0,
        inclusive_min=True,
        inclusive_max=False,
    ),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.03,after:q_proj=0.1",
)
_STEPS_SPEC = make_positive_int_override_spec(
    arg_name="--steps",
    allowed_selectors=_AFTER_SELECTORS,
    example="default=5000,after:q_proj=200",
    min_value=0,
)
_BATCH_SIZE_SPEC = make_positive_int_override_spec(
    arg_name="--batch_size",
    allowed_selectors=_AFTER_SELECTORS,
    example="default=4,after:q_proj=2",
)
_LEARNING_RATE_SPEC = make_override_spec(
    arg_name="--learning_rate",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--learning_rate"),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=1e-4,after:q_proj=5e-5",
)
_DECODER_LR_SPEC = make_override_spec(
    arg_name="--decoder_lr",
    parse_value=lambda raw: (
        None
        if str(raw).strip().lower() == "none"
        else parse_float_text(raw, arg_name="--decoder_lr")
    ),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=none,after:gate_proj=3e-5",
)
_WEIGHT_DECAY_SPEC = make_override_spec(
    arg_name="--weight_decay",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--weight_decay"),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.001,after:q_proj=0.01",
)
_LOGGING_STEPS_SPEC = make_positive_int_override_spec(
    arg_name="--logging_steps",
    allowed_selectors=_AFTER_SELECTORS,
    example="default=1,after:q_proj=10",
)
_LOSS_TYPE_SPEC = make_override_spec(
    arg_name="--loss_type",
    parse_value=lambda raw: parse_loss_type(raw),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=kl_top,after:q_proj=kd",
)
_TOP_K_SPEC = make_positive_int_override_spec(
    arg_name="--top_k",
    allowed_selectors=_AFTER_SELECTORS,
    example="default=100,after:q_proj=50",
)
_TEMPERATURE_SPEC = make_override_spec(
    arg_name="--temperature",
    parse_value=lambda raw: parse_float_text(
        raw, arg_name="--temperature", min_value=0.0, inclusive_min=False
    ),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=1.0,after:q_proj=2.0",
)
_ALPHA_SPEC = make_override_spec(
    arg_name="--alpha",
    parse_value=lambda raw: parse_float_text(
        raw, arg_name="--alpha", min_value=0.0, max_value=1.0
    ),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.5,after:q_proj=0.3",
)
_PROMPT_LOSS_WEIGHT_SPEC = make_override_spec(
    arg_name="--prompt_loss_weight",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--prompt_loss_weight", min_value=0.0),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.0,after:q_proj=0.05",
)
_HIDDEN_LOSS_WEIGHT_SPEC = make_override_spec(
    arg_name="--hidden_loss_weight",
    parse_value=lambda raw: parse_float_text(raw, arg_name="--hidden_loss_weight", min_value=0.0),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.0,after:q_proj=0.1",
)
_PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC = make_override_spec(
    arg_name="--pre_mlp_hidden_loss_weight",
    parse_value=lambda raw: parse_float_text(
        raw, arg_name="--pre_mlp_hidden_loss_weight", min_value=0.0
    ),
    allowed_selectors=_AFTER_SELECTORS,
    example="default=0.0,after:o_proj=0.001",
)


def collect_explicit_cli_flags(argv: Sequence[str]) -> List[str]:
    flags = set()
    for token in argv:
        text = str(token)
        if text.startswith("--"):
            flags.add(text.split("=", 1)[0].strip())
    return sorted(flag for flag in flags if flag)


def _bool_type(arg_name: str):
    return lambda value: parse_bool_text(value, arg_name=arg_name)


def _error_from_exc(parser: argparse.ArgumentParser, exc: BaseException) -> None:
    parser.error(str(exc))


def reject_deleted_cli_flags(parser: argparse.ArgumentParser, argv: Sequence[str]) -> None:
    explicit = set(collect_explicit_cli_flags(argv))
    deleted = sorted(explicit & DELETED_CLI_FLAGS)
    if deleted:
        parser.error("unrecognized arguments: " + " ".join(deleted))


def _parse_channel_mlp_fuse_weights(raw: object) -> Tuple[float, float, float]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--channel_mlp_fuse_weights must contain exactly 3 comma-separated floats, got {raw!r}."
        )
    parsed = tuple(parse_float_text(part, arg_name="--channel_mlp_fuse_weights", min_value=0.0, inclusive_min=False) for part in parts)
    return parsed  # type: ignore[return-value]


def _parse_gc_kwargs(raw: object) -> Dict[str, object]:
    if raw is None:
        return {"use_reentrant": False}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {"use_reentrant": False}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --gradient_checkpointing_kwargs {raw!r}. Expected JSON object."
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--gradient_checkpointing_kwargs must be a JSON object.")
    return parsed


def _add_data_args(parser: argparse.ArgumentParser, *, dataset_task_default: str) -> None:
    parser.add_argument("--dataset_mix", type=str, default=None)
    parser.add_argument("--dataset_task", type=str, default=dataset_task_default)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--dynamic_padding", type=_bool_type("--dynamic_padding"), default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--group_by_length", type=_bool_type("--group_by_length"), default=True)


def _add_loss_args(parser: argparse.ArgumentParser, *, cat_overrides: bool) -> None:
    if cat_overrides:
        parser.add_argument("--loss_type", type=str, default="default=sft")
        parser.add_argument("--top_k", type=str, default="default=100")
        parser.add_argument("--temperature", type=str, default="default=1.0")
        parser.add_argument("--alpha", type=str, default="default=0.5")
        parser.add_argument("--prompt_loss_weight", type=str, default="default=0.0")
        parser.add_argument("--hidden_loss_weight", type=str, default="default=0.0")
        parser.add_argument("--pre_mlp_hidden_loss_weight", type=str, default="default=0.0")
    else:
        parser.add_argument("--loss_type", type=parse_loss_type, default="sft")
        parser.add_argument("--top_k", type=int, default=100)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--prompt_loss_weight", type=float, default=0.0)
        parser.add_argument("--hidden_loss_weight", type=float, default=0.0)
        parser.add_argument("--pre_mlp_hidden_loss_weight", type=float, default=0.0)
    parser.add_argument("--hidden_layer_weighting", type=parse_hidden_layer_weighting, default="uniform")
    parser.add_argument(
        "--selective_student_topk",
        type=_bool_type("--selective_student_topk"),
        default=False,
    )
    parser.add_argument("--selective_student_topk_chunk_rows", type=int, default=32)


def _add_opt_args(parser: argparse.ArgumentParser, *, cat_overrides: bool) -> None:
    if cat_overrides:
        parser.add_argument("--steps", type=str, default="default=50")
        parser.add_argument("--batch_size", type=str, default="default=2")
        parser.add_argument("--learning_rate", type=str, default="default=1e-4")
        parser.add_argument("--decoder_lr", type=str, default="default=none")
        parser.add_argument("--weight_decay", type=str, default="default=0.0")
        parser.add_argument("--logging_steps", type=str, default="default=1")
    else:
        parser.add_argument("--steps", type=int, default=50)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--decoder_lr", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument(
        "--gradient_checkpointing",
        type=_bool_type("--gradient_checkpointing"),
        default=True,
    )
    parser.add_argument(
        "--gradient_checkpointing_kwargs",
        type=_parse_gc_kwargs,
        default='{"use_reentrant": false}',
    )


def _add_lora_aux_args(parser: argparse.ArgumentParser, *, cat_overrides: bool) -> None:
    if cat_overrides:
        parser.add_argument("--lora_rank", type=str, default="default=12")
        parser.add_argument("--lora_alpha", type=str, default="default=24.0")
        parser.add_argument("--lora_dropout", type=str, default="default=0.03")
    else:
        parser.add_argument("--lora_rank", type=int, default=12)
        parser.add_argument("--lora_alpha", type=float, default=24.0)
        parser.add_argument("--lora_dropout", type=float, default=0.03)
    parser.add_argument("--norm_train_mode", type=str, default="none")
    parser.add_argument("--norm_lr", type=float, default=None)
    parser.add_argument("--lm_head_train_mode", type=str, default="none")
    parser.add_argument("--lm_head_lr", type=float, default=None)


def _add_runtime_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teacher_output_offload", type=str, default="none")
    parser.add_argument("--teacher_model_offload", type=str, default="none")
    parser.add_argument(
        "--teacher_output_pin_memory",
        type=_bool_type("--teacher_output_pin_memory"),
        default=True,
    )
    parser.add_argument("--teacher_output_chunk_tokens", type=int, default=8)
    parser.add_argument(
        "--vae_decoder_checkpoint",
        type=_bool_type("--vae_decoder_checkpoint"),
        default=True,
    )
    parser.add_argument("--parallel_mode", type=str, default="dp")
    parser.add_argument("--layer_device_map", type=str, default="auto")
    parser.add_argument("--offload_mode", type=str, default="none")
    parser.add_argument("--offload_checkpoint", type=_bool_type("--offload_checkpoint"), default=True)
    parser.add_argument("--offload_prefetch_distance", type=int, default=1)
    parser.add_argument("--offload_min_tensor_bytes", type=int, default=1048576)
    parser.add_argument("--offload_pin_memory", type=_bool_type("--offload_pin_memory"), default=True)
    parser.add_argument("--distill_hif4_act", type=_bool_type("--distill_hif4_act"), default=False)
    parser.add_argument("--eval_tasks", type=str, default=None)
    parser.add_argument("--eval_num_fewshot", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=str, default="auto")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--eval_after_save", type=_bool_type("--eval_after_save"), default=False)
    parser.add_argument("--skip_ppl_eval", type=_bool_type("--skip_ppl_eval"), default=False)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_limit", type=int, default=-1)
    parser.add_argument("--eval_hif4_act", type=_bool_type("--eval_hif4_act"), default=False)
    parser.add_argument("--eval_prewarm_group_size", type=int, default=8)


def _add_vae_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codebook_bits", type=str, default="default=16")
    parser.add_argument("--codebook_dim", type=str, default="default=8")
    parser.add_argument("--residual_stages", type=str, default="default=1")
    parser.add_argument("--base_ch", type=str, default="default=128")
    parser.add_argument("--num_res_blocks", type=str, default="default=1")
    parser.add_argument("--quantizer_type", type=str, default="BSQ")
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)
    parser.add_argument("--normalize_weight", type=_bool_type("--normalize_weight"), nargs="?", const=True, default=False)
    parser.add_argument("--new_quant", type=_bool_type("--new_quant"), nargs="?", const=True, default=False)
    parser.add_argument("--transpose_modules", type=str, default="v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--intra_parallel", type=str, default="default=1x1")
    parser.add_argument("--linear_group_size", type=int, default=32)
    parser.add_argument("--allow_tail_group", type=_bool_type("--allow_tail_group"), default=True)
    parser.add_argument("--decoder_base_ch", type=str, default="default=none")
    parser.add_argument("--decoder_num_res_blocks", type=str, default="default=none")
    parser.add_argument("--norm_type", type=str, default="default=group")
    parser.add_argument("--activation_type", type=str, default="default=swish")
    parser.add_argument("--decoder_type", type=str, default="default=linear")
    parser.add_argument("--recon_loss_type", type=str, default="default=mse")
    parser.add_argument("--vae_steps", type=str, default="default=2000")
    parser.add_argument("--vae_batch_size", type=int, default=256)
    parser.add_argument("--vae_learning_rate", type=float, default=1e-4)
    parser.add_argument("--vae_weight_decay", type=float, default=1e-2)
    parser.add_argument(
        "--vae_gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vae_max_grad_norm",
        type=lambda raw: parse_optional_positive_float(raw, arg_name="--vae_max_grad_norm"),
        default=None,
    )
    parser.add_argument("--vae_warmup_ratio", type=float, default=0.0)
    parser.add_argument("--vae_lr_scheduler_type", type=str, default="constant", choices=list(VAE_LR_SCHEDULERS))
    parser.add_argument("--vae_optim", type=str, default="adamw", choices=list(VAE_OPTIMS))
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lfq_weight", type=float, default=1.0)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
    parser.add_argument("--gpu_resident_data", type=_bool_type("--gpu_resident_data"), default=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_blocks", type=int, default=256)


def _add_channel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel_protect_mode", type=str, default="channel", choices=list(CHANNEL_PROTECT_MODES))
    parser.add_argument("--channel_rank_metric", type=str, default="channel_weight_abs", choices=list(CHANNEL_RANK_METRICS))
    parser.add_argument(
        "--channel_mlp_rank_metric",
        type=str,
        default="none",
        choices=list(CHANNEL_MLP_RANK_METRICS),
    )
    parser.add_argument("--channel_mlp_fuse_weights", type=_parse_channel_mlp_fuse_weights, default="1,1,1")
    parser.add_argument("--channel_scope", type=str, default="layer", choices=list(CHANNEL_SCOPES))
    parser.add_argument("--channel_min_per_layer", type=int, default=0)
    parser.add_argument("--channel_quant", type=str, default="none")
    parser.add_argument("--channel_axis", type=str, default="input", choices=list(CHANNEL_AXES))
    parser.add_argument("--channel_protect_count", type=str, default="default=0")


def _add_sparse_bit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bit_active_ratio", type=float, default=0.01)
    parser.add_argument("--bit_optimizer", type=str, default="rms_sgd")
    parser.add_argument("--bit_lr", type=str, default="auto")
    parser.add_argument("--bit_weight_decay", type=float, default=0.0)
    parser.add_argument("--bit_round_steps", type=str, default="auto")


def build_e2e_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end fine-tune compressed checkpoints.",
        allow_abbrev=False,
    )
    parser.add_argument("--student_checkpoint_dir", type=str, required=True)
    parser.add_argument("--run_root_dir", type=str, default=".result/compressed_e2e_fintuning")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_model_path", type=str, default=None)
    parser.add_argument("--train_mode", type=parse_train_mode, default="decoder")
    parser.add_argument("--target_layers", type=parse_target_layers, default="all")
    parser.add_argument("--target_modules", type=parse_target_modules, default="all")
    parser.add_argument("--save_tokenizer", type=_bool_type("--save_tokenizer"), default=True)
    _add_data_args(parser, dataset_task_default="lm")
    _add_loss_args(parser, cat_overrides=False)
    _add_opt_args(parser, cat_overrides=False)
    _add_lora_aux_args(parser, cat_overrides=False)
    _add_runtime_eval_args(parser)
    _add_sparse_bit_args(parser)
    return parser


def build_cat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAT category-wise VAE compression.", allow_abbrev=False)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--compression_categories", type=parse_compression_categories, required=True)
    parser.add_argument("--target_layers", type=parse_target_layers, default="all")
    parser.add_argument("--skip_layers", type=parse_skip_layers, default="")
    parser.add_argument("--after_category_mode", type=parse_after_category_mode, default="none")
    parser.add_argument("--output_dir", type=str, default="./output_linear_by_category")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--deterministic", type=_bool_type("--deterministic"), default=False)
    parser.add_argument("--rot_llm", type=_bool_type("--rot_llm"), nargs="?", const=True, default=False)
    parser.add_argument("--convert", type=_bool_type("--convert"), nargs="?", const=True, default=False)
    parser.add_argument("--convert_device", type=str, default="cuda")
    parser.add_argument("--save_model", type=_bool_type("--save_model"), nargs="?", const=True, default=False)
    parser.add_argument("--save_candidate_artifact", type=_bool_type("--save_candidate_artifact"), default=False)
    parser.add_argument("--candidate_artifact_spec", type=str, default=None)
    parser.add_argument("--candidate_artifact_output_dir", type=str, default=None)
    parser.add_argument("--distill_reset_completed", type=_bool_type("--distill_reset_completed"), default=False)
    parser.add_argument(
        "--distill_independent_categories",
        type=_bool_type("--distill_independent_categories"),
        default=False,
    )
    parser.add_argument("--activation_calib_dataset", type=str, default="")
    parser.add_argument("--activation_calib_nsamples", type=int, default=512)
    parser.add_argument("--activation_calib_seqlen", type=int, default=512)
    parser.add_argument("--activation_calib_seed", type=int, default=0)
    parser.add_argument("--activation_calib_device", type=str, default="")
    parser.add_argument("--activation_calib_log_every", type=int, default=0)
    parser.add_argument("--teacher_model_path", type=str, default=None)
    _add_data_args(parser, dataset_task_default="sft")
    _add_loss_args(parser, cat_overrides=True)
    _add_vae_args(parser)
    _add_channel_args(parser)
    _add_opt_args(parser, cat_overrides=True)
    _add_lora_aux_args(parser, cat_overrides=True)
    _add_runtime_eval_args(parser)
    return parser


def _build_data_config(ns, *, require_source: bool) -> DistillDataConfig:
    cfg = DistillDataConfig(
        dataset_mix=ns.dataset_mix,
        dataset_task=ns.dataset_task,
        train_file=ns.train_file,
        text_field=ns.text_field,
        model_max_length=int(ns.model_max_length),
        dynamic_padding=bool(ns.dynamic_padding),
        seed=int(ns.seed),
        data_seed=int(ns.data_seed),
        group_by_length=bool(ns.group_by_length),
    )
    cfg.validate()
    if require_source and not cfg.dataset_mix and not cfg.train_file:
        raise ValueError("Choose a data source: either --dataset_mix or --train_file.")
    return cfg


def _build_runtime_config(ns) -> DistillRuntimeConfig:
    cfg = DistillRuntimeConfig(
        teacher_output_offload=ns.teacher_output_offload,
        teacher_model_offload=ns.teacher_model_offload,
        teacher_output_pin_memory=bool(ns.teacher_output_pin_memory),
        teacher_output_chunk_tokens=int(ns.teacher_output_chunk_tokens),
        vae_decoder_checkpoint=bool(ns.vae_decoder_checkpoint),
        parallel_mode=ns.parallel_mode,
        layer_device_map=ns.layer_device_map,
        offload_mode=ns.offload_mode,
        offload_checkpoint=bool(ns.offload_checkpoint),
        offload_prefetch_distance=int(ns.offload_prefetch_distance),
        offload_min_tensor_bytes=int(ns.offload_min_tensor_bytes),
        offload_pin_memory=bool(ns.offload_pin_memory),
        distill_hif4_act=bool(ns.distill_hif4_act),
        evaluation=EvaluationRuntimeConfig(
            eval_tasks=ns.eval_tasks,
            eval_num_fewshot=int(ns.eval_num_fewshot),
            eval_batch_size=str(ns.eval_batch_size),
            eval_limit=ns.eval_limit,
            eval_device=str(ns.eval_device),
            eval_after_save=bool(ns.eval_after_save),
            skip_ppl_eval=bool(ns.skip_ppl_eval),
            ppl_seqlen=int(ns.ppl_seqlen),
            ppl_limit=int(ns.ppl_limit),
            eval_hif4_act=bool(ns.eval_hif4_act),
            eval_prewarm_group_size=int(ns.eval_prewarm_group_size),
        ),
    )
    cfg.validate()
    return cfg


def _build_aux_config(ns) -> AuxTrainableConfig:
    cfg = AuxTrainableConfig(
        norm_train_mode=ns.norm_train_mode,
        norm_lr=ns.norm_lr,
        lm_head_train_mode=ns.lm_head_train_mode,
        lm_head_lr=ns.lm_head_lr,
    )
    cfg.validate()
    return cfg


@dataclass
class E2ECLIConfig:
    student_checkpoint_dir: str
    train_mode: str
    data: DistillDataConfig
    loss: DistillLossConfig
    opt: DistillOptimizationConfig
    lora: LoRAConfig
    aux: AuxTrainableConfig
    runtime: DistillRuntimeConfig
    target_layers: TargetLayers
    target_modules: TargetModules
    remaining_argv: Tuple[str, ...]
    explicit_cli_flags: Tuple[str, ...]
    run_root_dir: str
    resume_from_checkpoint: Optional[str]
    teacher_model_path: Optional[str]
    save_tokenizer: bool
    bit_active_ratio: float
    bit_optimizer: str
    bit_lr: str
    bit_weight_decay: float
    bit_round_steps: str


@dataclass
class CatCLIConfig:
    model_path: str
    compression_categories: Tuple[str, ...]
    target_layers: TargetLayers
    skip_layers: FrozenSet[Tuple[int, str]]
    after_category_mode: str
    data: DistillDataConfig
    aux: AuxTrainableConfig
    runtime: DistillRuntimeConfig
    remaining_argv: Tuple[str, ...]
    explicit_cli_flags: Tuple[str, ...]
    vae_steps: OverrideTable
    codebook_bits: OverrideTable
    codebook_dim: OverrideTable
    residual_stages: OverrideTable
    base_ch: OverrideTable
    num_res_blocks: OverrideTable
    decoder_base_ch: OverrideTable
    decoder_num_res_blocks: OverrideTable
    recon_loss_type: OverrideTable
    norm_type: OverrideTable
    activation_type: OverrideTable
    decoder_type: OverrideTable
    intra_parallel: OverrideTable
    channel_protect_count_table: Optional[OverrideTable]
    channel_protect_count_ratio: Optional[float]
    channel_protect_mode: str
    channel_rank_metric: str
    channel_mlp_rank_metric: str
    channel_mlp_fuse_weights: Tuple[float, float, float]
    channel_scope: str
    channel_min_per_layer: int
    channel_quant: str
    channel_axis: str
    vae_opt_template: VAEOptimizationConfig
    core_template: VAECoreConfig
    lora_rank: OverrideTable
    lora_alpha: OverrideTable
    lora_dropout: OverrideTable
    steps: OverrideTable
    batch_size: OverrideTable
    learning_rate: OverrideTable
    decoder_lr: OverrideTable
    weight_decay: OverrideTable
    logging_steps: OverrideTable
    loss_type: OverrideTable
    top_k: OverrideTable
    temperature: OverrideTable
    alpha: OverrideTable
    prompt_loss_weight: OverrideTable
    hidden_loss_weight: OverrideTable
    pre_mlp_hidden_loss_weight: OverrideTable
    hidden_layer_weighting: str
    selective_student_topk: bool
    selective_student_topk_chunk_rows: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: Dict[str, object]
    lora_rank_explicit: bool
    lora_alpha_explicit: bool
    lora_dropout_explicit: bool
    distill_reset_completed: bool
    distill_independent_categories: bool
    teacher_model_path: Optional[str]
    output_dir: str
    resume_from_checkpoint: Optional[str]
    train_device: str
    deterministic: bool
    rot_llm: bool
    convert: bool
    convert_device: str
    save_model: bool
    save_candidate_artifact: bool
    candidate_artifact_spec: Optional[str]
    candidate_artifact_output_dir: Optional[str]
    activation_calib_dataset: str
    activation_calib_nsamples: int
    activation_calib_seqlen: int
    activation_calib_seed: int
    activation_calib_device: str
    activation_calib_log_every: int
    eval_tasks: Optional[str]
    skip_ppl_eval: bool
    ppl_limit: int
    eval_hif4_act: bool

    def resolve_category_config(self, category: str) -> Tuple[VAECompressionConfig, VAEOptimizationConfig]:
        if category not in self.compression_categories:
            raise ValueError(
                f"category {category!r} is not in compression_categories {list(self.compression_categories)}."
            )
        if self.channel_scope == "global":
            count: Union[int, float] = float(self.channel_protect_count_ratio)
        else:
            count = int(resolve_category_value(self.channel_protect_count_table, category))
        compression = VAECompressionConfig(
            core=VAECoreConfig(
                codebook_bits=int(resolve_category_value(self.codebook_bits, category)),
                codebook_dim=int(resolve_category_value(self.codebook_dim, category)),
                residual_stages=int(resolve_category_value(self.residual_stages, category)),
                base_ch=int(resolve_category_value(self.base_ch, category)),
                num_res_blocks=int(resolve_category_value(self.num_res_blocks, category)),
                quantizer_type=self.core_template.quantizer_type,
                gamma0=self.core_template.gamma0,
                gamma=self.core_template.gamma,
                zeta=self.core_template.zeta,
                inv_temperature=self.core_template.inv_temperature,
                normalize_weight=self.core_template.normalize_weight,
                new_quant=self.core_template.new_quant,
                transpose_modules=self.core_template.transpose_modules,
                intra_parallel=tuple(resolve_category_value(self.intra_parallel, category)),
                linear_group_size=self.core_template.linear_group_size,
                allow_tail_group=self.core_template.allow_tail_group,
            ),
            decoder=VAEDecoderConfig(
                decoder_base_ch=resolve_category_value(self.decoder_base_ch, category),
                decoder_num_res_blocks=resolve_category_value(self.decoder_num_res_blocks, category),
                norm_type=resolve_category_value(self.norm_type, category),
                activation_type=resolve_category_value(self.activation_type, category),
                decoder_type=resolve_category_value(self.decoder_type, category),
            ),
            channel=ChannelProtectionConfig(
                channel_protect_mode=self.channel_protect_mode,
                channel_rank_metric=self.channel_rank_metric,
                channel_mlp_rank_metric=self.channel_mlp_rank_metric,
                channel_mlp_fuse_weights=self.channel_mlp_fuse_weights,
                channel_scope=self.channel_scope,
                channel_min_per_layer=self.channel_min_per_layer,
                channel_quant=self.channel_quant,
                channel_axis=self.channel_axis,
                channel_protect_count=count,
            ),
            recon_loss_type=resolve_category_value(self.recon_loss_type, category),
        )
        compression.validate()
        opt = VAEOptimizationConfig(
            vae_steps=int(resolve_category_value(self.vae_steps, category)),
            vae_batch_size=self.vae_opt_template.vae_batch_size,
            vae_learning_rate=self.vae_opt_template.vae_learning_rate,
            vae_weight_decay=self.vae_opt_template.vae_weight_decay,
            vae_gradient_accumulation_steps=self.vae_opt_template.vae_gradient_accumulation_steps,
            vae_max_grad_norm=self.vae_opt_template.vae_max_grad_norm,
            vae_warmup_ratio=self.vae_opt_template.vae_warmup_ratio,
            vae_lr_scheduler_type=self.vae_opt_template.vae_lr_scheduler_type,
            vae_optim=self.vae_opt_template.vae_optim,
            beta1=self.vae_opt_template.beta1,
            beta2=self.vae_opt_template.beta2,
            l1_weight=self.vae_opt_template.l1_weight,
            lfq_weight=self.vae_opt_template.lfq_weight,
            commitment_loss_weight=self.vae_opt_template.commitment_loss_weight,
            entropy_loss_weight=self.vae_opt_template.entropy_loss_weight,
            gpu_resident_data=self.vae_opt_template.gpu_resident_data,
            log_every=self.vae_opt_template.log_every,
            eval_every=self.vae_opt_template.eval_every,
            eval_blocks=self.vae_opt_template.eval_blocks,
        )
        opt.validate()
        return compression, opt

    def resolve_after_category_config(self, category: str) -> AfterCategoryResolvedConfig:
        if category not in self.compression_categories:
            raise ValueError(
                f"category {category!r} is not in compression_categories {list(self.compression_categories)}."
            )
        loss = DistillLossConfig(
            loss_type=resolve_after_category_value(self.loss_type, category),
            top_k=int(resolve_after_category_value(self.top_k, category)),
            temperature=float(resolve_after_category_value(self.temperature, category)),
            alpha=float(resolve_after_category_value(self.alpha, category)),
            prompt_loss_weight=float(resolve_after_category_value(self.prompt_loss_weight, category)),
            hidden_loss_weight=float(resolve_after_category_value(self.hidden_loss_weight, category)),
            pre_mlp_hidden_loss_weight=float(
                resolve_after_category_value(self.pre_mlp_hidden_loss_weight, category)
            ),
            hidden_layer_weighting=self.hidden_layer_weighting,
            selective_student_topk=self.selective_student_topk,
            selective_student_topk_chunk_rows=self.selective_student_topk_chunk_rows,
        )
        loss.validate()
        opt = DistillOptimizationConfig(
            steps=int(resolve_after_category_value(self.steps, category)),
            batch_size=int(resolve_after_category_value(self.batch_size, category)),
            learning_rate=float(resolve_after_category_value(self.learning_rate, category)),
            decoder_lr=resolve_after_category_value(self.decoder_lr, category),
            weight_decay=float(resolve_after_category_value(self.weight_decay, category)),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optim,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs=dict(self.gradient_checkpointing_kwargs),
            logging_steps=int(resolve_after_category_value(self.logging_steps, category)),
        )
        opt.validate()
        lora = LoRAConfig(
            rank=int(resolve_after_category_value(self.lora_rank, category)),
            alpha=float(resolve_after_category_value(self.lora_alpha, category)),
            dropout=float(resolve_after_category_value(self.lora_dropout, category)),
            rank_explicit=self.lora_rank_explicit,
            alpha_explicit=self.lora_alpha_explicit,
            dropout_explicit=self.lora_dropout_explicit,
        )
        lora.validate()
        return AfterCategoryResolvedConfig(
            data=self.data,
            loss=loss,
            opt=opt,
            lora=lora,
            aux=self.aux,
            runtime=self.runtime,
        )


def _parse_override(raw: str, spec: OverrideSpec) -> OverrideTable:
    return parse_override_table(raw, spec)


def parse_e2e_cli(argv: Optional[Sequence[str]] = None) -> E2ECLIConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_e2e_parser()
    reject_deleted_cli_flags(parser, raw_argv)
    try:
        ns, remaining = parser.parse_known_args(raw_argv)
        explicit = tuple(collect_explicit_cli_flags(raw_argv))
        data = _build_data_config(ns, require_source=True)
        loss = DistillLossConfig(
            loss_type=ns.loss_type,
            top_k=int(ns.top_k),
            temperature=float(ns.temperature),
            alpha=float(ns.alpha),
            prompt_loss_weight=float(ns.prompt_loss_weight),
            hidden_loss_weight=float(ns.hidden_loss_weight),
            pre_mlp_hidden_loss_weight=float(ns.pre_mlp_hidden_loss_weight),
            hidden_layer_weighting=ns.hidden_layer_weighting,
            selective_student_topk=bool(ns.selective_student_topk),
            selective_student_topk_chunk_rows=int(ns.selective_student_topk_chunk_rows),
        )
        loss.validate()
        opt = DistillOptimizationConfig(
            steps=int(ns.steps),
            batch_size=int(ns.batch_size),
            learning_rate=float(ns.learning_rate),
            decoder_lr=ns.decoder_lr,
            weight_decay=float(ns.weight_decay),
            gradient_accumulation_steps=int(ns.gradient_accumulation_steps),
            max_grad_norm=float(ns.max_grad_norm),
            warmup_ratio=float(ns.warmup_ratio),
            lr_scheduler_type=str(ns.lr_scheduler_type),
            optim=str(ns.optim),
            gradient_checkpointing=bool(ns.gradient_checkpointing),
            gradient_checkpointing_kwargs=_parse_gc_kwargs(ns.gradient_checkpointing_kwargs),
            logging_steps=int(ns.logging_steps),
        )
        opt.validate()
        lora = LoRAConfig(
            rank=int(ns.lora_rank),
            alpha=float(ns.lora_alpha),
            dropout=float(ns.lora_dropout),
            rank_explicit="--lora_rank" in explicit,
            alpha_explicit="--lora_alpha" in explicit,
            dropout_explicit="--lora_dropout" in explicit,
        )
        lora.validate()
        aux = _build_aux_config(ns)
        validate_train_mode_aux(ns.train_mode, aux)
        runtime = _build_runtime_config(ns)
        return E2ECLIConfig(
            student_checkpoint_dir=str(ns.student_checkpoint_dir),
            train_mode=str(ns.train_mode),
            data=data,
            loss=loss,
            opt=opt,
            lora=lora,
            aux=aux,
            runtime=runtime,
            target_layers=ns.target_layers,
            target_modules=ns.target_modules,
            remaining_argv=tuple(remaining),
            explicit_cli_flags=explicit,
            run_root_dir=str(ns.run_root_dir),
            resume_from_checkpoint=ns.resume_from_checkpoint,
            teacher_model_path=ns.teacher_model_path,
            save_tokenizer=bool(ns.save_tokenizer),
            bit_active_ratio=float(ns.bit_active_ratio),
            bit_optimizer=str(ns.bit_optimizer),
            bit_lr=str(ns.bit_lr),
            bit_weight_decay=float(ns.bit_weight_decay),
            bit_round_steps=str(ns.bit_round_steps),
        )
    except (ValueError, argparse.ArgumentTypeError) as exc:
        _error_from_exc(parser, exc)
        raise


def parse_cat_cli(argv: Optional[Sequence[str]] = None) -> CatCLIConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_cat_parser()
    reject_deleted_cli_flags(parser, raw_argv)
    try:
        ns, remaining = parser.parse_known_args(raw_argv)
        explicit = tuple(collect_explicit_cli_flags(raw_argv))
        categories = tuple(ns.compression_categories)
        target_layers = ns.target_layers
        skip_layers = frozenset(ns.skip_layers)
        validate_skip_layers_scope(
            skip_layers,
            target_layers=target_layers,
            compression_categories=categories,
        )
        after_mode = str(ns.after_category_mode)
        data = _build_data_config(ns, require_source=False)
        if after_mode != "none" and not data.dataset_mix and not data.train_file:
            raise ValueError("--dataset_mix must be set when --after_category_mode is not none.")
        if data.dataset_mix:
            parse_dataset_mix_spec(data.dataset_mix)
        aux = _build_aux_config(ns)
        runtime = _build_runtime_config(ns)
        candidate_artifact = CandidateArtifactConfig(
            save_candidate_artifact=bool(ns.save_candidate_artifact),
            candidate_artifact_spec=ns.candidate_artifact_spec,
            candidate_artifact_output_dir=ns.candidate_artifact_output_dir,
            save_model=bool(ns.save_model),
            convert=bool(ns.convert),
        )
        candidate_artifact.validate()

        vae_steps = _parse_override(ns.vae_steps, _VAE_STEPS_SPEC)
        codebook_bits = _parse_override(ns.codebook_bits, _CODEBOOK_BITS_SPEC)
        codebook_dim = _parse_override(ns.codebook_dim, _CODEBOOK_DIM_SPEC)
        residual_stages = _parse_override(ns.residual_stages, _RESIDUAL_STAGES_SPEC)
        base_ch = _parse_override(ns.base_ch, _BASE_CH_SPEC)
        num_res_blocks = _parse_override(ns.num_res_blocks, _NUM_RES_BLOCKS_SPEC)
        decoder_base_ch = _parse_override(ns.decoder_base_ch, _DECODER_BASE_CH_SPEC)
        decoder_num_res_blocks = _parse_override(ns.decoder_num_res_blocks, _DECODER_NUM_RES_BLOCKS_SPEC)
        recon_loss_type = _parse_override(ns.recon_loss_type, _RECON_LOSS_TYPE_SPEC)
        norm_type = _parse_override(ns.norm_type, _NORM_TYPE_SPEC)
        activation_type = _parse_override(ns.activation_type, _ACTIVATION_TYPE_SPEC)
        decoder_type = _parse_override(ns.decoder_type, _DECODER_TYPE_SPEC)
        intra_parallel = _parse_override(ns.intra_parallel, _INTRA_PARALLEL_SPEC)
        for table in (
            vae_steps,
            codebook_bits,
            codebook_dim,
            residual_stages,
            base_ch,
            num_res_blocks,
            decoder_base_ch,
            decoder_num_res_blocks,
            recon_loss_type,
            norm_type,
            activation_type,
            decoder_type,
            intra_parallel,
        ):
            validate_category_keys(table, categories, table.arg_name)

        scope = str(ns.channel_scope)
        count_raw = str(ns.channel_protect_count)
        if scope == "global":
            if looks_like_override_string(count_raw):
                raise ValueError(
                    "channel_scope=global requires a scalar channel_protect_count ratio; "
                    "default=/cat: override strings are not allowed."
                )
            count_table = None
            count_ratio = parse_float_text(
                count_raw,
                arg_name="--channel_protect_count",
                min_value=0.0,
                max_value=1.0,
                inclusive_min=True,
                inclusive_max=False,
            )
        else:
            count_table = _parse_override(count_raw, _CHANNEL_PROTECT_COUNT_SPEC)
            validate_category_keys(count_table, categories, count_table.arg_name)
            count_ratio = None

        lora_rank = _parse_override(ns.lora_rank, _LORA_RANK_SPEC)
        lora_alpha = _parse_override(ns.lora_alpha, _LORA_ALPHA_SPEC)
        lora_dropout = _parse_override(ns.lora_dropout, _LORA_DROPOUT_SPEC)
        steps = _parse_override(ns.steps, _STEPS_SPEC)
        batch_size = _parse_override(ns.batch_size, _BATCH_SIZE_SPEC)
        learning_rate = _parse_override(ns.learning_rate, _LEARNING_RATE_SPEC)
        decoder_lr = _parse_override(ns.decoder_lr, _DECODER_LR_SPEC)
        weight_decay = _parse_override(ns.weight_decay, _WEIGHT_DECAY_SPEC)
        logging_steps = _parse_override(ns.logging_steps, _LOGGING_STEPS_SPEC)
        loss_type = _parse_override(ns.loss_type, _LOSS_TYPE_SPEC)
        top_k = _parse_override(ns.top_k, _TOP_K_SPEC)
        temperature = _parse_override(ns.temperature, _TEMPERATURE_SPEC)
        alpha = _parse_override(ns.alpha, _ALPHA_SPEC)
        prompt_loss_weight = _parse_override(ns.prompt_loss_weight, _PROMPT_LOSS_WEIGHT_SPEC)
        hidden_loss_weight = _parse_override(ns.hidden_loss_weight, _HIDDEN_LOSS_WEIGHT_SPEC)
        pre_mlp_hidden_loss_weight = _parse_override(
            ns.pre_mlp_hidden_loss_weight, _PRE_MLP_HIDDEN_LOSS_WEIGHT_SPEC
        )
        for table in (
            lora_rank,
            lora_alpha,
            lora_dropout,
            steps,
            batch_size,
            learning_rate,
            decoder_lr,
            weight_decay,
            logging_steps,
            loss_type,
            top_k,
            temperature,
            alpha,
            prompt_loss_weight,
            hidden_loss_weight,
            pre_mlp_hidden_loss_weight,
        ):
            validate_category_keys(table, categories, table.arg_name)

        vae_opt_template = VAEOptimizationConfig(
            vae_steps=0,
            vae_batch_size=int(ns.vae_batch_size),
            vae_learning_rate=float(ns.vae_learning_rate),
            vae_weight_decay=float(ns.vae_weight_decay),
            vae_gradient_accumulation_steps=int(ns.vae_gradient_accumulation_steps),
            vae_max_grad_norm=ns.vae_max_grad_norm,
            vae_warmup_ratio=float(ns.vae_warmup_ratio),
            vae_lr_scheduler_type=str(ns.vae_lr_scheduler_type),
            vae_optim=str(ns.vae_optim),
            beta1=float(ns.beta1),
            beta2=float(ns.beta2),
            l1_weight=float(ns.l1_weight),
            lfq_weight=float(ns.lfq_weight),
            commitment_loss_weight=float(ns.commitment_loss_weight),
            entropy_loss_weight=float(ns.entropy_loss_weight),
            gpu_resident_data=bool(ns.gpu_resident_data),
            log_every=int(ns.log_every),
            eval_every=int(ns.eval_every),
            eval_blocks=int(ns.eval_blocks),
        )
        vae_opt_template.validate()
        core_template = VAECoreConfig(
            quantizer_type=str(ns.quantizer_type),
            gamma0=float(ns.gamma0),
            gamma=float(ns.gamma),
            zeta=float(ns.zeta),
            inv_temperature=float(ns.inv_temperature),
            normalize_weight=bool(ns.normalize_weight),
            new_quant=bool(ns.new_quant),
            transpose_modules=str(ns.transpose_modules),
            linear_group_size=int(ns.linear_group_size),
            allow_tail_group=bool(ns.allow_tail_group),
        )
        core_template.validate()

        cfg = CatCLIConfig(
            model_path=str(ns.model_path),
            compression_categories=categories,
            target_layers=target_layers,
            skip_layers=skip_layers,
            after_category_mode=after_mode,
            data=data,
            aux=aux,
            runtime=runtime,
            remaining_argv=tuple(remaining),
            explicit_cli_flags=explicit,
            vae_steps=vae_steps,
            codebook_bits=codebook_bits,
            codebook_dim=codebook_dim,
            residual_stages=residual_stages,
            base_ch=base_ch,
            num_res_blocks=num_res_blocks,
            decoder_base_ch=decoder_base_ch,
            decoder_num_res_blocks=decoder_num_res_blocks,
            recon_loss_type=recon_loss_type,
            norm_type=norm_type,
            activation_type=activation_type,
            decoder_type=decoder_type,
            intra_parallel=intra_parallel,
            channel_protect_count_table=count_table,
            channel_protect_count_ratio=count_ratio,
            channel_protect_mode=str(ns.channel_protect_mode),
            channel_rank_metric=str(ns.channel_rank_metric),
            channel_mlp_rank_metric=str(ns.channel_mlp_rank_metric),
            channel_mlp_fuse_weights=tuple(ns.channel_mlp_fuse_weights),
            channel_scope=scope,
            channel_min_per_layer=int(ns.channel_min_per_layer),
            channel_quant=str(ns.channel_quant),
            channel_axis=str(ns.channel_axis),
            vae_opt_template=vae_opt_template,
            core_template=core_template,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            decoder_lr=decoder_lr,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            loss_type=loss_type,
            top_k=top_k,
            temperature=temperature,
            alpha=alpha,
            prompt_loss_weight=prompt_loss_weight,
            hidden_loss_weight=hidden_loss_weight,
            pre_mlp_hidden_loss_weight=pre_mlp_hidden_loss_weight,
            hidden_layer_weighting=str(ns.hidden_layer_weighting),
            selective_student_topk=bool(ns.selective_student_topk),
            selective_student_topk_chunk_rows=int(ns.selective_student_topk_chunk_rows),
            gradient_accumulation_steps=int(ns.gradient_accumulation_steps),
            max_grad_norm=float(ns.max_grad_norm),
            warmup_ratio=float(ns.warmup_ratio),
            lr_scheduler_type=str(ns.lr_scheduler_type),
            optim=str(ns.optim),
            gradient_checkpointing=bool(ns.gradient_checkpointing),
            gradient_checkpointing_kwargs=_parse_gc_kwargs(ns.gradient_checkpointing_kwargs),
            lora_rank_explicit="--lora_rank" in explicit,
            lora_alpha_explicit="--lora_alpha" in explicit,
            lora_dropout_explicit="--lora_dropout" in explicit,
            distill_reset_completed=bool(ns.distill_reset_completed),
            distill_independent_categories=bool(ns.distill_independent_categories),
            teacher_model_path=ns.teacher_model_path,
            output_dir=str(ns.output_dir),
            resume_from_checkpoint=ns.resume_from_checkpoint,
            train_device=str(ns.train_device),
            deterministic=bool(ns.deterministic),
            rot_llm=bool(ns.rot_llm),
            convert=bool(ns.convert),
            convert_device=str(ns.convert_device),
            save_model=bool(ns.save_model),
            save_candidate_artifact=bool(ns.save_candidate_artifact),
            candidate_artifact_spec=ns.candidate_artifact_spec,
            candidate_artifact_output_dir=ns.candidate_artifact_output_dir,
            activation_calib_dataset=str(ns.activation_calib_dataset),
            activation_calib_nsamples=int(ns.activation_calib_nsamples),
            activation_calib_seqlen=int(ns.activation_calib_seqlen),
            activation_calib_seed=int(ns.activation_calib_seed),
            activation_calib_device=str(ns.activation_calib_device),
            activation_calib_log_every=int(ns.activation_calib_log_every),
            eval_tasks=ns.eval_tasks,
            skip_ppl_eval=bool(ns.skip_ppl_eval),
            ppl_limit=int(ns.ppl_limit),
            eval_hif4_act=bool(ns.eval_hif4_act),
        )
        if after_mode != "none":
            cfg.resolve_after_category_config(categories[0])
        cfg.resolve_category_config(categories[0])
        return cfg
    except (ValueError, argparse.ArgumentTypeError) as exc:
        _error_from_exc(parser, exc)
        raise

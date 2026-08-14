import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from transformers import HfArgumentParser

from e2e_common.data import MCQA_DATASET_MIX_ALIASES, VAELLM_EDGERAZOR_SFT_ALIASES, normalize_dataset_mix_spec
from e2e_common.e2e_args import parse_decoder_layers, parse_target_modules
from train_utils.lora_training import parse_distill_hidden_alignment_layer_weighting
from train_utils.model_checkpoint_io import resolve_checkpoint_dir
from train_utils.train_args import HFArguments, TrainingArguments, _parse_bool_like, _parse_lora_loss_type


_DEFAULT_RUN_ROOT = ".result/compressed_e2e_fintuning"
_SFT_DATASET_MIX_ALIASES = {"openorca", "alpaca", "longalpaca", "longalign", "race", "sciq"} | VAELLM_EDGERAZOR_SFT_ALIASES
_MCQA_LOSS_TYPES = {"choice_kd", "choice_kd_ce"}
_VALID_FINETUNE_MODES = {"decoder", "compressed_lora", "both"}
_VALID_VAE_TRAIN_MODES = {"decoder", "compressed_lora", "both"}
_VALID_PARALLEL_MODES = {"layer_mp", "dp"}
_VALID_DECODE_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(?::\d+)?)$", re.IGNORECASE)
_DISALLOWED_DENSE_LORA_FLAGS = {
    "--lora_variant",
    "--lora_rank",
    "--lora_alpha",
    "--lora_dropout",
    "--lora_tune_bias",
    "--lora_init_mode",
    "--lora_hif4_act",
    "--adalora_target_r",
    "--adalora_init_r",
    "--adalora_tinit",
    "--adalora_tfinal",
    "--adalora_delta_t",
    "--adalora_beta1",
    "--adalora_beta2",
    "--adalora_orth_reg_weight",
}


@dataclass
class VAEDecoderE2EArguments:
    student_checkpoint_dir: str
    run_root_dir: str = _DEFAULT_RUN_ROOT
    resume_from_checkpoint: Optional[str] = None
    teacher_model_path: Optional[str] = None
    loss_type: str = "sft"
    distill_temperature: float = 1.0
    distill_alpha: float = 0.5
    hidden_loss_weight: float = 0.0
    prompt_kd_weight: float = 0.0
    eakld_confidence_k: int = 16
    hidden_layer_weighting: str = "uniform"
    teacher_output_offload: str = "none"
    teacher_output_pin_memory: bool = True
    teacher_output_chunk_tokens: int = 8
    decoder_layers: str = "all"
    target_modules: str = "all"
    finetune_mode: str = "decoder"
    decode_device: str = "auto"
    decode_group_size: int = 8
    parallel_mode: str = "layer_mp"
    layer_device_map: str = "auto"
    parallel_stage_decode: bool = True
    vae_decoder_checkpoint: bool = True
    tune_final_norm: bool = False
    use_post_norm_head_linear: bool = False
    vae_tune_bias: bool = False
    offload_mode: str = "streaming"
    offload_checkpoint: bool = True
    offload_prefetch_distance: int = 1
    offload_min_tensor_bytes: int = 1048576
    offload_pin_memory: bool = True
    eval_hif4_act: bool = False
    eval_tasks: Optional[str] = None
    eval_num_fewshot: int = 0
    eval_lm_batch_size: str = "auto"
    eval_lm_limit: Optional[int] = None
    eval_device: str = "cuda"
    eval_prewarm_group_size: int = 8
    eval_after_save: bool = False
    skip_ppl_eval: bool = False
    ppl_seqlen: int = 2048
    ppl_limit: int = -1
    dataset_mix: Optional[str] = None
    dataset_task: str = "lm"
    train_file: Optional[str] = None
    text_field: str = "text"
    max_train_samples: Optional[int] = None
    dynamic_padding: bool = False
    save_tokenizer: bool = True
    decoder_layer_ids: Optional[List[int]] = field(default=None, init=False)
    target_module_names: Optional[List[str]] = field(default=None, init=False)
    dataset_mix_sources: Optional[List[str]] = field(default=None, init=False)
    dataset_mix_weights: Optional[List[float]] = field(default=None, init=False)
    dataset_mix_spec: Optional[str] = field(default=None, init=False)
    explicit_cli_flags: Optional[List[str]] = field(default=None, init=False)
    vae_train_mode: str = field(default="decoder", init=False)
    internal_vae_train_mode: str = field(default="decoder", init=False)
    e2e_stage: str = field(default="compressed_e2e_fintuning", init=False)
    e2e_args_key: str = field(default="compressed_e2e_args", init=False)


def _collect_explicit_cli_flags(argv: Sequence[str]) -> List[str]:
    flags = set()
    for token in argv:
        text = str(token)
        if text.startswith("--"):
            flags.add(text.split("=", 1)[0].strip())
    return sorted(flag for flag in flags if flag)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end fine-tune compressed checkpoints.")
    parser.add_argument("--student_checkpoint_dir", type=str, required=True)
    parser.add_argument("--run_root_dir", type=str, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_model_path", type=str, default=None)
    parser.add_argument("--loss_type", type=_parse_lora_loss_type, default="sft")
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--hidden_loss_weight", type=float, default=0.0)
    parser.add_argument("--prompt_kd_weight", type=float, default=0.0)
    parser.add_argument("--eakld_confidence_k", type=int, default=16)
    parser.add_argument("--hidden_layer_weighting", type=str, default="uniform")
    parser.add_argument("--teacher_output_offload", type=str, default="none")
    parser.add_argument(
        "--teacher_output_pin_memory",
        type=lambda value: _parse_bool_like(
            value,
            arg_name="--teacher_output_pin_memory",
        ),
        default=True,
    )
    parser.add_argument("--teacher_output_chunk_tokens", type=int, default=8)
    parser.add_argument("--decoder_layers", type=str, default="all")
    parser.add_argument("--target_modules", type=str, default="all")
    parser.add_argument("--finetune_mode", type=str, default="decoder")
    parser.add_argument("--decode_device", type=str, default="auto")
    parser.add_argument("--decode_group_size", type=int, default=8)
    parser.add_argument("--parallel_mode", type=str, default="layer_mp")
    parser.add_argument("--layer_device_map", type=str, default="auto")
    parser.add_argument(
        "--parallel_stage_decode",
        type=lambda v: _parse_bool_like(v, arg_name="--parallel_stage_decode"),
        default=True,
    )
    parser.add_argument(
        "--vae_decoder_checkpoint",
        type=lambda v: _parse_bool_like(v, arg_name="--vae_decoder_checkpoint"),
        default=True,
    )
    parser.add_argument(
        "--tune_final_norm",
        type=lambda v: _parse_bool_like(v, arg_name="--tune_final_norm"),
        default=False,
    )
    parser.add_argument(
        "--use_post_norm_head_linear",
        type=lambda v: _parse_bool_like(v, arg_name="--use_post_norm_head_linear"),
        default=False,
    )
    parser.add_argument(
        "--vae_tune_bias",
        type=lambda v: _parse_bool_like(v, arg_name="--vae_tune_bias"),
        default=False,
    )
    parser.add_argument("--offload_mode", type=str, default="streaming")
    parser.add_argument(
        "--offload_checkpoint",
        type=lambda v: _parse_bool_like(v, arg_name="--offload_checkpoint"),
        default=True,
    )
    parser.add_argument("--offload_prefetch_distance", type=int, default=1)
    parser.add_argument("--offload_min_tensor_bytes", type=int, default=1048576)
    parser.add_argument(
        "--offload_pin_memory",
        type=lambda v: _parse_bool_like(v, arg_name="--offload_pin_memory"),
        default=True,
    )
    parser.add_argument("--eval_hif4_act", type=lambda v: _parse_bool_like(v, arg_name="--eval_hif4_act"), default=False)
    parser.add_argument("--eval_tasks", "--tasks", dest="eval_tasks", type=str, default=None)
    parser.add_argument("--eval_num_fewshot", "--num_fewshot", dest="eval_num_fewshot", type=int, default=0)
    parser.add_argument("--eval_lm_batch_size", "--lm_batch_size", dest="eval_lm_batch_size", type=str, default="auto")
    parser.add_argument("--eval_lm_limit", "--lm_limit", dest="eval_lm_limit", type=int, default=None)
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--eval_prewarm_group_size", type=int, default=8)
    parser.add_argument(
        "--eval_after_save",
        type=lambda v: _parse_bool_like(v, arg_name="--eval_after_save"),
        default=False,
        help="After Trainer writes a checkpoint on save_steps, run distributed lm-eval.",
    )
    parser.add_argument("--skip_ppl_eval", type=lambda v: _parse_bool_like(v, arg_name="--skip_ppl_eval"), default=False)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_limit", type=int, default=-1)
    parser.add_argument("--dataset_mix", type=str, default=None)
    parser.add_argument("--dataset_task", type=str, default="lm")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--dynamic_padding",
        type=lambda v: _parse_bool_like(v, arg_name="--dynamic_padding"),
        default=False,
    )
    parser.add_argument("--save_tokenizer", type=lambda v: _parse_bool_like(v, arg_name="--save_tokenizer"), default=True)
    return parser


def _validate_dataset_inputs(parser: argparse.ArgumentParser, args: VAEDecoderE2EArguments) -> None:
    explicit_cli_flags = set(args.explicit_cli_flags or [])
    dataset_mix_raw = None if args.dataset_mix is None else str(args.dataset_mix).strip()
    if dataset_mix_raw:
        try:
            sources, weights, spec = normalize_dataset_mix_spec(dataset_mix_raw)
        except ValueError as exc:
            parser.error(str(exc))
        conflicts = ["--train_file", "--text_field", "--max_train_samples"]
        used_conflicts = [flag for flag in conflicts if flag in explicit_cli_flags]
        if used_conflicts:
            parser.error("--dataset_mix cannot be combined with: " + ",".join(used_conflicts))
        args.dataset_mix = dataset_mix_raw
        args.dataset_mix_sources = sources
        args.dataset_mix_weights = weights
        args.dataset_mix_spec = spec
        return

    if not str(args.train_file or "").strip():
        parser.error("Choose a data source: either --dataset_mix or --train_file.")
    train_file = os.path.abspath(str(args.train_file))
    if not os.path.exists(train_file):
        parser.error(f"--train_file does not exist: {train_file}")
    args.train_file = train_file


def validate_args(
    parser: argparse.ArgumentParser,
    args: VAEDecoderE2EArguments,
    training_args: Optional[TrainingArguments],
) -> None:
    checkpoint_dir = str(args.student_checkpoint_dir or "").strip()
    if not checkpoint_dir:
        parser.error("--student_checkpoint_dir is required.")
    try:
        args.student_checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    dataset_task = str(args.dataset_task or "lm").strip().lower()
    if dataset_task not in {"lm", "sft", "mcqa"}:
        parser.error("--dataset_task must be one of: lm | sft | mcqa.")
    args.dataset_task = dataset_task

    _validate_dataset_inputs(parser, args)
    if args.dataset_task == "sft":
        if not args.dataset_mix_spec:
            parser.error("--dataset_task sft currently requires --dataset_mix.")
        unsupported = sorted(set(args.dataset_mix_sources or []) - _SFT_DATASET_MIX_ALIASES)
        if unsupported:
            parser.error(
                "--dataset_task sft supports only these dataset_mix aliases: "
                + ",".join(sorted(_SFT_DATASET_MIX_ALIASES))
                + ". Unsupported: "
                + ",".join(unsupported)
            )
    if args.dataset_task == "mcqa":
        if not args.dataset_mix_spec:
            parser.error("--dataset_task mcqa requires --dataset_mix.")
        unsupported = sorted(set(args.dataset_mix_sources or []) - MCQA_DATASET_MIX_ALIASES)
        if unsupported:
            parser.error(
                "--dataset_task mcqa supports only these dataset_mix aliases: "
                + ",".join(sorted(MCQA_DATASET_MIX_ALIASES))
                + ". Unsupported: "
                + ",".join(unsupported)
            )
        if str(args.loss_type).strip().lower() not in _MCQA_LOSS_TYPES:
            parser.error("--dataset_task mcqa requires --loss_type choice_kd or choice_kd_ce.")
        if float(args.hidden_loss_weight) > 0.0:
            parser.error("--dataset_task mcqa does not support --hidden_loss_weight > 0.")
        if float(args.prompt_kd_weight) != 0.0:
            parser.error(
                "--dataset_task mcqa does not support --prompt_kd_weight != 0 "
                "(choice KD has no token mask)."
            )
    if float(args.distill_temperature) <= 0.0:
        parser.error("--distill_temperature must be > 0.")
    if not (0.0 <= float(args.distill_alpha) <= 1.0):
        parser.error("--distill_alpha must satisfy 0 <= alpha <= 1.")
    if float(args.hidden_loss_weight) < 0.0:
        parser.error("--hidden_loss_weight must be >= 0.")
    if float(args.prompt_kd_weight) < 0.0:
        parser.error("--prompt_kd_weight must be >= 0.")
    if int(args.eakld_confidence_k) < 2:
        parser.error("--eakld_confidence_k must be >= 2.")
    try:
        args.hidden_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
            str(args.hidden_layer_weighting or "uniform")
        )
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(
            str(exc).replace(
                "--distill_hidden_alignment_layer_weighting",
                "--hidden_layer_weighting",
            )
        )
    teacher_output_offload = str(args.teacher_output_offload or "").strip().lower()
    if teacher_output_offload not in {"none", "cpu"}:
        parser.error("--teacher_output_offload must be one of: none | cpu.")
    args.teacher_output_offload = teacher_output_offload

    if int(args.teacher_output_chunk_tokens) < 1:
        parser.error("--teacher_output_chunk_tokens must be >= 1.")
    finetune_mode = str(args.finetune_mode or "").strip().lower()
    if finetune_mode not in _VALID_FINETUNE_MODES:
        parser.error("--finetune_mode must be one of: decoder | compressed_lora | both.")
    args.finetune_mode = finetune_mode
    if finetune_mode not in _VALID_VAE_TRAIN_MODES:
        parser.error("Internal error: invalid --finetune_mode.")
    args.vae_train_mode = finetune_mode
    args.internal_vae_train_mode = finetune_mode
    decode_device = str(args.decode_device or "").strip().lower()
    if not _VALID_DECODE_DEVICE_PATTERN.fullmatch(decode_device):
        parser.error("--decode_device only supports: auto | cpu | cuda | cuda:<index>.")
    args.decode_device = decode_device
    if int(args.decode_group_size) < 1:
        parser.error("--decode_group_size must be >= 1.")
    if finetune_mode == "compressed_lora":
        if bool(args.vae_tune_bias):
            parser.error("--finetune_mode compressed_lora does not support --vae_tune_bias=true.")
        if bool(args.tune_final_norm):
            parser.error("--finetune_mode compressed_lora does not support --tune_final_norm=true.")
        if bool(args.use_post_norm_head_linear):
            parser.error("--finetune_mode compressed_lora does not support --use_post_norm_head_linear=true.")
    if int(args.ppl_seqlen) < 1:
        parser.error("--ppl_seqlen must be >= 1.")
    if int(args.ppl_limit) == 0 or int(args.ppl_limit) < -1:
        parser.error("--ppl_limit must be -1 or >= 1.")
    eval_tasks = None if args.eval_tasks is None else str(args.eval_tasks).strip()
    args.eval_tasks = eval_tasks or None
    if int(args.eval_num_fewshot) < 0:
        parser.error("--eval_num_fewshot must be >= 0.")
    if args.eval_lm_limit is not None and int(args.eval_lm_limit) < 1:
        parser.error("--eval_lm_limit must be >= 1 when provided.")
    if not str(args.eval_lm_batch_size or "").strip():
        parser.error("--eval_lm_batch_size cannot be empty.")
    if not str(args.eval_device or "").strip():
        parser.error("--eval_device cannot be empty.")
    if int(args.eval_prewarm_group_size) < 1:
        parser.error("--eval_prewarm_group_size must be >= 1.")
    if args.max_train_samples is not None and int(args.max_train_samples) < 1:
        parser.error("--max_train_samples must be >= 1 when provided.")
    offload_mode = str(args.offload_mode or "").strip().lower()
    if offload_mode not in {"none", "saved_tensors", "streaming"}:
        parser.error("--offload_mode must be one of: none | saved_tensors | streaming.")
    args.offload_mode = offload_mode
    parallel_mode = str(args.parallel_mode or "").strip().lower()
    if parallel_mode not in _VALID_PARALLEL_MODES:
        parser.error("--parallel_mode must be one of: layer_mp | dp.")
    args.parallel_mode = parallel_mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if parallel_mode == "layer_mp" and world_size != 1:
        parser.error(
            "--parallel_mode layer_mp requires single-process launch (WORLD_SIZE=1). "
            "Use python instead of torchrun, or set --parallel_mode dp."
        )
    if parallel_mode == "dp" and offload_mode == "streaming":
        parser.error("--parallel_mode dp does not support --offload_mode streaming.")
    if offload_mode == "streaming" and world_size != 1:
        parser.error("offload_mode=streaming only supports single-process multi-GPU. Do not launch it with torchrun/DDP.")
    if int(args.offload_prefetch_distance) < 0:
        parser.error("--offload_prefetch_distance must be >= 0.")
    if int(args.offload_min_tensor_bytes) < 0:
        parser.error("--offload_min_tensor_bytes must be >= 0.")

    resume_path = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()
    if resume_path:
        resume_path = os.path.abspath(resume_path)
        if not os.path.isdir(resume_path):
            parser.error(f"--resume_from_checkpoint must be a directory: {resume_path}")
        trainer_state_path = os.path.join(resume_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            parser.error(f"--resume_from_checkpoint must contain trainer_state.json: {resume_path}")
        args.resume_from_checkpoint = resume_path
    else:
        args.resume_from_checkpoint = None

    if bool(args.eval_after_save):
        if not args.eval_tasks:
            parser.error("--eval_after_save=true requires non-empty --eval_tasks.")
        if training_args is None:
            parser.error("--eval_after_save=true requires TrainingArguments (save_strategy/save_steps).")
        save_strategy = getattr(training_args, "save_strategy", None)
        save_strategy_value = getattr(save_strategy, "value", save_strategy)
        if str(save_strategy_value).strip().lower() != "steps":
            parser.error("--eval_after_save=true requires --save_strategy steps.")
        if int(getattr(training_args, "save_steps", 0) or 0) < 1:
            parser.error("--eval_after_save=true requires --save_steps > 0.")

    if training_args is not None:
        fsdp = getattr(training_args, "fsdp", "")
        if not (fsdp is None or fsdp == "" or fsdp == []):
            parser.error("compressed_e2e_fintuning does not support FSDP.")

    args.decoder_layer_ids = parse_decoder_layers(args.decoder_layers)
    args.target_module_names = parse_target_modules(args.target_modules)
    args.layer_device_map = str(args.layer_device_map or "auto").strip().lower()
    args.e2e_stage = "compressed_e2e_fintuning"
    args.e2e_args_key = "compressed_e2e_args"


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[VAEDecoderE2EArguments, HFArguments, TrainingArguments]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    removed_refresh_flags = [
        flag
        for flag in _collect_explicit_cli_flags(raw_argv)
        if flag.startswith("--refresh_sparse_residual")
    ]
    if removed_refresh_flags:
        parser.error("unrecognized arguments: " + " ".join(removed_refresh_flags))
    explicit_flags = set(_collect_explicit_cli_flags(raw_argv))
    if "--vae_train_mode" in explicit_flags:
        parser.error("compressed_e2e_fintuning uses --finetune_mode; do not pass --vae_train_mode.")
    disallowed_lora_flags = sorted(explicit_flags & _DISALLOWED_DENSE_LORA_FLAGS)
    if disallowed_lora_flags:
        parser.error(
            "compressed_e2e_fintuning does not expose dense PEFT flags: "
            + ",".join(disallowed_lora_flags)
        )
    raw_ns, remaining = parser.parse_known_args(raw_argv)
    vae_e2e_args = VAEDecoderE2EArguments(**vars(raw_ns))
    vae_e2e_args.explicit_cli_flags = _collect_explicit_cli_flags(raw_argv)
    # Validate parallel_mode before HF TrainingArguments device setup, which may
    # initialize Accelerate when WORLD_SIZE>1.
    parallel_mode = str(vae_e2e_args.parallel_mode or "").strip().lower()
    if parallel_mode not in _VALID_PARALLEL_MODES:
        parser.error("--parallel_mode must be one of: layer_mp | dp.")
    vae_e2e_args.parallel_mode = parallel_mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if parallel_mode == "layer_mp" and world_size != 1:
        parser.error(
            "--parallel_mode layer_mp requires single-process launch (WORLD_SIZE=1). "
            "Use python instead of torchrun, or set --parallel_mode dp."
        )
    offload_mode = str(vae_e2e_args.offload_mode or "").strip().lower()
    if parallel_mode == "dp" and offload_mode == "streaming":
        parser.error("--parallel_mode dp does not support --offload_mode streaming.")

    # WORLD_SIZE>1 时 HF TrainingArguments 会经 Accelerate 初始化 process group，
    # 默认 ddp_timeout=1800。先按 DISTILL_NCCL_TIMEOUT_SEC 注入/初始化，避免 mid-eval barrier 30min 超时。
    remaining_list = list(remaining)
    if world_size > 1:
        from train_utils.lora_utils import (
            _resolve_distill_process_group_timeout_sec,
            ensure_distill_process_group_initialized,
        )

        distill_pg_timeout_sec = _resolve_distill_process_group_timeout_sec()
        if "--ddp_timeout" not in remaining_list:
            remaining_list.extend(["--ddp_timeout", str(distill_pg_timeout_sec)])
        ensure_distill_process_group_initialized()

    hf_parser = HfArgumentParser((HFArguments, TrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=remaining_list)
    if world_size > 1:
        training_args.ddp_timeout = int(distill_pg_timeout_sec)
        from train_utils.lora_utils import ensure_distill_process_group_initialized

        # Accelerate 可能已用默认超时建组；再次强制写回 DISTILL_NCCL_TIMEOUT_SEC。
        ensure_distill_process_group_initialized()
    validate_args(parser, vae_e2e_args, training_args)
    return vae_e2e_args, hf_args, training_args

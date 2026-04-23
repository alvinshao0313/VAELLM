import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from transformers import HfArgumentParser

from e2e_common.data import normalize_dataset_mix_spec
from train_utils.train_args import HFArguments, TrainingArguments, _parse_bool_like, _parse_lora_loss_type


_DEFAULT_RUN_ROOT = ".result/e2e_raw_fintuning"
_VALID_LORA_VARIANTS = {"plain", "rslora", "dora", "adalora"}
_VALID_LORA_INIT_MODES = {"zero", "gaussian"}
_TARGET_MODULE_ALIASES = {
    "q": "q_proj",
    "query": "q_proj",
    "k": "k_proj",
    "key": "k_proj",
    "v": "v_proj",
    "value": "v_proj",
    "o": "o_proj",
    "out": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


@dataclass
class RawE2EFinetuneArguments:
    student_model_path: str
    run_root_dir: str = _DEFAULT_RUN_ROOT
    resume_from_checkpoint: Optional[str] = None
    teacher_model_path: Optional[str] = None
    loss_type: str = "sft"
    distill_temperature: float = 1.0
    distill_alpha: float = 0.5
    post_attn: bool = False
    decoder_layers: str = "all"
    target_modules: str = "all"
    lora_variant: str = "plain"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_tune_bias: bool = False
    lora_init_mode: str = "zero"
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_delta_t: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5
    lora_hif4_act: bool = False
    eval_hif4_act: bool = False
    skip_ppl_eval: bool = False
    ppl_seqlen: int = 2048
    ppl_limit: int = -1
    dataset_mix: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    text_field: str = "text"
    dataset_num_proc: int = 1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    save_tokenizer: bool = False
    raw_merge_and_save: bool = False
    tune_final_norm: bool = False
    use_post_norm_head_linear: bool = False
    decoder_layer_ids: Optional[List[int]] = field(default=None, init=False)
    target_module_names: Optional[List[str]] = field(default=None, init=False)
    dataset_mix_sources: Optional[List[str]] = field(default=None, init=False)
    dataset_mix_weights: Optional[List[float]] = field(default=None, init=False)
    dataset_mix_spec: Optional[str] = field(default=None, init=False)
    explicit_cli_flags: Optional[List[str]] = field(default=None, init=False)


def parse_decoder_layers(value: Optional[str]) -> Optional[List[int]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None

    out = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            parts = [p.strip() for p in token.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise argparse.ArgumentTypeError(
                    f"Invalid --decoder_layers token '{token}'. Expected <idx> or <begin>-<end>."
                )
            begin = int(parts[0])
            end = int(parts[1])
            if begin < 0 or end < 0 or end < begin:
                raise argparse.ArgumentTypeError(
                    f"Invalid --decoder_layers range '{token}'. Expected non-negative begin <= end."
                )
            out.update(range(begin, end + 1))
            continue

        idx = int(token)
        if idx < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid --decoder_layers token '{token}'. Expected non-negative layer index."
            )
        out.add(idx)

    if not out:
        raise argparse.ArgumentTypeError("--decoder_layers cannot be empty.")
    return sorted(out)


def parse_target_modules(value: Optional[str]) -> Optional[List[str]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None

    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        normalized = _TARGET_MODULE_ALIASES.get(token, token)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    if not out:
        raise argparse.ArgumentTypeError("--target_modules cannot be empty.")
    return out


def needs_teacher(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return norm not in {"", "sft", "origin"}


def parse_lora_variant(value: Optional[str]) -> str:
    norm = str(value or "").strip().lower()
    if not norm:
        norm = "plain"
    if norm not in _VALID_LORA_VARIANTS:
        raise argparse.ArgumentTypeError(
            f"Invalid --lora_variant '{value}'. Expected one of: {sorted(_VALID_LORA_VARIANTS)}."
        )
    return norm


def parse_lora_init_mode(value: Optional[str]) -> str:
    norm = str(value or "").strip().lower()
    if not norm:
        norm = "zero"
    if norm not in _VALID_LORA_INIT_MODES:
        raise argparse.ArgumentTypeError(
            f"Invalid --lora_init_mode '{value}'. Expected one of: {sorted(_VALID_LORA_INIT_MODES)}."
        )
    return norm


def _collect_explicit_cli_flags(argv: Sequence[str]) -> List[str]:
    flags = set()
    for token in argv:
        text = str(token)
        if not text.startswith("--"):
            continue
        option = text.split("=", 1)[0].strip()
        if option:
            flags.add(option)
    return sorted(flags)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw model e2e finetuning with HF/PEFT checkpoints.")
    parser.add_argument("--student_model_path", type=str, required=True)
    parser.add_argument("--run_root_dir", type=str, default=_DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional HF Trainer checkpoint dir, e.g. trainer_state/checkpoint-10000.",
    )
    parser.add_argument("--teacher_model_path", type=str, default=None)
    parser.add_argument(
        "--loss_type",
        type=_parse_lora_loss_type,
        default="sft",
        help="sft/origin/kl/rkl/dual_rkl/mse/kd/kd_top[_K]/dual_kd_top[_K]/dual_kl/dual_kd/kl_top[_K]/r_kl_top[_K]/dual_r_kl_top[_K]/dual_kl_top[_K].",
    )
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument(
        "--post_attn",
        type=lambda v: _parse_bool_like(v, arg_name="--post_attn"),
        default=False,
    )
    parser.add_argument("--decoder_layers", type=str, default="all")
    parser.add_argument("--target_modules", type=str, default="all")
    parser.add_argument("--lora_variant", type=parse_lora_variant, default="plain")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_tune_bias",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_tune_bias"),
        default=False,
    )
    parser.add_argument("--lora_init_mode", type=parse_lora_init_mode, default="zero")
    parser.add_argument("--adalora_target_r", type=int, default=8)
    parser.add_argument("--adalora_init_r", type=int, default=12)
    parser.add_argument("--adalora_tinit", type=int, default=0)
    parser.add_argument("--adalora_tfinal", type=int, default=0)
    parser.add_argument("--adalora_delta_t", type=int, default=1)
    parser.add_argument("--adalora_beta1", type=float, default=0.85)
    parser.add_argument("--adalora_beta2", type=float, default=0.85)
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5)
    parser.add_argument(
        "--lora_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_hif4_act"),
        default=False,
    )
    parser.add_argument(
        "--eval_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--eval_hif4_act"),
        default=False,
    )
    parser.add_argument(
        "--skip_ppl_eval",
        type=lambda v: _parse_bool_like(v, arg_name="--skip_ppl_eval"),
        default=False,
    )
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_limit", type=int, default=-1)
    parser.add_argument("--dataset_mix", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--save_tokenizer",
        type=lambda v: _parse_bool_like(v, arg_name="--save_tokenizer"),
        default=False,
    )
    parser.add_argument(
        "--raw_merge_and_save",
        type=lambda v: _parse_bool_like(v, arg_name="--raw_merge_and_save"),
        default=False,
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
    return parser


def _validate_dataset_inputs(parser: argparse.ArgumentParser, args: RawE2EFinetuneArguments) -> None:
    explicit_cli_flags = set(getattr(args, "explicit_cli_flags", []) or [])
    dataset_mix_raw = None if args.dataset_mix is None else str(args.dataset_mix).strip()
    if dataset_mix_raw:
        try:
            dataset_mix_sources, dataset_mix_weights, dataset_mix_spec = normalize_dataset_mix_spec(dataset_mix_raw)
        except ValueError as exc:
            parser.error(str(exc))
        conflicting_flags = [
            "--dataset_name",
            "--dataset_config_name",
            "--train_split",
            "--eval_split",
            "--train_file",
            "--eval_file",
            "--text_field",
            "--max_train_samples",
            "--max_eval_samples",
        ]
        used_conflicts = [flag for flag in conflicting_flags if flag in explicit_cli_flags]
        if used_conflicts:
            parser.error(
                "--dataset_mix cannot be combined with single-source dataset args: "
                + ",".join(used_conflicts)
            )
        args.dataset_mix = dataset_mix_raw
        args.dataset_mix_sources = dataset_mix_sources
        args.dataset_mix_weights = dataset_mix_weights
        args.dataset_mix_spec = dataset_mix_spec
        return

    use_hf_dataset = bool(str(args.dataset_name or "").strip())
    use_local_files = bool(str(args.train_file or "").strip())
    if use_hf_dataset == use_local_files:
        parser.error(
            "Choose exactly one data source mode: either --dataset_name or --train_file."
        )
    if use_hf_dataset and (args.train_file or args.eval_file):
        parser.error("--train_file/--eval_file cannot be combined with --dataset_name.")
    if use_local_files:
        train_file = os.path.abspath(str(args.train_file))
        if not os.path.exists(train_file):
            parser.error(f"--train_file does not exist: {train_file}")
        if args.eval_file:
            eval_file = os.path.abspath(str(args.eval_file))
            if not os.path.exists(eval_file):
                parser.error(f"--eval_file does not exist: {eval_file}")
    args.dataset_mix = None
    args.dataset_mix_sources = None
    args.dataset_mix_weights = None
    args.dataset_mix_spec = None


def _validate_numeric_inputs(parser: argparse.ArgumentParser, args: RawE2EFinetuneArguments) -> None:
    if float(args.distill_temperature) <= 0.0:
        parser.error("--distill_temperature must be > 0.")
    if float(args.distill_alpha) < 0.0 or float(args.distill_alpha) > 1.0:
        parser.error("--distill_alpha must satisfy 0 <= alpha <= 1.")
    if int(args.lora_rank) < 1:
        parser.error("--lora_rank must be >= 1.")
    if float(args.lora_alpha) <= 0.0:
        parser.error("--lora_alpha must be > 0.")
    if float(args.lora_dropout) < 0.0 or float(args.lora_dropout) >= 1.0:
        parser.error("--lora_dropout must satisfy 0 <= dropout < 1.")
    if int(args.ppl_seqlen) < 1:
        parser.error("--ppl_seqlen must be >= 1.")
    if int(args.ppl_limit) == 0 or int(args.ppl_limit) < -1:
        parser.error("--ppl_limit must be -1 or >= 1.")
    if args.max_train_samples is not None and int(args.max_train_samples) < 1:
        parser.error("--max_train_samples must be >= 1 when provided.")
    if args.max_eval_samples is not None and int(args.max_eval_samples) < 1:
        parser.error("--max_eval_samples must be >= 1 when provided.")
    if int(args.dataset_num_proc) < 1:
        parser.error("--dataset_num_proc must be >= 1.")


def _validate_variant_inputs(
    parser: argparse.ArgumentParser,
    args: RawE2EFinetuneArguments,
    training_args: Optional[TrainingArguments],
) -> None:
    args.lora_variant = parse_lora_variant(args.lora_variant)
    args.lora_init_mode = parse_lora_init_mode(args.lora_init_mode)

    if args.lora_variant == "adalora":
        if int(args.adalora_target_r) < 1:
            parser.error("--adalora_target_r must be >= 1.")
        if int(args.adalora_init_r) < int(args.adalora_target_r):
            parser.error("--adalora_init_r must be >= --adalora_target_r.")
        if int(args.adalora_tinit) < 0:
            parser.error("--adalora_tinit must be >= 0.")
        if int(args.adalora_tfinal) < 0:
            parser.error("--adalora_tfinal must be >= 0.")
        if int(args.adalora_delta_t) < 1:
            parser.error("--adalora_delta_t must be >= 1.")
        if not (0.0 < float(args.adalora_beta1) < 1.0):
            parser.error("--adalora_beta1 must satisfy 0 < beta1 < 1.")
        if not (0.0 < float(args.adalora_beta2) < 1.0):
            parser.error("--adalora_beta2 must satisfy 0 < beta2 < 1.")
        if float(args.adalora_orth_reg_weight) < 0.0:
            parser.error("--adalora_orth_reg_weight must be >= 0.")
        if training_args is None or int(getattr(training_args, "max_steps", -1)) <= 0:
            parser.error("--lora_variant adalora requires TrainingArguments.max_steps > 0.")


def validate_args(
    parser: argparse.ArgumentParser,
    args: RawE2EFinetuneArguments,
    training_args: Optional[TrainingArguments] = None,
) -> None:
    explicit_cli_flags = set(getattr(args, "explicit_cli_flags", []) or [])
    if "--student_checkpoint_dir" in explicit_cli_flags:
        parser.error("raw 训练轨不接受 --student_checkpoint_dir，请使用 --student_model_path。")

    student_model_path = str(args.student_model_path or "").strip()
    if not student_model_path:
        parser.error("--student_model_path is required.")
    args.student_model_path = student_model_path

    _validate_dataset_inputs(parser, args)
    _validate_numeric_inputs(parser, args)
    _validate_variant_inputs(parser, args, training_args)

    resume_path = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()
    if resume_path:
        resume_path = os.path.abspath(resume_path)
        if not os.path.isdir(resume_path):
            parser.error(f"--resume_from_checkpoint must be a directory: {resume_path}")
        trainer_state_path = os.path.join(resume_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            parser.error(
                "--resume_from_checkpoint must point to a HF trainer checkpoint dir "
                f"containing trainer_state.json: {resume_path}"
            )
        args.resume_from_checkpoint = resume_path
    else:
        args.resume_from_checkpoint = None

    args.decoder_layer_ids = parse_decoder_layers(args.decoder_layers)
    args.target_module_names = parse_target_modules(args.target_modules)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[RawE2EFinetuneArguments, HFArguments, TrainingArguments]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    raw_ns, remaining = parser.parse_known_args(raw_argv)
    raw_args = RawE2EFinetuneArguments(**vars(raw_ns))
    raw_args.explicit_cli_flags = _collect_explicit_cli_flags(raw_argv)
    explicit_flags = set(raw_args.explicit_cli_flags or [])
    if "--student_checkpoint_dir" in explicit_flags:
        parser.error("raw 训练轨不接受 --student_checkpoint_dir，请使用 --student_model_path。")
    legacy_raw_flags = sorted(
        flag
        for flag in explicit_flags
        if flag.startswith("--vae_lora_") or flag.startswith("--vae_adalora_")
    )
    if legacy_raw_flags:
        parser.error(
            "raw 训练轨不再接受旧的 vae 前缀 LoRA 参数，请改用无前缀名字："
            + ",".join(legacy_raw_flags)
        )

    hf_parser = HfArgumentParser((HFArguments, TrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=list(remaining))
    validate_args(parser, raw_args, training_args)
    return raw_args, hf_args, training_args

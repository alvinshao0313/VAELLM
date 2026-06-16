import argparse
import sys
from typing import Optional, Sequence, Tuple

from train_utils.train_args import HFArguments, TrainingArguments
from vae_e2e_fintuning.args import VAEDecoderE2EArguments, parse_args as parse_vae_args


_VALID_FINETUNE_MODES = {"decoder", "lora", "both"}
_MODE_TO_VAE_TRAIN_MODE = {
    "decoder": "decoder",
    "lora": "low_rank",
    "both": "both",
}
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


def _build_error_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Unified compressed checkpoint e2e finetuning.")


def _collect_explicit_cli_flags(argv: Sequence[str]) -> list[str]:
    flags = set()
    for token in argv:
        text = str(token)
        if not text.startswith("--"):
            continue
        option = text.split("=", 1)[0].strip()
        if option:
            flags.add(option)
    return sorted(flags)


def _normalize_finetune_mode(parser: argparse.ArgumentParser, value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in _VALID_FINETUNE_MODES:
        parser.error("--finetune_mode must be one of: decoder | lora | both.")
    return mode


def _translate_argv(raw_argv: Sequence[str]) -> Tuple[list[str], str]:
    parser = _build_error_parser()
    translated = []
    finetune_mode: Optional[str] = None
    idx = 0
    raw = [str(item) for item in raw_argv]
    while idx < len(raw):
        token = raw[idx]
        option = token.split("=", 1)[0].strip() if token.startswith("--") else token
        if option == "--vae_train_mode":
            parser.error("compressed_e2e_fintuning uses --finetune_mode; do not pass --vae_train_mode.")
        if option in _DISALLOWED_DENSE_LORA_FLAGS:
            parser.error(f"compressed_e2e_fintuning does not expose dense PEFT flag: {option}")
        if option == "--finetune_mode":
            if "=" in token:
                value = token.split("=", 1)[1]
                idx += 1
            else:
                if idx + 1 >= len(raw) or raw[idx + 1].startswith("--"):
                    parser.error("--finetune_mode requires a value: decoder | lora | both.")
                value = raw[idx + 1]
                idx += 2
            finetune_mode = _normalize_finetune_mode(parser, value)
            continue
        translated.append(token)
        idx += 1

    if finetune_mode is None:
        finetune_mode = "decoder"
    translated.extend(["--vae_train_mode", _MODE_TO_VAE_TRAIN_MODE[finetune_mode]])
    return translated, finetune_mode


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[VAEDecoderE2EArguments, HFArguments, TrainingArguments]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    translated_argv, finetune_mode = _translate_argv(raw_argv)
    args, hf_args, training_args = parse_vae_args(translated_argv)
    args.finetune_mode = str(finetune_mode)
    args.internal_vae_train_mode = str(args.vae_train_mode)
    args.e2e_stage = "compressed_e2e_fintuning"
    args.e2e_args_key = "compressed_e2e_args"
    args.explicit_cli_flags = sorted(set(args.explicit_cli_flags or []) | set(_collect_explicit_cli_flags(raw_argv)))
    return args, hf_args, training_args

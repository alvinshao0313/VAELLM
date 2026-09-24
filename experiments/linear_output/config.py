"""CLI for the isolated experiment; CAT/E2E arguments are not modified."""
from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

DEFAULT_MIX = (
    "edgerazor_ii_7m=0.614,edgerazor_ii_gen=0.121,edgerazor_tulu=0.050,"
    "edgerazor_am=0.115,vaellm_eval_task=0.100"
)
DEFAULT_CATEGORIES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def boolean(text: str) -> bool:
    if text.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false.")
    return text.lower() == "true"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--objective", choices=("weight_mse", "linear_output_mse"), default="linear_output_mse")
    p.add_argument("--compression_categories", default=DEFAULT_CATEGORIES)
    p.add_argument("--target_linears", nargs="+", help="Exact module names; otherwise all requested categories.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=5000, help="Total optimizer updates per Linear, across residual stages.")
    p.add_argument("--batch_size", type=int, default=8, help="Calibration sequences per update, NOT weight vectors.")
    p.add_argument("--calibration_microbatch_size", type=int, default=1)
    p.add_argument("--dataset_mix", default=DEFAULT_MIX)
    p.add_argument("--dataset_task", choices=("sft", "lm"), default="sft")
    p.add_argument("--model_max_length", type=int, default=1024)
    p.add_argument("--dynamic_padding", type=boolean, default=True)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--data_seed", type=int, default=31)
    p.add_argument("--codebook_dim", type=int, default=32)
    p.add_argument("--codebook_bits", type=int, default=64)
    p.add_argument("--residual_stages", type=int, default=1)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--num_res_blocks", type=int, default=0)
    p.add_argument("--decoder_base_ch", type=int, default=128)
    p.add_argument("--decoder_num_res_blocks", type=int, default=1)
    p.add_argument("--norm_type", choices=("layer", "rms", "group", "no"), default="layer")
    p.add_argument("--activation_type", choices=("swish", "relu", "none", "sigmoid", "gelu", "hard_swish"), default="swish")
    p.add_argument("--decoder_type", choices=("linear", "symmetric", "asymmetric"), default="symmetric")
    p.add_argument("--normalize_weight", type=boolean, default=True)
    p.add_argument("--new_quant", type=boolean, default=True)
    p.add_argument("--vae_learning_rate", type=float, default=0.003)
    p.add_argument("--vae_weight_decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--lfq_weight", type=float, default=2.5)
    p.add_argument("--commitment_loss_weight", type=float, default=0.25)
    p.add_argument("--entropy_loss_weight", type=float, default=0.01)
    p.add_argument("--gamma0", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--inv_temperature", type=float, default=100.0)
    p.add_argument("--vae_autocast_dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--vae_chunk_vectors", type=int, default=8192, help="Memory chunk only; BSQ entropy remains global.")
    p.add_argument("--cache_weight_blocks", type=boolean, default=True, help="Keep static normalized/source blocks on GPU between updates; exact objective, more VRAM.")
    p.add_argument("--output_chunk_tokens", type=int, default=512, help="Exact tokenwise accumulation, not sampling.")
    p.add_argument("--log_every", type=int, default=100)
    return p


def validate(args) -> None:
    for key in ("steps", "batch_size", "calibration_microbatch_size", "codebook_dim", "codebook_bits",
                "residual_stages", "base_ch", "decoder_base_ch", "vae_chunk_vectors", "output_chunk_tokens", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive.")
    if args.model_max_length < 2 or args.num_res_blocks < 0 or args.decoder_num_res_blocks < 0:
        raise ValueError("Invalid sequence length or residual block count.")
    if args.residual_stages != 1:
        raise ValueError("The first linear-output algorithm is single-stage; residual_stages must be 1.")
    if args.steps < args.residual_stages:
        raise ValueError("steps must give each residual stage at least one optimizer update.")
    if args.codebook_dim != 32:
        raise ValueError("This experiment fixes input-axis weight vectors to 32 elements.")
    for key in ("vae_learning_rate", "vae_weight_decay", "l1_weight", "lfq_weight",
                "commitment_loss_weight", "entropy_loss_weight", "gamma0", "gamma", "zeta", "inv_temperature"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if args.vae_learning_rate == 0 or args.inv_temperature == 0:
        raise ValueError("Learning rate and inverse temperature must be positive.")
    if not all(0 <= b < 1 for b in (args.beta1, args.beta2)):
        raise ValueError("AdamW betas must lie in [0, 1).")


def stage_steps(total: int, stages: int) -> list[int]:
    if total < stages or stages < 1:
        raise ValueError("Each stage needs at least one step.")
    return [total // stages + (i < total % stages) for i in range(stages)]


def vae_arguments(args) -> SimpleNamespace:
    values = vars(args).copy()
    values.update(
        parallel_layers=1, quantizer_type="BSQ", recon_loss_type="mse",
        vae_weight_dtype="fp32", vae_decoder_checkpoint=False,
        optimizer="adamw", weight_decay=args.vae_weight_decay,
    )
    return SimpleNamespace(**values)

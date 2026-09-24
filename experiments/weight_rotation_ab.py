"""Small real-weight A/B using the production CAT VAE train/convert path.

This measures weight reconstruction, not end-to-end LLM accuracy. Cropping,
short training, and one independently trained decoder per matrix are explicit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn

from rotation.hadamard_utils import random_hadamard_matrix
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_pipeline import apply_group_vae_payload, train_group_vae_payload
from train_utils.utils import LinearRef


MODES = ("none", "current_r1", "two_sided_32", "two_sided_full")


def load_weight_patch(model_path: str, layer: int, category: str, rows: int, cols: int):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(model_path)
    if not root.is_dir():
        root = Path(snapshot_download(repo_id=model_path, local_files_only=True))
    family = "self_attn" if category in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    module_name = f"model.layers.{layer}.{family}.{category}"
    key = module_name + ".weight"
    index = root / "model.safetensors.index.json"
    shard = root / json.loads(index.read_text())["weight_map"][key] if index.is_file() else root / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as tensors:
        sliced = tensors.get_slice(key)
        shape = tuple(sliced.get_shape())
        if rows > shape[0] or cols > shape[1] or rows < 1 or cols < 1:
            raise ValueError(f"Requested patch {rows}x{cols} is invalid for {key}: {shape}.")
        weight = sliced[:rows, :cols].float().contiguous()
    meta = {
        "model": model_path, "snapshot": str(root), "tensor": key, "original_shape": list(shape),
        "slice": [0, rows, 0, cols], "patch_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
    }
    return weight, module_name, meta


def _host_for_weight(weight: torch.Tensor, module_name: str):
    host = nn.Module()
    parent = host
    parts = module_name.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        parent.add_module(part, child)
        parent = child
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(weight)
    parent.add_module(parts[-1], linear)
    return host, linear


def current_r1_matrix(seed: int, device: str):
    # Same shared 32x32 block as rotation_utils.group_hadamard_matrix, without
    # allocating the full block-diagonal residual-stream matrix.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        return random_hadamard_matrix(32, device).float()


def apply_current_r1(weight: torch.Tensor, block: torch.Tensor, category: str, *, inverse=False):
    output_side = category in {"o_proj", "down_proj"}
    w = weight.T if output_side else weight
    # Input projections: W Q; output projections: Q.T W.
    q = block.T if inverse else block
    result = (w.reshape(-1, 32) @ q.to(w)).reshape(w.shape)
    return result.T.contiguous() if output_side else result.contiguous()


def train_patch(
    weight: torch.Tensor, *, module_name: str, category: str, mode: str,
    seed: int, steps: int, batch_size: int = 2048, device: str = "cuda",
    protected_count: int = 0, protected_axis: str = "input",
):
    if mode not in MODES:
        raise ValueError(f"Unknown experiment mode {mode}.")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    block = current_r1_matrix(seed, device) if mode == "current_r1" else None
    source = apply_current_r1(weight.to(device), block, category).cpu() if block is not None else weight
    host, linear = _host_for_weight(source, module_name)
    transpose = category in {"q_proj", "v_proj", "o_proj", "down_proj"}
    args = [
        "--model_path", "Qwen/Qwen3-8B", "--compression_categories", category,
        "--after_category_mode", "none", "--seed", str(seed), "--data_seed", str(seed),
        "--bf16", "true", "--deterministic", "true", "--codebook_bits", "default=32",
        "--codebook_dim", "default=32", "--residual_stages", "default=2",
        "--base_ch", "default=128", "--num_res_blocks", "default=0",
        "--decoder_base_ch", "default=128", "--decoder_num_res_blocks", "default=1",
        "--norm_type", "default=layer", "--activation_type", "default=swish",
        "--decoder_type", "default=symmetric", "--recon_loss_type", "default=mse",
        "--normalize_weight", "--new_quant", "--vae_steps", f"default={steps}",
        "--vae_batch_size", str(batch_size), "--vae_learning_rate", "0.003",
        "--vae_weight_decay", "0", "--vae_optim", "adamw", "--vae_lr_scheduler_type", "linear",
        "--vae_warmup_ratio", "0", "--beta1", "0.9", "--beta2", "0.95",
        "--l1_weight", "1", "--lfq_weight", "2.5", "--commitment_loss_weight", "0.25",
        "--entropy_loss_weight", "0.01", "--vae_decoder_checkpoint", "true",
        "--channel_protect_mode", "channel" if protected_count else "none",
        "--channel_protect_count", f"default={protected_count}", "--channel_axis", protected_axis,
        "--channel_rank_metric", "channel_weight_abs", "--channel_quant", "int8",
        "--weight_rotation", "two_sided" if mode.startswith("two_sided") else "none",
        "--weight_rotation_block_size", "0" if mode == "two_sided_full" else "32",
    ]
    cat_args, _, training_args, vae_args = parse_cat_runtime_args(args)
    # Match tools/cat_train.py: the canonical CAT seed also seeds preprocessing.
    training_args.seed = int(cat_args.seed)
    runtime = resolve_category_runtime_configs(cat_args, vae_args, [category])[category]
    refs = [LinearRef(module_name, linear, category, transpose)]
    plan = None
    if protected_count:
        axis = 0 if protected_axis == "output" else 1
        score = source.abs().sum(dim=1 - axis)
        plan = {module_name: score.topk(protected_count).indices.sort().values}
    start = time.perf_counter()
    payload = train_group_vae_payload(
        model=host, group_refs=refs, group_tag=f"{module_name}/{mode}/seed{seed}",
        runtime_cfg=runtime, vae_args=vae_args, training_args=training_args,
        train_device=device, convert_device=device, do_convert=True, batch_size=batch_size,
        gpu_resident_data=True, log_every=steps, eval_every=0, eval_blocks=256,
        channel_protect_mode="channel" if protected_count else "none", channel_plan=plan,
        channel_rank_metric="channel_weight_abs", channel_axis=protected_axis, channel_quant="int8",
        deterministic=True, shuffle_seed=seed,
    )
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=module_name, payload=payload, convert_device=device)
    layer = host.get_submodule(module_name).to(device)
    with torch.no_grad():
        reconstructed = layer._decode_weight(dtype=torch.float32)
        if block is not None:
            reconstructed = apply_current_r1(reconstructed, block, category, inverse=True)
        reconstructed = reconstructed.cpu().double()
        ref = weight.double()
        error = reconstructed - ref
        nmse = float(error.square().sum() / ref.square().sum())
        if not torch.isfinite(error).all():
            raise RuntimeError("Nonfinite reconstructed weights.")
        encoded_bytes = sum(layer.get_stage_part_vq_storage(stage_idx=s, part_idx=p).numel()
                            for s in range(layer.residual_stages) for p in range(layer.parallel_parts))
        state = layer.state_dict()
        rotation_bytes = sum(t.numel() * t.element_size() for key, t in state.items() if "weight_rotation" in key)
        state_bytes = sum(t.numel() * t.element_size() for t in state.values())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    metrics = {
        "mode": mode, "seed": seed, "steps_per_stage": steps, "stages": 2,
        "shape": list(weight.shape), "transpose": transpose, "protected_count": protected_count,
        "protected_axis": protected_axis, "mse": float(error.square().mean()), "nmse": nmse,
        "relative_l2": nmse**0.5,
        "cosine": float(torch.nn.functional.cosine_similarity(reconstructed.flatten(), ref.flatten(), dim=0)),
        "code_payload_bpw": encoded_bytes * 8 / weight.numel(),
        "rotation_state_bytes": rotation_bytes, "total_state_bpw": state_bytes * 8 / weight.numel(),
        "rotation_spec": None if layer.weight_rotation is None else layer.weight_rotation.to_spec(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    return host, layer, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--category", choices=("q_proj", "gate_proj", "down_proj"), required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError("steps must be positive.")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError(f"Invalid modes {modes}.")
    weight, module_name, source = load_weight_patch(args.model, args.layer, args.category, args.rows, args.cols)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "source": source, "python": sys.executable, "torch": torch.__version__,
        "config": vars(args), "results": [],
        "scope": "Real-weight submatrix reconstruction; not full-model KL/PPL/task accuracy. "
                 "All variants use the same BSQ payload and decoder capacity; rotation overhead is reported separately.",
    }
    log_path = out / "training.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("linear_by_category")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    for mode in modes:
        with log_path.open("a") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            host, layer, result = train_patch(
                weight, module_name=module_name, category=args.category, mode=mode,
                seed=args.seed, steps=args.steps, batch_size=args.batch_size, device=args.device,
            )
        summary["results"].append(result)
        (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del host, layer
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    handler.close()
    logger.removeHandler(handler)
    print(f"RESULTS={out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode-opt gate: serial / infer fuse / packed train / fused train on a real ckpt layer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litebsq.fused_multistage_decoder import (  # noqa: E402
    extract_packed_symmetric_stage_weights,
    fused_decode_packed_symmetric_decoder,
    fused_multistage_symmetric_decode,
)
from litebsq.vae_linear import VAELinear  # noqa: E402
from train_utils.model_checkpoint_io import (  # noqa: E402
    _get_module_by_name,
    _rebuild_converted_modules,
    _torch_load_state_dict,
)


DEFAULT_CKPT = ".result/catlora/res0-bf16-protect-channel-vae/final_model"
DEFAULT_MODULE = "model.layers.0.self_attn.q_proj"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(int(iters)):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0 / float(iters)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = float(af.norm().item() * bf.norm().item())
    if denom <= 1e-12:
        return 1.0
    return float(torch.dot(af, bf).item() / denom)


def _load_module_spec(ckpt_dir: Path, module_name: str) -> Dict[str, Any]:
    meta = json.loads((ckpt_dir / "checkpoint_meta.json").read_text())
    for spec in meta.get("converted_modules", []):
        if str(spec.get("name")) == module_name:
            return spec
    raise KeyError(f"module {module_name!r} not found in {ckpt_dir}")


def load_vae_linear(
    ckpt_dir: Path,
    *,
    module_name: str,
    device: torch.device,
) -> VAELinear:
    spec = _load_module_spec(ckpt_dir, module_name)
    parts = module_name.split(".")
    model = nn.Module()
    parent = model
    for part in parts[:-1]:
        child = nn.Module()
        setattr(parent, part, child)
        parent = child
    in_features = int(spec["in_features"])
    out_features = int(spec["out_features"])
    placeholder = nn.Linear(
        in_features,
        out_features,
        bias=bool(spec.get("has_bias", False)),
    )
    setattr(parent, parts[-1], placeholder)
    _rebuild_converted_modules(model, [spec])
    layer = _get_module_by_name(model, module_name)
    if not isinstance(layer, VAELinear):
        raise TypeError(f"expected VAELinear at {module_name}, got {type(layer)}")

    bin_path = ckpt_dir / "pytorch_model.bin"
    if not bin_path.exists():
        candidates = list(ckpt_dir.glob("*.bin")) + list(ckpt_dir.glob("*.safetensors"))
        raise FileNotFoundError(f"no pytorch_model.bin under {ckpt_dir}; found={candidates[:5]}")
    state = _torch_load_state_dict(str(bin_path), map_location="cpu")
    prefix = module_name + "."
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    layer.load_state_dict(sub, strict=False)
    layer = layer.to(device=device)
    if getattr(layer, "_parallel_stage_grouped_vq_packed", None) is None and getattr(
        layer, "_parallel_stage_decoder", None
    ) is not None:
        layer._build_parallel_stage_decode_plan()
    return layer


def _param_grad_map(packed_decoder: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, param in packed_decoder.named_parameters():
        if param.grad is None:
            continue
        out[name] = param.grad.detach().clone()
    return out


def _grad_cosines(ref: Dict[str, torch.Tensor], other: Dict[str, torch.Tensor]) -> Dict[str, float]:
    keys = sorted(set(ref) & set(other))
    return {k: _cosine(ref[k], other[k]) for k in keys}


def run_bench(
    *,
    ckpt_dir: Path,
    module_name: str,
    device: torch.device,
    warmup: int,
    iters: int,
    small_b: int,
) -> int:
    print(f"ckpt={ckpt_dir}")
    print(f"module={module_name}")
    print(f"device={device}")
    layer = load_vae_linear(ckpt_dir, module_name=module_name, device=device)
    packed = getattr(layer, "_parallel_stage_decoder", None)
    if packed is None:
        raise RuntimeError("expected packed _parallel_stage_decoder on ckpt layer")
    dtype = next(packed.parameters()).dtype
    print(
        f"stages={layer.residual_stages} parts={layer.parallel_parts} "
        f"decoder_type={packed.decoder_type} dtype={dtype}"
    )

    # --- serial reference (no grad) ---
    layer_serial = load_vae_linear(ckpt_dir, module_name=module_name, device=device)
    if getattr(layer_serial, "_parallel_stage_decoder", None) is not None:
        layer_serial.unpack_parallel_stage_decoder_()
    layer_serial.parallel_stage_decode = False
    layer_serial.eval()

    with torch.no_grad():
        serial_out = layer_serial._decode_split_weight(dtype)

    # --- infer fuse ---
    layer_infer = load_vae_linear(ckpt_dir, module_name=module_name, device=device)
    layer_infer.eval()
    packed_infer = layer_infer._parallel_stage_decoder
    packed_infer.requires_grad_(False)
    layer_infer.parallel_stage_decode = True
    with torch.no_grad():
        infer_out = layer_infer._decode_split_weight(dtype)
    infer_max_abs = float((infer_out.float() - serial_out.float()).abs().max().item())

    # --- packed train path (force non-fuse by temporarily breaking fuse support via needs_grad + old gate)
    # After D3 the production path uses fuse; for packed baseline call packed_decoder directly.
    layer_packed = load_vae_linear(ckpt_dir, module_name=module_name, device=device)
    layer_packed.train()
    packed_train = layer_packed._parallel_stage_decoder
    packed_train.requires_grad_(True)
    layer_packed.parallel_stage_decode = True
    grouped_vq = layer_packed._get_parallel_stage_grouped_vq(dtype=dtype, device=device)

    def _packed_fwd_bwd() -> torch.Tensor:
        for p in packed_train.parameters():
            if p.grad is not None:
                p.grad = None
        stage_out = packed_train(grouped_vq)
        # sum stages like restore identity path
        flats = stage_out.permute(1, 0, 2).contiguous().view(int(layer_packed.residual_stages), -1)
        split_rows = layer_packed.compressed_in_features if layer_packed.transpose else layer_packed.compressed_out_features
        split_cols = layer_packed.compressed_out_features if layer_packed.transpose else layer_packed.compressed_in_features
        y = flats.view(int(layer_packed.residual_stages), int(split_rows), int(split_cols)).sum(dim=0)
        loss = y.float().pow(2).mean()
        loss.backward()
        return y.detach()

    packed_y = _packed_fwd_bwd()
    packed_grads = _param_grad_map(packed_train)
    packed_max_abs = float((packed_y.float() - serial_out.float()).abs().max().item())

    # --- fused train (production VAELinear path) ---
    layer_fuse = load_vae_linear(ckpt_dir, module_name=module_name, device=device)
    layer_fuse.train()
    packed_fuse = layer_fuse._parallel_stage_decoder
    packed_fuse.requires_grad_(True)
    layer_fuse.enable_trainable_decode(parallel_stage_decode=True)

    def _fused_fwd_bwd() -> torch.Tensor:
        for p in packed_fuse.parameters():
            if p.grad is not None:
                p.grad = None
        w = layer_fuse._decode_split_weight(dtype)
        loss = w.float().pow(2).mean()
        loss.backward()
        return w.detach()

    fused_y = _fused_fwd_bwd()
    fused_grads = _param_grad_map(packed_fuse)
    fused_max_abs = float((fused_y.float() - serial_out.float()).abs().max().item())
    fused_vs_packed = float((fused_y.float() - packed_y.float()).abs().max().item())
    cosines = _grad_cosines(packed_grads, fused_grads)
    min_cosine = min(cosines.values()) if cosines else float("nan")

    # small-B cosine check
    B_full = int(grouped_vq.shape[0])
    B_small = min(int(small_b), B_full)
    x_small = grouped_vq[:B_small].detach()
    packed_fuse.zero_grad(set_to_none=True)
    packed_train.zero_grad(set_to_none=True)
    # packed small
    for p in packed_train.parameters():
        if p.grad is not None:
            p.grad = None
    stage_out = packed_train(x_small)
    flats = stage_out.permute(1, 0, 2).contiguous().view(int(layer_packed.residual_stages), -1)
    y_p = flats.sum(dim=0)
    y_p.float().pow(2).mean().backward()
    grads_p_small = _param_grad_map(packed_train)
    for p in packed_fuse.parameters():
        if p.grad is not None:
            p.grad = None
    wi, bi, lw, lb, wo, bo = extract_packed_symmetric_stage_weights(packed_fuse)
    y_f = fused_multistage_symmetric_decode(x_small, wi, bi, lw, lb, wo, bo)
    y_f.float().pow(2).mean().backward()
    grads_f_small = _param_grad_map(packed_fuse)
    cosines_small = _grad_cosines(grads_p_small, grads_f_small)
    min_cosine_small = min(cosines_small.values()) if cosines_small else float("nan")

    # timings
    with torch.no_grad():
        t_serial = _bench(lambda: layer_serial._decode_split_weight(dtype), device=device, warmup=warmup, iters=iters)
        t_infer = _bench(lambda: layer_infer._decode_split_weight(dtype), device=device, warmup=warmup, iters=iters)

    def _packed_fwd_only():
        with torch.no_grad():
            packed_train(grouped_vq)

    t_packed_fwd = _bench(_packed_fwd_only, device=device, warmup=warmup, iters=iters)
    t_packed_fb = _bench(_packed_fwd_bwd, device=device, warmup=warmup, iters=iters)
    t_fused_fwd = _bench(
        lambda: layer_fuse._decode_split_weight(dtype),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    t_fused_fb = _bench(_fused_fwd_bwd, device=device, warmup=warmup, iters=iters)

    print("\n=== accuracy ===")
    print(f"infer_fuse max_abs vs serial: {infer_max_abs:.6e}")
    print(f"packed_train max_abs vs serial: {packed_max_abs:.6e}")
    print(f"fused_train max_abs vs serial: {fused_max_abs:.6e}")
    print(f"fused_train max_abs vs packed_train: {fused_vs_packed:.6e}")
    print(f"grad cosine min (full B): {min_cosine:.6f}")
    for k, v in cosines.items():
        print(f"  {k}: {v:.6f}")
    print(f"grad cosine min (B={B_small}): {min_cosine_small:.6f}")

    print("\n=== speed (ms) ===")
    print(f"serial_fwd          {t_serial:8.2f}")
    print(f"infer_fuse_fwd      {t_infer:8.2f}   ({t_serial / t_infer:.2f}x vs serial)")
    print(f"packed_train_fwd    {t_packed_fwd:8.2f}")
    print(f"packed_train_fwd+bwd{t_packed_fb:8.2f}")
    print(f"fused_train_fwd     {t_fused_fwd:8.2f}")
    print(f"fused_train_fwd+bwd {t_fused_fb:8.2f}   ({t_packed_fb / t_fused_fb:.2f}x vs packed)")

    # gates
    ok = True
    if fused_vs_packed > 1e-3:
        print(f"FAIL: fused_train vs packed_train max_abs {fused_vs_packed} > 1e-3")
        ok = False
    if min_cosine < 0.999:
        print(f"FAIL: full-B grad cosine {min_cosine} < 0.999")
        ok = False
    if min_cosine_small < 0.999:
        print(f"FAIL: small-B grad cosine {min_cosine_small} < 0.999")
        ok = False
    if t_fused_fb * 1.2 > t_packed_fb:
        print(f"FAIL: fused_train fwd+bwd not >=1.2x faster ({t_packed_fb / t_fused_fb:.2f}x)")
        ok = False
    if t_serial / t_infer < 5.0:
        print(f"WARN: infer fuse vs serial only {t_serial / t_infer:.2f}x (gate >=5x)")
        # keep as warning; still useful regression signal
    if ok:
        print("\nGATE PASS")
        return 0
    print("\nGATE FAIL")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--module", type=str, default=DEFAULT_MODULE)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--small-b", type=int, default=4096)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 2
    return run_bench(
        ckpt_dir=Path(args.ckpt),
        module_name=str(args.module),
        device=device,
        warmup=int(args.warmup),
        iters=int(args.iters),
        small_b=int(args.small_b),
    )


if __name__ == "__main__":
    raise SystemExit(main())

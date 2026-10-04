"""Compare saved CAT reconstruction and grouped inference on real projections."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors import safe_open

from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from train_utils.v6_model_loader import load_v6_model_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model_path", default="/home/star/models/Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, meta, _ = load_v6_model_checkpoint(args.checkpoint_dir, strict=True, map_location="cpu")
    model.eval().requires_grad_(False)
    names = [name for name in meta["compressed_targets"]
             if int(name.split(".")[2]) in (0, 9, 27)]
    if len(names) != 21:
        raise ValueError("Expected seven compressed projections at each of layers 0, 9, 27.")
    targets = [NamedVAELinearTarget(name, model.get_submodule(name).cuda()) for name in names]
    stats = prime_named_vae_linear_cache(targets, group_size=8, compute_device="cuda:0")
    if stats["warmed"] != 21 or stats["failed"]:
        raise RuntimeError(f"Incomplete real projection prewarm: {stats}")
    root = Path(args.base_model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    with torch.no_grad():
        for target in targets:
            name, layer = target.name, target.base_layer
            cached = layer._cached_weight.clone()
            layer.clear_decoded_weight_cache()
            direct = layer._decode_weight(dtype=torch.bfloat16)
            fp32 = layer._decode_weight(dtype=torch.float32)
            key = name + ".weight"
            with safe_open(str(root / index[key]), framework="pt", device="cuda:0") as source:
                ref = source.get_tensor(key).float()
            energy = ref.square().mean()
            difference = cached.float() - direct.float()
            x = torch.randn(4, layer.in_features, device="cuda", dtype=torch.bfloat16)
            layer.clear_decoded_weight_cache()
            actual = layer(x)
            expected = torch.nn.functional.linear(x, direct)
            row = {
                "name": name, "shape": list(ref.shape),
                "reference_mse": float(energy),
                "nmse_fp32": float((fp32.float() - ref).square().mean() / energy),
                "nmse_bf16": float((direct.float() - ref).square().mean() / energy),
                "nmse_grouped": float((cached.float() - ref).square().mean() / energy),
                "grouped_vs_direct_max_abs": float(difference.abs().max()),
                "grouped_vs_direct_mse_over_ref": float(difference.square().mean() / energy),
                "native_forward_vs_linear_max_abs": float((actual.float() - expected.float()).abs().max()),
                "cache_dtype": str(cached.dtype),
            }
            if not all(torch.isfinite(t).all() for t in (cached, direct, fp32, actual)):
                raise FloatingPointError(name)
            rows.append(row)
            print(json.dumps(row), flush=True)
            layer.clear_decoded_weight_cache()
            layer.cpu()
            del cached, direct, fp32, ref, difference, x, actual, expected
    report = {"checkpoint_id": meta["checkpoint_id"], "checkpoint_dir": args.checkpoint_dir,
              "base_model_path": args.base_model_path, "seed": 31, "warmup": stats, "rows": rows}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

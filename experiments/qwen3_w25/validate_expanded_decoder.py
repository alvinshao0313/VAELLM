"""Compare real R0/R1 native weights and record the deployment-kernel difference."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.qwen3_w25.assemble_categories import PROJECTION_SHAPES, _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.expand_decoder_checkpoint import CODE_FILES, REPO, sha256
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


def difference(value, reference):
    _require(value.shape == reference.shape and torch.isfinite(value).all() and torch.isfinite(reference).all(),
             "Probe shape or finite-value contract failed.")
    delta = value.float() - reference.float()
    return {"equal": torch.equal(value, reference), "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-12))}


def validate(source, expanded, report_path):
    source, expanded, report_path = (Path(path).resolve() for path in (source, expanded, report_path))
    _require(not report_path.exists(), "Numerical report already exists.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly one free validation GPU.")
    migration = json.loads((expanded.parent / "migration.json").read_text())
    _require(migration["status"] == "verified" and Path(migration["source_checkpoint"]) == source
             and Path(migration["checkpoint_dir"]) == expanded, "Verified expansion provenance is required.")
    code_files = (*CODE_FILES, "experiments/qwen3_w25/validate_expanded_decoder.py",
                  "litebsq/fused_multistage_decoder.py", "litebsq/packed_bit_linear.py")
    pins = {name: sha256(REPO / name) for name in code_files}
    started = time.monotonic()
    report = {"status": "running", "source_checkpoint": str(source), "expanded_checkpoint": str(expanded),
              "source_checkpoint_id": migration["source_checkpoint_id"], "checkpoint_id": migration["checkpoint_id"],
              "code_sha256": pins, "migration_sha256": sha256(expanded.parent / "migration.json"),
              "worker_pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
              "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "modules": {},
              "scope": "seven real layer-0 projection geometries; native identity; deployment difference is measured, not assumed zero"}
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        old, old_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        new, new_meta, _ = load_v6_model_checkpoint(str(expanded), map_location="cpu", strict=True, expected_kind="final_model")
        _require(old_meta["checkpoint_id"] == report["source_checkpoint_id"]
                 and new_meta["checkpoint_id"] == report["checkpoint_id"], "Loaded model identity changed.")
        old.eval().requires_grad_(False)
        new.eval().requires_grad_(False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for suffix in PROJECTION_SHAPES:
                name = "model.layers.0." + suffix
                left, right = old.get_submodule(name).cuda(), new.get_submodule(name).cuda()
                _require(packed_symmetric_decoder_supports_fuse(left._parallel_stage_decoder)
                         and not packed_symmetric_decoder_supports_fuse(right._parallel_stage_decoder),
                         "Expected R0 wholefused and R1 native packed deployment routes.")
                VAELinear.reset_fuse_stats()
                fused = left._decode_weight(torch.bfloat16)
                fused_stats = VAELinear.get_fuse_stats()
                left.cache_decoded_weight = False
                VAELinear.reset_fuse_stats()
                native = left._decode_weight(torch.bfloat16)
                native_stats = VAELinear.get_fuse_stats()
                left.cache_decoded_weight = True
                VAELinear.reset_fuse_stats()
                recovered = right._decode_weight(torch.bfloat16)
                expanded_stats = VAELinear.get_fuse_stats()
                _require(fused_stats["packed_u8_linear_hit"] == 0
                         and native_stats["packed_u8_linear_hit"] > 0
                         and expanded_stats["packed_u8_linear_hit"] > 0,
                         "The three decoder routes did not execute their intended kernels.")
                native_delta = difference(recovered, native)
                _require(native_delta["equal"], f"Identity expansion changed native reconstructed weights: {name}.")
                x = torch.randn(1, 4, right.in_features, device="cuda", dtype=torch.bfloat16)
                expected_output = F.linear(x, native)
                output_delta = difference(right(x), expected_output)
                _require(output_delta["equal"], f"Expanded VAELinear forward differs from native reference: {name}.")
                report["modules"][name] = {
                    "native_weight_identity": native_delta, "native_forward_identity": output_delta,
                    "original_fused_to_expanded_native": difference(recovered, fused),
                    "original_fused_stats": fused_stats, "original_native_stats": native_stats,
                    "expanded_native_stats": expanded_stats, "shape": list(recovered.shape),
                }
                print(json.dumps({"module": name, **report["modules"][name]}), flush=True)
                left.clear_decoded_weight_cache()
                right.clear_decoded_weight_cache()
                left.cpu()
                right.cpu()
                del fused, native, recovered, expected_output, x
        report["source_persistent_tensors_compared"] = _assert_state_equal(old.state_dict(), _load_state(source), "Source after CUDA probes")
        report["expanded_persistent_tensors_compared"] = _assert_state_equal(new.state_dict(), _load_state(expanded), "Expanded after CUDA probes")
        del old, new
        gc.collect()
        for name, digest in pins.items():
            _require(sha256(REPO / name) == digest, f"Probe source changed: {name}.")
        report.update(status="verified", full_model_accuracy_measured=False, native_identity_verified=True)
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, completed_at=datetime.now().astimezone().isoformat(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        _atomic_write_json(report, str(report_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--expanded_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_dir, args.expanded_dir, args.report)
    print(json.dumps({"status": result["status"], "modules": len(result["modules"]),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

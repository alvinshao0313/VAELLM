"""Measure an isolated width initialization on the two real decoder geometries."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.expand_decoder_checkpoint import REPO, sha256
from experiments.qwen3_w25.validate_expanded_decoder import difference
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


def validate(source, report_path):
    source, report_path = Path(source).resolve(), Path(report_path).resolve()
    _require(not report_path.exists(), "Width probe report already exists.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose one free validation GPU.")
    files = (
        "experiments/qwen3_w25/widen_decoder.py", "experiments/qwen3_w25/validate_widened_decoder.py",
        "experiments/qwen3_w25/expand_decoder.py", "experiments/qwen3_w25/assemble_categories.py",
        "experiments/qwen3_w25/validate_expanded_decoder.py",
        "litebsq/autoencoder.py", "litebsq/parallel_layers.py", "litebsq/vae_linear.py",
        "litebsq/packed_bit_linear.py", "litebsq/fused_multistage_decoder.py",
        "train_utils/v6_model_loader.py", "train_utils/checkpoint_v6.py",
    )
    pins = {name: sha256(REPO / name) for name in files}
    source_pins = {name: sha256(source / name) for name in ("checkpoint_meta.json", "config.json")}
    source_stat = (source / "pytorch_model.bin").stat()
    started = time.monotonic()
    report = {
        "status": "running", "source_checkpoint": str(source), "code_sha256": pins,
        "source_sha256": source_pins,
        "source_weight_stat": {"size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns, "inode": source_stat.st_ino},
        "worker_pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
        "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "modules": {},
        "scope": "isolated real L0 gate K32 and down K48; numerical and state validation only; no full model export or quality claim",
        "native_relative_l2_limit": torch.finfo(torch.bfloat16).eps,
    }
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        _require(meta["checkpoint_id"] == "552282f2-868a-4105-90df-46f3c6f6a818", "Unexpected probe source.")
        report["source_checkpoint_id"] = meta["checkpoint_id"]
        model.eval().requires_grad_(False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for suffix in ("mlp.gate_proj", "mlp.down_proj"):
                name = "model.layers.0." + suffix
                old = model.get_submodule(name).cuda()
                new = deepcopy(old)
                expansion = widen_packed_decoder_h128_to_h256_(new)
                _require(packed_symmetric_decoder_supports_fuse(old._parallel_stage_decoder)
                         and not packed_symmetric_decoder_supports_fuse(new._parallel_stage_decoder), "Unexpected decoder route.")
                VAELinear.reset_fuse_stats()
                fused = old._decode_weight(torch.bfloat16)
                fused_stats = VAELinear.get_fuse_stats()
                old.cache_decoded_weight = False
                VAELinear.reset_fuse_stats()
                native = old._decode_weight(torch.bfloat16)
                native_stats = VAELinear.get_fuse_stats()
                old.cache_decoded_weight = True
                VAELinear.reset_fuse_stats()
                widened = new._decode_weight(torch.bfloat16)
                widened_stats = VAELinear.get_fuse_stats()
                _require(fused_stats["packed_u8_linear_hit"] == 0 and native_stats["packed_u8_linear_hit"] > 0
                         and widened_stats["packed_u8_linear_hit"] > 0, "Probe did not use the intended routes.")
                delta = difference(widened, native)
                _require(delta["relative_l2"] <= report["native_relative_l2_limit"], f"Native width discrepancy exceeds BF16 scale: {name}.")
                x = torch.randn(1, 4, new.in_features, device="cuda", dtype=torch.bfloat16)
                output = new(x)
                forward = difference(output, F.linear(x, widened))
                _require(forward["equal"], f"Widened VAELinear forward differs from its decoded weight: {name}.")
                outside = {key: value for key, value in old.state_dict().items() if not key.startswith("_parallel_stage_decoder.")}
                _assert_state_equal({key: new.state_dict()[key] for key in outside}, outside, "Nondecoder width probe state")
                report["modules"][name] = {
                    "expansion": expansion, "native_weight_difference": delta,
                    "source_deployment_to_widened_difference": difference(widened, fused),
                    "source_native_to_widened_output_difference": difference(output, F.linear(x, native)),
                    "widened_forward_identity": forward,
                    "source_fused_stats": fused_stats, "source_native_stats": native_stats,
                    "widened_native_stats": widened_stats,
                }
                print(json.dumps({"module": name, **report["modules"][name]}), flush=True)
                old.clear_decoded_weight_cache()
                new.clear_decoded_weight_cache()
                old.cpu()
                del new, fused, native, widened, x, output, outside
        report["source_persistent_tensors_compared"] = _assert_state_equal(model.state_dict(), _load_state(source), "Source after width probes")
        del model
        gc.collect()
        for name, digest in pins.items():
            _require(sha256(REPO / name) == digest, f"Width probe source changed: {name}.")
        for name, digest in source_pins.items():
            _require(sha256(source / name) == digest, f"Protected source changed: {name}.")
        current_stat = (source / "pytorch_model.bin").stat()
        _require((current_stat.st_size, current_stat.st_mtime_ns, current_stat.st_ino)
                 == (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino), "Protected weights changed.")
        report.update(status="verified", full_model_accuracy_measured=False, full_checkpoint_created=False)
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
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_dir, args.report)
    print(json.dumps({"status": result["status"], "modules": len(result["modules"]),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

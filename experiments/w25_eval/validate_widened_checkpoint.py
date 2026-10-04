"""Check the saved L0 down width initialization on its real decoded matrix."""
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

from experiments.qwen3_w25.validate_expanded_decoder import difference
from experiments.w25_eval.model_inventory import sha256
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGET = "model.layers.0.mlp.down_proj"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stat(path):
    value = path.stat()
    return {"path": str(path), "size": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}


def validate(source_dir, checkpoint_dir, migration_report, report_path):
    source, widened, migration_path, report_path = (
        Path(p).resolve() for p in (source_dir, checkpoint_dir, migration_report, report_path))
    _require(not report_path.exists(), "The numerical report already exists.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose one authorized GPU.")
    migration = json.loads(migration_path.read_text())
    _require(migration["status"] == "verified" and migration["selected_targets"] == [TARGET],
             "Require the verified single L0 down migration.")
    metadata = {str(p): sha256(p) for directory in (source, widened)
                for p in (directory / "checkpoint_meta.json", directory / "config.json")}
    stats = [_stat(directory / "pytorch_model.bin") for directory in (source, widened)]
    started = time.monotonic()
    report = {"status": "running", "source_checkpoint": str(source), "checkpoint_dir": str(widened),
              "migration_report": str(migration_path), "migration_sha256": sha256(migration_path),
              "source_checkpoint_id": migration["source_checkpoint_id"], "checkpoint_id": migration["checkpoint_id"],
              "selected_targets": [TARGET], "metadata_sha256": metadata, "weight_stats": stats,
              "probe_source_sha256": sha256(__file__), "worker_pid": os.getpid(),
              "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
              "started_at": datetime.now().astimezone().isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "native_relative_l2_limit": torch.finfo(torch.bfloat16).eps,
              "scope": "Saved L0 down complete merged matrix and one BF16 forward; deployment kernel delta is measured separately; no quality claim."}
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        old, old_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        new, new_meta, _ = load_v6_model_checkpoint(str(widened), map_location="cpu", strict=True, expected_kind="final_model")
        _require(old_meta["checkpoint_id"] == report["source_checkpoint_id"]
                 and new_meta["checkpoint_id"] == report["checkpoint_id"], "Loaded migration identities disagree.")
        old.eval().requires_grad_(False)
        new.eval().requires_grad_(False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            left, right = old.get_submodule(TARGET).cuda(), new.get_submodule(TARGET).cuda()
            _require(left._parallel_stage_decoder.hidden_dim == 128 and right._parallel_stage_decoder.hidden_dim == 256
                     and right._parallel_stage_decoder.decoder_type == "asymmetric"
                     and right._parallel_stage_decoder.num_res_blocks == 0, "Wrong saved width topology.")
            _require(packed_symmetric_decoder_supports_fuse(left._parallel_stage_decoder)
                     and not packed_symmetric_decoder_supports_fuse(right._parallel_stage_decoder), "Unexpected decoder routes.")
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
            widened_stats = VAELinear.get_fuse_stats()
            _require(fused_stats["packed_u8_linear_hit"] == 0 and native_stats["packed_u8_linear_hit"] > 0
                     and widened_stats["packed_u8_linear_hit"] > 0, "The three actual routes did not execute.")
            delta = difference(recovered, native)
            _require(delta["relative_l2"] <= report["native_relative_l2_limit"], "Native width delta exceeds the existing BF16 tolerance.")
            x = torch.randn(1, 4, right.in_features, device="cuda", dtype=torch.bfloat16)
            actual = right(x)
            forward = difference(actual, F.linear(x, recovered))
            _require(forward["equal"], "VAELinear output differs from its actual decoded weight.")
            report["numerical"] = {"weight_shape": list(recovered.shape), "input_shape": list(x.shape),
                "dtype": str(recovered.dtype), "native_weight_difference": delta,
                "source_deployment_to_widened_difference": difference(recovered, fused),
                "source_native_to_widened_output_difference": difference(actual, F.linear(x, native)),
                "widened_forward_identity": forward, "source_fused_stats": fused_stats,
                "source_native_stats": native_stats, "widened_native_stats": widened_stats}
            for layer in (left, right):
                layer.clear_decoded_weight_cache()
                layer.cpu()
            del left, right, fused, native, recovered, x, actual
        del old, new
        gc.collect()
        for path, digest in metadata.items():
            _require(sha256(path) == digest, "Checkpoint metadata changed during the read-only probe.")
        _require([_stat(Path(item["path"])) for item in stats] == stats, "Checkpoint weight identity changed.")
        _require(sha256(migration_path) == report["migration_sha256"], "Migration evidence changed.")
        report.update(status="verified", inputs_unchanged=True, full_model_accuracy_measured=False)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        _atomic_write_json(report, str(report_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--migration_report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = validate(args.source_dir, args.checkpoint_dir, args.migration_report, args.report)
    print(json.dumps({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
                      "numerical": result["numerical"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

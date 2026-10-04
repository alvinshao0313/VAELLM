"""Four-step real-source validation of the isolated final-norm export path."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.e2e_norm_audit import audit_checkpoint
from experiments.qwen3_w25.train_norm_with_step_exports import run_training


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.launch_record.read_text())
    _require(record["status"] == "prepared", "Validation must start from its prepared record.")
    argv = record["training_argv"]
    for name, value in (("--steps", "4"), ("--save_steps", "2"), ("--batch_size", "4"),
                        ("--model_max_length", "1024"), ("--warmup_ratio", "0.01")):
        _require(argv[argv.index(name) + 1] == value, f"Validation requires {name}={value}.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    record.update(status="running", worker_pid=os.getpid(),
                  worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]),
                  started_at=datetime.now().astimezone().isoformat(), actual_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    started = time.monotonic()
    args.launch_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    try:
        for path, digest in record["source_sha256"].items():
            _require(_sha(path) == digest, f"Pinned source changed: {path}.")
        torch.cuda.reset_peak_memory_stats()
        result = run_training(argv, diagnostic_eval_limit=64)
        _require(result["global_step"] == 4, "Real training did not complete all four steps.")
        output = Path(argv[argv.index("--run_root_dir") + 1])
        steps = {}
        for step in (2, 4):
            folder = output / f"step_{step:04d}"
            boundary = json.loads((folder / "diagnostic.json").read_text())
            audit = json.loads((folder / "audit.json").read_text())
            _require(boundary["status"] == "diagnostic_complete" and boundary["diagnostic_eval_limit"] == 64,
                     f"Selected boundary {step} did not complete.")
            _require(audit["status"] == "verified" and audit["frozen_tensors_compared"] == 1910
                     and audit["code_bank_count"] == 504 and audit["changed_code_bits"] == 0
                     and audit["changed_norm_fp32_elements"] > 0
                     and audit["norm_matches_selected_fp32_cast_exactly"]
                     and audit["frozen_backbone_norm_tensors_compared"] == 144,
                     f"Selected boundary {step} did not preserve the frozen source.")
            steps[str(step)] = {"diagnostic_sha256": _sha(folder / "diagnostic.json"),
                               "audit_sha256": _sha(folder / "audit.json"), "checkpoint_id": audit["checkpoint_id"],
                               "changed_norm_fp32_elements": audit["changed_norm_fp32_elements"],
                               "changed_norm_bf16_elements": audit["changed_norm_bf16_elements"]}
        native_audit = audit_checkpoint(record["source_checkpoint"], result["saved_model_dir"], output / "native_final_audit.json",
            training_checkpoint_dir=Path(result["saved_model_dir"]).parent / "trainer_state" / "checkpoint-4")
        for path, digest in record["source_sha256"].items():
            _require(_sha(path) == digest, f"Pinned source changed during validation: {path}.")
        record.update(status="verified", training_result=result, steps=steps,
                      native_final_checkpoint_id=native_audit["checkpoint_id"],
                      native_final_audit_sha256=_sha(output / "native_final_audit.json"),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      training_resumed_after_first_export=True, protocol_is_diagnostic_not_quality=True)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
        args.launch_record.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

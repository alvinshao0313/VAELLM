"""Generate and score fixed retrieval cases without changing candidate weights."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from experiments.longbench_attribution.protocol import TASKS
from experiments.longbench_attribution.restore import frozen_inventory
from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.models import load_model
from experiments.longbench_v1.runner import generate_one, read_input
from experiments.longbench_v1.scoring import score_example
from .prepare import check_candidate_preparation


def run(preparation, output):
    from transformers import AutoTokenizer

    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    record, reference, manifest = check_candidate_preparation(preparation)
    if (record["status"] != "bound_not_started" or record["queued"]
            or type(record["gpu"]) is not int or os.environ.get("CUDA_VISIBLE_DEVICES") != str(record["gpu"])):
        raise ValueError("Bind one assigned GPU explicitly before starting this diagnostic.")
    output.mkdir(parents=True, exist_ok=False)
    config = record["config"]
    report = {"schema": "longbench_candidate_retrieval_result_v1", "status": "running",
              "checkpoint_id": config["model"]["checkpoint_id"], "checkpoint": config["model"]["checkpoint"],
              "preparation_sha256": sha256(preparation), "pid": os.getpid(),
              "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_unix": time.time(),
              "scope": record["scope"], "sample_count": 0, "cases": [], "goal_acceptance": False,
              "model_mutations": False, "model_exported": False}
    report_path = output / "report.json"
    write_json(report_path, report)
    model = tokens = None
    try:
        seed = config["protocol"]["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], use_fast=True, local_files_only=True)
        model, report["load_audit"] = load_model(config, logging.getLogger(__name__))
        before = frozen_inventory(model, ())
        tokens = np.memmap(manifest["inputs"]["path"], dtype="<u4", mode="r")
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
            for i, row in enumerate(record["cases"]):
                generated = generate_one(model, tokenizer, read_input(tokens, row), row,
                                         diagnostic=True, protocol=config["protocol"])
                result = {key: row[key] for key in ("dataset", "row_index", "_id", "answers", "all_classes",
                                                   "length", "input_ids_sha256", "input_tokens")}
                result.update(generated, diagnostic=True, checkpoint_id=report["checkpoint_id"],
                              preparation_sha256=report["preparation_sha256"])
                result["official_score"] = score_example(row["dataset"], generated["pred"], row["answers"], row["all_classes"])
                result["control_scores"] = {kind: reference["controls"][kind][i]["official_score"]
                                            for kind in reference["controls"]}
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                report["cases"].append({key: result[key] for key in ("dataset", "row_index", "input_ids_sha256",
                    "official_score", "control_scores", "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")})
                report["sample_count"] += 1
                write_json(report_path, report)
                logging.info("candidate=%s task=%s row=%d completed=%d/16", report["checkpoint_id"],
                             row["dataset"], row["row_index"], report["sample_count"])
        after = frozen_inventory(model, ())
        if before != after:
            raise ValueError("Generation changed candidate state, buffers or parameter identity.")
        check_candidate_preparation(preparation)
        digest = hashlib.sha256(json.dumps(before["state"], sort_keys=True).encode()).hexdigest()
        report.update(status="diagnostic_complete", exit_code=0, input_pins_unchanged_after=True,
                      frozen_state_and_identity_unchanged=True, frozen_state_sha256=digest,
                      frozen_state_tensors=len(before["state"]), frozen_buffers=len(before["buffers"]),
                      predictions_sha256=sha256(output / "predictions.jsonl"))
        report["selected_task_scores"] = {task: {
            "candidate": 100 * sum(c["official_score"] for c in report["cases"] if c["dataset"] == task) / 8,
            **{kind: 100 * sum(c["control_scores"][kind] for c in report["cases"] if c["dataset"] == task) / 8
               for kind in reference["controls"]}} for task in TASKS}
    except BaseException as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report["started_unix"])
        write_json(report_path, report)
        del model, tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.preparation, args.output)


if __name__ == "__main__":
    main()

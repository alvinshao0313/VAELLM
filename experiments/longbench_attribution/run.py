"""One restoration arm, sixteen existing inputs, existing LongBench generation."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.models import load_model
from experiments.longbench_v1.runner import generate_one, read_input
from experiments.longbench_v1.scoring import score_example
from experiments.w25_eval.model_inventory import model_inventory
from .protocol import ARMS, TASKS, check_preparation
from .restore import frozen_inventory, restore_targets


def run(preparation, arm, output):
    from transformers import AutoTokenizer
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    record, manifest = check_preparation(preparation)
    if arm not in ARMS:
        raise ValueError("Unknown restoration arm.")
    output.mkdir(parents=True, exist_ok=False)
    config = record["config"]
    architecture = model_inventory(config["model"]["hf_path"])
    targets = tuple(name for name in architecture["projection_shapes"] if name.rsplit(".", 1)[1] in ARMS[arm])
    if (architecture != record["architecture"] or list(targets) != record["selected_targets"][arm]
            or len(targets) != architecture["num_hidden_layers"] * len(ARMS[arm])):
        raise ValueError("Incomplete actual layer/category inventory.")
    report = {"schema": "longbench_retrieval_restoration_result_v1", "status": "running", "arm": arm,
        "preparation_sha256": sha256(preparation), "checkpoint_id": config["model"]["checkpoint_id"],
        "pid": os.getpid(), "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_unix": time.time(),
        "scope": record["scope"], "sample_count": 0, "cases": []}
    path = output / "report.json"
    write_json(path, report)
    model = tokens = None
    try:
        seed = config["protocol"]["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], use_fast=True, local_files_only=True)
        model, report["source_load_audit"] = load_model(config, logging.getLogger(__name__))
        torch.cuda.reset_peak_memory_stats()
        report["restoration"] = restore_targets(model, config["model"]["hf_path"], targets)
        report["restoration"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()
        tokens = np.memmap(manifest["inputs"]["path"], dtype="<u4", mode="r")
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
            for i, row in enumerate(record["cases"]):
                generated = generate_one(model, tokenizer, read_input(tokens, row), row, diagnostic=True)
                result = {key: row[key] for key in ("dataset", "row_index", "_id", "answers", "all_classes", "length", "input_ids_sha256", "input_tokens")}
                result.update(generated, diagnostic=True, arm=arm, preparation_sha256=report["preparation_sha256"])
                score = score_example(row["dataset"], generated["pred"], row["answers"], row["all_classes"])
                result["official_score"] = score
                result["control_scores"] = {kind: record["controls"][kind][i]["official_score"] for kind in record["controls"]}
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                report["cases"].append({key: result[key] for key in ("dataset", "row_index", "official_score", "control_scores", "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")})
                report["sample_count"] += 1
                write_json(path, report)
                logging.info("arm=%s task=%s row=%d completed=%d/16", arm, row["dataset"], row["row_index"], report["sample_count"])
        if frozen_inventory(model, targets) != report["restoration"]["frozen_inventory"]:
            raise ValueError("Inference changed a non-target persistent parameter/buffer.")
        check_preparation(preparation)
        report.update(status="diagnostic_complete", exit_code=0, input_pins_unchanged_after=True,
                      non_target_state_unchanged_after=True, predictions_sha256=sha256(output / "predictions.jsonl"))
        report["selected_task_scores"] = {task: 100 * sum(c["official_score"] for c in report["cases"] if c["dataset"] == task) / 8 for task in TASKS}
    except BaseException as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report["started_unix"])
        write_json(path, report)
        del model, tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.preparation, args.arm, args.output)


if __name__ == "__main__":
    main()

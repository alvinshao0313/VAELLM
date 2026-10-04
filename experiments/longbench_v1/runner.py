from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from .common import sha256, write_json
from .data import load_archive
from .models import check_config, load_model
from .protocol import DATASETS, generation_options
from .scoring import check_prediction


def validation_rows(rows):
    # Input geometry chooses cases, never predictions/labels/scores.
    longest = max(rows, key=lambda row: row["input_tokens"])
    summary = max((row for row in rows if row["dataset"] == "gov_report"), key=lambda row: row["input_tokens"])
    samsum = max((row for row in rows if row["dataset"] == "samsum"), key=lambda row: row["input_tokens"])
    return [longest, summary, samsum]


class ForwardTrace:
    """Short-test evidence from the actual model.generate forward calls."""
    def __init__(self):
        self.calls = 0
        self.first_calls = []
        self.cache_decode_calls = 0

    def hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        cache = kwargs.get("past_key_values")
        cache_length = int(cache.get_seq_length()) if cache is not None else 0
        row = {"input_shape": list(ids.shape), "cache_length_before": cache_length,
               "cache_type": None if cache is None else type(cache).__name__,
               "use_cache": kwargs.get("use_cache"), "logits_to_keep": kwargs.get("logits_to_keep")}
        self.calls += 1
        if ids.shape[-1] == 1 and cache_length > 0:
            self.cache_decode_calls += 1
        if len(self.first_calls) < 5:
            self.first_calls.append(row)

    def record(self):
        return {"forward_calls": self.calls, "cache_decode_calls": self.cache_decode_calls,
                "first_calls": self.first_calls}


def read_input(tokens, row):
    start = row["token_offset"]
    values = np.array(tokens[start:start + row["input_tokens"]], dtype="<u4", copy=True)
    if hashlib.sha256(values.tobytes()).hexdigest() != row["input_ids_sha256"]:
        raise ValueError("A shared input token slice changed.")
    return torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0)


def generate_one(model, tokenizer, input_ids, row, *, diagnostic):
    from transformers import GenerationConfig
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    options = generation_options(tokenizer, row["dataset"], input_ids.shape[-1], row["max_new_tokens"])
    effective, _ = model._prepare_generation_config(GenerationConfig(**options), use_model_defaults=False)
    if any(getattr(effective, key) != value for key, value in options.items()):
        raise ValueError("Transformers changed explicit generation settings.")
    trace = ForwardTrace() if diagnostic else None
    handle = model.register_forward_pre_hook(trace.hook, with_kwargs=True) if diagnostic else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            # A fresh shared configuration prevents checkpoint-specific hidden defaults.
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    generation_config=effective, use_model_defaults=False)
        torch.cuda.synchronize()
    finally:
        if handle is not None:
            handle.remove()
    generated = output[0, input_ids.shape[-1]:].detach().cpu().tolist()
    if not torch.equal(output[0, :input_ids.shape[-1]], input_ids[0]):
        raise ValueError("Generation did not preserve the exact input token prefix.")
    report = {"pred": tokenizer.decode(generated, skip_special_tokens=True),
              "output_token_ids": generated, "generated_tokens": len(generated),
              "max_new_tokens": row["max_new_tokens"],
              "hit_generation_limit": len(generated) == row["max_new_tokens"],
              "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "generation_options": options,
              "effective_generation_options": {key: getattr(effective, key) for key in options},
              "use_model_defaults": False}
    if trace is not None:
        report["forward_trace"] = trace.record()
    return report


def run(config_path, output, *, datasets=None, resume=False, max_runtime_seconds=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    diagnostic = config["mode"] == "validation"
    if diagnostic and (datasets is not None or resume or max_runtime_seconds is not None):
        raise ValueError("Validation has exactly three deterministic cases and no partial/resume mode.")
    selected = list(DATASETS) if datasets is None else list(datasets)
    if not selected or len(set(selected)) != len(selected) or any(name not in DATASETS for name in selected):
        raise ValueError("Unknown or repeated dataset partition.")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("A runtime budget must be positive.")
    check_config(config)
    manifest, rows = load_archive(config["archive"]["path"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], use_fast=True, local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config.json"
    if snapshot.exists():
        if snapshot.read_bytes() != config_path.read_bytes():
            raise ValueError("Prediction directory belongs to a different pinned configuration.")
    else:
        shutil.copyfile(config_path, snapshot)
    config_sha = sha256(snapshot)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    run_id = f"{time.time_ns()}_{os.getpid()}"
    reports = output / "runs"
    reports.mkdir(exist_ok=True)
    report_path = reports / (run_id + ".json")
    report = {"schema": "longbench_v1_execution_v1", "mode": config["mode"], "status": "running",
              "config_sha256": config_sha, "archive_sha256": config["archive"]["sha256"],
              "pid": os.getpid(), "start_tick": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "datasets": selected,
              "started_unix": time.time(), "new_predictions": 0, "already_present": 0,
              "quality_evaluation": not diagnostic, "max_runtime_seconds": max_runtime_seconds,
              "runtime_budget_note": "checked between samples; an in-flight official generation is never shortened"}
    write_json(report_path, report)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("longbench_v1")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, tokens, active_files = None, None, []
    try:
        targets = validation_rows(rows) if diagnostic else [row for row in rows if row["dataset"] in selected]
        remaining, handles = [], {}
        for dataset in dict.fromkeys(row["dataset"] for row in targets):
            path = prediction_dir / (dataset + ".jsonl")
            stream = path.open("a+", encoding="utf-8")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            active_files.append(stream)
            stream.seek(0)
            existing = [json.loads(line) for line in stream]
            reference = [row for row in targets if row["dataset"] == dataset]
            if existing and not resume:
                raise FileExistsError(f"Predictions exist for {dataset}; use explicit --resume for the same full run.")
            if len(existing) > len(reference):
                raise ValueError("Prediction file contains too many rows.")
            for pred, row in zip(existing, reference):
                check_prediction(pred, row, config_sha)
            report["already_present"] += len(existing)
            remaining.extend(reference[len(existing):])
            handles[dataset] = stream
        model, report["load_audit"] = load_model(config, logger)
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "total_memory": torch.cuda.get_device_properties(0).total_memory}
        write_json(report_path, report)
        tokens = np.memmap(manifest["inputs"]["path"], dtype="<u4", mode="r")
        diagnostic_records = []
        for row in remaining:
            if max_runtime_seconds is not None and time.time() - report["started_unix"] >= max_runtime_seconds:
                report["status"] = "partial_runtime_budget"
                break
            result = generate_one(model, tokenizer, read_input(tokens, row), row, diagnostic=diagnostic)
            prediction = {key: row[key] for key in
                          ("dataset", "row_index", "_id", "answers", "all_classes", "length", "input_ids_sha256")}
            prediction.update(result, input_tokens=row["input_tokens"], removed_tokens=row["removed_tokens"],
                              config_sha256=config_sha, diagnostic=diagnostic)
            stream = handles[row["dataset"]]
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            report["new_predictions"] += 1
            logger.info("dataset=%s row=%d input=%d generated=%d elapsed=%.3fs", row["dataset"], row["row_index"],
                        row["input_tokens"], result["generated_tokens"], result["elapsed_seconds"])
            if diagnostic:
                diagnostic_records.append(prediction)
        else:
            report["status"] = "diagnostic_complete" if diagnostic else "partition_complete"
        if diagnostic:
            report["cases"] = diagnostic_records
            if len(diagnostic_records) != 3 or max(item["input_tokens"] for item in diagnostic_records) < 32000:
                raise ValueError("Validation did not execute the real long-context case.")
            if not any(item["forward_trace"]["cache_decode_calls"] >= 2 for item in diagnostic_records):
                raise ValueError("Actual multi-step KV-cache generation was not covered.")
            if not any(item["dataset"] == "gov_report" and item["max_new_tokens"] == 512 for item in diagnostic_records):
                raise ValueError("The real 512-token summary generation path was not covered.")
            if not any(item["dataset"] == "samsum" and "min_length" in item["generation_options"] for item in diagnostic_records):
                raise ValueError("The official SAMSum stopping path was not covered.")
        check_config(config)
        report["input_pins_unchanged_after"] = True
        report["exit_code"] = 0
    except BaseException as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        report["duration_seconds"] = report["finished_unix"] - report["started_unix"]
        for stream in active_files:
            stream.close()
        del tokens, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        write_json(report_path, report)
    return report

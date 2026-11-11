"""Official LongBench on the same complete SC-CAT native boundary, without conversion."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint, measure_native
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.longbench_v1 import runner
from experiments.longbench_v1.common import check_hashes, file_stat, sha256
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config as original_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS, generation_options, model_policy
from experiments.longbench_v1.scoring import check_prediction, score_directory
from experiments.sc_cat_live_cmcc_20261106.recovery_io import binding, read_bound_json, read_json, write_json
from tools.cat_eval import _prepare_model_for_eval
from train_utils import checkpoint_v6 as v6


def make_config(deployment_report, inventory_receipt, archive):
    """Partial endpoints can be inspected on CPU; prediction requires final completeness."""
    report, inventory = read_bound_json(deployment_report), read_bound_json(inventory_receipt)["inventory"]
    directory = Path(report["path"]).resolve()
    meta = v6.load_v6_meta(str(directory))
    proof = read_bound_json(report["fresh_proof_reference"])
    require(meta["checkpoint_kind"] == "category_boundary" and report["phase"] == "formal"
        and proof == report["fresh_forward_evidence"] and proof["checkpoint_id"] == meta["checkpoint_id"] == report["uuid"]
        and proof["weight_digest"] == report["digest"] and proof["state_exact"]
        and proof["forward_executed"] and proof["gpu_validation"] and proof["finite_logits"] and proof["exit_code"] == 0,
        "Use the actual formal same-UUID deployment and its own independent fresh proof.")
    hf_path = Path(inventory["source_evidence"]["path"]).resolve()
    archive_path = Path(archive["path"] if isinstance(archive, dict) else archive).resolve()
    if isinstance(archive, dict):
        require(sha256(archive_path) == archive["sha256"], "The original LongBench archive changed.")
    config = original_config(archive_path, kind="dense", hf_path=hf_path, mode="full")
    ledger = measure_checkpoint(directory, inventory)
    complete = not meta["pending_dense_targets"] and not meta["skip_targets"] and (
        set(meta["compressed_targets"]) == {row["actual_name"] for row in inventory["projections"]}
        and meta["completed_categories"] == meta["compression_categories"])
    require(meta.get("lora_config") is None, "All future-dense LoRA must already be merged into this deployed artifact.")
    config["model"].update(kind="compressed", checkpoint=str(directory), checkpoint_id=meta["checkpoint_id"],
        metadata_sha256=sha256(directory / "checkpoint_meta.json"),
        weight_stat=file_stat(directory / meta["state_dict_file"]),
        sc_cat=dict(schema="sc_cat_native_boundary_longbench_v1", complete=bool(complete),
            max_logical_bpw="2.6", max_lora_rank=0, weight_digest=report["digest"],
            deployment_report=deployment_report, inventory_receipt=inventory_receipt, ledger=ledger))
    config["source_sha256"][str(Path(__file__).resolve())] = sha256(__file__)
    check_config(config)
    return config


def load_model(config, logger):
    record, declaration = config["model"], config["model"]["sc_cat"]
    require(declaration["complete"] and declaration["max_logical_bpw"] == "2.6"
        and declaration["max_lora_rank"] == 0 and torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Complete LongBench requires one assigned GPU and a genuinely complete rank0 native boundary.")
    model, fresh = fresh_native(record["checkpoint"], model_factory=lambda path:
        architecture(path, attn_implementation=config["model_policy"]["attn_implementation"]))
    require(fresh["checkpoint_id"] == record["checkpoint_id"]
        and fresh["weight_digest"] == declaration["weight_digest"], "LongBench must load the exact already-evaluated artifact.")
    meta = v6.load_v6_meta(record["checkpoint"])
    inventory = read_bound_json(declaration["inventory_receipt"])["inventory"]
    ledger = measure_native(meta, model.state_dict(), inventory)
    require(ledger["final_budget_applicable"] and ledger["final_budget_passed"]
        and all(ledger[key] == declaration["ledger"][key] for key in ("N", "logical_bits", "packed_bits", "component_fees")),
        "The actual complete native logical/packed/rank ledger changed.")
    model.eval().requires_grad_(False)
    warmup = _prepare_model_for_eval(model, config["device"], 8, logger, {})
    policy = model_policy(record["hf_path"])
    require(not warmup["failed"] and warmup["warmed"] == warmup["total"]
        and model.config._attn_implementation == policy["attn_implementation"]
        and model.config.rope_scaling == policy["rope_scaling"] and model.config.rope_theta == policy["rope_theta"]
        and model.get_input_embeddings().weight.dtype == torch.bfloat16,
        "Keep the original BF16/attention/RoPE policy and native group8 consumer.")
    return model, dict(checkpoint_id=meta["checkpoint_id"], strict_missing=[], strict_unexpected=[],
        same_artifact_weight_digest=fresh["weight_digest"], state_exact=True, bit_budget=ledger,
        warmup=warmup, model_policy=policy, persistent_tensors=len(model.state_dict()),
        rope_buffers={name: dict(dtype=str(value.dtype), shape=list(value.shape))
            for name, value in model.named_buffers() if "inv_freq" in name})


def prepare(request):
    config = make_config(request["deployment_report"], request["inventory_receipt"], request["archive"])
    require(config["model"]["sc_cat"]["complete"] and config["model"]["sc_cat"]["ledger"]["final_budget_passed"],
        "A partial category prefix cannot enter full21 or final-nine selection.")
    output = Path(request["output"]).resolve()
    write_json(output / "config.json", config)
    report = dict(status="prepared_complete_same_artifact_LongBench", exit_code=0,
        config=binding(output / "config.json"), checkpoint_id=config["model"]["checkpoint_id"],
        deployment_report=request["deployment_report"], actual_model_policy=config["model_policy"],
        official_tasks=21, official_samples=4750, GPU_executed=False)
    write_json(output / "prepared.json", report)
    return report


def predict(request):
    from .bindings import bound_path, seconds, validate_worker_request
    authorization = validate_worker_request(request, kind="longbench")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == str(request["gpu"]),
        "Expose exactly this assigned LongBench GPU in caller-activated bitvae.")
    require(0 < request["deadline_epoch"] - time.time() <= seconds(authorization, "longbench_shard"),
        "The LongBench partition deadline exceeds its actual approved window.")
    config_path = Path(request["config"]["path"]).resolve()
    config = read_bound_json(request["config"])
    require(config["mode"] == "full" and config["model"]["sc_cat"]["complete"], "Only complete official task partitions are scheduled.")
    report = read_bound_json(config["model"]["sc_cat"]["deployment_report"])
    raw = read_bound_json(report["source_raw_reference"])
    producer = read_bound_json(raw["request"])
    origin = "core" if request["stage"] == "terminal_longbench" else "replay"
    require(producer["phase"] == report["phase"] == "formal" and producer["stage"] == origin
        and bound_path(producer["authorization"]) == bound_path(request["authorization"])
        and producer["source_manifest"] == authorization["source_manifest"],
        "LongBench must consume this approved search/replay artifact, never a relabeled historical endpoint.")
    tasks = request["datasets"]
    require(tasks and len(set(tasks)) == len(tasks) and set(tasks) <= set(DATASETS), "Use unique complete official tasks.")
    output = Path(request["output"]).resolve()
    require(not output.exists(), "Do not replace or duplicate an existing LongBench partition.")
    runner.load_model = load_model
    remaining = request["deadline_epoch"] - time.time() - 20
    require(remaining > 0, "No time remains for a bounded original-generation task.")
    native = runner.run(config_path, output, datasets=tasks, max_runtime_seconds=remaining)
    require(native["status"] == "partition_complete" and native["exit_code"] == 0
        and native["already_present"] == 0 and native["new_predictions"] == sum(EXPECTED_COUNTS[task] for task in tasks),
        "Partial prediction is a resource failure, never a complete quality result.")
    paths = list((output / "runs").glob("*.json"))
    require(len(paths) == 1 and read_json(paths[0]) == native, "Bind this invocation's unique actual native execution report.")
    report = dict(status="completed_same_artifact_LongBench_partition", exit_code=0,
        config=request["config"], datasets=tasks, output=str(output), native_report=binding(paths[0]),
        checkpoint_id=config["model"]["checkpoint_id"],
        prediction_sha256={str(output / "predictions" / (task + ".jsonl")):
            sha256(output / "predictions" / (task + ".jsonl")) for task in tasks})
    write_json(output / "partition_result.json", report)
    return report


def _retired(native):
    path = Path(f"/proc/{native['pid']}/stat")
    if path.exists():
        tick = int(path.read_text().rsplit(")", 1)[1].split()[19])
        require(tick != native["start_tick"], "Only a retired LongBench worker may publish terminal merge input.")


def partition_sources(config_reference, partition_references):
    """Validate actual retired partition outputs independently of model availability."""
    config = read_bound_json(config_reference)
    manifest, rows = load_archive(config["archive"]["path"])
    config_sha = config_reference["sha256"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], local_files_only=True, use_fast=True)
    sources, partitions, outputs = {}, [], set()
    for reference in partition_references:
        part = read_bound_json(reference)
        native = read_bound_json(part["native_report"])
        _retired(native)
        owner = str(Path(part["output"]).resolve())
        require(owner not in outputs, "LongBench partitions must have distinct output directories.")
        outputs.add(owner)
        require(part["status"] == "completed_same_artifact_LongBench_partition" and part["exit_code"] == 0
            and native["status"] == "partition_complete" and native["exit_code"] == 0
            and part["checkpoint_id"] == config["model"]["checkpoint_id"]
            and native["config_sha256"] == config_sha and read_bound_json(part["config"]) == config
            and native["datasets"] == part["datasets"] and native["input_pins_unchanged_after"]
            and native["already_present"] == 0
            and native["new_predictions"] == sum(EXPECTED_COUNTS[task] for task in part["datasets"]),
            "Every complete task owner must belong to this exact model/config/archive.")
        check_hashes(part["prediction_sha256"])
        for task in part["datasets"]:
            require(task not in sources and task in DATASETS, "No task may have two prediction owners.")
            path = Path(part["output"]) / "predictions" / (task + ".jsonl")
            predicted = [json.loads(line) for line in path.read_text().splitlines()]
            expected = [row for row in rows if row["dataset"] == task]
            require(len(predicted) == len(expected) == EXPECTED_COUNTS[task], "Full21 cannot consume missing or extra rows.")
            for pred, row in zip(predicted, expected, strict=True):
                check_prediction(pred, row, config_sha)
                options = generation_options(tokenizer, task, row["input_tokens"], row["max_new_tokens"], protocol=config["protocol"])
                require(pred["generation_options"] == pred["effective_generation_options"] == options
                    and pred["use_model_defaults"] is False, "The official generation options changed.")
            sources[task] = path
        partitions.append(reference)
    require(set(sources) == set(DATASETS) and sum(EXPECTED_COUNTS[task] for task in sources) == 4750,
        "Require exactly one complete owner for all21 tasks/4750 rows.")
    return config, sources, partitions


def merge(request):
    config = read_bound_json(request["config"])
    check_config(config)
    require(config["mode"] == "full" and config["model"]["sc_cat"]["complete"], "Merge only the complete actual final artifact.")
    config, sources, partitions = partition_sources(request["config"], request["partitions"])
    output = Path(request["output"]).resolve()
    require(not output.exists(), "Do not replace a prior full21 merge.")
    (output / "predictions").mkdir(parents=True)
    shutil.copyfile(request["config"]["path"], output / "config.json")
    for task, path in sources.items():
        shutil.copyfile(path, output / "predictions" / (task + ".jsonl"))
    scores = score_directory(config["archive"]["path"], output, output / "scores.json")
    report = dict(status="complete_same_artifact_LongBench21", exit_code=0, sample_count=4750,
        checkpoint_id=config["model"]["checkpoint_id"], config=request["config"], partitions=partitions,
        scores=binding(output / "scores.json"), task_macro_mean=scores["task_macro_mean"],
        no_model_conversion=True, generation_protocol_unchanged=True)
    write_json(output / "longbench_result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    if request["operation"] == "predict":
        def deadline_exhausted(_signum, _frame):
            raise TimeoutError("The assigned LongBench absolute deadline expired.")
        signal.signal(signal.SIGALRM, deadline_exhausted)
        signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_epoch"] - time.time()))
    torch.set_num_threads(4)
    result = {"prepare": prepare, "predict": predict, "merge": merge}[request["operation"]](request)
    print(json.dumps(dict(status=result["status"], output=request["output"])), flush=True)


if __name__ == "__main__":
    main()

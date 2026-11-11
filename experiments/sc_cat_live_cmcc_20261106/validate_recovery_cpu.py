"""Actual tiny native KD → retained nodes → new export → independent fresh CPU proof."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from transformers import BertTokenizerFast

from experiments.cat_auto_protocol_cmcc_20261104 import data as d
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.native_builder import RecoveryExecutionPolicy, make_native_recovery_trainer
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_auto_protocol_cmcc_20261104.validate_native_builder_cpu import fixture_sources, tiny_root

from .recovery_io import binding, node_receipts, package_pins, read_json, write_json
from .recovery_worker import export, export_zero


def run(directory):
    vocab = directory / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "tiny", "train", "answer"]
        + [f"unused{index}" for index in range(10, 128)]) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    inventory, raw_dir, raw_meta = tiny_root(directory, tokenizer)
    sources = fixture_sources(directory)
    prepared = d.freeze_data_plan(sources, sources[0], tokenizer,
        model_identity=inventory["model_identity"], category_ids=["k_proj"], search_seed=31)
    data_path, inventory_path = directory / "data_plan.pt", directory / "inventory.json"
    torch.save(prepared, data_path)
    write_json(inventory_path, dict(inventory=inventory))
    settings = RecoverySettings()
    raw = dict(round_base_dir=str(raw_dir), round_base_uuid=raw_meta["checkpoint_id"], current_category="k_proj", phase="cpu_fixture")
    policy = RecoveryExecutionPolicy(train_device="cpu", attention_implementation="sdpa",
        teacher_model_offload="none", teacher_output_offload="none", teacher_output_pin_memory=False)
    recovery = make_native_recovery_trainer(round_base_dir=raw_dir, inventory=inventory,
        category_id="k_proj", settings=settings, prepared_data=prepared, sources=sources,
        tokenizer=tokenizer, output_dir=directory / "KD", execution_policy=policy,
        phase="cpu_fixture", technical_stop_after=2)
    try:
        result = recovery.train()
        receipts = node_receipts(recovery, live_raw=raw, trajectory_id="actual_CPU_trajectory",
            recipe_id="actual_CPU_recipe", source_candidate_id="actual_CPU_candidate")
        require([row["step"] for row in receipts] == [1, 2]
            and all(row["schedule_max"] == 500 for row in receipts), "Save1→return→step2 must preserve the real500-step schedule.")
        require(all(row["mutable_manifest"] and row["rng_files"] for row in receipts), "Both real nodes must retain all mutable and RNG manifests.")
        result_path = directory / "recovery_result.json"
        write_json(result_path, dict(status="completed_native_recovery", phase="cpu_fixture",
            settings=asdict(settings), nodes=receipts, live_raw_reference="explicit_CPU_raw", actual_run=result))
    finally:
        recovery.close()
    request = dict(operation="export", phase="cpu_fixture", settings=asdict(settings),
        inventory_receipt=str(inventory_path), data_plan_path=str(data_path),
        recovery_result=binding(result_path), node_step=2, destination=str(directory / "deploy"),
        output=str(directory / "export"), execution_policy=asdict(policy))
    report = export(request)
    require(report["uuid"] not in (raw_meta["checkpoint_id"], receipts[-1]["source_node_id"])
        and report["fresh_evidence"]["separate_process"]
        and report["fresh_evidence"]["pid"] != os.getpid()
        and report["fresh_evidence"]["state_exact"]
        and report["fresh_evidence"]["forward_executed"]
        and not report["gpu_validation"], "Only the actual independent CPU deployment proof may be claimed.")
    meta = read_json(Path(report["path"]) / "checkpoint_meta.json")
    require(meta["completed_categories"] == ["q_proj", "k_proj"]
        and meta["extra_meta"]["distill_stage"]["actual_updates"] == 2,
        "The new deployment must contain actual2-update provenance and advance the real prefix.")
    raw_path = directory / "live_raw.json"
    write_json(raw_path, raw)
    zero_request = {**request, "operation": "export_zero", "live_raw": binding(raw_path),
        "trajectory_id": "actual_CPU_raw0", "recipe_id": "actual_CPU_recipe",
        "source_candidate_id": "actual_CPU_candidate", "destination": str(directory / "zero_deploy"),
        "output": str(directory / "zero_export")}
    zero = export_zero(zero_request)
    zero_meta = read_json(Path(zero["path"]) / "checkpoint_meta.json")
    require(zero["uuid"] not in (raw_meta["checkpoint_id"], report["uuid"])
        and zero_meta["completed_categories"] == ["q_proj", "k_proj"]
        and zero_meta["extra_meta"]["distill_stage"]["actual_updates"] == 0
        and not zero_meta["extra_meta"]["distill_stage"]["did_train"]
        and zero["fresh_forward_evidence"]["finite_logits"],
        "The raw0 worker must use the actual compressed raw, advance progress and independently execute its saved deployment.")
    return dict(status="recovery_worker_actual_CPU_verified", phase="cpu_fixture",
        original_schedule_max=500, actual_updates=result["actual_steps"],
        saved_nodes=[row["step"] for row in receipts],
        actual_save1_return_to_step2=True, all_mutable_optimizer_schedule_RNG_retained=True,
        new_node_export_and_runtime_progress=True, independent_fresh_state_and_forward=True,
        actual_zero_KD_export_and_independent_fresh_forward=True,
        source_pins=package_pins(), formal_eligible=False,
        boundary="Tiny real native CPU mechanism evidence; full-shape GPU save-return/export/fresh is still required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "This is a CPU-only check.")
    output = Path(args.output).resolve()
    require(not output.exists(), "Keep every validation's original receipt.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="recovery_worker_CPU_", dir=output.parent) as directory:
        report = run(Path(directory))
    report.update(elapsed_seconds=time.monotonic() - began, temporary_models_removed=True)
    write_json(output, report)
    print(json.dumps(dict(status=report["status"], elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()

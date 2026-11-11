"""Run one real native KD trajectory, one node export, or an independent fresh check."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import functools
import gc
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.data import batch_digest, tensor_digest
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, build_live_recovery, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.native_builder import RecoveryExecutionPolicy
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds, training_snapshot

from .recovery_io import (binding, node_receipts, package_pins, read_bound_json, read_json,
    selected_from_receipt, validate_request, write_json)


def technical_peak_proof(run, prepared):
    """One separately identified true1024 batch: native KD backward, no third update."""
    trainer = run.trainer
    require(run.provenance["phase"] == "technical_validation" and trainer.technical_stop_after == 2
        and trainer.state.global_step == 2, "Only run the authorized resource proof after both technical updates were saved.")
    manifest = prepared["manifest"]["updates"][run.provenance["category_id"]]
    target_index = next(index for index, row in enumerate(manifest["batches"]) if row["shape"] == [4, 1024])
    loader, observer = trainer.get_train_dataloader(), trainer.input_observer
    iterator = iter(loader)
    began = time.monotonic()
    try:
        for index in range(target_index + 1):
            inputs = next(iterator)
        expected = manifest["batches"][target_index]
        actual_digest = batch_digest(inputs)
        require(list(inputs["input_ids"].shape) == [4, 1024] and actual_digest == expected["input_sha256"],
            "The resource probe must use the exact real1024 batch from the original frozen loader/trace.")
        trainer.model.zero_grad(set_to_none=True)
        before = training_snapshot(trainer)
        trainer.input_observer = None
        loss = trainer.training_step(trainer.model, inputs)
        require(bool(torch.isfinite(loss)), "The original1024 KD forward/backward produced a nonfinite loss.")
        gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.requires_grad and parameter.grad is not None]
        require(gradients and all(bool(torch.isfinite(value).all()) for value in gradients), "The true formal-shape native backward must produce finite actual selected gradients.")
        after = training_snapshot(trainer)
        stable = ("mutable", "optimizer_tensors", "optimizer_scalars", "optimizer_groups", "optimizer_identity", "scheduler")
        require(all(before[key] == after[key] for key in stable) and trainer.state.global_step == 2,
            "The resource-only backward must not apply an optimizer update or alter the two saved update states.")
        return dict(status="actual_formal_shape_resource_only_verified", actual_batch_index=target_index,
            batch_identity=actual_digest, expected_batch=expected, input_shape=[4, 1024],
            original_loss="native_kl_top_mass_top100_T1_prompt.3", loss=float(loss.cpu()),
            gradient_tensors=len(gradients), finite_gradients=True, additional_optimizer_updates=0,
            native_saved_nodes_unchanged=True, formal_trajectory_eligible=False,
            dropout_rng_consumed_only_after_saved_technical_trajectory=True,
            elapsed_seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        trainer.input_observer = observer
        trainer.model.zero_grad(set_to_none=True)
        if hasattr(iterator, "close"):
            iterator.close()
        del iterator, loader
        gc.collect()


def train(request):
    from .live_io import load_inputs

    began = time.monotonic()
    settings = validate_request(request)
    inventory, prepared, sources, tokenizer = load_inputs(request, with_sources=True)
    raw = read_bound_json(request["live_raw"])
    require(raw["phase"] == request["phase"], "The source raw cannot change phase during KD.")
    policy = RecoveryExecutionPolicy(**request["execution_policy"])
    require(policy.train_device == ("cpu" if request["phase"] == "cpu_fixture" else "cuda:0"), "Use the assigned device with the unchanged native execution policy.")
    require(request["phase"] == "cpu_fixture" or policy.attention_implementation == "flash_attention_2",
        "This Qwen live batch keeps the original FlashAttention2 training and deployment policy.")
    update_manifest = prepared["manifest"]["updates"][raw["current_category"]]
    require(update_manifest["schedule_reserved"] == 1000 and len(update_manifest["batches"]) == 1000
        and update_manifest["workers"] == 16, "Keep the actual frozen1000-update trace for both legal schedules.")
    from train_utils import checkpoint_v6 as v6
    raw_meta = v6.load_v6_meta(raw["round_base_dir"])
    require((raw_meta.get("extra_meta") or {}).get("cat_auto_phase") == request["phase"],
        "The actual native raw metadata must preserve technical/CPU/formal provenance.")
    raw_state = torch.load(Path(raw["round_base_dir"]) / raw_meta["state_dict_file"],
        map_location="cpu", weights_only=True, mmap=True)
    source_dtype_counts = dict(Counter(str(value.dtype) for value in raw_state.values()))
    require(all(raw_state[name + ".weight"].dtype == torch.bfloat16 for name in raw_meta["pending_dense_targets"]),
        "Actual own-raw future dense weights must retain the original BF16 source precision.")
    del raw_state
    trajectory_id = request["trajectory_id"]
    recipe_id = request["recipe_id"]
    source_candidate_id = request["source_candidate_id"]
    run = build_live_recovery(live_raw=raw, inventory=inventory, settings=settings,
        prepared_data=prepared, sources=sources, tokenizer=tokenizer, output_dir=request["output"],
        execution_policy=policy, technical_stop_after=request.get("technical_stop_after"))
    try:
        actual_master_dtypes = dict(Counter(str(parameter.dtype) for parameter in run.trainer.model.parameters() if parameter.requires_grad))
        require(set(actual_master_dtypes) == {"torch.float32"}, "All actual selected decoder/LoRA masters must be FP32.")
        result = run.train()
        nodes = node_receipts(run, live_raw=raw, trajectory_id=trajectory_id,
            recipe_id=recipe_id, source_candidate_id=source_candidate_id)
        peak_proof = technical_peak_proof(run, prepared) if request.get("technical_peak_proof") else None
        report = dict(status="completed_native_recovery", exit_code=0, phase=request["phase"],
            trajectory_id=trajectory_id, recipe_id=recipe_id, source_candidate_id=source_candidate_id,
            live_raw_reference=request["live_raw"], live_raw=raw, settings=asdict(settings),
            inventory_receipt=request["inventory_receipt"], data_plan_path=request["data_plan_path"],
            data_identity=prepared["data_identity"], source_pins=package_pins(),
            actual_run=result, nodes=nodes, elapsed_seconds=time.monotonic() - began,
            technical_peak_proof=peak_proof,
            actual_raw_source_dtype_counts=source_dtype_counts,
            actual_trainable_master_dtype_counts=actual_master_dtypes,
            reserved_update_trace=dict(schedule_reserved=1000, batches=1000, workers=16,
                token_order_digest=update_manifest["token_order_digest"]),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            deployments_complete=False, full8_complete=False, exact_resume_verified=False)
        write_json(Path(request["output"]) / "recovery_request.json", request)
        write_json(Path(request["output"]) / "recovery_result.json", report)
        return report
    finally:
        run.close()


def export(request):
    from .boundary import export_selected_node
    from .live_io import load_inputs

    began = time.monotonic()
    validate_request(request)
    result = read_bound_json(request["recovery_result"])
    require(result["status"] == "completed_native_recovery" and result["phase"] == request["phase"], "Export only a completed own trajectory without relabeling its phase.")
    require(result["settings"] == request["settings"], "Export the unchanged actual recovery settings.")
    inventory, _prepared, _sources, tokenizer = load_inputs(request, with_sources=False)
    matches = [row for row in result["nodes"] if row["step"] == request["node_step"]]
    require(len(matches) == 1, "Export exactly one existing retained node.")
    receipt = matches[0]
    selected = selected_from_receipt(receipt, request["phase"])
    current = [row["actual_name"] for row in inventory["projections"] if row["category_id"] == receipt["category_id"]]
    attention = request["execution_policy"]["attention_implementation"]
    report = export_selected_node(node_directory=receipt["node_directory"], destination=request["destination"],
        selected=selected, current_targets=current, phase=request["phase"],
        model_factory=functools.partial(architecture, attn_implementation=attention))
    report.update(source_node=receipt, recovery_result_reference=request["recovery_result"],
        source_raw_reference=result["live_raw_reference"], elapsed_seconds=time.monotonic() - began,
        source_pins=package_pins())
    return _finish_export(request, report, tokenizer, began)


def export_zero(request):
    """Build the real compressed0-update endpoint, without any training object."""
    from .boundary import export_zero_kd_boundary
    from .live_io import load_inputs
    from .selection import SelectedCategory

    began = time.monotonic()
    settings = validate_request(request)
    require(settings == RecoverySettings(), "Raw0 records the unchanged500-step baseline settings without applying any KD updates.")
    inventory, prepared, _sources, tokenizer = load_inputs(request, with_sources=False)
    raw = read_bound_json(request["live_raw"])
    require(raw["phase"] == request["phase"], "The compressed raw cannot change phase during zero-KD export.")
    from train_utils import checkpoint_v6 as v6
    raw_meta = v6.load_v6_meta(raw["round_base_dir"])
    require((raw_meta.get("extra_meta") or {}).get("cat_auto_phase") == request["phase"],
        "The actual compressed raw metadata must keep its original phase during zero-KD export.")
    seeds = recovery_seeds(inventory["model_identity"], raw["current_category"], prepared["manifest"]["search_seed"])
    selected = SelectedCategory(category_id=raw["current_category"], recipe_id=request["recipe_id"],
        source_candidate_id=request["source_candidate_id"], settings=settings, selected_step=0,
        source_trajectory_id=request["trajectory_id"], source_node_id=raw["round_base_uuid"],
        source_round_base_id=raw["round_base_uuid"], seed_identity=seeds, phase=request["phase"])
    current = [row["actual_name"] for row in inventory["projections"] if row["category_id"] == raw["current_category"]]
    report = export_zero_kd_boundary(round_base_dir=raw["round_base_dir"], destination=request["destination"],
        selected=selected, current_targets=current, phase=request["phase"],
        model_factory=functools.partial(architecture,
            attn_implementation=request["execution_policy"]["attention_implementation"]))
    report.update(source_raw_reference=request["live_raw"], source_raw=raw,
        selected_recipe=selected.training_plan(), no_training_objects_constructed=True,
        elapsed_seconds=time.monotonic() - began, source_pins=package_pins())
    return _finish_export(request, report, tokenizer, began)


def _finish_export(request, report, tokenizer, began):
    tokenizer.save_pretrained(request["destination"])
    output = Path(request["output"])
    write_json(output / "deployment_export.json", report)
    gc.collect()
    fresh_request = {**request, "operation": "fresh", "checkpoint_dir": request["destination"],
        "expected_uuid": report["uuid"], "expected_digest": report["digest"],
        "output": str(output / "fresh")}
    fresh_request_path = output / "fresh_request.json"
    write_json(fresh_request_path, fresh_request)
    command = [sys.executable, "-m", "experiments.sc_cat_live_cmcc_20261106.recovery_worker", "--request", str(fresh_request_path)]
    remaining = None if request["phase"] == "cpu_fixture" else max(.01, request["deadline_epoch"] - time.time())
    subprocess.run(command, check=True, timeout=remaining, stdin=subprocess.DEVNULL)
    proof_path = output / "fresh" / "fresh_result.json"
    proof = read_json(proof_path)
    require(proof["checkpoint_id"] == report["uuid"] and proof["weight_digest"] == report["digest"]
        and proof["state_exact"] and proof["forward_executed"], "Independent fresh proof must belong to this exact deployed node.")
    report["fresh_evidence"] = proof
    report["fresh_forward_evidence"] = proof
    report["gpu_validation"] = proof["gpu_validation"]
    report["fresh_proof_reference"] = binding(proof_path)
    report["elapsed_seconds"] = time.monotonic() - began
    write_json(output / "deployment_report.json", report)
    return report


def fresh(request):
    from .live_io import load_inputs

    began = time.monotonic()
    validate_request(request)
    _inventory, prepared, _sources, _tokenizer = load_inputs(request, with_sources=False)
    model_factory = functools.partial(architecture,
        attn_implementation=request["execution_policy"]["attention_implementation"])
    model, proof = fresh_native(request["checkpoint_dir"], model_factory=model_factory)
    require(proof["checkpoint_id"] == request["expected_uuid"] and proof["weight_digest"] == request["expected_digest"], "Fresh reconstruction must use this deployment's whole payload and UUID.")
    device = "cpu" if request["phase"] == "cpu_fixture" else "cuda:0"
    model.to(device).eval().requires_grad_(False)
    from tools.cat_eval import _prepare_model_for_eval
    warmup = _prepare_model_for_eval(model, device, 8, logging.getLogger(__name__), {})
    require(not warmup["failed"] and warmup["warmed"] == warmup["total"],
        "Warm every real compressed module through the full8 deployment consumer before fresh forward.")
    input_ids = prepared["calibration"][0].to(device)
    require(input_ids.ndim == 2 and input_ids.shape[0] == 1 and input_ids.shape[1] > 1, "Exercise a real frozen calibration sequence in the deployment consumer.")
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
    require(bool(torch.isfinite(logits).all()), "The independently reconstructed deployed node produced nonfinite logits.")
    proof.update(status="independent_native_fresh_forward_verified", exit_code=0,
        forward_executed=True, gpu_validation=device.startswith("cuda"), phase=request["phase"],
        separate_process=True, pid=os.getpid(), input_shape=list(input_ids.shape),
        input_digest=tensor_digest({"input_ids": input_ids}), data_identity=prepared["data_identity"],
        logits_shape=list(logits.shape), logits_digest=tensor_digest({"logits": logits}),
        logits_finite=True, finite_logits=True, device=device,
        native_deployment_warmup=warmup,
        attention_implementation=model.config._attn_implementation,
        elapsed_seconds=time.monotonic() - began, source_pins=package_pins())
    write_json(Path(request["output"]) / "fresh_result.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    validate_request(request)
    if request["phase"] != "cpu_fixture":
        def deadline_exhausted(_signum, _frame):
            raise TimeoutError("The shared authorized recovery deadline expired.")
        signal.signal(signal.SIGALRM, deadline_exhausted)
        signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_epoch"] - time.time()))
    torch.set_num_threads(4)
    try:
        report = {"train": train, "export": export, "export_zero": export_zero, "fresh": fresh}[request["operation"]](request)
    except Exception:
        failure = Path(request["output"]) / "recovery_failed.json"
        if not failure.exists():
            write_json(failure, dict(status="failed", exit_code=1,
                operation=request["operation"], request_reference=binding(args.request),
                traceback=traceback.format_exc()))
        raise
    print(json.dumps(dict(status=report.get("status", "node_deployment_complete"),
        operation=request["operation"], output=request["output"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

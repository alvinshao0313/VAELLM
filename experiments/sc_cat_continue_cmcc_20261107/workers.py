"""New-authorization adapters over frozen actual fitter/recovery/export consumers."""
from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import types

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_offline_execution_cmcc_20261104 import evaluate as native_evaluate
from experiments.cat_offline_execution_cmcc_20261104.protocol import load_baseline
from experiments.sc_cat_live_cmcc_20261106 import fit_worker as old_fit
from experiments.sc_cat_live_cmcc_20261106 import recovery_worker as old_recovery
from experiments.sc_cat_live_cmcc_20261106 import evaluate as old_evaluate
from experiments.sc_cat_live_cmcc_20261106 import live_io
from experiments.sc_cat_live_cmcc_20261106 import recovery_io
from experiments.sc_cat_pipeline_cmcc_20261106 import node_export as old_node
from train_utils import checkpoint_v6 as v6
from . import bindings as io


def _function(function, **replacements):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **replacements), function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _package_pins():
    return io.runtime_pins(live_io.source_manifest(_active_request))


_active_request = None


def _source_fields(request, authorization):
    require(request["stage"] in ("core", "replay")
        and request["source_manifest"] == authorization["source_manifest"],
        "Fit/recovery must retain the actual new approval's frozen source and stage.")
    manifest = live_io.source_manifest(request)
    require(io.bound_path(request["inventory_receipt"]) == io.bound_path(manifest["inventory"])
        and io.bound_path(request["data_plan_path"]) == io.bound_path(manifest["data_plan"]),
        "The worker must use the frozen actual inventory and DataPlan.")
    if request.get("destination"):
        require(Path(request["destination"]).resolve().is_relative_to(Path(authorization["output_root"]).resolve()),
            "Deployment output must remain inside the actual new authorized batch root.")
    return manifest


def _deadline(request, authorization, kind, *, category=None, settings=None):
    limit = io.seconds(authorization, kind, category=category, settings=settings)
    require(0 < request["deadline_epoch"] - time.time() <= limit,
        "The actual worker deadline exceeds its approved category/operation window.")


def _category(request, category):
    allowed = io.CATEGORIES if request["stage"] == "core" else io.ORDER
    require(category in allowed and request.get("category_id", category) == category,
        "This operation must preserve its actual source category and approved stage.")


def _raw_provenance(request, authorization, raw):
    _category(request, raw["current_category"])
    producer = io.read(io.bound_path(raw["request"]))
    require(raw["phase"] == "formal" and raw["formal_eligible"]
        and producer.get("phase") == "formal" and producer.get("stage") == request["stage"]
        and io.bound_path(producer["authorization"]) == io.bound_path(request["authorization"])
        and producer["source_manifest"] == authorization["source_manifest"]
        and producer["category_id"] == raw["current_category"]
        and producer["recipe"] == raw["recipe"],
        "Recovery/replay must start from its own genuinely new-authorized raw, not relabel an old candidate.")


def _fresh_source(request):
    report = io.read(io.bound_path(request["source_export_report"]))
    kind = "raw_export" if report["export_evidence"]["actual_updates"] == 0 else "node_export"
    require(report["phase"] == request["phase"] == "formal"
        and report["uuid"] == request["expected_uuid"] and report["digest"] == request["expected_digest"]
        and Path(report["path"]).resolve() == Path(request["checkpoint_dir"]).resolve(),
        "The independent fresh request must retain the exact actual export artifact and phase.")
    return report, kind


def _validate_fit(request):
    auth = io.validate_worker_request(request, kind="fit")
    _source_fields(request, auth)
    _category(request, request["category_id"])
    _deadline(request, auth, "fit", category=request["category_id"])
    if request["stage"] == "replay":
        from .replay import read_manifest, stage_plan
        selected = stage_plan(read_manifest(request["replay_manifest"]), request["category_id"])
        require(request["recipe"] == selected["resolved_recipe"]
            and request["candidate_id"] == selected["source_candidate_id"],
            "Replay fits only the previously frozen actual recipe, without another anchor/challenger choice.")
    elif request["category_id"] == io.CATEGORIES[0]:
        parent = live_io.actual_parent(request)
        previous = io.read(io.bound_path(auth["prefix"]["after_o"]))
        require(parent["uuid"] == previous["uuid"] and parent["digest"] == previous["digest"],
            "The first continuation fit must start from the authorized unique measured after_o artifact.")
    require(request["phase"] == "formal" and not request.get("technical_steps")
        and request["execution_policy"]["attention_implementation"] == "sdpa"
        and request["category_id"] in (io.CATEGORIES if request["stage"] == "core" else io.ORDER)
        and request["recipe"]["category_id"] == request["category_id"]
        and request["source_manifest"] == auth["source_manifest"]
        and os.environ.get("CONDA_DEFAULT_ENV") == "bitvae"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == str(request["gpu"])
        and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "Keep the actual approved own-parent fit/data/environment/device.")
    return auth


def _validate_recovery(request):
    require(request["operation"] in ("train", "export", "export_zero", "fresh"), "Execute one concrete real native operation.")
    operation = request["operation"]
    kind = "train" if operation == "train" else "raw_export" if operation == "export_zero" else "node_export"
    if operation == "fresh":
        report, kind = _fresh_source(request)
        raw_reference = report["source_raw_reference"]
    elif operation in ("train", "export_zero") or operation == "export" and "recovery_result" not in request:
        # The published-node exporter retains the actual original training
        # request while the full trajectory is still running.
        raw_reference = request["live_raw"]
    else:
        result = io.read(io.bound_path(request["recovery_result"]))
        require(result["phase"] == request["phase"] and result["settings"] == request["settings"],
            "Export the actual unchanged completed recovery settings and phase.")
        raw_reference = result["live_raw_reference"]
    if request["stage"] == "replay" and operation != "train":
        kind = "replay_export"
    raw = io.read(io.bound_path(raw_reference))
    actual_request = dict(request, category_id=raw["current_category"])
    auth = io.validate_worker_request(actual_request, kind=kind)
    _source_fields(request, auth)
    _raw_provenance(request, auth, raw)
    require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == str(request["gpu"])
        and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
        and request.get("technical_stop_after") is None and not request.get("technical_peak_proof"), "Formal continuation preserves actual device/policy and cannot carry technical restrictions.")
    settings = RecoverySettings(**request["settings"])
    require(request["execution_policy"]["attention_implementation"] == "flash_attention_2",
        "Keep the actual native FA2 recovery and deployment policy.")
    _deadline(request, auth, kind, category=raw["current_category"], settings=settings)
    if request["operation"] == "train":
        require(not Path(request["output"]).exists(), "Do not overwrite or resume an existing trajectory.")
    return settings


def execute_fit(request, *, request_path):
    global _active_request
    _active_request = request
    proxy = types.SimpleNamespace(**dict(vars(live_io), validate_execution=_validate_fit))
    return _function(old_fit.execute, io=proxy)(request, request_path=request_path)


def make_fresh_request(request, report):
    output = Path(request["output"])
    return dict(request, operation="fresh", checkpoint_dir=request["destination"],
        expected_uuid=report["uuid"], expected_digest=report["digest"], output=str(output / "fresh"),
        source_export_report=recovery_io.binding(output / "deployment_export.json"),
        fresh_origin_kind="raw_export" if report["export_evidence"]["actual_updates"] == 0 else "node_export")


def finish_export(request, report, tokenizer, began):
    """Preserve actual independent fresh proof, with the real new approval boundary."""
    tokenizer.save_pretrained(request["destination"])
    output = Path(request["output"])
    recovery_io.write_json(output / "deployment_export.json", report)
    gc.collect()
    fresh_request = make_fresh_request(request, report)
    path = output / "fresh_request.json"
    recovery_io.write_json(path, fresh_request)
    command = [sys.executable, "-B", "-u", "-m", "experiments.sc_cat_continue_cmcc_20261107.workers", "--kind", "recovery_worker", "--request", str(path)]
    subprocess.run(command, check=True, timeout=max(.01, request["deadline_epoch"] - time.time()), stdin=subprocess.DEVNULL)
    proof_path = output / "fresh" / "fresh_result.json"
    proof = io.read(proof_path)
    require(proof["checkpoint_id"] == report["uuid"] and proof["weight_digest"] == report["digest"]
        and proof["phase"] == request["phase"] and proof["exit_code"] == 0
        and proof["state_exact"] and proof["forward_executed"] and proof["gpu_validation"]
        and proof["finite_logits"], "Fresh proof must belong to this actual formal deployed node.")
    report.update(fresh_evidence=proof, fresh_forward_evidence=proof, gpu_validation=proof["gpu_validation"],
        fresh_proof_reference=recovery_io.binding(proof_path), elapsed_seconds=time.monotonic() - began)
    recovery_io.write_json(output / "deployment_report.json", report)
    return report


def recovery(request):
    global _active_request
    _active_request = request
    operation = request["operation"]
    return _function(getattr(old_recovery, operation), validate_request=_validate_recovery,
        package_pins=_package_pins, _finish_export=finish_export)(request)


def node_export(request):
    global _active_request
    training = recovery_io.read_bound_json(request["training_request"])
    require(io.bound_path(request["authorization"]) == io.bound_path(training["authorization"])
        and request.get("stage", training["stage"]) == training["stage"],
        "A published node retains its actual training authorization and stage.")
    _active_request = training
    return _function(old_node.export, validate_request=_validate_recovery,
        package_pins=_package_pins, _finish_export=finish_export)(request)


def full8(request):
    """Use the original complete evaluator, changing only the actual approval source."""
    auth = io.validate_worker_request(request, kind="full8")
    manifest = _source_fields(request, auth)
    require(io.bound_path(request["baseline"]) == io.bound_path(manifest["baseline"]),
        "The complete evaluator must retain the frozen actual BF16 protocol reference.")
    require(request["deadline_epoch"] - time.time() <= io.seconds(auth, "full8")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "The native full8 outer controller keeps CUDA hidden and its original900second window.")
    report_path, baseline_path = io.bound_path(request["deployment_report"]), io.bound_path(request["baseline"])
    report, baseline = io.read(report_path), load_baseline(baseline_path)
    directory = Path(report["path"]).resolve()
    meta = v6.load_v6_meta(str(directory))
    proof, source = report["fresh_forward_evidence"], meta["extra_meta"]["cat_auto_selected_node"]
    require(report["phase"] == "formal" and meta["checkpoint_kind"] == "category_boundary"
        and proof["checkpoint_id"] == report["uuid"] == meta["checkpoint_id"]
        and proof["weight_digest"] == report["digest"] and proof["exit_code"] == 0
        and proof["state_exact"] and proof["forward_executed"] and proof["finite_logits"] and proof["gpu_validation"]
        and str(proof["device"]).startswith("cuda:") and io.read(io.bound_path(report["fresh_proof_reference"])) == proof
        and source == report["export_evidence"] and source["phase"] == "formal", "Evaluate this exact formal deployment, source and independent GPU fresh proof.")
    raw = io.read(io.bound_path(report["source_raw_reference"]))
    _raw_provenance(request, auth, raw)
    require(raw["phase"] == "formal" and raw["formal_eligible"] and raw["run_identity"]["model"] == baseline["model_identity"]
        and raw["round_base_uuid"] == source["round_base_id"], "The deployment must retain its actual model and own formal raw.")
    endpoint = dict(kind="native_v6", path=str(directory), checkpoint_id=report["uuid"], model_identity=baseline["model_identity"],
        metadata_sha256=io.binding(directory / "checkpoint_meta.json")["sha256"],
        state_sha256=io.binding(directory / meta["state_dict_file"])["sha256"], phase="formal", formal_eligible=True)
    require(endpoint["state_sha256"] == report["digest"], "The actual deployed payload changed.")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    allocation = dict(authorization_reference=auth["authorization_reference"], physical_GPU=request["gpu"],
        authorized_GPU_hours=.25, authorized_start_epoch=started, absolute_deadline_epoch=request["deadline_epoch"])
    plan = native_evaluate.make_plan(result=dict(endpoint=endpoint, phase="formal", formal_eligible=True),
        baseline=baseline, baseline_path=baseline_path, output=output, authorization=allocation)
    plan["live_source"] = dict(deployment_report=io.binding(report_path), source_node_id=source["source_node_id"],
        source_round_base_id=source["round_base_id"], qualification="Actual deployment/fresh proof; task quality still requires full evaluator completion.")
    plan_path = output / "plan.json"
    io.write(plan_path, plan)
    native_evaluate.run(plan_path)
    return old_evaluate.collect(plan_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("fit_worker", "recovery_worker", "node_export", "evaluate"), required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    request = io.read(request_path)
    def expired(_signum, _frame):
        raise TimeoutError("The actual authorized finite worker window expired.")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_epoch"] - time.time()))
    try:
        if args.kind == "fit_worker":
            result = execute_fit(request, request_path=request_path)
        else:
            result = {"recovery_worker": recovery, "node_export": node_export, "evaluate": full8}[args.kind](request)
    except BaseException:
        failure_trace = traceback.format_exc()
        try:
            authorization = io.read_authorization(request["authorization"])
            output = Path(request["output"]).resolve()
            allowed = output.is_relative_to(Path(authorization["output_root"]).resolve())
        except Exception:
            allowed = False
        if allowed:
            output.mkdir(parents=True, exist_ok=True)
            failure = output / "worker_failed.json"
            if not failure.exists():
                io.write(failure, dict(status="failed", exit_code=1, kind=args.kind, request=io.binding(request_path), traceback=failure_trace))
        raise
    print(dict(status="completed_actual_worker", kind=args.kind, output=request["output"]), flush=True)


if __name__ == "__main__":
    main()

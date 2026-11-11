"""Read atomic native node commits and export them without claiming trajectory completion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import functools
import json
from pathlib import Path
import signal
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture
from experiments.cat_auto_protocol_cmcc_20261104.native_builder import RecoveryExecutionPolicy
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106 import live_io
from experiments.sc_cat_live_cmcc_20261106.boundary import export_selected_node
from experiments.sc_cat_live_cmcc_20261106.recovery_io import (
    binding, package_pins, read_bound_json, read_json, selected_from_receipt, validate_request,
)
from experiments.sc_cat_live_cmcc_20261106.recovery_worker import _finish_export
from train_utils import checkpoint_v6 as v6


def read_committed_node(training_request_ref, node_step):
    """Return the final node-receipt schema; only an unpublished commit returns None."""
    require(isinstance(training_request_ref, dict) and training_request_ref.get("sha256"),
        "Bind the actual original training request bytes before reading a published node.")
    request = read_bound_json(training_request_ref)
    require(request["operation"] == "train", "A published node belongs to an actual native training request.")
    settings = RecoverySettings(**request["settings"])
    technical = request.get("technical_stop_after")
    legal = (1, 2) if technical == 2 else settings.node_steps
    require(request["phase"] in ("formal", "technical_validation", "cpu_fixture")
        and technical in (None, 2) and (request["phase"] != "formal" or technical is None)
        and type(node_step) is int and node_step in legal, "Read an actual node of this unchanged requested schedule and phase.")
    directory = (Path(request["output"]) / "trainer_state" / f"checkpoint-{node_step}").resolve()
    meta_path = directory / "checkpoint_meta.json"
    if not meta_path.is_file():
        return None
    files = ("checkpoint_meta.json", "training_model_state.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    rng_files = sorted(directory.glob("rng_state*.pth"))
    require(all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in files)
        and rng_files and all(path.stat().st_size > 0 for path in rng_files),
        "A published native checkpoint_meta commit must contain every mutable/optimizer/scheduler/RNG/trainer file.")
    meta_binding = binding(meta_path)
    meta = v6.load_v6_training_step_meta(str(directory))
    state = read_json(directory / "trainer_state.json")
    raw_directory, raw_meta = v6.resolve_training_step_round_base_ref(str(directory), meta)
    raw = read_bound_json(request["live_raw"])
    require(Path(raw_directory).resolve() == Path(raw["round_base_dir"]).resolve()
        and raw_meta["checkpoint_id"] == raw["round_base_uuid"]
        and raw_meta["checkpoint_kind"] == "round_base"
        and raw["phase"] == request["phase"] == (raw_meta.get("extra_meta") or {}).get("cat_auto_phase"),
        "The published node must retain this training request's actual own raw and phase.")
    require(state["global_step"] == node_step and state["max_steps"] == settings.schedule_max,
        "The committed trainer state changed its actual node or original schedule maximum.")
    inventory = read_json(live_io.bound_path(request["inventory_receipt"]))["inventory"]
    prepared = torch.load(live_io.bound_path(request["data_plan_path"]), map_location="cpu", weights_only=False)
    require(digest(prepared["manifest"]) == prepared["data_identity"]
        and prepared["manifest"]["model_identity"] == inventory["model_identity"], "Preserve the frozen original data/model identity.")
    category = raw["current_category"]
    seeds = recovery_seeds(inventory["model_identity"], category, prepared["manifest"]["search_seed"])
    audit = meta["runtime_audit"]["cat_auto_recovery"]
    builder = meta["runtime_audit"]["cat_auto_native_builder"]
    require(audit["semantic_seeds"] == seeds == builder["semantic_seeds"]
        and audit["schedule_max"] == settings.schedule_max and audit["selected_scope"] == settings.scope
        and audit["technical_stop_after"] == technical and audit["formal_eligible"] == (technical is None)
        and builder["phase"] == request["phase"] and builder["settings"] == asdict(settings)
        and builder["execution_policy"] == asdict(RecoveryExecutionPolicy(**request["execution_policy"]))
        and builder["round_base_id"] == raw_meta["checkpoint_id"] and builder["category_id"] == category
        and builder["root_data_identity"] == prepared["data_identity"], "The published scope/seed/LR/schedule/runtime differs from the bound training request.")
    contract = meta["immutable_resume_contract"]
    opt, data = contract["optimization"], contract["data"]
    require(contract["round_base_checkpoint_id"] == raw_meta["checkpoint_id"]
        and contract["active_category"] == category and contract["after_category_mode"] == settings.scope
        and opt["steps"] == settings.schedule_max and opt["decoder_lr"] == settings.decoder_lr
        and opt["learning_rate"] == settings.lora_lr and opt["batch_size"] == 4
        and opt["gradient_accumulation_steps"] == 1 and opt["warmup_ratio"] == .01
        and opt["lr_scheduler_type"] == "cosine" and opt["optim"] == "adamw_torch"
        and data["model_max_length"] == 1024 and data["dynamic_padding"]
        and data["data_seed"] == data["seed"] == seeds["data"]
        and contract["dataset_identity"]["root_data_identity"] == prepared["data_identity"]
        and contract["dataset_identity"]["update_manifest"] == prepared["manifest"]["updates"][category],
        "The committed native immutable contract changed actual data, optimizer, LR or original schedule.")
    current = {row["actual_name"] for row in inventory["projections"] if row["category_id"] == category}
    decoders = set(raw_meta["compressed_targets"]) if settings.scope == "remaining_lora_prefix_decoder" else current
    future = set(raw_meta["pending_dense_targets"]) if settings.scope.startswith("remaining_lora_") else set()
    lora = contract["lora"]
    require(all(meta[key] == raw_meta[key] == contract[key] for key in
        ("compressed_targets", "pending_dense_targets", "completed_categories", "compression_categories", "target_layers", "target_modules"))
        and meta["lora_config"] == lora
        and (lora is None if not future else lora["rank"] == settings.lora_rank
            and lora["alpha"] == settings.lora_alpha and lora["dropout"] == settings.lora_dropout
            and set(lora["target_modules"]) == future), "The committed prefix/future inventory or actual LoRA topology differs from its own raw and request.")
    require(set(contract["decoder_target_names"]) == decoders and set(contract["lora_target_names"]) == future
        and contract["precision"]["bf16"] and not contract["precision"]["fp16"]
        and binding(meta_path) == meta_binding, "Actual committed decoder/LoRA owners, precision or metadata changed.")
    return dict(step=node_step, schedule_max=settings.schedule_max, node_directory=str(directory),
        source_node_id=meta["checkpoint_id"], source_round_base_id=raw_meta["checkpoint_id"],
        source_round_base=str(Path(raw_directory).resolve()), trajectory_id=request["trajectory_id"],
        recipe_id=request["recipe_id"], source_candidate_id=request["source_candidate_id"], category_id=category,
        settings=asdict(settings), semantic_seeds=seeds, checkpoint_meta=meta_binding,
        native_files={name: dict(path=str((directory / name).resolve()), bytes=(directory / name).stat().st_size) for name in files},
        rng_files=[dict(path=str(path.resolve()), bytes=path.stat().st_size) for path in rng_files],
        mutable_manifest=meta[v6.MUTABLE_STATE_MANIFEST_KEY], exact_resume_verified=False)


def export(request):
    began = time.monotonic()
    training = read_bound_json(request["training_request"])
    common = dict(training, operation="export", output=request["output"], gpu=request["gpu"],
        deadline_epoch=request["deadline_epoch"], authorization=request["authorization"], destination=request["destination"])
    validate_request(common)
    receipt = read_committed_node(request["training_request"], request["node_step"])
    require(receipt is not None, "The requested native checkpoint commit has not been published yet.")
    inventory, _prepared, _sources, tokenizer = live_io.load_inputs(common, with_sources=False)
    selected = selected_from_receipt(receipt, training["phase"])
    current = [row["actual_name"] for row in inventory["projections"] if row["category_id"] == receipt["category_id"]]
    report = export_selected_node(node_directory=receipt["node_directory"], destination=common["destination"],
        selected=selected, current_targets=current, phase=training["phase"],
        model_factory=functools.partial(architecture,
            attn_implementation=training["execution_policy"]["attention_implementation"]))
    report.update(source_node=receipt, training_request_reference=request["training_request"],
        source_raw_reference=training["live_raw"], trajectory_completion_required_for_selection=True,
        source_pins={**package_pins(), str(Path(__file__).resolve()): binding(__file__)["sha256"]},
        elapsed_seconds=time.monotonic() - began)
    return _finish_export(common, report, tokenizer, began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    def deadline_exhausted(_signum, _frame):
        raise TimeoutError("The assigned published-node export window expired.")
    signal.signal(signal.SIGALRM, deadline_exhausted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_epoch"] - time.time()))
    torch.set_num_threads(4)
    report = export(request)
    print(json.dumps(dict(status="committed_node_export_fresh_complete", uuid=report["uuid"],
        output=request["output"])), flush=True)


if __name__ == "__main__":
    main()

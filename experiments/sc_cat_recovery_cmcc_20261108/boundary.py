"""Real native node/raw0 finalization and one-prefix SC-CAT boundary commits."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import torch

from e2e_common.full_lora import collect_exact_peft_lora_config, finalize_model_level_lora, iter_named_peft_lora_layers
from litebsq.vae_linear import clear_model_vae_linear_cache
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
from train_utils.cat_train_runtime import load_cat_resume_distill_progress
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.distill_precision import prepare_model_export

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, inventory_id, require, validate_boundary
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native, _sha
from .identity import source_pins
from .mutable import restore_mutable, select_native_scope
from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
from .selection import SelectedCategory


def _pins():
    return dict(source_pins(), **{str(path): _sha(path) for path in
        (Path(__file__).resolve(), Path(__file__).with_name("selection.py").resolve())})


def _validate_raw(raw, selected, current_targets, phase, model_factory):
    require(phase in ("formal", "technical_validation", "pilot", "cpu_fixture"), "Declare the actual export phase.")
    require(phase == "cpu_fixture" or model_factory is not architecture,
        "Real model exports need the explicit original attention/runtime construction policy.")
    require(phase != "formal" or selected.phase == "formal", "Technical nodes cannot acquire formal provenance.")
    raw_phase = (raw.get("extra_meta") or {}).get("cat_auto_phase")
    require(raw_phase != "cpu_fixture" or phase == "cpu_fixture", "CPU raw state cannot acquire model-scale provenance.")
    order, completed = raw["compression_categories"], raw["completed_categories"]
    require(raw["checkpoint_kind"] == "round_base" and raw["checkpoint_id"] == selected.source_round_base_id
        and completed == order[:len(completed)] and len(completed) < len(order)
        and order[len(completed)] == selected.category_id,
        "Use this selected raw and the next category of its own complete prefix.")
    actual_current = {name for name in raw["compressed_targets"] if name.rsplit(".", 1)[-1] == selected.category_id}
    require(actual_current and set(current_targets) == actual_current and len(current_targets) == len(actual_current),
        "Current decoder owners must equal the actual compressed current category.")
    require(len(completed) + 1 == len(order) or bool(raw["pending_dense_targets"]),
        "Future dense targets may be empty only at the last category.")


def _save_boundary(*, model, raw, raw_directory, destination, selected, phase,
        model_factory, info, manifest, stage, resolved_learning_rates):
    destination = Path(destination).resolve()
    require(not destination.exists(), "Do not replace an existing candidate boundary.")
    source_runtime = v6.load_v6_cat_runtime_state(str(raw_directory), required=True, map_location="cpu")
    require(not source_runtime["runtime_identity"].get("cat_shared", {}).get("channel_refresh_after_category", False),
        "Channel-refresh runtime needs its actual adapter before boundary export.")
    completed = [*raw["completed_categories"], selected.category_id]
    state_digest = tensor_digest(model.state_dict())
    runtime_identity = dict(schema="sc_cat_finalized_boundary_runtime_v1", phase=phase,
        source_runtime_identity=source_runtime["runtime_identity"],
        source_runtime_identity_digest=digest(source_runtime["runtime_identity"]),
        compression_categories=raw["compression_categories"], completed_prefix=completed,
        compressed_targets=raw["compressed_targets"], pending_dense_targets=raw["pending_dense_targets"],
        active_category=None, stage="after_category", actual_updates=selected.selected_step,
        source_node_id=selected.source_node_id, round_base_id=raw["checkpoint_id"],
        resolved_recipe_id=selected.recipe_id, state_digest=state_digest,
        activation_statistics_policy="resample_next_live_parent", training_exact_resume=False)
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None,
        runtime_identity=runtime_identity)
    info.update(source_pins=_pins(), phase=phase, runtime_identity=runtime_identity,
        exported_state_digest=state_digest, training_exact_resume=False, gpu_validation=False)
    extra = deepcopy(raw.get("extra_meta") or {})
    history = deepcopy(extra.get("distill_stage_history") or [])
    history.append(stage)
    extra.update(cat_auto_selected_node=info, cat_auto_phase=phase, distill_stage=stage,
        distill_stage_history=history, stage="after_category", category=selected.category_id,
        active_category=None, round_idx=len(completed))
    saved = v6.save_v6_full_checkpoint(model, str(destination), checkpoint_kind="category_boundary",
        compressed_targets=raw["compressed_targets"], pending_dense_targets=raw["pending_dense_targets"],
        skip_targets=raw["skip_targets"], legacy_original_only_sources=raw["legacy_original_only_sources"],
        completed_categories=completed, compression_categories=raw["compression_categories"],
        target_layers=raw["target_layers"], target_modules=raw["target_modules"], train_mode="none",
        after_category_mode="none" if selected.selected_step == 0 else selected.settings.scope,
        lora_config=None, resolved_learning_rates=resolved_learning_rates,
        base_model_path=raw["base_model_path"], finalized_status=dict(lora_finalized=True,
            decoder_finalized=True, aux_finalized=True, stable_category_boundary=True, inference_ready=False),
        cat_runtime_state=runtime, extra_meta=extra)
    meta = v6.load_v6_meta(str(destination))
    require(saved["checkpoint_id"] == meta["checkpoint_id"]
        and meta["checkpoint_id"] not in (raw["checkpoint_id"], selected.source_node_id),
        "A real finalized boundary requires its own new native UUID.")
    progress = load_cat_resume_distill_progress(str(destination))
    require(list(progress.completed_categories) == completed and progress.active_category is None
        and progress.distill_stage_history[-1]["actual_updates"] == selected.selected_step
        and progress.distill_stage_history[-1]["did_train"] == (selected.selected_step > 0),
        "Actual saved CAT progress must advance the category and preserve zero/nonzero KD history.")
    restored_runtime = v6.load_v6_cat_runtime_state(str(destination), required=True, map_location="cpu")
    require(restored_runtime["runtime_identity"] == runtime_identity
        and restored_runtime["activation_runtime"] is None and restored_runtime["global_adaptive_plan"] is None,
        "Save the rebuilt finalized runtime, rather than retaining the raw cache identity.")
    _fresh, fresh = fresh_native(destination, model_factory=model_factory)
    require(tensor_digest(_fresh.state_dict()) == state_digest,
        "The independent fresh model must match the actual finalized exported state.")
    return dict(uuid=meta["checkpoint_id"], path=str(destination), digest=fresh["weight_digest"],
        completed_prefix=completed, mutable_manifest=list(manifest), export_evidence=info,
        fresh_evidence=fresh, formal_eligible=False, phase=phase, runtime_identity=runtime_identity,
        ledger_status="required_before_formal_chain_commit", gpu_validation=False,
        progress_evidence=dict(completed_prefix=completed, active_category=progress.active_category,
            actual_updates=selected.selected_step, lora_round_idx=progress.lora_round_idx),
        limitation="Native finalization/runtime/progress/fresh-state evidence; model-scale forward and full8 remain separate.")


def export_zero_kd_boundary(*, round_base_dir, destination, selected: SelectedCategory,
        current_targets: Sequence[str], phase: str, model_factory: Callable = architecture) -> dict:
    """Finalize the actual compressed raw without constructing training objects."""
    require(selected.selected_step == 0 and selected.source_node_id == selected.source_round_base_id,
        "Zero-KD export requires the actual selected raw UUID and zero updates.")
    require(not Path(destination).exists(), "Do not replace an existing candidate boundary.")
    raw_directory = Path(round_base_dir).resolve()
    raw = v6.load_v6_meta(str(raw_directory))
    _validate_raw(raw, selected, current_targets, phase, model_factory)
    model, actual_raw, loaded = v6.load_v6_full_checkpoint_into_model(model_factory(raw_directory),
        str(raw_directory), expected_kind="round_base", strict=True)
    require(actual_raw["checkpoint_id"] == raw["checkpoint_id"] and not loaded.missing_keys and not loaded.unexpected_keys,
        "Reconstruct the complete actual compressed raw, never a training node or donor.")
    require(not list(iter_named_peft_lora_layers(model)), "A finalized raw must not contain live future-LoRA adapters.")
    before = tensor_digest(model.state_dict())
    targets = tuple(NamedMainDecoderTarget(name, model.get_submodule(name)) for name in raw["compressed_targets"])
    finalized = finalize_main_decoder_targets(targets)
    model.eval().requires_grad_(False)
    clear_model_vae_linear_cache(model)
    require(not list(iter_named_peft_lora_layers(model)) and all(not target.base_layer.trainable_decode for target in targets)
        and tensor_digest(model.state_dict()) == before,
        "Zero-KD finalization must preserve every actual raw tensor and remove runtime trainability.")
    info = dict(source_kind="raw_zero", zero_kd=True, source_node_id=raw["checkpoint_id"],
        round_base_id=raw["checkpoint_id"], source_round_base=str(raw_directory), actual_updates=0,
        schedule_max=selected.settings.schedule_max, source_trajectory_id=selected.source_trajectory_id,
        source_candidate_id=selected.source_candidate_id, recipe_id=selected.recipe_id,
        all_mutable_restored=False, raw_state_exact=True, raw_state_digest=before,
        optimizer_constructed=False, trainer_constructed=False, native_decoders_finalized=finalized,
        future_dense_lora_merged=0)
    stage = dict(mode="none", category=selected.category_id, did_train=False, actual_updates=0,
        planned_steps=selected.settings.schedule_max, source_raw_checkpoint_id=raw["checkpoint_id"],
        round_base_checkpoint_id=raw["checkpoint_id"], decoder_targets=[], remaining_lora_targets=[])
    return _save_boundary(model=model, raw=raw, raw_directory=raw_directory, destination=destination,
        selected=selected, phase=phase, model_factory=model_factory, info=info, manifest=[],
        stage=stage, resolved_learning_rates=None)


def export_selected_node(*, node_directory, destination, selected: SelectedCategory,
        current_targets: Sequence[str], phase: str, model_factory: Callable = architecture) -> dict:
    """Restore all actual masters from the node's full raw, then finalize once."""
    require(selected.selected_step > 0, "Use export_zero_kd_boundary for the actual raw0 path.")
    require(not Path(destination).exists(), "Do not replace an existing candidate boundary.")
    node_directory = Path(node_directory).resolve()
    node = v6.load_v6_training_step_meta(str(node_directory))
    raw_directory, raw = v6.resolve_training_step_round_base_ref(str(node_directory), node)
    raw_directory = Path(raw_directory)
    _validate_raw(raw, selected, current_targets, phase, model_factory)
    require(node["checkpoint_id"] == selected.source_node_id, "Use the selected actual training node UUID.")
    v6.validate_training_step_round_base(node, raw)
    contract = node["immutable_resume_contract"]
    import json
    state = json.loads((node_directory / "trainer_state.json").read_text(encoding="utf-8"))
    require(state["global_step"] == selected.selected_step and state["max_steps"] == selected.settings.schedule_max
        and contract["optimization"]["steps"] == selected.settings.schedule_max,
        "Preserve the real selected step and original scheduler time domain.")
    require(contract["round_base_checkpoint_id"] == raw["checkpoint_id"]
        and contract["active_category"] == selected.category_id and contract["after_category_mode"] == selected.settings.scope,
        "The immutable node contract must match its raw/category/scope.")
    audit = (node.get("runtime_audit") or {}).get("cat_auto_recovery")
    require(audit is not None and audit["semantic_seeds"] == dict(selected.seed_identity)
        and audit["schedule_max"] == selected.settings.schedule_max
        and (phase != "formal" or audit.get("formal_eligible")),
        "Bind actual semantic seeds and formal/technical provenance.")
    require(contract["sc_cat_recovery_settings"] == asdict(selected.settings), "Saved D/P/L settings differ.")
    optimization = contract["optimization"]
    decoder_lr = optimization["learning_rate"] if optimization.get("decoder_lr") is None else optimization["decoder_lr"]
    require(decoder_lr == selected.settings.decoder_lr, "The selected decoder LR differs from its actual source.")
    precision, aux = SimpleNamespace(**contract["precision"]), AuxTrainableConfig(**contract["aux"])
    require(precision.bf16 and not precision.fp16 and aux.norm_train_mode == "none"
        and aux.lm_head_train_mode == "none" and aux.residual_lora_mode == "none", "Keep the confirmed BF16 core export recipe.")
    model, actual_raw, loaded = v6.load_v6_full_checkpoint_into_model(model_factory(raw_directory),
        str(raw_directory), expected_kind="round_base", strict=True)
    require(actual_raw["checkpoint_id"] == raw["checkpoint_id"] and not loaded.missing_keys and not loaded.unexpected_keys,
        "Reconstruct the complete actual native raw.")
    original = {name: tensor_digest({name: value}) for name, value in model.state_dict().items()}
    use_cache = getattr(model.config, "use_cache", None)
    selection = select_native_scope(model, settings=selected.settings, prefix_targets=raw["compressed_targets"],
        current_targets=current_targets, future_dense_targets=raw["pending_dense_targets"], precision=precision,
        lora_init_seed=selected.seed_identity["lora_init"], aux=aux,
        fp32_components=optimization.get("distill_fp32_components", ()))
    require(list(contract["decoder_target_names"]) == [name for name, _layer in selection.decoder_targets]
        and list(contract["lora_target_names"]) == list(selection.dense_targets), "Restore the exact saved decoder/LoRA owner inventory.")
    lora = None if not selection.dense_targets else collect_exact_peft_lora_config(selection.model,
        default_rank=selected.settings.lora_rank, alpha=selected.settings.lora_alpha, dropout=selected.settings.lora_dropout)
    require(lora == contract["lora"] == node["lora_config"], "Restore the exact saved actual LoRA topology.")
    values, manifest = v6.load_v6_training_model_state(str(node_directory), map_location="cpu")
    restoration = restore_mutable(selection, values, manifest)
    protection_changed = selection.protection.finalize()
    finalized = finalize_main_decoder_targets(tuple(NamedMainDecoderTarget(name, layer) for name, layer in selection.execution_targets))
    model = finalize_model_level_lora(selection.model, compressed_proxy_names=None) if selection.dense_targets else selection.model
    prepare_model_export(model, precision)
    model.eval().requires_grad_(False)
    clear_model_vae_linear_cache(model)
    if use_cache is not None:
        model.config.use_cache = use_cache
    require(not list(iter_named_peft_lora_layers(model)) and not any(layer.trainable_decode for _name, layer in selection.decoder_targets),
        "Finalize every actual selected decoder and merge every future-dense adapter.")
    changed = {entry["name"].removeprefix("base_model.model.") for entry in manifest if entry["component_class"] != "lora"}
    changed.update(protection_changed)
    changed.update(name + ".weight" for name in selection.dense_targets)
    actual = model.state_dict()
    require(set(actual) == set(original) and all(name in changed or tensor_digest({name: actual[name]}) == value
        for name, value in original.items()), "Selected-node export must preserve every unselected persistent raw field.")
    info = dict(source_kind="recovery_node", zero_kd=False, source_node_id=node["checkpoint_id"],
        round_base_id=raw["checkpoint_id"], source_round_base=str(raw_directory), actual_updates=selected.selected_step,
        schedule_max=selected.settings.schedule_max, source_trajectory_id=selected.source_trajectory_id,
        source_candidate_id=selected.source_candidate_id, recipe_id=selected.recipe_id,
        all_mutable_restored=True, mutable=restoration, native_decoders_finalized=finalized,
        future_dense_lora_merged=len(selection.dense_targets), frozen_fields_exact=len(set(original) - changed))
    stage = dict(mode=selected.settings.scope, category=selected.category_id, did_train=True,
        actual_updates=selected.selected_step, planned_steps=selected.settings.schedule_max,
        source_training_checkpoint_id=node["checkpoint_id"], round_base_checkpoint_id=raw["checkpoint_id"],
        decoder_targets=[name for name, _layer in selection.decoder_targets], remaining_lora_targets=list(selection.dense_targets),
        source_immutable_contract=contract)
    return _save_boundary(model=model, raw=raw, raw_directory=raw_directory, destination=destination,
        selected=selected, phase=phase, model_factory=model_factory, info=info, manifest=manifest,
        stage=stage, resolved_learning_rates=node["resolved_learning_rates"])


def complete_boundary_reference(*, export_report, inventory, full8_score=None) -> dict:
    """Accept only this artifact's actual state, runtime, ledger and full-eight."""
    from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
    directory = Path(export_report["path"]).resolve()
    meta = v6.load_v6_meta(str(directory))
    extra = meta.get("extra_meta") or {}
    provenance = extra.get("cat_auto_selected_node")
    fresh = export_report["fresh_evidence"]
    runtime = v6.load_v6_cat_runtime_state(str(directory), required=True, map_location="cpu")["runtime_identity"]
    require(meta["checkpoint_kind"] == "category_boundary" and provenance == export_report["export_evidence"]
        and meta["checkpoint_id"] == export_report["uuid"] == fresh["checkpoint_id"]
        and meta["completed_categories"] == export_report["completed_prefix"] == runtime["completed_prefix"]
        and runtime == export_report["runtime_identity"] == provenance["runtime_identity"], "Boundary UUID/provenance/progress/runtime changed.")
    require(_sha(directory / meta["state_dict_file"]) == export_report["digest"] == fresh["weight_digest"]
        and fresh["state_exact"] and extra["cat_auto_phase"] == export_report["phase"] == provenance["phase"],
        "Bind the actual whole payload and independent fresh proof without relabeling phase.")
    zero = provenance.get("zero_kd") is True
    require((zero and provenance["actual_updates"] == 0 and provenance["raw_state_exact"] and not export_report["mutable_manifest"]
            and provenance["source_node_id"] == provenance["round_base_id"]
            and not provenance["optimizer_constructed"] and not provenance["trainer_constructed"])
        or (not zero and provenance["actual_updates"] > 0 and provenance["all_mutable_restored"] and export_report["mutable_manifest"]),
        "Raw0 and positive-node boundaries require their distinct actual source contracts.")
    ledger = measure_checkpoint(directory, inventory)
    phase, eligible = extra["cat_auto_phase"], False
    if phase == "formal":
        require(full8_score is not None and full8_score.evidence["phase"] == "formal"
            and full8_score.evidence.get("validated_complete_summary")
            and full8_score.evidence["checkpoint_id"] == meta["checkpoint_id"]
            and full8_score.evidence["source_checkpoint_id"] == provenance["source_node_id"],
            "The actual deployment artifact needs its own validated complete-eight evidence.")
        require(not ledger["final_budget_applicable"] or ledger["final_budget_passed"], "The last boundary must meet actual final logical bits.")
        forward = export_report.get("fresh_forward_evidence") or {}
        require(forward.get("checkpoint_id") == meta["checkpoint_id"]
            and forward.get("weight_digest") == export_report["digest"]
            and forward.get("exit_code") == 0 and forward.get("forward_executed") is True
            and forward.get("finite_logits") is True and str(forward.get("device", "")).startswith("cuda:"),
            "A formal boundary requires this artifact's actual independent GPU fresh-forward evidence.")
        eligible = True
    elif full8_score is not None:
        require(full8_score.evidence["phase"] == "cpu_fixture", "Technical state evidence cannot borrow formal task metrics.")
    reference = dict(uuid=meta["checkpoint_id"], path=str(directory), digest=export_report["digest"],
        inventory_hash=inventory_id(inventory), completed_prefix=meta["completed_categories"],
        mutable_manifest=export_report["mutable_manifest"], export_evidence=provenance, fresh_evidence=fresh,
        actual_ledger=ledger, formal_eligible=eligible, phase=phase, runtime_identity=runtime,
        selection_evaluation=None if full8_score is None else dict(full8_score.evidence))
    if export_report.get("fresh_forward_evidence") is not None:
        reference["fresh_forward_evidence"] = export_report["fresh_forward_evidence"]
    return validate_boundary(reference)


class StrictMainChain:
    def __init__(self, *, bf16_root_id: str, categories: Sequence[str]):
        self.bf16_root_id, self.categories = bf16_root_id, tuple(categories)
        require(bool(bf16_root_id) and self.categories and len(set(self.categories)) == len(self.categories),
            "Freeze an identified BF16 root and unique complete category order.")
        self.boundaries: list[dict] = []

    @property
    def live_parent_id(self):
        return self.boundaries[-1]["boundary_id"] if self.boundaries else self.bf16_root_id

    def commit(self, selected: SelectedCategory, *, live_parent_id: str, boundary_reference: Mapping,
            selection_certificate: Mapping, actual_round_base_id: str, restored_node_id: str,
            runtime_identity: Mapping) -> dict:
        boundary = validate_boundary(boundary_reference)
        index = len(self.boundaries)
        require(selected.phase == "formal" and boundary["formal_eligible"]
            and index < len(self.categories) and selected.category_id == self.categories[index]
            and live_parent_id == self.live_parent_id, "Only the next formal own-parent selection extends this sole chain.")
        require(restored_node_id == selected.source_node_id and actual_round_base_id == selected.source_round_base_id,
            "Commit this actual selected source and its own complete raw.")
        certificate = dict(selection_certificate)
        selection_id = certificate.pop("selection_id", None)
        expected = dict(category_id=selected.category_id, parent_id=live_parent_id,
            selected_trajectory_id=selected.source_trajectory_id, selected_node_id=restored_node_id,
            selected_round_base_id=actual_round_base_id, selected_step=selected.selected_step,
            selected_kind="raw_zero" if selected.selected_step == 0 else "recovery_node")
        require(digest(certificate) == selection_id and certificate.get("all_announced_cells_complete")
            and all(certificate.get(key) == value for key, value in expected.items()), "Bind the complete immutable category selection certificate.")
        export, fresh = boundary["export_evidence"], boundary["fresh_evidence"]
        require(boundary["completed_prefix"] == list(self.categories[:index + 1])
            and dict(runtime_identity) == boundary["runtime_identity"] == export["runtime_identity"]
            and export["source_node_id"] == restored_node_id and export["round_base_id"] == actual_round_base_id
            and export["actual_updates"] == selected.selected_step
            and fresh["checkpoint_id"] == boundary["uuid"] and fresh["state_exact"], "Commit the actual finalized prefix/runtime/fresh source.")
        require((selected.selected_step == 0 and certificate["raw_zero_strictly_above_all_recovery"]
                and export.get("raw_state_exact") and export.get("zero_kd") and not boundary["mutable_manifest"])
            or (selected.selected_step > 0 and export.get("all_mutable_restored") and boundary["mutable_manifest"]),
            "Zero-KD qualification and positive-node mutable restoration cannot substitute for each other.")
        used = {self.bf16_root_id, actual_round_base_id, restored_node_id, *(row["boundary_id"] for row in self.boundaries)}
        require(boundary["uuid"] not in used and certificate["phase"] == "formal", "Require a new deployment UUID and formal complete selection.")
        row = dict(parent_id=live_parent_id, boundary_id=boundary["uuid"], selected=asdict(selected),
            boundary_reference=boundary, selection_id=selection_id, runtime_identity=dict(runtime_identity),
            checkpoint_kind="category_boundary", training_exact_resume=False)
        self.boundaries.append(row)
        return row

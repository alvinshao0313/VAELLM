"""Read-only native node export from its own full round_base, never a donor prefix."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from e2e_common.full_lora import collect_exact_peft_lora_config, finalize_model_level_lora, iter_named_peft_lora_layers
from litebsq.vae_linear import clear_model_vae_linear_cache
from train_utils import checkpoint_v6 as v6
from train_utils import cat_after_category_common as native_training
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.distill_precision import prepare_model_export

from .contracts import capability, digest, inventory_id, validate_boundary, validate_recipe, validate_run_identity
from .mutable import restore_mutable, select_native_scope
from .nodes import RecoverySettings
from .replay import SelectedCategory


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_pins() -> dict:
    """Reject an accidental historical snapshot import instead of mixing runtimes."""
    root = Path(__file__).resolve().parents[2]
    paths = (Path(v6.__file__).resolve(), Path(native_training.__file__).resolve(), Path(__file__).resolve(),
        *(Path(__file__).with_name(name) for name in ("mutable.py", "recovery.py", "native_builder.py", "nodes.py", "replay.py", "validate_recovery_cpu.py", "contracts.py", "identity.py", "data.py")),
        *(root / name for name in ("train_utils/model_level_checkpoint_state.py", "train_utils/model_level_trainables.py",
            "train_utils/model_level_optimizer.py", "train_utils/distill_precision.py", "train_utils/distill_decoder.py", "e2e_common/full_lora.py")))
    if paths[0] != root / "train_utils/checkpoint_v6.py" or paths[1] != root / "train_utils/cat_after_category_common.py":
        raise ValueError("This isolated native-only adapter requires the actual PUBLIC runtime, not a mixed/N snapshot import.")
    return {str(path): _sha(path) for path in paths}


def architecture(directory: str | Path, *, attn_implementation: str = "eager"):
    config = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation, trust_remote_code=False)
    return model


def fresh_native(directory: str | Path, *, model_factory: Callable = architecture) -> tuple[object, dict]:
    model = model_factory(directory)
    model, meta, loaded = v6.load_v6_full_checkpoint_into_model(model, str(directory), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Fresh native load did not restore the complete model inventory.")
    state = torch.load(Path(directory) / meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
    actual = model.state_dict()
    if set(actual) != set(state) or any(actual[name].dtype != value.dtype or not torch.equal(actual[name].cpu(), value) for name, value in state.items()):
        raise ValueError("Independent architecture reconstruction differs from the saved complete state.")
    return model, dict(checkpoint_id=meta["checkpoint_id"], state_exact=True, tensor_count=len(state),
        own_weight_path=str(Path(directory) / meta["state_dict_file"]),
        weight_digest=_sha(Path(directory) / meta["state_dict_file"]),
        forward_executed=False, gpu_validation=False)


def export_selected_node(*, node_directory: str | Path, destination: str | Path,
                         selected: SelectedCategory, current_targets: Sequence[str],
                         phase: str, model_factory: Callable = architecture) -> dict:
    """Deployment export only; does not construct a Trainer or claim exact resume."""
    if phase not in ("cpu_fixture", "pilot", "formal"):
        raise ValueError("Explicitly distinguish CPU state fixtures, real pilot and formal exports.")
    if phase != "cpu_fixture" and model_factory is architecture:
        raise ValueError("A real pilot/formal export needs the explicit model adapter's original attention/runtime construction policy.")
    if phase == "formal" and selected.phase != "formal":
        raise ValueError("A technical/CPU fixture node cannot acquire formal provenance during export.")
    pins = source_pins()
    node_directory, destination = Path(node_directory).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refuse to replace an existing candidate export: {destination}")
    node = v6.load_v6_training_step_meta(str(node_directory))
    raw_directory, raw = v6.resolve_training_step_round_base_ref(str(node_directory), node)
    raw_directory = Path(raw_directory)
    if node["checkpoint_id"] != selected.source_node_id or raw["checkpoint_id"] != selected.source_round_base_id:
        raise ValueError("Selected node/raw UUIDs differ from the actual native source payload.")
    v6.validate_training_step_round_base(node, raw)
    contract = node["immutable_resume_contract"]
    state = json.loads((node_directory / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] != selected.selected_step or state["max_steps"] != selected.settings.schedule_max or contract["optimization"]["steps"] != selected.settings.schedule_max:
        raise ValueError("The selected node must preserve actual selected_step and its original schedule_max.")
    if contract["round_base_checkpoint_id"] != raw["checkpoint_id"] or contract["active_category"] != selected.category_id or contract["after_category_mode"] != selected.settings.scope:
        raise ValueError("The immutable source contract does not describe this selected category/scope/raw.")
    audit = (node.get("runtime_audit") or {}).get("cat_auto_recovery")
    if audit is None or audit["semantic_seeds"] != dict(selected.seed_identity) or audit["schedule_max"] != selected.settings.schedule_max:
        raise ValueError("The node must bind the actual semantic random streams and original schedule.")
    if phase == "formal" and not audit.get("formal_eligible"):
        raise ValueError("A technical save-return fixture cannot be relabeled as a formal recovery node.")
    order, completed = raw["compression_categories"], raw["completed_categories"]
    if completed != order[:len(completed)] or len(completed) >= len(order) or order[len(completed)] != selected.category_id:
        raise ValueError("The source raw must be the next category of its own frozen complete prefix.")
    if len(completed) + 1 < len(order) and not raw["pending_dense_targets"]:
        raise ValueError("Future dense LoRA may be absent only at the final category boundary.")
    optimization = contract["optimization"]
    decoder_lr = optimization["learning_rate"] if optimization.get("decoder_lr") is None else optimization["decoder_lr"]
    if decoder_lr != selected.settings.decoder_lr:
        raise ValueError("The source decoder LR differs from the frozen selected recovery settings.")
    precision = SimpleNamespace(**contract["precision"])
    if not precision.bf16 or precision.fp16:
        raise ValueError("The core protocol retains BF16 compute/export and actual FP32 masters.")
    aux = AuxTrainableConfig(**contract["aux"])
    if aux.norm_train_mode != "none" or aux.lm_head_train_mode != "none" or aux.residual_lora_mode != "none":
        raise NotImplementedError("The confirmed core exporter has no enabled norm/head/residual auxiliary branch.")
    runtime = v6.load_v6_cat_runtime_state(str(raw_directory), required=True, map_location="cpu")
    if runtime["runtime_identity"].get("cat_shared", {}).get("channel_refresh_after_category", False):
        raise NotImplementedError("Post-recovery channel-refresh runtime requires its actual refresh adapter before export.")
    model, actual_raw, loaded = v6.load_v6_full_checkpoint_into_model(model_factory(raw_directory),
        str(raw_directory), expected_kind="round_base", strict=True)
    if actual_raw["checkpoint_id"] != raw["checkpoint_id"] or loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Failed to reconstruct the exact complete native round_base.")
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    previous_use_cache = getattr(model.config, "use_cache", None)
    selection = select_native_scope(model, settings=selected.settings,
        prefix_targets=raw["compressed_targets"], current_targets=current_targets,
        future_dense_targets=raw["pending_dense_targets"], precision=precision,
        lora_init_seed=selected.seed_identity["lora_init"], aux=aux,
        fp32_components=contract["optimization"].get("distill_fp32_components", ()))
    if list(contract["decoder_target_names"]) != [name for name, _layer in selection.decoder_targets] or list(contract["lora_target_names"]) != list(selection.dense_targets):
        raise ValueError("The reconstructed actual decoder/LoRA owners differ from the source immutable manifest.")
    actual_lora = None if not selection.dense_targets else collect_exact_peft_lora_config(selection.model,
        default_rank=12, alpha=24, dropout=.1)
    if actual_lora != contract["lora"] or actual_lora != node["lora_config"]:
        raise ValueError("The reconstructed actual LoRA topology differs from the saved contract.")
    values, manifest = v6.load_v6_training_model_state(str(node_directory), map_location="cpu")
    restoration = restore_mutable(selection, values, manifest)
    finalized = finalize_main_decoder_targets(tuple(NamedMainDecoderTarget(name, layer)
        for name, layer in selection.decoder_targets))
    model = finalize_model_level_lora(selection.model, compressed_proxy_names=None) if selection.dense_targets else selection.model
    prepare_model_export(model, precision)
    model.eval().requires_grad_(False)
    clear_model_vae_linear_cache(model)
    if previous_use_cache is not None:
        model.config.use_cache = previous_use_cache
    if list(iter_named_peft_lora_layers(model)) or any(layer.trainable_decode for _name, layer in selection.decoder_targets):
        raise ValueError("Finalize every actual native decoder and merge every selected future-dense adapter.")
    actual = model.state_dict()
    changed = {entry["name"].removeprefix("base_model.model.") for entry in manifest if entry["component_class"] != "lora"}
    changed.update(name + ".weight" for name in selection.dense_targets)
    if set(actual) != set(original):
        raise ValueError("Finalization changed the persistent round_base inventory.")
    for name, value in original.items():
        if name not in changed and (actual[name].dtype != value.dtype or not torch.equal(actual[name].cpu(), value)):
            raise ValueError(f"An unselected persistent round_base field changed: {name}")
    extra = deepcopy(raw.get("extra_meta") or {})
    info = dict(source_node_id=node["checkpoint_id"], round_base_id=raw["checkpoint_id"],
        source_round_base=str(raw_directory), actual_updates=selected.selected_step,
        schedule_max=selected.settings.schedule_max, source_trajectory_id=selected.source_trajectory_id,
        source_candidate_id=selected.source_candidate_id, recipe_id=selected.recipe_id,
        all_mutable_restored=True, mutable=restoration, native_decoders_finalized=finalized,
        future_dense_lora_merged=len(selection.dense_targets), frozen_fields_exact=len(set(original) - changed),
        training_exact_resume=False, source_pins=pins, phase=phase, gpu_validation=False)
    history = deepcopy(extra.get("distill_stage_history") or [])
    stage = dict(mode=selected.settings.scope, category=selected.category_id, did_train=True,
        actual_updates=selected.selected_step, planned_steps=selected.settings.schedule_max,
        source_training_checkpoint_id=node["checkpoint_id"], round_base_checkpoint_id=raw["checkpoint_id"],
        decoder_targets=[name for name, _layer in selection.decoder_targets],
        remaining_lora_targets=list(selection.dense_targets), source_immutable_contract=contract)
    history.append(stage)
    extra.update(cat_auto_selected_node=info, cat_auto_phase=phase, distill_stage=stage,
        distill_stage_history=history, stage="after_category", category=selected.category_id,
        active_category=None, round_idx=len(completed) + 1)
    saved = v6.save_v6_full_checkpoint(model, str(destination), checkpoint_kind="category_boundary",
        compressed_targets=raw["compressed_targets"], pending_dense_targets=raw["pending_dense_targets"],
        skip_targets=raw["skip_targets"], legacy_original_only_sources=raw["legacy_original_only_sources"],
        completed_categories=[*completed, selected.category_id], compression_categories=order,
        target_layers=raw["target_layers"], target_modules=raw["target_modules"], train_mode="none",
        after_category_mode=selected.settings.scope, lora_config=None, resolved_learning_rates=node["resolved_learning_rates"],
        base_model_path=node["base_model_path"], finalized_status=dict(lora_finalized=True,
            decoder_finalized=True, aux_finalized=True, stable_category_boundary=True, inference_ready=False),
        cat_runtime_state=runtime, extra_meta=extra)
    meta = v6.load_v6_meta(str(destination))
    if saved["checkpoint_id"] != meta["checkpoint_id"] or meta["checkpoint_id"] in (raw["checkpoint_id"], node["checkpoint_id"]):
        raise ValueError("A selected deployment boundary requires its own new native UUID.")
    _fresh, evidence = fresh_native(destination, model_factory=model_factory)
    return dict(uuid=meta["checkpoint_id"], path=str(destination), digest=evidence["weight_digest"],
        completed_prefix=meta["completed_categories"], mutable_manifest=manifest,
        export_evidence=info, fresh_evidence=evidence, formal_eligible=False,
        phase=phase, runtime_identity=runtime["runtime_identity"],
        ledger_status="required_before_formal_chain_commit", gpu_validation=False,
        limitation="CPU serialization/fresh state proof only; real model-scale save-return/export/forward and own live successor require the separately budgeted pilot.")


def next_live_fit_request(*, boundary_reference, run_identity, resolved_recipe, phase="formal") -> dict:
    """Validate a controlled new fit request; it never asserts old exact-resume."""
    boundary = validate_boundary(boundary_reference)
    identity, recipe = validate_run_identity(run_identity), validate_recipe(resolved_recipe)
    if phase not in ("formal", "technical_validation", "cpu_fixture"):
        raise ValueError("Use an explicit formal/model-scale technical/CPU bridge phase.")
    if phase == "formal" and not boundary["formal_eligible"]:
        raise ValueError("A formal live fit requires its own formally accepted previous boundary.")
    if phase != "cpu_fixture" and boundary.get("phase") == "cpu_fixture":
        raise ValueError("A CPU fixture boundary cannot acquire model-scale/formal provenance.")
    if identity["parent_uuid"] != boundary["uuid"] or identity["parent_weights_digest"] != boundary["digest"]:
        raise ValueError("A live fit must read its own validated finalized boundary and exact fused weights.")
    if recipe["category_id"] in boundary["completed_prefix"]:
        raise ValueError("The next live fit cannot rewrite an already selected prefix category.")
    return dict(parent_uuid=boundary["uuid"], own_live_weights_path=boundary["path"],
        parent_weights_digest=boundary["digest"], resolved_recipe=recipe,
        controlled_new_fork=True, training_exact_resume=False,
        phase=phase, capability=capability("constructible", "fit_next_live_category calls the real isolated PUBLIC category fitter, saves its actual own-parent raw and binds the new runtime identity.",
            verification_scope="actual_code_adapter; model-scale GPU pilot still required"),
        requirement="Construct the next raw from these fused future-dense weights and actual refreshed statistics; do not install BF16/donor banks or reuse the old global runtime identity.")


def _boundary_reference(directory, inventory, *, phase, export_evidence, mutable_manifest=()):
    from .ledger import measure_checkpoint
    meta = v6.load_v6_meta(str(directory))
    ledger = measure_checkpoint(directory, inventory)
    _model, fresh = fresh_native(directory)
    reference = dict(uuid=meta["checkpoint_id"], path=str(Path(directory).resolve()),
        digest=fresh["weight_digest"], inventory_hash=inventory_id(inventory),
        completed_prefix=meta["completed_categories"], mutable_manifest=list(mutable_manifest),
        export_evidence=export_evidence, fresh_evidence=fresh, actual_ledger=ledger,
        formal_eligible=phase == "formal", phase=phase,
        runtime_identity=v6.load_v6_cat_runtime_state(str(directory), required=True)["runtime_identity"])
    return validate_boundary(reference)


def complete_boundary_reference(*, export_report, inventory, full8_score=None) -> dict:
    """Bind a real node export and ledger; only its complete formal score admits it."""
    from .contracts import require
    from .ledger import measure_checkpoint
    directory = Path(export_report["path"]).resolve()
    meta = v6.load_v6_meta(str(directory))
    provenance = (meta.get("extra_meta") or {}).get("cat_auto_selected_node")
    require(meta["checkpoint_kind"] == "category_boundary" and provenance is not None,
        "Accept the actual finalized selected-node deployment, not a raw or historical donor.")
    require(meta["checkpoint_id"] == export_report["uuid"]
        and provenance == export_report["export_evidence"]
        and meta["completed_categories"] == export_report["completed_prefix"],
        "The exported node identity/provenance/prefix changed before boundary acceptance.")
    require(_sha(directory / meta["state_dict_file"]) == export_report["digest"]
        and export_report["fresh_evidence"]["checkpoint_id"] == meta["checkpoint_id"]
        and export_report["fresh_evidence"]["state_exact"]
        and export_report["fresh_evidence"]["weight_digest"] == export_report["digest"],
        "Boundary acceptance must bind the exported whole payload and its actual independent fresh proof.")
    phase = (meta.get("extra_meta") or {})["cat_auto_phase"]
    require(phase == export_report["phase"] and provenance["all_mutable_restored"],
        "Technical/CPU provenance cannot be relabeled during boundary assembly.")
    ledger = measure_checkpoint(directory, inventory)
    eligible = False
    if phase == "formal":
        require(full8_score is not None and full8_score.evidence["phase"] == "formal"
            and full8_score.evidence.get("validated_complete_summary")
            and full8_score.evidence["checkpoint_id"] == meta["checkpoint_id"]
            and full8_score.evidence["source_checkpoint_id"] == provenance["source_node_id"],
            "A formal selected boundary needs the exact deployment UUID's validated complete-eight evidence.")
        require(not ledger["final_budget_applicable"] or ledger["final_budget_passed"],
            "The last category must meet the actual final logical bit cap.")
        eligible = True
    elif full8_score is not None:
        require(full8_score.evidence["phase"] == "cpu_fixture", "A technical boundary cannot borrow a formal endpoint score.")
    reference = dict(uuid=meta["checkpoint_id"], path=str(directory), digest=export_report["digest"],
        inventory_hash=inventory_id(inventory), completed_prefix=meta["completed_categories"],
        mutable_manifest=export_report["mutable_manifest"], export_evidence=provenance,
        fresh_evidence=export_report["fresh_evidence"], actual_ledger=ledger,
        formal_eligible=eligible, phase=phase,
        runtime_identity=v6.load_v6_cat_runtime_state(str(directory), required=True)["runtime_identity"],
        selection_evaluation=None if full8_score is None else dict(full8_score.evidence))
    return validate_boundary(reference)


def materialize_bf16_root(*, inventory, frozen_order: Sequence[str], destination,
        attention_implementation: str, tokenizer=None, phase="formal") -> dict:
    """Save a real dense v6 root once; an HF source is never relabeled as v6."""
    from .contracts import require, validate_inventory
    from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
    inventory = validate_inventory(inventory)
    require(phase in ("formal", "cpu_fixture"), "Root materialization is formal source handling or an explicit CPU fixture.")
    require(len(frozen_order) == len(set(frozen_order)) and set(frozen_order) == set(inventory["category_ids"]),
        "Freeze the full online category order explicitly before materializing its native root.")
    source_pins()
    destination = Path(destination).resolve()
    require(not destination.exists(), "The root destination must be new; do not overwrite a source or another chain.")
    source = inventory["source_evidence"]
    if phase == "formal":
        require(inventory["model_identity"].get("schema") == "dense_model_content_identity_v0" and source.get("verification"),
            "A formal BF16 root needs the actual verified original model content identity and source receipt.")
    source_path = Path(source["path"]).resolve()
    require(source_path.is_dir(), "Load the original BF16 root from its actual local source directory.")
    model = AutoModelForCausalLM.from_pretrained(str(source_path), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation=attention_implementation)
    require(model.config.model_type == inventory["model_identity"]["model_type"]
        and all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()),
        "The original source must reconstruct the actual dense BF16 architecture.")
    projections = {row["actual_name"]: row for row in inventory["projections"]}
    for name, row in projections.items():
        layer = model.get_submodule(name)
        require(type(layer) is torch.nn.Linear and list(layer.weight.shape) == row["shape"],
            "Original BF16 root projection differs from its real complete inventory: " + name)
    model.eval().requires_grad_(False)
    identity = dict(schema="cat_auto_BF16_native_root_v0", model_identity=inventory["model_identity"],
        compression_categories=list(frozen_order), online_order_frozen=True,
        offline_source_identity=inventory["model_identity"], training_exact_resume=False)
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None, runtime_identity=identity)
    evidence = dict(original_BF16_root=True, source_content_identity=inventory["model_identity"],
        source_evidence=source, converted_modules=0, complete_pending_dense=len(projections),
        no_compression_or_KD_executed=True, training_exact_resume=False,
        source_pins=source_pins(), phase=phase)
    v6.save_v6_full_checkpoint(model, str(destination), checkpoint_kind="category_boundary",
        compressed_targets=(), pending_dense_targets=tuple(projections), skip_targets=(),
        completed_categories=(), compression_categories=frozen_order,
        target_layers=sorted({row["layer_id"] for row in projections.values()}), target_modules=inventory["category_ids"],
        train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=str(source_path), tokenizer=tokenizer, cat_runtime_state=runtime,
        finalized_status=dict(lora_finalized=True, decoder_finalized=True, aux_finalized=True,
            stable_category_boundary=True, inference_ready=False),
        extra_meta=dict(cat_auto_bf16_root=evidence, cat_auto_phase=phase, distill_stage_history=[], round_idx=0))
    del model
    return _boundary_reference(destination, inventory, phase=phase, export_evidence=evidence)


def fit_next_live_category(*, boundary_reference, run_identity, resolved_recipe, inventory,
        calibration_bundle, frozen_order: Sequence[str], device: str, output_dir,
        attention_implementation: str, phase="formal", technical_steps=None,
        activation_bundle=None) -> dict:
    """Load own fused parent -> actual category fit -> new complete native raw."""
    from .adapter import category_projections
    from .contracts import require, validate_inventory
    from .fit_adapter import fit_category
    from .ledger import measure_checkpoint
    from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
    inventory = validate_inventory(inventory)
    request = next_live_fit_request(boundary_reference=boundary_reference, run_identity=run_identity,
        resolved_recipe=resolved_recipe, phase=phase)
    boundary = validate_boundary(boundary_reference)
    identity, recipe = validate_run_identity(run_identity), validate_recipe(resolved_recipe)
    require(identity["model"] == inventory["model_identity"], "Fit and parent source must share the same BF16 model content identity.")
    require(boundary["inventory_hash"] == inventory_id(inventory), "The own parent boundary changed its complete target inventory.")
    require(phase != "formal" or technical_steps is None, "A technical fit cannot be offered as a full 10K formal candidate.")
    require(phase != "cpu_fixture" or device == "cpu", "CPU bridge validation cannot allocate a GPU.")
    prefix, order = tuple(boundary["completed_prefix"]), tuple(frozen_order)
    require(len(order) == len(set(order)) and set(order) == set(inventory["category_ids"])
        and prefix == order[:len(prefix)] and len(prefix) < len(order)
        and recipe["category_id"] == order[len(prefix)], "Fit exactly the next semantic category of the frozen single chain.")
    parent_dir, output_dir = Path(boundary["path"]).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), "Use a new controlled live fork output directory.")
    parent = v6.load_v6_meta(str(parent_dir))
    require(parent["checkpoint_id"] == boundary["uuid"] and tuple(parent["completed_categories"]) == prefix
        and tuple(parent["compression_categories"]) == order, "Parent native metadata is not this selected order/prefix/UUID.")
    require(_sha(parent_dir / parent["state_dict_file"]) == boundary["digest"], "The own merged parent weight payload changed.")
    model, loaded_meta, loaded = v6.load_v6_full_checkpoint_into_model(
        architecture(parent_dir, attn_implementation=attention_implementation), str(parent_dir),
        expected_kind="category_boundary", strict=True)
    require(loaded_meta["checkpoint_id"] == boundary["uuid"] and not loaded.missing_keys and not loaded.unexpected_keys,
        "The live fitter must reconstruct its complete own-parent fused model.")
    current_names = {row["actual_name"] for row in category_projections(inventory, recipe["category_id"])}
    require(current_names <= set(parent["pending_dense_targets"]), "Current fitting targets must be the actual future dense weights of this parent.")
    fitted = fit_category(model, inventory=inventory, recipe=recipe, run_identity=identity,
        calibration_bundle=calibration_bundle, activation_bundle=activation_bundle,
        device=device, output_dir=output_dir / "fit", technical_steps=technical_steps)
    model, fit, activation = fitted["model"], fitted["fit_report"], fitted["activation_bundle"]
    require(fit["status"] == "completed" and fit["exit_code"] == 0 and set(fit["actual_targets"]) == current_names
        and fit["parent_identity"] == identity and fit["frozen_prefix_before_digest"] == fit["frozen_prefix_after_digest"],
        "Only a complete actual own-parent category fit may become a new raw.")
    require(activation["source_parent_uuid"] == boundary["uuid"]
        and activation["source_weights_digest"] == boundary["digest"]
        and activation["calibration_data_identity"] == calibration_bundle["data_identity"],
        "Input statistics must come from this own parent and its frozen calibration blocks.")
    compressed = tuple([*parent["compressed_targets"], *recipe["group_signature"]])
    pending = tuple(name for name in parent["pending_dense_targets"] if name not in current_names)
    v6.validate_model_target_inventories(model, compressed, pending, parent["skip_targets"])
    runtime_identity = dict(schema="cat_auto_controlled_native_category_fork_v0",
        model_identity=inventory["model_identity"], compression_categories=list(order),
        source_parent_uuid=boundary["uuid"], source_weights_digest=boundary["digest"],
        resolved_current_recipe=recipe, data_identity=identity["data_identity"],
        consumer_identity=identity["consumer_identity"], code_identity=identity["code_identity"],
        activation_statistics_identity=activation["statistics_identity"],
        controlled_new_fork=True, training_exact_resume=False)
    runtime = build_cat_runtime_state(activation_runtime=activation["activation_runtime"],
        global_adaptive_plan=activation["global_adaptive_plan"], runtime_identity=runtime_identity)
    extra = deepcopy(parent.get("extra_meta") or {})
    extra.update(cat_auto_phase=phase, cat_auto_live_fit=dict(request=request,
        candidate_id=fit["candidate_id"], artifact_id=fit["artifact_id"], actual_fit_report=fit,
        source_parent_uuid=boundary["uuid"], source_parent_path=str(parent_dir),
        source_weights_digest=boundary["digest"], new_runtime_identity=runtime_identity,
        previous_runtime_identity=boundary.get("runtime_identity"), training_exact_resume=False),
        stage="round_base", active_category=recipe["category_id"], category=recipe["category_id"], round_idx=len(prefix))
    model.eval().requires_grad_(False)
    raw_dir = output_dir / "round_base"
    saved = v6.save_v6_full_checkpoint(model, str(raw_dir), checkpoint_kind="round_base",
        compressed_targets=compressed, pending_dense_targets=pending, skip_targets=parent["skip_targets"],
        legacy_original_only_sources=parent["legacy_original_only_sources"], completed_categories=prefix,
        compression_categories=order, target_layers=parent["target_layers"], target_modules=parent["target_modules"],
        train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=parent["base_model_path"], cat_runtime_state=runtime, extra_meta=extra,
        finalized_status=dict(lora_finalized=True, decoder_finalized=True, aux_finalized=True,
            stable_category_boundary=False, inference_ready=False))
    raw = v6.load_v6_meta(str(raw_dir))
    require(saved["checkpoint_id"] == raw["checkpoint_id"] != boundary["uuid"]
        and tuple(raw["completed_categories"]) == prefix and set(raw["compressed_targets"]) == set(compressed),
        "The fresh fitted raw needs a new UUID, current compressed and the old completion prefix.")
    ledger = measure_checkpoint(raw_dir, inventory)
    return dict(round_base_dir=str(raw_dir), round_base_uuid=raw["checkpoint_id"],
        source_parent_uuid=boundary["uuid"], candidate_id=fit["candidate_id"], artifact_id=fit["artifact_id"],
        completed_prefix=list(prefix), current_category=recipe["category_id"],
        compressed_targets=list(compressed), pending_dense_targets=list(pending),
        fit_report=fit, activation_bundle=activation, runtime_identity=runtime_identity,
        actual_ledger=ledger, phase=phase, formal_eligible=False,
        training_exact_resume=False, next_step="make_native_recovery_trainer on this exact round_base_dir")


def build_live_recovery(*, live_raw, inventory, settings, prepared_data, sources,
        tokenizer, output_dir, execution_policy, technical_stop_after=None, replay_selected_step=None):
    """Real fit artifact to the real native KD builder; no executor placeholder."""
    from .native_builder import make_native_recovery_trainer
    meta = v6.load_v6_meta(live_raw["round_base_dir"])
    if meta["checkpoint_id"] != live_raw["round_base_uuid"]:
        raise ValueError("The fitted raw identity changed before native recovery construction.")
    return make_native_recovery_trainer(round_base_dir=live_raw["round_base_dir"], inventory=inventory,
        category_id=live_raw["current_category"], settings=settings, prepared_data=prepared_data,
        sources=sources, tokenizer=tokenizer, output_dir=output_dir, execution_policy=execution_policy,
        phase=live_raw["phase"], technical_stop_after=technical_stop_after,
        replay_selected_step=replay_selected_step)

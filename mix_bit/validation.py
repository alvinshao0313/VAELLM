from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from litebsq.vae_linear import VAELinear
from mix_bit.assembler import (
    REFERENCE_LOGITS_FILENAME,
    _validate_allocation_payload,
    build_model_from_assignments,
    build_uniform_assignments,
)
from mix_bit.checkpoint_pool import (
    FORBIDDEN_PAYLOAD_LEAVES,
    PROTECTED_SPARSE_SPEC_KEYS,
    CandidatePoolIndex,
)
from mix_bit.cost_search import (
    CostWorkerContext,
    TeacherCacheView,
    _load_calibration_examples,
    _validate_dataset_manifest,
    evaluate_student_per_sample_kl,
    load_teacher_cache_for_worker,
    load_teacher_model,
)
from mix_bit.calibration import (
    TOKENIZER_FINGERPRINT_VERSION,
    compute_tokenizer_config_sha256,
)
from mix_bit.kl_metric import (
    KL_MODE_TEACHER_TOPK,
    resolve_metric_contract,
    validate_kl_mode_arguments,
)
from mix_bit.model_adapter import get_model_adapter, normalize_tokenizer_for_mix_bit
from mix_bit.model_inventory import ModelInventory
from mix_bit.schema import ResolvedRunConfig, sha256_file
from mix_bit.solver import OBJECTIVE_MATCH_REL, bit_to_units, load_cost_table_for_solve
from mix_bit.teacher_cache import validate_teacher_cache_against_inputs
from train_utils.eval_utils import calculate_ppl, run_lm_eval
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    STATE_DICT_FILENAME,
    load_model_checkpoint,
)
from transformers import AutoTokenizer

RELATIVE_GAP_EPS = 1e-12
SAVE_RELOAD_RTOL = 1e-4
SAVE_RELOAD_ATOL = 1e-4

DOWNSTREAM_TASKS = (
    "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
)
DOWNSTREAM_SEQLEN = 2048

def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_tokenizer_for_profile(resolved: ResolvedRunConfig, *, access_token: str | None = None):
    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    return adapter.load_tokenizer(profile, access_token=access_token)


def _load_tokenizer_from_final_dir(final_dir: Path) -> Any:
    """Local-only reload of the tokenizer saved alongside the mixed checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(final_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    return normalize_tokenizer_for_mix_bit(tokenizer, source_label=str(final_dir))


def _verify_final_tokenizer_fingerprint(
    *,
    final_dir: Path,
    mix: Mapping[str, Any],
) -> dict[str, Any]:
    """Load tokenizer local-only, recompute fingerprint, compare against meta."""
    version = mix.get("tokenizer_fingerprint_version")
    if version != TOKENIZER_FINGERPRINT_VERSION:
        raise ValueError(
            f"checkpoint mix_bit tokenizer_fingerprint_version={version!r}, "
            f"expected {TOKENIZER_FINGERPRINT_VERSION}"
        )
    expected_sha = mix.get("tokenizer_fingerprint_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError("checkpoint mix_bit missing tokenizer_fingerprint_sha256")
    try:
        tokenizer = _load_tokenizer_from_final_dir(final_dir)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed to local-only reload tokenizer from {final_dir}: {exc}"
        ) from exc
    reload_sha = compute_tokenizer_config_sha256(tokenizer)
    if reload_sha != expected_sha:
        raise ValueError(
            f"tokenizer fingerprint mismatch: meta={expected_sha!r} reload={reload_sha!r}"
        )
    return {
        "fingerprint_version": TOKENIZER_FINGERPRINT_VERSION,
        "fingerprint_sha256": reload_sha,
        "reported_name_or_path": str(final_dir.resolve()),
        "local_reload_passed": True,
    }


def _reject_forbidden_checkpoint_payload(
    *,
    meta: Mapping[str, Any],
    state_keys: Sequence[str],
) -> None:
    extra = meta.get("extra_meta")
    if isinstance(extra, Mapping):
        for key in extra:
            lowered = str(key).lower()
            if lowered == "mix_bit":
                continue
            if any(
                token in lowered
                for token in (
                    "peft",
                    "lora",
                    "adapter",
                    "distill",
                    "e2e",
                    "protected",
                    "sparse",
                    "original_weight",
                )
            ):
                raise ValueError(f"Forbidden extra_meta key: {key}")

    for spec in meta.get("converted_modules", []) or []:
        if not isinstance(spec, Mapping):
            continue
        name = str(spec.get("name", "<unknown>"))
        if bool(spec.get("has_original_weight")):
            raise ValueError(
                f"{name}: forbidden original_weight payload (has_original_weight=true)"
            )
        for key in PROTECTED_SPARSE_SPEC_KEYS:
            value = spec.get(key)
            if value is None:
                continue
            if isinstance(value, (list, dict)) and len(value) == 0:
                continue
            raise ValueError(f"{name}: forbidden protected/sparse meta field {key!r}")
        stages = int(spec.get("protected_residual_stages", 0) or 0)
        if stages > 0:
            raise ValueError(f"{name}: forbidden protected_residual_stages={stages}")

    for key in state_keys:
        leaf = key.split(".")[-1]
        if leaf in FORBIDDEN_PAYLOAD_LEAVES or leaf.startswith("protected_") or leaf.startswith(
            "sparse_residual_"
        ):
            raise ValueError(f"Forbidden state_dict key: {key}")
        lowered = leaf.lower()
        if "lora" in lowered or "adapter" in lowered or "peft" in lowered:
            raise ValueError(f"Forbidden adapter state_dict key: {key}")


def _require_finite_number(payload: Mapping[str, Any], key: str, *, label: str) -> float:
    if key not in payload or payload[key] is None:
        raise ValueError(f"{label} missing required field {key}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} field {key} must be a finite number, got {payload[key]!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} field {key} must be finite, got {value!r}")
    return value


def _validate_baseline_overlay_against_cost_meta(
    baseline_overlay_path: str | Path,
    cost_meta: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(baseline_overlay_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline overlay: {path}")
    expected = cost_meta.get("baseline_overlay_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("cost_meta missing required baseline_overlay_sha256")
    found = sha256_file(path)
    if found != expected:
        raise ValueError(
            f"baseline_overlay_sha256 mismatch: file={found!r} cost_meta={expected!r}"
        )
    overlay = _read_json(path)
    if overlay.get("kind") != "uniform_baseline_overlay":
        raise ValueError(
            f"Unexpected baseline overlay kind={overlay.get('kind')!r} in {path}"
        )
    return {
        "baseline_overlay_path": str(path.resolve()),
        "baseline_overlay_sha256": found,
        "kind": overlay.get("kind"),
    }


def _validate_teacher_cache_against_cost_meta(
    *,
    teacher_cache: str | Path | None,
    cost_meta: Mapping[str, Any],
    contract,
) -> TeacherCacheView | None:
    if contract.kl_mode != KL_MODE_TEACHER_TOPK:
        return None
    expected = cost_meta.get("teacher_cache_index_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "cost_meta missing required teacher_cache_index_sha256 for teacher_topk"
        )
    if teacher_cache is None:
        raise ValueError("teacher_topk validation requires --teacher_cache")
    view = load_teacher_cache_for_worker(teacher_cache)
    if view.index_sha256 != expected:
        raise ValueError(
            "teacher_cache_index_sha256 mismatch: "
            f"cache={view.index_sha256!r} cost_meta={expected!r}"
        )
    cache_k = int(view.index["teacher_topk"])
    expected_k = int(contract.teacher_topk)
    if cache_k != expected_k:
        raise ValueError(
            f"teacher_topk K mismatch: cache={cache_k} contract={expected_k}"
        )
    if cost_meta.get("teacher_topk") != expected_k:
        raise ValueError(
            f"cost_meta teacher_topk mismatch vs contract: "
            f"meta={cost_meta.get('teacher_topk')!r} contract={expected_k}"
        )
    return view


def _recompute_budget_and_objective(
    *,
    allocation: Mapping[str, Any],
    cost_meta: Mapping[str, Any],
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    cost_rows_by_key: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    entries = allocation["entries"]
    used_units = 0
    total_params = 0
    weighted_bits = 0.0
    objective = 0.0
    for entry in entries:
        name = str(entry["module_name"])
        mode = str(entry["mode"])
        target = next(t for t in inventory.targets if t.module_name == name)
        cand = pool_index.candidates[(name, mode)]
        used_units += int(target.param_count) * bit_to_units(float(cand.nominal_bit))
        total_params += int(target.param_count)
        weighted_bits += int(target.param_count) * float(cand.nominal_bit)
        row = cost_rows_by_key.get((name, mode))
        if row is None:
            raise ValueError(f"Missing cost row for selected assignment {name!r}/{mode!r}")
        cost = float(row.mean_delta_kl)
        if abs(cost - float(entry["mean_delta_kl"])) > OBJECTIVE_MATCH_REL * max(
            1.0, abs(cost)
        ):
            raise ValueError(
                f"Allocation mean_delta_kl mismatch for {name}/{mode}: "
                f"entry={entry['mean_delta_kl']} cost_row={cost}"
            )
        objective += cost

    target_bit = float(allocation["target_average_bit"])
    budget_units = bit_to_units(target_bit) * total_params
    achieved = weighted_bits / float(total_params) if total_params else 0.0
    if int(allocation["used_bit_units"]) != used_units:
        raise ValueError(
            f"used_bit_units mismatch: allocation={allocation['used_bit_units']} recomputed={used_units}"
        )
    if int(allocation["budget_bit_units"]) != budget_units:
        raise ValueError(
            f"budget_bit_units mismatch: allocation={allocation['budget_bit_units']} recomputed={budget_units}"
        )
    if abs(float(allocation["achieved_average_bit"]) - achieved) > 1e-9:
        raise ValueError(
            f"achieved_average_bit mismatch: allocation={allocation['achieved_average_bit']} recomputed={achieved}"
        )
    tol = OBJECTIVE_MATCH_REL * max(1.0, abs(objective))
    if abs(float(allocation["objective_delta_kl"]) - objective) > tol:
        raise ValueError(
            f"objective_delta_kl mismatch: allocation={allocation['objective_delta_kl']} recomputed={objective}"
        )

    baseline_alloc = _require_finite_number(
        allocation, "baseline_kl_mean", label="allocation"
    )
    baseline_meta = _require_finite_number(
        cost_meta, "baseline_kl_mean", label="cost_meta"
    )
    if abs(baseline_alloc - baseline_meta) > tol:
        raise ValueError(
            f"baseline_kl_mean mismatch: allocation={baseline_alloc} cost_meta={baseline_meta}"
        )
    predicted = _require_finite_number(
        allocation, "predicted_mixed_model_kl", label="allocation"
    )
    expected_pred = float(baseline_alloc) + float(objective)
    if abs(float(predicted) - expected_pred) > tol:
        raise ValueError(
            f"predicted_mixed_model_kl mismatch: allocation={predicted} recomputed={expected_pred}"
        )
    return {
        "used_bit_units": used_units,
        "budget_bit_units": budget_units,
        "achieved_average_bit": achieved,
        "objective_delta_kl": objective,
        "total_target_parameters": total_params,
        "baseline_kl_mean": float(baseline_alloc),
        "predicted_mixed_model_kl": float(predicted),
    }


def _validate_live_modules_against_assignments(
    *,
    model: nn.Module,
    meta: Mapping[str, Any],
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    allocation: Mapping[str, Any],
) -> dict[str, Any]:
    mix = meta.get("extra_meta", {}).get("mix_bit")
    if not isinstance(mix, Mapping):
        raise ValueError("checkpoint meta missing extra_meta.mix_bit provenance")
    if mix.get("kind") != "optimal_mixed_bit":
        raise ValueError(f"Unexpected mix_bit kind={mix.get('kind')!r}")

    expected_names = [t.module_name for t in inventory.targets]
    expected_set = set(expected_names)
    live_vae = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, VAELinear)
    }
    live_names = set(live_vae)
    missing = sorted(expected_set - live_names)
    extra = sorted(live_names - expected_set)
    if missing or extra:
        raise ValueError(
            f"VAELinear module coverage mismatch vs inventory; missing={missing} extra={extra}"
        )

    alloc_by_name = {str(e["module_name"]): e for e in allocation["entries"]}
    if set(alloc_by_name) != expected_set:
        raise ValueError("allocation entries coverage mismatch vs inventory")

    prov_entries = mix.get("assignments")
    if not isinstance(prov_entries, list):
        raise ValueError("mix_bit.assignments must be a list")
    prov_by_name = {str(e["module_name"]): e for e in prov_entries}
    missing_assign = sorted(expected_set - set(prov_by_name))
    extra_assign = sorted(set(prov_by_name) - expected_set)
    if missing_assign or extra_assign:
        raise ValueError(
            f"mix_bit assignment coverage mismatch; missing={missing_assign} extra={extra_assign}"
        )

    converted = {
        str(spec["name"]): spec
        for spec in (meta.get("converted_modules") or [])
        if isinstance(spec, Mapping) and "name" in spec
    }
    if set(converted) != expected_set:
        raise ValueError(
            "converted_modules name set mismatch vs inventory: "
            f"missing={sorted(expected_set - set(converted))} "
            f"extra={sorted(set(converted) - expected_set)}"
        )

    module_reports = []
    for name in expected_names:
        module = live_vae[name]
        alloc_entry = alloc_by_name[name]
        prov = prov_by_name[name]
        mode = str(alloc_entry["mode"])
        if str(prov.get("mode_name", prov.get("mode"))) != mode:
            raise ValueError(
                f"{name}: provenance mode mismatch assignment={prov.get('mode_name')!r} "
                f"allocation={mode!r}"
            )
        cand = pool_index.candidates[(name, mode)]
        compact_sha = str(
            prov.get("compact_state_sha256") or alloc_entry.get("compact_state_sha256") or ""
        )
        if compact_sha != cand.source.compact_state_sha256:
            raise ValueError(
                f"{name}: compact source hash mismatch for mode={mode!r}: "
                f"found={compact_sha!r} candidate={cand.source.compact_state_sha256!r}"
            )
        if str(alloc_entry.get("compact_state_sha256")) != cand.source.compact_state_sha256:
            raise ValueError(
                f"{name}: allocation compact_state_sha256 mismatch for mode={mode!r}"
            )

        spec = cand.module_spec
        checks = {
            "in_features": int(module.in_features) == int(spec["in_features"]),
            "out_features": int(module.out_features) == int(spec["out_features"]),
            "has_bias": (module.bias is not None) == bool(spec.get("has_bias", False)),
            "transpose": bool(getattr(module, "transpose", False))
            == bool(spec.get("transpose", False)),
            "codebook_dim": int(module.codebook_dim) == int(spec["codebook_dim"]),
            "residual_stages": int(module.residual_stages)
            == int(spec.get("residual_stages", 1) or 1),
        }
        bad = [key for key, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"{name}: live module config mismatch for fields {bad}")
        stage_dims = [int(v) for v in getattr(module, "stage_codebook_dims", [])]
        expected_stage = [int(v) for v in spec.get("stage_codebook_dims", [])]
        if expected_stage and stage_dims != expected_stage:
            raise ValueError(
                f"{name}: stage_codebook_dims mismatch live={stage_dims} expected={expected_stage}"
            )
        if module.original_weight is not None:
            raise ValueError(f"{name}: live original_weight must be None")
        conv = converted[name]
        if bool(conv.get("has_original_weight")):
            raise ValueError(f"{name}: converted_modules reports has_original_weight=true")
        if int(conv.get("codebook_dim", -1)) != int(spec["codebook_dim"]):
            raise ValueError(f"{name}: converted_modules codebook_dim mismatch vs candidate")
        if int(conv.get("residual_stages", -1)) != int(spec.get("residual_stages", 1) or 1):
            raise ValueError(f"{name}: converted_modules residual_stages mismatch vs candidate")
        module_reports.append(
            {
                "module_name": name,
                "mode": mode,
                "compact_state_sha256": compact_sha,
                "nominal_bit": float(cand.nominal_bit),
            }
        )
    return {
        "module_count": len(module_reports),
        "modules": module_reports,
        "passed": True,
    }


def _validate_provenance_hashes(
    *,
    mix: Mapping[str, Any],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    pool_index: CandidatePoolIndex,
    allocation_path: Path,
    allocation_sha256: str,
    cost_table_sha256: str,
    cost_table_meta_sha256: str,
) -> None:
    checks = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "model_inventory_sha256": sha256_file(inventory_path),
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "cost_table_sha256": cost_table_sha256,
        "cost_table_meta_sha256": cost_table_meta_sha256,
        "allocation_sha256": allocation_sha256,
    }
    for key, expected in checks.items():
        found = mix.get(key)
        if found != expected:
            raise ValueError(
                f"checkpoint mix_bit provenance mismatch for {key}: "
                f"found={found!r} expected={expected!r}"
            )
    if Path(mix.get("allocation_path", "")).resolve() != allocation_path.resolve():
        # Path string may differ by symlink; require matching sha already checked.
        if mix.get("allocation_sha256") != allocation_sha256:
            raise ValueError("allocation_path/sha mismatch in mix_bit provenance")


def validate_save_reload_logits(
    *,
    model: nn.Module,
    mixed_model_dir: Path,
) -> dict[str, Any]:
    ref_path = mixed_model_dir / REFERENCE_LOGITS_FILENAME
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing reference logits: {ref_path}")
    payload = torch.load(ref_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid reference_logits payload in {ref_path}")
    input_ids = payload["input_ids"]
    reference = payload["logits"]
    if not torch.is_tensor(input_ids) or not torch.is_tensor(reference):
        raise ValueError("reference_logits.pt must contain tensor input_ids and logits")

    device = next(model.parameters()).device
    model.eval()
    use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config") and use_cache is not None:
        model.config.use_cache = False
    try:
        batch = input_ids.to(device)
        attention_mask = torch.ones_like(batch)
        with torch.inference_mode():
            outputs = model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits_cpu = logits.detach().cpu()
        torch.testing.assert_close(
            logits_cpu,
            reference,
            rtol=SAVE_RELOAD_RTOL,
            atol=SAVE_RELOAD_ATOL,
        )
        abs_err = (logits_cpu - reference).abs()
        rel_err = abs_err / reference.abs().clamp_min(1e-12)
        return {
            "passed": True,
            "reference_logits_path": str(ref_path.resolve()),
            "logits_shape": list(logits_cpu.shape),
            "max_abs_error": float(abs_err.max().item()),
            "max_rel_error": float(rel_err.max().item()),
            "rtol": SAVE_RELOAD_RTOL,
            "atol": SAVE_RELOAD_ATOL,
        }
    finally:
        if hasattr(model, "config") and use_cache is not None:
            model.config.use_cache = use_cache


def _measure_actual_kl(
    *,
    model: nn.Module,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    dataset_path: Path,
    dataset_manifest_path: Path,
    kl_mode: str,
    teacher_topk: int | None,
    teacher_cache: str | Path | None,
    device: str,
    access_token: str | None,
) -> dict[str, Any]:
    contract = validate_kl_mode_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
        vocab_size=None,
    )
    manifest = _read_json(dataset_manifest_path)
    _validate_dataset_manifest(
        manifest,
        resolved=resolved,
        inventory=inventory,
        dataset_path=dataset_path,
    )
    examples = _load_calibration_examples(dataset_path)
    sample_ids = np.asarray([int(ex.sample_id) for ex in examples], dtype=np.int64)
    pad_token_id = manifest.get("pad_token_id")
    if pad_token_id is None:
        raise ValueError("dataset manifest missing pad_token_id")

    teacher_view: TeacherCacheView | None = None
    teacher_model: nn.Module | None = None
    teacher_cache_sha = ""
    if contract.kl_mode == KL_MODE_TEACHER_TOPK:
        assert teacher_cache is not None
        teacher_view = load_teacher_cache_for_worker(teacher_cache)
        validate_teacher_cache_against_inputs(
            teacher_view.index,
            expected_sample_ids=[int(x) for x in sample_ids.tolist()],
            run_config_sha256=resolved.run_config_sha256,
            model_inventory_fingerprint=inventory.fingerprint_sha256,
            dataset_file_sha256=sha256_file(dataset_path),
            teacher_topk=int(contract.teacher_topk),
            vocab_size=int(teacher_view.index["vocab_size"]),
            cache_prob_dtype=str(teacher_view.index["cache_prob_dtype"]),
        )
        teacher_cache_sha = teacher_view.index_sha256
        if teacher_model is not None:
            raise ValueError("teacher_topk validation must not hold a teacher model")
    else:
        teacher_model = load_teacher_model(resolved, device=device, access_token=access_token)

    model.eval()
    model.to(device)
    ctx = CostWorkerContext(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        overlay={},
        overlay_path="",
        overlay_sha256="",
        assignments={},
        model=model,
        examples=examples,
        sample_ids=sample_ids,
        baseline_kl=np.zeros(sample_ids.shape[0], dtype=np.float64),
        cost_run_root=Path(resolved.canonical_run_root) / "validation_scratch",
        device=torch.device(device),
        batch_size=1,
        pad_token_id=int(pad_token_id),
        kl_mode=contract.kl_mode,
        metric_name=contract.metric_name,
        teacher_topk=contract.teacher_topk,
        teacher_cache_index=teacher_view,
        teacher_cache_index_sha256=teacher_cache_sha,
        teacher_model=teacher_model,
        teacher_model_id=resolved.config.model_profile.model_id,
        calibration_manifest_sha256=sha256_file(dataset_manifest_path),
        candidate_manifest_sha256=sha256_file(pool_index.manifest_path),
        run_config_sha256=resolved.run_config_sha256,
        model_inventory_sha256=inventory.fingerprint_sha256,
        baseline_mode=str(resolved.config.candidate_space.baseline_mode),
    )
    evaluated = evaluate_student_per_sample_kl(ctx)
    actual = float(np.asarray(evaluated["per_sample_kl"], dtype=np.float64).mean())
    if teacher_model is not None:
        del teacher_model
        gc.collect()
    return {
        "kl_mode": contract.kl_mode,
        "metric_name": contract.metric_name,
        "teacher_topk": contract.teacher_topk,
        "teacher_cache_index_sha256": teacher_cache_sha or None,
        "actual_mixed_model_kl": actual,
        "sample_count": int(sample_ids.size),
    }


def _run_downstream_eval(
    *,
    mixed_model: nn.Module,
    baseline_model: nn.Module,
    resolved: ResolvedRunConfig,
    device: str,
    lm_batch_size: str | int,
    access_token: str | None,
    tokenizer: Any,
) -> dict[str, Any]:
    profile = resolved.config.model_profile
    try:
        import lm_eval  # noqa: F401

        lm_eval_version = getattr(__import__("lm_eval"), "__version__", "unknown")
    except Exception:
        lm_eval_version = None

    def _eval_one(model: nn.Module, label: str) -> dict[str, Any]:
        model.eval()
        model.to(device)
        ppl_args = SimpleNamespace(
            model_path=profile.model_path,
            seqlen=DOWNSTREAM_SEQLEN,
            limit=-1,
        )
        # calculate_ppl uses model.device
        if not hasattr(model, "device"):
            try:
                model.device = next(model.parameters()).device
            except StopIteration:
                model.device = torch.device(device)
        ppl = calculate_ppl(model, ppl_args)
        lm_args = SimpleNamespace(
            model_path=profile.model_path,
            tasks=DOWNSTREAM_TASKS,
            num_fewshot=0,
            batch_size=lm_batch_size,
            lm_limit=None,
            mmlu_debug_samples=0,
            eval_log_dir=None,
            eval_run_ts=None,
        )
        lm_raw = run_lm_eval(model, tokenizer, lm_args)
        compact = {
            "wiki_ppl": ppl.get("wiki_ppl"),
            "seqlen": ppl.get("seqlen"),
            "nsamples": ppl.get("nsamples"),
            "task_metrics": (
                lm_raw.get("artifact_payload", {}).get("task_metrics")
                if isinstance(lm_raw, Mapping)
                else None
            )
            or (lm_raw.get("results") if isinstance(lm_raw, Mapping) else lm_raw),
        }
        return {
            "label": label,
            "ppl": ppl,
            "lm_eval": lm_raw if isinstance(lm_raw, Mapping) else {"raw": lm_raw},
            "compact": compact,
        }

    baseline_result = _eval_one(baseline_model, "uniform_baseline")
    mixed_result = _eval_one(mixed_model, "mixed_model")
    return {
        "lm_eval_version": lm_eval_version,
        "tasks": [t.strip() for t in DOWNSTREAM_TASKS.split(",") if t.strip()],
        "num_fewshot": 0,
        "limit": None,
        "device": device,
        "batch_size": lm_batch_size,
        "seqlen": DOWNSTREAM_SEQLEN,
        "uniform_baseline": baseline_result,
        "mixed_model": mixed_result,
    }


def _render_validation_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mixed-bit validation report",
        "",
        f"- passed: `{report.get('passed')}`",
        f"- mixed_model_dir: `{report.get('mixed_model_dir')}`",
        f"- kl_mode: `{report.get('kl', {}).get('kl_mode')}`",
        "",
        "## Budget",
        "",
        f"- used_bit_units: `{report.get('budget', {}).get('used_bit_units')}`",
        f"- budget_bit_units: `{report.get('budget', {}).get('budget_bit_units')}`",
        f"- achieved_average_bit: `{report.get('budget', {}).get('achieved_average_bit')}`",
        "",
        "## KL",
        "",
        f"- baseline_kl_mean: `{report.get('kl', {}).get('baseline_kl_mean')}`",
        f"- predicted_mixed_model_kl: `{report.get('kl', {}).get('predicted_mixed_model_kl')}`",
        f"- actual_mixed_model_kl: `{report.get('kl', {}).get('actual_mixed_model_kl')}`",
        f"- absolute_gap: `{report.get('kl', {}).get('absolute_gap')}`",
        f"- relative_gap: `{report.get('kl', {}).get('relative_gap')}`",
        "",
        "## Save/reload",
        "",
        f"- passed: `{report.get('save_reload', {}).get('passed')}`",
        f"- max_abs_error: `{report.get('save_reload', {}).get('max_abs_error')}`",
        f"- max_rel_error: `{report.get('save_reload', {}).get('max_rel_error')}`",
        "",
    ]
    if "downstream" in report:
        lines.extend(
            [
                "## Downstream",
                "",
                f"- skipped: `{report['downstream'].get('skipped', False)}`",
            ]
        )
        if not report["downstream"].get("skipped"):
            lines.append(
                f"- tasks: `{','.join(report['downstream'].get('tasks', []))}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def validate_mixed_model(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    pool_index: CandidatePoolIndex,
    allocation_path: str,
    cost_table_path: str,
    cost_table_meta_path: str,
    baseline_overlay_path: str,
    mixed_model_dir: str,
    dataset_path: str,
    dataset_manifest_path: str,
    teacher_cache: str | None = None,
    device: str = "cuda",
    skip_downstream_eval: bool = False,
    lm_batch_size: str | int = "auto",
    access_token: str | None = None,
    allow_suboptimal: bool = False,
) -> dict[str, Any]:
    """Validate an assembled mixed-bit checkpoint without recovery training."""
    alloc_path = Path(allocation_path)
    if not alloc_path.is_file():
        raise FileNotFoundError(f"Missing allocation file: {alloc_path}")
    allocation_sha256 = sha256_file(alloc_path)
    allocation = _read_json(alloc_path)
    # Structural allocation validation (coverage / budget / objective consistency).
    _validate_allocation_payload(
        allocation,
        allocation_path=alloc_path,
        allocation_sha256=allocation_sha256,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        allow_suboptimal=allow_suboptimal,
    )

    cost_meta = _read_json(Path(cost_table_meta_path))
    cost_table_sha = sha256_file(cost_table_path)
    cost_meta_sha = sha256_file(cost_table_meta_path)
    if allocation.get("cost_table_sha256") != cost_table_sha:
        raise ValueError(
            "allocation cost_table_sha256 mismatch against provided cost table file"
        )
    if allocation.get("cost_table_meta_sha256") != cost_meta_sha:
        raise ValueError(
            "allocation cost_table_meta_sha256 mismatch against provided cost table meta"
        )
    if cost_meta.get("cost_table_sha256") != cost_table_sha:
        raise ValueError("cost_table_meta.cost_table_sha256 mismatch against cost table file")

    # Metric contract comes from allocation/cost meta, not CLI retyping.
    contract = resolve_metric_contract(
        kl_mode=str(allocation["kl_mode"]),
        teacher_topk=allocation.get("teacher_topk"),
    )
    if cost_meta.get("kl_mode") != contract.kl_mode:
        raise ValueError(
            f"cost meta kl_mode mismatch: meta={cost_meta.get('kl_mode')!r} "
            f"allocation={contract.kl_mode!r}"
        )
    if cost_meta.get("teacher_topk") != contract.teacher_topk:
        raise ValueError(
            f"cost meta teacher_topk mismatch: meta={cost_meta.get('teacher_topk')!r} "
            f"allocation={contract.teacher_topk!r}"
        )
    # Reject incompatible cache args for the recorded contract.
    validate_kl_mode_arguments(
        kl_mode=contract.kl_mode,
        teacher_topk=contract.teacher_topk,
        teacher_cache=teacher_cache,
        vocab_size=None,
    )
    overlay_info = _validate_baseline_overlay_against_cost_meta(
        baseline_overlay_path, cost_meta
    )
    teacher_cache_view = _validate_teacher_cache_against_cost_meta(
        teacher_cache=teacher_cache,
        cost_meta=cost_meta,
        contract=contract,
    )

    cost_rows = load_cost_table_for_solve(
        cost_table_path,
        cost_table_meta_path,
        inventory=inventory,
        candidate_space=resolved.config.candidate_space,
        expected_hashes={
            "run_config_sha256": resolved.run_config_sha256,
            "model_inventory_sha256": inventory.fingerprint_sha256,
            "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
            "candidate_space_sha256": resolved.candidate_space_sha256,
        },
    )
    cost_by_key = {(r.module_name, r.mode): r for r in cost_rows}
    budget = _recompute_budget_and_objective(
        allocation=allocation,
        cost_meta=cost_meta,
        inventory=inventory,
        pool_index=pool_index,
        cost_rows_by_key=cost_by_key,
    )

    out_dir = Path(mixed_model_dir)
    if not (out_dir / META_FILENAME).is_file() or not (out_dir / STATE_DICT_FILENAME).is_file():
        raise FileNotFoundError(f"Incomplete mixed model checkpoint under {out_dir}")

    # Reject forbidden meta before reload — has_original_weight=true would otherwise
    # request missing original_weight tensors during strict load.
    meta_pre = _read_json(out_dir / META_FILENAME)
    state_probe = torch.load(
        out_dir / STATE_DICT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    try:
        state_keys = list(state_probe.keys()) if isinstance(state_probe, Mapping) else []
        _reject_forbidden_checkpoint_payload(meta=meta_pre, state_keys=state_keys)
    finally:
        del state_probe
        gc.collect()

    # Verify tokenizer fingerprint before any KL/downstream work proceeds.
    mix_pre = meta_pre.get("extra_meta", {}).get("mix_bit")
    if not isinstance(mix_pre, Mapping):
        raise ValueError("checkpoint missing extra_meta.mix_bit")
    tokenizer_report = _verify_final_tokenizer_fingerprint(
        final_dir=out_dir,
        mix=mix_pre,
    )
    final_tokenizer = _load_tokenizer_from_final_dir(out_dir)

    model, meta, _load_result = load_model_checkpoint(
        str(out_dir),
        access_token=access_token,
        map_location=device,
        strict=True,
    )
    try:
        # Re-check after load for live module original_weight / adapter leaks.
        _reject_forbidden_checkpoint_payload(
            meta=meta, state_keys=list(model.state_dict().keys())
        )
        mix = meta.get("extra_meta", {}).get("mix_bit")
        if not isinstance(mix, Mapping):
            raise ValueError("checkpoint missing extra_meta.mix_bit")
        _validate_provenance_hashes(
            mix=mix,
            resolved=resolved,
            inventory=inventory,
            inventory_path=inventory_path,
            pool_index=pool_index,
            allocation_path=alloc_path,
            allocation_sha256=allocation_sha256,
            cost_table_sha256=cost_table_sha,
            cost_table_meta_sha256=cost_meta_sha,
        )
        structural = _validate_live_modules_against_assignments(
            model=model,
            meta=meta,
            inventory=inventory,
            pool_index=pool_index,
            allocation=allocation,
        )
        save_reload = validate_save_reload_logits(model=model, mixed_model_dir=out_dir)

        kl_actual = _measure_actual_kl(
            model=model,
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            dataset_path=Path(dataset_path),
            dataset_manifest_path=Path(dataset_manifest_path),
            kl_mode=contract.kl_mode,
            teacher_topk=contract.teacher_topk,
            teacher_cache=teacher_cache,
            device=device,
            access_token=access_token,
        )
        baseline_kl = float(budget["baseline_kl_mean"])
        predicted = float(budget["predicted_mixed_model_kl"])
        actual = float(kl_actual["actual_mixed_model_kl"])
        absolute_gap = actual - predicted
        if abs(predicted) < RELATIVE_GAP_EPS:
            relative_gap = None
        else:
            relative_gap = absolute_gap / abs(predicted)
        cache_sha = kl_actual.get("teacher_cache_index_sha256")
        if teacher_cache_view is not None:
            cache_sha = teacher_cache_view.index_sha256
        kl_report = {
            "kl_mode": contract.kl_mode,
            "metric_name": contract.metric_name,
            "teacher_topk": contract.teacher_topk,
            "teacher_cache_index_sha256": cache_sha,
            "baseline_kl_mean": baseline_kl,
            "predicted_mixed_model_kl": predicted,
            "actual_mixed_model_kl": actual,
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
            "sample_count": kl_actual["sample_count"],
            "baseline_overlay_sha256": overlay_info["baseline_overlay_sha256"],
        }

        downstream: dict[str, Any]
        if skip_downstream_eval:
            downstream = {"skipped": True}
        else:
            baseline_assignments = build_uniform_assignments(
                pool_index, str(resolved.config.candidate_space.baseline_mode)
            )
            baseline_model = build_model_from_assignments(
                resolved=resolved,
                inventory=inventory,
                pool_index=pool_index,
                assignments=baseline_assignments,
                device=device,
            )
            try:
                downstream = _run_downstream_eval(
                    mixed_model=model,
                    baseline_model=baseline_model,
                    resolved=resolved,
                    device=device,
                    lm_batch_size=lm_batch_size,
                    access_token=access_token,
                    tokenizer=final_tokenizer,
                )
                downstream["skipped"] = False
                downstream["baseline_overlay_path"] = overlay_info["baseline_overlay_path"]
                downstream["baseline_overlay_sha256"] = overlay_info["baseline_overlay_sha256"]
            finally:
                del baseline_model
                gc.collect()

        report: dict[str, Any] = {
            "kind": "mix_bit_validation",
            "passed": True,
            "mixed_model_dir": str(out_dir.resolve()),
            "allocation_path": str(alloc_path.resolve()),
            "allocation_sha256": allocation_sha256,
            "cost_table_sha256": cost_table_sha,
            "cost_table_meta_sha256": cost_meta_sha,
            "structural": structural,
            "budget": budget,
            "objective": {
                "objective_delta_kl": budget["objective_delta_kl"],
                "predicted_mixed_model_kl": predicted,
            },
            "save_reload": save_reload,
            "kl": kl_report,
            "tokenizer": tokenizer_report,
            "downstream": downstream,
            "recovery_training": False,
        }
        validation_json = out_dir / "validation.json"
        validation_md = out_dir / "validation.md"
        _write_json_atomic(validation_json, report)
        _write_text_atomic(validation_md, _render_validation_md(report))
        report["validation_json"] = str(validation_json.resolve())
        report["validation_md"] = str(validation_md.resolve())
        return report
    finally:
        del model
        del final_tokenizer
        gc.collect()

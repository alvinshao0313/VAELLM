from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from mix_bit.model_inventory import ModelInventory, TargetLinearSpec
from mix_bit.schema import CandidateSpaceConfig, sha256_file

BIT_UNIT_DENOMINATOR = 2
BIT_CONVERSION_TOL = 1e-12
BINARY_TOL = 1e-6
OBJECTIVE_MATCH_REL = 1e-9


@dataclass(frozen=True)
class CostRow:
    module_name: str
    category: str
    module_suffix: str
    block_index: int
    in_features: int
    out_features: int
    has_bias: bool
    param_count: int
    mode: str
    nominal_bit: float
    mean_delta_kl: float
    kl_mode: str = ""
    metric_name: str = ""
    teacher_topk: int | None = None
    run_config_sha256: str = ""
    model_inventory_sha256: str = ""
    candidate_manifest_sha256: str = ""
    candidate_space_sha256: str = ""
    compact_state_sha256: str = ""
    per_sample_sha256: str = ""


@dataclass(frozen=True)
class AllocationEntry:
    module_name: str
    category: str
    module_suffix: str
    block_index: int
    in_features: int
    out_features: int
    has_bias: bool
    param_count: int
    mode: str
    nominal_bit: float
    mean_delta_kl: float
    compact_state_sha256: str = ""
    per_sample_sha256: str = ""


@dataclass(frozen=True)
class AllocationResult:
    entries: tuple[AllocationEntry, ...]
    objective_delta_kl: float
    objective_scale: float
    is_globally_optimal: bool
    allow_suboptimal: bool
    used_bit_units: int
    budget_bit_units: int
    achieved_average_bit: float
    total_target_parameters: int
    target_average_bit: float
    bit_unit_denominator: int
    baseline_mode: str
    baseline_objective_delta_kl: float
    predicted_mixed_model_kl: float | None
    solver_name: str
    solver_status: int
    solver_message: str
    scipy_version: str
    time_limit_sec: float | None


def bit_to_units(nominal_bit: float) -> int:
    scaled = float(nominal_bit) * float(BIT_UNIT_DENOMINATOR)
    units = int(round(scaled))
    if abs(scaled - units) > BIT_CONVERSION_TOL:
        raise ValueError(
            f"nominal_bit={nominal_bit!r} is not representable in half-bit units "
            f"(conversion error {abs(scaled - units)} > {BIT_CONVERSION_TOL})"
        )
    return units


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _cost_row_from_mapping(raw: Mapping[str, Any]) -> CostRow:
    teacher_topk = raw.get("teacher_topk")
    if teacher_topk is not None:
        teacher_topk = int(teacher_topk)
    compact_sha = raw.get("compact_state_sha256") or raw.get("source_compact_state_sha256") or ""
    return CostRow(
        module_name=str(raw["module_name"]),
        category=str(raw["category"]),
        module_suffix=str(raw["module_suffix"]),
        block_index=int(raw["block_index"]),
        in_features=int(raw.get("in_features", 0)),
        out_features=int(raw.get("out_features", 0)),
        has_bias=bool(raw.get("has_bias", False)),
        param_count=int(raw["param_count"]),
        mode=str(raw["mode"]),
        nominal_bit=float(raw["nominal_bit"]),
        mean_delta_kl=float(raw["mean_delta_kl"]),
        kl_mode=str(raw.get("kl_mode", "")),
        metric_name=str(raw.get("metric_name", "")),
        teacher_topk=teacher_topk,
        run_config_sha256=str(raw.get("run_config_sha256", "")),
        model_inventory_sha256=str(raw.get("model_inventory_sha256", "")),
        candidate_manifest_sha256=str(raw.get("candidate_manifest_sha256", "")),
        candidate_space_sha256=str(raw.get("candidate_space_sha256", "")),
        compact_state_sha256=str(compact_sha),
        per_sample_sha256=str(raw.get("per_sample_sha256", "")),
    )


def _assert_row_matches_target(row: CostRow, target: TargetLinearSpec) -> None:
    checks = {
        "category": target.category,
        "module_suffix": target.module_suffix,
        "block_index": int(target.block_index),
        "param_count": int(target.param_count),
    }
    for field, expected in checks.items():
        found = getattr(row, field)
        if field in ("block_index", "param_count"):
            found = int(found)
        if found != expected:
            raise ValueError(
                f"Row inventory metadata mismatch for {(row.module_name, row.mode)}: "
                f"{field} row={found!r} inventory={expected!r}"
            )
    if row.in_features and int(row.in_features) != int(target.in_features):
        raise ValueError(
            f"Row inventory metadata mismatch for {(row.module_name, row.mode)}: "
            f"in_features row={row.in_features!r} inventory={target.in_features!r}"
        )
    if row.out_features and int(row.out_features) != int(target.out_features):
        raise ValueError(
            f"Row inventory metadata mismatch for {(row.module_name, row.mode)}: "
            f"out_features row={row.out_features!r} inventory={target.out_features!r}"
        )


def validate_cost_rows_for_solve(
    rows: Sequence[CostRow],
    *,
    inventory: ModelInventory,
    candidate_space: CandidateSpaceConfig,
    target_average_bit: float,
) -> dict[tuple[str, str], CostRow]:
    modes = list(candidate_space.modes)
    mode_names = [m.name for m in modes]
    if len(mode_names) != len(set(mode_names)):
        raise ValueError("candidate_space has duplicate mode names")
    if candidate_space.baseline_mode not in set(mode_names):
        raise ValueError(
            f"baseline_mode {candidate_space.baseline_mode!r} missing from candidate space"
        )
    baseline = next(m for m in modes if m.name == candidate_space.baseline_mode)
    if abs(float(baseline.nominal_bit) - float(target_average_bit)) > BIT_CONVERSION_TOL:
        raise ValueError(
            f"baseline nominal_bit={baseline.nominal_bit} must equal "
            f"target_average_bit={target_average_bit}"
        )
    # Ensure every mode bit converts exactly.
    for mode in modes:
        bit_to_units(mode.nominal_bit)
    bit_to_units(target_average_bit)

    expected_keys = {
        (t.module_name, m.name) for t in inventory.targets for m in modes
    }
    seen: dict[tuple[str, str], CostRow] = {}
    for row in rows:
        key = (row.module_name, row.mode)
        if key in seen:
            raise ValueError(f"duplicate module-mode row for {key}")
        if key not in expected_keys:
            raise ValueError(f"unexpected cost row for {key}")
        if not math.isfinite(float(row.mean_delta_kl)):
            raise ValueError(f"non-finite cost for {key}: {row.mean_delta_kl}")
        target_map = {t.module_name: t for t in inventory.targets}
        target = target_map.get(row.module_name)
        if target is None:
            raise ValueError(f"cost row module not in inventory: {row.module_name}")
        _assert_row_matches_target(row, target)
        mode_bit = next(m.nominal_bit for m in modes if m.name == row.mode)
        if abs(float(row.nominal_bit) - float(mode_bit)) > BIT_CONVERSION_TOL:
            raise ValueError(
                f"row nominal_bit mismatch for {key}: "
                f"row={row.nominal_bit} candidate_space={mode_bit}"
            )
        seen[key] = row

    if set(seen) != expected_keys:
        missing = sorted(expected_keys - set(seen))
        raise ValueError(
            f"incomplete cost table: expected L * R = "
            f"{len(inventory.targets) * len(modes)} unique rows, "
            f"got {len(seen)}; missing={missing[:5]}"
        )

    # Every module must have the same complete ordered mode set.
    for target in inventory.targets:
        module_modes = [m for m in mode_names if (target.module_name, m) in seen]
        if module_modes != mode_names:
            raise ValueError(
                f"module {target.module_name} mode set mismatch: "
                f"got={module_modes} expected={mode_names}"
            )
    return seen


def parse_exclude_modes(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    parts = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if len(parts) != len(set(parts)):
        raise ValueError(f"duplicate exclude_modes: {parts}")
    return parts


def with_excluded_modes(
    space: CandidateSpaceConfig,
    rows: Sequence[CostRow],
    exclude_modes: Sequence[str],
) -> tuple[CandidateSpaceConfig, list[CostRow]]:
    """Drop named modes from the MILP while keeping the loaded cost table intact.

    The original cost table must still be complete for the unfiltered candidate
    space (provenance / hash checks happen before this call). Exclusion only
    shrinks the search space.
    """
    exclude = tuple(str(name).strip() for name in exclude_modes if str(name).strip())
    if not exclude:
        return space, list(rows)
    if len(exclude) != len(set(exclude)):
        raise ValueError(f"duplicate exclude_modes: {exclude}")
    names = {mode.name for mode in space.modes}
    unknown = [name for name in exclude if name not in names]
    if unknown:
        raise ValueError(f"exclude_modes not in candidate space: {unknown}")
    excluded = set(exclude)
    if space.baseline_mode in excluded:
        raise ValueError(
            f"cannot exclude baseline_mode {space.baseline_mode!r}"
        )
    kept_modes = tuple(mode for mode in space.modes if mode.name not in excluded)
    if len(kept_modes) < 2:
        raise ValueError("candidate space must keep at least two modes after exclusion")
    filtered_space = CandidateSpaceConfig(
        candidate_space_id=space.candidate_space_id,
        baseline_mode=space.baseline_mode,
        target_average_bit=space.target_average_bit,
        modes=kept_modes,
    )
    filtered_rows = [row for row in rows if row.mode not in excluded]
    return filtered_space, filtered_rows


def load_cost_table_for_solve(
    cost_table_path: str | Path,
    cost_table_meta_path: str | Path,
    *,
    inventory: ModelInventory,
    candidate_space: CandidateSpaceConfig,
    expected_hashes: Mapping[str, str],
) -> list[CostRow]:
    table_path = Path(cost_table_path)
    meta_path = Path(cost_table_meta_path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing cost table: {table_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing cost table meta: {meta_path}")

    meta = _read_json(meta_path)
    table_sha = sha256_file(table_path)
    meta_sha = meta.get("cost_table_sha256")
    if meta_sha != table_sha:
        raise ValueError(
            f"cost_table_sha256 mismatch: meta={meta_sha!r} file={table_sha!r}"
        )

    hash_keys = (
        "run_config_sha256",
        "model_inventory_sha256",
        "candidate_manifest_sha256",
        "candidate_space_sha256",
    )
    for key in hash_keys:
        if key not in expected_hashes:
            raise ValueError(f"expected_hashes missing required key {key}")
        found = meta.get(key)
        if found != expected_hashes[key]:
            raise ValueError(
                f"cost table meta provenance mismatch for {key}: "
                f"meta={found!r} expected={expected_hashes[key]!r}"
            )

    rows: list[CostRow] = []
    with open(table_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {table_path}:{line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Cost row at {table_path}:{line_no} must be an object")
            rows.append(_cost_row_from_mapping(raw))

    validate_cost_rows_for_solve(
        rows,
        inventory=inventory,
        candidate_space=candidate_space,
        target_average_bit=float(candidate_space.target_average_bit),
    )
    return rows


def _flatten_index(module_idx: int, mode_idx: int, n_modes: int) -> int:
    return module_idx * n_modes + mode_idx


def _verify_solution(
    *,
    x: np.ndarray,
    costs: np.ndarray,
    bit_units: np.ndarray,
    param_counts: np.ndarray,
    n_modules: int,
    n_modes: int,
    target_units: int,
    scaled_objective: float,
    objective_scale: float,
    baseline_mode_idx: int,
    mode_bits: np.ndarray,
) -> tuple[np.ndarray, float, int, int, float]:
    if x.shape != (n_modules * n_modes,):
        raise ValueError(f"unexpected solution shape {x.shape}")

    rounded = np.empty_like(x, dtype=np.int64)
    for i, val in enumerate(x):
        if abs(float(val) - 0.0) <= BINARY_TOL:
            rounded[i] = 0
        elif abs(float(val) - 1.0) <= BINARY_TOL:
            rounded[i] = 1
        else:
            raise ValueError(
                f"solver returned fractional/ambiguous binary variable at index {i}: "
                f"value={val} (require within {BINARY_TOL} of 0 or 1)"
            )

    selected = []
    for l in range(n_modules):
        chosen = [
            r
            for r in range(n_modes)
            if int(rounded[_flatten_index(l, r, n_modes)]) == 1
        ]
        if len(chosen) != 1:
            raise ValueError(
                f"module index {l} must select exactly one mode, got {chosen}"
            )
        selected.append(chosen[0])
    selected_arr = np.asarray(selected, dtype=np.int64)

    unscaled_obj = 0.0
    used_units = 0
    weighted_bits = 0.0
    total_params = int(param_counts.sum())
    for l, r in enumerate(selected_arr):
        idx = _flatten_index(l, int(r), n_modes)
        unscaled_obj += float(costs[idx])
        used_units += int(param_counts[l]) * int(bit_units[r])
        weighted_bits += float(param_counts[l]) * float(mode_bits[r])

    budget_units = int(target_units) * int(total_params)
    if used_units > budget_units:
        raise ValueError(
            f"budget violation: used_bit_units={used_units} > budget_bit_units={budget_units}"
        )

    # Solver reports scaled objective; compare after unscaling.
    tol = OBJECTIVE_MATCH_REL * max(1.0, abs(unscaled_obj))
    if objective_scale == 0.0:
        raise ValueError("objective_scale must be positive")
    solver_unscaled = float(scaled_objective) / float(objective_scale)
    if abs(solver_unscaled - unscaled_obj) > tol:
        raise ValueError(
            f"solver objective disagreement: recomputed={unscaled_obj} "
            f"solver_unscaled={solver_unscaled} tol={tol}"
        )

    baseline_obj = 0.0
    for l in range(n_modules):
        idx = _flatten_index(l, baseline_mode_idx, n_modes)
        baseline_obj += float(costs[idx])
    if unscaled_obj > baseline_obj + tol:
        raise ValueError(
            f"claimed optimum worse than uniform baseline: "
            f"obj={unscaled_obj} baseline={baseline_obj}"
        )

    achieved = weighted_bits / float(total_params) if total_params else 0.0
    return selected_arr, unscaled_obj, used_units, budget_units, achieved


def solve_mixed_bit_allocation(
    rows: Sequence[CostRow],
    *,
    inventory: ModelInventory,
    candidate_space: CandidateSpaceConfig,
    target_average_bit: float,
    time_limit_sec: float | None = None,
    allow_suboptimal: bool = False,
) -> AllocationResult:
    row_map = validate_cost_rows_for_solve(
        rows,
        inventory=inventory,
        candidate_space=candidate_space,
        target_average_bit=target_average_bit,
    )
    modes = list(candidate_space.modes)
    n_modes = len(modes)
    targets = list(inventory.targets)
    n_modules = len(targets)
    if n_modules == 0:
        raise ValueError("inventory has no target modules")

    mode_bits = np.asarray([float(m.nominal_bit) for m in modes], dtype=np.float64)
    bit_units = np.asarray([bit_to_units(b) for b in mode_bits], dtype=np.int64)
    target_units = bit_to_units(target_average_bit)
    param_counts = np.asarray([int(t.param_count) for t in targets], dtype=np.int64)
    total_params = int(param_counts.sum())
    if total_params != int(inventory.total_target_parameters):
        raise ValueError(
            f"inventory.total_target_parameters={inventory.total_target_parameters} "
            f"!= sum(param_count)={total_params}"
        )

    baseline_mode_idx = next(
        i for i, m in enumerate(modes) if m.name == candidate_space.baseline_mode
    )

    n_vars = n_modules * n_modes
    costs = np.zeros(n_vars, dtype=np.float64)
    for l, target in enumerate(targets):
        for r, mode in enumerate(modes):
            costs[_flatten_index(l, r, n_modes)] = float(
                row_map[(target.module_name, mode.name)].mean_delta_kl
            )

    max_abs = float(np.max(np.abs(costs))) if costs.size else 0.0
    objective_scale = 1.0 if max_abs == 0.0 else 1.0 / max_abs
    solver_c = costs * objective_scale

    # One equality per module: sum_r x_l,r = 1
    a_eq = np.zeros((n_modules, n_vars), dtype=np.float64)
    for l in range(n_modules):
        for r in range(n_modes):
            a_eq[l, _flatten_index(l, r, n_modes)] = 1.0
    eq_constraint = LinearConstraint(a_eq, lb=np.ones(n_modules), ub=np.ones(n_modules))

    # Integer budget: sum N_l * u_r * x <= u_target * sum N_l
    a_budget = np.zeros(n_vars, dtype=np.float64)
    for l in range(n_modules):
        for r in range(n_modes):
            a_budget[_flatten_index(l, r, n_modes)] = float(param_counts[l] * bit_units[r])
    budget_rhs = float(target_units * total_params)
    budget_constraint = LinearConstraint(a_budget, lb=-np.inf, ub=budget_rhs)

    options: dict[str, Any] = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit_sec is not None:
        options["time_limit"] = float(time_limit_sec)

    result = milp(
        c=solver_c,
        integrality=np.ones(n_vars, dtype=np.int64),
        bounds=Bounds(0, 1),
        constraints=[eq_constraint, budget_constraint],
        options=options,
    )

    # SciPy HiGHS: success True means optimal for default; time-limited feasible
    # solutions may set success False or a non-optimal status.
    status = int(getattr(result, "status", -1))
    message = str(getattr(result, "message", ""))
    x = getattr(result, "x", None)
    fun = getattr(result, "fun", None)

    globally_optimal = bool(result.success) and status == 0
    if x is None or fun is None:
        raise ValueError(
            f"MILP failed without a solution: success={result.success} "
            f"status={status} message={message}"
        )

    if not globally_optimal:
        if not allow_suboptimal:
            raise ValueError(
                "MILP did not prove global optimality "
                f"(success={result.success}, status={status}, message={message}). "
                "Pass allow_suboptimal=True / --allow_suboptimal to accept a "
                "time-limited feasible solution."
            )
        # Still require a usable binary feasible point.
        if not np.all(np.isfinite(x)):
            raise ValueError(f"MILP returned non-finite solution: {message}")

    selected, unscaled_obj, used_units, budget_units, achieved = _verify_solution(
        x=np.asarray(x, dtype=np.float64),
        costs=costs,
        bit_units=bit_units,
        param_counts=param_counts,
        n_modules=n_modules,
        n_modes=n_modes,
        target_units=target_units,
        scaled_objective=float(fun),
        objective_scale=objective_scale,
        baseline_mode_idx=baseline_mode_idx,
        mode_bits=mode_bits,
    )

    entries: list[AllocationEntry] = []
    for l, target in enumerate(targets):
        r = int(selected[l])
        mode = modes[r]
        row = row_map[(target.module_name, mode.name)]
        entries.append(
            AllocationEntry(
                module_name=target.module_name,
                category=target.category,
                module_suffix=target.module_suffix,
                block_index=target.block_index,
                in_features=target.in_features,
                out_features=target.out_features,
                has_bias=target.has_bias,
                param_count=target.param_count,
                mode=mode.name,
                nominal_bit=mode.nominal_bit,
                mean_delta_kl=float(row.mean_delta_kl),
                compact_state_sha256=row.compact_state_sha256,
                per_sample_sha256=row.per_sample_sha256,
            )
        )

    return AllocationResult(
        entries=tuple(entries),
        objective_delta_kl=float(unscaled_obj),
        objective_scale=float(objective_scale),
        is_globally_optimal=bool(globally_optimal),
        allow_suboptimal=bool(allow_suboptimal),
        used_bit_units=int(used_units),
        budget_bit_units=int(budget_units),
        achieved_average_bit=float(achieved),
        total_target_parameters=int(total_params),
        target_average_bit=float(target_average_bit),
        bit_unit_denominator=int(BIT_UNIT_DENOMINATOR),
        baseline_mode=candidate_space.baseline_mode,
        baseline_objective_delta_kl=0.0,
        predicted_mixed_model_kl=None,
        solver_name="scipy.optimize.milp",
        solver_status=status,
        solver_message=message,
        scipy_version=str(scipy.__version__),
        time_limit_sec=time_limit_sec,
    )


def allocation_result_to_dict(
    result: AllocationResult,
    *,
    model_id: str,
    run_id: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_kl = provenance.get("baseline_kl_mean")
    predicted = None
    if baseline_kl is not None:
        predicted = float(baseline_kl) + float(result.objective_delta_kl)
    payload = {
        "kind": "mix_bit_allocation",
        "model_id": model_id,
        "run_id": run_id,
        "solver_name": result.solver_name,
        "solver_status": result.solver_status,
        "solver_message": result.solver_message,
        "scipy_version": result.scipy_version,
        "is_globally_optimal": result.is_globally_optimal,
        "allow_suboptimal": result.allow_suboptimal,
        "time_limit_sec": result.time_limit_sec,
        "objective_scale": result.objective_scale,
        "objective_delta_kl": result.objective_delta_kl,
        "baseline_mode": result.baseline_mode,
        "baseline_objective_delta_kl": result.baseline_objective_delta_kl,
        "baseline_kl_mean": baseline_kl,
        "predicted_mixed_model_kl": (
            predicted if predicted is not None else result.predicted_mixed_model_kl
        ),
        "kl_mode": provenance.get("kl_mode"),
        "metric_name": provenance.get("metric_name"),
        "teacher_topk": provenance.get("teacher_topk"),
        "target_average_bit": result.target_average_bit,
        "bit_unit_denominator": result.bit_unit_denominator,
        "used_bit_units": result.used_bit_units,
        "budget_bit_units": result.budget_bit_units,
        "budget_slack_bit_units": result.budget_bit_units - result.used_bit_units,
        "achieved_average_bit": result.achieved_average_bit,
        "total_target_parameters": result.total_target_parameters,
        "run_config_sha256": provenance.get("run_config_sha256"),
        "model_inventory_sha256": provenance.get("model_inventory_sha256"),
        "candidate_manifest_sha256": provenance.get("candidate_manifest_sha256"),
        "candidate_space_sha256": provenance.get("candidate_space_sha256"),
        "cost_table_sha256": provenance.get("cost_table_sha256"),
        "cost_table_meta_sha256": provenance.get("cost_table_meta_sha256"),
        "entries": [asdict(e) for e in result.entries],
    }
    excluded = provenance.get("excluded_modes")
    if excluded:
        payload["excluded_modes"] = list(excluded)
    return payload


def _build_allocation_markdown(payload: Mapping[str, Any]) -> str:
    entries = list(payload["entries"])
    mode_counts: dict[str, int] = {}
    mode_params: dict[str, int] = {}
    cat_bits_num: dict[str, float] = {}
    cat_bits_den: dict[str, float] = {}
    cat_obj: dict[str, float] = {}
    total_params = float(payload["total_target_parameters"]) or 1.0
    for entry in entries:
        mode = str(entry["mode"])
        cat = str(entry["category"])
        n = int(entry["param_count"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        mode_params[mode] = mode_params.get(mode, 0) + n
        cat_bits_num[cat] = cat_bits_num.get(cat, 0.0) + n * float(entry["nominal_bit"])
        cat_bits_den[cat] = cat_bits_den.get(cat, 0.0) + n
        cat_obj[cat] = cat_obj.get(cat, 0.0) + float(entry["mean_delta_kl"])

    lines = [
        "# Mixed-bit allocation summary",
        "",
        f"- is_globally_optimal: {payload['is_globally_optimal']}",
        f"- allow_suboptimal: {payload['allow_suboptimal']}",
        f"- solver_status: {payload['solver_status']}",
        f"- solver_message: {payload['solver_message']}",
        f"- objective_delta_kl: {payload['objective_delta_kl']}",
        f"- achieved_average_bit: {payload['achieved_average_bit']}",
        f"- target_average_bit: {payload['target_average_bit']}",
    ]
    excluded = payload.get("excluded_modes")
    if excluded:
        lines.append(f"- excluded_modes: {', '.join(str(m) for m in excluded)}")
    lines.extend(
        [
            f"- used_bit_units: {payload['used_bit_units']}",
            f"- budget_bit_units: {payload['budget_bit_units']}",
            f"- budget_slack_bit_units: {payload['budget_slack_bit_units']}",
            "",
            "## Unweighted mode counts",
        ]
    )
    for mode, count in sorted(mode_counts.items()):
        lines.append(f"- {mode}: {count}")
    lines.append("")
    lines.append("## Parameter-weighted mode shares")
    for mode, params in sorted(mode_params.items()):
        lines.append(f"- {mode}: {params / total_params:.6f}")
    lines.append("")
    lines.append("## Category-wise bit averages")
    for cat in sorted(cat_bits_num):
        avg = cat_bits_num[cat] / cat_bits_den[cat] if cat_bits_den[cat] else 0.0
        lines.append(f"- {cat}: {avg:.6f}")
    lines.append("")
    lines.append("## Objective contribution by category")
    for cat, obj in sorted(cat_obj.items()):
        lines.append(f"- {cat}: {obj}")
    lines.append("")
    return "\n".join(lines)


def write_allocation_outputs(
    result: AllocationResult,
    *,
    output_dir: str | Path,
    model_id: str,
    run_id: str,
    provenance: Mapping[str, Any],
    stem: str = "optimal_2bit",
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = allocation_result_to_dict(
        result, model_id=model_id, run_id=run_id, provenance=provenance
    )
    if payload["predicted_mixed_model_kl"] is None and provenance.get("baseline_kl_mean") is not None:
        payload["predicted_mixed_model_kl"] = (
            float(provenance["baseline_kl_mean"]) + float(result.objective_delta_kl)
        )

    json_path = out / f"{stem}.json"
    csv_path = out / f"{stem}.csv"
    md_path = out / f"{stem}_summary.md"

    _write_json_atomic(json_path, payload)

    fieldnames = [
        "module_name",
        "category",
        "module_suffix",
        "block_index",
        "in_features",
        "out_features",
        "has_bias",
        "param_count",
        "mode",
        "nominal_bit",
        "mean_delta_kl",
        "compact_state_sha256",
        "per_sample_sha256",
    ]
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(csv_tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in payload["entries"]:
            writer.writerow({k: entry.get(k) for k in fieldnames})
    os.replace(csv_tmp, csv_path)

    _write_text_atomic(md_path, _build_allocation_markdown(payload))
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
    }


def derive_allocation_dir(cost_table_path: str | Path) -> Path:
    """Derive <run_root>/allocation/<kl-run-name> from a cost table path."""
    path = Path(cost_table_path).resolve()
    # .../runs/<run_id>/costs/<kl_run>/cost_table.jsonl
    kl_run = path.parent.name
    costs_dir = path.parent.parent
    if costs_dir.name != "costs":
        raise ValueError(
            f"Cannot derive allocation dir from cost table path {path}: "
            "expected .../costs/<kl_run>/cost_table.jsonl"
        )
    run_root = costs_dir.parent
    return run_root / "allocation" / kl_run

from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from mix_bit.model_inventory import ModelInventory, TargetLinearSpec, with_fingerprint
from mix_bit.schema import CandidateMode, CandidateSpaceConfig


def _modes_1_2_3() -> tuple[CandidateMode, ...]:
    return (
        CandidateMode(
            name="b16d32s2",
            nominal_bit=1.0,
            codebook_bits=16,
            codebook_dim=32,
            residual_stages=2,
        ),
        CandidateMode(
            name="b32d32s2",
            nominal_bit=2.0,
            codebook_bits=32,
            codebook_dim=32,
            residual_stages=2,
        ),
        CandidateMode(
            name="b48d32s2",
            nominal_bit=3.0,
            codebook_bits=48,
            codebook_dim=32,
            residual_stages=2,
        ),
    )


def _space(
    modes: tuple[CandidateMode, ...] | None = None,
    *,
    baseline: str = "b32d32s2",
    target: float = 2.0,
) -> CandidateSpaceConfig:
    modes = modes or _modes_1_2_3()
    return CandidateSpaceConfig(
        candidate_space_id="toy_space",
        baseline_mode=baseline,
        target_average_bit=target,
        modes=modes,
    )


def _target(
    name: str,
    *,
    category: str = "q_proj",
    block: int = 0,
    in_f: int = 4,
    out_f: int = 4,
    param_count: int | None = None,
) -> TargetLinearSpec:
    pc = int(param_count) if param_count is not None else int(in_f * out_f)
    return TargetLinearSpec(
        module_name=name,
        category=category,
        module_suffix=category,
        block_index=block,
        in_features=in_f,
        out_features=out_f,
        has_bias=False,
        param_count=pc,
        transpose=True,
    )


def _inventory(targets: list[TargetLinearSpec]) -> ModelInventory:
    cats = tuple(dict.fromkeys(t.category for t in targets))
    inv = ModelInventory(
        model_id="toy",
        model_path="toy-model",
        transformers_model_type="toy",
        resolved_model_class="ToyLM",
        adapter_name="generic_decoder",
        model_profile_sha256="profile-sha",
        category_order=cats,
        block_count=len({t.block_index for t in targets}),
        targets=tuple(targets),
        total_target_parameters=int(sum(t.param_count for t in targets)),
        fingerprint_sha256="",
    )
    return with_fingerprint(inv)


def _cost_map(
    inventory: ModelInventory,
    space: CandidateSpaceConfig,
    costs: dict[tuple[str, str], float],
) -> list[Any]:
    from mix_bit.solver import CostRow

    rows: list[CostRow] = []
    for target in inventory.targets:
        for mode in space.modes:
            key = (target.module_name, mode.name)
            if key not in costs:
                raise KeyError(f"missing cost for {key}")
            rows.append(
                CostRow(
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
                    mean_delta_kl=float(costs[key]),
                    kl_mode="teacher_topk",
                    metric_name="forward_kl_teacher_topk_renorm",
                    teacher_topk=256,
                    run_config_sha256="run-sha",
                    model_inventory_sha256=inventory.fingerprint_sha256,
                    candidate_manifest_sha256="manifest-sha",
                    candidate_space_sha256="space-sha",
                    compact_state_sha256=f"compact-{mode.name}",
                    per_sample_sha256=f"ps-{target.module_name}-{mode.name}",
                )
            )
    return rows


def test_solver_selects_one_mode_per_module():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=100),
            _target("m1", category="k_proj", block=1, param_count=100),
        ]
    )
    costs = {}
    for t in inv.targets:
        costs[(t.module_name, "b16d32s2")] = 1.0
        costs[(t.module_name, "b32d32s2")] = 0.0
        costs[(t.module_name, "b48d32s2")] = 2.0
    costs[("m0", "b16d32s2")] = -0.5
    costs[("m1", "b48d32s2")] = -0.4
    # Compensate: m0@1bit + m1@3bit keeps weighted average at 2.0
    rows = _cost_map(inv, space, costs)
    result = solve_mixed_bit_allocation(
        rows,
        inventory=inv,
        candidate_space=space,
        target_average_bit=2.0,
    )
    selected = {e.module_name: e.mode for e in result.entries}
    assert set(selected) == {"m0", "m1"}
    assert selected["m0"] == "b16d32s2"
    assert selected["m1"] == "b48d32s2"
    assert len(result.entries) == 2


def test_parameter_weighted_budget_not_arithmetic_mean():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    # Large module dominates weighted average; arithmetic mean would allow L@3 + S@1.
    inv = _inventory(
        [
            _target("large", in_f=10, out_f=10, param_count=100),
            _target("small", category="k_proj", block=1, in_f=1, out_f=1, param_count=1),
        ]
    )
    costs = {
        ("large", "b16d32s2"): 5.0,
        ("large", "b32d32s2"): 0.0,
        ("large", "b48d32s2"): -10.0,  # tempting under arithmetic mean
        ("small", "b16d32s2"): -10.0,
        ("small", "b32d32s2"): 0.0,
        ("small", "b48d32s2"): 5.0,
    }
    rows = _cost_map(inv, space, costs)
    result = solve_mixed_bit_allocation(
        rows,
        inventory=inv,
        candidate_space=space,
        target_average_bit=2.0,
    )
    selected = {e.module_name: e.mode for e in result.entries}
    # Arithmetic-mean-feasible but weighted-infeasible: large@3 + small@1
    assert not (selected["large"] == "b48d32s2" and selected["small"] == "b16d32s2")
    assert selected == {"large": "b32d32s2", "small": "b16d32s2"}
    assert result.achieved_average_bit <= 2.0 + 1e-12
    # Weighted units: 100*4 + 1*2 = 402 <= 4*101 = 404
    assert result.used_bit_units <= result.budget_bit_units


def test_half_bit_units_are_converted_exactly_to_integers():
    from mix_bit.solver import BIT_UNIT_DENOMINATOR, bit_to_units

    assert BIT_UNIT_DENOMINATOR == 2
    assert bit_to_units(1.0) == 2
    assert bit_to_units(1.5) == 3
    assert bit_to_units(2.0) == 4
    assert bit_to_units(2.5) == 5
    assert bit_to_units(3.0) == 6


def test_solver_rejects_bit_not_representable_in_half_bit_units():
    from mix_bit.solver import bit_to_units, solve_mixed_bit_allocation

    with pytest.raises(ValueError, match="half-bit|representable|1e-12|conversion"):
        bit_to_units(1.25)

    bad_modes = (
        CandidateMode(
            name="bad",
            nominal_bit=1.25,
            codebook_bits=20,
            codebook_dim=32,
            residual_stages=2,
        ),
        CandidateMode(
            name="b32d32s2",
            nominal_bit=2.0,
            codebook_bits=32,
            codebook_dim=32,
            residual_stages=2,
        ),
    )
    space = _space(bad_modes, baseline="b32d32s2", target=2.0)
    inv = _inventory([_target("m0", param_count=16)])
    costs = {("m0", "bad"): -1.0, ("m0", "b32d32s2"): 0.0}
    rows = _cost_map(inv, space, costs)
    with pytest.raises(ValueError, match="half-bit|representable|1e-12|conversion"):
        solve_mixed_bit_allocation(
            rows,
            inventory=inv,
            candidate_space=space,
            target_average_bit=2.0,
        )


def test_solver_keeps_negative_cost_candidate():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=50),
            _target("m1", category="k_proj", block=1, param_count=50),
        ]
    )
    costs = {}
    for t in inv.targets:
        costs[(t.module_name, "b16d32s2")] = 1.0
        costs[(t.module_name, "b32d32s2")] = 0.0
        costs[(t.module_name, "b48d32s2")] = 1.0
    costs[("m0", "b16d32s2")] = -0.25  # negative KL delta must be kept
    rows = _cost_map(inv, space, costs)
    result = solve_mixed_bit_allocation(
        rows,
        inventory=inv,
        candidate_space=space,
        target_average_bit=2.0,
    )
    selected = {e.module_name: e.mode for e in result.entries}
    assert selected["m0"] == "b16d32s2"
    assert result.objective_delta_kl < 0.0
    m0_entry = next(e for e in result.entries if e.module_name == "m0")
    assert m0_entry.mean_delta_kl == -0.25


def test_uniform_2bit_solution_is_always_feasible():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=10),
            _target("m1", category="k_proj", block=1, param_count=20),
            _target("m2", category="v_proj", block=2, param_count=30),
        ]
    )
    # All non-baseline modes worse; optimum must be uniform baseline.
    costs = {}
    for t in inv.targets:
        costs[(t.module_name, "b16d32s2")] = 100.0
        costs[(t.module_name, "b32d32s2")] = 0.0
        costs[(t.module_name, "b48d32s2")] = 100.0
    rows = _cost_map(inv, space, costs)
    result = solve_mixed_bit_allocation(
        rows,
        inventory=inv,
        candidate_space=space,
        target_average_bit=2.0,
    )
    assert all(e.mode == "b32d32s2" for e in result.entries)
    assert result.objective_delta_kl == 0.0
    assert result.is_globally_optimal is True
    assert math.isclose(result.achieved_average_bit, 2.0, abs_tol=1e-12)


def test_objective_scaling_does_not_change_argmin():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=40),
            _target("m1", category="k_proj", block=1, param_count=60),
        ]
    )
    base_costs = {
        ("m0", "b16d32s2"): -0.03,
        ("m0", "b32d32s2"): 0.0,
        ("m0", "b48d32s2"): 0.05,
        ("m1", "b16d32s2"): 0.04,
        ("m1", "b32d32s2"): 0.0,
        ("m1", "b48d32s2"): -0.02,
    }
    rows_a = _cost_map(inv, space, base_costs)
    rows_b = _cost_map(
        inv,
        space,
        {k: v * 1e-9 for k, v in base_costs.items()},
    )
    res_a = solve_mixed_bit_allocation(
        rows_a, inventory=inv, candidate_space=space, target_average_bit=2.0
    )
    res_b = solve_mixed_bit_allocation(
        rows_b, inventory=inv, candidate_space=space, target_average_bit=2.0
    )
    assert {e.module_name: e.mode for e in res_a.entries} == {
        e.module_name: e.mode for e in res_b.entries
    }
    assert res_a.objective_scale == 1.0 / max(abs(v) for v in base_costs.values())
    assert res_b.objective_scale == pytest.approx(1.0 / (0.05 * 1e-9))


def test_solver_matches_bruteforce_on_tiny_problem():
    from mix_bit.solver import BIT_UNIT_DENOMINATOR, bit_to_units, solve_mixed_bit_allocation

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=8),
            _target("m1", category="k_proj", block=1, param_count=24),
        ]
    )
    costs = {
        ("m0", "b16d32s2"): 0.2,
        ("m0", "b32d32s2"): 0.0,
        ("m0", "b48d32s2"): -0.3,
        ("m1", "b16d32s2"): -0.4,
        ("m1", "b32d32s2"): 0.0,
        ("m1", "b48d32s2"): 0.5,
    }
    rows = _cost_map(inv, space, costs)
    mode_names = [m.name for m in space.modes]
    bit_of = {m.name: m.nominal_bit for m in space.modes}
    total_n = sum(t.param_count for t in inv.targets)
    target_units = bit_to_units(2.0)
    best_obj = None
    best_assign = None
    for assign in itertools.product(mode_names, repeat=len(inv.targets)):
        used = 0
        obj = 0.0
        mapping = {}
        for target, mode in zip(inv.targets, assign, strict=True):
            mapping[target.module_name] = mode
            used += target.param_count * bit_to_units(bit_of[mode])
            obj += costs[(target.module_name, mode)]
        if used > target_units * total_n:
            continue
        if best_obj is None or obj < best_obj - 1e-15:
            best_obj = obj
            best_assign = mapping
    assert best_assign is not None
    result = solve_mixed_bit_allocation(
        rows, inventory=inv, candidate_space=space, target_average_bit=2.0
    )
    selected = {e.module_name: e.mode for e in result.entries}
    assert selected == best_assign
    assert result.objective_delta_kl == pytest.approx(best_obj, abs=1e-12)
    assert result.bit_unit_denominator == BIT_UNIT_DENOMINATOR


def test_solver_rejects_fractional_or_ambiguous_solution():
    from mix_bit.solver import solve_mixed_bit_allocation

    space = _space()
    inv = _inventory([_target("m0", param_count=16), _target("m1", category="k_proj", block=1, param_count=16)])
    costs = {}
    for t in inv.targets:
        costs[(t.module_name, "b16d32s2")] = 1.0
        costs[(t.module_name, "b32d32s2")] = 0.0
        costs[(t.module_name, "b48d32s2")] = 1.0
    rows = _cost_map(inv, space, costs)

    class FakeRes:
        success = True
        status = 0
        message = "fake fractional"
        fun = 0.0
        x = np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    with mock.patch("mix_bit.solver.milp", return_value=FakeRes()):
        with pytest.raises(ValueError, match="fractional|ambiguous|binary|0 or 1"):
            solve_mixed_bit_allocation(
                rows,
                inventory=inv,
                candidate_space=space,
                target_average_bit=2.0,
            )


def test_incomplete_or_wrong_provenance_cost_table_is_rejected(tmp_path: Path):
    from mix_bit.solver import (
        load_cost_table_for_solve,
        solve_mixed_bit_allocation,
        write_allocation_outputs,
    )
    from mix_bit.schema import sha256_file

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=16),
            _target("m1", category="k_proj", block=1, param_count=16),
        ]
    )
    costs = {}
    for t in inv.targets:
        costs[(t.module_name, "b16d32s2")] = 1.0
        costs[(t.module_name, "b32d32s2")] = 0.0
        costs[(t.module_name, "b48d32s2")] = 1.0
    rows = _cost_map(inv, space, costs)

    table_path = tmp_path / "cost_table.jsonl"
    with open(table_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
    table_sha = sha256_file(table_path)
    meta = {
        "kind": "mix_bit_cost_table_meta",
        "cost_table_sha256": table_sha,
        "row_count": len(rows),
        "L": len(inv.targets),
        "R": len(space.modes),
        "run_config_sha256": "run-sha",
        "model_inventory_sha256": inv.fingerprint_sha256,
        "candidate_manifest_sha256": "manifest-sha",
        "candidate_space_sha256": "space-sha",
        "kl_mode": "teacher_topk",
        "metric_name": "forward_kl_teacher_topk_renorm",
        "teacher_topk": 256,
        "baseline_kl_mean": 1.23,
    }
    meta_path = tmp_path / "cost_table_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected = {
        "run_config_sha256": "run-sha",
        "model_inventory_sha256": inv.fingerprint_sha256,
        "candidate_manifest_sha256": "manifest-sha",
        "candidate_space_sha256": "space-sha",
    }
    loaded = load_cost_table_for_solve(
        table_path,
        meta_path,
        inventory=inv,
        candidate_space=space,
        expected_hashes=expected,
    )
    assert len(loaded) == len(rows)

    bad_meta = copy.deepcopy(meta)
    bad_meta["cost_table_sha256"] = "0" * 64
    bad_meta_path = tmp_path / "bad_meta.json"
    bad_meta_path.write_text(json.dumps(bad_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cost_table_sha256|SHA256|hash"):
        load_cost_table_for_solve(
            table_path,
            bad_meta_path,
            inventory=inv,
            candidate_space=space,
            expected_hashes=expected,
        )

    wrong_expected = dict(expected)
    wrong_expected["candidate_manifest_sha256"] = "wrong-manifest"
    with pytest.raises(ValueError, match="provenance|manifest|hash"):
        load_cost_table_for_solve(
            table_path,
            meta_path,
            inventory=inv,
            candidate_space=space,
            expected_hashes=wrong_expected,
        )

    incomplete = rows[:-1]
    with pytest.raises(ValueError, match="L \\* R|incomplete|row"):
        solve_mixed_bit_allocation(
            incomplete,
            inventory=inv,
            candidate_space=space,
            target_average_bit=2.0,
        )

    # Output writer smoke: allocation artifacts are written atomically.
    result = solve_mixed_bit_allocation(
        rows, inventory=inv, candidate_space=space, target_average_bit=2.0
    )
    out_dir = tmp_path / "allocation" / "topk_k256"
    paths = write_allocation_outputs(
        result,
        output_dir=out_dir,
        model_id=inv.model_id,
        run_id="toy_run",
        provenance={
            **expected,
            "cost_table_sha256": table_sha,
            "cost_table_meta_sha256": sha256_file(meta_path),
            "kl_mode": "teacher_topk",
            "metric_name": "forward_kl_teacher_topk_renorm",
            "teacher_topk": 256,
            "baseline_kl_mean": 1.23,
        },
    )
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    assert Path(paths["markdown"]).is_file()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["is_globally_optimal"] is True
    assert payload["objective_delta_kl"] == pytest.approx(result.objective_delta_kl)
    assert len(payload["entries"]) == len(inv.targets)
    assert "excluded_modes" not in payload


def test_parse_exclude_modes_rejects_duplicates():
    from mix_bit.solver import parse_exclude_modes

    assert parse_exclude_modes(None) == ()
    assert parse_exclude_modes("b16d32s2") == ("b16d32s2",)
    assert parse_exclude_modes(" b16d32s2 , b24d32s2 ") == ("b16d32s2", "b24d32s2")
    with pytest.raises(ValueError, match="duplicate"):
        parse_exclude_modes("b16d32s2,b16d32s2")


def test_with_excluded_modes_rejects_baseline_and_unknown():
    from mix_bit.solver import with_excluded_modes

    space = _space()
    inv = _inventory([_target("m0", param_count=16)])
    costs = {
        ("m0", "b16d32s2"): -1.0,
        ("m0", "b32d32s2"): 0.0,
        ("m0", "b48d32s2"): 1.0,
    }
    rows = _cost_map(inv, space, costs)
    with pytest.raises(ValueError, match="baseline_mode"):
        with_excluded_modes(space, rows, ["b32d32s2"])
    with pytest.raises(ValueError, match="not in candidate space"):
        with_excluded_modes(space, rows, ["missing_mode"])


def test_excluding_1bit_never_selects_it_and_meets_budget():
    from mix_bit.solver import solve_mixed_bit_allocation, with_excluded_modes

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=100),
            _target("m1", category="k_proj", block=1, param_count=100),
        ]
    )
    costs = {
        ("m0", "b16d32s2"): -10.0,
        ("m0", "b32d32s2"): 0.0,
        ("m0", "b48d32s2"): -0.1,
        ("m1", "b16d32s2"): -10.0,
        ("m1", "b32d32s2"): 0.0,
        ("m1", "b48d32s2"): 0.2,
    }
    rows = _cost_map(inv, space, costs)
    full = solve_mixed_bit_allocation(
        rows, inventory=inv, candidate_space=space, target_average_bit=2.0
    )
    assert any(entry.mode == "b16d32s2" for entry in full.entries)

    filtered_space, filtered_rows = with_excluded_modes(space, rows, ["b16d32s2"])
    assert all(row.mode != "b16d32s2" for row in filtered_rows)
    result = solve_mixed_bit_allocation(
        filtered_rows,
        inventory=inv,
        candidate_space=filtered_space,
        target_average_bit=2.0,
    )
    assert all(entry.mode != "b16d32s2" for entry in result.entries)
    assert result.is_globally_optimal is True
    assert result.used_bit_units <= result.budget_bit_units
    assert result.achieved_average_bit <= 2.0 + 1e-12


def test_write_allocation_records_excluded_modes(tmp_path: Path):
    from mix_bit.solver import (
        solve_mixed_bit_allocation,
        with_excluded_modes,
        write_allocation_outputs,
    )

    space = _space()
    inv = _inventory(
        [
            _target("m0", param_count=16),
            _target("m1", category="k_proj", block=1, param_count=16),
        ]
    )
    costs = {}
    for target in inv.targets:
        costs[(target.module_name, "b16d32s2")] = -1.0
        costs[(target.module_name, "b32d32s2")] = 0.0
        costs[(target.module_name, "b48d32s2")] = 1.0
    rows = _cost_map(inv, space, costs)
    filtered_space, filtered_rows = with_excluded_modes(space, rows, ["b16d32s2"])
    result = solve_mixed_bit_allocation(
        filtered_rows,
        inventory=inv,
        candidate_space=filtered_space,
        target_average_bit=2.0,
    )
    paths = write_allocation_outputs(
        result,
        output_dir=tmp_path / "allocation",
        model_id=inv.model_id,
        run_id="toy_run",
        provenance={
            "run_config_sha256": "run-sha",
            "model_inventory_sha256": inv.fingerprint_sha256,
            "candidate_manifest_sha256": "manifest-sha",
            "candidate_space_sha256": "space-sha",
            "cost_table_sha256": "a" * 64,
            "cost_table_meta_sha256": "b" * 64,
            "kl_mode": "teacher_topk",
            "metric_name": "forward_kl_teacher_topk_renorm",
            "teacher_topk": 256,
            "baseline_kl_mean": 1.23,
            "excluded_modes": ["b16d32s2"],
        },
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["excluded_modes"] == ["b16d32s2"]
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "excluded_modes: b16d32s2" in markdown
    assert all(entry["mode"] != "b16d32s2" for entry in payload["entries"])

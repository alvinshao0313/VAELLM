> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-5-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Review: Aggregate Phase-1 With Hard Validity Gates Before Ranking

**Reviewer:** spec + quality review  
**Artifacts:** `task-5-brief.md`, `task-5-report.md`, `task-5-review-package.diff`  
**Verdict:** **Spec ✅** · **Approved**

---

## Summary

Task 5 delivers `summarize_phase1()` with the required hard gates, canonical baseline handling, ranking semantics, artifact schema, and unit coverage. Implementation matches the brief; no silent fallbacks observed. Tests verified locally: `13 passed` (`pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py`, `bitvae` / Python 3.11.13).

---

## Spec Compliance

| Requirement | Status | Evidence |
|---|---|---|
| Single entry `summarize_phase1(*, run_dir, selected_gpus) -> list[int]` | ✅ | `summarize.py:430–547`; returns ranked `layer_idx` in memory |
| Exact job inventory `38+W`, no missing/duplicate | ✅ | `_expected_formal_job_ids`, `_assert_job_inventory` |
| Canonical baseline = `compressed_baseline_worker00` only | ✅ | `CANONICAL_BASELINE_JOB_ID`; all metrics vs worker00 |
| Worker00 repeat + cross-GPU baseline determinism (`1e-12`, population) | ✅ | `_validate_baseline_determinism`, `_assert_matching_accuracies` |
| Evaluation population consistency for all phase-1 jobs | ✅ | `_assert_same_population` loop over all jobs |
| Homogeneous `device_name` or hard fail | ✅ | `_assert_homogeneous_devices` |
| `A_all_down_original > A_compressed` or stop (diagnostic, no ranking) | ✅ | Lines 474–477: diagnostic JSON + `ValueError`; no CSV/Spearman/`ranked_layers` |
| Historical 41.71% / 51.99% recorded, not gated | ✅ | `_historical_reference()` |
| `delta_mmlu_pp`, `single_recovery_fraction`, weight merge by `layer_idx` | ✅ | Per-layer loop + `_load_weight_metrics` |
| CSV columns exact order | ✅ | `SENSITIVITY_CSV_COLUMNS` matches brief |
| Subject improved/worsened/unchanged with `1e-12`; NumPy median | ✅ | `_subject_diagnostics` |
| Rank `(-delta_mmlu_pp, layer_idx)`, ranks 1..36 | ✅ | `sorted(...)` + enumerate |
| Local Spearman (average ranks + Pearson), no SciPy | ✅ | `spearman_rank_correlation` |
| `phase1_summary.json` scientific fields (+ allowed metadata) | ✅ | Success: all fields; down-gap fail: diagnostic subset only |
| `cross_gpu_baseline_probes` per-worker records, not boolean pass | ✅ | `_probe_record` list |
| Smoke must not produce formal ranking | ✅ | `summarize_phase1` rejects `mode=smoke`; `validate_smoke` writes no formal artifacts |
| Scope limited to `experiments/down_layer_sensitivity/` | ✅ | Only `summarize.py` + `tests/test_summarize.py` added |
| Unit tests for all listed failure/math cases | ✅ | 13 tests cover brief Step 11 checklist |

---

## Global Constraints

| Constraint | Status |
|---|---|
| Hard validity gates before ranking; no silent fallback | ✅ All gate failures raise `ValueError`; no degraded ranking path |
| Canonical baseline = `compressed_baseline_worker00` | ✅ |
| Ranking: `delta_mmlu_pp` desc, `layer_idx` asc | ✅ Tie test (layers 7/10) confirms |
| `A_all_down_original <= A_compressed` → stop phase-1, no Top-K | ✅ Diagnostic JSON only; process aborts before phase 2 |
| Smoke must not produce formal sensitivity ranking | ✅ |
| Only `experiments/down_layer_sensitivity/` | ✅ |

---

## Implementer Concern: `run_config.status` Stays `"running"` on Down-Gap Failure

**Not a Task 5 brief violation.**

- Task 5 brief covers aggregation logic and phase-1 artifacts only. It does not assign `summarize_phase1()` responsibility for updating `run_config.json`.
- Step 5 requirement—“write diagnostic summary but stop before sensitivity ranking and phase 2”—is satisfied: diagnostic `phase1_summary.json` is written with `status: all_down_original_not_greater_than_compressed`, then `ValueError` is raised; `run.py` never reaches `build_phase2_manifests()`.
- `run_config.status` lifecycle belongs to `run.py` orchestration (Task 6 explicitly modifies `run.py`). Today, uncaught exceptions leave `status="running"`; that is an orchestration/observability gap, not incorrect aggregation semantics.

**Recommendation (non-blocking, defer to Task 6):** Have `run.py` catch aggregation gate failures and set `run_config.status="failed"` (with reason), mirroring existing worker-failure handling in `launch_phase_workers`.

---

## Quality Notes (Non-Blocking)

1. **Extra validations beyond brief** — Job filename/`job_id` consistency, `restore_layers` checks, and formal/smoke `mode` enforcement strengthen gate integrity without contradicting the spec.
2. **`validate_smoke` / `summarize_final` stub** — Not in Task 5 brief, but required because `run.py` already imports them; appropriate forward hooks.
3. **Gate-failure artifact policy** — Only down-gap failure writes `phase1_summary.json`; other gate failures raise with no summary. This matches brief (Step 5 explicitly requires diagnostic summary only for the all-down-original direction check).
4. **Integration gap** — No end-to-end test that `run.py` exits non-zero and skips phase 2 on gate failure; acceptable at Task 5 scope given unit coverage and raise semantics.

---

## Test Verification

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py
# 13 passed in 4.65s  (reviewer re-run)
```

---

## Decision

| Gate | Result |
|---|---|
| **Spec** | ✅ |
| **Approval** | **Approved** |

Task 5 is complete per brief. Proceed to Task 6; address `run_config.status` there if desired for operational clarity.

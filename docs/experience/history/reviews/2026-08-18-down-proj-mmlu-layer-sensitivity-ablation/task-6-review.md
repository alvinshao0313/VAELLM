> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-6-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Review: Build and Run Phase-2 Cumulative Validation

**Reviewer:** spec + quality review  
**Artifacts:** `task-6-brief.md`, `task-6-report.md`, `task-6-review-package.diff`  
**Verdict:** **Spec ✅** · **Approved**

---

## Summary

Task 6 implements phase-2 manifest construction, orchestration, and final aggregation per brief Steps 1–11. `build_phase2_manifests()` consumes the in-memory `list[int]` from `summarize_phase1()`; `summarize_final()` validates phase-2 gates, reuses phase-1 Top-1 / `all_down_original`, writes `cumulative_results.csv` and `random8_aggregate` into `final_summary.json`, and leaves plots / `report.md` to Task 7. Global constraints are satisfied. Tests verified locally: `55 passed` (Task 6 scope), `77 passed` (full package), `bitvae` / Python 3.11.13.

---

## Spec Compliance

| Requirement | Status | Evidence |
|---|---|---|
| `build_phase2_manifests(*, selected_gpus, ranked_layers) -> list[dict]` | ✅ | `run.py:265–330` |
| `summarize_final(*, run_dir, selected_gpus) -> None` | ✅ | `summarize.py:1129–1314` |
| **Step 1** — `ranked_layers` permutation of `0..35`; Top-K = `ranked[:K]`; Top-1 not scheduled | ✅ | Validation at `run.py:273–277`; scientific jobs omit `top1`; tests assert `"top1" not in all_ids` |
| **Step 2** — Random-8 seeds `(31..35)`, `random.Random(s).sample`, `sorted`, `random8_seed{s}`, no redraw | ✅ | `RANDOM_CONTROL_SEEDS`, `run.py:306–315`; hardcoded restore sets in tests match |
| **Step 3** — Exactly 9 scientific jobs in fixed order; `W2 = min(len(gpus), 9)`; `selected_gpus[:W2]` | ✅ | `run.py:279–280, 300–315`; W=10 test uses GPUs `0..8` only |
| **Step 4** — Each worker starts with baseline; worker0 repeat second; total `9+W2+1`; same greedy scheduler | ✅ | `_least_loaded_worker_id` reused; `EXPECTED_PHASE2_JOB_IDS` for W∈{1,2,4,8,9} |
| **Step 5** — `phase2/manifests/worker_XX.json`; reuse `launch_phase_workers()`; all `write_weight_metrics=false` | ✅ | `main()` phase2 launch; post-build overwrite `manifest["write_weight_metrics"] = False` |
| **Step 6** — Phase-2 baselines + worker00 repeat vs phase-1 canonical: accuracy `1e-12`, population, per-subject accuracy, `device_name`; fail closed | ✅ | `summarize_final` baseline loop; failure tests write no CSV / `final_summary.json` |
| **Step 7** — Scientific jobs match population, device, formal `lm_limit=None` | ✅ | `_assert_same_population`, `_assert_device_name`, `_assert_formal_lm_limit` |
| **Step 8** — Recovery `(A_topK - A_compressed) / (A_all - A_compressed)`; Top-1 from phase-1 `restore_Lxx` | ✅ | `_configuration_metrics`; Top-1 loaded from phase-1 job, not phase-2 |
| **Step 9** — Random-8 mean/std with `np.mean`, `np.std(..., ddof=0)`; `top8_minus_random8_mean_recovery` | ✅ | `summarize.py:1251–1267`; test asserts `ddof=0` and delta |
| **Step 10** — No monotonicity enforcement / smoothing | ✅ | Test preserves Top-12 accuracy `<` Top-8; no clamp in `_configuration_metrics` |
| **Step 11** — `cumulative_results.csv` exact columns and row order; no `random8_mean` row | ✅ | `CUMULATIVE_CSV_COLUMNS`, `CUMULATIVE_ROW_ORDER`; test asserts no `random8_mean` |
| Orchestration: phase-2 on `selected_gpus[:W2]` only | ✅ | `main()` lines 410–417 |
| Orchestration: phase-1 summarize / manifest failure → `run_config.status="failed"` | ✅ | Shared `try/except` around `summarize_phase1` + `build_phase2_manifests`; integration test for phase-1 failure |
| Orchestration: `summarize_final` failure → `status="failed"` | ✅ | `main()` lines 419–423 |

---

## Global Constraints

| Constraint | Status |
|---|---|
| Phase-2 Top-K `{1,2,4,8,12}`; Top-1 reuse phase-1, no rerun | ✅ |
| Random-8 seeds `31..35`, sample 8 without replacement from `0..35` | ✅ |
| Launch phase-2 on `selected_gpus[:W2]` only | ✅ |
| Use ranked `list[int]` from `summarize_phase1()` return value | ✅ (`main()` passes return value directly to `build_phase2_manifests`) |
| Do not clamp `cumulative_recovery_fraction` | ✅ (tests include `<0` and `>1`) |
| Phase-1 summarize failure → `run_config.status=failed` in orchestration | ✅ |
| Charts / `report.md` deferred to Task 7 | ✅ (brief Steps 1–11 end at CSV; header mention is out of scope for this task) |

---

## Implementer Concerns (Assessed)

### 1. Brief header lists plots / `report.md` but Steps 1–11 stop at CSV

**Not a spec violation.** Numbered steps define Task 6 deliverables; Task 7 owns figures and markdown report. `final_summary.json` (with `random8_aggregate`) is appropriate here because Step 9 assigns aggregate statistics to JSON / report / plot — JSON is the durable handoff before Task 7 rendering.

### 2. `summarize_final` reads ranking from `phase1_summary.json`, not a function argument

**Acceptable.** Brief interface is `summarize_final(*, run_dir, selected_gpus)`. Ranking is persisted by `summarize_phase1()` immediately before phase-2 launch; `_ranked_layer_ids()` validates the same permutation. In-memory consumption requirement applies to manifest build, which uses the return value directly.

### 3. Formal `lm_limit=None` checked via `mode=formal` when job JSON omits `lm_limit`

**Acceptable.** Manifests set `lm_limit=None` at schedule time (`_make_job`); worker job results store `mode` but not necessarily `lm_limit`. `_assert_formal_lm_limit` rejects non-`None` if present. This satisfies Step 7 intent without requiring worker schema changes.

---

## Quality Notes (Non-Blocking)

1. **Strong test matrix** — Greedy allocation tables for W∈{1,2,4,8,9}, W=10 GPU slice, Random-8 restore sets, Top-1 reuse, recovery edge values, fail-closed aggregation, and `main()` phase-2 launch / phase-1-failure paths.
2. **Minor integration gap** — No test that `main()` sets `status="failed"` when `summarize_final` raises (only phase-1 failure is covered). Behavior is straightforward from `main()` source; low risk.
3. **`build_phase2_manifests` invalid-input failure in `main()`** — Covered by unit tests on `build_phase2_manifests`; shares the same `try/except` as `summarize_phase1` failure.
4. **Scope discipline** — Changes confined to `experiments/down_layer_sensitivity/`; production train/compress/eval code untouched.

---

## Test Verification

```bash
PYTHONPATH=. pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py \
  experiments/down_layer_sensitivity/tests/test_summarize.py
# 55 passed in 5.44s  (reviewer re-run)

PYTHONPATH=. pytest -q experiments/down_layer_sensitivity/tests
# 77 passed in 5.61s  (reviewer re-run)
```

---

## Decision

| Gate | Result |
|---|---|
| **Spec** | ✅ |
| **Approval** | **Approved** |

Task 6 meets brief Steps 1–11 and global constraints. Proceed to Task 7 for plots and `report.md`.

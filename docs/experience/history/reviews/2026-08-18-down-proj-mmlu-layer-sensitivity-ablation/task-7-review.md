> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-7-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Review: Reproducible Report and Three Diagnostic Figures

**Reviewer:** spec + quality review  
**Artifacts:** `task-7-brief.md`, `task-7-report.md`, `task-7-review-package.diff`  
**Verdict:** **Spec ✅** · **Approved**

---

## Summary

Task 7 adds plot/report rendering to `summarize_final()`: after writing `cumulative_results.csv` and `final_summary.json`, `_generate_report_and_plots()` emits three fixed PNGs under `plots/` and a Chinese `report.md` sourced from JSON/CSV artifacts. `ranked_layers` and Spearman rho are reused from `phase1_summary.json`; `topk` / `random8_controls` / `random8_aggregate` align with the cumulative CSV row set. Global constraints hold (NumPy/Matplotlib only, scope confined to `experiments/down_layer_sensitivity/`). Tests verified locally: **17 passed** (`bitvae` / Python 3.11.13).

---

## Spec Compliance

| Requirement | Status | Evidence |
|---|---|---|
| **Step 1** — `plots/layer_sensitivity.png`: x=`layer_idx` 0..35, y=`delta_mmlu_pp`, one bar/layer, y=0 line, Top-8 annotate layer/rank, x not reordered | ✅ | `_plot_layer_sensitivity()` bar on `range(EXPECTED_DOWN_LAYERS)`, `axhline(0)`, annotate `L{idx}/#{rank}` |
| **Step 2** — `plots/nmse_vs_mmlu_sensitivity.png`: x=`weight_nmse`, y=`delta_mmlu_pp`, Top-8 layer ID labels, Spearman rho in title, diagnostic framing | ✅ | `_plot_nmse_vs_mmlu_sensitivity()`; report §6 states “相关不等于因果” |
| **Step 3** — `plots/cumulative_recovery.png`: Top-K K∈{1,2,4,8,12,36}, single Random-8 point at x=8 (mean ± std), no five random curves, no dual y-axis | ✅ | `_plot_cumulative_recovery()` curve over `top1..top12, all_down_original`; one `errorbar` at x=8 |
| **Step 4** — Write `final_summary.json` **before** report/plots; required scientific fields; no re-rank / re-sample / alternate MMLU | ✅ | `_dump_json(final_summary)` then `_generate_report_and_plots()`; `ranked_layers` / Spearman from `phase1_summary`; topk/random8 from same `metrics_by_config` as CSV |
| **Step 5** — Chinese `report.md` with fixed §1–§9; all mandated numeric bullets; numbers from JSON/CSV not re-derived | ✅ | `_render_report()` sections and required fields; ranking table `weight_nmse` from CSV; aggregates from `final_summary` |
| **Step 6** — Conclusions bounded to MMLU setting; no cross-task generalization | ✅ | §9 uses mandated “在当前 final_model 与 0-shot full-MMLU 设置下，Lx/…” form |
| Modify `summarize.py`, create `README.md` | ✅ | Per diff |
| Fail-closed: no report/plots on aggregation failure | ✅ | Failure tests assert absent `report.md` / `plots/` |

---

## Global Constraints

| Constraint | Status |
|---|---|
| Three fixed plot filenames under `plots/` | ✅ |
| `report.md` reproducible from run artifacts (JSON/CSV-driven renderer) | ✅ |
| No new third-party deps (NumPy/Matplotlib only) | ✅ |
| Changes only under `experiments/down_layer_sensitivity/` | ✅ |

---

## Implementer Concerns (Assessed)

### 1. Formal run not executed; charts/report validated on synthetic fixtures only

**Acceptable for Task 7 scope.** Unit/integration tests cover artifact wiring, section structure, and fail-closed behavior. Visual correctness on real runs remains an operational check when phase-1/2 jobs complete.

### 2. Task 8 shell usage not in README

**Out of scope.** Task 7 brief does not require orchestration docs beyond output artifacts.

---

## Quality Notes (Non-Blocking)

1. **§3 worker00 repeat line** — Report prints `worker00_baseline_repeat accuracy` using `compressed['accuracy']` instead of a dedicated repeat probe field. Gate guarantees equality; prefer reading `phase1_summary["worker00_baseline_repeat"]` or an explicit final-summary field for clarity.
2. **Historical reference labels (41.71% / 51.99%)** — Display uses module constants; diff values correctly come from `historical_reference` in JSON. For strict “all numbers from JSON/CSV”, labels could use `historical['historical_compressed_mmlu']` etc.
3. **Phase-2 baseline detail** — §3 prose asserts phase-2 baseline consistency but enumerates only phase-1 `cross_gpu_baseline_probes` (the only probes persisted in `final_summary.json`). Sufficient for brief’s “consistency status” requirement; optional enhancement would note phase-2 re-validation explicitly.
4. **Test depth** — Success path asserts file existence and section headers, not plot pixel content or report↔JSON numeric equality. Adequate for wiring; formal-run spot check still useful.
5. **Minor test duplication** — `test_summarize_final_phase2_baseline_mismatch_writes_no_csv` repeats `report.md` / `plots` assertions twice (harmless).

---

## Test Verification

```bash
PYTHONPATH=. pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py
# 17 passed in 5.37s  (reviewer re-run, bitvae / Python 3.11.13)
```

---

## Decision

| Gate | Result |
|---|---|
| **Spec** | ✅ |
| **Approval** | **Approved** |

Task 7 meets brief Steps 1–6 and global constraints. Proceed to Task 8 (or formal run + visual spot-check as operational follow-up).

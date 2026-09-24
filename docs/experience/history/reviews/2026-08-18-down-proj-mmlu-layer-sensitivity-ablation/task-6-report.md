> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-6-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Report: Build and Run Phase-2 Cumulative Validation

**Status:** DONE_WITH_CONCERNS  
**Commits:** none

## Files

- Modify: `experiments/down_layer_sensitivity/run.py`
- Modify: `experiments/down_layer_sensitivity/summarize.py`
- Extend: `experiments/down_layer_sensitivity/tests/test_job_manifest.py`
- Extend: `experiments/down_layer_sensitivity/tests/test_summarize.py`
- Production training/compression/eval code was not modified.
- `scripts/catlora_simple.sh` was not touched.

## What changed

### `run.py`

- Implemented `build_phase2_manifests(*, selected_gpus, ranked_layers)`:
  - Requires `ranked_layers` to be a permutation of `0..35` (the in-memory list returned by `summarize_phase1()`, not a CSV re-rank).
  - Restore sets: `top2/4/8/12 = ranked[:K]`; Top-1 is not scheduled.
  - Random-8: seeds `(31, 32, 33, 34, 35)` via `random.Random(s).sample(range(36), 8)` then `sorted`; no redraw.
  - Scientific order: `top2, top4, top8, top12, random8_seed31..35`.
  - `W2 = min(len(selected_gpus), 9)`, GPUs `selected_gpus[:W2]`.
  - Each worker starts with `compressed_baseline_workerXX`; worker 0 also has `compressed_baseline_worker00_repeat`.
  - Total jobs `9 + W2 + 1`. Scientific jobs use the same greedy scheduler as phase 1 (`_least_loaded_worker_id`).
  - All manifests `write_weight_metrics=false`, `mode=formal`, `lm_limit=None`.
- Formal `main()` now:
  - launches phase 2 with `launch_phase_workers(..., selected_gpus=phase2_gpus, manifests=phase2_manifests)` (reuse, no new CLI);
  - on `summarize_phase1` / `build_phase2_manifests` failure sets `run_config.status="failed"`;
  - on `summarize_final` failure also sets `status="failed"`.

### `summarize.py`

- Implemented `summarize_final(*, run_dir, selected_gpus)`:
  - Phase-2 inventory must be exactly `9 + W2 + 1`.
  - Phase-2 baselines and worker0 repeat must match phase-1 `compressed_baseline_worker00` on accuracy (`1e-12`), subject set, per-subject counts, `n_samples_total`, per-subject accuracy, and `device_name`.
  - Scientific jobs must match the same evaluation population, device name, and formal `lm_limit=None` (via `mode=formal`; job results do not store `lm_limit`).
  - Ranking is read from `phase1_summary.json` (`ranked_layers`), not recomputed from CSV.
  - Top-1 accuracy is reused from phase-1 `restore_Lxx` of rank-1 layer.
  - Recovery: `(A - A_compressed) / (A_all - A_compressed)` with no `[0, 1]` clamp and no monotonicity enforcement.
  - Random-8 aggregate uses `np.mean` / `np.std(..., ddof=0)`.
  - Writes `cumulative_results.csv` with the exact columns and row order; no `random8_mean` row.
  - Writes `final_summary.json` with Task-7 scientific fields (`topk`, `random8_controls`, `random8_aggregate`, plus reused phase-1 ranking / baselines / Spearman / historical reference).
  - Does **not** write `report.md` or plots (Task 7).

## Tests

Environment: `bitvae` (`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`, Python 3.11.13).

```bash
PYTHONPATH=. pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py experiments/down_layer_sensitivity/tests/test_summarize.py
# 55 passed

PYTHONPATH=. pytest -q experiments/down_layer_sensitivity/tests
# 77 passed
```

Covered: phase-2 greedy allocation for W∈{1,2,4,8,9}; W=10 uses first 9 GPUs; Random-8 restore sets for seeds 31–35; Top-1 not scheduled; `write_weight_metrics=false`; formal `main()` phase-2 launch slice; `summarize_phase1` failure sets `status=failed`; Top-1 reuse; recovery math including values `<0` and `>1`; non-monotonic Top-12 vs Top-8 preserved; no `random8_mean` CSV row; Random-8 mean/std `ddof=0`; phase-2 baseline / population / device / missing-job failures write no CSV or `final_summary.json`.

## Self-review

- Phase-2 scheduling reuses `_least_loaded_worker_id` / `_make_job` / `launch_phase_workers`; worker CLI unchanged.
- Manifest restore lists keep ranked order for Top-K and sorted order for Random-8, matching the brief.
- Aggregation fails closed: no CSV / `final_summary.json` on gate failure.
- Did not clamp recovery, did not smooth Top-K, did not redraw overlapping Random-8 seeds.

## Concerns

- Brief header lists plots and `report.md` as Task 6 products, but numbered steps 1–11 stop at `cumulative_results.csv` and Task 7 owns the three figures plus markdown report. This task writes `final_summary.json` (needed for Random-8 aggregates) and leaves plots/`report.md` to Task 7.
- Worker job JSON does not store `lm_limit`; formal `lm_limit=None` is enforced on manifests and checked in aggregation as `mode=formal` plus optional `lm_limit is None`.
- `_make_manifest` still defaults `write_weight_metrics` from `worker_id==0`; phase 2 overwrites every manifest to `False` after construction.

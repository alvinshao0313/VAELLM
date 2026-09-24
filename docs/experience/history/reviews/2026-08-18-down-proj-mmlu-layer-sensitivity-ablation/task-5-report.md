> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-5-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Report: Aggregate Phase-1 With Hard Validity Gates Before Ranking

**Status:** complete  
**Commits:** none

## Files

- Created `experiments/down_layer_sensitivity/summarize.py`
- Created `experiments/down_layer_sensitivity/tests/test_summarize.py`
- Production training/compression/eval code was not modified.

## What changed

`summarize.py` implements the Task 5 formal entry point:

```python
def summarize_phase1(*, run_dir: str, selected_gpus: list[str]) -> list[int]:
```

Formal aggregation order is fixed:

1. Refuse `mode=smoke` (no formal ranking).
2. Require exact job inventory `38+W`.
3. Canonical baseline is always `compressed_baseline_worker00`.
4. Worker00 repeat and every other GPU baseline must match canonical accuracy, subject set, per-subject sample counts, `n_samples_total`, and per-subject accuracy within `1e-12`.
5. Every phase-1 job must share the canonical evaluation population.
6. All phase-1 `device_name` values must be homogeneous.
7. Require `A_all_down_original > A_compressed`. If not, write diagnostic `phase1_summary.json` and raise; no ranking CSV, no Spearman, no `ranked_layers`.
8. Compute per-layer `delta_mmlu_pp` / `single_recovery_fraction`, merge worker0 `phase1/weight_metrics_worker.json` by `layer_idx`, compute subject improved/worsened/unchanged with `1e-12`, rank by `(-delta_mmlu_pp, layer_idx)`, assign rank `1..36`.
9. Spearman is local average-rank + NumPy Pearson; no SciPy.
10. Write `phase1_summary.json`, `single_layer_sensitivity.csv`, `weight_metrics.csv` under `run_dir`. Return the 36 ranked `layer_idx` values in memory.

Task 6/7 hooks:

- `validate_smoke(...)` is implemented because `run.py` already calls it after smoke phase-1. It checks the four smoke jobs and writes no formal ranking artifacts.
- `summarize_final(...)` raises `NotImplementedError("summarize_final is implemented in Task 6.")`.

## Tests

Environment: `bitvae` (`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`, Python 3.11.13).

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py
# 13 passed

pytest -q experiments/down_layer_sensitivity/tests
# 63 passed
```

Covered: delta pp, recovery fraction, rank order, equal-delta tie by `layer_idx`, subject improved/worsened counts, baseline mismatch, sample-count mismatch, all-original `<=` baseline diagnostic stop, heterogeneous GPU, Spearman monotonic/reverse/tied, smoke must not rank.

## Concerns

- `A_all_down_original <= A_compressed` raises `ValueError` after writing diagnostic summary. `run.py` does not catch it, so the formal process exits non-zero, but `run_config.json` stays `status=running` rather than `failed`.
- `phase1_summary.json` `ranked_layers` is a list of dicts (`rank`, `layer_idx`, metrics). The function return value is `list[int]` for `build_phase2_manifests()`. Task 6 must use the return value, not re-rank from CSV.
- Historical 41.71% / 51.99% are recorded only; they are not hard gates.

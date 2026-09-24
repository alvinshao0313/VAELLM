> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-3-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Report: Per-GPU Worker and Self-Describing Jobs

## Status

DONE

## What Was Implemented

Created `experiments/down_layer_sensitivity/worker.py` as the per-GPU job runner.

### CLI (fixed, no extra flags)

```text
--checkpoint_dir
--manifest_path
--jobs_dir
--worker_meta_path
--worker_id
--physical_gpu_id
```

`worker_id` / `physical_gpu_id` must match CLI and manifest exactly.

### Manifest validation (before model load)

Rejects:

- duplicate `job_id`
- restore layer outside `0..35`
- duplicate layer in one restore list
- mode not in `{smoke, formal}`
- formal job with `lm_limit is not None`
- smoke job with `lm_limit != 2`
- CLI vs manifest `worker_id` / `physical_gpu_id` mismatch

Also fail-loud on: missing/non-bool `write_weight_metrics`, non-list `jobs`, non-int restore entries, job mode ≠ manifest mode.

### Runtime sequence

1. Parse CLI + load/validate manifest
2. Seeds once: `random/np/torch/cuda` seed **31** (no deterministic-algorithm mode)
3. `load_worker_model(..., prewarm_group_size=8)` once; `build_tokenizer` once; logical device `cuda:0`
4. If `write_weight_metrics=true`: `compute_down_weight_metrics` from prewarmed cache; write `weight_metrics_worker.json` to `dirname(jobs_dir)`
5. Write worker metadata
6. Execute jobs in order; on first exception: print traceback, `sys.exit(1)`, do not continue

Each job:

```text
reset_all_vae_to_compressed → assert empty
set_down_restore_set → assert restore set
try: evaluate_mmlu
finally: reset + assert empty
```

Per-job JSON fields: `job_id`, `mode`, `restore_layers`, `accuracy`, `accuracy_percent`, `metric_key`, `subject_metrics`, `n_samples_total`, `runtime_sec`, `worker_id`, `physical_gpu_id`, `device_name`, `prewarm_stats`. No model state / raw logits.

## Files Changed

| File | Action |
|------|--------|
| `experiments/down_layer_sensitivity/worker.py` | Created — worker CLI, validation, load-once, job loop |
| `experiments/down_layer_sensitivity/tests/test_job_manifest.py` | Created — manifest schema tests |

No production code modified. No git commit.

## Tests and Results

Environment:

```text
which python -> /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V      -> Python 3.11.13
```

Command:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py
```

Result: **20 passed in 5.61s**

### Required coverage

Valid:

1. `test_valid_baseline_manifest`
2. `test_valid_single_restore_manifest`
3. `test_valid_all_down_original_manifest`

Invalid (listed schemas):

4. `test_duplicate_job_id_raises`
5. `test_restore_layer_outside_range_raises` (`-1`, `36`, `100`)
6. `test_duplicate_layer_in_restore_list_raises`
7. `test_mode_not_smoke_or_formal_raises`
8. `test_formal_job_with_lm_limit_not_none_raises`
9. `test_smoke_job_with_lm_limit_not_2_raises` (`None`, `0`, `1`, `3`)

Plus CLI identity:

10. `test_worker_id_mismatch_raises`
11. `test_physical_gpu_id_mismatch_raises`

TDD: tests written first; RED was `ModuleNotFoundError: worker`; then implementation; GREEN 20 passed.

## Self-Review Against Brief

| Requirement | Met? | Notes |
|-------------|------|-------|
| Code only under `experiments/down_layer_sensitivity/` | Yes | |
| Fixed CLI only | Yes | |
| Manifest + CLI identity check | Yes | |
| Validation before model load | Yes | |
| Seed 31 once; no deterministic mode | Yes | |
| Load model + tokenizer once | Yes | `prewarm_group_size=8` |
| Weight metrics only if flag true | Yes | worker 0 expected by Task 4 |
| Per-job reset/assert/try-finally | Yes | exception not swallowed |
| Exact per-job JSON fields | Yes | |
| Fail on first job; non-zero exit | Yes | |
| Unit tests for listed schemas | Yes | 20 passed in bitvae |
| No git commit | Yes | |

## Concerns

1. **`weight_metrics_worker.json` path is not named in the brief.** Written to `dirname(jobs_dir)/weight_metrics_worker.json` (phase directory), not inside `jobs/`, so Task 5 inventory glob of job JSONs will not pick it up. Task 5 must read this same path.
2. **Per-job filename is `{jobs_dir}/{job_id}.json`.** Brief only says one JSON per job under `phase*/jobs/`; Task 4/5 must use the same naming.
3. **Job execution / reset / JSON write are not unit-tested.** Brief Step 7 only requires manifest validation; no GPU/MMLU in these tests.
4. **Worker metadata JSON keys** are snake_case (`device_name`, `total_memory`, `python_version`, …) derived from the field list; brief listed descriptions, not exact key names.
5. Extra fail-loud checks (job mode must equal manifest mode; restore entries must be `int`) are not in the listed schema table; they reject mixed/invalid input instead of falling back.

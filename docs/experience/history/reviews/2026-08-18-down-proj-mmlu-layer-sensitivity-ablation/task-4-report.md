> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Report: Deterministic Multi-GPU Phase-1 Scheduling

## Status

DONE

## What Was Implemented

Created `experiments/down_layer_sensitivity/run.py` as the experiment coordinator.

### CLI (fixed, no extra flags)

```text
--checkpoint_dir
--output_dir
--gpus            comma-separated physical GPU IDs
--mode            smoke|formal
```

Scientific settings are module constants, not flags:

```text
SEED=31
RANDOM_CONTROL_SEEDS=(31, 32, 33, 34, 35)
PREWARM_GROUP_SIZE=8
FORMAL_LM_LIMIT=None
SMOKE_LM_LIMIT=2
EXPECTED_DOWN_LAYERS=36
HISTORICAL_COMPRESSED_MMLU=0.4171
HISTORICAL_PRE_DOWN_MMLU=0.5199
```

### Run directory + `run_config.json`

Run ID: `YYYYMMDD_HHMMSS_{mode}`. Writes a single `run_config.json` before workers launch, including all constants, checkpoint path, selected GPUs, mode, `phase1_worker_count=W`, `status=running`. Later the same file is updated with `phase2_worker_count`, `failed_workers`, and completion status. No second config file.

### Phase-1 manifests

`build_phase1_manifests(*, selected_gpus, mode)`:

- Formal inventory: `W` per-worker baselines + `compressed_baseline_worker00_repeat` + `all_down_original` + `restore_L00..L35` = `38+W` jobs.
- Worker 0: `jobs[0]=compressed_baseline_worker00`, `jobs[1]=compressed_baseline_worker00_repeat`.
- Worker `i>0`: `jobs[0]=compressed_baseline_workerXX`.
- Scientific jobs assigned greedily after baselines: fewest jobs, tie → smaller `worker_id`; every job cost = 1.
- Only `worker_00.json` has `write_weight_metrics=true`.
- Smoke: `W=1` only; exactly four jobs in order `compressed_baseline_worker00`, `compressed_baseline_worker00_repeat`, `restore_L00`, `all_down_original`; `lm_limit=2`. Smoke never enters phase 2.

### Worker launch

`launch_phase_workers(...)` writes `phase*/manifests/worker_XX.json` and starts one `subprocess.Popen` per GPU:

```text
<sys.executable> experiments/down_layer_sensitivity/worker.py
  --checkpoint_dir ...
  --manifest_path <phase_dir>/manifests/worker_XX.json
  --jobs_dir <phase_dir>/jobs
  --worker_meta_path <phase_dir>/worker_logs/worker_XX_meta.json
  --worker_id <int>
  --physical_gpu_id <g>
```

Env = parent copy + `CUDA_VISIBLE_DEVICES=<g>`. No torchrun / DDP / FSDP / multiprocessing worker entry.

If any worker exit code != 0: wait for all, write `status=failed` + `failed_workers` into `run_config.json`, `SystemExit(1)`. Summarize / phase 2 do not run.

### Top-level call chain

Formal: parse CLI → run_dir + `run_config.json` → `build_phase1_manifests` → `launch_phase_workers(phase1)` → `summarize_phase1` → `build_phase2_manifests` → `launch_phase_workers(phase2)` → `summarize_final` → `status=completed`.

Smoke: parse CLI → run_dir + `run_config.json` → `build_phase1_manifests(mode=smoke)` → `launch_phase_workers(phase1)` → `validate_smoke` → `status=smoke_completed` → return.

`summarize_phase1` / `summarize_final` / `validate_smoke` are lazy-imported from `summarize.py` (Task 5+). `build_phase2_manifests` is a Task-6 stub (`NotImplementedError`).

## Files Changed

| File | Action |
|------|--------|
| `experiments/down_layer_sensitivity/run.py` | Created — CLI, run_config, phase-1 scheduling, worker launch, locked main() chain |
| `experiments/down_layer_sensitivity/tests/test_job_manifest.py` | Extended — formal greedy allocation for N∈{1,2,4,8}, smoke four-job order, launch command shape, failed-worker status |

No production code modified (`scripts/catlora_simple.sh` untouched). No git commit.

## Tests and Results

Environment:

```text
which python -> /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V      -> Python 3.11.13
```

Commands:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py
pytest -q experiments/down_layer_sensitivity/tests
```

Results:

- `test_job_manifest.py`: **28 passed in 5.38s**
- full `experiments/down_layer_sensitivity/tests`: **50 passed in 5.31s**

TDD: allocation/launch tests written first; RED was `ModuleNotFoundError: run`; then implementation; GREEN 28/50.

### Required coverage (Step 10)

For N GPU in `{1,2,4,8}`:

- total formal phase-1 jobs exactly `38+N`
- every worker `jobs[0] = compressed_baseline_workerXX`
- worker0 `jobs[1] = compressed_baseline_worker00_repeat`
- `all_down_original` exactly once
- `restore_L00..L35` each exactly once
- no scientific job before that worker’s baseline probe
- repeated calls produce byte-equivalent manifest job ordering
- greedy assignment equals hardcoded `EXPECTED_FORMAL_JOB_IDS` worker→job_id lists

Plus smoke four-job order, smoke `W!=1` reject, launch argv/`CUDA_VISIBLE_DEVICES` contract, failed-worker `run_config` update.

## Self-Review Against Brief

| Requirement | Met? | Notes |
|-------------|------|-------|
| Code only under `experiments/down_layer_sensitivity/` | Yes | |
| CLI only 4 experiment-control flags | Yes | |
| Constants not exposed as flags | Yes | |
| Run ID + single `run_config.json` | Yes | includes `phase1_worker_count` |
| Formal inventory `38+W` | Yes | |
| Worker0 baseline then repeat first | Yes | |
| Other workers baseline first | Yes | |
| Greedy fewest-jobs / smaller `worker_id` | Yes | cost=1 |
| Manifest paths `phase1/manifests/worker_XX.json` | Yes | |
| `write_weight_metrics` only worker 0 | Yes | |
| One Popen per GPU; fixed argv; `sys.executable` | Yes | |
| `CUDA_VISIBLE_DEVICES=<g>` only env overlay | Yes | |
| Wait all; failed_workers; non-zero; no summarize | Yes | |
| Locked formal/smoke `main()` chain | Yes | |
| Smoke never phase 2; W=1; four jobs | Yes | |
| Unit tests with hardcoded expected lists | Yes | |
| No git commit | Yes | |

## Concerns

1. **`summarize.py` does not exist yet.** Formal/smoke `main()` will `ImportError` after a successful phase-1 launch until Task 5 implements `summarize_phase1` / `validate_smoke`. This is required by the locked call chain, not a missing Task-4 scheduler.
2. **`build_phase2_manifests` is a stub.** Task 6 owns it. Formal `main()` currently passes the full `selected_gpus` list into the phase-2 launch; Task 6 must slice to `W2=min(len(selected_gpus), 9)` when it returns fewer manifests, or launch will reject a length mismatch.
3. **`validate_smoke` signature is assumed** as `validate_smoke(*, run_dir, selected_gpus)`. Brief named the call but not the signature; Task 5/9 should keep this or update `run.py`.
4. **`run_config.json` key names** were not schema-frozen beyond “all constants + checkpoint + GPUs + mode + `phase1_worker_count`”. Keys are snake_case mirrors of the Python constants.
5. **`launch_phase_workers` infers `run_dir` as `dirname(phase_dir)`** when writing `failed_workers`. Correct for `run_dir/phase1` and `run_dir/phase2`.
6. **Same-second run ID collision** uses `os.makedirs(..., exist_ok=False)` and fails loud rather than overwriting.
7. Pre-existing dirty `scripts/catlora_simple.sh` was not modified.

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-3-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Report: Pin Candidate Training to the Parent Python Interpreter

## Status

**Complete** — uncommitted working tree changes only (no commit per instructions).

## Changes

### `mix_bit/candidate_pool.py`

- Added `_resolved_python_executable()` using `Path(sys.executable).resolve()` with existence check.
- `build_trial_command` now emits `[script, gpu_id, python_executable, *cat_train_args]`.
- `trial_spec_to_dict` records `python_executable` from `command[2]`.
- `run_candidate_pool` writes `python_executable` into `scheduler_meta.json`.

### `mix_bit/scripts/train_candidate_single.sh`

- argv contract: GPU id → argv[1], resolved Python → argv[2], cat_train args → argv[3:].
- Validates executable bit; exits 2 on usage / non-executable errors.
- `exec "${PYTHON_EXECUTABLE}" tools/cat_train.py "$@"` (no bare `python`).
- Preserved env exports: `PYTHONPATH`, `PYTORCH_CUDA_ALLOC_CONF`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, `TOKENIZERS_PARALLELISM`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`.

### `mix_bit/tests/test_candidate_pool.py`

Added 6 tests (TDD-first):

- `test_trial_command_passes_resolved_sys_executable_as_second_argument`
- `test_scheduler_meta_records_python_executable`
- `test_trial_spec_records_python_executable`
- `test_candidate_shell_requires_python_argument`
- `test_candidate_shell_rejects_non_executable_python`
- `test_candidate_shell_executes_explicit_interpreter`

### `mix_bit/README.md`

Documented that parent CLI must start under `bitvae` Python and workers inherit the same interpreter via `sys.executable`.

## Tests

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_candidate_pool.py -q
# 29 passed in 5.05s
```

Shell smoke (no model):

```bash
mix_bit/scripts/train_candidate_single.sh 4 /bin/echo --sentinel
# stdout: tools/cat_train.py --sentinel
```

## Concerns

- `python_executable` in metadata is provenance-only; resume still keyed on artifact/model/recipe/hash contract (Task 1–2).
- If parent is launched with a wrapper or symlinked interpreter, workers follow the resolved path at pool start time.

## Commits

None (skipped per task instructions).

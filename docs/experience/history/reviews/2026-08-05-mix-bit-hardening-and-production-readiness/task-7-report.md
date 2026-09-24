> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-7-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Report: Make Spawn Worker Startup and Runtime Fail Fast

## Scope

- Modified: `mix_bit/cost_table.py`
- Modified: `mix_bit/tests/test_cost_table.py` (tests were already present from prior TDD scaffolding; no edits required)

## Constants Added

```python
BASELINE_STARTUP_TIMEOUT_SECONDS = 900.0
WORKER_STARTUP_TIMEOUT_SECONDS = 900.0
RESULT_QUEUE_POLL_SECONDS = 1.0
WORKER_JOIN_TIMEOUT_SECONDS = 30.0
```

No new CLI parameters; fixed production-safe values.

## Helpers Added (`mix_bit/cost_table.py`)

- `_dead_process_descriptions(processes)` — stable `pid=X exitcode=Y` strings for non-alive processes.
- `_terminate_and_join(process, join_timeout=5.0)` — terminate + join used by error paths.
- `_wait_for_single_process_message(*, process, result_queue, expected_type, timeout_seconds, label)` — polls `result_queue.get(timeout=min(RESULT_QUEUE_POLL_SECONDS, remaining))`; raises on `failure` (with traceback), raises on dead child with pid/exitcode, raises `TimeoutError` after total deadline (terminate + join first), and after receiving the expected message joins with `WORKER_JOIN_TIMEOUT_SECONDS` then fails if still alive or exitcode != 0.
- `_wait_for_workers_ready(*, processes, result_queue, timeout_seconds)` — tracks `logical_id` ready set; rejects duplicates; fails on any dead process before all ready; fails on `failure`; raises `TimeoutError` after total deadline (terminates all survivors).

## Production Code Changes

- `_ensure_baseline_per_sample_spawn`: removed blocking `result_queue.get()` + manual join/terminate; now calls `_wait_for_single_process_message` with `BASELINE_STARTUP_TIMEOUT_SECONDS` and `expected_type="baseline_ready"`.
- `run_cost_search_scheduler`: replaced the `while ready < len(processes)` blocking `result_queue.get()` loop with `_wait_for_workers_ready` using `WORKER_STARTUP_TIMEOUT_SECONDS`.
- Runtime loop `queue.Empty` branch: changed from "fail only when ALL processes dead" to "fail when ANY process is dead", capturing `failure_detail = "; ".join(_dead_process_descriptions(dead))`. Runtime `failure` messages also populate `failure_detail` (with traceback). Final `RuntimeError` includes `failure_detail`.
- `drain_job_queue_and_stop_workers` join timeout now uses `WORKER_JOIN_TIMEOUT_SECONDS`.

## TDD Evidence

### RED (before implementation)

Focused subset (excluding the hanging runtime test, which would block indefinitely under the old implementation):

```
FAILED mix_bit/tests/test_cost_table.py::test_baseline_wait_fails_when_child_exits_without_message
FAILED mix_bit/tests/test_cost_table.py::test_baseline_wait_surfaces_failure_traceback
FAILED mix_bit/tests/test_cost_table.py::test_baseline_wait_times_out_and_terminates_child
FAILED mix_bit/tests/test_cost_table.py::test_baseline_wait_accepts_exact_ready_message
FAILED mix_bit/tests/test_cost_table.py::test_baseline_wait_fails_when_process_exits_nonzero_after_ready
FAILED mix_bit/tests/test_cost_table.py::test_worker_ready_wait_rejects_duplicate_logical_id
FAILED mix_bit/tests/test_cost_table.py::test_worker_ready_wait_fails_if_one_worker_dies
FAILED mix_bit/tests/test_cost_table.py::test_worker_ready_wait_times_out_and_terminates_all
FAILED mix_bit/tests/test_cost_table.py::test_worker_ready_wait_accepts_all_unique_workers
FAILED mix_bit/tests/test_cost_table.py::test_worker_ready_wait_surfaces_failure_message
10 failed, 30 deselected in 5.34s
```

All failures were `ImportError: cannot import name '_wait_for_single_process_message' / '_wait_for_workers_ready'`, confirming the helpers did not exist. The runtime partial-death test (`test_runtime_partial_death_fails_and_does_not_keep_polling`) hung under the old implementation because the runtime loop only failed when every worker was dead — direct evidence the old blocking implementation cannot satisfy the test.

### GREEN (after implementation)

Focused Task 7 subset:

```
...........                                                              [100%]
11 passed, 29 deselected in 5.05s
```

Full `mix_bit/tests/test_cost_table.py` suite:

```
........................................                                 [100%]
40 passed in 6.21s
```

No regressions. Linter: clean.

## Test Approach

All tests use ~0.01s timeouts via `_FakeProc` / `_FakeQueue` / `_ScriptedQueue` helpers; no test waits the 900s production deadline. The runtime partial-death test pre-creates `baseline_per_sample.npz` so `_ensure_baseline_per_sample_spawn` returns early, then uses a scripted queue that serves two `ready` messages and raises `Empty` (killing worker 0 with exitcode 137 on the first empty), verifying the scheduler fails fast with `pid=100 exitcode=137` in the error message instead of polling forever.

## Commits

None. Workspace changes left uncommitted per task instructions.

## Concerns

None.

---

# Round 1/5 Fix Report

## Findings Addressed

### Important #1: Startup/runtime failure paths did not drain surviving workers

**Problem:** `_wait_for_workers_ready` raises on worker death / failure / duplicate / unexpected-type without terminating survivors, and `run_cost_search_scheduler` had no try/finally around the ready wait, so `drain_job_queue_and_stop_workers` never ran on startup failure → GPU-holding daemons could linger. Violated brief Step 8.

**Fix (smallest correct approach):** Wrapped the ready wait + runtime loop in `run_cost_search_scheduler` with `try/finally`; the `finally` block always calls `drain_job_queue_and_stop_workers(...)` (which joins, then terminates any still-alive worker). This covers every exit path: startup failure, runtime failure, and success. No need to push termination into `_wait_for_workers_ready` itself, keeping the helper pure.

### Minor #2: `_wait_for_single_process_message` failure/unexpected-type paths left child alive

**Fix:** Added `_terminate_and_join(process)` before the `raise` on both the `failure` message path and the unexpected-type path, so the child is never left running when the helper raises.

## Test Added

`test_startup_failure_terminates_remaining_workers` (scheduler-level): worker 0 already dead before ready, worker 1 alive. Asserts the scheduler raises `RuntimeError` matching "died before ready" and that the still-alive sibling (`procs[1]`) is drained (`is_alive() is False`, `_joined is True`). This proves the try/finally drain runs on startup failure.

## Test Results

Full suite after fixes:

```
.........................................                                [100%]
41 passed in 6.10s
```

(40 prior + 1 new). Linter: clean. No regressions.

## Commits

None.


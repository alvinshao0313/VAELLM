> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-8-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Report: `--pool_manifest` Authoritative Candidate Pool Location

## Status
Complete. All focused tests pass; no commits made (per workspace rule).

## Commits
None (workspace rule forbids auto-commit).

## Test summary
- New `mix_bit/tests/test_cli_pool_manifest.py`: 11 tests
  - 6 helper tests (`test_index_from_manifest_*`, `test_build_index_write_manifest_false_creates_no_manifest`)
  - 5 CLI tests (`test_prepare_baseline_cli_*`, `test_cost_cli_*`, `test_solve_cli_*`, `test_assemble_cli_*`, `test_validate_cli_*`)
- New `mix_bit/tests/test_cost_table.py::test_spawned_worker_args_preserve_pool_manifest_path`: 1 test
- Focused suite (`test_cli_pool_manifest.py` + `test_cost_table.py` + `test_checkpoint_pool.py`): 73 passed
- Regression (`test_solver.py` + `test_tiny_integration.py`): 12 passed
- TDD confirmed: tests failed on old code (helper missing) before implementation.

## Changes
- `mix_bit/checkpoint_pool.py`
  - Added `write_manifest: bool = True` to `build_candidate_pool_index`; manifest written only when enabled.
  - Extracted `_candidate_manifest_payload(index)` (canonical payload from validated index).
  - Added `build_candidate_pool_index_from_manifest(resolved, inventory, manifest_path)`: resolves to absolute, requires name `candidate_manifest.json`, requires existence, records `supplied_sha256`, validates `kind`, sets `pool_root = parent`, calls builder with `write_manifest=False`, requires `index.manifest_path == manifest_path`, requires supplied JSON == expected payload (artifact order, absolute paths, SHAs), re-checks on-disk SHA, returns index. No rewrite on mismatch.
  - Added `hashlib` import.
- `mix_bit/cost_table.py`
  - `_baseline_init_process_main` and `_worker_process_main` now use `build_candidate_pool_index_from_manifest` with `pool_manifest_path` from args (no canonical rebuild).
  - `_ensure_baseline_per_sample_spawn` gains `pool_index` param and writes `pool_manifest_path` into init_args.
  - `run_cost_search_scheduler` computes `pool_manifest_path = str(Path(pool_index.manifest_path).resolve())`, passes `pool_index` to baseline spawn, and writes `pool_manifest_path` into every worker_args.
- `mix_bit/cli/prepare_uniform_baseline.py`: calls helper directly; removed redundant manifest path/existence checks; dropped unused `Path` import.
- `mix_bit/cli/compute_cost_table.py`: calls helper directly; removed redundant checks.
- `mix_bit/cli/assemble_mixed_model.py`: calls helper directly; removed redundant checks.
- `mix_bit/cli/validate_mixed_model.py`: calls helper directly; removed redundant checks; dropped unused `Path` import.
- `mix_bit/cli/solve_allocation.py`: imports both builders; if `--pool_manifest` provided uses helper, else canonical build; help text "optional; provided path is authoritative"; removed redundant post-check.
- `mix_bit/README.md`: added "自定义 candidate pool 根目录" section documenting `--output_root X` then `--pool_manifest X/candidate_manifest.json` flow and spawn worker manifest propagation.

## Concerns
- `solve_allocation.py` removed the `if not manifest_path.is_file()` post-check. Both paths (helper validates existence; canonical writes) guarantee the manifest exists, so the check was redundant. Behavior preserved for the canonical path; the helper path now relies on the helper's own FileNotFoundError.
- Spawn worker test uses fake procs that die after ready to terminate the scheduler loop (avoids hang); consistent with existing Task 7 test patterns.
- `build_candidate_pool_index_from_manifest` compares parsed JSON objects (not raw bytes), so formatting differences (indent/spacing) are tolerated but field/value/artifact-order differences are rejected, matching the brief's "完全相等" requirement on payload content.

## Report path
/home/shaoyuantian/program/VAELLM/.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-8-report.md

---

## Fix Round 1/5

**Finding (Important / brief Step 4 gap):** `test_spawned_worker_args_preserve_pool_manifest_path` pre-created `baseline_per_sample.npz` so `_ensure_baseline_per_sample_spawn` returned early and never captured baseline init_args. Brief Step 4 requires capturing **both** baseline and worker args.

**Fix:** Rewrote `test_spawned_worker_args_preserve_pool_manifest_path` in `mix_bit/tests/test_cost_table.py` to exercise the real baseline spawn path:
- No longer pre-creates `baseline_per_sample.npz` (asserts it is absent so the baseline spawn fires).
- Added `_BaselineResultQueue` that serves a `baseline_ready` message on first `get` and materializes the npz file as a side effect (the fake baseline proc never runs), so `_ensure_baseline_per_sample_spawn`'s post-message existence check passes.
- Single shared `_CapturingSpawnCtx` instance returned by `mp.get_context` so Queue()/Process() call counters persist across the baseline and scheduler spawn contexts.
- Queue() call order: 1) baseline result_queue, 2) job_queue, 3) scheduler result_queue. Process() call order: 1) baseline, 2-3) workers.
- Captures all three arg dicts; asserts `len(captured_args) == 3`, baseline init_args and both worker_args each contain `pool_manifest_path`, all resolve to the same absolute path as `pool_index.manifest_path`.

**No production code changed** — `from_manifest` immutability and worker wiring already correct; this is a test-only fix.

**Verification:**
- `test_spawned_worker_args_preserve_pool_manifest_path`: 1 passed.
- Focused suite (`test_cli_pool_manifest.py` + `test_cost_table.py` + `test_checkpoint_pool.py`): 73 passed.
- No linter errors.

**Commits:** none.

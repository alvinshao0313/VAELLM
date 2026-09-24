> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-1-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Report: Isolated Core and Layer-State Semantics

## Status

DONE

## What Was Implemented

Created the isolated experiment core under `experiments/down_layer_sensitivity/` with the following public API:

| Symbol | Purpose |
|--------|---------|
| `DownLayerRef` | Frozen dataclass: `layer_idx`, `name`, `module` |
| `discover_down_layers(model)` | Discover exactly 36 Qwen3-8B `down_proj` VAELinear refs with validation |
| `reset_all_vae_to_compressed(model)` | Force all VAELinear to compressed path; reject `always_use_original` |
| `set_down_restore_set(down_layers, restore_layers)` | Apply leave-one-out / multi-restore temporary flags |
| `assert_down_restore_set(down_layers, restore_layers)` | Strict post-condition check for temporary/original state |
| `unload_non_down_original_weights(model, down_names)` | Memory optimization: unload non-down originals with count dict |
| `prewarm_compressed_weights(model, device, group_size)` | One-time grouped decode cache priming via existing prewarm |
| `compute_down_weight_metrics(down_layers)` | NMSE diagnostics from prewarmed `_cached_weight` only |
| `load_worker_model(checkpoint_dir, device, prewarm_group_size)` | Full worker bootstrap in fixed order |

### Semantics locked down

- `_DOWN_RE = r"^model\.layers\.(\d+)\.mlp\.down_proj$"` for name parsing
- `discover_down_layers` requires `num_hidden_layers == 36`, indexes `0..35`, all `VAELinear`, `original_weight is not None`, `always_use_original == False`
- `set_temporary(True)` = compressed; `set_temporary(False)` = original path
- Weight metrics use `_cached_weight` (no re-decode); fields: `layer_idx`, `name`, `numel`, `mse`, `nmse`, `relative_fro_error`, `original_rms`, `error_rms`
- `load_worker_model` order: load checkpoint → eval → reset → discover → unload → to(device) → prewarm → assert all compressed → return dict

## Files Changed

| File | Action |
|------|--------|
| `experiments/__init__.py` | Created (empty) — makes `experiments` importable from repo root |
| `experiments/down_layer_sensitivity/__init__.py` | Created — re-exports core API |
| `experiments/down_layer_sensitivity/core.py` | Created — all core logic |
| `experiments/down_layer_sensitivity/tests/__init__.py` | Created (empty) |
| `experiments/down_layer_sensitivity/tests/test_core.py` | Created — unit tests |

No production code modified.

## Tests and Results

Environment:

```text
which python -> /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V      -> Python 3.11.13
```

Command:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_core.py
```

Result: **13 passed in 8.77s**

### Discovery tests (6 required cases — all present, not merged)

1. `test_valid_contiguous_down_refs_are_sorted_by_layer_index`
2. `test_missing_layer_raises_value_error`
3. `test_duplicate_non_contiguous_layer_index_raises_value_error`
4. `test_matched_down_module_not_vae_linear_raises_type_error`
5. `test_down_original_weight_none_raises_value_error`
6. `test_always_use_original_true_raises_value_error`

### Additional tests

- `test_no_cross_job_state_leakage` — exact reset/restore sequence from brief
- `test_unknown_restore_layer_raises_value_error`
- `test_unloading_does_not_change_compressed_forward`
- `test_unload_counts_and_down_original_kept`
- `test_unload_retained_original_without_protection_raises_runtime_error`
- `test_uses_prewarmed_cache_without_redecode`
- `test_missing_cache_raises_runtime_error`

Synthetic tests use minimal module trees with real small `VAELinear` instances; no production class changes.

## Self-Review Against Brief

| Requirement | Met? | Notes |
|-------------|------|-------|
| All code under `experiments/down_layer_sensitivity/` | Yes | Plus minimal `experiments/__init__.py` for import path |
| No production modifications | Yes | |
| Exact interfaces and semantics | Yes | |
| 6 discovery failure cases | Yes | Separate test methods, names match brief |
| State leakage test sequence | Yes | Uses `reset_all_vae_to_compressed` between jobs |
| Unload counts dict keys | Yes | Exact five count fields |
| Prewarm via `iter_named_vae_module_refs` + `prime_named_vae_linear_cache` | Yes | `failed == 0` enforced |
| Weight metrics from cache only | Yes | Raises if `_cached_weight is None` |
| `load_worker_model` fixed order | Yes | Returns `{model, meta, down_layers, prewarm_stats}` |
| pytest in bitvae | Yes | All pass |

## Concerns

1. **`experiments/__init__.py` not listed in brief** — Required so `from experiments.down_layer_sensitivity.core import ...` resolves when pytest runs from repo root. Without it, collection fails with `ModuleNotFoundError`.
2. **`load_worker_model` not integration-tested** — Brief Step 11 only requires unit tests; full checkpoint load would need the multi-GB final model and is deferred to later smoke/formal tasks.
3. **Duplicate vs non-contiguous** — Brief lists one combined case; test uses missing layer 0 (non-contiguous indexes). A true duplicate index is structurally impossible with the fixed `_DOWN_RE` naming scheme unless module tree is manually corrupted.

## Commits

None (per project rules).

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Report: Split CPU-Offloaded EAKLD Teacher Scalars by Region

## Status

**DONE**

## Summary

Extended `TeacherTargetBatch` with three prompt-region scalar fields
(`eakld_prompt_gamma_cpu`, `teacher_prompt_entropy_mean_cpu`,
`teacher_prompt_valid_token_count_cpu`) and reset them in `clear()`. Replaced
the E2E trainer's `_build_distill_token_mask()` with
`_build_distill_token_regions()` returning the shared `DistillTokenRegions`
dataclass (no telemetry recording in the builder). In
`_build_cpu_teacher_targets()`, a single CPU copy of the full teacher logits
is now reused for both regional entropy/gamma computations: response scalars
are always computed when logits are required, and prompt scalars are computed
only when `prompt_kd_weight > 0.0` (otherwise the prompt path is skipped and
the new fields stay unset). Updated both dispatcher call sites
(`_compute_legacy_dense_loss`, `_compute_teacher_first_cpu_loss`) to use the
new region builder's `response_mask`.

## Files Changed

### `compressed_e2e_fintuning/teacher_targets.py`

- Added three `Optional[torch.Tensor]` prompt-region scalar fields to
  `TeacherTargetBatch`.
- Extended `clear()` to reset all three new fields.

### `compressed_e2e_fintuning/trainer.py`

- Imports: replaced `build_distill_token_mask` with
  `build_distill_token_regions` + `DistillTokenRegions`.
- Replaced `_build_distill_token_mask()` method with
  `_build_distill_token_regions()` (keyword-only, returns
  `DistillTokenRegions`, no telemetry).
- `_build_cpu_teacher_targets()`: copy teacher logits to CPU once, then
  compute response entropy/gamma/count on that CPU copy with
  `regions.response_mask` (stored in existing fields). When
  `prompt_kd_weight > 0.0`, compute prompt entropy/gamma/count on the same
  CPU copy with `regions.prompt_mask` (stored in new fields). When
  `prompt_kd_weight == 0.0`, skip the prompt path entirely.
- `_compute_legacy_dense_loss()` and `_compute_teacher_first_cpu_loss()`:
  call site switched to `self._build_distill_token_regions(...).response_mask`.

### `tests/test_e2e_teacher_first.py`

- Added `prompt_kd_weight` parameter to `_build_trainer` (default `0.0`).
- Added `_inputs_with_prompt_prefix()` helper (2-token prompt prefix → -100).
- Added `_install_counting_entropy_helper()` monkeypatch fixture capturing
  logits id, mask, and confidence_k per call.
- `test_cpu_teacher_targets_zero_prompt_weight_response_only`: asserts
  response scalars populated, prompt scalars absent, helper called once.
- `test_cpu_teacher_targets_positive_prompt_weight_both_regions`: asserts
  both scalar sets populated, helper called twice with distinct binary
  masks, each valid count equals its region-mask sum, and both calls reuse
  the same single CPU logits copy.

### `tests/smoke/test_one_step_train_smoke.py`

- `test_cpu_offload_eakld_one_step_trainer_smoke`: updated
  `entropy_calls["n"]` assertion from `== 1` to `== 2` and refreshed the
  comment, since positive `prompt_kd_weight` now triggers a second
  (prompt-region) entropy/gamma call inside the CPU target builder. This
  smoke test was previously failing with a `TypeError` from Task 1's
  `prompt_kd_weight` removal; it now passes.

## Self-Review

| Requirement | Met? | Notes |
|---|---|---|
| Add 3 prompt-region scalar fields | Yes | `eakld_prompt_gamma_cpu`, `teacher_prompt_entropy_mean_cpu`, `teacher_prompt_valid_token_count_cpu` |
| `clear()` resets new fields | Yes | All three set to `None` |
| Replace `_build_distill_token_mask` with `_build_distill_token_regions` | Yes | Returns `DistillTokenRegions`; no telemetry |
| Always compute response scalars when logits required | Yes | On CPU copy with `response_mask` |
| Prompt scalars only when prompt weight > 0 | Yes | Guarded by `self.prompt_kd_weight > 0.0` |
| Zero weight leaves prompt fields unset, skips prompt path | Yes | Tested |
| Reuse single CPU copy of teacher logits | Yes | One `copy_detached_tensor_to_cpu` call; both helper calls share logits id |
| Test zero weight: response populated, prompt absent, helper once | Yes | `test_cpu_teacher_targets_zero_prompt_weight_response_only` |
| Test positive weight: both populated, helper twice, distinct masks, counts = mask sums | Yes | `test_cpu_teacher_targets_positive_prompt_weight_both_regions` |
| No git commit | Yes | Working tree only |

## Test Summary

```
PYTHONPATH=. pytest tests/test_e2e_teacher_first.py -q
→ 14 passed in 9.37s
```

```
PYTHONPATH=. pytest tests/test_e2e_teacher_first.py tests/test_teacher_target_offload.py tests/test_distill_losses.py -q
→ 89 passed in 8.98s
```

```
PYTHONPATH=. pytest tests/smoke/test_one_step_train_smoke.py -q
→ 5 passed in 5.60s
```

## Concerns

1. **Temporary wiring for Task 5**: `_compute_teacher_first_cpu_loss` still
   calls `compute_dense_loss_from_offloaded_teacher` with only
   `mask=regions.response_mask` (no `prompt_mask`/`prompt_kd_weight`). The
   precomputed prompt scalars (`eakld_prompt_gamma_cpu` etc.) are populated
   by the builder but not yet consumed by the dispatcher. This means the
   CPU-offload path currently applies response-region EAKLD only; the
   prompt-region loss combination in the dispatcher is deferred to Task 5,
   which will wire the dispatcher to use the precomputed prompt scalars
   instead of recomputing them.

2. **Legacy dense path** (`teacher_output_offload="none"`,
   `_compute_legacy_dense_loss`) now uses `regions.response_mask` only.
   `prompt_kd_weight > 0` is silently ignored on this path. This matches the
   post-Task-1 contract (fractional prompt mask removed) and is outside
   Task 4's CPU-offload-scalar scope; wiring prompt-region combination into
   the legacy dense dispatcher is a separate concern.

3. Updated one assertion in `tests/smoke/test_one_step_train_smoke.py`
   (outside the brief's file list) because the entropy/gamma call count
   changed from 1 to 2 for positive prompt weight — a direct consequence of
   this task's region split. Noted here for transparency.

## Commits

None (per instructions).

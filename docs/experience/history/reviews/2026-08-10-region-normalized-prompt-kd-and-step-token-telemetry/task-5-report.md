> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-5-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Report: Keep E2E Dense and CPU-Offload Paths Mathematically Identical

## Status

**DONE**

## Summary

Wired both E2E student loss paths so the legacy dense dispatcher
(`compute_dense_loss_from_logits`) and the CPU-offload dispatcher
(`compute_dense_loss_from_offloaded_teacher`) consume the same shared
`DistillTokenRegions` (response mask + prompt mask) and `prompt_kd_weight`,
and the offload dispatcher now consumes the precomputed prompt-region
teacher scalars (`eakld_prompt_gamma_cpu`, `teacher_prompt_entropy_mean_cpu`,
`teacher_prompt_valid_token_count_cpu`) populated by
`_build_cpu_teacher_targets` in Task 4, instead of recomputing prompt gamma
from `teacher_logits_cpu`.

## Files Changed

### `e2e_common/dense_loss.py`

- `compute_dense_loss_from_offloaded_teacher`: added three keyword args
  `teacher_prompt_gamma_cpu`, `teacher_prompt_entropy_mean_cpu`,
  `teacher_prompt_valid_token_count_cpu` matching the `TeacherTargetBatch`
  prompt-region fields.
- Added validation: when `prompt_kd_weight > 0`, all three prompt scalar
  values are required (in addition to `prompt_mask`); missing data raises
  `ValueError` explicitly — no silent response-only fallback.
- Replaced the inline `compute_teacher_entropy_mean_and_gamma` recomputation
  inside `_prompt_eakld` with the precomputed `teacher_prompt_gamma_cpu`.
  Response EAKLD still uses `mask` + `teacher_gamma_cpu` + response telemetry
  (`telemetry_out=telemetry_out`); prompt EAKLD uses `prompt_mask` +
  `teacher_prompt_gamma_cpu` with `telemetry_out=None` so response telemetry
  is preserved. Regional combination happens only after both regional means
  are computed (via `_combine_region_loss`).
- `eakld_kd` still mixes CE exactly once after regional EAKLD combination.
- Removed the now-unused `compute_teacher_entropy_mean_and_gamma` import.

### `compressed_e2e_fintuning/trainer.py`

- `_compute_legacy_dense_loss`: builds student regions once
  (`self._build_distill_token_regions(inputs, logits)`) and passes
  `mask=regions.response_mask`, `prompt_mask=regions.prompt_mask`,
  `prompt_kd_weight=self.prompt_kd_weight` to `compute_dense_loss_from_logits`.
  Previously only `mask=response_mask` was passed and `prompt_kd_weight > 0`
  was silently ignored on this path.
- `_compute_teacher_first_cpu_loss`: builds student regions once and passes
  `mask=regions.response_mask`, `prompt_mask=regions.prompt_mask`,
  `prompt_kd_weight=self.prompt_kd_weight`, plus the three prompt-region
  scalars from `targets` to `compute_dense_loss_from_offloaded_teacher`.
  Added a guard: when `prompt_kd_weight > 0` and any prompt scalar on the
  built `TeacherTargetBatch` is `None`, raise `RuntimeError` (the builder is
  expected to populate them for positive weight).

### `tests/test_distill_losses.py`

Added a `_prompt_region_scalar_fixtures` helper plus 6 new tests:

- `test_offloaded_eakld_positive_prompt_weight_matches_dense_value_and_gradient`
  (parametrized over chunk sizes 1/2/3): full EAKLD, positive prompt weight;
  asserts offload loss and student gradients match the dense path within
  existing tolerances (rtol=5e-6/atol=5e-6 value, rtol=1e-5/atol=1e-5 grad).
- `test_offloaded_eakld_topk_positive_prompt_weight_matches_dense_value_and_gradient`
  (parametrized over chunk sizes 1/3/8): small top-k EAKLD variant; same
  value + gradient equality.
- `test_offloaded_zero_prompt_weight_matches_response_only_value_and_gradient`:
  zero weight with `prompt_mask` set but no prompt scalars matches the
  response-only offload path in value and gradient.
- `test_offloaded_empty_prompt_mask_positive_weight_remains_finite`: empty
  prompt mask + positive weight + scalars computed on the empty mask stays
  finite with finite student grads (offload analogue of the dense
  `test_empty_prompt_mask_eakld_positive_weight_remains_finite`).
- `test_offloaded_positive_weight_without_prompt_scalars_is_error`: positive
  weight with `prompt_mask` but missing prompt scalars raises `ValueError`.
- `test_offloaded_eakld_kd_positive_prompt_weight_mixes_ce_once`: `eakld_kd`
  with positive prompt weight mixes CE exactly once (gradient w.r.t. the CE
  tensor equals `1 - alpha`).

## Self-Review

| Requirement | Met? | Notes |
|---|---|---|
| Legacy dense path builds regions once, passes response mask + prompt mask + weight | Yes | `_compute_legacy_dense_loss` |
| CPU-offload path builds regions once, passes both masks + response + prompt scalar sets | Yes | `_compute_teacher_first_cpu_loss` |
| Dispatcher extended with prompt gamma / entropy mean / valid-token count args matching `TeacherTargetBatch` | Yes | Three new kwargs on `compute_dense_loss_from_offloaded_teacher` |
| Positive prompt weight requires prompt mask + all three prompt scalars; missing = explicit error | Yes | `ValueError` in dispatcher; `RuntimeError` guard in trainer |
| Offloaded response EAKLD uses response telemetry; prompt EAKLD does not overwrite it | Yes | Prompt call passes `telemetry_out=None` |
| Regional means computed before combination | Yes | `_combine_region_loss` after both `_response_eakld` and `_prompt_eakld` |
| `eakld_kd` mixes CE once after regional combination | Yes | Tested |
| Dense-vs-offload equality tests with positive weight (full + top-k) | Yes | Value + student grad equality |
| Zero-weight test: prompt scalars unnecessary, matches response-only | Yes | Value + grad equality |
| Empty prompt mask with positive weight remains finite (offload path) | Yes | New test |
| No git commit | Yes | Working tree only |
| No shell hyperparam / CLI name changes | Yes | None touched |

## Test Summary

```
PYTHONPATH=. pytest tests/test_distill_losses.py -q
→ 67 passed in 5.70s
```

```
PYTHONPATH=. pytest tests/test_e2e_teacher_first.py tests/test_teacher_target_offload.py \
  tests/smoke/test_loss_pipeline_smoke.py tests/smoke/test_one_step_train_smoke.py -q
→ 40 passed in 8.51s
```

```
PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py -q
→ 28 passed in 6.08s
```

`test_distill_losses.py` grew from 60 to 67 passed (7 new test cases, several
parametrized). No pre-existing tests required modification.

## Concerns

1. The offload dispatcher no longer recomputes prompt gamma from
   `teacher_logits_cpu`. Callers that previously relied on the dispatcher
   recomputing prompt gamma (passing `prompt_mask` + `prompt_kd_weight > 0`
   without prompt scalars) now raise `ValueError`. No existing test or
   production call site used that path — the only production caller is
   `_compute_teacher_first_cpu_loss`, which now supplies the scalars from
   the `TeacherTargetBatch` built by Task 4.

2. The legacy dense path now honors `prompt_kd_weight > 0` (previously
   silently ignored). This is the intended behavior change per the brief;
   the dense EAKLD smoke test (`test_dense_eakld_one_step_trainer_smoke`)
   already sets `prompt_kd_weight=0.1` and continues to pass.

3. `eakld_confidence_k` remains a validated parameter on the offload
   dispatcher for API symmetry, but is no longer read inside the dispatcher
   body (the precomputed prompt gamma already reflects it). The response
   path does not need it because `compute_eakld_from_cpu_teacher_logits`
   receives the precomputed response gamma directly.

## Commits

None (per instructions).

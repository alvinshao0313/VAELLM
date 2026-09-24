> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-5-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Review: Dense vs CPU-Offload Mathematical Identity at Positive Prompt Weight

## Spec Compliance: ✅

| Brief item | Status |
|---|---|
| Legacy dense builds regions once, passes response mask + prompt mask + weight | ✅ `_compute_legacy_dense_loss` |
| CPU-offload builds regions once, passes both masks + response + prompt scalar sets | ✅ `_compute_teacher_first_cpu_loss` |
| Offload dispatcher extended with prompt gamma / entropy mean / valid-token count args | ✅ Three new kwargs |
| Positive weight requires prompt mask + all three scalars; missing = explicit error | ✅ `ValueError` in dispatcher + `RuntimeError` guard in trainer |
| Response EAKLD uses response telemetry; prompt EAKLD does not overwrite | ✅ `_prompt_eakld` passes `telemetry_out=None` |
| Regional means computed before combination | ✅ `_combine_region_loss` after both |
| `eakld_kd` mixes CE once after regional combination | ✅ Tested (dense + offload) |
| Dense-vs-offload equality tests, positive weight, full + top-k, value + grads | ✅ Chunks 1/2/3 + 1/3/8 |
| Zero-weight test: scalars unnecessary, matches response-only | ✅ Value + grad |
| No auto-commit | ✅ Working tree only |

## Quality: High

- `_validate_prompt_weight` + `_combine_region_loss` are clean shared helpers; lazy `prompt_loss_fn` means zero-weight never touches prompt scalars — mathematically sound and matches the zero-weight test.
- Prompt EAKLD call passes `teacher_entropy_mean=None` / `teacher_valid_token_count=None` with `telemetry_out=None`; verified in `_compute_eakld_from_cpu_teacher_logits_impl` that those fields are only read when `telemetry_out is not None`, so the math is unaffected.
- `test_cpu_offload_eakld_one_step_trainer_smoke` asserts `entropy_calls["n"] == 2` — strong guard confirming the dispatcher no longer recomputes prompt gamma from `teacher_logits_cpu`.
- Equality tolerances (rtol=5e-6/atol=5e-6 value, rtol=1e-5/atol=1e-5 grad) match existing conventions.

## Findings

1. **Naming asymmetry (non-blocking):** Dispatcher arg `teacher_prompt_gamma_cpu` vs `TeacherTargetBatch` field `eakld_prompt_gamma_cpu`. This mirrors the existing response-side convention (`teacher_gamma_cpu` vs `eakld_gamma_cpu`), so it is consistent, but the brief said "corresponding exactly." Trainer wiring is correct.
2. **Response gamma device change (non-blocking):** `_build_cpu_teacher_targets` now computes the *response* gamma from `logits_cpu` (CPU) instead of `teacher_logits` (GPU). Mathematically equivalent because `compute_teacher_entropy_mean_and_gamma` converts to fp32 before entropy, and equality tests pass. Worth noting as a subtle behavior change bundled into Task 5.
3. **`eakld_confidence_k` unused in offload dispatcher (non-blocking):** Still validated (`< 2` raises), but not read in the body since precomputed gamma already reflects it. Acknowledged in report Concern #3.
4. **No direct `eakld_kd` dense-vs-offload value+grad equality test (minor gap):** Only CE-once mixing is asserted for the offload `eakld_kd` path. Since `eakld_kd = ce*(1-α) + eakld_region*α` and `eakld_region` equality is already tested for `eakld`/`eakld_top_7`, the gap is low-risk.

## ⚠️ Warnings

None blocking. Implementation is correct, tested, and matches spec.

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-5-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Report: Compute Student Top-k KL by Direct K-Way Gather on Device

## Status

Complete. All Task 5 steps executed (Step 9 commit skipped per override).

## Commits

None. Workspace changes left uncommitted per `no-auto-git-commit` rule and task override.

## Files modified

- `mix_bit/kl_metric.py`
  - Added private helper `_gather_topk_student_logits(shifted_student_logits, valid_mask, teacher_topk_indices)` returning `[N_valid, K]` via on-device padded gather (pads indices to `[B, T, K]`, gathers along vocab axis, then boolean-selects valid rows). Never materializes `[N_valid, V]`.
  - Refactored `per_sample_teacher_topk_forward_kl` to use the helper instead of `shifted_student_logits.float()[mask]`. Validation logic preserved verbatim (operates only on small mask/offsets/indices/probs). KL computation now runs on the student device; `per_sample_exact_forward_kl` is unchanged.

- `mix_bit/cost_search.py`
  - `evaluate_student_per_sample_kl` teacher_topk branch now passes `shifted_student_logits=shifted_student` (on-device) instead of `shifted_student.detach().cpu()`.
  - exact_full_vocab branch now passes `shifted_teacher.detach()` / `shifted_student.detach()` (on-device) instead of `.detach().cpu()`; `per_sample_exact_forward_kl` is device-agnostic (handles mask device internally), so behavior is preserved while removing full-vocab CPU transfers.
  - Only the final `kl.detach().cpu()` remains (for collecting per-sample floats).

- `mix_bit/tests/test_kl_metric.py`
  - Added `_dense_teacher_topk_kl_reference` (test-only `[N_valid, V]` oracle) and parametrized equivalence tests covering batch 1/3, K=1/3/V, bf16 probs, mixed-sign logits, non-contiguous masks, varying per-sample token counts.
  - Added `test_gather_topk_student_logits_shape_is_n_valid_by_k` (V=1000, K=4 → shape `[N_valid, 4]`).
  - Added CUDA test asserting helper input/output stay on CUDA and match CPU dense reference.
  - Added `test_kl_source_does_not_gather_full_valid_student_rows` source guard.

- `mix_bit/tests/test_cost_table.py`
  - Added `test_evaluate_student_per_sample_kl_source_does_not_cpu_full_logits` source guard forbidding `shifted_student.detach().cpu()` / `shifted_student.cpu()` and requiring on-device logits handoff.

- `mix_bit/tests/test_tiny_integration.py`
  - No changes required; existing fixtures (b4d4s1/s2/s3) left untouched.

## Test summary

```
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_kl_metric.py mix_bit/tests/test_cost_table.py mix_bit/tests/test_tiny_integration.py -q
→ 66 passed in 6.92s
```

- Step 5 pre-implementation confirmation: 3 new kl_metric tests failed (helper missing, source guard tripped) and 1 new cost_table source guard failed — as required before implementing.
- Post-implementation: all 66 tests pass, including the CUDA test (CUDA available, not skipped).
- Existing equivalence tests (`test_teacher_topk_kl_matches_manual_renormalized_subset`, `test_teacher_topk_equals_exact_when_k_equals_vocab_size`, `test_bf16_cached_probs_renormalized_to_float32_before_kl`) continue to pass, confirming the refactor preserves numeric behavior.

## Concerns

- The exact_full_vocab path in `evaluate_student_per_sample_kl` now runs on the worker device instead of CPU. `per_sample_exact_forward_kl` is device-agnostic (mask is moved to `token_kl.device` internally), so behavior is preserved, but this is a behavior-adjacent change. It was required because the Step 4 source guard forbids `shifted_student.detach().cpu()` anywhere in the function. If CPU execution of the exact path is ever required, the guard would need to be relaxed or the exact path restructured.
- The per-sample mean loop in `per_sample_teacher_topk_forward_kl` still uses Python-level `.item()` calls over `offsets_dev` (small `[B+1]` tensor), which syncs per sample. This is acceptable for the small batch sizes used in calibration cost search; vectorizing it is out of Task 5 scope.

## Report path

`/home/shaoyuantian/program/VAELLM/.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-5-report.md`

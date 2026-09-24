> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-4-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Review: Split CPU-Offloaded EAKLD Teacher Scalars by Region

## Spec Compliance: ✅

| Requirement | Status |
|---|---|
| Add exactly 3 named prompt fields | ✅ `eakld_prompt_gamma_cpu`, `teacher_prompt_entropy_mean_cpu`, `teacher_prompt_valid_token_count_cpu` |
| `clear()` resets all 3 | ✅ |
| Replace `_build_distill_token_mask` → `_build_distill_token_regions` returning `DistillTokenRegions` | ✅ |
| Builder records no telemetry | ✅ only delegates to `build_distill_token_regions` |
| Response entropy/gamma always when logits required | ✅ on `logits_cpu` with `regions.response_mask` |
| Prompt path only when `prompt_kd_weight > 0.0` | ✅ guarded; zero weight leaves fields unset |
| Single CPU teacher logits copy, no duplicate | ✅ one `copy_detached_tensor_to_cpu`; both helper calls share `logits_cpu` |
| Zero-weight test: response populated, prompt absent, helper once | ✅ `test_cpu_teacher_targets_zero_prompt_weight_response_only` |
| Positive-weight test: both populated, helper twice, distinct masks, counts = mask sums | ✅ `test_cpu_teacher_targets_positive_prompt_weight_both_regions` |
| No auto-commit | ✅ working tree only |

Tests run: `tests/test_e2e_teacher_first.py` → 14 passed; both smoke files → 8 passed.

## Quality

Clean, minimal implementation. Behavior change (entropy/gamma now computed on the CPU copy instead of the original GPU `teacher_logits`) is intentional and aligns with the brief's "single CPU copy reused for both regions" directive; detached float32 scalar storage is preserved. Tests are well-designed: the counting helper captures `logits_id`, cloned `mask`, and `confidence_k` per call, enabling precise assertions on call count, mask distinctness, count-vs-mask-sum equality, and single-copy reuse via `id()` equality. Task 5 deferral (dispatcher not yet consuming prompt scalars) is correctly noted in report Concern 1.

## Findings

1. ⚠️ **`tests/smoke/test_loss_pipeline_smoke.py` modified but omitted from report.** It is not in the brief's file list and is NOT mentioned in the report's "Files Changed" section (only `test_one_step_train_smoke.py` is disclosed in Concern 3). The change is defensible — the old test called `build_distill_token_mask(..., prompt_kd_weight=0.1)`, which has been broken since Task 1 removed that parameter — but the report's silence is a transparency gap. The fix also goes beyond a minimal API repair: it switches to `build_distill_token_regions` and passes `prompt_mask`/`prompt_kd_weight=0.03` into `compute_dense_loss_from_logits`, exercising the dispatcher's prompt path (Task 5 territory) ahead of Task 4's scope.

2. ⚠️ **Report inaccuracy: "keyword-only".** Report states `_build_distill_token_regions` is "keyword-only", but the signature has no `*` — `inputs` and `reference_logits` are positional-or-keyword. Not a spec violation (brief does not require keyword-only), just a misdescription.

3. **Edge case (not required by brief):** if `prompt_kd_weight > 0` but `labels` contain no `-100` (no prompt tokens), `prompt_mask` is all zeros → `valid_count = 0`, `gamma` may be non-finite. Untested. Worth a guard or test in Task 5.

## Verdict

Spec ✅. Quality good. Two transparency/accuracy nits (findings 1–2) and one deferred edge case (finding 3). No blocking issues; Task 4 deliverables are present and correct.

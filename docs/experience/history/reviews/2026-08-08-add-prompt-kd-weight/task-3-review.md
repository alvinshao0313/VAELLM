> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-3-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Review: EAKLD Fractional-Mask Verification Tests

**Reviewer:** read-only code review  
**Date:** 2026-08-10  
**Scope:** `tests/test_distill_losses.py` — 5 new tests (Task 3 only; no production changes)

---

## Verdicts

| Dimension | Verdict |
|-----------|---------|
| **Spec** | ✅ |
| **Quality** | Approved |

---

## Spec Checklist (task-3-brief.md)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Teacher entropy/gamma fractional-mask test with hand-computed `weighted entropy / mask.sum()` | ✅ | `test_eakld_teacher_entropy_gamma_uses_fractional_mask` uses `_dense_teacher_entropy_stats_reference` + `gamma_from_entropy_sums`; mask contains 0.1/1.0/0; asserts `entropy_mean`, `gamma_reverse`, `valid_count` |
| Retain existing binary dense-vs-CPU tests | ✅ | `test_cpu_teacher_eakld_matches_dense_value_and_gradient` and `test_cpu_teacher_eakld_topk_matches_dense_value_and_gradient` unchanged (no deletions in diff) |
| Add fractional-mask versions for `compute_eakld` and `compute_eakld_topk` | ✅ | `test_cpu_teacher_eakld_fractional_mask_*` and `test_cpu_teacher_eakld_topk_fractional_mask_*` |
| Fractional: dense loss == CPU chunked loss, student gradients match | ✅ | Both CPU fractional tests compare loss (`rtol=5e-6`) and grad (`rtol=1e-5`) across multiple `sequence_chunk_size` values |
| `valid_tokens` = `mask.sum()` (non-integer allowed); no telemetry key/schema changes | ✅ | `test_eakld_telemetry_valid_tokens_is_fractional_mask_sum` asserts `4.3`; production diff is empty |
| Prefer tests-only; no EAKLD refactor if tests pass | ✅ | Report and diff confirm zero production changes; 51/51 tests pass |

**Additional coverage beyond minimum:** `test_eakld_topk_fractional_mask_matches_dense_output_and_gradient` compares `compute_eakld_topk` against `_dense_eakld_topk_reference` (loss + gradient). This mirrors the existing binary top-k dense-reference pattern and strengthens Task 3 without expanding scope.

---

## Findings

No findings.

No correctness, security, or maintainability defects were identified in the Task 3 additions. Tests follow existing file conventions (shared helpers, parametrized chunk sizes, explicit non-integer `mask.sum()` literals).

---

## Residual Risks (non-blocking)

1. **Telemetry path is partially covered for fractional masks.** `test_eakld_telemetry_valid_tokens_is_fractional_mask_sum` only asserts `valid_tokens`. Entropy/gamma fractional semantics are verified via `compute_teacher_entropy_mean_and_gamma` directly, not through `telemetry_out["teacher_entropy_mean"]` / `telemetry_out["gamma_reverse"]`. Risk is low because `_write_eakld_telemetry` is a thin passthrough and the binary test `test_eakld_telemetry_reuses_existing_entropy_and_kl_computation` already guards telemetry schema parity.

2. **Full-vocab `compute_eakld` has no standalone dense reference test for fractional masks** (only dense-vs-CPU). This matches the existing binary test layout and satisfies the brief, which requires CPU chunked parity rather than a hand-built full EAKLD reference.

---

## Test Verification (independent re-run)

```bash
conda activate bitvae
cd /home/shaoyuantian/program/VAELLM
PYTHONPATH=. pytest tests/test_distill_losses.py -q
# 51 passed in ~5.6s

PYTHONPATH=. pytest tests/test_distill_losses.py -q -k "fractional_mask or teacher_entropy_gamma_uses_fractional or telemetry_valid_tokens_is_fractional"
# 9 passed (5 test functions, parametrized chunk sizes)
```

---

## Summary

Task 3 is complete and spec-compliant. Five focused tests lock in fractional-mask semantics for teacher entropy/gamma, telemetry `valid_tokens`, top-k dense reference parity, and CPU-offload dense parity for both `compute_eakld` and `compute_eakld_topk`. No production refactor was introduced. **Spec ✅ · Quality Approved.**

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-3-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Report: Verify EAKLD Uses the Same Fractional Mask Everywhere

## Status

**Complete** — all Task 3 tests added; no production code changes required.

## Production Changes

None. Existing EAKLD implementation already applies the same fractional mask to teacher entropy/gamma, KL terms, and telemetry `valid_tokens` (= `mask.sum()`).

## Tests Added (`tests/test_distill_losses.py`)

| Test | Purpose |
|------|---------|
| `test_eakld_teacher_entropy_gamma_uses_fractional_mask` | Hand-computed weighted entropy / `mask.sum()` vs `compute_teacher_entropy_mean_and_gamma` for mask with 0.1/1.0/0 |
| `test_eakld_telemetry_valid_tokens_is_fractional_mask_sum` | `telemetry_out["valid_tokens"]` equals non-integer `mask.sum()` |
| `test_eakld_topk_fractional_mask_matches_dense_output_and_gradient` | Dense reference vs `compute_eakld_topk` with fractional mask |
| `test_cpu_teacher_eakld_fractional_mask_matches_dense_value_and_gradient` | Dense vs CPU-chunked `compute_eakld` (chunk sizes 1, 2, 5) |
| `test_cpu_teacher_eakld_topk_fractional_mask_matches_dense_value_and_gradient` | Dense vs CPU-chunked `compute_eakld_topk` (chunk sizes 1, 3, 8) |

Existing binary dense-vs-CPU tests retained unchanged.

## Test Run

```bash
conda activate bitvae
cd /home/shaoyuantian/program/VAELLM
PYTHONPATH=. pytest tests/test_distill_losses.py -q
```

**Result:** 51 passed in ~5s

## Concerns

None. Fractional mask semantics are consistent across entropy, gamma, KL, telemetry, and CPU-offload paths without refactoring.

## Commits

None (per task instructions).

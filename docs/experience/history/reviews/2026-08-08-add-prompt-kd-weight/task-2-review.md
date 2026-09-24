> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-2-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Review: Shared Weighted Causal Mask

**Reviewer:** code review subagent  
**Date:** 2026-08-10  
**Scope:** `train_utils/distill_losses.py` + minimal Task 1 test fixes  
**Artifacts:** `task-2-brief.md`, `task-2-report.md`, `task-2-review-package.diff`

---

## Verdict

| Dimension | Result |
|-----------|--------|
| **Spec** | ✅ |
| **Quality** | **Approved** |

---

## Spec Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Add `prompt_kd_weight: float = 0.0` kwarg | ✅ | Signature extended; default preserves call sites |
| Keep `reference_logits.ndim` validation | ✅ | Unchanged guard at entry |
| Reject `< 0`; no upper bound | ✅ | `float()` cast + `ValueError`; `p=2.0` test passes |
| `p=0` + labels: legacy `labels != -100`, no attention | ✅ | Dedicated branch; ignores `attention_mask` |
| `p>0` + labels: response/prompt validity + optional attention AND | ✅ | Padding wins when attention present |
| Response weight 1.0; prompt weight = config value | ✅ | `response_validity.float() + prompt_validity.float() * p` |
| `labels=None`: unchanged fallback; ignore prompt weight | ✅ | Same attention / all-ones paths as before |
| Causal shift; `[B,L]` float32; logits device; last pos 0 | ✅ | `causal_mask[:, :-1] = source_weights[:, 1:]` |
| No reducer changes | ✅ | Diff touches only `build_distill_token_mask` in production file |
| Task 1 tests GREEN | ✅ | Report: 42 passed (not re-run; evidence accepted) |

---

## Implementation Analysis

### Core logic (`build_distill_token_mask`)

The implementation cleanly splits three paths:

1. **Labels present, `p == 0`:** `source_weights = labels.ne(-100).float()` — binary, labels-only precedence, exactly matching pre-Task-2 behavior (attention ignored even if passed).
2. **Labels present, `p > 0`:** builds mutually exclusive `response_validity` / `prompt_validity`, optionally ANDs both with `attention_validity`, combines as `1.0 + p * prompt`. Matches plan semantics in `docs/superpowers/plans/2026-08-08-add-prompt-kd-weight.md` §Required Mathematical Semantics.
3. **Labels absent:** unchanged attention-mask or all-ones fallback; `prompt_kd_weight` never applied.

Causal shift and output contract are unchanged from legacy: float32 on `reference_logits.device`, final column hard-zero.

### Plan examples verified by tests

- `[-100,-100,-100,A,B,EOS]` with `p=0.1` → `[0.1,0.1,1,1,1,0]` — `test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift`
- Padding case → `[0.1,1,1,0,0,0]` — `test_distill_mask_padding_excludes_prompt_weight`
- Interleaved `-100` mid-sequence → per-token prompt weight — `test_distill_mask_interleaved_prompt_tokens_use_prompt_weight`

### Minor improvement (non-blocking)

Moving `labels.to(device=device)` before weight construction avoids a potential device mismatch when labels and logits live on different devices. This is a safe hardening, not a behavior change for typical same-device usage.

---

## Task 1 Test Fixes — Legitimacy Assessment

Both fixes are **legitimate expectation corrections**, not test weakening.

### 1. `test_distill_mask_accepts_prompt_kd_weight_above_one`

| | Original (Task 1) | Fixed |
|---|-------------------|-------|
| Labels | `[-100, -100, 10, 2]` | `[-100, -100, -100, 2]` |
| Expected | `[[2, 2, 1, 0]]` | `[[2, 2, 1, 0]]` (unchanged) |

**Why original was wrong:** With causal shift, logit weight at index `t` comes from target weight at `t+1`. Expected `[2, 2, 1, 0]` requires target positions 1 and 2 both be prompt (`weight=2`). Original labels have index 2 = `10` (response, weight 1), yielding `[2, 1, 1, 0]`. The fix aligns labels with the intended assertion — still tests that `p>2` is accepted with no upper bound.

**Verdict:** ✅ Correct fix.

### 2. `test_forward_kl_gradient_respects_fractional_prompt_weights` — `grad_zero[1]`

| | Original (Task 1) | Fixed |
|---|-------------------|-------|
| Assertion at index 1, `p=0` | `== 0` | `> 0` |

**Why original was wrong:** Labels `[-100, -100, 10, 11, 2]` with `p=0` produce mask `[0, 1, 1, 1, 0]` after causal shift. Index 0 predicts prompt token (mask 0 → grad 0). Index 1 predicts response token `10` (mask 1 → grad non-zero). Task 1 brief intent — *"prompt-only logits 梯度严格为 0"* — applies to index 0, not index 1.

The fix preserves the strict zero check at index 0 and correctly requires non-zero gradient at response-predicting positions. No assertions were removed or relaxed elsewhere in the gradient test.

**Verdict:** ✅ Correct fix.

---

## Quality Assessment

**Approved.** No P0–P3 findings.

- Logic matches brief and plan semantics end-to-end.
- Backward compatibility at `p=0` is structurally guaranteed by a separate branch that does not touch attention.
- Shape validation preserved and correctly scoped (labels shape when labels present; attention shape when attention used in `p>0` or labels-absent path).
- Reducers untouched; existing float-mask / `mask.sum()` normalization path reused.
- Test fixes correct Task 1 authoring errors without diluting coverage.

### Residual risks (informational, out of scope)

- `p>0` without `attention_mask`: padded positions identifiable only via labels cannot be zeroed — pre-existing limitation, explicitly allowed by brief ("if attention mask exists").
- Trainer wiring (Tasks 3+) not reviewed here.

---

## Findings

No findings.

---

## Recommendation

Proceed to Task 3. No changes required for Task 2.

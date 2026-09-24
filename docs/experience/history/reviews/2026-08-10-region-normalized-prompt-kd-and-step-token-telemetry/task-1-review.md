> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-1-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Review: Replace Fractional Prompt Mask with Two Binary Region Masks

**Reviewer:** task-scoped gate review  
**Base:** `d393552508cbfbf44a5f0bcf7a26b5bdbe422bf3`  
**Head:** working tree (uncommitted)  
**Sources:** `task-1-brief.md`, `task-1-report.md`, `task-1-review-package.diff`

---

## Verdict

| Dimension | Result |
|---|---|
| **Spec compliance** | ✅ Pass |
| **Task quality** | **Approved** |

Task 1 meets its brief and acceptance criterion. Implementation is focused, test coverage matches the specified mask contracts, and fractional prompt-weight production logic is fully removed from `distill_losses.py`.

---

## Spec Compliance

### Required API changes

| Requirement | Verified | Evidence |
|---|---|---|
| `build_distill_token_mask()` restored to 3 keyword args only | ✅ | Diff removes `prompt_kd_weight`; signature is `(labels, attention_mask, reference_logits)` |
| No fractional values from mask builders | ✅ | `build_distill_token_mask` uses binary `labels.ne(-100)` / attention validity / ones; `build_distill_token_regions` prompt path uses binary `labels.eq(-100)` |
| Frozen `DistillTokenRegions` with `response_mask`, `prompt_mask` | ✅ | `@dataclass(frozen=True)` added |
| `build_distill_token_regions()` keyword-only, returns dataclass | ✅ | Function signature and return type match brief |
| `response_mask` reuses `build_distill_token_mask()` | ✅ | First line of `build_distill_token_regions` delegates to helper |
| Prompt: `labels == -100`, AND attention when present, causal shift, final zero | ✅ | Lines 135–148 in diff |
| Labels absent → all-zero prompt | ✅ | `else` branch zeros prompt mask |
| Labels present → `labels != -100` for response (weight-zero path) | ✅ | Restored single branch without weight branching |
| Shape validation preserved | ✅ | `_validate_distill_mask_shape` called for labels/attention |
| Only `train_utils/distill_losses.py` and `tests/test_distill_losses.py` touched | ✅ | Diff stat shows exactly 2 files |

### Acceptance criterion

> No production prompt-weight path creates a fractional causal mask.

✅ **Met.** Grep of current `distill_losses.py` shows no `prompt_kd_weight` references. Fractional mask logic is removed from `build_distill_token_mask`. New region builder emits only `{0.0, 1.0}` masks. Downstream reducers still accept arbitrary float masks by design (brief allows this).

### Global constraints (binding on this task)

| Constraint | Status |
|---|---|
| Keep existing CLI names / experiment weight unchanged | ✅ N/A at Task 1 scope — no CLI or caller wiring changed |
| Weight zero preserves response-only behavior | ✅ Restored mask equals former `prompt_kd_weight=0.0` path; existing `test_distill_mask_exactly_matches_next_label_validity` retained |
| Prompt coefficient no longer creates fractional causal mask | ✅ |
| Response and prompt use disjoint binary causal masks | ✅ Implemented and tested |
| Callers may break until later tasks | ✅ Expected; not a Task 1 defect |

### Required tests and expected values

Manual trace of causal left-shift (`mask[:, :-1] = source[:, 1:]`, final position zero) confirms:

| Test | Expected (brief) | Diff expected tensors | Match |
|---|---|---|---|
| Single-turn `[-100,-100,-100,A,B,EOS]` | response `[0,0,1,1,1,0]`, prompt `[1,1,0,0,0,0]` | Same | ✅ |
| Padding `[-100,-100,A,EOS,-100,-100]` + attn `[1,1,1,1,0,0]` | response `[0,1,1,0,0,0]`, prompt `[1,0,0,0,0,0]` | Same | ✅ |
| Interleaved `[-100,A,-100,B,EOS]` | response `[1,0,1,1,0]`, prompt `[0,1,0,0,0]` | Same | ✅ |
| Labels-none fallback | response = mask helper; prompt all zero | Asserts equality to `build_distill_token_mask` + zero prompt (with and without attention) | ✅ |
| Invariants: float32, binary, shape/device, disjoint, final zero | Required | `test_distill_regions_masks_are_binary_disjoint_with_zero_final` | ✅ |

### Fractional test removal

✅ Nine prompt-weight / fractional-KL tests tied to `build_distill_token_mask(..., prompt_kd_weight=...)` removed as required.

✅ Independent fractional-mask reducer tests retained (e.g. `test_cpu_teacher_eakld_fractional_mask_matches_dense_value_and_gradient` still present in diff tail).

---

## Task Quality

### Strengths

1. **Clean decomposition.** Extracting `_apply_causal_shift` removes duplication between response and prompt paths without changing semantics.
2. **Correct delegation.** `response_mask` is not reimplemented; brief’s “directly reuses” requirement is satisfied literally.
3. **Tests mirror brief.** All five new region tests map 1:1 to checklist items with exact expected tensors.
4. **Scope discipline.** No drive-by changes outside the two allowed files.
5. **Backward-compatible response path.** Restored `build_distill_token_mask` logic matches pre-feature weight-zero behavior (binary label validity → causal shift).

### Findings

#### Minor

1. **`tensor_name` parameter unused in `_validate_distill_mask_shape`.** Passed at call sites but not included in the `ValueError` message. Old inline validation identified the failing tensor (`labels` vs `attention_mask`); new helper loses that detail. No functional impact; slightly worse DX on shape mismatch.

2. **Explicit weight-zero equivalence test removed.** `test_distill_mask_prompt_weight_zero_is_exact_current_behavior` was deleted with the fractional suite. Behavior is now the only code path and is still covered by `test_distill_mask_exactly_matches_next_label_validity` plus region tests. Acceptable for Task 1; optional follow-up if later tasks want an explicit regression anchor.

#### Important / Critical

None.

---

## Report Claims vs Diff

| Claim | Diff verification |
|---|---|
| `prompt_kd_weight` removed from `build_distill_token_mask` | ✅ Confirmed |
| `DistillTokenRegions` frozen dataclass added | ✅ Confirmed |
| Five new region tests added | ✅ Confirmed |
| Nine fractional prompt-weight tests removed | ✅ Confirmed |
| Independent fractional reducer tests kept | ✅ Confirmed (diff shows retained tests) |
| Only two files modified | ✅ Confirmed |
| No git commit | ✅ Out of review scope; diff is uncommitted working tree |
| 46 tests passed | ⚠️ Cannot verify from diff alone (see below) |
| TDD RED → GREEN evidence | ⚠️ Cannot verify from diff alone |

---

## Cannot Verify from Diff

- **`pytest tests/test_distill_losses.py -q` → 46 passed.** Report claim only; not re-run per review instructions.
- **TDD RED phase** (`AttributeError` before implementation). Plausible sequencing; not independently evidenced in diff.

---

## Out of Scope (Not Defects)

Downstream callers still passing `prompt_kd_weight` to `build_distill_token_mask()` will `TypeError` until later tasks wire `build_distill_token_regions()`. Brief and report both acknowledge this.

---

## Recommendation

**Approve Task 1.** Proceed to downstream wiring tasks. Optional polish (non-blocking): include `tensor_name` in `_validate_distill_mask_shape` error text.

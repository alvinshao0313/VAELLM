> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/final-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Final Whole-Branch Review — Region-Normalized Prompt KD + Step-Token Telemetry

**Plan:** `docs/superpowers/plans/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry.md`
**Base:** `d393552`  **Head:** working tree (no commit, per project policy)
**Reviewer:** Senior Code Reviewer (final whole-branch)
**Diff package:** `final-review-package.diff` (9 production `diff --git` hunks + 4 new files appended under `==== NEW FILES ====`)

## Verdict: MERGE-READY (no critical / no important blockers)

All locked semantics from the plan are correctly implemented and verified by the
focused suites (184 passed in combined run; only 9 pre-existing failures, see §4).

### 1. Critical findings
None.

### 2. Important findings
None.

### 3. Deferred-minors triage (from progress ledger)

| Source | Deferred item | Must-fix before merge? | Reason |
|---|---|---|---|
| Task 1 | `tensor_name` field in mask-shape error message | No | Cosmetic; error text already names the tensor via the dedicated validator. |
| Task 2 → 5 | Offload prompt gamma wiring (F2/F3/F4) | No | Closed in Task 5; `TeacherTargetBatch` carries `eakld_prompt_gamma_cpu` / `teacher_prompt_entropy_mean_cpu` / `teacher_prompt_valid_token_count_cpu`, dispatcher requires all three when `weight > 0`, and `_prompt_eakld` reuses the single CPU logits copy. |
| Task 4 | "empty prompt_mask gamma finiteness" parked note | No | Covered by `test_empty_prompt_mask_eakld_positive_weight_remains_finite` and `test_offloaded_empty_prompt_mask_positive_weight_remains_finite`. |
| Task 5 | "naming" | No | Cosmetic. |
| Task 5 | "eakld_kd equality test gap" | No | Gap closed: `test_offloaded_eakld_kd_positive_prompt_weight_mixes_ce_once` (tests/test_distill_losses.py:1964) asserts CE is mixed exactly once after regional EAKLD combination on the offload path. |
| Task 6 / 7 | Unspecified minors | No | Token-stats accumulator + callback tests (10 + 7 + 11) green; contract matches plan. |
| Task 9 | `docs/cat_train_args.md` §2.2 override-syntax example still uses `after:q_proj=0.05` (line 94) while §6.11.1 uses `0.03` | No (optional polish) | Line 94 is a generic demonstration of the `--key default=...,after:<cat>=<v>` override *syntax*, not a value recommendation. §6.11.1 and the README both consistently document `0.03` as the experiment example and explicitly disclaim optimality. Safe to merge; optionally align to `0.03` later. |

No deferred minor rises to must-fix.

### 4. Note on 9 pre-existing `tests/test_e2e_dataset_mix.py` failures

Reproduced on the **base commit `d393552`** with the `bitvae` env (working tree
stashed): `9 failed, 39 passed` — identical failure set and classes as the head
run:

1. `DatasetMixArgsTest::test_parse_args_eval_before_save_requires_tasks_and_save_steps` — HfArgumentParser unused-arg on `--eval_before_save`.
2. Seven `DatasetMixBuilderTest::*` cases — `Weighted lazy mix with multiple text_format values is not supported` raised by `e2e_common/lazy_datasets.py` (plan forbids modifying this file).
3. `DatasetMixBuilderTest::*` missing `dummy.txt` fixture (1).

These are unrelated to region-normalized loss or token telemetry, pre-date this
branch, and are explicitly out of plan scope. **Not blocking.**

### 5. Spot-checked correctness against locked semantics

- `build_distill_token_mask` no longer accepts `prompt_kd_weight`; `build_distill_token_regions` returns disjoint binary float32 masks with final-position zero (verified in `train_utils/distill_losses.py` hunk + `test_distill_regions_*`).
- `L_logit = L_response + w * L_prompt` via `_combine_region_loss` / `combine_region_loss`; weight applied only after independent region means; zero-weight short-circuits prompt criterion (category `lora_training.py`, dense `e2e_common/dense_loss.py`).
- CE+KD families (`kd`, `kd_top*`, `dual_kd`, `dual_kd_top*`, `eakld_kd`) mix CE exactly once after regional KD combination (`test_kd_ce_not_double_counted_across_regions`, `test_offloaded_eakld_kd_positive_prompt_weight_mixes_ce_once`).
- EAKLD: response call writes `telemetry_out`; prompt call passes `telemetry_out=None` and (offload) reuses precomputed `teacher_prompt_gamma_cpu` with `teacher_entropy_mean=None`/`teacher_valid_token_count=None` (guard at `train_utils/distill_losses.py:881` only raises when `telemetry_out is not None`). Existing `eakld/*` telemetry remains response-region only.
- Dense/offload parity at positive weight: `test_offloaded_eakld_positive_prompt_weight_matches_dense_value_and_gradient` and the topk variant compare scalar + student gradient.
- CPU teacher targets: response scalars always populated; prompt scalars only when `prompt_kd_weight > 0`; single CPU logits copy reused for both regions (`test_cpu_teacher_targets_zero_prompt_weight_response_only`, `test_cpu_teacher_targets_positive_prompt_weight_both_regions`).
- Token telemetry: one `update(labels, attention_mask)` per student micro-batch at the top of `compute_loss` only when `model.training`; MCQA/choice paths skipped (no rank-2 labels); no update from teacher/hidden/CPU-staging paths. `consume_global` runs on every rank before rank-0 filtering (DDP-safe); empty local state still participates in the collective via a zero tensor.
- Callback boundary: uses resolved `state.logging_steps` (positive int required), `window_start_step` initialized on first `on_step_end` (resume-safe), `global_step % logging_steps != 0` returns before collective; special step-1 log therefore does not consume/reset the window. Verified by `test_window_one_to_ten_no_consume_until_step_ten`, `test_cadence_resolution_uses_state_logging_steps_not_raw_args`, `test_resume_from_non_boundary_reports_partial_first_window`, `test_nonzero_rank_invokes_consume_but_does_not_write`.
- Sample log line matches plan: `LoRA token stats: step=10 window_optimizer_steps=10 avg_prompt_tokens=3.0000 avg_response_tokens=3.0000 global_samples=10`.
- Compatibility: `scripts/catlora_distill_4gpu_res0.sh` retains `--distill_prompt_kd_weight "default=0.03"`; `e2e_decoder.sh`, `lazy_datasets.py`, checkpoint/eval/VAE code untouched (confirmed via `git diff --stat HEAD` on forbidden paths = empty).
- No auto-commit; changes remain in working tree.

### 6. Caveats / follow-ups (non-blocking)

- Diff package did not embed hunks for the *modified* test files (`tests/test_distill_losses.py`, `tests/test_e2e_dataset_mix.py`, `tests/test_e2e_teacher_first.py`, `tests/test_cat_eval_adapter_match.py`, smoke tests); only the 4 new files were appended. Review of those modified tests was done by reading the working tree directly. Recommend future review packages include all touched files' hunks.
- Optional: align the `0.05` override-syntax example at `docs/cat_train_args.md:94` to `0.03` for consistency with §6.11.1.
- No downstream-accuracy claim is made or implied by this branch (plan explicitly forbids it); a controlled run is still required before any such conclusion.

## Summary

The branch faithfully implements region-normalized prompt KD (`L_response + w * L_prompt`
with independent binary region masks and per-region EAKLD gamma) and logging-window
token telemetry (true non-padding `labels`+`attention_mask` counts, not KD masks)
for both category and E2E paths, with dense/offload parity and unchanged loss
logging cadence. All focused tests pass; the only failures are 9 pre-existing
`test_e2e_dataset_mix` cases reproduced on the base commit and explicitly out
of scope. **Approve for merge.**

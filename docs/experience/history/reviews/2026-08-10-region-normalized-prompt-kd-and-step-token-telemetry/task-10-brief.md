> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-10-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 10: Closed-Loop Verification

Environment:
- [ ] Activate conda environment `bitvae`.
- [ ] Verify the selected Python interpreter can import PyTorch before pytest.

Run focused tests individually:

```text
pytest tests/test_distill_losses.py -q
pytest tests/test_distill_token_stats.py -q
pytest tests/test_cat_eval_adapter_match.py -q
pytest tests/test_e2e_dataset_mix.py -q
pytest tests/test_e2e_teacher_first.py -q
pytest tests/smoke/test_loss_pipeline_smoke.py -q
pytest tests/smoke/test_one_step_train_smoke.py -q
```

- [ ] Every focused command exits successfully.
- [ ] Then run those same files in one combined pytest invocation to detect shared-state or test-order problems.

Static audit A:

```text
rg -n "build_distill_token_mask|build_distill_token_regions|prompt_kd_weight" train_utils e2e_common compressed_e2e_fintuning
```

Manual acceptance:
- no production path passes prompt weight into mask construction;
- prompt/response region masks are binary;
- prompt coefficient is applied only after independent criterion means;
- zero-weight control flow skips prompt criterion.

Static audit B:

```text
rg -n "compute_teacher_entropy_mean_and_gamma" compressed_e2e_fintuning train_utils e2e_common
```

Confirm CPU teacher-target code contains the normal response entropy/gamma path and a conditional prompt entropy/gamma path.

Static audit C:

```text
rg -n "DistillTokenStatsAccumulator|consume_global|token stats" train_utils compressed_e2e_fintuning
```

Confirm:
- one accumulator update per student micro-batch from original `labels` plus `attention_mask`;
- no consume/reset on ordinary optimizer steps or the special step-1 log;
- one consume only at each regular `logging_steps` boundary;
- distributed reduction occurs before rank-zero output filtering.

Compatibility checks:
- [ ] category script retains the user's current 0.03 setting;
- [ ] no shell hyperparameter changes;
- [ ] hidden loss, data truncation, dataset mix, checkpoint and eval logic are untouched.

Minimal real-run smoke, if repository conventions permit:
- [ ] startup resolves prompt coefficient 0.03;
- [ ] with `distill_log_every=10`, the special step-1 loss log contains no token-window line, and step 10 emits exactly one token-window line covering steps 1-10;
- [ ] the token-window line reports true average prompt/response token counts per sample, plus `window_optimizer_steps=10` and global sample count;
- [ ] regular loss lines remain on the existing cadence; no new per-step logging is introduced;
- [ ] loss remains finite;
- [ ] no downstream-accuracy conclusion is drawn from smoke behavior.

---

## Acceptance Criteria

- [ ] Prompt coefficient no longer creates a fractional causal mask.
- [ ] response and prompt use disjoint binary causal masks.
- [ ] Exact logit loss is `L_response + w * L_prompt`.
- [ ] Each region normalizes by its own valid-token count.
- [ ] Prompt length cannot change coefficient `w`.
- [ ] Zero weight skips prompt computation and preserves response-only behavior.
- [ ] All KL/RKL/Top-K/dual/MSE branches use region-level combination.
- [ ] CE+KD families include CE exactly once.
- [ ] response and prompt EAKLD have independent entropy/gamma.
- [ ] dense and CPU-offload EAKLD agree at positive weight in scalar value and student gradient.
- [ ] CPU prompt support adds only scalar gamma/entropy/count state and does not duplicate full teacher logits.
- [ ] Existing EAKLD telemetry remains response-region telemetry.
- [ ] Category and E2E use identical prompt-loss semantics.
- [ ] Token telemetry is emitted only at regular logging boundaries, never every optimizer step.
- [ ] With cadence 10, the normal first telemetry window covers steps 1-10; the special step-1 loss log does not consume/reset the window.
- [ ] Counts include all gradient-accumulation micro-batches, all optimizer steps in the window, and all DDP ranks.
- [ ] Prompt/response counts are true non-padding token counts from `labels` plus `attention_mask`, not causal KD-mask counts and not weighted mass.
- [ ] Existing loss logging cadence and `logging_first_step` behavior are unchanged.
- [ ] Current category script value 0.03 is unchanged.
- [ ] hidden loss, data truncation, dataset mix, checkpoint and eval behavior are unchanged.
- [ ] All focused and combined tests pass in `bitvae`.
- [ ] Implementation does not automatically stage or commit changes.

## Completion Report Requirements

The implementation agent must report concrete evidence for:
1. final loss formula and zero-weight compatibility test result;
2. separate response/prompt EAKLD gamma behavior for dense and CPU-offload paths;
3. dense/offload value-and-gradient comparison result;
4. whether token telemetry is aggregated over the configured regular logging window, includes all gradient-accumulation micro-batches plus DDP ranks, and leaves the special step-1 log outside the window-consume path;
5. one real token-statistics log line from a smoke run showing the window length and average true prompt/response tokens per sample;
6. every pytest command actually run and its result;
7. confirmation that the current 0.03 setting and hidden/data/checkpoint/eval behavior were not changed.

Do not report downstream-accuracy improvement without a new controlled evaluation.

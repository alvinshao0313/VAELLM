> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 2: Implement Region-Level Combination in Dense Loss Dispatch

**Files:** `e2e_common/dense_loss.py`, `tests/test_distill_losses.py`, `tests/smoke/test_loss_pipeline_smoke.py`

Keep existing `mask` parameter as the response mask for backward compatibility. Add `prompt_mask` and `prompt_kd_weight` parameters to both `compute_dense_loss_from_logits()` and `compute_dense_loss_from_offloaded_teacher()`.

- [ ] Add a manual forward-KL test. Compute response KL mean and prompt KL mean separately using their binary masks; expected total is `response_mean + 0.03 * prompt_mean`.
- [ ] Add a critical anti-regression test where the prompt region is repeated many more times while every prompt token has the same per-token KL. The prompt-region mean and final coefficient must remain unchanged. This test must distinguish the new formula from the old shared-denominator formula.
- [ ] Add p=0 compatibility test comparing the current response-only call against the new call with prompt mask plus zero weight. Compare scalar loss and student gradients.
- [ ] Add empty-region tests: an all-zero prompt mask with positive prompt weight must contribute exactly zero and remain finite; an all-zero response mask must contribute zero response loss and leave only the weighted prompt-region loss. Cover at least forward KL and EAKLD so zero valid-token handling cannot create NaN.
- [ ] Validate negative prompt weight as an error. If weight is positive but prompt mask is absent, raise instead of silently using response-only.
- [ ] For each pure logit criterion, compute response criterion first. If weight is zero, return immediately without computing prompt. Otherwise compute prompt criterion separately and add the weighted prompt scalar.
- [ ] Apply this to `rkl`, `dual_rkl`, `kl`, `dual_kl`, `r_kl_top*`, `dual_r_kl_top*`, `kl_top*`, `dual_kl_top*`, `mse`, `eakld`, and `eakld_top*`.
- [ ] For `kd`, `kd_top*`, `dual_kd`, `dual_kd_top*`, and `eakld_kd`, first form regional logit KD, then mix CE once using the existing alpha formula.
- [ ] Add a CE anti-double-count test with a deliberately large known CE scalar; verify CE is not evaluated once per region.
- [ ] EAKLD response call receives existing telemetry output. Prompt EAKLD call does not write existing telemetry. Thus current `eakld/*` fields continue to mean response-region EAKLD.
- [ ] Run `pytest tests/test_distill_losses.py tests/smoke/test_loss_pipeline_smoke.py -q`.

---


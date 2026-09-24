> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 3: Convert Category Distillation to Region-Normalized Loss

**Files:** `train_utils/lora_training.py`, `tests/test_cat_eval_adapter_match.py`, `tests/smoke/test_one_step_train_smoke.py`

Inside `CustomSFTTrainer.compute_loss()`, replace the current local token-mask builder with one shared `build_token_regions(reference_logits)` calling `build_distill_token_regions()` on the original labels and attention mask.

Add one common local region combiner with exactly this control flow:

```text
response_loss = loss_for_mask(response_mask)
if prompt weight is zero: return response_loss
prompt_loss = loss_for_mask(prompt_mask)
return response_loss + prompt weight * prompt_loss
```

- [ ] All pure tokenwise branches use the common combiner: `rkl`, `dual_rkl`, `kl`, `r_kl_top*`, `dual_r_kl_top*`, `kl_top*`, `mse`, `dual_kl`, `dual_kl_top*`, `eakld_top*`, `eakld`.
- [ ] All CE+KD branches obtain regional KD first, then apply original CE/alpha formula once: `kd_top*`, `kd`, `dual_kd_top*`, `dual_kd`, `eakld_kd`.
- [ ] For EAKLD with positive prompt weight, add a focused mock/test proving the EAKLD criterion is called twice with different masks. For zero weight it is called once on response only.
- [ ] Do not change SFT/origin, hidden loss, pre-MLP hidden loss, teacher-logit staging, LoRA merge or restore logic.
- [ ] Static audit `train_utils/lora_training.py`: there must be one shared region-builder path and no call that passes prompt weight into `build_distill_token_mask()`.

---


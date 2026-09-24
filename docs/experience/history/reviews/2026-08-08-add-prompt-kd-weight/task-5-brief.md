> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-5-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 5: Integrate Prompt Weight into Every Category KD Branch

**File:** `train_utils/lora_training.py`

- [ ] `CustomSFTTrainer.__init__` 增加 `prompt_kd_weight: float = 0.0`；保存并拒绝负值。trainer 层必须自行验证，不能只依赖 CLI。
- [ ] 在 `compute_loss()` 中定义一个局部 `build_token_mask(reference_logits)`，内部统一调用共享 `build_distill_token_mask()`，传 full inputs 的 labels、attention 和 `self.prompt_kd_weight`。
- [ ] 当前所有 tokenwise distill branch 都改成调用该局部 helper，不允许每个分支自己拼参数。
- [ ] 必须覆盖：`rkl`、`dual_rkl`、`kl`、`r_kl_top*`、`dual_r_kl_top*`、`kl_top*`、`kd_top*`、`mse`、`kd`、`dual_kl`、`dual_kl_top*`、`dual_kd_top*`、`dual_kd`、`eakld_top*`、`eakld`、`eakld_kd`。
- [ ] SFT/origin 分支不改变。
- [ ] `kd` / `kd_top` / `dual_kd*` / `eakld_kd` 的 CE 项保持原 response-only labels 与原 `alpha` 混合公式；prompt weight 只作用于其 KD 项。
- [ ] hidden/pre-MLP hidden loss 继续使用 attention mask，禁止传 weighted KD mask。
- [ ] 实现后运行 `rg -n "build_distill_token_mask" train_utils/lora_training.py`。生产代码应收敛到 1 个 import + 1 个局部 helper 内真实调用；不能遗留某些 branch 的独立调用。
- [ ] 运行 `pytest tests/test_distill_losses.py tests/test_cat_eval_adapter_match.py -q`。

---


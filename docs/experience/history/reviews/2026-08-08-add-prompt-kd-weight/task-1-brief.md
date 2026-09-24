> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-1-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 1: Add Weighted-Mask Regression Tests First

**File:** `tests/test_distill_losses.py`

- [ ] 新增 `test_distill_mask_prompt_weight_zero_is_exact_current_behavior`：同一 labels/attention 下，省略参数和显式传 0.0 的结果必须 `torch.equal`。
- [ ] 新增 `test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift`：使用 `[-100,-100,-100,A,B,EOS]`，权重 0.1，精确得到 `[0.1,0.1,1,1,1,0]`。
- [ ] 新增 padding 测试：`labels==-100` 但 `attention_mask==0` 的位置不能获得 prompt 权重。
- [ ] 新增 `prompt_kd_weight=1.0` 测试：正常 SFT labels/attention 下结果等于 shifted attention validity，最后一个仍为 0。
- [ ] 新增多轮/interleaved 测试：例如 labels 中间再次出现 `-100`，这些 context token 应按 prompt weight 处理。
- [ ] 新增 `labels=None` fallback 测试：即使传 `prompt_kd_weight=0.1`，attention-only 模式仍必须返回当前 shifted binary attention mask；labels/attention 都不存在时仍只保留前 `L-1` 个位置。prompt 权重不得作用于无法识别 prompt/response 的输入。
- [ ] 新增负值拒绝测试：小于 0 必须 `ValueError`。
- [ ] 新增大于 1 测试：例如 2.0 必须被接受并真实出现在 prompt mask 中，防止实现者自行加上界。
- [ ] 新增 gradient 测试：p=0 时 prompt-only logits 梯度严格为 0；p>0 时变为非零；response 非零；padding 和最后 logits 始终为 0。
- [ ] 新增 weighted-mean 数值测试：手动计算 per-token forward KL，再按 fractional mask 做加权平均，与 `compute_forward_kl_loss` 完全一致。
- [ ] 运行 `pytest tests/test_distill_losses.py -q`，确认新增测试在生产代码修改前失败。

---


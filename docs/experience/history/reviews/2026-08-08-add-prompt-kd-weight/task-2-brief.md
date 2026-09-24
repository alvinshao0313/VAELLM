> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 2: Implement the Shared Weighted Causal Mask

**File:** `train_utils/distill_losses.py`

- [ ] 给 `build_distill_token_mask()` 增加 keyword 参数 `prompt_kd_weight: float = 0.0`。
- [ ] 保留当前 `reference_logits.ndim` 与 shape validation。
- [ ] 转成 float 后拒绝 `<0`；不加上界。
- [ ] labels 存在时先构造 `response_validity = labels != -100`。当 `prompt_kd_weight=0` 时，直接以它生成当前 legacy response 权重，不引入 attention mask，保证 exact backward compatibility。
- [ ] 当 `prompt_kd_weight>0` 时，如果存在 attention mask，先验证 shape 并构造 `attention_validity = attention_mask != 0`；此时 `response_validity` 也必须与 `attention_validity` 相与，使 padding 具有最高优先级。
- [ ] 当 `prompt_kd_weight>0` 时构造 `prompt_validity = (labels == -100)`；若有 attention mask，同样与 `attention_validity` 相与。source weight = response validity 的 1.0 + prompt validity 的配置权重。
- [ ] response 的正常有效 target 权重始终 1.0；prompt 权重为配置值。`prompt_kd_weight=0` 与当前 labels precedence 完全一致；`prompt_kd_weight>0` 时所有 attention padding 不论 labels 内容都必须为 0。
- [ ] labels 不存在时保持当前 attention/no-metadata fallback，不应用 prompt 权重。
- [ ] 最后统一 causal shift，输出 `[B,L]`、float32、与 logits 同 device；最后一个位置固定 0。
- [ ] 不修改 `_masked_token_kl_mean`、reverse KL、Top-K reducer、MSE reducer；它们已支持 float mask 和 `mask.sum()` normalization。
- [ ] 运行 `pytest tests/test_distill_losses.py -q`，Task 1 全绿。

---


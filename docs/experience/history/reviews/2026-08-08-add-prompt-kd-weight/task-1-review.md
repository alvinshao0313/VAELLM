> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-1-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Review: Add Weighted-Mask Regression Tests First

**Reviewer:** gate review (task-scoped)  
**Base:** `b41153cb59bd6ce894c80e493a67d9d4ab7a4fa5`  
**Head:** uncommitted working tree  
**Date:** 2026-08-10

---

## 1. Spec Compliance: ✅

| Brief requirement | Status | Evidence (diff / file) |
|---|---|---|
| 仅改 `tests/test_distill_losses.py` | ✅ | `git diff --name-only` 仅 1 文件；+287 行 |
| 不改生产代码 | ✅ | 无 `train_utils/distill_losses.py` 变更 |
| `prompt_kd_weight` 省略 vs 显式 `0.0` 必须 `torch.equal` | ✅ | L28–47：`test_distill_mask_prompt_weight_zero_is_exact_current_behavior`，期望 `[0,0,1,1,1,0]` |
| 分数 prompt 权重 + causal shift：`0.1` → `[0.1,0.1,1,1,1,0]` | ✅ | L50–63：`test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift` |
| padding：`labels==-100` 且 `attention_mask==0` 不得获 prompt 权重 | ✅ | L66–79：计划文档精确样例 `0.1,1,1,0,0,0` |
| `prompt_kd_weight=1.0` 等于 shifted attention validity，末位为 0 | ✅ | L82–109：`test_distill_mask_prompt_weight_one_equals_shifted_attention_validity`，动态构造 `expected[:, :-1] = attention_validity[:, 1:]` |
| interleaved / 多轮中间 `-100` 按 prompt weight | ✅ | L112–125：labels `[-100,10,-100,11,2]` → `[1.0,0.1,1.0,1.0,0.0]` |
| `labels=None` fallback 忽略 `prompt_kd_weight` | ✅ | L128–181：attention-only 与 no-metadata 两段均断言 `without == with_weight` |
| 负值 `< 0` 必须 `ValueError` | ✅ | L184–195：`pytest.raises(ValueError, match="prompt_kd_weight")` |
| `> 1`（如 `2.0`）必须接受并出现在 mask | ✅ | L198–211：期望 `[2.0,2.0,1.0,0.0]` |
| gradient：p=0 prompt 梯度 0；p>0 非零；response 非零；padding/末位 0 | ✅ | L214–280：`test_forward_kl_gradient_respects_fractional_prompt_weights` 三段断言 |
| weighted-mean 手算与 `compute_forward_kl_loss` 一致 | ✅ | L283–312：手算 `(token_kl * mask).sum() / mask.sum().clamp_min(1.0)`，与 `_masked_token_kl_mean` 一致 |
| TDD RED：实现前新增测试应失败 | ✅ | report 记录 `10 failed, 32 passed`；当前 API 无 `prompt_kd_weight` → `TypeError` |

### Missing

无。

### Extra

无越界文件或额外测试类别。

### Misunderstood

无。所有期望向量与 `docs/superpowers/plans/2026-08-08-add-prompt-kd-weight.md` 的 Required Mathematical Semantics 一致（含 padding 精确样例、interleaved 逐 token 判定、fallback 语义）。

### ⚠️ Notes (non-blocking)

- **RED 失败形态**：`test_distill_mask_rejects_negative_prompt_kd_weight`（L184–195）当前因未知参数以 `TypeError` 失败，而非 `ValueError`。测试断言本身正确；Task 2 加入参数与校验后应自然转绿。report 已说明。
- **未在本 review 中重跑 pytest**：依据 diff 与当前 `build_distill_token_mask` 签名（无 `prompt_kd_weight`），RED 结论可信。

---

## 2. Task Quality: Approved

实现质量良好：命名遵循现有 `test_distill_mask_*` / `test_forward_kl_*` 惯例；数值期望与计划文档一致；gradient 测试沿用文件中既有模式（`grad.abs().sum(dim=-1)`）；weighted-mean 手算路径与 `train_utils/distill_losses.py` 中 `_masked_token_kl_mean` + `compute_forward_kl_loss(temp=1.0)` 对齐。

### Critical

无。

### Important

无。

### Minor

1. **weighted-mean 手算未显式乘 `temp * temp`**（L835–841）：`compute_forward_kl_loss` 在 `temp != 1` 时会乘 `temp²`；当前 `temperature=1.0` 下正确，但若日后改温度需同步手算。
2. **padding gradient 子场景**（L262–280）只断言 padding 位置 3/4 梯度为 0，未再断言有效 response 位置仍非零；brief 对 padding/末位的约束已覆盖，非缺陷。

---

## Verdict Summary

| Dimension | Result |
|---|---|
| **Spec compliance** | ✅ |
| **Task quality** | **Approved** |

**Gate decision:** 通过。可进入 Task 2（在 `build_distill_token_mask` 实现 `prompt_kd_weight: float = 0.0` 及加权逻辑）。

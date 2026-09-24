> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-10-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 10: Closed-Loop Verification

- [ ] 激活 `bitvae` 并确认解释器。
- [ ] 依次运行：

```text
pytest tests/test_distill_losses.py -q
pytest tests/test_cat_eval_adapter_match.py -q
pytest tests/test_e2e_dataset_mix.py -q
pytest tests/smoke/test_loss_pipeline_smoke.py -q
pytest tests/smoke/test_one_step_train_smoke.py -q
```

- [ ] 再组合运行上述五个文件，确认没有测试间状态污染。
- [ ] 条件允许时运行 `pytest tests -q`。若有已有无关失败，只记录和验证其与本 patch 无关，不扩展范围修复。
- [ ] 全仓搜索 `prompt_kd_weight|distill_prompt_kd_weight`，人工核对完整链路：category CLI -> normalized override -> runtime config -> stage config -> CustomSFTTrainer -> shared mask；E2E CLI -> args -> runtime -> trainer -> dense/CPU/gamma 三条路径。
- [ ] 搜索两个 trainer 中 `build_distill_token_mask`，确保没有漏掉某个 loss branch。
- [ ] 检查最终 diff，不得出现数据截断、样本过滤、loss type、checkpoint、数据比例、优化器/学习率等无关改动。
- [ ] 不执行 git commit。

## Acceptance Criteria

- [ ] 两套 CLI 均存在且默认 0.0。
- [ ] 不传参数与显式 0.0 的 mask 完全相同。
- [ ] p=0 时现有 response-target KD 数值/梯度语义不变。
- [ ] p>0 时非 padding prompt target 得到指定权重。
- [ ] response target 始终 1.0。
- [ ] negative weight 被拒绝；>1.0 被允许。
- [ ] padding 和 final logits 始终 0。
- [ ] 预测 EOS 的 logits 仍为 1.0。
- [ ] CE、hidden loss、pre-MLP hidden loss 均未改变。
- [ ] 所有 category tokenwise KD 分支都使用统一 weighted mask。
- [ ] E2E dense、CPU student loss、CPU teacher gamma 使用统一 weighted mask。
- [ ] fractional mask 下 EAKLD dense/offload loss 与 gradient 测试通过。
- [ ] 默认脚本行为保持不变。
- [ ] checkpoint、推理、数据截断和数据配比未改变。

## Completion Report Template

实现者完成后只报告事实：新增了可配置 prompt-weighted logit KD；response 固定 1.0，prompt 默认 0.0；0.0 精确保留当前行为；CE/hidden loss 不变；EAKLD teacher entropy/gamma 与 KL 使用同一 weighted mask；列出实际测试命令和结果。不要在没有控制实验前宣称 downstream accuracy 得到提升。

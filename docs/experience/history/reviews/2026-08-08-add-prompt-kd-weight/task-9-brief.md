> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-9-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 9: Documentation

- [ ] `docs/cat_train_args.md` 增加 category 参数：默认 `default=0.0`、范围 `>=0`、支持 after-category override。
- [ ] `compressed_e2e_fintuning/README.md` 增加 Prompt KD weighting 小节。
- [ ] 文档明确：0.0 = 当前 response-target-only KD；0.05 表示 prompt token 的相对权重为 response 的 5%；1.0 表示所有有效 next-token 位置等权。
- [ ] 明确该权重不改变 CE 和 hidden loss。
- [ ] 明确 padding/final logits 始终排除，EOS target 仍包含。
- [ ] 明确 EAKLD entropy/gamma 和 KL 共用同一 weighted mask。
- [ ] 0.05/0.1 只能作为实验示例，不写成推荐最优值或已验证结论。

---


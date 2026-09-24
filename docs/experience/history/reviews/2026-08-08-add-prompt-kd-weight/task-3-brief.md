> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 3: Verify EAKLD Uses the Same Fractional Mask Everywhere

**File:** `tests/test_distill_losses.py`

- [ ] 新增 teacher entropy/gamma fractional-mask 测试。用含 0.1/1.0/0 的 mask 手算 `weighted entropy / mask.sum()`，验证 `teacher_entropy_mean`、gamma 与 `valid_count`。
- [ ] 保留现有 binary dense-vs-CPU tests，再增加 fractional-mask 版本，覆盖 `compute_eakld` 和 `compute_eakld_topk`。
- [ ] fractional 情况必须验证 dense loss == CPU chunked loss，且 student gradients 一致。
- [ ] `valid_tokens` 在 fractional mask 下表示 effective token-weight mass，即 `mask.sum()`，允许非整数。不要改 telemetry key 或 checkpoint 数据结构。
- [ ] 如果这些测试无需生产改动即可通过，不要为了“显式支持”而重构 EAKLD。

---


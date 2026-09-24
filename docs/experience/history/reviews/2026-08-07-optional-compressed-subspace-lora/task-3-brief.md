> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 3 — `VAELinear` scope + single shape validator + patch semantics

- [ ] constructor 增 `low_rank_scope=LOW_RANK_SCOPE_FULL`。
- [ ] 实现 `_expected_low_rank_shape_for_scope()` 与 `_validate_low_rank_payload_tensors()`，constructor 复用。
- [ ] 实现 scope-aware low-rank patch helper。
- [ ] 固定 full/subspace finalize 顺序，full 原位置不移动。
- [ ] 跑 input/output protected-channel 数值测试与 no-protection equivalence。
- [ ] 跑现有 VAELinear 相关测试，确认 full 没回归。


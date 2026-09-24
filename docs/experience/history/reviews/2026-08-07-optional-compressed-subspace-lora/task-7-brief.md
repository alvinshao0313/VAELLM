> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-7-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 7 — category PEFT-carrier routing + surrounding traversals

- [ ] full 原 `PeftVAELinearProxy` 路径不变。
- [ ] subspace 使用 `CompressedSubspacePeftProxy + inject_adapter_in_model()`，不创建 root `PeftModel`。
- [ ] scope mismatch early failure；续蒸只允许相同 scope/rank。
- [ ] `both` 支持 decoder + PEFT carrier LoRA A/B。
- [ ] carrier base storage 恒为 O(1)，不得 materialize full/compressed dense zero matrix。
- [ ] 修改 HiF4 logical-linear collector，确保只 hook 顶层 subspace proxy。
- [ ] 修改 temporary-mode traversal，确保只由 proxy 同步 base temporary 状态一次。
- [ ] 修改 logical VAELinear iterator，使其返回顶层 subspace proxy/base ref 而不是嵌套 `.base_layer`；不改变现有 root-PeftModel 前缀行为。
- [ ] 修改 after-category/final save guard，禁止未导出的 subspace proxy 进入 saver。
- [ ] export 后恢复 bare `VAELinear`，carrier/PEFT wrapper 均不存在。
- [ ] route/integration tests。


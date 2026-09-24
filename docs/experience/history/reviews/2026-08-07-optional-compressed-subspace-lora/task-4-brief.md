> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 4 — 完成 subspace PEFT proxy/lifecycle helpers

- [ ] 实现 `CompressedSubspacePeftProxy` 与 compressed input/output index 预计算。
- [ ] 实现 `_subspace_proxy_root()`、iterator、精确 target resolver。
- [ ] 实现 category `inject_compressed_subspace_peft_lora()`。
- [ ] 实现 PEFT effective payload restore/extract，并复用 `VAELinear._validate_low_rank_payload_tensors()`。
- [ ] 实现 wrap/export/unwrap；export 必须先校验 candidate payload，再原子式更新 scope+A/B。
- [ ] 跑 proxy forward/export equivalence、root-PeftModel path resolver、O(1) carrier storage tests。


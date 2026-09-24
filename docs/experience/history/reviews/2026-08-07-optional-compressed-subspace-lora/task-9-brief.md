> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-9-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 9 — compressed E2E root-PEFT subspace path

- [ ] 新增 uniform scope validator；`compressed_lora` 保留现有 uniform-rank，`both` 不新增 uniform-rank。
- [ ] full `compressed_lora` -> 当前 `_build_low_rank_peft_model()`，行为不变。
- [ ] subspace `compressed_lora` -> `_build_subspace_low_rank_peft_model()`：先 proxy/carrier，再 `get_peft_model()`。
- [ ] full/subspace `compressed_lora` 都保持 root `PeftModel`，继续共用 `_peft_base_model()` 的 device-map/streaming 逻辑。
- [ ] E2E subspace `LoraConfig` 固定 `alpha=rank, dropout=0.0`。
- [ ] root `PeftModel` 下 payload init/extract 必须通过 `_subspace_proxy_root()` 保留原 module path。
- [ ] Trainer checkpoint/resume 保持现有 PEFT 语义；增加 step1 save -> fresh reconstruction -> resume step2 测试。
- [ ] final extraction 按 scope 分支，随后统一 reload source + `expected_scope` writeback。
- [ ] `both` 直接训练 subspace payload，不创建 proxy，且不改变现有 rank 灵活性。
- [ ] E2E routing/roundtrip/resume tests。


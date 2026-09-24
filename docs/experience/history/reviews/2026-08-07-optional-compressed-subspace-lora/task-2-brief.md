> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 2 — scope truth source + PEFT 0.10.0 carrier compatibility gate

- [ ] 创建 `litebsq/low_rank_scope.py`，只放 scope constants + `normalize_low_rank_scope()`，不 import `VAELinear`。
- [ ] 创建 `e2e_common/compressed_subspace_lora.py`，此阶段只实现 `_resolve_proxy_device_dtype()` 与 `PeftZeroLinearCarrier`。
- [ ] 写两个 PEFT 0.10.0 compatibility gate tests：`inject_adapter_in_model` 与 `get_peft_model`。
- [ ] 在 `bitvae` 下先跑 gate；任一失败则停止后续实现并汇报，不得切换架构。
- [ ] gate 通过后确认 sentinel `weight.numel()==1`、A/B shape 为 `[r,Ic]`/`[Oc,r]`、backward finite。


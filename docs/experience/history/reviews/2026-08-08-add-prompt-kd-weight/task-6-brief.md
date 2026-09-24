> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-6-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 6: Add E2E CLI and Runtime Chain

**Files:** `compressed_e2e_fintuning/args.py`, `compressed_e2e_fintuning/runtime.py`, `tests/test_e2e_dataset_mix.py`

- [ ] 参数测试先行：默认 0.0；0.05 可接受；2.0 可接受；负值 `SystemExit`。
- [ ] 对 `dataset_task=mcqa`：0.0 可保留，非零必须 parser error，因为 choice KD 不使用 token mask，不能 silent no-op。
- [ ] `VAEDecoderE2EArguments` 增加 `prompt_kd_weight: float = 0.0`。
- [ ] parser 增加 `--prompt_kd_weight`，type=float，default=0.0。
- [ ] parse validation 拒绝 `<0`。
- [ ] runtime 增加独立日志：prompt KD weight 的 resolved 值，以及 response weight 固定 1.0。
- [ ] runtime 构造 `VAEDecoderE2ETrainer` 时传 `prompt_kd_weight=float(args.prompt_kd_weight)`。
- [ ] 运行 `pytest tests/test_e2e_dataset_mix.py -q`。

---


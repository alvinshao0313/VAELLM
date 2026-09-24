> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-7-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 7: Unify E2E Dense, CPU-Offload and Teacher Gamma Mask

**Files:** `compressed_e2e_fintuning/trainer.py`, `tests/test_e2e_dataset_mix.py`, two smoke-test files

- [ ] `VAEDecoderE2ETrainer.__init__` 增加 `prompt_kd_weight=0.0`，保存并拒绝负值。
- [ ] 新增 private `_build_distill_token_mask(inputs, reference_logits)`，内部唯一调用共享 helper 并传 `self.prompt_kd_weight`。
- [ ] `_compute_legacy_dense_loss()` 使用 private helper 生成 token mask。
- [ ] `_compute_teacher_first_cpu_loss()` 使用同一个 private helper。
- [ ] `_build_cpu_teacher_targets()` 中计算 teacher entropy/gamma 的 `gamma_mask` 也必须使用同一个 private helper。这一点是硬性要求，否则 CPU EAKLD 的 gamma 与 KL 会使用不同权重。
- [ ] 在 `tests/test_e2e_dataset_mix.py` 用轻量方式验证 private helper 确实把 0.1 转发到共享 mask；不需要启动大模型。
- [ ] 运行 `rg -n "build_distill_token_mask" compressed_e2e_fintuning/trainer.py`。目标是 1 个 import + 1 个 private helper 内真实调用，三个生产路径都走 private helper。
- [ ] `tests/smoke/test_loss_pipeline_smoke.py` 至少一组 synthetic loss pipeline 使用 `prompt_kd_weight=0.1` 生成 fractional mask；断言存在 0~1 之间的权重，并让全部 `DENSE_LOSS_TYPES` forward/backward 通过。
- [ ] 同一 smoke 中 EAKLD dense 与 CPU-offload 必须使用相同 fractional mask，loss、telemetry、gradient 保持一致。
- [ ] `tests/smoke/test_one_step_train_smoke.py` 的 E2E trainer builder 增加 prompt weight 参数；dense EAKLD 和 CPU-offload EAKLD 至少各跑一组 0.1 one-step，验证有限 loss、反传、参数更新、telemetry 和 CPU entropy 单次计算逻辑。
- [ ] 运行 `pytest tests/smoke/test_loss_pipeline_smoke.py tests/smoke/test_one_step_train_smoke.py -q`。

---


> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 4: Split CPU-Offloaded EAKLD Teacher Scalars by Region

**Files:** `compressed_e2e_fintuning/teacher_targets.py`, `compressed_e2e_fintuning/trainer.py`, `tests/test_e2e_teacher_first.py`

Keep existing `TeacherTargetBatch` EAKLD scalar fields as response-region fields. Add exactly these prompt-region fields:

```python
eakld_prompt_gamma_cpu: Optional[torch.Tensor] = None
teacher_prompt_entropy_mean_cpu: Optional[torch.Tensor] = None
teacher_prompt_valid_token_count_cpu: Optional[torch.Tensor] = None
```

- [ ] Extend `TeacherTargetBatch.clear()` to reset all three new fields.
- [ ] Replace E2E `_build_distill_token_mask()` with `_build_distill_token_regions()` returning the shared region dataclass. This builder must not record telemetry counts itself.
- [ ] In `_build_cpu_teacher_targets()`, when teacher logits are required, always compute response entropy/gamma/count using response mask and store them in the existing fields.
- [ ] Only when prompt weight is positive, compute prompt entropy/gamma/count from prompt mask and store detached CPU float32 scalars in the new fields.
- [ ] When prompt weight is zero, do not call the prompt entropy/gamma path and leave all new fields unset.
- [ ] Reuse the same single CPU copy of full teacher logits for both regional losses; do not duplicate the logits tensor.
- [ ] Test zero weight: response scalars populated, prompt scalars absent, entropy/gamma helper called once.
- [ ] Test positive weight: both scalar sets populated, helper called twice with distinct binary masks, and each valid count equals its region-mask sum.

---


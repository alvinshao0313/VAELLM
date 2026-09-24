> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-5-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 5: Keep E2E Dense and CPU-Offload Paths Mathematically Identical

**Files:** `compressed_e2e_fintuning/trainer.py`, `e2e_common/dense_loss.py`, E2E smoke tests

- [ ] Legacy dense E2E path builds student regions once and passes response mask, prompt mask, and prompt weight to `compute_dense_loss_from_logits()`.
- [ ] CPU-offload student path builds student regions once and passes both masks plus response and prompt EAKLD scalar sets to `compute_dense_loss_from_offloaded_teacher()`.
- [ ] Extend the offloaded dispatcher with prompt gamma, prompt entropy mean, and prompt valid-token count arguments corresponding exactly to the new `TeacherTargetBatch` fields.
- [ ] Positive prompt weight requires prompt mask plus all three prompt scalar values. Missing data is an explicit error; no silent response-only fallback.
- [ ] Offloaded response EAKLD uses response mask/gamma and existing telemetry. Prompt EAKLD uses prompt mask/prompt gamma and does not overwrite existing response telemetry. Combine only after both regional means are computed.
- [ ] `eakld_kd` mixes CE once after regional EAKLD combination.
- [ ] Add dense-versus-offload equality tests with positive prompt weight for full EAKLD and a small top-k EAKLD variant. Compare scalar value and student gradients using existing numerical tolerances.
- [ ] Add zero-weight test proving prompt scalar fields are unnecessary and result matches current response-only path.

---


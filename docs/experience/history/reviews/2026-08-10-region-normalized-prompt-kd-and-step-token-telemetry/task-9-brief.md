> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-9-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 9: Documentation Uses the New Region-Level Meaning

**Files:** `docs/cat_train_args.md`, `compressed_e2e_fintuning/README.md`

- [ ] Remove wording that describes prompt weighting as a fractional per-token mask inside one common denominator.
- [ ] Document exact formula `L_logit = L_response + prompt_kd_weight * L_prompt`.
- [ ] State response and prompt independently compute their own token mean before combination.
- [ ] State value 0.03 means the prompt-region mean contributes with coefficient 0.03, independent of prompt/response length ratio.
- [ ] State EAKLD computes response and prompt entropy/gamma independently.
- [ ] State existing `eakld/*` telemetry continues to describe response-region EAKLD.
- [ ] Document prompt/response token telemetry as true post-truncation, non-padding token counts derived from `labels` plus `attention_mask`, averaged over each regular logging window across gradient accumulation and DDP; explicitly state it is not causal KD-mask counting.
- [ ] Do not claim 0.03 is empirically optimal or claim downstream improvement before a controlled run.

---


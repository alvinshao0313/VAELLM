> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-9-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Report: Documentation Uses the New Region-Level Meaning

## Status

Complete. Documentation updated to match region-normalized prompt KD and logging-window token telemetry semantics.

## Commits

None (per instructions).

## Files Changed

| File | Change |
|------|--------|
| `docs/cat_train_args.md` | Rewrote §6.11.1 Prompt KD weighting: removed fractional per-token mask / shared-denominator wording; documented `L_logit = L_response + prompt_kd_weight * L_prompt`; independent region token means; 0.03 as region-mean coefficient example; EAKLD per-region entropy/gamma with `eakld/*` telemetry scoped to response region; added §6.11.2 LoRA token telemetry. Updated parameter table row for `--distill_prompt_kd_weight`. |
| `compressed_e2e_fintuning/README.md` | Rewrote Prompt KD weighting section with same region-level formula and semantics; added Token telemetry section for `E2E token stats:` lines. Removed old mask/relative-weight wording. |

## Brief Checklist

- [x] Remove fractional per-token mask / shared denominator wording
- [x] Document exact formula `L_logit = L_response + prompt_kd_weight * L_prompt`
- [x] State response and prompt independently compute their own token mean before combination
- [x] State 0.03 means prompt-region mean coefficient 0.03, independent of length ratio
- [x] State EAKLD independent entropy/gamma per region
- [x] State existing `eakld/*` telemetry remains response-region
- [x] Document token telemetry as post-truncation non-padding counts from labels+attention, averaged over logging window across grad accum + DDP; not causal KD-mask counting
- [x] Do not claim 0.03 is empirically optimal or claim downstream improvement

## Verification

Grep sanity check: no remaining `weighted mask`, `相对权重`, `5%`, or `sum(token_loss * mask)` in either doc file.

No pytest run (docs-only change; grep check sufficient).

## Concerns

None. The old plan doc `docs/superpowers/plans/2026-08-08-add-prompt-kd-weight.md` still describes the superseded weighted-mask semantics; out of scope for Task 9 unless a follow-up doc sweep is requested.

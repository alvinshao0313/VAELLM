> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-9-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Review: Documentation Updates

**Scope:** `docs/cat_train_args.md`, `compressed_e2e_fintuning/README.md` (docs-only)

## Spec: ✅

All eight brief checklist items are satisfied in the changed sections.

| Brief item | Verdict |
|---|---|
| Remove fractional / shared-denominator wording | ✅ Grep clean; old mask/5%/相对权重/`sum(token_loss * mask)` removed |
| `L_logit = L_response + prompt_kd_weight * L_prompt` | ✅ Both files, with region definitions |
| Independent region token means before combination | ✅ Explicit in §6.11.1 and E2E README |
| 0.03 as region-mean coefficient, length-ratio independent | ✅ Stated with example |
| EAKLD independent entropy/gamma per region | ✅ Both files |
| `eakld/*` telemetry = response region only | ✅ Both files |
| Token telemetry: post-truncation labels+mask counts, logging-window avg over grad accum+DDP, not KD-mask | ✅ §6.11.2 + E2E Token telemetry |
| No optimality / downstream claims | ✅ Disclaimed in both files |

## Findings

1. **Minor (nit):** `docs/cat_train_args.md` §2.2 after-category override example (line ~94) still uses `after:q_proj=0.05` while §6.11.1 examples were updated to `0.03`. Syntax-demo only—not old semantics—but inconsistent; consider aligning to `0.03`.

No other issues. Diff touches only the two in-scope doc files.

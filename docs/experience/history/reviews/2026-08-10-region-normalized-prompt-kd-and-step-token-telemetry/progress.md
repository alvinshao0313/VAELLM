> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/progress.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# SDD ledger — plan: docs/superpowers/plans/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry.md

Workspace constraints (plan + AGENTS.md override SDD defaults):
- No worktree, no branch switch, work on current checkout
- No git commit unless user explicitly asks
- Tests in conda `bitvae` only
- Do not modify scripts/catlora_distill_4gpu_res0.sh, e2e_decoder.sh, lazy_datasets, checkpoint/eval/VAE

BASE (pre-Task-1): d393552

Task 1: complete (uncommitted working tree, review clean)
- Implementer: c6657d41-9cb2-4171-9e4d-e96a7fb24974
- Reviewer: bea8cb89-3383-45e2-8ca4-d1a36fa44956
- Notes: 46 passed; minors deferred (tensor_name in error msg; removed zero-weight equivalence test covered elsewhere)
- Controller verified: pytest tests/test_distill_losses.py -q → 46 passed

Task 2: complete (uncommitted, review clean after fix round 1)
- Implementer: e41d8fca-d171-4341-a011-c2c381ec56a4
- Reviewer: ab49fe53-cbb9-4b62-90eb-013a4e37e0d4
- Re-review: cacca0c9-2e9a-4188-9693-0ac898cd5c35
- Notes: 60 passed; F1 anti-regression test fixed to distinguish shared-denominator formula
- Deferred minors: F2/F3/F4 from initial review (offload prompt gamma → Task 5)

Task 3: complete (uncommitted, review clean)
- Implementer: ff732eb6-132f-4122-8254-73badc04abb7
- Reviewer: 32e5fca8-6272-44b9-9ea4-ca9b69274bcc
- Notes: shared build_token_regions + combine_region_loss; 5 new tests; smoke eakld failures are E2E trainer Task1 leftover → Tasks 4–5

Task 4: complete (uncommitted, review clean with deferred notes)
- Implementer: ddfe3566-e446-43c5-b7be-deeaca9c1ead
- Reviewer: fdff8851-fec2-49b4-82a0-0227accf10c6
- Notes: prompt scalar fields + regions builder; dispatcher wiring deferred to Task 5
- Parked for Task 5: smoke already partially passes prompt_mask; empty prompt_mask gamma finiteness

Task 5: complete (uncommitted, review clean)
- Implementer: f3da74cc-2d4d-461c-a634-536f83c036b2
- Reviewer: 10288b58-0dcd-4c75-a23d-d56e1731370d
- Notes: dense/offload parity at positive weight; minors deferred (naming, eakld_kd equality test gap)

Task 6: complete (uncommitted, review clean)
- Implementer: 86022927-6558-46b7-a60a-5deef5964df1
- Reviewer: aaceeb09-2fa3-4684-9c8d-23db298dc43c
- Notes: DistillTokenStatsAccumulator + 10 tests; minors deferred

Task 7: complete (uncommitted, review clean)
- Implementer: 2c682929-a2e6-4ffd-a4fa-5f2a95938b11
- Reviewer: 01e32e60-e756-4c6c-a1b3-76bc406cbb87
- Notes: LoRA token stats callback + 7 tests; minors deferred

Task 7: complete (uncommitted, review clean)
- Implementer: 2c682929-a2e6-4ffd-a4fa-5f2a95938b11
- Reviewer: 01e32e60-e756-4c6c-a1b3-76bc406cbb87
- Notes: LoRA token stats callback + 7 tests; minors deferred

Task 8: complete (uncommitted, review clean after fix round 1)
- Implementer: 3b4496d7-a096-4b8c-a3ab-be20eff57159
- Reviewer: c8c98f6a-1a65-4dca-abda-d3806cc016d5
- Re-review: 2394c9e6-bd96-4897-b7c7-37c9eee036a1
- Notes: E2E token stats; fixed stale PromptKdMaskHelperTest; other dataset_mix failures pre-existing/out of scope

Task 9: complete (uncommitted, review clean)
- Implementer: 456dcef2-5b84-4135-bb26-fc2d91117edc
- Reviewer: ebfbcc7b-094d-434a-997d-ca35364f9932
- Notes: docs updated; minor §2.2 example still 0.05 deferred

Task 10: complete (uncommitted, DONE_WITH_CONCERNS)
- Controller ran full verification directly
- Plan-focused suites green except 9 pre-existing test_e2e_dataset_mix failures (lazy_datasets/dummy.txt/eval_before_save; out of plan scope)
- PromptKdMaskHelperTest fixed and passing
- Sample line: LoRA token stats: step=10 window_optimizer_steps=10 avg_prompt_tokens=3.0000 avg_response_tokens=3.0000 global_samples=10
- Deferred minors from earlier tasks: see individual reviews

Final review: clean (fd68aac0-cba0-465b-916c-8cdadd209bd1) — merge-ready; no Critical/Important
- 9 dataset_mix failures confirmed pre-existing at base d393552
- SDD workspace retained until user commits (work still uncommitted)

Finish choice: 3 — keep working tree as-is (user will handle later). No commit/push/merge. SDD workspace retained.

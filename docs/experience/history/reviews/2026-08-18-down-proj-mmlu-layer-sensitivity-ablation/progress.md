> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/progress.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# SDD ledger — plan: docs/superpowers/plans/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation.md

Controller notes:
- No git worktree (plan + AGENTS.md forbid)
- No auto git commit (plan + AGENTS.md forbid); reviews use working-tree diffs
- Workdir: /home/shaoyuantian/program/VAELLM
- Checkpoint: .result/catlora/res0-bf16-protect-channel-vae/final_model
- GPUs: 8x A800 80GB
- Env: bitvae


Task 1: complete (working tree, review clean)
Task 1: minor (deferred): unload helper does not re-assert non-down temporary=True
Task 1: minor (deferred): prewarm redoes reset/to/eval already done in load_worker_model
Task 1: minor (deferred): duplicate-index discovery case not isolatable; optional coverage gaps for n!=36 and always_use_original in reset

Task 2: complete (working tree, review clean)

Task 3: complete (working tree, review clean)
Task 3: minor (deferred): optional untested extra validations (job/manifest mode consistency etc.)

Task 4: complete (working tree, review clean)
Task 4: note — summarize_phase1/build_phase2_manifests wired in main; implementations land in Task 5/6

Task 5: complete (working tree, review clean)
Task 5: note for Task 6 — on summarize_phase1 failure, set run_config.status=failed; use ranked list[int] return value not CSV

Task 6: complete (working tree, review clean)

Task 7: complete (working tree, review clean)
Task 7: minor (deferred): report §3 worker00 repeat may reuse compressed accuracy field; historical refs from module constants

Task 8: complete (working tree, review clean)

Task 9: BLOCKED — all 8 GPUs occupied by other users; smoke OOM on GPU 5 during prewarm (~45GB model + existing ~33GB process)
Task 9: env+pytest PASS (77); smoke FAIL OOM; git isolation PASS; waiting on free GPUs

Prewarm CPU staging (follow-up): implemented in experiments/down_layer_sensitivity/core.py
- 81 unit tests passed
- smoke on GPU5 progressed past up_proj to down_proj, still OOM (~45GB self + ~33GB co-tenant); staging works, hoist/co-tenant still tight

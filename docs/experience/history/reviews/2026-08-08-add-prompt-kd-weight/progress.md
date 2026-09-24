> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/progress.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# SDD ledger — plan: docs/superpowers/plans/2026-08-08-add-prompt-kd-weight.md

Workspace constraints (plan + AGENTS.md override SDD defaults):
- No worktree, no branch switch, work on current checkout
- No git commit unless user explicitly asks
- Tests in conda `bitvae` only
- Strict TDD: failing tests before production code

BASE (pre-Task-1): b41153cb59bd6ce894c80e493a67d9d4ab7a4fa5

Task 1: complete (uncommitted working tree, review clean)
- Implementer: b7539c7e-a739-46af-8f07-817dfbb0c3f4
- Reviewer: db20087c-4fb3-47c1-aba3-c3be3472ca89
- Notes: 10 failing tests RED as expected; no production changes

Task 2: complete (uncommitted, review clean)
- Implementer: 05163834-9cf5-457f-9aa6-d85d0f573d9c
- Reviewer: e20de202-2dd1-49b8-a350-f0bb317ba5f8
- Notes: build_distill_token_mask prompt_kd_weight; 42 passed; 2 Task1 expectation fixes legitimated

Task 3: complete (uncommitted, review clean)
- Implementer: 2a2dbc68-4f4f-4d62-8518-05a7bcf87225
- Reviewer: 764542fb-155a-443a-9a07-e7e1797db57b
- Notes: 5 EAKLD fractional tests; no production changes; 51 passed

Task 4: complete (uncommitted, review clean)
- Implementer: e7b6c5e2-05d1-4767-959a-9da2f4b84749
- Reviewer: 37ee9ddd-b010-4981-a113-c058c2f2ec0b
- Notes: CLI OverrideTable chain + trainer init stub; 23 passed

Task 5: complete (uncommitted, review clean)
- Implementer: 22fbd57c-9b0e-4714-9187-b5a8f59ba9e9
- Reviewer: b4bdc4ed-a165-4156-a213-6acff1eac2ec
- Notes: local build_token_mask; all KD branches; 74 passed

Task 6: complete (uncommitted, awaiting review)
- Notes: E2E CLI/runtime chain + trainer init stub; VAEE2EPromptKdWeightArgsTest 6 passed; full file 9 pre-existing failures unrelated

Task 6: complete (uncommitted, review clean)
- Implementer: 18830e57-b858-4356-b5b8-a13046479f94
- Reviewer: 71edc3d9-58bd-48f8-a074-5bb86723a2dd
- Notes: E2E CLI+runtime+trainer stub; 6 new tests pass; 9 pre-existing failures unrelated

Task 7: complete (uncommitted, review clean)
- Implementer: cb6f1e3f-3857-4f0b-a414-8498aca1b9cd
- Reviewer: 4078954a-532c-4441-9c22-bf1045bb953a
- Notes: E2E private helper for dense/CPU/gamma; smokes 8-9 passed

Task 8: complete (uncommitted, review clean)
- Implementer: 674a5346-784f-4902-8899-f667b561c45e
- Reviewer: a1839312-531a-49ba-b933-2d307c9d4a33

Task 9: complete (uncommitted, review clean)
- Implementer: 10c2e65b-daca-4469-9427-154c4f1c5862
- Reviewer: 74ca7cf5-3bc9-4f35-9e86-581bb253cce6

Task 10: complete (uncommitted, verification PASS with notes)
- Implementer: e55940b2-5135-4c14-9577-ce1ee3eee030
- Notes: required suites green except 9 pre-existing e2e_dataset_mix; fixed gitignored test_cat_compressed_lora_scope.py _fake_cfg
- Deferred minors from earlier tasks: none blocking

Final review: clean (7d44ffbb-e13a-4f02-b826-99cbbc1c5cb7) — merge-ready; no Critical/Important

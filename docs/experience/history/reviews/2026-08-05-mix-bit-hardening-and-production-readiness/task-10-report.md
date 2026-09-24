> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-10-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 10 Report — Full Regression, Static Guards and Production Smoke Gates

Plan: `docs/superpowers/plans/2026-08-05-mix-bit-hardening-and-production-readiness.md`
Work dir: `/home/shaoyuantian/program/VAELLM`
Python: `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`
BASE before Task 1: `bd5c253f28f7a8c086ef002b0320e97c4faf0b2d`
Commits: none (AGENTS.md overrides Step 11; working-tree changes only).

## Gate 1 — Static search gates

| Search | Scope | Result |
|---|---|---|
| `exec python tools/cat_train.py\|python tools/cat_train.py` | `mix_bit/scripts/train_candidate_single.sh` | PASS (no matches) |
| `shifted\.detach\(\)\.cpu\(\)` | `mix_bit/teacher_cache.py`, `mix_bit/cost_search.py` | PASS (no matches) |
| `reference_state[[:space:]]=\|cpu\(\)\.clone\(\)` | `mix_bit/assembler.py` | PASS (no matches) |

Manual `result_queue.get` audit in `mix_bit/cost_table.py`:

| Line | Context | Timeout | Liveness | Verdict |
|---|---|---|---|---|
| 449 | `_wait_for_startup_message` (startup) | `timeout=poll` + total deadline | `is_alive()` on Empty -> RuntimeError | PASS |
| 520 | `_wait_for_workers_ready` (ready) | `timeout=poll` + total deadline | dead-process check on Empty -> RuntimeError | PASS |
| 945 | runtime job loop | `timeout=RESULT_QUEUE_POLL_SECONDS` | dead-process check on Empty -> failure | PASS |

All startup/ready paths carry timeout; no unbounded `result_queue.get`. **Gate 1 PASS.**

## Gate 2 — `pytest mix_bit/tests -q`

```
301 passed in 28.67s
```
No failures, no unexpected skips. **Gate 2 PASS.**

## Gate 3 — Repository integration regressions

```
tests/test_cat_train_candidate_artifact_hook.py
tests/test_model_utils_auto_loader.py
tests/test_e2e_checkpoint_io_legacy.py
tests/test_temporary_switch_residency.py
tests/test_distill_losses.py
-> 57 passed in 6.60s
```
**Gate 3 PASS.**

## Gate 4 — All CLI --help exit 0

10/10 PASS: build_model_inventory, train_candidate_pool, inventory_candidate_pool, prepare_uniform_baseline, prepare_calibration, build_teacher_cache, compute_cost_table, solve_allocation, assemble_mixed_model, validate_mixed_model. **Gate 4 PASS.**

## Gate 5 — Build real Qwen3-8B inventory

```
model_id=qwen3_8b
C=7
L=252
block_count=36
total_target_parameters=6945767424
fingerprint_sha256=75033486b95bbdb49df2fbe973959684fb68f94627209a887fe2b9763073195e
```
Required C=7 L=252 block_count=36 — all present. **Gate 5 PASS.**

## Gate 6 — Candidate dry-run

```
total_trials=35
dry_run_unique_commands=35
```
Argv[2] audit: all 35 commands use `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python3.11` (absolute bitvae interpreter). **Gate 6 PASS.**

## Gate 7 — Tiny integration covers both KL modes

`mix_bit/tests/test_tiny_integration.py` imports `KL_MODE_EXACT_FULL_VOCAB`, `KL_MODE_TEACHER_TOPK`; computes `per_sample_exact_forward_kl` (line 560) and `per_sample_teacher_topk_forward_kl` (line 568); asserts `KL_MODE_TEACHER_TOPK == "teacher_topk"` (line 656); covers compact teacher cache, student K-way gather, atomic cost rows, solver, final tokenizer save, state fingerprint, strict reload. Both KL modes exercised within the 301 passed tests. **Gate 7 PASS.**

## Gate 8 — Qwen 1-step production smoke (Step 7)

Pre: Gates 1-6 PASS. GPU 4 free (3 MiB, 0%). GPUs 0-3 occupied by user's `cat_distill_from_vae_checkpoint.py` (seed 33, eakld_top_1000); smoke used GPU 4 only — no conflict.

Temp configs under `.result/mix_bit/_smoke_config/`: qwen3_8b.json (q_proj only), vae_b16d32s2.json (b16d32s2 only), vae_smoke_1step.json (steps=1, batch_size=128), run_smoke.json. Smoke inventory: C=1 L=36 block_count=36.

Real run exit 0. Artifact dir contains exactly: `module_state.pt`, `candidate_meta.json`, `completed.json`. Meta: 36 expected_module_names, 36 module_specs, mode = 16 bits / 32 dim / 2 stages. Pool index: C=1 L=36 R=1 artifact_count=1, candidate_manifest.json written.

4-token forward (loaded Qwen3-8B on GPU 4, installed 36 q_proj candidates):
```
candidates=36
installed=36
logits shape=(1, 4, 151936)
logits_all_finite=True
SMOKE_FORWARD_OK
```

Cleanup: `rm -rf .result/mix_bit/_smoke_config .result/mix_bit/_smoke_candidate_pool` — both removed. Temp configs not added to git. **Gate 8 PASS (smoke ran).**

## Gate 9 — No accidental full production workload (Step 8)

Post-smoke GPU 4 back to 3 MiB / 0%. No 35-job pool launched; no full Cost rows; user's `.result/catlora_distill/...` and `.result/catlora/...` untouched. Only new `.result/mix_bit/` artifact is `qwen3_8b/model_inventory.json` (Step 4). **Gate 9 PASS.**

## Gate 10 — README failure gates update (Step 9)

Modified `mix_bit/README.md` "执行门禁（失败即停）" to explicitly enumerate the seven required gates:

1. mode/payload mismatch
2. wrong Python executable
3. top-k full-logits CPU transfer regression
4. worker startup/runtime death
5. tokenizer fingerprint mismatch
6. custom manifest root mismatch
7. final state fingerprint mismatch

**Gate 10 PASS.**

## Gate 11 — Final git diff scope review (Step 10)

Plan-related changes (in plan file tables / implementation dependencies of Tasks 1-9):
- `mix_bit/**` (all new source, configs, scripts, tests, README)
- `train_utils/cat_train_args.py` — adds `--save_candidate_artifact` args (candidate artifact hook, tested by `test_cat_train_candidate_artifact_hook.py`)
- `train_utils/cat_train_pipeline.py` — adds candidate artifact save branch
- `train_utils/distill_losses.py` — causal mask change (tested by `test_distill_losses.py`)
- `rotation/model_utils.py` — adds `get_auto_causal_lm` auto-loader (tested by `test_model_utils_auto_loader.py`)
- `tests/test_cat_train_candidate_artifact_hook.py`, `tests/test_model_utils_auto_loader.py`, `tests/test_temporary_switch_residency.py` (new regression tests in Step 2 list)

Out-of-plan paths (pre-existing user experiment / workspace work, NOT in any task brief file table; NOT touched by this task; NOT committed):
- `.gitignore` — removes `tests/` `test/` ignore, adds `AGENTS.md` `.cursor/`
- `AGENTS.md` — adds no-auto-commit workspace rule
- `compressed_e2e_fintuning/scripts/e2e_decoder.sh` — user experiment config (loss_type, hidden_loss_weight)
- `scripts/catlora_distill_4gpu_res0.sh` — user experiment config (resume path, seed, lora_rank, distill_steps, loss_type, max_grad_norm) — matches the currently running distillation job
- `scripts/catlora_codebook_ab_down_channel_single.sh` — NEW user experiment script (146 lines)
- `output_linear_by_category/` — untracked experiment output dir

No plan source/test/README files were deleted or rewritten. No commit performed. **Gate 11 PASS (scope reviewed; out-of-plan paths listed above).**

## Acceptance Matrix

| Gate | Required | Result |
|---|---|---|
| Candidate mode metadata | five fields match candidate space | PASS (contract tests + smoke meta) |
| Actual candidate structure | stages/dim/logical bits/decoder dims match | PASS (smoke spec0: 16/32/2) |
| Candidate resume | stale/mislabeled retrains | PASS (test suite) |
| Candidate subprocess | parent absolute `sys.executable` | PASS (35/35 bitvae python3.11) |
| Teacher top-k | only `[N_valid,K]` to CPU | PASS (static gate + tests) |
| Student top-k | direct gather `[B,T,K]` | PASS (static gate + tests) |
| Exact KL | math matches legacy | PASS (test_distill_losses + tiny integration) |
| Final state verification | streaming 16 MiB SHA, no full CPU clone | PASS (static gate + tests) |
| Worker startup | 900s timeout, child death immediate fail | PASS (cost_table audit) |
| Worker runtime | any worker death immediate fail | PASS (cost_table line 945) |
| Custom pool root | `--pool_manifest.parent` is sole root | PASS (tests + README gate) |
| Calibration tokenizer | fingerprint v2, core/chat/added vocab protected | PASS (tests) |
| Final tokenizer | final dir local-only reload, same fingerprint | PASS (tests) |
| Qwen inventory | 36 blocks, 7 categories, 252 linears | PASS (C=7 L=252 block_count=36) |
| Candidate planner | 35 unique s2 jobs | PASS (total_trials=35) |
| Full tests | mixed-bit + selected regressions pass | PASS (301 + 57) |

## Summary

All 11 gates PASS. Smoke ran on GPU 4 (free), exit 0, all artifact/forward checks passed, temp dirs deleted. No production workload launched. README failure gates updated. No commit performed (AGENTS.md override). Out-of-plan paths are pre-existing user experiment/workspace changes, listed above for user awareness.

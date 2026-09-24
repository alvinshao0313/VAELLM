> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-10-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 10 Report — Closed-Loop Verification

**Status:** DONE_WITH_CONCERNS

## Environment

- conda `bitvae`: `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`
- Python 3.11.13, torch 2.6.0+cu124

## Focused pytest (individual)

| Command | Result |
|---|---|
| `PYTHONPATH=. pytest tests/test_distill_losses.py -q` | **67 passed** |
| `PYTHONPATH=. pytest tests/test_distill_token_stats.py -q` | **10 passed** |
| `PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py -q` | **28 passed** |
| `PYTHONPATH=. pytest tests/test_e2e_dataset_mix.py -q` | **39 passed, 9 failed** |
| `PYTHONPATH=. pytest tests/test_e2e_teacher_first.py -q` | **14 passed** |
| `PYTHONPATH=. pytest tests/smoke/test_loss_pipeline_smoke.py -q` | **3 passed** |
| `PYTHONPATH=. pytest tests/smoke/test_one_step_train_smoke.py -q` | **5 passed** |

Additional focused (this plan's new files):

| Command | Result |
|---|---|
| `PYTHONPATH=. pytest tests/test_lora_distill_token_stats_callback.py -q` | **7 passed** |
| `PYTHONPATH=. pytest tests/test_e2e_distill_token_stats.py -q` | **11 passed** |

## Combined invocation

```
PYTHONPATH=. pytest tests/test_distill_losses.py tests/test_distill_token_stats.py \
  tests/test_cat_eval_adapter_match.py tests/test_e2e_dataset_mix.py \
  tests/test_e2e_teacher_first.py tests/smoke/test_loss_pipeline_smoke.py \
  tests/smoke/test_one_step_train_smoke.py \
  tests/test_lora_distill_token_stats_callback.py \
  tests/test_e2e_distill_token_stats.py -q
```

Result: **184 passed, 9 failed** — all 9 failures in `tests/test_e2e_dataset_mix.py` only.

### Concern: 9 `test_e2e_dataset_mix` failures (pre-existing / out of scope)

Failure classes:
1. `eval_before_save` HfArgumentParser unused arg (1)
2. `Weighted lazy mix with multiple text_format values is not supported` in `e2e_common/lazy_datasets.py` (7) — plan forbids modifying `lazy_datasets.py`
3. missing `dummy.txt` fixture (1)

`VAEE2ETrainerPromptKdMaskHelperTest` (API rename) was fixed in Task 8 and **passes**.

These match the pattern noted in the prior prompt-kd-weight plan verification (pre-existing dataset-mix suite issues). They are **not** caused by region-normalized loss or token telemetry.

## Static audits

### A — mask / weight paths

- `build_distill_token_mask` has **no** `prompt_kd_weight` parameter (3 kwargs only).
- Production callers use `build_distill_token_regions` / `_build_distill_token_regions`.
- `prompt_kd_weight` applied only after independent region means (`combine_region_loss` / dense `_combine_region_loss`).
- Zero-weight control flow skips prompt criterion.

### B — EAKLD entropy/gamma

`compressed_e2e_fintuning/trainer.py` `_build_cpu_teacher_targets`:
- always computes response entropy/gamma
- conditionally computes prompt entropy/gamma when `prompt_kd_weight > 0`

### C — token telemetry

- `DistillTokenStatsAccumulator` in category + E2E trainers
- `update` once per student micro-batch from labels+attention
- `consume_global` only at regular `state.logging_steps` boundaries in callbacks
- prefixes: `LoRA token stats` / `E2E token stats`

## Compatibility

- `scripts/catlora_distill_4gpu_res0.sh` retains `--distill_prompt_kd_weight "default=0.03"` (unchanged vs HEAD)
- `e2e_decoder.sh` / `lazy_datasets.py` unchanged vs HEAD
- No auto-commit

## Evidence for completion report items

1. **Formula / zero-weight:** covered by `tests/test_distill_losses.py` (incl. `test_zero_prompt_weight_matches_response_only_value_and_gradient`, anti-regression invariant test). 67 passed.
2. **Separate EAKLD gamma dense+offload:** Task 4/5 tests in `test_e2e_teacher_first.py` + dense-vs-offload in `test_distill_losses.py`.
3. **Dense/offload value+grad:** Task 5 tests; included in 67 + 14 + smoke passes.
4. **Token telemetry window:** callback tests prove steps 1–9 no consume; step 10 one line; step-1 special log outside window; grad-accum included; DDP reduce-before-rank0.
5. **Sample token stats line** (synthetic window of 10 identical samples with labels `[-100,-100,-100,A,B,EOS]`):

```
LoRA token stats: step=10 window_optimizer_steps=10 avg_prompt_tokens=3.0000 avg_response_tokens=3.0000 global_samples=10
```

(Matches unit-test assertions `window_optimizer_steps=10`, `avg_prompt_tokens=3.0000`.)

6. Pytest commands/results: tables above.
7. Script 0.03 + hidden/data/checkpoint/eval untouched: confirmed via `git diff --stat HEAD` on forbidden paths (empty) and script grep.

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-4-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Review: Category-Distill `prompt_kd_weight` Parameter Chain

**Reviewer:** read-only code review  
**Date:** 2026-08-10  
**Scope:** `task-4-brief.md`, `task-4-report.md`, `task-4-review-package.diff`

---

## Verdict

| Dimension | Result |
|-----------|--------|
| **Spec** | ✅ |
| **Quality** | **Approved** |

---

## Spec Checklist (brief + global constraints)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `--distill_prompt_kd_weight` CLI, `type=str`, default `default=0.0` | ✅ | `build_cat_train_parser()` adds arg with `default="default=0.0"` |
| Parse to `OverrideTable[float]` via `process_cat_train_args()` | ✅ | `_normalize_cat_train_script_args()` uses `_parse_cat_override(..., spec=_DISTILL_PROMPT_KD_WEIGHT_SPEC)` |
| Resolve by `after_category` to `float` | ✅ | `resolve_distill_runtime_config()` calls `resolve_after_category_value(cat_args.distill_prompt_kd_weight, after_category)` |
| `_DISTILL_PROMPT_KD_WEIGHT_SPEC` reuses `_parse_nonnegative_float_text()` | ✅ | Spec uses `_parse_nonnegative_float_text`; selectors `_AFTER_CATEGORY_OVERRIDE_SELECTORS` |
| Negative values rejected | ✅ | Test `test_distill_prompt_kd_weight_rejects_negative_weight`; CLI parse raises `ArgumentTypeError` |
| Values `>= 1` allowed (e.g. `2.0`) | ✅ | Test `test_distill_prompt_kd_weight_accepts_value_above_one`; no upper bound in parser |
| Default resolves to `0.0` | ✅ | Test `test_distill_prompt_kd_weight_defaults_to_zero` |
| After-category override parsing | ✅ | Test `test_distill_prompt_kd_weight_resolves_after_category_overrides` (`default=0.05,after:q_proj=0.1`) |
| `NormalizedCatArgs.distill_prompt_kd_weight` near other distill loss params | ✅ | Placed after `distill_pre_mlp_hidden_loss_weight` |
| `ResolvedDistillRuntimeConfig.prompt_kd_weight: float` | ✅ | Field added and populated in `resolve_distill_runtime_config()` |
| `distill_tables` includes new table (invalid category keys rejected) | ✅ | `cat_train_pipeline.py` adds `(cat_args.distill_prompt_kd_weight, "--distill_prompt_kd_weight")`; manual check confirms `validate_category_keys` rejects unknown `after:` keys |
| `_ResolvedDistillStageConfig.prompt_kd_weight` + `_resolve_distill_stage_config()` passthrough | ✅ | Stage config field and `float(runtime_cfg.prompt_kd_weight)` assignment |
| LoRA log prints resolved float, not raw OverrideTable | ✅ | `_log_lora_stage_start()` logs `float(cfg.prompt_kd_weight)` |
| `_build_lora_trainer()` passes `prompt_kd_weight` to `CustomSFTTrainer` | ✅ | `prompt_kd_weight=float(cfg.prompt_kd_weight)` in trainer kwargs |
| `CustomSFTTrainer` stub init only (no `compute_loss`) | ✅ | `lora_training.py` stores value and rejects `<0`; no mask/KD wiring (Task 5) |
| Tests: default / override / negative / `2.0` | ✅ | `CatDistillPromptKdWeightArgsTest` (4 cases) |
| Trainer-selection fixture updated | ✅ | `CatDistillTrainerSelectionTest` asserts `prompt_kd_weight` kwarg passthrough |
| `pytest tests/test_cat_eval_adapter_match.py -q` | ✅ | **23 passed** (re-run 2026-08-10, `bitvae`) |
| No git commit | ✅ | Report confirms none |

---

## Findings

No findings.

Reviewed end-to-end chain:

```text
CLI (--distill_prompt_kd_weight)
  → NormalizedCatArgs.distill_prompt_kd_weight (OverrideTable)
  → resolve_distill_runtime_config(after_category) → prompt_kd_weight: float
  → _ResolvedDistillStageConfig.prompt_kd_weight
  → _log_lora_stage_start (resolved float)
  → CustomSFTTrainer(prompt_kd_weight=...)  [store + validate only]
```

Implementation mirrors existing after-category distill float params (`distill_hidden_loss_weight`, `distill_pre_mlp_hidden_loss_weight`) and stays within Task 4 scope.

---

## Quality Notes (non-blocking)

1. **Task 4 / Task 5 boundary:** `CustomSFTTrainer.__init__` stub landed in Task 4 (required for `_build_lora_trainer` kwarg). Task 5 should only add `compute_loss` mask wiring; avoid duplicating init work.

2. **`use_custom_trainer` gate unchanged:** `_build_lora_trainer()` / `cat_after_category_distill.py` still select `CustomSFTTrainer` only when `loss_type ∉ {sft, none, ""}` or hidden losses `> 0`. `prompt_kd_weight > 0` alone does not force `CustomSFTTrainer`. Acceptable for Task 4: KD loss types already select `CustomSFTTrainer`, and plan Task 5 states SFT/origin branch stays unchanged (pure `loss_type=sft` + prompt weight would be a no-op even if routed).

3. **Test gaps (optional hardening, not spec failures):**
   - No dedicated test that unknown `after:<cat>` keys in `--distill_prompt_kd_weight` fail validation (covered indirectly via `distill_tables` + existing `validate_category_keys`).
   - No test for trainer-level `ValueError` on negative `prompt_kd_weight` (CLI rejection is tested; trainer check is defense-in-depth).

4. **Out of scope (later tasks):** shell script defaults (Task 8), docs (Task 9), `compute_loss` integration (Task 5).

---

## Files Reviewed

| File | Role |
|------|------|
| `train_utils/cat_train_args.py` | CLI, spec, normalize, resolve |
| `train_utils/cat_train_pipeline.py` | `distill_tables` validation |
| `train_utils/lora_utils.py` | stage config, log, trainer build |
| `train_utils/lora_training.py` | `CustomSFTTrainer` stub init |
| `tests/test_cat_eval_adapter_match.py` | arg + trainer passthrough tests |

**Diff size:** 5 files, +67 / −1 lines (matches review package).

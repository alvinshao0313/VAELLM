> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-2-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Review: Strict MMLU Evaluation Wrapper

**Reviewer scope:** Task-scoped gate only (spec compliance + implementation quality)  
**Artifacts reviewed:** `task-2-brief.md`, `task-2-report.md`, `task-2-review-package.diff`  
**Date:** 2026-08-19

---

## Spec Verdict: ✅

Implementation meets all Task 2 brief requirements.

### Scope & constraints

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Changes only under `experiments/down_layer_sensitivity/` | ✅ | Diff adds `mmlu_eval.py` and `tests/test_mmlu_eval.py` only |
| Do not modify `train_utils/eval_utils.py` | ✅ | Import-only reuse of `run_lm_eval`; no edits in diff |
| Wrap `run_lm_eval`; never call `simple_evaluate` directly | ✅ | `evaluate_mmlu` calls `run_lm_eval` inside `torch.no_grad()`; no `simple_evaluate` in module or tests |

### Public API

| Symbol | Status | Notes |
|--------|--------|-------|
| `build_tokenizer(checkpoint_dir, access_token=None)` | ✅ | Present; delegates to `AutoTokenizer.from_pretrained` with required kwargs |
| `evaluate_mmlu(model, tokenizer, checkpoint_dir, *, lm_limit)` | ✅ | Keyword-only `lm_limit`; default `None` for formal runs |
| `extract_subject_metrics(lm_result)` | ✅ | Public; deterministic subject extraction |

### Step 1 — Tokenizer

- Loads from `checkpoint_dir` (not base model): ✅
- `use_fast=False`, `trust_remote_code=True`, `token=access_token`: ✅
- Worker-level “build once per worker” is a Task 3 concern; this task correctly exposes a single factory function: ✅

### Step 2 — Namespace & `run_lm_eval` wrapper

All ten `argparse.Namespace` fields match the brief exactly:

```python
tasks="mmlu"
num_fewshot=0
batch_size="auto"
lm_limit=lm_limit          # caller-controlled
model_path=checkpoint_dir
eval_log_dir=None
eval_run_ts=None
mmlu_debug_samples=0
mmlu_debug_log_dir=None
mmlu_debug_run_ts=None
```

Invocation pattern matches brief: `with torch.no_grad(): result = run_lm_eval(model, tokenizer, lm_args)`.

### Step 3 — Aggregate validation & return shape

| Check | Status |
|-------|--------|
| `result["task_metrics"]["mmlu"]` must exist | ✅ raises `ValueError` if missing |
| Must be finite float in `[0, 1]` | ✅ type, finiteness, and range checks |
| `result["task_metric_keys"]["mmlu"]` must exist | ✅ |
| `raw_results` must be dict | ✅ checked before aggregate validation |
| Returns `accuracy`, `metric_key`, `raw_results`, `n_samples_total` | ✅ |

### Step 4 — Subject metrics

| Check | Status |
|-------|--------|
| Iterate sorted `raw_results` keys matching `mmlu_*` | ✅ `sorted(...)` + `startswith("mmlu_")` |
| Metric priority: `acc_norm,none` → `acc,none` → `acc_norm` → `acc` | ✅ `_SUBJECT_METRIC_KEYS` tuple order |
| Row fields: `subject_name`, `metric_key`, `accuracy`, `samples` | ✅ |
| `n_samples_total = sum(int(row["samples"]) for row in subject_metrics)` | ✅ never uses top-level `result["n_samples"]` |
| No hard-coded 57 subject names | ✅ |

### Step 5 — Unit tests (required six cases)

| # | Required case | Test function | Status |
|---|---------------|---------------|--------|
| 1 | `build_tokenizer("/ckpt")` kwargs | `test_build_tokenizer_calls_auto_tokenizer_from_checkpoint` | ✅ |
| 2 | Metric priority `acc_norm,none` over `acc,none` | `test_subject_metric_priority_prefers_acc_norm_none` | ✅ |
| 3 | Subject rows sorted by `subject_name` | `test_subject_rows_are_sorted_by_subject_name` | ✅ |
| 4 | `n_samples_total == sum(subject.samples)` | `test_n_samples_total_equals_sum_of_subject_samples` | ✅ |
| 5 | Non-finite / missing aggregate raises `ValueError` | `test_non_finite_or_missing_aggregate_mmlu_accuracy_raises` (4 parametrized cases) | ✅ |
| 6 | Fixed MMLU args + `lm_limit` passed to `run_lm_eval` | `test_evaluate_mmlu_passes_fixed_lm_eval_args` | ✅ all 10 Namespace fields asserted |

- Tests mock only `AutoTokenizer` / `run_lm_eval` on the isolated module: ✅
- No `simple_evaluate` in tests: ✅
- Implementer report: `pytest -q experiments/down_layer_sensitivity/tests/test_mmlu_eval.py` → 9 passed (6 functions, 1 parametrized with 4 cases). Consistent with diff; not re-run per review instructions.

### Compatibility with `run_lm_eval` return contract

Cross-checked against `train_utils/eval_utils.py`: `run_lm_eval` returns `task_metrics`, `task_metric_keys`, and `raw_results` (from `results_dict`) in the shape the wrapper expects. Field names align with repository conventions.

---

## Quality Assessment

### Strengths

1. **Focused scope** — Single-purpose module; no unrelated refactors or exports.
2. **Clear structure** — `_pick_subject_metric`, `_validate_mmlu_aggregate`, and public functions are separated logically.
3. **Strict validation** — Aggregate and per-subject paths fail loudly with descriptive `ValueError` messages; aligns with downstream cross-job consistency needs.
4. **Test fidelity** — Required cases map 1:1 to brief; parametrized failure cases cover missing key, NaN, out-of-range, and missing `task_metric_keys["mmlu"]`.
5. **Repository alignment** — Reuses existing MMLU path via `run_lm_eval` rather than duplicating lm-eval setup.

### Non-blocking observations (do not block approval)

1. **Missing `samples` defaults to 0** — `int(task_result.get("samples", 0) or 0)` could undercount `n_samples_total` if lm-eval ever omits `samples`. Acceptable for Task 2; real `run_lm_eval` output should populate the field. Task 3 integration will surface any mismatch.
2. **Per-subject strictness** — Subjects with no finite metric in the priority list raise `ValueError`. Brief is silent on skip-vs-fail; strict choice is reasonable and documented in the implementer report.
3. **Redundant `torch.no_grad()`** — `run_lm_eval` already runs under `no_grad`; outer wrapper duplicates per brief requirement. Harmless.
4. **Error message granularity** — Non-finite aggregate errors share the same message as out-of-range errors (`"must be a finite float in [0, 1]"`). Minor UX nit only.
5. **Optional future tests (not in brief)** — `lm_limit=None` default path, `torch.no_grad` context, `extract_subject_metrics` failure modes. Not required for this task gate.

### Risks deferred (expected)

- No real-checkpoint / integration smoke — explicitly deferred to Task 3 per brief Step 5.
- `__init__.py` not exporting symbols — brief does not require; direct import is fine.

---

## Findings Summary

| ID | Severity | Finding | Action |
|----|----------|---------|--------|
| — | — | No spec violations or blocking quality issues identified | None |

---

## Task Quality Verdict: **Approved**

Implementation is spec-complete, test-covered per brief, and ready for Task 3 worker integration. No changes requested for Task 2 gate.

> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-2-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Report: Strict MMLU Evaluation Wrapper

## Status

DONE

## What Was Implemented

Created `experiments/down_layer_sensitivity/mmlu_eval.py` with the following public API:

| Symbol | Purpose |
|--------|---------|
| `build_tokenizer(checkpoint_dir, access_token=None)` | Load tokenizer once per worker from checkpoint dir via `AutoTokenizer.from_pretrained(..., use_fast=False, trust_remote_code=True, token=access_token)` |
| `evaluate_mmlu(model, tokenizer, checkpoint_dir, *, lm_limit)` | Wrap `train_utils.eval_utils.run_lm_eval` with fixed MMLU args; validate aggregate; return primary fields |
| `extract_subject_metrics(lm_result)` | Deterministic per-subject extraction from `raw_results` keys matching `mmlu_*` |

### Wrapper contract

- `argparse.Namespace` fields fixed per brief: `tasks="mmlu"`, `num_fewshot=0`, `batch_size="auto"`, `lm_limit`, `model_path=checkpoint_dir`, debug/log fields all `None`/`0`
- Calls only `run_lm_eval(model, tokenizer, lm_args)` inside `torch.no_grad()` — no direct `simple_evaluate`
- Aggregate validation: `task_metrics["mmlu"]` must exist, be finite, in `[0, 1]`; `task_metric_keys["mmlu"]` must exist; `raw_results` must be dict
- Subject metric priority: `acc_norm,none` → `acc,none` → `acc_norm` → `acc`
- Subject rows sorted by `subject_name` (sorted `raw_results` keys filtered to `mmlu_*`)
- `n_samples_total = sum(int(row["samples"]) for row in subject_metrics)` — never uses top-level `result["n_samples"]`

### Return shape of `evaluate_mmlu`

```python
{
    "accuracy": float,
    "metric_key": str,
    "raw_results": dict,
    "n_samples_total": int,
}
```

## Files Changed

| File | Action |
|------|--------|
| `experiments/down_layer_sensitivity/mmlu_eval.py` | Created — MMLU wrapper |
| `experiments/down_layer_sensitivity/tests/test_mmlu_eval.py` | Created — unit tests |

No production code modified (`eval_utils.py` untouched).

## Tests and Results

Environment:

```text
which python -> /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V      -> Python 3.11.13
```

Command:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_mmlu_eval.py
```

Result: **9 passed in 6.09s**

### Required test cases (6 from brief — all present)

1. `test_build_tokenizer_calls_auto_tokenizer_from_checkpoint`
2. `test_subject_metric_priority_prefers_acc_norm_none`
3. `test_subject_rows_are_sorted_by_subject_name`
4. `test_n_samples_total_equals_sum_of_subject_samples`
5. `test_non_finite_or_missing_aggregate_mmlu_accuracy_raises` (parametrized: missing mmlu, NaN, out-of-range, missing metric key)
6. `test_evaluate_mmlu_passes_fixed_lm_eval_args`

All tests mock `run_lm_eval` / `AutoTokenizer` only — no real MMLU or `simple_evaluate` invocation.

## Self-Review Against Brief

| Requirement | Met? | Notes |
|-------------|------|-------|
| Code only under `experiments/down_layer_sensitivity/` | Yes | |
| No `eval_utils.py` modification | Yes | Import-only reuse |
| Wrap `run_lm_eval`, not `simple_evaluate` | Yes | |
| Tokenizer from checkpoint_dir with exact kwargs | Yes | |
| Fixed MMLU Namespace fields | Yes | All 10 fields set per brief |
| Aggregate validation + primary return fields | Yes | |
| Subject extraction with metric priority + sorted keys | Yes | |
| `n_samples_total` from subject samples sum | Yes | |
| 6 unit-test cases | Yes | |
| pytest in bitvae | Yes | All pass |
| No git commit | Yes | |

## Concerns

1. **Subjects without finite metrics raise `ValueError`** — Brief does not specify skip-vs-fail for individual subjects; strict raise chosen so formal runs cannot silently drop subjects. Real MMLU via `run_lm_eval` should always populate metrics.
2. **No integration test with real checkpoint** — Deferred to Task 3 smoke/formal; unit tests fully mock the eval path per brief Step 5.
3. **`__init__.py` not updated** — Brief does not require exporting `mmlu_eval` symbols yet; Task 3 worker will import directly from `mmlu_eval`.

## Commits

None (per project rules).

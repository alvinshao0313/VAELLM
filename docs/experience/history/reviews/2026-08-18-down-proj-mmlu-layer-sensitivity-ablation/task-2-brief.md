> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2: Add a Strict MMLU Evaluation Wrapper Without Modifying `eval_utils.py`

**Files:**
- Create: `experiments/down_layer_sensitivity/mmlu_eval.py`
- Create: `experiments/down_layer_sensitivity/tests/test_mmlu_eval.py`.

**Interfaces:**
- Consumes: `train_utils.eval_utils.run_lm_eval`, `transformers.AutoTokenizer`.
- Produces:
  - `build_tokenizer(checkpoint_dir, access_token=None)`
  - `evaluate_mmlu(model, tokenizer, checkpoint_dir, *, lm_limit)`
  - `extract_subject_metrics(lm_result)`

- [ ] **Step 1: Build tokenizer exactly once per worker**

Tokenizer **固定从本次 final checkpoint 目录加载**，不允许 Cursor 自行改为 base model tokenizer：

```python
AutoTokenizer.from_pretrained(
    checkpoint_dir,
    use_fast=False,
    trust_remote_code=True,
    token=access_token,
)
```

`checkpoint_dir` 固定就是 `.result/catlora/res0-bf16-protect-channel-vae/final_model`（由 `run.py --checkpoint_dir` 传入）。每个 worker 只创建一次 tokenizer，所有 job 复用；禁止每个 job 重新加载 tokenizer。

- [ ] **Step 2: Wrap existing `run_lm_eval()`**

Create a local `argparse.Namespace` with exactly:

```text
tasks="mmlu"
num_fewshot=0
batch_size="auto"
lm_limit=<None for formal; 2 for smoke>
model_path=<checkpoint_dir>
eval_log_dir=None
eval_run_ts=None
mmlu_debug_samples=0
mmlu_debug_log_dir=None
mmlu_debug_run_ts=None
```

Call only:

```python
with torch.no_grad():
    result = run_lm_eval(model, tokenizer, lm_args)
```

Do not reproduce `lm_eval.evaluator.simple_evaluate()` directly; consistency with the repository's current MMLU path is more important than shaving wrapper construction overhead.

- [ ] **Step 3: Validate primary MMLU result**

After `run_lm_eval()`:

```text
result["task_metrics"]["mmlu"] must exist
must be finite float in [0,1]
result["task_metric_keys"]["mmlu"] must exist
raw_results must be dict
```

Return primary fields:

```text
accuracy
metric_key
raw_results
n_samples_total
```

- [ ] **Step 4: Extract deterministic subject metrics**

`extract_subject_metrics()` iterates sorted keys matching `mmlu_*` from `raw_results`.

For each subject choose first finite metric from:

```python
("acc_norm,none", "acc,none", "acc_norm", "acc")
```

Store:

```text
subject_name
metric_key
accuracy
samples
```

`n_samples_total` 的定义必须唯一化为：

```python
n_samples_total = sum(int(row["samples"]) for row in subject_metrics)
```

禁止直接把 `run_lm_eval()` 顶层 `result["n_samples"]` 当成单个整数使用，因为该字段是 lm-eval 返回的结构化对象。所有跨 job 一致性检查都使用 `n_samples_total` + 每个 subject 的 `samples`。

Do not hard-code individual 57 subject names. Formal aggregation later checks all jobs have the same subject-key set as the baseline.

- [ ] **Step 5: Unit-test the wrapper contract without launching real MMLU**

In `test_mmlu_eval.py`, use `unittest.mock`/pytest monkeypatch against the isolated module only. Tests must cover exactly:

```text
1. build_tokenizer("/ckpt") calls AutoTokenizer.from_pretrained("/ckpt", use_fast=False, trust_remote_code=True, token=None)
2. subject metric priority chooses acc_norm,none before acc,none when both exist
3. subject rows are sorted by subject_name
4. n_samples_total equals sum(subject.samples)
5. non-finite or missing aggregate MMLU accuracy raises ValueError
6. evaluate_mmlu passes tasks=mmlu, num_fewshot=0, batch_size=auto and the requested lm_limit to run_lm_eval
```

Do not invoke `lm_eval.evaluator.simple_evaluate()` in unit tests.

Run:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_mmlu_eval.py
```

---


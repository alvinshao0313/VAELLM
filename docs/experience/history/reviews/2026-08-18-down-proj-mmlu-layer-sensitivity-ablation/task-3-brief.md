> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3: Implement the Per-GPU Worker and Make Every Job Self-Describing

**Files:**
- Create: `experiments/down_layer_sensitivity/worker.py`
- Create: `experiments/down_layer_sensitivity/tests/test_job_manifest.py`

**Interfaces:**
- Consumes: `core.load_worker_model`, `core.set_down_restore_set`, `core.assert_down_restore_set`, `mmlu_eval.evaluate_mmlu`.
- Produces: one JSON result per job under `phase*/jobs/`.

## Exact job schema

Each manifest job is:

```json
{
  "job_id": "restore_L17",
  "restore_layers": [17],
  "mode": "formal",
  "lm_limit": null
}
```

Baseline probe（示例为 worker 0；其它 worker 只改两位 worker id）：

```json
{
  "job_id": "compressed_baseline_worker00",
  "restore_layers": [],
  "mode": "formal",
  "lm_limit": null
}
```

Worker 0 repeat 的 job id 固定为 `compressed_baseline_worker00_repeat`，restore set 同样为空。

All-down-original:

```json
{
  "job_id": "all_down_original",
  "restore_layers": [0, 1, 2, ..., 35],
  "mode": "formal",
  "lm_limit": null
}
```

### Worker CLI is fixed

`worker.py` 只接受下面这些参数，禁止 Cursor 自行设计第二套启动协议：

```text
--checkpoint_dir
--manifest_path
--jobs_dir
--worker_meta_path
--worker_id
--physical_gpu_id
```

manifest 顶层结构固定为：

```json
{
  "worker_id": 0,
  "physical_gpu_id": "0",
  "mode": "formal",
  "write_weight_metrics": true,
  "jobs": []
}
```

`worker_id` / `physical_gpu_id` 必须同时出现在 CLI 与 manifest，worker 启动时要求两处值完全一致；这样输出 artifact 可追踪且不允许调用方和 manifest 各说一套。

- [ ] **Step 1: Implement manifest validation before model loading**

Worker must reject:

```text
- duplicate job_id
- restore layer outside 0..35
- duplicate layer in one restore list
- mode not in {smoke, formal}
- formal job with lm_limit not None
- smoke job with lm_limit != 2
```

- [ ] **Step 2: Set the exact inference seeds, then record environment once**

Worker 解析完 CLI/manifest、但在加载 tokenizer/model 之前，必须执行且只执行一次：

```python
random.seed(31)
np.random.seed(31)
torch.manual_seed(31)
torch.cuda.manual_seed_all(31)
```

禁止额外启用 `torch.use_deterministic_algorithms(True)`、修改 cuDNN/CUDA kernel 选择或关闭现有 TF32/bfloat16 行为；本实验要复用当前评测执行路径，而不是改变推理数值路径。

At startup record:

```text
worker_id
physical_gpu_id
logical_device="cuda:0"
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
seed=31
python version
torch version
transformers version
lm_eval version if available
checkpoint_dir
base_model_path_from_checkpoint_meta
```

Do not hash the multi-GB checkpoint.

- [ ] **Step 3: Load model exactly once per worker**

Worker sequence:

```text
load worker model
build tokenizer once
if write_weight_metrics=true: compute down weight metrics once from prewarmed cache
write worker metadata
execute manifest jobs in order
```

只有 worker 0 的 manifest 设置 `write_weight_metrics=true`；**其它 worker 不计算也不写 weight metrics**，避免在每张 GPU 上重复把 36 个大 down weight 转成 FP32 做同一份统计。worker 0 写 canonical `weight_metrics_worker.json`。

- [ ] **Step 4: Enforce state reset around every job**

Each job execution must be exactly:

```python
reset_all_vae_to_compressed(model)
assert_down_restore_set(down_layers, set())
set_down_restore_set(down_layers, restore_set)
assert_down_restore_set(down_layers, restore_set)
result = evaluate_mmlu(...)
reset_all_vae_to_compressed(model)
assert_down_restore_set(down_layers, set())
```

Use `try/finally` only for the final reset; do not swallow the evaluation exception.

- [ ] **Step 5: Write exact per-job JSON**

Each result contains:

```text
job_id
mode
restore_layers
accuracy
accuracy_percent
metric_key
subject_metrics
n_samples_total
runtime_sec
worker_id
physical_gpu_id
device_name
prewarm_stats
```

Do not dump full model state or raw logits.

- [ ] **Step 6: Fail worker on first failed job**

If any job raises, log the traceback, exit non-zero, and do not continue later jobs. Coordinator must treat any non-zero worker exit as formal-run failure.

- [ ] **Step 7: Unit-test manifest validation**

Run:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py
```

Cover all invalid schemas listed above plus valid baseline/single/all-original manifests.

---


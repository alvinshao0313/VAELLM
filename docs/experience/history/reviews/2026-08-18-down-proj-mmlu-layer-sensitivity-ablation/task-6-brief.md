> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-6-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6: Build and Run Phase-2 Cumulative Validation

**Files:**
- Modify: `experiments/down_layer_sensitivity/run.py`
- Modify: `experiments/down_layer_sensitivity/summarize.py`
- Extend: `experiments/down_layer_sensitivity/tests/test_job_manifest.py`
- Extend: `experiments/down_layer_sensitivity/tests/test_summarize.py`

**Interfaces:**
- Consumes the in-memory phase-1 ranked layer list returned by `summarize_phase1()`.
- Must implement exactly:

```python
def build_phase2_manifests(
    *,
    selected_gpus: list[str],
    ranked_layers: list[int],
) -> list[dict]:

def summarize_final(*, run_dir: str, selected_gpus: list[str]) -> None:
```

- Produces phase-2 worker manifests, `cumulative_results.csv`, `final_summary.json`, plots and `report.md`.

- [ ] **Step 1: Build restore sets from the actual phase-1 ranking**

Let:

```python
ranked = ranked_layers  # exact list returned by summarize_phase1()
```

Require `sorted(ranked) == list(range(36))` and `len(ranked)==36`, then build exactly:

```python
top1 = ranked[:1]   # reuse phase-1 restore_Lxx result only
top2 = ranked[:2]
top4 = ranked[:4]
top8 = ranked[:8]
top12 = ranked[:12]
```

Top-1 is not scheduled again.

- [ ] **Step 2: Build exactly five deterministic Random-8 controls**

Fixed seeds are **only**：

```python
RANDOM_CONTROL_SEEDS = (31, 32, 33, 34, 35)
```

For each seed `s` use exactly:

```python
rng = random.Random(s)
restore_layers = sorted(rng.sample(list(range(36)), 8))
job_id = f"random8_seed{s}"
```

Do not reject or redraw a seed merely because two controls overlap heavily or one random set happens to include Top-8 layers；真实随机对照就按固定种子保留。

- [ ] **Step 3: Phase-2 scientific jobs are exactly nine**

Scientific jobs fixed order:

```text
top2
top4
top8
top12
random8_seed31
random8_seed32
random8_seed33
random8_seed34
random8_seed35
```

Let:

```python
W2 = min(len(selected_gpus), 9)
phase2_gpus = selected_gpus[:W2]
```

禁止为了 phase 2 自行换卡或从 selected_gpus 中重新挑卡。

- [ ] **Step 4: Every phase-2 worker also starts with a baseline probe**

Phase 2 每个参与 worker 的第一项必须是：

```text
compressed_baseline_workerXX
```

Worker 0 第二项仍固定为：

```text
compressed_baseline_worker00_repeat
```

因此 phase-2 总 job 数固定为：

```text
9 + W2 + 1
```

Scientific jobs 再按 Task 4 同一个 greedy 规则分配：当前总 job 数最少，tie 选较小 worker_id。禁止为 phase 2 实现另一种调度器。

- [ ] **Step 5: Phase-2 manifest path and worker launch reuse Task 4 exactly**

Paths:

```text
phase2/manifests/worker_00.json
phase2/manifests/worker_01.json
...
```

Phase 2 **必须复用同一个 `launch_phase_workers()`**。不要复制 subprocess launch code，也不要修改 `worker.py` 的 CLI。

Phase 2 所有 manifest 的 `write_weight_metrics=false`；weight metrics 只允许 phase-1 worker0 计算一次。

- [ ] **Step 6: Validate phase-2 baseline probes against phase-1 canonical baseline**

对 phase-2 每个 worker baseline，以及 phase2 worker0 repeat，要求与 phase-1 `compressed_baseline_worker00`：

```text
accuracy difference <= 1e-12
same subject-name set
same per-subject sample count
same n_samples_total
for every subject: accuracy difference <= 1e-12
same device_name as formal phase-1 homogeneous device name
```

任一失败则 `summarize_final()` 非零失败，禁止生成 final report。

- [ ] **Step 7: Validate all phase-2 scientific jobs use the same evaluation population**

Every phase-2 scientific job must match the phase-1 canonical baseline on:

```text
subject-name set
per-subject sample count
n_samples_total
GPU device name
formal lm_limit=None
```

- [ ] **Step 8: Calculate Top-K cumulative recovery**

For K in `{1,2,4,8,12}`:

```python
recovery_fraction = (A_topK - A_compressed) / (A_all_down_original - A_compressed)
```

For Top-1, use the phase-1 result of rank-1 layer. Do not rerun it and do not recompute its ranking.

- [ ] **Step 9: Calculate five Random-8 controls and aggregate mean/std**

For each seed compute:

```python
control_recovery_fraction_s = (
    A_random8_s - A_compressed
) / (A_all_down_original - A_compressed)
```

Then use NumPy with population standard deviation `ddof=0`：

```python
random8_accuracy_mean = np.mean(random8_accuracies)
random8_accuracy_std = np.std(random8_accuracies, ddof=0)
random8_recovery_mean = np.mean(random8_recoveries)
random8_recovery_std = np.std(random8_recoveries, ddof=0)
top8_minus_random8_mean_recovery = top8_recovery - random8_recovery_mean
```

Do not use standard error, bootstrap CI or another random baseline statistic in this task.

- [ ] **Step 10: Do not assume monotonicity**

Do not enforce `A_top2 <= A_top4 <= ...` and do not smooth curves. If adding more restored layers lowers MMLU, preserve the observed result because it is evidence of interaction.

- [ ] **Step 11: Write `cumulative_results.csv`**

Columns exactly:

```text
configuration
num_restored_layers
restore_layers
mmlu_accuracy
mmlu_accuracy_percent
delta_from_compressed_pp
recovery_fraction
```

Rows exactly in this order:

```text
top1
top2
top4
top8
top12
random8_seed31
random8_seed32
random8_seed33
random8_seed34
random8_seed35
all_down_original
```

Do not add an artificial `random8_mean` row to this CSV，因为 mean 没有唯一 restore set；aggregate statistics 只写入 `final_summary.json` / report / plot。

---


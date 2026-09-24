> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-5-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5: Aggregate Phase-1 With Hard Validity Gates Before Ranking

**Files:**
- Create: `experiments/down_layer_sensitivity/summarize.py`
- Create: `experiments/down_layer_sensitivity/tests/test_summarize.py`

**Interfaces:**
- Produces `phase1_summary.json`, `single_layer_sensitivity.csv`, `weight_metrics.csv`, and ranked layer list for phase 2.
- 必须实现并由 `run.py` 直接调用唯一入口：

```python
def summarize_phase1(*, run_dir: str, selected_gpus: list[str]) -> list[int]:
```

返回值就是按 `delta_mmlu_pp` 排好的 36 个 `layer_idx`，供 `build_phase2_manifests()` 使用；禁止通过重新读取 CSV 再推导第二套 ranking。

- [ ] **Step 1: Require exact phase-1 job inventory**

设 `W=len(selected_gpus)`。Formal aggregation 必须找到恰好：

```text
compressed_baseline_worker00 ... compressed_baseline_worker{W-1:02d}
compressed_baseline_worker00_repeat
all_down_original
restore_L00 ... restore_L35
```

总数必须为 `38+W`，No missing or duplicate job IDs.

- [ ] **Step 2: Validate same-worker and cross-GPU baseline determinism**

Canonical baseline 固定为 `compressed_baseline_worker00`，不允许根据结果选择另一张卡作为 canonical。

先验证 worker 0 repeat：

```text
abs(A_worker00 - A_worker00_repeat) <= 1e-12
same n_samples_total
same subject-name set
same per-subject sample count
for every subject: abs(subject_acc_worker00 - subject_acc_repeat) <= 1e-12
```

然后对每个 `worker i=1..W-1` 的 baseline probe，逐项要求与 canonical baseline 完全满足同样条件：

```text
abs(A_workeri - A_worker00) <= 1e-12
same n_samples_total
same subject-name set
same per-subject sample count
for every subject: abs(subject_acc_workeri - subject_acc_worker00) <= 1e-12
```

任一项失败，formal aggregation 立即失败，不生成 36 层 ranking。禁止通过平均多个 GPU baseline 来掩盖差异。

- [ ] **Step 3: Validate evaluation population consistency**

Use canonical baseline subject keys and `n_samples_total` as reference.

For every phase-1 job require:

```text
same subject-name set
same per-subject sample count
same n_samples_total
```

This prevents comparing configurations evaluated on different MMLU subsets.

- [ ] **Step 4: Validate homogeneous GPU type for formal multi-GPU run**

Collect `device_name` from all phase-1 jobs. If more than one distinct device name is present, formal aggregation fails with explicit message to rerun on a homogeneous GPU set.

Do not try to normalize results across GPU architectures.

- [ ] **Step 5: Validate all-down-original intervention direction**

Require:

```text
A_all_down_original > A_compressed
```

If false, write diagnostic summary but stop before sensitivity ranking and phase 2.

Also record, but do not hard-fail on:

```text
current baseline vs historical 41.71%
current all-down-original vs historical pre-down 51.99%
```

- [ ] **Step 6: Compute single-layer primary metrics**

For each layer i:

```python
delta_pp = 100.0 * (A_i - A_compressed)
recovery_fraction = (A_i - A_compressed) / (A_all_down_original - A_compressed)
```

Merge canonical weight metrics by `layer_idx`.

CSV columns in exact order:

```text
rank
layer_idx
module_name
mmlu_accuracy
mmlu_accuracy_percent
delta_mmlu_pp
single_recovery_fraction
weight_mse
weight_nmse
relative_fro_error
original_rms
error_rms
subjects_improved
subjects_worsened
subjects_unchanged
median_subject_delta_pp
max_subject_gain_pp
max_subject_drop_pp
```

- [ ] **Step 7: Compute subject-level diagnostics**

For each subject:

```python
delta_subject_pp = 100.0 * (subject_acc_i - subject_acc_baseline)
```

Use tolerance `1e-12` only to classify exact numerical equality:

```text
> +1e-12 -> improved
< -1e-12 -> worsened
otherwise unchanged
```

`median_subject_delta_pp` uses NumPy median.

- [ ] **Step 8: Rank exactly by task sensitivity**

```python
sorted(rows, key=lambda r: (-r["delta_mmlu_pp"], r["layer_idx"]))
```

Assign rank 1..36 after sorting.

Do not rerank by NMSE or subject count.

- [ ] **Step 9: Compute NMSE-vs-sensitivity Spearman without SciPy**

No new dependency. Implement rank correlation locally in `summarize.py` using average ranks for ties and NumPy Pearson correlation over the two rank vectors.

Record:

```text
spearman_weight_nmse_vs_delta_mmlu
```

Unit-test with known monotonic, reverse and tied examples.

- [ ] **Step 10: Write phase-1 artifacts**

`phase1_summary.json` must contain exactly these top-level scientific fields（可以另外有 `status` / `run_id` 这类运行元数据，但不得改名或重复计算指标）：

```text
compressed_baseline
cross_gpu_baseline_probes
worker00_baseline_repeat
all_down_original
down_gap_pp
historical_reference
spearman_weight_nmse_vs_delta_mmlu
ranked_layers
```

其中 `compressed_baseline` 永远来自 `compressed_baseline_worker00`；`cross_gpu_baseline_probes` 保存每个 worker 的 baseline accuracy/device/sample count，不能只保存一个“passed=true”。

Write `single_layer_sensitivity.csv` and `weight_metrics.csv`.

- [ ] **Step 11: Unit-test aggregation math**

Synthetic test must verify exact calculations for:

```text
- delta pp
- recovery fraction
- rank order
- equal-delta tie by layer index
- subject improved/worsened counts
- baseline mismatch failure
- sample-count mismatch failure
- all-original <= baseline failure
- heterogeneous GPU failure
```

Run:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py
```

---


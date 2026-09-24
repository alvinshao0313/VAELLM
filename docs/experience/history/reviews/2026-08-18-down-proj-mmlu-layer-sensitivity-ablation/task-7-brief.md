> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-7-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7: Produce a Reproducible Report and Three Diagnostic Figures

**Files:**
- Modify: `experiments/down_layer_sensitivity/summarize.py`
- Create: `experiments/down_layer_sensitivity/README.md`

- [ ] **Step 1: Generate `layer_sensitivity.png`**

Plot:

```text
x = layer_idx 0..35
y = delta_mmlu_pp
```

Requirements:

- one point/bar per layer;
- horizontal y=0 reference line;
- annotate rank Top-8 with layer ID/rank;
- do not reorder x by sensitivity; x remains model depth.

This figure answers whether sensitive layers concentrate at shallow/middle/deep depth.

- [ ] **Step 2: Generate `nmse_vs_mmlu_sensitivity.png`**

Plot:

```text
x = weight_nmse
y = delta_mmlu_pp
```

Annotate Top-8 task-sensitive layers with layer ID.

Title/caption in report includes computed Spearman rho.

This figure is diagnostic only; do not draw a causal claim from correlation.

- [ ] **Step 3: Generate `cumulative_recovery.png`**

Main curve uses Top-K for K=`1,2,4,8,12,36` where K=36 is all-down-original.

At x=8, plot exactly one Random-8 control summary point:

```text
y = random8_recovery_mean
errorbar = ± random8_recovery_std
```

Do not draw five random lines and do not use a dual y-axis. Figure axes are fixed:

```text
x = restored layer count
y = cumulative recovery fraction
```

- [ ] **Step 4: Generate `final_summary.json` before generating the markdown report**

`final_summary.json` is the single machine-readable source of truth for final conclusions. It must contain exactly these scientific sections/fields:

```text
compressed_baseline
cross_gpu_baseline_probes
all_down_original
down_gap_pp
ranked_layers
spearman_weight_nmse_vs_delta_mmlu
topk
  top1
  top2
  top4
  top8
  top12
random8_controls
  seed31
  seed32
  seed33
  seed34
  seed35
random8_aggregate
  accuracy_mean
  accuracy_std
  recovery_mean
  recovery_std
  top8_minus_random8_mean_recovery
historical_reference
```

`ranked_layers` 必须直接复用 `phase1_summary.json` 的排序内容；`topk` / `random8_controls` 必须直接来自 `cumulative_results.csv` 对应配置结果。禁止在写 `final_summary.json` 时再次跑排序、再次抽随机层或重新计算另一套 MMLU 指标。

- [ ] **Step 5: Generate final report in Chinese from `final_summary.json` + CSV artifacts**

`report.md` sections fixed:

```text
1. 实验目标
2. 固定实验设置
3. 有效性检查
4. Down 压缩总损失
5. 36 层敏感度排名
6. Weight NMSE 与 MMLU 敏感度关系
7. Top-K 累积恢复
8. Random-8 五组对照
9. 结论与后续压缩建议
```

Report must explicitly state:

```text
- current compressed baseline
- all phase-1/phase-2 GPU baseline consistency status
- historical 41.71% reference difference
- all-down-original accuracy
- historical 51.99% reference difference
- total recoverable down gap pp
- Top-8 layer IDs
- Top-1/2/4/8/12 recovery fractions
- five Random-8 layer sets and their individual recovery fractions
- Random-8 mean/std recovery fraction
- Top-8 minus Random-8 mean recovery
- Spearman rho
```

Markdown 中的数字必须读取已生成的 JSON/CSV 字段；禁止在 report renderer 里另写一套公式重新算结果。

- [ ] **Step 6: Keep conclusions bounded to MMLU**

Allowed conclusion form:

```text
“在当前 final_model 与 0-shot full-MMLU 设置下，Lx/Ly/... 对 down VAE 压缩最敏感。”
```

Do not write:

```text
“这些层对所有任务都最敏感。”
```

If future work needs general sensitive layers, that requires ARC/MMLU-Pro/etc. cross-task confirmation and is outside this plan.

---


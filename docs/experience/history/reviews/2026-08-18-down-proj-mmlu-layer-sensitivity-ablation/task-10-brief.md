> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-10-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 10: Run the Formal Experiment and Enforce Completion Criteria

- [ ] **Step 1: Launch formal multi-GPU run**

Example for four GPUs:

```bash
GPUS=0,1,2,3 bash experiments/down_layer_sensitivity/scripts/run_formal.sh
```

Use more homogeneous GPUs by changing only `GPUS`, e.g. `GPUS=0,1,2,3,4,5,6,7`.

- [ ] **Step 2: Phase-1 must complete the exact worker-dependent inventory**

Let `W=len(selected_gpus)`. Before phase 2, verify exactly:

```text
W worker baseline probes
1 worker00 baseline repeat
1 all-down-original
36 single-layer restore
Total = 38 + W
```

All are full MMLU with `lm_limit=None`.

- [ ] **Step 3: Same-worker repeat + cross-GPU baselines are the determinism/comparability gate**

Formal run only continues if:

```text
compressed_baseline_worker00 == compressed_baseline_worker00_repeat
and every compressed_baseline_workerXX == compressed_baseline_worker00
```

Equality uses the fixed `1e-12` tolerance for aggregate and every subject accuracy, plus exact subject/sample population equality. Never average disagreeing GPU baselines.

This gate is more important than exactly reproducing historical `41.71%` because software versions may have changed; historical difference is reported, not force-corrected.

- [ ] **Step 4: Validate all-down-original restores a positive gap**

Require:

```text
A_all_down_original > A_compressed
```

Report whether it approaches historical pre-down `51.99%`, but do not force it to equal 51.99%.

- [ ] **Step 5: Confirm 36-layer ranking is complete**

`single_layer_sensitivity.csv` must contain exactly 36 data rows and rank values exactly 1..36.

- [ ] **Step 6: Run the fixed phase-2 cumulative/control phase**

Scientific jobs are exactly:

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

In addition, every participating phase-2 worker must run its own compressed baseline first and worker0 must repeat it once, exactly as specified in Task 6. Top1 is reused from phase 1.

- [ ] **Step 7: Confirm all final artifacts exist**

Required:

```text
phase1_summary.json
single_layer_sensitivity.csv
weight_metrics.csv
cumulative_results.csv
final_summary.json
report.md
plots/layer_sensitivity.png
plots/nmse_vs_mmlu_sensitivity.png
plots/cumulative_recovery.png
```

- [ ] **Step 8: Final report must answer the actual research question**

At minimum, Cursor's completion message must quote from artifacts:

```text
1. 当前 compressed full-MMLU canonical baseline 是多少？所有 phase-1/phase-2 GPU baseline 是否通过一致性门禁？
2. 36 个 down 全部恢复 original 后是多少？总 down gap 是多少 pp？
3. Top-8 最敏感 layer ID 是哪些？
4. 每个 Top-8 单独恢复分别带来多少 ΔMMLU pp？
5. Top-2/4/8/12 联合恢复分别拿回多少 down gap？
6. seed=31..35 五组 Random-8 各拿回多少？mean/std 是多少？Top-8 比 Random-8 mean 高多少 recovery？
7. weight NMSE 与 MMLU sensitivity 的 Spearman rho 是多少？
8. 高敏感 layer 是否集中在某个深度区间，还是分散？
```

- [ ] **Step 9: Do not automatically change compression policy**

This plan ends at experimentally identifying sensitive layers. Do **not** automatically modify VAE bit allocation, skip_layers, protected-channel counts, mixed-bit solver or retrain model.

The next design decision should be based on the measured ranking/cumulative curve, e.g. whether to:

```text
- keep a small number of down layers BF16,
- give sensitive down layers higher bit budget,
- increase protection only for sensitive layers,
- or redesign down-specific compression.
```

Those are follow-up tasks, not part of this experiment implementation.

---

## Final Acceptance Checklist

Implementation is complete only when all items below are true:

- [ ] All new runtime code is under `experiments/down_layer_sensitivity/`.
- [ ] No production training/compression/eval Python file was modified.
- [ ] Existing user modification to `scripts/catlora_simple.sh` was not touched.
- [ ] Formal checkpoint is the specified `final_model`.
- [ ] Exactly 36 `down_proj` VAELinear modules are discovered.
- [ ] All 36 have original weights and no down uses `always_use_original` at baseline.
- [ ] Non-down original-weight unload is attempted exactly once per module; legally protected originals may remain, and unload statistics are recorded.
- [ ] All non-down VAELinear modules remain on the compressed (`temporary=True`) path.
- [ ] Compressed cache prewarm succeeds once per worker with `failed=0`.
- [ ] Every job resets state before and after evaluation.
- [ ] Formal MMLU is 0-shot, full set, batch size auto, no HiF4 activation.
- [ ] Tokenizer is loaded from the same final checkpoint directory on every worker.
- [ ] Worker seeds are fixed to 31 using Python/NumPy/PyTorch/CUDA calls specified in the plan.
- [ ] Multi-GPU parallelism is one independent worker per GPU, not DDP.
- [ ] Worker00 repeat passes exact determinism checks in both formal phases.
- [ ] Every participating phase-1 and phase-2 GPU baseline matches phase-1 worker00 canonical baseline within the fixed tolerance.
- [ ] All formal jobs use identical MMLU subject/sample population and `n_samples_total` definition.
- [ ] All selected formal GPUs are homogeneous by device name.
- [ ] `all_down_original > compressed_baseline` before phase 2.
- [ ] 36 single-layer results are complete and ranked only by ΔMMLU.
- [ ] Weight NMSE is reported but does not influence primary rank.
- [ ] Top-1/2/4/8/12 cumulative recovery is reported.
- [ ] Random-8 seeds 31,32,33,34,35 are all reported, together with fixed `ddof=0` mean/std and Top-8-minus-random-mean recovery.
- [ ] `final_summary.json` is generated before `report.md` and is the machine-readable source of truth for the report.
- [ ] Three required plots and Chinese report are generated.
- [ ] Smoke results are never mixed into formal ranking.
- [ ] No checkpoint copies are generated for individual layers.
- [ ] No model retraining occurs.
- [ ] No automatic compression-policy change occurs after the experiment.

## Expected Scientific Interpretation

The experiment should allow exactly the following classes of conclusion:

```text
Case A: 少数 layer 有很大的 positive ΔMMLU，Top-4/8 可恢复大部分 down gap
=> down sensitivity 高度集中，后续最适合做 selective high precision / selective higher bit。

Case B: 36 层 ΔMMLU 都较小，但 Top-K cumulative 增益明显
=> 单层效应弱、layer interaction 强，不应只靠独立 sensitivity 做 bit allocation。

Case C: weight NMSE 与 ΔMMLU 高相关
=> 当前 down 难点较大程度来自特定层 reconstruction quality，可优先改这些层 VAE/bit budget。

Case D: weight NMSE 与 ΔMMLU 低相关
=> 单纯按 weight reconstruction error 找敏感层会误导，应以 task-aware sensitivity 为主。

Case E: Top-8 recovery 明显高于五组 Random-8 的 mean，且差值相对 random std 也有清晰优势
=> 排名确实捕获了结构性敏感层，而不只是“少压 8 层自然变好”；报告同时保留五组 individual control，不能只展示 mean。
```

所有结论限定在当前 Qwen3-8B final checkpoint 与 0-shot full-MMLU 设置内。

# Down Layer Sensitivity Ablation

本目录实现 down_proj VAE 压缩的 MMLU 层敏感度消融实验。

## 运行方式

在已激活 `bitvae` 环境的 shell 中执行：

```bash
conda activate bitvae
GPUS=0 bash experiments/down_layer_sensitivity/scripts/run_smoke.sh
GPUS=0,1,2,3 bash experiments/down_layer_sensitivity/scripts/run_formal.sh
```

`--gpus` 传入的是物理 GPU ID 列表，每个 ID 对应一个独立的 MMLU 评测 worker；这是多 job 并行调度，不是 DDP 单模型多卡训练。

## Prewarm 显存策略

实验包 prewarm（smoke / formal 共用）仍使用 `group_size=8`。预热期间模型主体与 36 个 down 的 `original_weight` 留在 CPU，仅按 batch 上 GPU 解码；每批 decoded cache 先暂存 CPU，全部解完后再 hoist cache 到 GPU，并把 down original 钉回 CPU。推理时只有当前 job 的 restore 集合才会把对应 down original 临时搬上 GPU，job 结束再卸回 CPU。

## 输出产物

正式 run 完成后，`summarize_final()` 会在 run 目录写入：

- `final_summary.json` — 机器可读最终结论
- `report.md` — 中文实验报告
- `plots/layer_sensitivity.png`
- `plots/nmse_vs_mmlu_sensitivity.png`
- `plots/cumulative_recovery.png`

报告与图表均从已生成的 JSON/CSV 读取数值，不在渲染阶段重新计算 MMLU 指标。

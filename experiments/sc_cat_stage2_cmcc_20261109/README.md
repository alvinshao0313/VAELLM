# SC-CAT二阶段独立准备

[唯一批次记录](../../docs/experience/records/sc_cat/qwen--qwen3-8b/2026-11-09_194001_s31_results.md)。已有完整SC-CAT研究基线和现有合法train来源的CPU核对已通过；本包目前只有准备入口，没有GPU训练入口。按规范先D／D＋P范围预选，再在同一完整父上比较A–E数据，通常六份训练。

调用者激活bitvae后，使用新的输出文件并以tmux或nohup后台执行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -B -u -m experiments.sc_cat_stage2_cmcc_20261109.prepare --output <new-preparation.json>
```

`prepare.py`核真实权重与已评artifact身份、252投影及保护拓扑，逐条回连原任务池的99842条知识题，核Web／Med原始train与card哈希。它不训练模型、不重跑评测、不把未定的优化／日程／预算填成默认值，也不复用K局部配置替代二阶段规范。

# 接续原SC-CAT批次的up250中断

**11月8日21:12当前入口：** up及末类双九项已完成，后续BF16重放在o导出写盘失败。用户确认重启后，当前使用`experiments.sc_cat_replay_restart_cmcc_20261108`及`repair_20261108/replay_restart/restart.json`，tmux `SC_CAT_replay_restart_20261108`。复用k完整边界和o125训练，只续未完成导出及后继；旧失败根和费用保留。以下up250入口是历史已完成段，不能重复启动。

仅处理`2026-11-05_184620_s31/live_gate_up_20261107/gpu0_3`已授权批次。原控制器因协议正文更新触发SHA检查退出；前六类和up两条完整基准直接复用，末类第三臂从实际250保存态继续原500日程。新版Top 2／恢复超参搜索不进入本批。

`frozen.py`只将授权中两份旧文档的“原路径＋原SHA”解析到字节相同的独立快照；原批准文件、新正文及85份原运行源码都不改，其他绑定仍严格检查。snapshot记录原失败state／budget／quota／exit和新增实现SHA。

`resume.py`使用已有原生v6加载器恢复完整mutable、Adam、scheduler、trainer、RNG及原Iterable数据跳过。真实续段逐批核原冻结输入哈希。125／250节点为原件的只读链接，375／500才是新保存；已有125导出／八项复用，不编造旧段逐批记录，不把恢复保存态证明称为已比较未中断轨迹的逐位一致。

GPU0先跑一次从250继续两个真实更新的必要技术验证，检查保存态／全部RNG和实际输入位置；诊断另计原technical额度，不作为正式候选。通过后再保存`formal_snapshot.json`并做CPU闭合。正式只执行余下250次更新，不新增类别或候选；原失败费用保留，各任务只用原拥有窗余额，原正式KD总额度和208.5GPUh不变。卡仅0–3，4–7保留。

由已激活`bitvae`的调用者shell在tmux中运行：

```bash
CUDA_VISIBLE_DEVICES= python -B -u -m experiments.sc_cat_finish_cmcc_20261108.run \
  --snapshot artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_gate_up_20261107/gpu0_3/repair_20261108/after_fresh_fix/formal_snapshot.json \
  --validation artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_gate_up_20261107/gpu0_3/repair_20261108/after_fresh_fix/cpu_formal.json
```

正式状态、预算和逐任务命令／PID／日志位于`repair_20261108/after_fresh_fix/formal/`。前一个`repair_20261108/formal/`为已失败的导出入口接线尝试，原费用保留；其250实际导出经独立GPU fresh验证后引用，不删除它的权重。候选日程计数仍8500；失败尝试的重复更新和拥有窗另列，不把废弃段拼接进有效轨迹。首个真实续训252步核验通过后停止主动监控；原up第三臂全部节点闭合后，自动末类双九项选品、原BF16固定recipe重放及终点九项。失败立即停止依赖，不自动重试。交付原结果与重放结果各自的质量及位宽，不将程序完成当总目标达标。

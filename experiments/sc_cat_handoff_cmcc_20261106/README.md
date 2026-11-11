# 本批两份在途 k 拟合的调度交接

本入口只处理 `formal/controller_handoff_request.json` 已固定的两份 k 正式拟合。原控制器、worker、算法和配额源码保持冻结；恢复后的工作仍写入原 `formal/`，沿用两份原拟合及其原日志。训练卡初始不入池，真实拟合退出且完整2×10K、保存及独立 GPU fresh 核证通过后，才归还对应0或1；后续导出／完整八项只使用2/3。

调用者先激活 `bitvae`、隐藏 CUDA。`prepare` 只保存交接快照，不运行 GPU、不停止进程：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_handoff_cmcc_20261106.run prepare \
  --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/authorization.json \
  --root-boundary artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/root_boundary.json \
  --parent-request artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/formal/controller_handoff_request.json \
  --output artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/formal/controller_handoff_snapshot.json
```

准备完成后由 root 统一后台启动，并把输出写入 `controller_handoff.log`：

```bash
CUDA_VISIBLE_DEVICES='' nohup python -B -u -m experiments.sc_cat_handoff_cmcc_20261106.run run \
  --snapshot artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/formal/controller_handoff_snapshot.json \
  > artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/controller_handoff.log 2>&1 < /dev/null &
```

快照绑定旧源码 SHA、新入口 SHA、原状态／预算／quota、原 PID/start_ticks/PGID、请求和窗口。只有旧控制器已停止且状态仍为“两份 k 拟合在途、其余全部 reserved、未做恢复选择”时才能接管。首份 k 依赖预约只精确复用一次，不增加 raw、轨迹、更新、评测或 GPUh 配额，不作为任意续跑接口。

对原 worker 用 pidfd 等待终态，从原始 `started_epoch` 到终态观察一次计费。若接管前已退出，采用接管时刻保守上界并明确实际退出时间未知。成功依据原 worker 自写的 completed/raw/fresh 证据，`os_exit_code=None` 始终保留，不能把 receipt 的0写成观察到的 OS 退出码。失败同样结算原拥有窗；另一已接管拟合继续自然结束并计费。原始有限窗口耗尽时，仅按保存的 PID/start_ticks/独立 PGID 终止该份本批任务，保留实际耗时并停止依赖。

CPU 验证记录在本批 `audit/handoff_cpu_final.json`：检查真实限定状态、原预约与计数未双计、成功／失败计费、未知 OS 状态、空训练池和2/3评测池、quota 恢复及越界状态拒绝；纯 CPU 子进程实际检查 pidfd 等待与超时关停。该记录不宣告原拟合、恢复或质量评测已经完成。

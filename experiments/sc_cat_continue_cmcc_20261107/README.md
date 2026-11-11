# 从实际 after_o 接续后五类

保持 `k→o` 已选前缀和全链配额，执行 `gate→q→v→down→up`；末类两份真实端点分别完成九项，选择既有 artifact，再从原 BF16 按固定七类 recipe 重放。旧源码、旧结果及旧36 GPUh额度不变。镜像名额已用完，1000名额仍只有两类。

调用者先激活 `bitvae`，控制器隐藏 CUDA。准备真实输入与成本草稿：

```bash
SC_CAT_RUN=artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_gate_up_20261107
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_continue_cmcc_20261107.prepare_plan \
  --cost-audit artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/audit/continuation_cost_20261107.json \
  --output "$SC_CAT_RUN/plan_draft.json"
```

`plan_draft.json` 明确 `confirmed=false`、`user_approved=false`、`not_authorization=true`，运行入口拒绝执行。它提出新208.5 GPUh：technical4、core120.5、末类LongBench24、固定recipe重放60、历史参考复用0。实际用户确认后，root 才据此写独立的 `authorization.json`，填真实授权来源、`phase=formal`、`not_authorization=false`及正式状态；本包不会自行批准。

CPU 总装验证保留真实 after_o／配额、类别成本、完整依赖预算和实际接口证据，不启动 GPU：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_continue_cmcc_20261107.validate_cpu \
  --plan-draft "$SC_CAT_RUN/plan_draft.json" \
  --workers-validation artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/audit/continue_workers_cpu_v3_20261107.json \
  --terminal-validation artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/audit/terminal_interfaces_cpu_final.json \
  --output "$SC_CAT_RUN/audit/continuation_cpu_v2.json"
```

上述文件已存在时复用成功证据；代码变化后用新的验证输出，不覆盖旧证据。CPU 验证不证明新五类收敛、完整原生 LongBench 或正式 selected-stop 重放已经运行。实际首个 gate 拟合主循环和重放首个 `k250/T500` 分别承担其正式路径核证，不另加重复模型短测。

root 获新授权后后台启动，当前4/5另有任务时明确选择读出2/3：

```bash
CUDA_VISIBLE_DEVICES='' nohup python -B -u -m experiments.sc_cat_continue_cmcc_20261107.run \
  --authorization "$SC_CAT_RUN/authorization.json" \
  --validation "$SC_CAT_RUN/audit/continuation_cpu_v2.json" \
  --readout-gpus 2 3 \
  > "$SC_CAT_RUN/controller.log" 2>&1 < /dev/null &
```

训练用0/1；读出默认使用新授权内的全部读出卡，也可选其非空唯一子集。只检查和使用实际选卡，不抢占、不轮询等待其他任务、不使用6/7。末类每个 artifact 固定两个完整任务分片，各6小时，两个 artifact 共4份拥有窗；少卡时排队，队列等待不计GPU费用。节点导出／八项继续与训练流水重叠。

唯一运行状态和费用在 `formal/controller_state.json`、`formal/budget.json`，命令、PID、卡、日志和退出状态逐任务落盘；失败费用照计、不自动重试。`formal/after_up.json`、`frozen_recipe.json` 是主链选择结果；固定 recipe 文件保持不可变，重放实际进度／九项在状态及 `formal/replay/`。正常结束写 `formal/controller.exit.json`，不自动启动其他批次或新参考训练。质量 `not_met` 是有效实测结果，不冒充技术失败或自动扩实验；当前批次完成也不等于跨模型总目标完成。

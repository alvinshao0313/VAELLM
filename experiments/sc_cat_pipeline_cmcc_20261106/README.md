# 同一 SC-CAT 批次的节点流水

本包复用原k→o全部配方、训练、125节点和36 GPUh预算，只让已保存节点的导出／完整八项在2/3上与0/1训练重叠。旧包和已启动训练源码保持冻结。没有新增研究分支、训练、节点、重复导出或重复评测。

`handoff.py`只接受已完成两份k raw拟合／导出／八项、两条baseline已登记且正在训练、全部节点任务仍reserved的精确状态。快照绑定原状态、预算、quota、全部旧runtime源码、worker PID/start_ticks/request及原拥有窗。它不停止进程；root另行核实并只停止旧controller PID。训练独立进程继续运行。

原训练meta是commit标记：Canonical先保存完整optimizer/scheduler/RNG/trainer状态，再原子保存mutable，最后原子发布`checkpoint_meta.json`。`node_export.read_committed_node`读取这一真实发布状态，核对原raw、500/1000日程、数据、scope、LR、seed与完整字段；未发布返回None，已发布但损坏立即失败。它返回与最终`recovery_result.nodes`完全相同的receipt。

`node_export --request`只接收`training_request`绑定、`node_step`、`output`、`destination`、`gpu`、`deadline_epoch`和`authorization`。其余实际条件从原训练请求继承，直接调用既有`boundary.export_selected_node`和`recovery_worker._finish_export`完成独立部署、fresh与GPU前向。它不伪造整条轨迹完成报告。

流水controller同时等待训练终态和每个固定节点的commit；每个节点只消费一次，使用原reserved task ID执行自身一次导出/fresh及完整八项。训练失败立即停止新增依赖；已接管的另一原训练按原有限窗口自然终态并结算。完整500/1000结束后，所有早评receipt必须与最终正式节点清单逐字段相等，全部节点八项齐备才返回原选择逻辑。raw0不能绕过失败或缺失依赖。

新driver从原窗口开始时间计费一次，保留未知`os_exit_code=None`与实际worker成功／失败receipt。初始训练卡池为空，0/1各自退出并核证后才归池；读出池仅2/3。先前六个完成raw阶段直接复用，两baseline不重复register，k依赖预算只精确复用一次。后继已批准条件臂和o的新训练使用同一流水方法。预算、类别依赖、完整选点与after_o停止规则保持原样。

准备快照由root执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_pipeline_cmcc_20261106.handoff \
  --authorization <batch>/authorization.json --root-boundary <batch>/root_boundary.json \
  --output <batch>/formal/controller_pipeline_snapshot.json
```

CPU验证绑定快照、原计费／去重／终态闭合和已验证的6份真实技术节点，不重复模型短验或完整八项：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_pipeline_cmcc_20261106.validate_cpu \
  --snapshot <batch>/formal/controller_pipeline_snapshot.json \
  --node-reader-validation <batch>/technical/pipeline_node_reader_cpu.json \
  --output <batch>/audit/pipeline_controller_cpu_final.json
```

由已激活`bitvae`的root统一后台启动，保持CUDA隐藏并落盘stdout/stderr：

```bash
CUDA_VISIBLE_DEVICES='' nohup python -B -u -m experiments.sc_cat_pipeline_cmcc_20261106.run \
  --snapshot <batch>/formal/controller_pipeline_snapshot.json \
  --validation <batch>/audit/pipeline_controller_cpu_final.json \
  > <batch>/controller_pipeline.log 2>&1 < /dev/null &
```

启动时CPU报告中的新包所有源码SHA必须与实际完全相同；原runtime pins也须仍匹配。原`formal/controller_state.json`和`budget.json`继续是唯一运行状态与费用来源，新增`pipeline_handoff`记录旧／新controller与验证来源，不另建第二条结果链。

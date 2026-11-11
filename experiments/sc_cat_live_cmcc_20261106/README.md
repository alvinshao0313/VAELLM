# SC-CAT 首批 k→o 有限执行器

本包只执行已批准的 k_proj→o_proj。冻结的七类顺序、离线 anchor／最佳不同配方、原始数据、类别 cap 和 BF16 根均由实际文件及 SHA256 绑定。后五类、LongBench、固定参考与 BF16 recipe 重放没有进入本入口。

由调用者先激活 `bitvae`，保持控制器 CUDA 隐藏；正式启动须等待两份拟合和完整恢复技术核证完成。启动形式如下，stdout/stderr 应由调用者落盘，并记录后台 PID：

```bash
CUDA_VISIBLE_DEVICES='' nohup python -B -u -m experiments.sc_cat_live_cmcc_20261106.run \
  --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/authorization.json \
  --root-boundary artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/root_boundary.json \
  --technical-fit artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/technical/fit_k_execution.json \
                  artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/technical/fit_o_execution.json \
  --technical-recovery artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/technical/recovery_execution.json \
  --readout-gpus 2 3 5 \
  > artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_k_o_20261106/controller.log 2>&1 < /dev/null &
```

执行前深核技术输出的两次真实更新、完整节点保存、k/raw0 导出、o 的 prefix/current 两种范围、正式 1024 形状反向及独立 GPU fresh。全部代码在正式启动时冻结，每个 worker 启动前重核源码 SHA 和所分配 GPU 的真实占用；卡忙即失败退出，不等待其他任务让卡。0/1 只拟合、训练；`--readout-gpus` 是2/3/4/5的非空唯一子集，默认四卡，专用于导出与完整八项。实际启动只核0/1及所选评测卡，状态记录同一列表。当前GPU4另有任务，命令明确使用2/3/5；6/7不进入本入口。

`fit_worker --request` 使用 source9 CPU gather 与真实当前父 fresh 拟合；`recovery_worker --request` 分别执行完整轨迹、单节点导出或 raw0 边界；`evaluate --request` 外层隐藏 CUDA，原生完整八项子进程使用请求中的物理卡。正式请求均保留原 B、长度、500/1000 日程与每125节点；技术参数不会带入正式请求。

每份 raw 先真实 zero-KD finalize、新 UUID、独立 fresh、完整八项，再从它自己的 raw 开始原500恢复。两个基准所有节点齐备后才按协议提名唯一第三臂及合格镜像，提前持久化全模型两类1000／两类镜像配额。所有宣布的节点齐备后统一选择；raw0 严格高于全部恢复点才有资格，同分保留恢复点。选择复用原已评部署，经过 `complete_boundary_reference` 和 `StrictMainChain.commit`，其唯一赢家才成为下一类的父。

| 部分 | 单项上限 | 计费归属 |
| --- | --- | --- |
| 实际技术核证 | 4 GPUh合计 | technical，明确外层拥有窗，含失败 |
| 每份 live raw | 拟合4500秒＋zero导出/fresh900秒 | fit，最多4份，共6 GPUh |
| 原500完整轨迹 | 训练5400秒＋4次导出各450秒 | recovery，每份2 GPUh |
| 原1000完整轨迹 | 训练10800秒＋8次导出各450秒 | recovery，每份4 GPUh |
| 完整八项 | 每次900秒，最多4raw＋32节点 | evaluation，共9 GPUh |
| 额外边界额度 | 1 GPUh | 本入口直接复用已评部署，验收 CPU 窗另计 |

每类开始前整体预约两份 raw 与两条基准的全部依赖；第三／镜像整体预约全部训练、导出和评测后才启动。实际拥有窗从启动准备至进程及本轮独立评测子进程终止，单卡不重叠；成功提前结束释放未用预约，失败同样扣实际费用。训练与导出分别拥有独立窗口。硬顶为36 GPUh、4raw、8轨迹、4000更新及36次完整八项，既往离线额度不转入本批。

产物在授权 root 的 `formal/`。`budget.json` 保存预约、实际费用、命令、PID、卡、日志、资源检查和退出状态；`controller_state.json` 保存依赖状态；`categories/` 保存全部比较、证书与边界。完成时保存 `after_o.json`、`quotas.json`、`frozen_recipe.json` 和 `controller.exit.json` 后退出，不占卡、不守护、不自动重试。后续批次可用 `run.restore_quota(绑定的quotas.json引用)` 恢复全七类配额；不能把两个批次分别当新模型重置名额。

`source_candidate_id` 是离线入围来源；`actual_live_candidate_id` 是实际新父上拟合的身份，两者单列。训练 source node UUID、raw UUID 与部署 UUID 分别记录，恢复点与 raw0 不混用。

每类全部比较、选择与主链提交后，精确清除非选中部署的权重文件，先记录路径、原因、大小，再验证删除。保留其元数据、配置、绑定报告与全部原始八项结果，以及所有 raw、完整 mutable/optimizer/scheduler/RNG 节点和已选部署。峰值 dense 副本保守估计为18份当前类部署＋4份 raw＋1份此前赢家，每份以共享 BF16 payload 字节上估；另有训练节点和日志。`storage_projection` 记录该估计、实际源字节及启动时可用空间；它不代表实测磁盘峰值。

CPU 验证入口为 `validate_controller_cpu --authorization ... --root-boundary ... --technical-fit 已完成的实际fit执行记录 --output 新的audit.json`。它检查真实四配方和技术输出、最坏依赖预算闭合、计数硬顶、失败扣费、单GPU重叠拒绝、源码变更拒绝及全模型配额持久化，不启动 GPU，也不提供正式质量证据。

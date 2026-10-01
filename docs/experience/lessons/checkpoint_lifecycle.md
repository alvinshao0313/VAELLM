# Checkpoint：先持久化可恢复状态，再导出并独立重载

## 问题与证据

Linear输出实验在导出第4个模块时触发误差门限，旧runner此前没有保存整分片训练状态，导致后续60个Linear最终状态无法从日志恢复。校验还混合BF16训练、FP32融合与TF32部署路径。[导出故障记录](../records/experiments/linear_output/RECOVERY_NOTES.md)

真实硬件审查还发现：finalization重新unpack已训练packed decoder、提前合并LoRA支路、保留FP32输出wrapper，以及已到max_steps的恢复额外多跑一步等问题。[硬件审查](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

## 下次怎么做

- 在不可逆导出前原子保存恢复必需状态；每个模块导出成功立即更新manifest，失败写明FAILED及目标，不能靠过期TRAINING标记判断进程状态。
- 区分同精度同路径的保存等价性与部署精度变化；各自记录误差，不通过放宽一个总门限掩盖语义混用。
- 采用fresh process strict重载；核对压缩范围、码、拓扑、冻结状态、推理输出，不能只验证文件存在。
- step恢复验证optimizer/scheduler/RNG、可变参数和Sparse Bit状态，并比较下一步更新；到达max_steps后应进入finalization。
- 运行缓存要在状态加载后刷新，训练评测、finalization与部署的表示必须明确。低精度矩阵运算重结合不自动保持同值。
- 成功验收后，依据后续用途清理checkpoint；恢复安全不意味着永久保留所有权重。

## 边界

历史硬件审查只通过单卡/单rank链路，真实多卡为hardware-deferred。本次没有复测。结果报告里旧output_alignment_v6_partial192路径与后续distill_init交付名称不同；读取当前模型必须以实际路径和最新交付记录核对，不能照搬历史“当前状态”。[后续交付说明](../records/experiments/linear_output/EXPERIMENT_SUMMARY.md)

## 层边界恢复须重建运行缓存（2026-09-23）

**本配置实测支持**：Qwen3-8B单stage VAE逐层恢复中，block9/10各1步的连续训练与层边界恢复，在native码/decoder、输出、loss/LR、RNG和冻结状态上严格一致；注入7处过期grouped packed副本后，复用native refresh能恢复正确前向。[工程验证及原始证据](../records/experiments/liftquant_recovery/2026-09-23_engineering_equivalence.md)

**动作**：边界原子保存仅完成块的native状态和必要身份/样例/RNG；恢复从原初始模型开始，验证完成列表是目标前缀，匹配实际初始权重、FP teacher权重、代码、配置与token内容。复制persistent状态后同步grouped packed并清除decoded缓存，再严格比较已完成块输出。teacher前缀重放后才在保存边界恢复RNG。

**边界**：新block原本就建立fresh optimizer，因此层边界可不保存前一块Adam；未完成block的连续proxy/Adam/scheduler不能仅从硬码恢复。本轮各1步只验证恢复路径，未证明长训练或下游收益。

## E2E中途评估改变RNG时的阶段恢复（2026-09-24）

**证据状态：CPU精确对照通过、真实双卡流程已覆盖。** 当前lm-eval会重设随机状态，而Trainer的checkpoint原在评估前保存RNG；阶段暂停再恢复可能重放评估前状态，偏离连续训练。修复是在成功评估并恢复训练状态后，将各rank实际RNG写回本次checkpoint，再允许暂停；不改变连续训练原有语义。[验证及边界](../records/e2e_0910/2026-09-24_rank8_search.md)

**当前实现约束**：启用eval_after_save时，save_steps和save_strategy属于不可变恢复契约；评估本身影响RNG，不能把降低评测频率当作无语义变化的续训加速。阶段预算只改stop_after_step，并保持其为原save_steps的整倍数；待评估完成、RNG写回并成功暂停后再恢复。当前恢复入口从checkpoint反推原run目录，`run_root_dir`不会将续训迁移到新目录；状态以resolved output为准，同一实验按step追加结果。依据为`v6_runtime_state.py`的evaluation_execution契约及`runtime_v6_pipeline.py`的阶段验证，具体运行见上述记录。

阶段预算应独立于scheduler总steps。真实CPU Trainer含dropout/累积的连续与恢复参数、Adam、scheduler逐位一致；真实GPU非确定运行在暂停前也有差异，不能据此宣称GPU逐位等价。最终导出核切换问题已独立修复并通过同路径验收，见下节；阶段恢复通过本身不能替代导出验证。

## 顺序搜索保留中途最佳，部署排名须独立重载（2026-10-01收尾）

**本配置实测支持**：0920 checkpoint、rank8、四卡DP搜索于北京时间2026-10-01 02:20:35正常完成；15组配置中12组到5000步（3组1000→5000、9组2500→5000），另3组到2500步。阶段退出码、run_meta和原始指标支持真实续训完成，不等于连续训练/resume逐位审计。[配置、指标与来源](../records/e2e_0920/2026-09-24_rank8_search.md)

**导出证据边界**：早期baseline `lr3e5/step1000`训练内66.564804%→fresh process四卡strict重载66.554029%，差−0.010775个百分点。收尾前三个step2500快照也各自严格重载成功：pre-MLP权重0为68.009998%→68.024489%，0.03为67.974884%→67.859254%，0.01为67.883982%→67.742981%。排序未变但导出差值不同，不能用早期baseline差值替代各候选复评，更不能推断位级一致。

**动作**：保留中途最佳训练checkpoint直至实际导出重载比较完成，5000步终点不能替代2500步权重；搜索最高分与最终交付排名分别报告。本轮`best_result.json`已指向唯一保留的`deployments/c05_pre_mlp_hidden_loss_weight_1_step2500/model`，重载均分68.024489%，未达到69%。

**清理与恢复边界**：控制器已清理其他生成权重及全部optimizer训练checkpoint，原数据、初始checkpoint、配置、核心日志和原始指标保留；交付模型仍可推理，但无法恢复原训练的optimizer/scheduler/RNG状态做exact-resume。147条cleanup累计124.8008 GiB为控制器历史删除文件的逻辑大小，不是本次文档收尾手动释放的空间。

## 结构转换与解码核切换分开验证（2026-09-24）

**本配置实测支持，已通过真实导出及fresh strict重载，并在用户授权的隔离worktree落地。** 0910 Qwen3-8B的decoder+LoRA真实8步checkpoint：冻结decoder保持packed路径时logits逐元素相同，转换LoRA相对同路径也逐元素相同；仅由packed切到修复候选fused时，logits相对L2约0.00794。原core校验将结构转换和计算核切换合在一起，不能将这一报错直接解释为LoRA参数丢失。[同模型三阶段证据与边界](../records/e2e_0910/2026-09-24_rank8_search.md)

先排除真实实现失配：本次旧fused的LayerNorm/SiLU与当前原生decoder不同，FP32 bias与末Linear舍入也有陈旧行为；真实模块对照与CUDA回归通过后才讨论不同核的正常浮点差异。结构校验应在同一部署计算路径比较转换前后，保持原门限与部署缓存；跨核误差单独记录，保存后仍须fresh strict加载复核。不能只关缓存或放宽总门限掩盖错误，也不把不同计算核全链路逐位一致当作通用要求。

## 源码身份拒绝恢复时保留原运行版本（2026-09-24）

**本配置实测支持**：LiftQuant全层恢复被SIGINT中断后，保存边界与当前源代码仅有通用E2E CLI六行差异，严格恢复因此拒绝。隔离源码副本还原这六行后，101个训练依赖文件SHA与边界逐个相同，初始权重、teacher、参数、token、runtime及保存张量检查通过。[中断与重启记录](../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**动作**：先定位身份差异，不修改boundary或关闭检查，也不回滚其他活动实验使用的共享文件。已有原内容可精确重建时，在独立源码目录保留原版本，cwd/PYTHONPATH都指向它，模型/数据/产物路径显式指向原位置。运行必须持有明确源码副本或持续保护源码，不能仅依赖启动时已导入模块来保证后续子进程一致。

**边界**：本次CLI差异是重启阻断，不是SIGINT来源。只有KeyboardInterrupt/子进程signal退出时，不能认定OOM或算法失败，也不能推断信号发送者。无控制终端并断开stdin可隔离终端操作，但不能保证外部显式信号永不再次终止任务；仍保留正常的SIGINT/SIGTERM停止能力。本配置在独立session、stdin=/dev/null下仍第二次收到SIGINT，用户明确表示未主动停止；这证明先前终端隔离措施未解决未知发送源，不能将中断归因于SSH断线。最新边界的101个源码SHA及CPU结构校验仍通过，信号中断与源码恢复问题须分别判断。层边界恢复只保留已完成层，未完成层需从该层首步重做。


重复SIGINT中断而缺少发送者审计时，可在既有运行外层加入系统现有的信号记录工具。本次strace 5.16以seccomp过滤全部syscall，仅记录指定中断信号；真实CPU探针验证能给出原始发送者PID/UID，且保持原有SIGINT退出语义。Python最终可能再向自身发SIGINT以还原退出状态，溯源应看第一条外部SI_USER记录，不能把随后的self-signal误判为训练主动中断。此探针未查明此前两次发送者；工具只改善取证，不表示外部中断原因已经修复，也不改变训练协议。

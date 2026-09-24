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

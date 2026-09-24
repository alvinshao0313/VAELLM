# Mixed-bit：缓存契约、代理目标与最终质量

## 问题与证据

历史工程审查列出候选结构不匹配、子进程解释器漂移、top-k全logits搬运、worker永久等待、自定义pool路径断链以及tokenizer指纹不完整等问题。后续报告记录301项mixed-bit检查、57项集成检查及真实q_proj单步冒烟。[验收记录](../history/reviews/2026-08-05-mix-bit-hardening-and-production-readiness/task-10-report.md)

## 下次怎么做

- resume前核对实际artifact结构和metadata：stage、dim、logical bits、decoder尺寸及hash，而非只看completed文件名。
- 子进程使用父进程实际解释器；在bitvae shell里启动不代表子进程硬编码python也正确。
- top-k只传所需有效token的K项，避免先把完整logits搬CPU再截取；exact KL与top-k KL明确区分。
- worker启动和运行都有超时与死亡检查；队列无消息不能无限等待。
- 自定义manifest所在目录是pool路径来源，禁止偷偷回退默认pool。
- tokenizer指纹覆盖content、chat template和added vocab；路径只是provenance，不能因source/reload路径不同产生伪不一致。
- final checkpoint独立保存tokenizer并local-only strict重载；大state用流式hash，避免为校验clone整模型。

## 边界

这些是缓存正确性和运行工程经验，不是混合位宽能提升质量的证据。上述工程验收中的单步冒烟没有执行完整35候选训练及全部cost评估；后续独立实验见下一节。当前接口和门禁仍应阅读[mix_bit说明](../../../mix_bit/README.md)。

## 代理目标与最终质量必须分别验证

最后核查：2026-09-23。证据状态：历史报告及分配摘要支持，未在本轮重跑下游评测。来源：[Qwen3-8B 混合位宽结果整理](../records/experiments/mixed_bit/qwen3_8b_vae_1to3bit.md)。

**问题与条件：** 252 个 Linear、均匀 2-bit 基线、top-k=256 单层替换代价、平均码流位宽≤2.0、无蒸馏恢复的历史实验中，混合方案校准 KL 降低，但 wiki-PPL 恶化。排除 1-bit 后仍未优于均匀基线的 PPL。

**已知与解释：** 求解器的全局最优只针对 ΣΔKL 近似目标；将基线 KL 加上该目标产生负预测，而实际联合替换 KL 为正，表明可加预测不能直接代表最终模型。大量负代价本身不证明实现错误；估计噪声、跨层相互作用、候选质量各占多少尚属待验证解释。

**下次动作：** 小规模验证组合替换的预测与实测偏差；分开记录求解目标、组装后完整模型 KL/PPL/下游质量及重载一致性。预算为≤时不强制填满，实际存储比较另外计入 decoder 和未压缩部分。不要只因 optimal、较低校准 KL 或稍高任务均值扩大实验。

**适用边界：** 这是该候选池、校准集和未恢复配置的经验，不能泛化为所有混合位宽方法无效。排除低位宽档不是已证实的通用修复。

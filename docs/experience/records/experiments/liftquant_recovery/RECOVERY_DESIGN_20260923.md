> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`experiments/liftquant_recovery/RECOVERY_DESIGN_20260923.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# VAE 逐 block 蒸馏：一致范围与恢复设计

本记录是 2026-09-23 设计审查的后续。目标是在现有无旋转、单 stage、native v6 checkpoint 上做好逐 block 蒸馏；本轮没有启动新训练或正式实验。当前证据支持明确修正方向，尚不支持声称某组配置全局最优或已经超过 LiftQuant。

## 决策：什么应一致

| 范围 | 决策 | 原因 |
| --- | --- | --- |
| 教师输入和目标 | FP teacher 前缀、align=1、整 block 最终 hidden-state MSE | 保持官方 Stage2/ALL 的优化任务 |
| 训练范围 | 当前 block 的全部压缩 Linear，所有 bits 与全部 decoder 联合训练；其他参数冻结 | 对齐离散编码＋连续重建器的功能范围 |
| 数据预算 | RedPajama 4096×2048、留出128、batch2、2epochs、3968 updates/block、seed42 | 固定可比较的训练条件 |
| 优化器和调度 | 官方 AdamW 参数及实际 scheduler 调用语义；无额外损失、交替训练、warmup 或裁剪 | 尽可能直接复用无关表示的代码 |
| VAE 表示 | 原 decoder、原 bits、保护范围、共享方式、无旋转、native v6 全部保留 | 不能为凑相同参数组而变成另一个压缩模型 |
| 连续参数 | decoder 自身 norm/bias 继续训练，原 LLM norm/bias 不训练 | 完整非线性重建器是合法适配，不应随意删减 |
| 编码坐标和 lr | 按 VAE 的解码敏感度建立尺度，明确不是 FWT 的等价重参数化 | 0/1 位坐标直接套 2e-5 已证明存在实质问题 |
| decoder lr | 下一次有限 smoke 候选 1.25e-5；通过固定更新方向诊断得到 | 当前 2e-4 更新过大，官方 lt_lr 不直接代表 VAE 的函数变化 |
| 精度/attention | 优先保证真实原生 BF16 解码前后与梯度约定一致；保留 bitvae 环境和原有 FA2 路径并记录 | 此处的合理差异不能靠强换依赖消除 |

官方依据：[Stage2 源码](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/quantize/liftq.py#L495)、[FWT 编码参数化](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/quantize/tmplinear.py#L134)。

“所有参数名字和数字相同”不适用于 VAE 与 FWT 两种不同表示。应该严格一致的是优化问题、数据、预算、冻结原则与外围训练实现；表示特有部分必须给出定义和验证依据。

## 为什么之前的编码配置必须改

当前代理从 0/1 出发，到阈值 0.5 的距离为 0.5。fresh AdamW、beta1=0.9、beta2=0.999、wd=0 时，用 Cauchy–Schwarz 可对任意梯度序列给出逐坐标界：

\[
\frac{|\hat m_t|}{\sqrt{\hat v_t}}
\le B_t
=\sqrt{\frac{(1-\beta_1)^2}{1-\beta_2}
\frac{1-\beta_2^t}{(1-\beta_1^t)^2}
\frac{1-r^t}{1-r}},\quad r=\beta_1^2/\beta_2.
\]

其全时上界约 7.2702918。证明中的关键不等式为：令 u=beta2^t、v=r^t，则 (1−sqrt(uv))²−(1−u)(1−v)=(sqrt(u)−sqrt(v))²≥0。eps>0 只减小步幅。

3968 步、初始 lr≤2e-5、余弦下限为初始的1/20时，已用学习率之和为 0.0416735，所以累计绝对位移上界为 **0.302978506<0.5**；逐步计算得到更紧的界 **0.228344416**。官方实际 step()+get_lr() 序列在这个下降周期更小，其逐步界约 **0.228221634**。

因此，在当前完整预算、fresh moments、wd0、无额外 proxy 重置的前提下，精确算术中所有硬码都无法翻转。这不是只做一次更新造成的现象。正常 FP32 舍入远不足以跨过约0.197的保守余量；这不是完整 GPU 浮点形式证明，但不应期待数值误差绕过它。

将 0/1 平移成 −0.5/+0.5 不改变这一事实。随意改成 ±RMS、缩小 margin 或只增大梯度也没有等价保证。Adam 近似消去整体梯度倍数，单纯放大 STE 梯度不是可靠解决办法。

## 编码适配的推荐候选及边界

推荐把编码尺度绑定到实际 decoder 的重构敏感度，作为下一步待验证的 VAE 适配，而不是再次借用原始 FP 权重 std：

1. 从原 checkpoint 的真实码中固定抽样，测量翻转单个 bit 后 decoder 输出权重的 RMS 变化；按 Linear 得到固定的正尺度 s。
2. 定义训练代理 p=s(b−1/2)，硬码仍为 b=clamp(round(p/s+1/2),0,1)。p 只存在于训练，不新增模型 scale、旋转或推理存储。
3. STE 明确包含 1/s 的链式因子及量化器的饱和门控；与独立 dense STE 对照。采用 FP32 代理和 Adam 状态。
4. min(lw_lr, proxy.std()/50) 中的 std 取实际代理本身。记录 margin、lr/margin、累计可达位移、真实 bit 变化及重构函数变化。

这样 p 的单位由实际解码函数确定，避免把不同 decoder 下的“一个 bit 坐标单位”当成相同权重变化。它仍不是官方 FWT 坐标的严格等价，也不保证最优；非线性相互作用和激活重要性仍要由 block MSE 检查。s 应在训练前固定并记录，不随损失偷调；零/非有限敏感度应明确报错分析，不强行造值。

这套代理适配尚未实现进训练入口。本轮已证明旧配置不合适，不应为了立即跑起来而把尚未测量的 s 任意设为一个常数。也不应把每一步必须翻码作为通过条件。

## 已完成的真实数值检查

使用 GPU4、同一 bitvae 环境，从 block9 七类 decoder 各抽取512行真实 packed codes。CPU mmap 读取 checkpoint，不构造完整模型、不更新参数。比较完整的 linear→LayerNorm→SiLU→linear，而非只检查第一层。

旧 code VJP 用 FP32 decoder 权重，前向却使用其 BF16 舍入值，导致约0.106%–0.133%的编码梯度差异。本轮已修正为用前向实际权重值做 VJP，并保持 FP32 梯度累加。

修正后的 BF16 测量：

- 七类 decoder 的前向与独立 packed 算术参考逐元素一致。
- 七类编码梯度与参考逐元素一致。
- decoder 全部参数梯度的最大相对 L2 误差为 2.2433e-8；差异来自第一层 FP32 求和顺序，其余参数梯度一致。
- 峰值 allocated 为18.37MiB，reserved为24MiB（不含 CUDA context）。

普通 dense BF16 和原生 packed kernel 的中间舍入规则不同，前向可相差约0.3%–0.58%；不能把它们混作同一个严格参考。FP32 路径也有 Triton TF32 运算影响，相关差异完整保留，不用宽容差输出笼统 PASS。

这证明了被测真实 decoder 切片的 BF16 数值约定，尚不等于完整 block 的梯度和多步恢复效果全部验证。上述 code VJP 差异也不是旧模型 MSE 大涨的直接原因，因为旧步骤没有改变硬码。

结果：远程 `.result/liftquant_recovery/decoder_gradient_20260923_01.json`（修正前）及 `_02.json`（修正后），均退出0；程序 `verify_decoder_gradients.py`。

## decoder 步长的实际依据

用已有 A/B decoder 差量 δ，仅做 θA+αδ 的前向测量，硬码完全固定，使用历史相同4条64-token、同一 FP teacher 前缀和原生 packed 算术。无优化器、无新训练、无 checkpoint 写入。

| 原更新量比例 α | 训练2条 MSE | 留出2条 MSE | 全4条 MSE |
| --- | --- | --- | --- |
| 0，原 A | 0.07141407 | 0.07525356 | 0.07333381 |
| 1/16 | 0.07021061 | 0.07424438 | 0.07222749 |
| 1/4 | 0.07160889 | 0.07525326 | 0.07343107 |
| 1，原 B | 1.17409277 | 0.43979335 | 0.80694306 |

两端准确复现历史指标。小步使训练和留出同时改善，而完整步大幅恶化，支持“原 decoder 更新过大”的解释。该测量不证明 1/16 是最优比例，也不证明所有 block 都应使用相同步长。

因此下一次有限 smoke 的 decoder lr 可先取 2e-4/16=**1.25e-5**，固定配置观察多步趋势；不能据此宣称已经训练得到更好的 B。当前已有保存的 B 仍是旧失败效果的模型，不应当成最终成果。

结果：`.result/liftquant_recovery/saved_step_20260923_01.json`，退出0；峰值 allocated2.25GiB、reserved2.88GiB；程序 `diagnose_saved_step.py`。任务已完成，无遗留训练进程。

## 本轮代码改动与后续门槛

已改：

- `liftquant_optimizer.py`：提取官方 AdamW 与实际逐组调度；CPU 四步序列回归通过。
- `block_train.py`：接入上述逻辑，保持 block.eval；选择的参数仍反传。std 学习率上限改取实际 proxy，移除对原 FP teacher 权重 std 的借用；这本身不会解决旧0/1坐标的冻结问题。
- `recover.py` 和两个 checkpoint 运行脚本：seed42。
- `all_bits.py`、`verify_recovery.py`：编码 VJP 使用前向实际舍入权重。
- 两个永久诊断入口：完整 decoder 梯度、历史更新方向插值。

未完成：敏感度归一化代理、decoder 新 lr 的多步真实恢复、修正后两 block 保存/重载/统一评测复验。现有入口仍保留旧代理及旧 lr 参数，不能因为本轮局部修正就直接启动全量脚本。

后续先完成代理及独立梯度对照，再做两个相邻 block 的有限多步验证。看训练/留出趋势、编码可达性、实际权重与输出变化、冻结和保存一致性。不要求每步下降，不强制翻码，不用下游测试集挑超参数，不增加新损失或 LoRA，不扩大到端到端训练。

## 如何判断“超过 LiftQuant”

非线性 VAE decoder 提供了不同的表达能力，也可能使同一参数步长引起更大的函数扰动；两者都要测，不能只依据参数更多预测胜出。

比较必须使用同一基座/版本、同一评测配置和相近的实际存储预算，包含 decoder、dense、norm/head 等额外开销。本 checkpoint 有60个dense Linear，压缩目标的2-bit payload不能代表整个模型2-bit。恢复增益可比较 A→B；最终精度超过原版 LiftQuant 则需要同条件原版基线，当前一个 A/B 还不能证明。

另外，[LiftQuant 论文 Appendix A](https://arxiv.org/html/2606.04050v1#A1) 给出的变换 lr 是1e-3，README 示例是2e-4；论文主结果的部分设置还包含 Appendix B 的端到端微调。应明确比较 block-only 的同阶段结果，不混用不同训练预算的成绩。本轮没有扩展为原版 LiftQuant 全量实验。

## 收尾

本轮语法检查退出0，官方优化器调度回归通过，两个decoder数值诊断和历史方向诊断均退出0。没有新增训练、checkpoint、依赖或Git提交，也没有删除文件。新增结果/日志合计约456KiB，保留作为修复依据；输入A和旧B各约6.8GiB。最终盘点远程剩余987GiB，磁盘变化包含其他作业，不能归因于本次清理。

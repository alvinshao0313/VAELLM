# 渐进残差门控：文献事实与可借鉴候选

整理日期：2026-10-30。依据已阅读的一手论文、作者项目页及VRVQ官方实现；没有执行作者代码或本项目实验。

[返回实验设计与候选池](README.md#设计候选池)。本文维护文献事实与迁移边界，候选的实际取舍和结果唯一写在实验README／原始运行记录，不将外部结果升级成本项目经验。

## 按设计问题阅读

| 要解决的问题 | 优先来源 | 对应候选或边界 |
| --- | --- | --- |
| 每向量如何选择有序残差深度，硬mask怎样传梯度 | VRVQ §III-A/B/C | 单重要性标量、有序阈值、平滑STE |
| 怎样保证最终码流真的满足预算 | BAMU §2.1—2.3 | 候选边际收益、prefix硬分配与实包核算 |
| 低阶前缀如何得到训练 | SoundStream §III-C、§V-C | quantizer dropout候选，不能代替内容门控 |
| 怎样学习可关闭结构 | L₀ regularization §2.1/2.2/2.4 | hard-concrete备选，需另加prefix和费用模型 |
| LLM残差多精度是否已有先例 | RRQ | 多残差checkpoint参考，不是端到端门控证明 |
| 码／decoder联合优化、功能评分和初始化 | 相邻背景文献表 | 支持问题拆解，不直接确定本实验配置 |

## VRVQ：可训练的残差前缀门控

**来源：** [Variable Bitrate Residual Vector Quantization for Audio Coding，ICASSP 2025最终v3](https://arxiv.org/html/2410.06016v3)；[Sony项目页](https://sonyresearch.github.io/VRVQ/)；官方[门控工具](https://github.com/SonyResearch/VRVQ/blob/main/models/utils.py)与[量化器](https://github.com/SonyResearch/VRVQ/blob/main/models/quantize.py)。

**原文机制。** 每帧一个重要性 p∈(0,1)，以码率因子ℓ生成 `m_k=1[ℓp≥k]`，k从0计，因此至少第一阶开启且天然prefix。前向用硬mask，反向用log-cosh平滑代理构造STE。训练用固定λ的重要性正则并随机采样ℓ，学习多个码率；没有全四阶起步后单调缩全局预算的日程。

通用STE形式为：

\[
m_{\mathrm{STE}}=m_{\mathrm{soft}}+\operatorname{sg}(m_{\mathrm{hard}}-m_{\mathrm{soft}}).
\]

**实现经验。** 官方先计算各阶，再mask，关闭阶的commitment／码本损失也被mask。改善gate梯度不等于关闭尾阶仍有训练。全阶样本比例的消融没有普遍收益，低码率还可能变差。论文计入每帧深度标签，但不是完整容器实包的硬预算方法。

**迁移候选。** 把“帧”替换为静态权重向量，训练重要性、最终固化长度；先比较一个标量控制深度，而非四个独立开关。权重侧不照搬音频预测器的时序卷积、λ、ℓ范围或全阶比例。

**仍需我们研究。** 预算递减日程、LLM功能梯度、BSQ stage完整贡献、尾阶维护、门控内存及最终压缩契约。该音频结果不提供MMLU恢复幅度证据。

## BAMU：真实预算下的硬前缀分配

**来源：** [Bitstream-Aware Marginal-Utility Allocation for Frozen Pretrained Neural Speech Codecs，2026预印本](https://arxiv.org/html/2608.08432v1)，§2.1—2.3及§3.3。

**原文机制。** 冻结codec，训练每帧每阶的边际残差收益预测器，再用约束分配器选择硬prefix深度。收益标签来自非负截断的残差失真差；分配包含候选λ、Viterbi及贪心补阶。最终按实际序列化大小验收，计活动索引、深度RLE、头部和分别字节对齐的payload；代理切换惩罚不等于实际费用。

**关键反例。** 某些DAC低码率条件下，连真实latent收益分配也不优于固定深度；保留负收益的变体有所改善但未反超。因此局部残差收益不是最终质量的充分代理。

**迁移候选。** 训练侧费用代理与最终真实费用分开；使用合法完整前缀候选，保留码率可行性证据。LLM侧还须计共享decoder、保护、旋转、深度信息及packed差异。

**不能直接迁移。** 它不训练codec、没有STE或渐进预算日程；不能当作尾阶训练已解决的证据。音频相邻帧的深度平滑、时间预测和RLE收益，不能默认适用于相邻权重向量。实际费用模型需要本项目自己的serializer。

## SoundStream：残差前缀的多码率训练

**来源：** [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/html/2107.03312)，§III-C、§V-C。

**原文机制。** 使用RVQ并在训练时对量化器层施加结构化dropout，让同一个模型支持不同残差前缀和码率。

**迁移候选。** 在全阶训练之外，为较短前缀提供学习机会，研究前阶能否承担核心信息；作为可选前缀训练策略，与无dropout的同框架比较。

**边界。** 随机抽取前缀不等于每向量内容自适应，也不等于全局预算随训练下降。重建／感知音频目标不适用于LLM知识恢复，dropout概率和阶段组合不直接照搬。

## L₀ regularization：可学习稀疏结构的备选机制

**来源：** [Learning Sparse Neural Networks through L₀ Regularization，ICLR 2018](https://arxiv.org/html/1712.01312v2)，§2.1、§2.2、§2.4。

**原文机制。** 随机非负门控与hard-concrete分布，使预期非零数可对分布参数优化，并联合学习网络参数与稀疏结构。训练样本可为0、1或中间值，并非默认全程二值前向。

**迁移候选。** 学习整阶残差的启用概率，研究随机门控是否优于确定性STE。prefix可通过条件门控等方式另行约束；四个独立门控本身不保证残差次序。

**边界。** 预期L₀不是实际bitstream大小；普通组稀疏也不包含共享decoder启用费、长度图或BSQ特殊语义。最终必须明确硬化、适配与部署，不按软幅度少计位宽。

## RRQ：LLM残差多精度的相邻先例

**来源：** [Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs，2026预印本](https://arxiv.org/abs/2608.04048)。

它从2bit基础加2bit RTN残差，形成4／6／8bit等版本，强调同checkpoint支持多精度；原文无校准、无联合多位宽优化。可参考残差包和多精度表示，但不能称它验证了逐向量门控、全四阶E2E或训练中降预算。

## 相邻背景：只借对应机制

| 一手来源 | 本专题可借鉴的问题 | 必须保留的限制 |
| --- | --- | --- |
| [PV-Tuning](https://arxiv.org/html/2405.14852v2) | 离散分配与连续参数共同优化整模型目标；附录H提醒采样预处理有偏差 | 不是残差深度门控；已有局部借鉴不等于完整算法已验证，数据附录也不是学科覆盖证明 |
| [EfficientQAT](https://aclanthology.org/2025.acl-long.498.pdf) | 块级与全局训练阶段、可训练空间对恢复的影响 | Block-AP开放W及量化参数，不等价固定码decoder；没有本方案的逐向量深度门控 |
| [UPQ](https://arxiv.org/html/2506.09104v1) | 更好初始化与蒸馏目标共同影响低位宽知识恢复 | 标量渐进位宽不是残差删阶；训练预算和最终MMLU差距不能省略 |
| [KronQ](https://arxiv.org/html/2607.07964v2) | 输入与输出侧统计共同影响功能敏感度、跨层预算分配 | 统计评分不等于本BSQ候选的真实误差；低2.x分配证据主要为PPL |
| [ThinkBit](https://arxiv.org/html/2610.34886v1) | 敏感度、候选代价、全局预算、码搜索及后续KD分开组织 | 标量量化噪声表不适用BSQ；推理指标优胜不意味着MMLU近无损 |
| [SliM-LLM](https://proceedings.mlr.press/v267/huang25aa.html) | 层内组级精度分配与敏感度加权 | 不直接证明跨层预算＋E2E渐进门控，名义bit与完整费用不同 |
| [BitDistiller](https://aclanthology.org/2024.acl-long.7/) | 低位宽QAT与自蒸馏目标的关系 | 目标变化应独立比较，不在首个日程对照中同时换散度 |

## 汇总为候选经验，而不是预设配方

| 文献／项目线索 | 对本方案的候选动作 | 怎样判断才有下一步价值 |
| --- | --- | --- |
| VRVQ硬prefix及梯度代理 | 优先审查单重要性标量＋硬前向／平滑反向 | 路由合法、梯度有效、费用清楚，再用完整功能指标比较 |
| BAMU真实费用与latent失效 | 对训练代理和实际部署成本双重记录 | 预算真实可行；LLM功能不能由局部MSE替代 |
| SoundStream多前缀训练 | 把随机前缀列为辅助策略候选 | 同预算下低阶功能更好且最终模型不退化，而非只看使用率 |
| L₀可学习结构 | 把有序随机门控列为后续备选 | 同框架下改善优化或码率，不靠软贡献隐藏成本 |
| 关闭尾阶梯度消失 | 单独观察尾阶使用与训练 | 确有饥饿再增加维护策略，不把全阶样本比例当通用答案 |
| 静态权重与音频输入不同 | 训练可变、部署固化，谨慎使用邻接平滑 | 不引入逐token权重路由，不假设RLE天然有效 |
| 原有退出下界与局部输出反例 | 预算内允许重新分配，并保持功能验证 | 不重复固定几何退出人数；不把权重误差下降写成知识恢复 |

这次找到的工作分别覆盖了门控学习、前缀训练、硬预算或残差表示；没有在同一设置中验证“全四阶起步＋LLM功能E2E＋全局预算渐进收紧＋静态硬前缀部署”。该组合仍是本实验的待验证设计。

当前阅读顺序建议为VRVQ→BAMU→SoundStream→L₀，RRQ及相邻背景按设计问题补读。后续新增论文先更新本页证据与候选关联；只有新证据支持才更新实验README的取舍，不按每篇论文新增一个实验分支。

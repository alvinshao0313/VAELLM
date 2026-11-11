<!-- lesson-redirect -->

# KD 与数据边界：同名参数可能已有不同数学语义

本专题已按“一级分类 → 二级专题 → 经验条目”拆分；以下保留旧标题锚点和完整去向，原经验正文只在目标专题维护。新总结请写入[经验分类入口](../README.md)对应专题。若并行任务仍向本文件补写，须先保留增量、核对来源后归并，不能覆盖或直接删除。

<a id="按条件选择动作"></a>

- **按条件选择动作** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--按条件选择动作)

<a id="下次怎么做"></a>

- **下次怎么做** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--下次怎么做)

<a id="同父恢复须同时看完整净收益与能力代价2026-10-30"></a>

- **同父恢复须同时看完整净收益与能力代价（2026-10-30）** → [压缩后恢复微调经验 / 恢复监督目标、更新保持与能力取舍](post_recovery/objectives.md#kd-and-data--同父恢复须同时看完整净收益与能力代价2026-10-30)

<a id="逐类别恢复的综合选点与单任务保持"></a>

- **逐类别恢复的综合选点与单任务保持** → [逐类别恢复经验 / 恢复预算、优化与阶段选点](category_recovery/optimization.md#kd-and-data--逐类别恢复的综合选点与单任务保持)

<a id="逐类别恢复的本段损伤与整链净收益"></a>

- **逐类别恢复的本段损伤与整链净收益** → [逐类别恢复经验 / 恢复预算、优化与阶段选点](category_recovery/optimization.md#kd-and-data--逐类别恢复的本段损伤与整链净收益)

<a id="softmax诊断须保留缓存目标与真实前向语义2026-10-29"></a>

- **Softmax诊断须保留缓存目标与真实前向语义（2026-10-29）** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#kd-and-data--softmax诊断须保留缓存目标与真实前向语义2026-10-29)

<a id="长文本sft须核对截断后的真实监督2026-10-04"></a>

- **长文本SFT须核对截断后的真实监督（2026-10-04）** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--长文本sft须核对截断后的真实监督2026-10-04)

<a id="问题与证据"></a>

- **问题与证据** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--问题与证据)

<a id="冻结参数不保证能力保持"></a>

- **冻结参数不保证能力保持** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#kd-and-data--冻结参数不保证能力保持)

<a id="平均nll与实际修复损坏的分解"></a>

- **平均NLL与实际修复、损坏的分解** → [评测与诊断经验 / 阶段损伤、能力退化与因果诊断](evaluation/diagnostics.md#kd-and-data--平均nll与实际修复损坏的分解)

<a id="gas归约和真实数据消费"></a>

- **GAS归约和真实数据消费** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--gas归约和真实数据消费)

<a id="来源比例有效正文及完整全链质量"></a>

- **来源比例、有效正文及完整全链质量** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--来源比例有效正文及完整全链质量)

<a id="跨模型恢复与知识来源配对"></a>

- **跨模型恢复与知识来源配对** → [搜索与对照设计经验 / 跨模型与跨条件迁移](experiment_design/transfer.md#kd-and-data--跨模型恢复与知识来源配对)

<a id="教师正确率真实知识覆盖与原始schema"></a>

- **教师正确率、真实知识覆盖与原始schema** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--教师正确率真实知识覆盖与原始schema)

<a id="历史prompt公式与causal边界"></a>

- **历史prompt公式与causal边界** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--历史prompt公式与causal边界)

<a id="边界"></a>

- **边界** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--边界)

<a id="2026-09-24-当前实现核查"></a>

- **2026-09-24 当前实现核查** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--2026-09-24-当前实现核查)

<a id="正式评分与真实错误类型"></a>

- **正式评分与真实错误类型** → [评测与诊断经验 / 阶段损伤、能力退化与因果诊断](evaluation/diagnostics.md#kd-and-data--正式评分与真实错误类型)

<a id="当前loss公式日志窗口与完整ce配方"></a>

- **当前loss公式、日志窗口与完整CE配方** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--当前loss公式日志窗口与完整ce配方)

<a id="真实artifactsplit与train-only替代关系"></a>

- **真实artifact、split与train-only替代关系** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--真实artifactsplit与train-only替代关系)

<a id="教师保护scale和整数qat的条件证据"></a>

- **教师、保护scale和整数QAT的条件证据** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#kd-and-data--教师保护scale和整数qat的条件证据)

<a id="覆盖目标与保留的完整配对"></a>

- **覆盖、目标与保留的完整配对** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--覆盖目标与保留的完整配对)

<a id="auxiliary_train来源与覆盖预算"></a>

- **auxiliary_train来源与覆盖预算** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--auxiliary_train来源与覆盖预算)

<a id="域损失clip和opd保留的真实贡献"></a>

- **域损失、clip和OPD保留的真实贡献** → [压缩后恢复微调经验 / 恢复监督目标、更新保持与能力取舍](post_recovery/objectives.md#kd-and-data--域损失clip和opd保留的真实贡献)

<a id="一阶投影与实际adambf16位移"></a>

- **一阶投影与实际Adam、BF16位移** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#kd-and-data--一阶投影与实际adambf16位移)

<a id="实际w更新量与局部lr候选"></a>

- **实际W更新量与局部LR候选** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#kd-and-data--实际w更新量与局部lr候选)

<a id="muonpcgrad和同期adam控制"></a>

- **Muon、PCGrad和同期Adam控制** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#kd-and-data--muonpcgrad和同期adam控制)

<a id="ce混合配方的短程筛选2026-09-24"></a>

- **CE混合配方的短程筛选（2026-09-24）** → [压缩后恢复微调经验 / 恢复监督目标、更新保持与能力取舍](post_recovery/objectives.md#kd-and-data--ce混合配方的短程筛选2026-09-24)

<a id="先核实实际数据文件再解释混合比例2026-09-24"></a>

- **先核实实际数据文件，再解释混合比例（2026-09-24）** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--先核实实际数据文件再解释混合比例2026-09-24)

<a id="生产路径重放token归因与模板审计2026-09-24"></a>

- **生产路径重放token归因与模板审计（2026-09-24）** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--生产路径重放token归因与模板审计2026-09-24)

<a id="数据修复后的短程收益与去包装对照2026-09-24"></a>

- **数据修复后的短程收益与去包装对照（2026-09-24）** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--数据修复后的短程收益与去包装对照2026-09-24)

<a id="0920的hidden与pre-mlp权重须分别对照2026-10-01"></a>

- **0920的hidden与pre-MLP权重须分别对照（2026-10-01）** → [压缩后恢复微调经验 / 恢复监督目标、更新保持与能力取舍](post_recovery/objectives.md#kd-and-data--0920的hidden与pre-mlp权重须分别对照2026-10-01)

<a id="dense-offload参数与实际显存边界2026-09-30"></a>

- **Dense offload参数与实际显存边界（2026-09-30）** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#kd-and-data--dense-offload参数与实际显存边界2026-09-30)

<a id="normhead与追加蒸馏的预算边界"></a>

- **norm、head与追加蒸馏的预算边界** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#kd-and-data--normhead与追加蒸馏的预算边界)

<a id="真实序列曝光与更长学习率时域"></a>

- **真实序列曝光与更长学习率时域** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#kd-and-data--真实序列曝光与更长学习率时域)

<a id="短答案的标签拟合与教师分布可能不一致2026-10-04"></a>

- **短答案的标签拟合与教师分布可能不一致（2026-10-04）** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--短答案的标签拟合与教师分布可能不一致2026-10-04)

<a id="教师格式概率与匹配训练的完整结果"></a>

- **教师格式概率与匹配训练的完整结果** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--教师格式概率与匹配训练的完整结果)

<a id="answer-only监督归约分母与联合码位对照"></a>

- **answer-only监督、归约分母与联合码位对照** → [数据与监督经验 / 监督位置、模板、归约与日志语义](data/supervision.md#kd-and-data--answer-only监督归约分母与联合码位对照)

<a id="答案kl自由生成与训练外迁移"></a>

- **答案KL、自由生成与训练外迁移** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--答案kl自由生成与训练外迁移)

<a id="混合gold目标的margin与固定身份读出"></a>

- **混合gold目标的margin与固定身份读出** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--混合gold目标的margin与固定身份读出)

<a id="tokenizer迁移和散度范围对照"></a>

- **tokenizer迁移和散度、范围对照** → [数据与监督经验 / 教师目标、缓存与模型适配](data/teachers.md#kd-and-data--tokenizer迁移和散度范围对照)

<a id="更换任务数据须同时核对实际监督量与采样路径2026-10-04"></a>

- **更换任务数据须同时核对实际监督量与采样路径（2026-10-04）** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--更换任务数据须同时核对实际监督量与采样路径2026-10-04)

<a id="连续预算与来源重配的实际曝光"></a>

- **连续预算与来源重配的实际曝光** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--连续预算与来源重配的实际曝光)

<a id="语言代码标签与实际监督token"></a>

- **语言、代码标签与实际监督token** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--语言代码标签与实际监督token)

<a id="任务lm与sft的格式监督量和码位配对"></a>

- **任务LM与SFT的格式、监督量和码位配对** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--任务lm与sft的格式监督量和码位配对)

<a id="gen来源家族交互与head长预算"></a>

- **Gen来源、家族交互与head长预算** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--gen来源家族交互与head长预算)

<a id="中文随机消费与短答案截断"></a>

- **中文随机消费与短答案截断** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--中文随机消费与短答案截断)

<a id="历史校准准备的重复身份修复边界"></a>

- **历史校准准备的重复身份修复边界** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#kd-and-data--历史校准准备的重复身份修复边界)

<!-- /lesson-redirect -->

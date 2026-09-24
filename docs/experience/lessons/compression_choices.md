# 压缩选择：相同码流位宽不代表相同模型或预算

## 问题与证据

单类别压缩A/B中，同约2bit/weight的32/32/2-stage与64/32/1-stage在不同类别表现不同：down_proj偏向后者，多个其他类别偏向前者，o_proj的PPL和任务均值还存在冲突。[码本A/B](../records/.result/catlora_codebook_ab/cat_codebook_ab_per_category.md)

局部双侧权重旋转最小测试显示某些NMSE改善，但32维分块没有稳定胜过单侧；去掉保护通道后4096变4064，完整Hadamard不再自动可用。[旋转实现与验证](../records/docs/exp_results/weight_rotation_preprocess.md)

## 下次怎么做

- 先固定类别、stage、向量维度、保护通道和训练预算；报告payload位宽及包含decoder/dense/norm/head等开销的实际存储。
- 单类压缩加完整模型评测，仍不等于所有类别同时压缩。按类挑出的组合要验证最终全压状态。
- PPL与下游指标冲突时分别报告，事先确定主指标，不事后合成有利分数。
- 旋转后核对坐标、通道保护顺序、矩阵整除及Hadamard支持条件；原坐标AMSE权重不能不经推导直接用于新坐标。
- 重构NMSE有收益只能作为进一步下游实验的依据，不能省略后者。

## 边界

码本结论来自旧配置下的14个单类job；不能迁移为所有模型/数据的默认最优配方。旋转记录为有限种子及预算的重构验证，不是全模型质量或速度结论。

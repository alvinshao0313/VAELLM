# 编码恢复：先验证阈值可达性和真实数值路径

## 问题与证据

旧0/1连续代理离阈值0.5有0.5距离。在记录限定的 fresh AdamW、wd0、3968步和lr≤2e-5条件下，累计位移上界低于0.5；因此“梯度非零、代理更新”不能证明硬码能改变。[设计推导](../records/experiments/liftquant_recovery/RECOVERY_DESIGN_20260923.md)

采用 p=s(b−1/2) 后，真实decoder敏感度提供s，STE包含1/s及clamp门控。16步两block测试仍没有翻码，MSE下降来自decoder；标量量化器的跨阈值回归只证明机制可达。[修复及冒烟记录](../records/experiments/liftquant_recovery/PROXY_FIX_20260923.md)

## 下次怎么做

1. 把代理坐标、阈值距离、实际学习率、调度序列和累计位移预算写进设计；检查完整预算内是否可能改变实际离散状态。
2. 不把“放大梯度”视为Adam下必然加大更新，也不把0/1平移误认为阈值变近。改坐标必须一起定义硬前向、链式梯度和饱和边界。
3. 用前向实际使用的BF16舍入权重做VJP参考；packed与dense不同舍入路径不得混作严格同一参考。分别核对前向、编码梯度和全部decoder梯度。
4. 同时记录代理位移、每步/累计bit变化、实际权重变化、训练/留出MSE、冻结hash与strict重载。各指标回答不同问题。
5. 旧decoder一步更新导致MSE上涨后，固定更新方向的小步插值支持先试较小步长；1.25e-5是有局部依据的候选，不是通用最佳LR。
6. 参考其他方法时区分“相同优化任务/数据/预算”和“表示特有适配”。该VAE实现不等价于官方FWT，不能因沿用名字就宣称严格复现。

## 边界

384步检查已于2026-09-23核实完成，具体证据见下节；归档时的进行中表述是历史状态。两个block、极少token和1题PIQA不能证明恢复质量；当前初始模型含192个压缩Linear及60个dense Linear，不能称全模型2-bit。正式全量、跨方法比较仍需匹配预算和下游评测。[历史审查](../records/experiments/liftquant_recovery/DESIGN_AUDIT_20260923.md)

## 384步真实翻码：机制可用不等于恢复有效

**证据**：单block9、2条训练/2条留出、每条64token，首次硬码翻转在第92步；384步后相对初始改变约1.374%的码位。训练MSE下降99.84%，留出MSE却增加7.99%，冻结和strict重载检查通过。[完整结果及原始指标索引](../records/experiments/liftquant_recovery/PROXY_FIX_20260923.md#2026-09-23384步检查已完成并核验)

**已证实与解释**：敏感度坐标确实允许真实梯度推动翻码；两条训练样本反复优化有明显过拟合表现，但未做同预算decoder-only对照，不能独立识别编码更新对留出恶化的贡献。

**下次动作**：机制门槛与质量门槛分别判断，不能用训练/留出混合MSE下降代替留出改善；增加有代表性的独立校准/验证数据后，使用受控配置检查泛化，再考虑正式训练。这里是后续设计原则，不是新实验启动授权。

**边界**：本次极小样本只验证真实翻码和完整保存链路，没有下游提升证据，也没有证明达到LiftQuant或超过它。

## 同预算对照能区分翻码收益与单纯拟合

**问题**：非零翻码和训练loss大幅下降都不足以证明编码学习比decoder-only更值得。

**证据**：后续单block9、16训练/16文档分离留出×128token、每臂384步的对照中，初始输出及decoder LR一致，前107步loss完全相同。joint相对decoder-only的留出block MSE下降11.48%（16/16改善），完整student留出NLL下降1.85%（12/16改善，配对差区间不跨0）。[配置、精确结果与原始指标](../records/experiments/liftquant_recovery/BIT_CONTRIBUTION_20260923.md)

**结论边界**：当前受控条件支持“允许编码学习优于仅decoder”，但joint相对原始初始化的NLL均值改善约1.01%，区间跨0；不能据此说稳定超过初始模型或提升正式下游分数。固定joint decoder还原初始码后block MSE全部恶化，NLL均值恶化但区间跨0，所以直接贡献在block层面更明确。该结论补充早前2训练样本的过拟合记录，不将不同数据条件混作因果对照。

**下次动作**：保留独立document划分、相同数值路径、固定终点、实际LR和首次翻码前一致性检查。分别报告“相对decoder-only”“相对初始模型”“固定decoder还原码”三个比较；配对bootstrap以文档为单位，不把token数量当独立验证量。更广泛的结论仍需有代表性的数据和跨seed/下游验证，本条不自动启动新实验。

## 工程提速先锁定数值轨迹（2026-09-23）

**证据状态**：在Qwen3-8B现有单stage无旋转checkpoint、block9、2训练/2留出×64token、33步条件下，合并诊断同步、每32步详细审计及单遍端点评测后，新旧每步loss/LR/翻码计数、最终FP32代理/梯度/Adam/decoder状态与native输出严格相等。14个真实decoder的独立梯度参考和7种requires_grad分支检查通过。[本轮工程验证与原始计时](../records/experiments/liftquant_recovery/2026-09-23_engineering_equivalence.md)

**可复用动作**：保留每步硬码投影和finite检查，只降低非训练所需详细统计的频率；在GPU汇总后一次传回CPU。跳过无需求梯度时检查decoder-only/code-only等分支，不把联合训练的数学一并改动。大型CPU激活预分配消除拼接副本，但不降低训练期间输入/目标本身的需求。

**缓存边界**：修改或加载persistent码后必须同步grouped packed与decoded缓存。首轮测试helper只恢复state_dict而漏grouped缓存，导致before-MSE比较失败；补齐缓存还原后原严格回归通过。仅比较state_dict不足以证明后续前向一致。

**适用边界**：小规模轨迹等价说明工程改动没有改变该配置的算法结果；不等于证明正式下游收益。交错组件计时与单次端到端时间分开报告，不把短序列倍率外推到2048token。新的投影/VJP核、低精度optimizer或不同LR均需另做数值与效果检查，本轮未启用。

## 全驻留评测可以保留原 packed 计算路径（2026-09-24）

**证据状态**：Qwen3-8B本次初始v6 checkpoint、192个压缩Linear、真实2×2048输入下，模型全驻留与逐层搬运的BF16 logits逐值一致，NLL差0、argmax一致率100%，192个已预热缓存指针未替换。两份恢复短测成品的八任务limit1入口均通过。完整训练形状batch2×2048的partial/full block前后向、保存重载已验证，allocated峰值约8.01GiB；全模型驻留验证约19.51GiB。[配置与原始证据](../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**可复用动作**：显存足够时先将native模型整体搬到GPU，再按原packed-u8 BF16路径一次预热权重缓存，沿用同一run_lm_eval。现成模型传给HFLM时必须确保model.device及embedding已在GPU；不把默认whole-decoder fused的旧分数混作新路径基线。受控多候选比较只需同路径重评初始基线一次，各候选评一次，避免重复搬运和重复基线计算。

**适用边界**：本次没有改解码精度、loss或评测口径；单批耗时6.65s对2.30s含驻留准备，不能外推为全量加速倍数。短测未跨码阈值不推翻此前384步翻码证据，也不证明正式下游改善。正式4096样本的CPU input+target仍需每任务约128GiB，应按并发任务总量核算，GPU峰值不能代替主机内存预算。


## 完整逐层恢复的下游收益（2026-09-24完成，2026-09-30核实）

**本配置实测支持**：Qwen3-8B无旋转、单stage VAE初始化，敏感度码坐标、code+decoder联合优化、FP teacher前缀对齐、4096×2048校准、每层3968步，28层恢复后，同路径完整八任务均值由39.90%升至59.38%或59.10%。冻结状态与严格native重载均通过。[完整配置与原始结果](../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**可复用结论与边界**：判断恢复是否有效应看同路径的初始化/成品全量下游对照；本次支持该整体流程的收益，不单独证明翻码的因果贡献。两档decoder LR只差约0.27个百分点且仅单seed，不据此认定稳定排名；未达到69%，未与官方LiftQuant同条件对比，不能声称超过官方。此前短测只验证链路的边界仍然有效。


## 迁移到 Sparse Bit 时同时处理坐标、轮次与恢复（2026-09-30）

**问题与证据**：原 Sparse Bit 用 ±1 score 和0.02/0.05默认LR，本来能翻码，不能沿用旧逐层恢复0/1代理冻结的判断。新增敏感度模式在真实packed计算、三种优化器、换轮/offload、Trainer精确续训检查中通过；固定小矩阵在2e-5下第9步翻码且目标MSE下降，原坐标12步不变。[实现、验证及边界](../records/experiments/liftquant_recovery/2026-09-30_sparse_bit_coordinates.md)

**可复用做法**：迁移时一起实现 p=s(b−0.5)、链式梯度1/s、与原训练器一致的限幅、FP32代理和固定尺度保存恢复。每轮重选active集合不等于可以重算s；精确续训须用初始保存值，不能用已更新decoder替代。新坐标要求显式LR，每轮更新步数仍决定阈值可达性。AMP计算dtype与decoder参数dtype分开处理，抽样校准不得消耗训练RNG或改变offload驻留。

**范围**：标定使用初始单bank完整decoder的eval输出；grouped抽取产生的正常低精度差异不要求逐位相同。新模式复用敏感度公式，保留Sparse Bit采样和优化器规则，不是复制完整LiftQuant训练法。小矩阵收益只证实机制，不构成Sparse Bit大模型下游提升证据；原unit默认及旧断点语义保留。

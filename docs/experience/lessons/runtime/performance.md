# 并行、算子、评测与吞吐

[经验分类入口](../../README.md) · 一级分类：数值与运行性能经验（`runtime`） · 二级专题：`performance`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [工程提速先锁定数值轨迹（2026-09-23）](#liftquant-recovery--工程提速先锁定数值轨迹2026-09-23)
- [全驻留评测可以保留原 packed 计算路径（2026-09-24）](#liftquant-recovery--全驻留评测可以保留原-packed-计算路径2026-09-24)
- [合批加速与缓存精度的数值对照](#memory-precision--合批加速与缓存精度的数值对照)
- [独立目标分组拟合的效率边界](#memory-precision--独立目标分组拟合的效率边界)
- [评测阶段耗时与离线数据访问](#memory-precision--评测阶段耗时与离线数据访问)

<a id="liftquant-recovery--工程提速先锁定数值轨迹2026-09-23"></a>

## 工程提速先锁定数值轨迹（2026-09-23）

原专题：[编码恢复：先验证阈值可达性和真实数值路径](../liftquant_recovery.md#工程提速先锁定数值轨迹2026-09-23)。

**证据状态**：在Qwen3-8B现有单stage无旋转checkpoint、block9、2训练/2留出×64token、33步条件下，合并诊断同步、每32步详细审计及单遍端点评测后，新旧每步loss/LR/翻码计数、最终FP32代理/梯度/Adam/decoder状态与native输出严格相等。14个真实decoder的独立梯度参考和7种requires_grad分支检查通过。[本轮工程验证与原始计时](../../records/experiments/liftquant_recovery/2026-09-23_engineering_equivalence.md)

**可复用动作**：保留每步硬码投影和finite检查，只降低非训练所需详细统计的频率；在GPU汇总后一次传回CPU。跳过无需求梯度时检查decoder-only/code-only等分支，不把联合训练的数学一并改动。大型CPU激活预分配消除拼接副本，但不降低训练期间输入/目标本身的需求。

**缓存边界**：修改或加载persistent码后必须同步grouped packed与decoded缓存。首轮测试helper只恢复state_dict而漏grouped缓存，导致before-MSE比较失败；补齐缓存还原后原严格回归通过。仅比较state_dict不足以证明后续前向一致。

**适用边界**：小规模轨迹等价说明工程改动没有改变该配置的算法结果；不等于证明正式下游收益。交错组件计时与单次端到端时间分开报告，不把短序列倍率外推到2048token。新的投影/VJP核、低精度optimizer或不同LR均需另做数值与效果检查，本轮未启用。

<a id="liftquant-recovery--全驻留评测可以保留原-packed-计算路径2026-09-24"></a>

## 全驻留评测可以保留原 packed 计算路径（2026-09-24）

原专题：[编码恢复：先验证阈值可达性和真实数值路径](../liftquant_recovery.md#全驻留评测可以保留原-packed-计算路径2026-09-24)。

**证据状态**：Qwen3-8B本次初始v6 checkpoint、192个压缩Linear、真实2×2048输入下，模型全驻留与逐层搬运的BF16 logits逐值一致，NLL差0、argmax一致率100%，192个已预热缓存指针未替换。两份恢复短测成品的八任务limit1入口均通过。完整训练形状batch2×2048的partial/full block前后向、保存重载已验证，allocated峰值约8.01GiB；全模型驻留验证约19.51GiB。[配置与原始证据](../../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**可复用动作**：显存足够时先将native模型整体搬到GPU，再按原packed-u8 BF16路径一次预热权重缓存，沿用同一run_lm_eval。现成模型传给HFLM时必须确保model.device及embedding已在GPU；不把默认whole-decoder fused的旧分数混作新路径基线。受控多候选比较只需同路径重评初始基线一次，各候选评一次，避免重复搬运和重复基线计算。

**适用边界**：本次没有改解码精度、loss或评测口径；单批耗时6.65s对2.30s含驻留准备，不能外推为全量加速倍数。短测未跨码阈值不推翻此前384步翻码证据，也不证明正式下游改善。正式4096样本的CPU input+target仍需每任务约128GiB，应按并发任务总量核算，GPU峰值不能代替主机内存预算。

<a id="memory-precision--合批加速与缓存精度的数值对照"></a>

## 合批加速与缓存精度的数值对照

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#合批加速与缓存精度的数值对照)。

**取样提速先保留原随机流，再减少逐行数据组织。** Qwen BF16 up三档新入口固定CPU取样，实测约.125–.130秒／VAE更新，45分钟三条正式链均未完成S2×10K。成熟GPU驻留路径虽较快，但本机PyTorch2.6的真实CPU索引对照表明，同seed下两路径跨epoch顺序不同：DataLoader迭代的base_seed及RandomSampler的空tail排列会消费generator，驻留路径每epoch仅一次randperm；eval的global RNG消费也不同。两者均无放回且保留尾批，不等于同轨迹，不能只改驻留flag便宣称等价。

针对逐8192行`__getitem__`／collate开销，[批量读取原型](../../../../experiments/cat_batch_gather_cmcc_20261104/README.md)保留原factory产生的batch_sampler和generator，仅用批量`index_select`及已成批collate。实际CPU五case包含BF16 M36/B8192/D32、跨3epoch、尾批、None seed、非连续源及train/eval交错，每批值／索引／layout与全局／私有RNG状态exact。两轮16384／16401行gather分别.224802→.011555秒、.236559→.013221秒，限CPU数据组织。隔离fit入口已绑定新源码并通过实际tiny Qwen3 S2×2：state、packed bank、初始化／seed、实际更新与host RNG exact，原formal资格检查保持。独立有限runner已路由新fit，8.106735秒CPU入口检查确认实际import／CLI、原namespace不变及旧身份拒绝；原search及在途源码未改。随后本批K32／40／48三条真实Qwen3-8B S2各2更新／原10K日程、M36／B8192及完整512路径均正常exit0，保存／独立fresh的state与同输入BF16 logits exact；原source9身份与GPU技术链已核，各自接fresh BF16正式，三正式已各推进到第一阶1000步，K32／40／48日志步时.0372／.0406／.0442秒；旧K32同1000位置.1219秒，新旧迭代耗时比约3.28，仅为同配置窗口。随后三正式S2各10K及编码／读出／保存／独立fresh／正式adapter全exit0，完整formal worker1152.259／1182.916／1238.636秒，整批含技术与协调1.352275GPUh；实际formal峰allocated约23.56／23.79／24.02GB，含整模型过程，不作纯VAE峰。旧批未完整结束，不能推导完整全链加速比，任务质量仍待；不用2步耗时外推10K。固定方法是保持样本、RNG、数值及调用时序后优化组织；各模型重算N/M/D和内存，batch与驻留方式须依实际吞吐／峰值有界选择，不推广本次8192为最优。后续同模型／来源／已覆盖资源路径复用该技术证据，关键形状或执行路径变化才补必要短验；不把CPU结果或旧驻留吞吐用于新的正式限时估算。

- 2026-10-11的七任务源输出保留短测表明，缓存教师的纯前向耗时不能估算live decoder训练成本：每步160条原生候选逐条反向，首步1759.37秒；保持样本、因果query、逐token/branch/doc/task权重和一次统一clip，按task合为7批后，两步为444.64/428.96秒。MQA仍走原逐题路径；首步总梯度范数相对差8.09e-5，代表BoolQ/Winogrande/PIQA完整分布mean TV最大.01390、mean KL最大.001122，2775冻结张量保持且两步均有非零保留梯度。合批须显式处理left padding/position_ids和各行续写位置，不能仅把输入拼接；本次峰allocated18.901GB、reserved72.117GB分别记录，不把初始化保留容量当作训练驻留。该证据支持执行加速，不证明九项质量提升或任意batch数值等价。[两更新报告与实际启动记录](../../history/w25_transfer/2026-10-04_29_research_log.md#10月11日-train-only-八组完整结果与下一轮覆盖对照)。

- 同日继续把M32短题合批，仅末query的原四选项KL，真实最大32×958输入（30656 padded tokens）通过两更新及冻结检查。相同group的M前后向281.880→16.995秒，但M-only梯度相对差6.56%、首次Adam更新向量相对差23.62%，即使32题mean TV仅.00358且top1不变，也不能把B1质量证据直接移植。后一个差值的分母是微小更新，绝非模型权重/准确率差；首Adam逐坐标归一化可能放大小梯度方向变化，尚未逐坐标证明。比较应单独保留各域梯度及真实optimizer更新，避免QA等共用梯度掩盖差异；无语义/状态异常且有明显吞吐收益时，用唯一matched正式预算/完整质量对照裁决即可，不无限追加低层误差诊断。参考总梯度可复用同一次QA/R并替换M梯度，但须声明FP32累积顺序不同，不能冒称串行全链逐位复现。峰allocated28.844GB，第二步总164.162秒；跨group步时不作为严格整体加速比。同日完整matched32结果随后证实正式更新时间5241.647对13726.882秒、加速2.618811倍；M32九项比B1高.083316pp，但M−.028486、LB−.095992，收益来自其他七项。这个同预算结果支持本轮后续采用M32，不等同逐位等价或一般泛化优势。[完整数值、资源及单一后续对照](../../history/w25_transfer/2026-10-04_29_research_log.md#10月11日-train-only-八组完整结果与下一轮覆盖对照)。

<a id="memory-precision--独立目标分组拟合的效率边界"></a>

## 独立目标分组拟合的效率边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#独立目标分组拟合的效率边界)。

**条件与固定方法。** 对同D/K、encoder／decoder结构、dtype、有效batch及LR时域的独立层，合批矩阵计算本身不要求共享decoder；本项目[IndependentGroupedVAE](../../../../experiments/adaptive_grouped_fit_20261029/grouped.py)打包各层参数，但每个原BSQ只读取自己的`[B,1,K]`，各层重建／辅助损失求和。若把层轴与样本轴一起归约熵，就改变了算法目标。各层保持原shuffle／游标和Adam状态；同组阶段依赖仍串行，不能将后阶所需残差提前替换为旧值。现有[整类拟合器](../../../../experiments/adaptive_class_20261030/fit.py)也明确按该独立目标求和，尾批形状不一致即拒绝，没有填充重采样。

**状态更新边界。** 不同层更新数／LR日程应分组，不能只把已完成层的梯度置零而假称冻结。PyTorch 2.6的[optimizer源码说明](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/optimizer.py)区分零梯度与`None`；[AdamW源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py)跳过`grad is None`的整个参数，否则仍可受已有动量及decoupled decay影响。打包参数的一个切片无法单独设为`None`，所以不同生命周期不能仅靠零梯度掩码等价处理。这里只解释源码契约，不宣称本项目新增了异步分组实现。

**已有证据及测量方法。** [原配对正文](../../../../experiments/adaptive_grouped_fit_20261029/README.md#10月29日完整配对结果)记录Qwen down L18／L35、D32/K48/H128/R0、各6000更新的完整流程由234.97529077秒变为199.53393471秒，即1.17762069倍、减时15.0830%；L18局部误差基本保持，L35留出输出NMSE增约0.15708%。计时包含准备、拟合、编码、分配、保存／重载与局部读出，且串行有重复MSE统计，不能全归因于GEMM，也不是速度／质量逐位等价。训练段allocated峰不等于全生命周期峰。2026-10-31本轮可复核源码和正文；旧`comparison.json`及两臂计时原件在已查原路径、迁移映射和清单中缺失，以上数值保留为历史记录证据，不能标成本机原始指标复核通过。

**按模型重算与调节范围。** 重算每层真实N、D/K/H、归一化、尾批、活跃码行及部署开销，独立统计更新／编码／路由／导出时间和allocated／reserved峰。组规模由目标形状与实测峰值限制，本条没有通用最优组数；已有M2数据不能证明M36、多卡或整模型恢复加速。少解码31.1%的码行相对的是全三阶，对固定两阶反而可能更多，也不能换算为实际提速。active-only拟合、共享decoder、周期重分配都改变算法条件，不能作为等价并行优化的经验。本轮仅完成资料核对，没有新增性能测量。

<a id="memory-precision--评测阶段耗时与离线数据访问"></a>

## 评测阶段耗时与离线数据访问

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#评测阶段耗时与离线数据访问)。

- 完整评测墙钟变慢时，先拆加载、预热、实际请求循环和循环外时间，再决定是否检查GPU算子。2026-10-04的Qwen普通LoRA400与rank0 CE5200同MMLU64/14042题，日志阶段664/296秒，但请求循环172/192秒；CE的auto batch探测占首批额外时间，扣除后约164/158秒。252模块预热均约2秒且缓存实际生效，没有反复解码的证据。LoRA未设HF离线变量、CE设置为1，但现有日志不足以将额外时间唯一归因网络；不能据总耗时指控LoRA forward退化，也不能反向宣称LoRA更快。[实际时间、环境与代码依据](../../history/w25_transfer/2026-10-04_29_research_log.md#无外置lora的长文本与norm恢复对照)。
- 本地评测数据已完整核验时，新实验显式复用HF离线缓存可避免无关元数据请求中断回调。同日OV8K400的seven及LoRA1200的MMLU分别在固定`super_glue`/`hails/mmlu_no_train`版本加载时发生ProxyError；已有模型导出与审计成功。只补失败的完整分片、设置三项HF离线变量后均通过，原失败终态保留。这证实两次中断来自数据访问，不追溯证明上条早期慢评测也是同一原因；不得用其他数据版本或缩减题数绕过失败。回调导致训练退出时，还须从原生FP32/optimizer/scheduler/RNG状态恢复原预算，不能从BF16推理成品假称exact-resume。[补评及恢复证据](../../history/w25_transfer/2026-10-04_29_research_log.md#qwen-norm长检索信号及后续诊断)。


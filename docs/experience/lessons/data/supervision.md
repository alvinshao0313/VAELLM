# 监督位置、模板、归约与日志语义

[经验分类入口](../../README.md) · 一级分类：数据与监督经验（`data`） · 二级专题：`supervision`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [KD 与数据边界：同名参数可能已有不同数学语义](#kd-and-data--kd-与数据边界同名参数可能已有不同数学语义)
- [按条件选择动作](#kd-and-data--按条件选择动作)
- [下次怎么做](#kd-and-data--下次怎么做)
- [长文本SFT须核对截断后的真实监督（2026-10-04）](#kd-and-data--长文本sft须核对截断后的真实监督2026-10-04)
- [问题与证据](#kd-and-data--问题与证据)
- [GAS归约和真实数据消费](#kd-and-data--gas归约和真实数据消费)
- [历史prompt公式与causal边界](#kd-and-data--历史prompt公式与causal边界)
- [边界](#kd-and-data--边界)
- [2026-09-24 当前实现核查](#kd-and-data--2026-09-24-当前实现核查)
- [当前loss公式、日志窗口与完整CE配方](#kd-and-data--当前loss公式日志窗口与完整ce配方)
- [生产路径重放token归因与模板审计（2026-09-24）](#kd-and-data--生产路径重放token归因与模板审计2026-09-24)
- [数据修复后的短程收益与去包装对照（2026-09-24）](#kd-and-data--数据修复后的短程收益与去包装对照2026-09-24)
- [answer-only监督、归约分母与联合码位对照](#kd-and-data--answer-only监督归约分母与联合码位对照)

<a id="kd-and-data--kd-与数据边界同名参数可能已有不同数学语义"></a>

## KD 与数据边界：同名参数可能已有不同数学语义

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#kd-与数据边界同名参数可能已有不同数学语义)。

<a id="kd-and-data--按条件选择动作"></a>

## 按条件选择动作

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#按条件选择动作)。

| 条件／要解决的问题 | 建议动作 | 证据状态与适用边界 |
| --- | --- | --- |
| loss、prompt权重或GAS语义不清 | 先核当前公式、causal位置和实际归约 | [下次怎么做](#kd-and-data--下次怎么做) |
| 怀疑softmax数值或缓存目标 | 固定原缓存与真实路径，按同目标高精度参照诊断 | [Softmax诊断须保留缓存目标与真实前向语义](../runtime/precision.md#kd-and-data--softmax诊断须保留缓存目标与真实前向语义2026-10-29) |
| 换数据、split或长文本长度 | 查真实文件与原生消费，统计截断后有效监督 | [长文本SFT须核对截断后的真实监督](#kd-and-data--长文本sft须核对截断后的真实监督2026-10-04) |
| 覆盖、教师或保护训练出现小收益 | 保持同父和预算，报告完整净收益及能力代价 | [同父恢复须同时看完整净收益与能力代价](../post_recovery/objectives.md#kd-and-data--同父恢复须同时看完整净收益与能力代价2026-10-30) |
| 判断Muon是否仍多任务或是否保能力 | 分开实际任务混合、优化器变换和完整能力结果 | [Muon、PCGrad和同期Adam控制](../post_recovery/optimization.md#kd-and-data--muonpcgrad和同期adam控制) |
| 新父拟接已有保持项 | 核当前梯度路径、保持缓存身份及导出约束，先处理真实适配缺口 | [冻结参数不保证能力保持](../post_recovery/parameters.md#kd-and-data--冻结参数不保证能力保持)；当前Compact／Rice结论仅为源码核查 |
| KL或答案NLL改善但任务变差 | 分开答案、格式、修复/损坏及自由生成 | [短答案的标签拟合与教师分布可能不一致](teachers.md#kd-and-data--短答案的标签拟合与教师分布可能不一致2026-10-04) |
| norm、head或追加预算 | 复用匹配完整轨迹，不把旧范围负例泛化 | [norm、head与追加蒸馏的预算边界](../post_recovery/parameters.md#kd-and-data--normhead与追加蒸馏的预算边界) |
| 跨模型或新来源迁移 | 重编码原始文本、重算教师缓存并核真实token曝光 | [更换任务数据须同时核对实际监督量与采样路径](sources.md#kd-and-data--更换任务数据须同时核对实际监督量与采样路径2026-10-04) |

以下按问题查动作；各节保留原核查日期、实测条件、未验证声明及原始来源，历史记录不代表当前运行状态。

<a id="kd-and-data--下次怎么做"></a>

## 下次怎么做

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#下次怎么做)。

- 明确目标位置到logits的单次shift，最后位置、padding及prompt/response边界用具体token例子校验；避免双重shift。
- 改prompt权重时先写数学定义，确认是区域均值组合还是token加权均值。历史同名CLI不保证同样公式。
- CE+KD组合中CE只加入一次；hidden/pre-MLP的mask不能因KD改动而无意改变。
- dense/offload路径比较标量和学生梯度；空区域应有明确语义，不能用任意默认值掩盖问题。
- token统计按学生micro-batch计一次，不能因teacher、重计算或CPU staging重复计数；恢复后的首个日志窗口按实际观察步数计算。
- 历史错误mask训练得到的checkpoint不能当作“干净对照起点”；以新设计明确是否从同一原始模型重训。

<a id="kd-and-data--长文本sft须核对截断后的真实监督2026-10-04"></a>

## 长文本SFT须核对截断后的真实监督（2026-10-04）

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#长文本sft须核对截断后的真实监督2026-10-04)。

2026-10-13补充原始训练集边界：LongBench v1发布本体只有test，但其部分原始任务另有train；使用原始train不等于可训练LongBench评测样本。NarrativeQA实际train的1102故事／32747问，完整原生prompt只有119故事／3574问≤32640，其中78故事／2354问在16–32K，其余长文不能仅因数据集名或作者词数就当作窗口内训练供给。官方旧HF reader统一latin-1而原件存在明确UTF8／UTF16 BOM；本次只按BOM修正读取、其余显式继承上游约定，保留原HTML与全部train样本，不能把所得长度称净正文长度或声称已确证所有作者编码。实际Qwen tokenizer还包含NFC，分词库存优化必须保留其真实预处理和边界语义，不能默认normalizer=None。上述均是来源／输入与长度实测，不证明长文恢复收益。[固定来源、完整库存和读取边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月13日longbench原始任务训练集与narrativeqa全文准备)。

现有canonical SFT保留前`max_length-1`个token并补EOS；如果截断后没有真实assistant payload，则跳过整条，不搬移末尾答案。LongAlign/LongAlpaca各固定随机64例CPU实测，Qwen在1K/8K/16K下可用数分别为0/5/39与21/37/55。配置8K不等于每批都是8K，也不表示原始长短比例会原样进入训练；这些是抽样计数，正式曝光须以native实际token记录为准。资源短验可单独选择真实满长且仍有监督的代表样本，但不把其选择规则带入正式流或用短验评价质量。

本轮还发现LongAlign4条训练记录含8条LongBench验收记录的完整文档，不等于相同问答；因此该候选暂不使用。LongAlpaca全12000条的有界全文匹配为0，仍不构成无污染证明，官方原始记录也缺少完整来源追溯。保留固定版本、原字段和既有模板，不以自动删除若干重叠样本绕过未批准的数据规则变化。[检查方法、来源、真实资源样本及边界](../../history/w25_transfer/2026-10-04_29_research_log.md#原生长文本数据检查与8k资源准备)。

<a id="kd-and-data--问题与证据"></a>

## 问题与证据

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#问题与证据)。

<a id="kd-and-data--gas归约和真实数据消费"></a>

## GAS归约和真实数据消费

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#gas归约和真实数据消费)。

2026-10-15从CAT GAS1迁移双卡GAS2时，须检查HF实际损失缩放，不能只核CLI数字。`CustomSFTTrainer`返回已归一的自定义KD均值、忽略`num_items_in_batch`；若HF从Qwen3的kwargs推断`model_accepts_loss_kwargs=True`，Trainer会跳过累积除数。新隔离CAT快照复用E2E已有声明，将该标志设为False；真实短验两rank各每更新两个微批、8192有效输入，非零LR下参数实际更新。DDP平均、微批均值与旧B4的token加权均值仍有计量差异，即使global batch同为4，也不能声称完全同一目标归约。此修正限于已确认的自定义loss路径，不根据参数名字推断任意Trainer都要重复除GAS。[新鲜配对运行记录](../../history/w25_transfer/2026-10-04_29_research_log.md#10月15日下午先准备广覆盖校准与蒸馏数据)。

2026-10-15广覆盖数据准备补充：池规模不等于逐类实际知识曝光。完整Web/Med/Qasper/Hotpot新增504680条train，但当前原生混合加载器以seed31、16 worker、4K/B4消费500批，只有2000个唯一有效样本；重建下一类别迭代器会完整重放同一身份序列。配置Qasper2%时为获得2000有效例实际读47条Qasper、28条因末尾答案截空跳过，最终只占19/2000。Web实际685条中法律0，而全池明确有452条；不能由单次未见认定全池缺席，也不能由大池存在推断训练已经覆盖。上述是当前CPU原生路径实测，旧GPU训练没有逐步身份遥测。后续先核源池、截断后有效例、token与类别间唯一曝光，再定数据/长度/预算；不自动引入新采样规则或把数据准备通过当作九项增益。[全量来源与真实消费证据](../../history/w25_transfer/2026-10-04_29_research_log.md#10月15日下午先准备广覆盖校准与蒸馏数据)。

2026-10-18补充：外层group循环次数不等于各域单例曝光次数。Llama原32组每组M32／QA8，只有M1024／QA128个唯一身份，因此一轮内M各一次、QA各两次；64／128更新分别使QA各曝光四／八次。覆盖对照固定64更新只将QA128扩至512，M1024仍各两次，QA改为各一次；必须沿实际ID计数并保持域时序，不能把原128步称为全部样本只重复四次，或把不同唯一数称为等token。重复池退化与扩池后的收益上限分开判断，本次编码／缓存检查通过尚不是质量收益。[隔离扩池与原始曝光证据](../../../../result/qwen3_8b_mmlu65_20261004/llama_fwbest_tf_qa512_64_20261018/README.md)。

<a id="kd-and-data--历史prompt公式与causal边界"></a>

## 历史prompt公式与causal边界

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#历史prompt公式与causal边界)。

早期prompt KD方案采用所有token的加权均值，后续改成响应区均值加权组合：L_response + w·L_prompt。两份历史审查都通过过各自版本，不能同时当成当前公式。[后续区域归一化审查](../../history/reviews/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/final-review.md)

causal KD曾存在一位错位问题：位置t的logits对应t+1目标。硬件审查又发现Qwen3多轮ChatML无法依赖独立prefix tokenization定位真实响应边界。[mask修复计划](../../history/plans/2026-08-04-fix-causal-kd-token-mask-alignment.md)、[硬件证据](../../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

<a id="kd-and-data--边界"></a>

## 边界

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#边界)。

来源是历史计划、审查和有限硬件验证。本次未核验当前所有loss实现；CLI与支持的loss以当前源码及根README为准。审查通过不证明某个prompt权重能提高下游成绩。

<a id="kd-and-data--2026-09-24-当前实现核查"></a>

## 2026-09-24 当前实现核查

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#2026-09-24-当前实现核查)。

<a id="kd-and-data--当前loss公式日志窗口与完整ce配方"></a>

## 当前loss公式、日志窗口与完整CE配方

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#当前loss公式日志窗口与完整ce配方)。

当前 `train_utils/distill_loss_core.py` 的 prompt reduction 是所有有效token的加权均值 `(sum_response + w*sum_prompt)/(N_response + w*N_prompt)`，不是上文历史审查中的区域均值组合。`kl_top_partial` 是纯KD，`alpha`仅在 `kd*` 的CE/KD组合中生效；不能因命令含 `alpha=0.5` 就认为训练加入了真实答案CE。`kl_top_mass` 在teacher top-K之外加入tail桶，`kl_top`则是top-K条件分布。以上为当前源码核查，不是不同loss精度优劣的实测。[本轮设计与后续证据](../../records/e2e_0910/2026-09-24_rank8_search.md)

2026-10-03补核当前E2E日志：`trainer._store_loss_parts`中的`distill_loss`是本次实际loss函数的完整返回值；`kd_top_mass`时包含`(1-alpha)*CE + alpha*KL`，不是纯KL，也未分别记录两项。该字段来自最后一次micro-batch，Trainer的`loss`则为日志窗口均值，不能直接相减求CE或把两者差异当异常。当前Qwen3 CE10与CE50/task100/response-only在前800步均训练3200条，但平均每条原始label统计的response token约143.28与11.26，混合目标权重、数据和mask不同，loss绝对值不可跨组直接排序；token数也不是梯度贡献。[实际日志及完整指标](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)。

同日这两组完成2000步和每400步完整八任务验收：49e来源66.57623885%，CE10最终66.44208228%，CE50/task100/response-only则从首400的60.38795485%逐点升至66.89468320%，成为当时总体最佳；相同2.456988791bpw/无外置LoRA，冻结硬码与非decoder、3423保存/strict及固定实际64/32通过。首400大幅下降不能代替本例最终结果，也不授权无依据延长任何失败配方。CE50最终五项上升，但MMLU较49e低3.72454066个百分点、OBQA/PIQA亦下降，平均提升不代表全面改善；CE比例、数据及prompt mask联合改变，不能单独归因某一项，更不能把此正确训练数据/decoder-only/2000步结果与旧rank8/800样本实验混作同一对照。[十个完整保存点及逐项证据](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的CE10/CE50最终结果。

随后在这两个相同硬码/部署规格的端点间使用现有类别组装，CE50 attention+CE10 MLP完整均分67.13033275%，高于双方来源；反向互补仅66.65448072%。两者实际2.456988791bpw，3423保存/fresh strict与完整固定64/32通过，没有重训或增加分支。前者MMLU较CE50恢复299题，但仍低于CE10，且RTE下降；平均提升不代表所有任务改善。共享1911非decoder张量与504硬码冻结是这次比较的前提，不同几何/不同硬码来源的旧组合负结果不构成此新条件已经失败，来源指标也不能相加预测组合质量。该单次组合收益仍不能拆分CE、数据、prompt或各子类别的因果贡献。[来源审计、完整指标及细分对照](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的08:54同硬码decoder组合。

<a id="kd-and-data--生产路径重放token归因与模板审计2026-09-24"></a>

## 生产路径重放token归因与模板审计（2026-09-24）

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#生产路径重放token归因与模板审计2026-09-24)。

**本配置实测支持、范围有限。** 新数据J/L的CPU重放调用真实混合流、permutation、canonical encoder及collator，保持实际16 workers与B4；J首40条和两组前800条的20个日志窗口全匹配。不能按运行前normalized的workers=0重放：入口后续将其改为16，iterable路径也会关闭group_by_length。重放必须覆盖真实跳过无有效回答样本的逻辑，raw draw比例不能替代编码后训练样本。1600条是同配置确定性重放，后800在审计时尚未获实际日志逐窗覆盖，不能写为1600条都已实际训练。[统计原件、源码位置与范围](../../records/e2e_0910/2026-09-24_rank8_search.md)

**观察与动作。** 已匹配前缀中，task样本权重0.5/0.8对应累计因果token分母份额仅约21.82%/48.88%；MMLU为18.61%/41.67%。长回答语料能显著改变样本权重在token归约中的份额，所以调整混合比例前先量化source级有效prompt/response，不能把task0.8当成优化贡献80%。本观察不是提高task权重有效的下游证明，也不授权改已运行配置。

**模板与内容分开。** MMLU auxiliary_train无subject时不能为凑评估description杜撰学科；Winogrande train的606条context差异经逐条比对仅为空格，目标与续写内容均正确，不能当作标签错配。ChatML空think与terminal EOS使已检RTE/MMLU的response目标包括包装token，不等于答案语义token数。格式差异的存在有源码/实际数据证据，对齐是否有收益仍待受控实验；保持真实内容，不用猜测填补缺失来源字段。

2026-10-15新增原生嵌入选项来源时，不能把题内标签的字母索引当作显示位次，也不能再套一次追加四选项的renderer。WebInstruct合格池13条奇序中6条的gold字母索引与显示位置不同；用完整raw question加答案后缀、既有原生pair编码和原ABCD标签映射可保留正文及选项引用。实际固定hash512题和这13条额外CPU接口检查通过，未改正式评测模板或活跃validator，也未生成教师目标。新512题作者标签偏商业／经济，M总输入token明显降低，因此结构可用不代表知识均衡或等token对照；比较须固定同一恢复阶段与真实控制，记录来源／格式变化整体影响。[CPU包、原标签语义与T1匹配边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月15日上午实际恢复数据与57科学科缺口核查)。

<a id="kd-and-data--数据修复后的短程收益与去包装对照2026-09-24"></a>

## 数据修复后的短程收益与去包装对照（2026-09-24）

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#数据修复后的短程收益与去包装对照2026-09-24)。

**当前配置实测，不能当作上限。** 修正RTE、使用MMLU辅助训练集的J/K/L/M均完成400步与全量八任务评估。同0910父、rank8／decoder冻结、B4、seq1024、LoRA LR1e-4、纯partial-KL／K100及5000步余弦前400步，J仅将整个task样本混合权重从0.5改为L的0.8，八项65.404405410506→64.982715182728，下降.421690227778pp；这是该比例方向的匹配负例，不能外推成所有比例或当前decoder恢复均无效。J基本持平初始，LR3e-4的K在200→400仍上升。纠正数据错误是保证比较有效性的必要条件，不保证该预算立即超过旧数据或达到69；旧大batch B与本轮同时存在数据、batch、预热和曝光差异，不能据其排名单独评价新数据。[全量指标、具体去留与保留用途](../../records/e2e_0910/2026-09-24_rank8_search.md#jklm的400步完成结果2026-09-24)

**已取消的数据对照的解释边界。** 原拟用原比例/每任务等概率与ChatML/裸续写组成2×2；用户随后固定本次实验的数据配置，因此该对照已停止，没有有效精度结论，不作为当前执行建议。裸续写保留原答案前导空格、使用评估pair分词，不附加EOS；取消包装会同时改变目标token、归约分母及截断后样本保留，不能单独归因于某个模板token。纯任务组相对半任务组同时移除通用数据并增加任务样本曝光，不能独立证明通用数据有害。MMLU缺subject与Question前缀、Winogrande空白差异仍保留；这些对照不等于完整模板对齐。CPU与短测只验证实现，效果以同预算全量指标判断。

<a id="kd-and-data--answer-only监督归约分母与联合码位对照"></a>

## answer-only监督、归约分母与联合码位对照

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#answer-only监督归约分母与联合码位对照)。

复用固定token archive测试已有其他loss配方时，训练器从labels现算分母，不代表外部诊断汇总也正确。上述archive缓存的是prompt0.3的分母；CE50/response-only短测须用同一production causal mask按实际prompt0重新计算训练遥测和heldout加权分母，保持archive不变。此时记录的Trainer loss是CE与KL混合总量，不能继续标作纯KL。两格式的response仍分别包含或排除ChatML格式/EOS，因此同样的样本数和alpha不等于同样的token监督；同时改变CE和prompt权重只能评价整体配方，不能拆出单因素归因。

同日同源/同固定数据/同L0 down/同256步补齐CE50/prompt0的完整2×2，ChatML decoder/joint为66.98004898%/67.23828170%，joint高0.25823272pp且高源552约0.10795pp；bare为67.04658171%/66.91719324%，joint仍低0.12938846pp。完整64叶24742题、固定64/32、实际2.45698879bpw及冻结/投射/码库保存审计均通过。这给出了现有联合训练在特定配方下的正向例子，不再笼统沿用“学码位必然较差”；单seed单层短程不证明普遍或显著优势，CE和prompt同时改变也不能认定只有CE是原因。教师在裸格式下更贴近标签，仍未预测训练后完整任务的优胜格式；目前优先沿匹配控制中有正向证据的ChatML CE50继续验证，而非把裸格式接入主框架。[完整对照及条件](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的05:21–05:40记录。

四臂原始初始状态、完整heldout0与teacher读数一致；同格式batch/token/response分母/LR逐步匹配。ChatML首次翻码41步、loss42步才分歧，joint在1–128/129–256在线训练loss分别低0.00699888/0.01159773。但终态同32条留出中，joint混合总loss高0.00590052（response加权也高0.00664757），答案序列NLL却低0.15104961，21/32改善。不是比较了不同留出数据，也不只是等batch归约：323个response token只有99个语义答案，另224为包装/EOS等；总目标含CE和KL，答案NLL仅答案CE。缺少分项forward记录时，不能反推具体由格式项或KL主导，也不能将在线批loss事后均值称为末态模型对整训练集的固定评估。

随后固定同552起点、ChatML完整输入、1024条训练/32条留出、CE50/prompt0和L0 down的256步预算，仅将训练labels限制到既有`answer_positions`，监督token从11911降至4743；留出labels保持原样。decoder-only/joint完整八任务为67.04741473%/67.10673513%，分别较匹配的完整response控制高0.067366pp、低0.131547pp，两者仍低于起点67.13033275%，同2.45698879bpw/rank0。留出答案序列NLL均改善，但完整response混合loss的等batch均值及token加权均值均上升，答案拟合改善没有稳定转为下游收益。这个单seed、单层、短预算反例不支持将新答案mask接入主框架，也不能认定包装监督是根因或外推其他预算；变化同时涉及监督覆盖与归约分母，并未删除输入中的包装。两个具体端点退役，保留[decoder完整验收](../../../../result/qwen3_8b_w25_20260930/format_answer_chatml_decoder_s256/full8/eval/acceptance.json)、[joint完整验收](../../../../result/qwen3_8b_w25_20260930/format_answer_chatml_joint_s256/full8/eval/acceptance.json)及相邻原始指标、导出审计和配置。


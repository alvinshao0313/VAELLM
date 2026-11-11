# 显存、卸载与缓存生命周期

[经验分类入口](../../README.md) · 一级分类：数值与运行性能经验（`runtime`） · 二级专题：`memory`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [Dense offload参数与实际显存边界（2026-09-30）](#kd-and-data--dense-offload参数与实际显存边界2026-09-30)
- [显存与精度：分清驻留对象、数值存储和验证范围](#memory-precision--显存与精度分清驻留对象数值存储和验证范围)
- [按条件选择动作](#memory-precision--按条件选择动作)
- [下次怎么做](#memory-precision--下次怎么做)
- [长文本生命周期的资源边界](#memory-precision--长文本生命周期的资源边界)
- [动态padding的长度上限与实际批形状](#memory-precision--动态padding的长度上限与实际批形状)
- [问题与证据](#memory-precision--问题与证据)
- [全MLP非保护权重学习的资源与部署初态](#memory-precision--全mlp非保护权重学习的资源与部署初态)
- [只读探针、autograd与强引用释放](#memory-precision--只读探针autograd与强引用释放)
- [主缓存、稀疏辅助缓存与恢复stamp](#memory-precision--主缓存稀疏辅助缓存与恢复stamp)
- [原缓存外部引用与无必要重复预热](#memory-precision--原缓存外部引用与无必要重复预热)
- [teacher驻留、搬运成本与真实峰值](#memory-precision--teacher驻留搬运成本与真实峰值)
- [新入口的allocator与CPU线程](#memory-precision--新入口的allocator与cpu线程)
- [CPU-only构造与GPU消费者的初始化边界](#memory-precision--cpu-only构造与gpu消费者的初始化边界)
- [精度切换后的缓存初始化和同步D2H观察](#memory-precision--精度切换后的缓存初始化和同步d2h观察)
- [OPD往返、完整词表峰值与Triton设备](#memory-precision--opd往返完整词表峰值与triton设备)
- [边界](#memory-precision--边界)
- [长度与CPU卸载的正式终态边界](#memory-precision--长度与cpu卸载的正式终态边界)
- [预热峰、训练驻留与卸载测量](#memory-precision--预热峰训练驻留与卸载测量)
- [完整启动环境与碎片化归因边界](#memory-precision--完整启动环境与碎片化归因边界)
- [CUDA allocator初始化与测量顺序](#memory-precision--cuda-allocator初始化与测量顺序)

<a id="kd-and-data--dense-offload参数与实际显存边界2026-09-30"></a>

## Dense offload参数与实际显存边界（2026-09-30）

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#dense-offload参数与实际显存边界2026-09-30)。

**源码核查和当前函数数值测试支持。** `e2e_common/dense_loss.py`的dense路径忽略`teacher_output_chunk_tokens`，将完整teacher logits转回student设备并转FP32，因此配置`teacher_output_offload=cpu, chunk8`不代表loss以8token分块。teacher前向输出存CPU与损失阶段GPU峰值是不同条件；启用mass/CE等分支时按真实B4/seq1024/accum4短测验收，不能从参数名推断峰值。当前mass/CE3/CE5在FP32/BF16的独立公式和学生梯度对照通过；CE同样受prompt权重作用。此观察不表示chunk配置改变了损失公式，也不提供精度提升结论。[真实流程和证据](../../records/e2e_0910/2026-09-30_focused_b16_search.md)

<a id="memory-precision--显存与精度分清驻留对象数值存储和验证范围"></a>

## 显存与精度：分清驻留对象、数值存储和验证范围

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#显存与精度分清驻留对象数值存储和验证范围)。

<a id="memory-precision--按条件选择动作"></a>

## 按条件选择动作

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#按条件选择动作)。

| 条件／要解决的问题 | 建议动作 | 证据状态与适用边界 |
| --- | --- | --- |
| 估算长文本、预热或卸载资源 | 按完整生命周期分别测峰值与常驻 | [长文本生命周期的资源边界](#memory-precision--长文本生命周期的资源边界) |
| 动态padding首批短于配置上限 | 核真实batch／长度cap／采样，按新增状态的尺寸与生命周期决定是否补峰值 | [动态padding的长度上限与实际批形状](#memory-precision--动态padding的长度上限与实际批形状) |
| 从压缩部署态诊断全MLP自由权重学习 | 固定保护列与共同部署初态，实测完整两卡反向／独立导出 | [全MLP非保护权重学习的资源与部署初态](#memory-precision--全mlp非保护权重学习的资源与部署初态) |
| 探针反传失败或收尾显存不释放 | 查图记录、强引用及实际缓存恢复范围 | [只读探针、autograd与强引用释放](#memory-precision--只读探针autograd与强引用释放) |
| CPU构造/GPU消费前提冲突 | 核真实导入和初始化条件，区分两种入口 | [CPU-only构造与GPU消费者的初始化边界](#memory-precision--cpu-only构造与gpu消费者的初始化边界) |
| 新入口同协议OOM，或CPU验证有线程开销 | 复用成熟完整环境；CPU构造前显式限制Torch及BLAS线程 | [新入口的allocator与CPU线程](#memory-precision--新入口的allocator与cpu线程) |
| dtype或状态身份异常 | 查effective dtype、原生RoPE和同步复制 | [GPU身份表示与effective dtype](precision.md#memory-precision--gpu身份表示与effective-dtype) |
| 同基底Muon κ复用因梯度SHA失败 | 保持参数／数据／分组精确身份，先实测同配置重复的数值差异，保留原κ | [固定κ复用与梯度数值身份](precision.md#memory-precision--固定κ复用与梯度数值身份) |
| 尝试合批、驻留或精度提速 | 保留数值与实际更新对照，再以完整质量裁决 | [合批加速与缓存精度的数值对照](performance.md#memory-precision--合批加速与缓存精度的数值对照) |
| 批量拟合不同层的残差codec | 合并矩阵计算，保持逐层损失、随机序列与优化状态独立 | [独立目标分组拟合的效率边界](performance.md#memory-precision--独立目标分组拟合的效率边界) |
| 评测慢、OOM或回调失败 | 先拆阶段、核完整启动环境与真实词表峰值 | [评测阶段耗时与离线数据访问](performance.md#memory-precision--评测阶段耗时与离线数据访问) |
| 引用CPU、单卡或特定kernel证据 | 只复用其实际dtype、形状和硬件范围 | [边界](#memory-precision--边界) |

以下按问题查动作；各节保留原核查日期、实测条件、未验证声明及原始来源，历史记录不代表当前运行状态。

<a id="memory-precision--下次怎么做"></a>

## 下次怎么做

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#下次怎么做)。

- 分别记录参数、梯度、optimizer moments和计算dtype；加载时先恢复存储精度，避免先低精度舍入再升回FP32。
- 同时报告allocated、reserved和外部GPU观测，注明context、工作区、激活与同卡作业影响。
- 训练能跑不代表中间评测/导出能跑；审计teacher targets、解码cache、scratch及wrapper的完整生命周期。
- offload或sequence chunk优化需要数值与学生梯度对照，保持原损失归一化；“转到CPU”不自动代表GPU峰值降了。
- 多进程队列需要超时和子进程存活检查；最后汇总前所有rank应参与collective。
- 测试矩阵明确CPU、单GPU、单rank、真实多GPU；不得把单rank torchrun或同卡layer_mp当成多卡证明。

<a id="memory-precision--长文本生命周期的资源边界"></a>

## 长文本生命周期的资源边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#长文本生命周期的资源边界)。

2026-10-13新增原生长故事验证：e3全部decoder在线OPD＋M＋七项保持，在一张A800 80GB上实际完成最长32561-token完整prompt、两次更新及第二次非空Adam的模式往返，771.875秒、无OOM。训练峰allocated约44.0GB，模型加载/解码预热约71.6GB，reserved峰75.04GB；仍必须按整个生命周期估算，不能以训练单段44GB声称可在48GB卡启动。此证据绑定原FA2/group8、FP32 master、checkpoint层级、只在真实续写位置计算全词表logits及既有F上下CPU路径；不证明任意batch/长度/全token logits都可承载，也不是质量收益。[完整两更新与原始报告](../../history/w25_transfer/2026-10-04_29_research_log.md#10月13日longbench原始任务训练集与narrativeqa全文准备)。

<a id="memory-precision--动态padding的长度上限与实际批形状"></a>

## 动态padding的长度上限与实际批形状

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#动态padding的长度上限与实际批形状)。

保留正式`batch_size`、最大长度及原数据采样，不等于要求每批都padding到最大长度。新优化器短验应记录实际输入形状及配置cap；不能为满足误设的固定形状断言而人工padding或重采样。是否另补峰值，取决于新增状态／临时张量与seqLen的依赖及已有同条件峰值和余量，而非仅看首批长度。

11月3日类别Muon短验的真实首batch为`[4,456]`，原cap1024、dynamic_padding、数据及seed均保持；新状态、save1后回训2及fresh均通过，step2峰约35.716GB。只有一次性汇总误断言`[4,1024]`，更正实际合同后，复用原C0完整500的B4/max1024资源证据：新optimizer状态尺寸不依赖seqLen且小于原Adam，首κ克隆只在已实测首批发生，80GB卡有明确余量，因此未追加GPU或改变样本。[实际短验证明](../../../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/short_verification.json)与[本臂配置及边界](../../../../experiments/cat_down_recovery_muon_cmcc_20261103/README.md)。这不证明所有新optimizer／模型可复用旧峰值；模型计算、并行、数据上限或状态生命周期改变时，只补受影响的必要条件。

<a id="memory-precision--问题与证据"></a>

## 问题与证据

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#问题与证据)。

<a id="memory-precision--全mlp非保护权重学习的资源与部署初态"></a>

## 全MLP非保护权重学习的资源与部署初态

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#全mlp非保护权重学习的资源与部署初态)。

需要区分固定码decoder学习路径与更大参数空间时，先从同一实际BF16部署矩阵初始化非保护原坐标FP32 master。保护值与索引保持buffer，合成完整BF16矩阵后只做一次`F.linear`；拆成两次GEMM或改读HF权重会引入不同初态。native应先按实际设备执行完整group8预热，再读取该缓存；逐模块传入一个对象即使写`group_size=8`也不会触发同一批融合路径。设置训练精度后再用普通`no_grad`预热冻结attention，保留hidden反传；参数、梯度及Adam状态留在各层设备，跨卡参数统一clip一次。独立CPU部署对象接收BF16更新，保持原FP32 master和RoPE，custom状态先物化为普通Linear再序列化。

2026-10-30 Qwen3-8B完整B原型实际覆盖36层／108MLP、两张A800的18／18层拆分、M32最长958及QA最长10593，静态教师目标在CPU。5,341,446,144个非保护元素的FP32参数／梯度／Adam双状态下界79.59375GiB，全形状BF16保护模板另占10.125GiB；不是按非零保护值计费。两更新实测dense单卡最大allocated54.442GiB／reserved62.816GiB，D为18.087／32.496GiB，而D部署预热allocated约30.09GiB。两臂全部投影梯度有限非零、保护与非MLP冻结，3423／2127持久字段独立strict及同64token输入BF16输出exact。dense实际logical13.1101bpw，属于超预算诊断，不能把接口通过当低比特成品。[配置、原始报告与数值边界](../../../../experiments/mlp_learnability_cmcc_20261030/README.md#两卡短验实测与取舍)。

D零更新live与共同BF16部署的probe相对差约1.18%，dense初态为exact；两组训练目标下降不解决该数值差对后续质量解释的影响。固定方法是共同部署源、保护冻结、单GEMM和实际阶段峰值测量；按模型重算层数、保护补集、矩阵元素数、分卡与最长输入。D的1e-6／dense的1e-5及两更新在该B原型中只是机制验证初值，当时正式预算尚未定稿，不称跨模型最优。上述B证据本身未覆盖新C0 compact、常驻在线teacher、正式长预算或下游收益；C0新增路径按下段补验，仍不外推其它未测条件。

2026-11-01同规则已补真实C0 compact与中点保存路径：[必要短验归并](../../../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/short_validation_summary.json)绑定两臂各两更新、step1保存后连续step2、四份独立fresh及单卡消费者。D／dense各3675／2127持久字段及同两卡BF16输出exact；MP转单卡完整fixture的实测差为0，但没有把跨设备逐位一致新增为通用门槛。两卡过程最大allocated为D35.44GB、dense58.37GB，单卡消费者已记录峰为57.27／29.62GB；分阶段峰值及reserved保留于原报告，不称snapshot独占峰或单卡正式batch32评测峰。原C0 D零live仍有约1.226%的logit相对差，dense初态exact；dense两步M改善而QA反弹，不能从两步单调性判长程收敛。固定的是共同父和完整状态契约，按新模型重算元素数、布局／分卡及消费者峰；LR只以D1e-6／dense1e-5作当前有界诊断，尚无通用调整区间。该128连续更新及完整九项现已闭合：D／dense均有综合净收益、M仍分别少17／126题，dense超预算，不支持据训练拟合判知识已恢复或D达到严格容量上限；[正式配方与读出边界](../../../../experiments/mlp_learnability_formal_cmcc_20261101/README.md)不将短验通过升级为质量结论。

补教师缓存时，先核身份检查器的设备契约，再搬模型。`muon_compact_cmcc_20261030.inputs.teacher_identity`逐tensor与CPU HF shard比较；先将教师搬GPU会在任何目标生成前因CPU／CUDA比较失败。M4096接续已用同一HF先在CPU核身份、再沿已有layer device-map统一映到单卡并保持FP32 RoPE alias，新增3072目标、冻结全状态及保存重载均完成；旧失败保留，没有换教师或目标数学。该规则不代表任意身份检查都只能CPU，按实际helper与数据位置适配；模型／shard身份重算，缓存仍绑定原样本、支持集合及teacher版本。[缓存来源与实际修复边界](../../../../experiments/mlp_m4096_cmcc_20261101/README.md)。

<a id="memory-precision--只读探针autograd与强引用释放"></a>

## 只读探针、autograd与强引用释放

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#只读探针autograd与强引用释放)。

2026-10-25的best abs全decoder Muon验证暴露了只读探针对autograd的副作用：`weight[row,col].detach()`先执行可导索引，仍会记录索引张量。首前向安装探针、反向checkpoint重算前卸载时，每层三MLP的两次索引各保存两份index，恰好造成saved tensors为115／103，首次M反传失败、optimizer更新为0。最小修复是`weight.detach()[row,col]`，保留实际sparse加法和模型返回的可导张量；同条件修复版两个完整M／QA／R更新及第二次非空优化器往返已实际通过。诊断中途停止追踪不能靠事后detach补救，应在诊断运算前脱离图或使用no_grad，且不得包住真实训练计算；这不是Muon或稀疏补偿的精度负例。完整两步、导出和正式训练状态见[唯一实验正文](../../../../result/qwen3_8b_mmlu65_20261004/qwen_abs005_adam_muon32_pair_20261025/README.md)，[研究记录](../../history/w25_transfer/2026-10-04_29_research_log.md)。

同次验证在两次更新、保存及源状态恢复完成后，收尾仍有18,786,174,976B allocated：Python3.11中`if 'modules' in locals(): del modules`生成的局部变量字典保留了模块引用，删除快速局部变量不足以释放它。弱引用复现定位后，将owner预先定义为None并在finally显式清空，移除locals查询；没有改训练或降低allocated归零要求。补验复用原两步compact，实际重载4071源字段、预热252缓存后释放allocated0，再完成CPU独导及fresh GPU重载，252缓存／aux读出exact，末次allocated0；未重复训练。原失败报告不回写，通过另一个completion记录绑定旧有效训练与新增收尾／导出证明。此例适用于被局部字典保留的强引用，不应把所有显存未释放都归因locals，也不由短验成功推断任务质量收益。证据和修改范围归上述唯一实验正文。

<a id="memory-precision--主缓存稀疏辅助缓存与恢复stamp"></a>

## 主缓存、稀疏辅助缓存与恢复stamp

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#主缓存稀疏辅助缓存与恢复stamp)。

同日，bestabs上的rank8初始空间配对又暴露了旧无sparse路径的复用边界：`install_precision_runtime`在override从None变BF16时调用`clear_decoded_weight_cache`，同时清空三个非持久sparse辅助缓存；旧LoRA安装仅恢复主权重缓存。两臂8更新及100观察计算结束后，包含全部named buffers的恢复stamp失败，而报告直到函数返回才写入，导致观察结果未落盘。真实小CPU VAELinear已复现三个辅助缓存消失、主cache identity/version与持久状态保持，恢复原辅助缓存后stamp恢复；108×3仅为拓扑推导，不能冒称已逐项检查失落的8B终态。应区分持久参数、主缓存与辅助缓存的契约，并将已经完成的结果先保存、清理状态单独记录；合并and断言在首项短路后，不能声称后续项也已核验。复用实际保存的BF16端点直接只读观察即可补结果，无需重训或构造假optimizer记录。[失败证据、CPU根因与独立接续](../../../../result/qwen3_8b_mmlu65_20261004/qwen_abs005_rank8_subspace_train8_pair_20261025/README.md)。

<a id="memory-precision--原缓存外部引用与无必要重复预热"></a>

## 原缓存外部引用与无必要重复预热

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#原缓存外部引用与无必要重复预热)。

2026-10-17 C1单层down的只读P/Q/N路径诊断，首次两臂各五面板均已算完，却在最终通用部署恢复中重复全252 group8预热而OOM：reader仍持有原251冻结native cache引用，清模块字段不等于释放旧GPU张量，再预热会额外分配另一套cache。修复仅在独立诊断入口恢复原6D值、native flag及原cache引用，避免无必要的全模型重解码，并逐面板保存真实读出；未改精度、完整prompt或算法。修后两臂各80次前向完成、完整3423源状态exact／3417冻结／原cache引用保持、GPU释放0；首次160加修后160共320次学生前向，共享8次教师前向未重算。下次只读干预收尾先追踪缓存的外部引用及实际恢复范围，复用已保留的原缓存，并核完整状态；不要将此错误归因allocator碎片或由主体前向完成推断收尾安全。这是缓存生命周期的本配置实测证据，不是精度或质量提升经验。[原始报告](../../../../result/qwen3_8b_mmlu65_20261004/qwen_c1_selected_down_mrqa32_20261017/path_probe/report.json)、[结果与限制](../../history/w25_transfer/2026-10-04_29_research_log.md#10月17日长预算结果阶段节点比较与文献对照)。

<a id="memory-precision--teacher驻留搬运成本与真实峰值"></a>

## teacher驻留、搬运成本与真实峰值

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#teacher驻留搬运成本与真实峰值)。

2026-10-16 native k恢复有限配对中，同round_base／初始792张量／每rank10微批／DDP2-B1-GAS2，去掉teacher模型CPU往返后四次非零更新均值26.742→8.358秒；分段同步计时表明原teacher上／下卡占主要时间，而teacher forward与student前后向接近。新增训练峰allocated约16.381GB，和实际teacher大小一致，支持在相同资源条件下优先检查搬运。但本次实际输入最大6128，未验证8192或全七类峰，不可据k的3.199倍外推整条链。两臂首预热采样梯度relL2 .000732，四次更新后采样位移relL2 .1713、cos .9854，loss有限但无exact轨迹／质量等价结论；位移差的分母是小更新，不是总权重或准确率。不得为此放宽training_step完整runtime契约或修改活动链。用户接受当前主线耗时后取消迁移，保留原A/B；诊断本身没有新质量结论。[实际计时、数值、资源与取消边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月16日上午完整终态与预算安排)。

<a id="memory-precision--新入口的allocator与cpu线程"></a>

## 新入口的allocator与CPU线程

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#新入口的allocator与cpu线程)。

**沿用已验证批量时，也要保留对应完整启动环境。** 2026-11-03同C0实际down raw、A800 80GB独占卡、MMLU B64的首个loglikelihood分配18.29GiB时OOM，当时allocated37.69GiB、reserved未用28.40GiB；新手写sequence漏了原C0 plan／成熟environment中的`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。在隔离入口、Torch导入前恢复原项后，相同原始模型／B64／0-shot／无limit越过原失败位置，实际loglikelihood请求已推进1025/56168，卡约68.9GB。这里支持修复该入口遗漏，不支持缩batch、修改协议或把所有OOM归于allocator。新入口优先调用成熟环境构造，资源条件改变时只验受影响峰值，不重做算法短链。

活动代码已冻结时不隐式改源或`__init__`。本轮只对新启未交付的CPU controller做单PID接续：核确切PID/start并持pidfd，确认timeout/fit独立进程组后，仅向controller发信号，原fit持续同一进程／阶段更新；新入口等实际产物和fresh合同完成后再接正确环境的后继。非亲缘进程拿不到OS退出码时明确记None，不能以PID消失伪造训练成功。此例不扩大停其它既有进程权限，也不构成自动重试体系。[实际失败、资源设置与接续证据](../../../../experiments/down_geometry_baseline_cmcc_20261103/README.md#实测验证与技术接续)。

**CPU拓扑核证慢时先看实际栈，线程限制要在构造前生效。** 同日SciQ数据／cache已完成验证，尾部`read_prepared`实际停留于Fixed2RiceLinear构造中的`torch.nonzero`，不是仍在生成教师目标。原默认64线程池被taskset限制到4核后调度开销大；保留已写prepared／原缓存和中断证据，仅在新进程以OMP／MKL4及`torch.set_num_threads(4)`重读未闭合的同一prepared，28.83秒exit0，源码／数据／目标均未改。不要把已建大线程池后的CPU affinity当成早期Torch线程限制，也不由单次耗时给出跨机器倍数；应复用已完成的GPU缓存及数据核证，而非从头重算。[原来源、CPU核证和缓存范围](../../../../experiments/c1_sciq_recovery_cmcc_20261103/README.md)。

<a id="memory-precision--cpu-only构造与gpu消费者的初始化边界"></a>

## CPU-only构造与GPU消费者的初始化边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#cpu-only构造与gpu消费者的初始化边界)。

2026-10-10，扩数据短测将显式set_device移到CPU重建后仍在入口失败。一次真实GPU可见import诊断确认，DeepSpeed import的Triton能力查询会调用get_device_capability并初始化CUDA，早于main；相同代码在CVD=''准备通过不能证明可见GPU路径满足CPU-only guard。对确需CPU-only构建的流程，应保留原guard并隔离构建进程，再完整验证数据保存重读身份交给GPU进程，不能简单删除检查或只移动set_device。后续真实GPU父进程已成功接收子进程重建的同hash数据，160条签名、2142个CPU int64张量和共享对象保持，入口故障已解决；随后四组真实更新、最长10593反向、连续Adam、2775冻结状态、BF16补丁保存重载和源恢复均通过，outer0。数据身份已绑定且明确保留为后续输入时，可直接CPU重读该归档并复用验证，免去重复构建；GPU消费者只需核实际输入张量留在CPU，不要求整个进程CUDA未初始化。17:21新aux入口误复制后一条件，两臂在任何模型加载/更新前停止；修复仅删GPU消费者的额外断言，独立CPU准备仍保留原guard及实际数据检查。此执行证据不关联精度收益。[初始化stack、两次失败及CPU子进程证据](../../history/w25_transfer/2026-10-04_29_research_log.md#10月10日六组答案蒸馏完整结果与保护通道可学习原型)。

<a id="memory-precision--精度切换后的缓存初始化和同步d2h观察"></a>

## 精度切换后的缓存初始化和同步D2H观察

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#精度切换后的缓存初始化和同步d2h观察)。

同日保护学习短测移到较大数据时，在安装adapter前遇到144个冻结attention缓存为空。源码确认 `install_precision_runtime` 将decoder compute dtype从None切BF16时会清缓存；短测中的额外控制前向曾重新建立缓存，删掉控制前向后不能假定加载时的group8缓存仍存在。独立修复在精度设置后逐模块调用已有 `prime_decoded_weight_cache(BF16)`，复用原gold冷cache前向的同一解码/no_grad/detach路径，保留108个MLP live解码；GPU6实际144个普通BF16缓存、无梯度以及2523冻结持久状态不变检查通过。无需为缓存初始化重跑40题控制，也不能以不同group解码替代后宣称纯保护变量；这里的通过仅说明初始化契约，不证明质量提升。[失败与原生缓存修复](../../history/w25_transfer/2026-10-04_29_research_log.md#10月10日六组答案蒸馏完整结果与保护通道可学习原型)。

2026-10-12的完整模型候选安装检查暴露了另一种观察错误：原生grouped packed getter用`to(CPU, non_blocking=True)`返回，reader立即CPU比较，首轮在0次任务前向时误报码库身份不一致。真实12,386,304字节原码库的一次独立复制诊断复现立即比较false、同步后true，阻塞`detach().cpu()`也true且SHA一致；后续快照已无差异，不能将不同时刻的读取拼成同一瞬时状态。修复限于reader先在源device取bank，再阻塞复制到CPU后保留exact比较，不改码库、冻结snapshot或数值容限。CPU层的安装通过和既有GPU端比较均未覆盖这个D2H读取窗口；后续遇到同类身份/哈希异常应先确保CPU观察所依赖的复制已完成，不把它当作量化误差或参数损坏。[真实复制证据与范围](../../../../result/qwen3_8b_mmlu65_20261004/qwen_native_joint_readout_20261012/copy_diagnosis/report.json)。

<a id="memory-precision--opd往返完整词表峰值与triton设备"></a>

## OPD往返、完整词表峰值与Triton设备

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#opd往返完整词表峰值与triton设备)。

- 2026-10-09同a0e5的greedy OPD两步原型首次在生成转训练时失败：训练精度切换清空全部252投影缓存，但生成前只以普通no_grad预热108个可训练MLP，144个冻结attention在inference_mode中懒生成缓存；被冻结的权重仍须为输入hidden反传保存，因而报 `Inference tensors cannot be saved for backward`。在教师已移CPU时按原group8/no_grad预热全252，生成后只清MLP缓存，保持144个普通冻结缓存，修后真实两步、第二轮Adam/master保持及导出重载通过。不能用关闭梯度或放弃完整前缀反传绕过此类错误。训练峰allocated34.73GB、group8预热峰71.56GB分别计量，不能将教师返回后的常驻量与预热峰混为一谈；这只是流程证据，未证明任务收益。[根因、精确补丁及两步结果](../../history/w25_transfer/2026-10-04_29_research_log.md#10月9日全层模块交互诊断与qwen优先)。
- 2026-10-07真实Qwen K38/QK decoder恢复将纯FineWeb上限1024改4096，保持B4/GAS1、原生2000调度和teacher-output/model均none：前两步正常，第三步首次[4,4096]在mass-KL教师完整FP32 logits计算处OOM，peak allocated75,535,696,896B、reserved83,154,173,952B，未完成四步。有效token7782不改变损失先物化完整padded词表张量的事实。S当前output-offload=cpu会把完整教师logits搬回GPU，dense路径忽略chunk-tokens；只有参数名不能证明分块省显存。旧B1×8192/output-cpu成功属于不同峰形状，不能直接外推B4×4096。随后使用现有教师模型与输出共同卸载，实际成功范围见下方10月7日运行证据；OOM本身不提供质量结论。[失败证据、源码及后续状态](../../history/w25_transfer/2026-10-04_29_research_log.md)。
- Triton启动应显式使用输入张量的CUDA设备；`fork_rng`只隔离随机数状态，不切换当前设备。2026-10-02四卡100%码位短测在校准第二张卡时触发`Pointer argument ... cannot be accessed from Triton`，张量本身均已在同一卡，不能误判为CPU张量或OOM。packed前向/反向、码位初始化/提交/梯度/Adam/RMS、单层及分组部署的12处launch增加device上下文后，真实双卡5项数值、状态与调用者设备恢复检查通过；随后全Qwen3四卡100%码位4步/save2完成两次独立导出、limited重载推理及回训；第二次反向传播及后续步最高allocated约72.06 GiB、reserved74.44 GiB。此资源证据限定于B4/max1024/FP32 sensitivity+Adam，不等于收敛或下游提升。证据与后续状态见[Qwen3专题](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)。

<a id="memory-precision--边界"></a>

## 边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#边界)。

<a id="memory-precision--长度与cpu卸载的正式终态边界"></a>

## 长度与CPU卸载的正式终态边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#长度与cpu卸载的正式终态边界)。

2026-10-12核对后续终态：下方K38／B4×4096的四步CPU卸载证据之后，正式2000步与五次节点八项、末点完整LB21均已完成，不能把历史短测边界当成当前实验进度。与同raw源／FineWeb／B4／2000步的1024对照相比，4096九项66.1868444754对66.2769237037，差−.0900792283pp；两者八项最高都在末点，未证明收益上限。长度与教师CPU卸载同时改变，不能单独归因于长度或offload；完整生命周期含导出／评测，也不能作为纯训练吞吐比。后续选预算应复用这组完整结果，不重复资源短测或把另一源的8192／400步局部检索结果混作同源全评。[预算复核与原始九项来源](../../history/w25_transfer/2026-10-04_29_research_log.md)。

<a id="memory-precision--预热峰训练驻留与卸载测量"></a>

## 预热峰、训练驻留与卸载测量

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#预热峰训练驻留与卸载测量)。

全模型多stage预热OOM时，先区分已缓存decoded权重与跨组遗留的展开码scratch；同一chunk输出已独立物化且不alias scratch时，可在chunk返回后只清该组运行缓存，保持原group大小、顺序、核和dtype。2026-10-31 Qwen3-8B全252投影S4／K=D的原group8在fitter现场因展开码持续累积而OOM（峰allocated83,546,853,888B）；隔离包装只提前清引用，复用已保存短态，真实最大down group8的8个decoded输出与原路径exact、无storage alias，全252预热／真实前向、MMLU入口和最长32704-token的三例LB均exit0，独立fresh峰allocated32,546,214,400B／reserved35,395,731,456B，随后七完整MMLU与末点LB21也全部完成。两峰驻留状态不同，不相减声称净节省；`released_scratch_bytes`仅为累计被清引用的名义tensor字节，不能当实测释放量。此证据支持该S4消费者的缓存生命周期修复，不证明其它核／stage或训练反向同样适用；未修改公共N快照。[实现、原OOM与完整基线证据](../../../../experiments/progressive_residual_bsq_20261030/README.md#第一阶段隔离实现与验证)。

全S4固定码decoder恢复时，推理预热成功不能替代反向峰证明。复用原native packed训练、decoder／层checkpoint与18／18层模型并行，可先在真实峰组核所有252投影的梯度和新增类别的Adam衔接，再按原预算运行。2026-11-01 Qwen3-8B两种恢复位置的短验各4更新／独立fresh均exit0，覆盖M32×958和QA B1×10593，完整冻结补集与3927持久字段exact；两卡峰allocated分别24,174,365,184／23,922,742,784B。该证据仅限当前H96答案级静态TF混合目标、FP32 master／BF16计算及固定四阶拓扑，不外推全tokenKD、其它模型或单卡可行性，也不证明恢复质量。live packed训练与BF16部署数值差量单列，不冒称逐位等价。[固定协议、实际反向与重载证据](../../../../experiments/progressive_residual_bsq_20261030/recovery_compare_20261101/README.md)。


2026-10-04 五个真实 BF16 部署模型的只读区域诊断，teacher 提前留 GPU 时，首个 group8 decoder 预热在 allocated78.56GiB、额外申请1.47GiB处 OOM；该异常时占用不是单独测得的峰值。改成 teacher 初载 CPU，并在每个 student 加载/预热前回 CPU、预热后才回 GPU，同样 group8/BF16/数据/损失下五模型全部完成，最大实测 allocated70,587,747,328B，teacher 返回后常驻约36.93GB，没有逐模型累积。不能将 native 逐模块缓存的约50GiB训练证据外推到 batched group8 的临时解码峰值，也不能将此高 live allocation 当成仅 allocator 碎片。修复只涉及隔离诊断的设备驻留，不是分块损失或精度变更；旧失败证据保留。[完整驻留修复和五模型结果](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的03:03记录。

2026-10-07上述K38/B4×4096改用现有teacher model/output共同CPU卸载，完全相同的四批1544/1080/4096/4096成功更新，run/outer均0；训练峰allocated74,114,097,664B、reserved81,770,053,632B、主进程peakRSS41,909,063,680B，116.43秒。末两长批稳定live allocated19,211,397,632B。这证明该真实前缀及连续padded4096可运行，未证明2000步质量或所有mask分布；没有none/cpu精确loss/gradient对照。现有CPU路径首步重置peak，不把训练峰冒称包括全部初始化。失败none峰来自中断过程，不能与成功CPU峰相减后宣称完整运行净省多少显存。[配置、真实形状和原生观察](../../history/w25_transfer/2026-10-04_29_research_log.md)。

2026-10-05进一步实测Qwen3完整原生group8预热与L18缓存叶子反向：加载预热峰allocated65.7087GiB，但教师已释放、完整学生预热后的实际allocated为19.2455GiB，逐篇max256的32文档最终mass-KL反向峰20.8536GiB。测量分开记录当前allocated、阶段重置后的peak和reserved；不能将曾出现的预热峰当作常驻量，再直接叠加估算梯度需求。此证据仅覆盖一个BF16缓存权重叶子、FP32累加、其余持久参数冻结，不证明全decoder/全码训练具有相同峰值。[完整测量和恢复证据](../../history/w25_transfer/2026-10-04_29_research_log.md)的`gate_final_kl_gradient_probe`。

现有单模块数值、历史实现审查和硬件冒烟各自只支持限定范围。设计新资源预算时检查当前可用GPU及共享任务，不继承旧日志中的卡号授权。[历史硬件范围](../../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

<a id="memory-precision--完整启动环境与碎片化归因边界"></a>

## 完整启动环境与碎片化归因边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#完整启动环境与碎片化归因边界)。

2026-10-03两份Qwen3类别快照的完整MMLU64诊断在log_softmax分配失败：allocated约37–46GiB，另有29–34GiB reserved但未用，单次请求约10–18GiB。两者BF16/strict加载、216或252模块预热均已成功，不能直接判断模型尺寸或batch本身放不下。实际诊断环境漏掉原成功流程的PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；补齐完整成功环境（含该项及既有CUBLAS/PYTHONHASHSEED等设置）后，不改模型、batch64/32、样本或算法，两份完整八任务均成功，约449秒。表现符合碎片化，但此次同时补齐数项环境，不能宣称已单独消融证明allocator是唯一原因。复用显存证据应同时绑定实际启动环境，环境修复后直接完成原协议，不以降低batch代替。[失败日志、启动差异和完整运行证据](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的down_category_transition_d16_prefix。

2026-10-15 Llama FW最佳成品接TF32后，同类新入口再次漏配成熟环境，MMLU batch64首批申请15.04GiB失败，另有34.13GiB reserved未用。恢复旧FW完整环境后，同一已导出成品和同batch64／seven32完成全部24742题，两个评测阶段均exit0、耗时238.56／170.64秒；LongBench另在运行。该复现增加Llama路径完整八项证据，未单独消融allocator，也没有改变评测精度或重训。已完成训练的实际环境必须保留，不能用补评环境倒填。[训练／导出、失败与独立补评记录](../../history/w25_transfer/2026-10-04_29_research_log.md#10月15日中午llama最佳fw成品接原tf32)。

<a id="memory-precision--cuda-allocator初始化与测量顺序"></a>

## CUDA allocator初始化与测量顺序

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#cuda-allocator初始化与测量顺序)。

2026-10-27，向量自适应RVQ的首次GPU短验在任何更新前因 `reset_peak_memory_stats("cuda:0")` 报 `Invalid device argument` 退出；`is_available()`／`device_count()` 已成功并不代表allocator完成lazy initialization。当前PyTorch 2.6该统计接口直接进入allocator，未主动初始化CUDA。把统计重置移到真实 `model.to(device)` 后，保持正式N=1,540,096、D32/K48/H128/R0、B8192和BF16路径，短验22.93秒exit0，全部阶段更新、完整计费、冻结及独立重载通过。修复只调整测量顺序；不把CPU契约通过替代GPU链验证，也不据此声称精度收益。原失败与修复证据均保留，随后正式配对才启动。[设计、原始证据路径与运行身份](../../../../experiments/adaptive_vector_rvq/README.md)。


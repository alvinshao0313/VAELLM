# 状态持久化、断点与精确续训

[经验分类入口](../../README.md) · 一级分类：保存与复现经验（`lifecycle`） · 二级专题：`checkpoint`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [Checkpoint：先持久化可恢复状态，再导出并独立重载](#checkpoint-lifecycle--checkpoint先持久化可恢复状态再导出并独立重载)
- [按条件选择动作](#checkpoint-lifecycle--按条件选择动作)
- [下次怎么做](#checkpoint-lifecycle--下次怎么做)
- [问题与证据](#checkpoint-lifecycle--问题与证据)
- [边界](#checkpoint-lifecycle--边界)
- [层边界恢复须重建运行缓存（2026-09-23）](#checkpoint-lifecycle--层边界恢复须重建运行缓存2026-09-23)
- [E2E中途评估改变RNG时的阶段恢复（2026-09-24）](#checkpoint-lifecycle--e2e中途评估改变rng时的阶段恢复2026-09-24)
- [源码身份拒绝恢复时保留原运行版本（2026-09-24）](#checkpoint-lifecycle--源码身份拒绝恢复时保留原运行版本2026-09-24)
- [源身份的内容校验与分配空间观测](#checkpoint-lifecycle--源身份的内容校验与分配空间观测)
- [信号中断与源码恢复分别取证](#checkpoint-lifecycle--信号中断与源码恢复分别取证)
- [持久冻结状态与可重建码缓存（2026-10-18）](#checkpoint-lifecycle--持久冻结状态与可重建码缓存2026-10-18)
- [CPU账本审计与实际侧车状态](#checkpoint-lifecycle--cpu账本审计与实际侧车状态)

<a id="checkpoint-lifecycle--checkpoint先持久化可恢复状态再导出并独立重载"></a>

## Checkpoint：先持久化可恢复状态，再导出并独立重载

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#checkpoint先持久化可恢复状态再导出并独立重载)。

<a id="checkpoint-lifecycle--按条件选择动作"></a>

## 按条件选择动作

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#按条件选择动作)。

| 条件／要解决的问题 | 建议动作 | 证据状态与适用边界 |
| --- | --- | --- |
| 保存、导出或恢复新路径 | 先原子保存，再核实际状态并fresh strict重载 | [下次怎么做](#checkpoint-lifecycle--下次怎么做) |
| 末类别输出dtype、compact或子集冻结异常 | 检查训练包装、依赖、容器与元数据合同 | [训练包装、compact格式与子集状态的导出](export.md#checkpoint-lifecycle--训练包装compact格式与子集状态的导出) |
| 层边界或中评后恢复 | 分别重建运行缓存并恢复评估后的RNG | [层边界恢复须重建运行缓存](#checkpoint-lifecycle--层边界恢复须重建运行缓存2026-09-23) |
| 训练内排名和部署排名不同 | 保留中途最佳，逐候选独立部署复评 | [顺序搜索保留中途最佳，部署排名须独立重载](export.md#checkpoint-lifecycle--顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾) |
| 结构或解码核改变 | 同路径核结构转换，跨核误差另记 | [结构转换与解码核切换分开验证](export.md#checkpoint-lifecycle--结构转换与解码核切换分开验证2026-09-24) |
| 独导或来源身份被拒 | 核实际导入、构造及配置规范化，保留原失败 | [独立导出须复现配置规范化顺序](export.md#checkpoint-lifecycle--独立导出须复现配置规范化顺序2026-10-17) |
| BF16参数配FP32优化器动量 | 核标准load是否隐式按参数dtype转换state，保持原动量数值 | [参数与优化器状态精度不同的重载](export.md#checkpoint-lifecycle--参数与优化器状态精度不同的重载) |
| 冻结检查或生成协议异常 | 区分持久状态与缓存，并核最终有效配置 | [持久冻结状态与可重建码缓存](#checkpoint-lifecycle--持久冻结状态与可重建码缓存2026-10-18) |

以下按问题查动作；各节保留原核查日期、实测条件、未验证声明及原始来源，历史记录不代表当前运行状态。

<a id="checkpoint-lifecycle--下次怎么做"></a>

## 下次怎么做

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#下次怎么做)。

- 在不可逆导出前原子保存恢复必需状态；每个模块导出成功立即更新manifest，失败写明FAILED及目标，不能靠过期TRAINING标记判断进程状态。
- 区分同精度同路径的保存等价性与部署精度变化；各自记录误差，不通过放宽一个总门限掩盖语义混用。
- 采用fresh process strict重载；核对压缩范围、码、拓扑、冻结状态、推理输出，不能只验证文件存在。
- step恢复验证optimizer/scheduler/RNG、可变参数和Sparse Bit状态，并比较下一步更新；到达max_steps后应进入finalization。
- 原日程含warmup零学习率时，中点保存后继续训练与非零更新导出分别核实；选实际已有非零更新的节点，不改正式warmup来让短验看似更新。
- 运行缓存要在状态加载后刷新，训练评测、finalization与部署的表示必须明确。低精度矩阵运算重结合不自动保持同值。
- 成功验收后，依据后续用途清理checkpoint；恢复安全不意味着永久保留所有权重。

SC-CAT末类要求两份VAE各保留合法端点、且raw0只有全局严格胜出才有资格时，须分别保留raw0与本配方最好恢复部署。例如A的raw高于全体恢复、B的raw也高于B恢复，B仍只能用自己的最好恢复节点参加双端点九项；按“每个raw只留八项最高部署”提前清理会误删B必需端点。新版控制器已据此分开保留，CPU反例核过全局raw0资格和B的恢复端点；真实末类GPU／九项尚待执行，不称已实测该分数例子。[实施与CPU证据](../../records/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31_results.md#剩余全部类别与验收执行准备2026-11-09)。

<a id="checkpoint-lifecycle--问题与证据"></a>

## 问题与证据

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#问题与证据)。

**warmup首节点的验证边界：** Qwen3-8B／SC-CAT原500日程的两步技术验证中，step1实际LR为0，216份未来dense LoRA B均为0；step2的216份B全部非零，另有216个decoder叶变化。因此step1成功保存／独立fresh只能证明该节点的生命周期，不能独自证明非零LoRA融合。固定方法是检查实际scheduler、保存的mutable和部署变化；迁移到新模型／日程时重算首个非零节点，不把“第2步”固化成通用阈值。可在既有两步短验中改选step2独导，无须增加正式训练或修改学习率。同批o两种decoder范围的step2部署均证实180个未来dense owner及持久decoder发生实际变化，并通过独立CUDA fresh；该证据限本次Qwen3-8B、原500日程与已核范围。节点变化和部署原件见[本批恢复记录](../../records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md#cat-recovery)和其`technical/k_nonzero_mutable_check.json`；短验不支持收敛或任务收益结论。

**训练与节点读出并行的条件：** 当前SC-CAT native保存器先落完整optimizer／scheduler／RNG／trainer状态，再原子保存mutable，最后原子发布`checkpoint_meta.json`。可在该提交标记出现后，只读校验实际训练请求、原父、日程、数据、seed、范围和全部必需文件，再用独立GPU执行既有节点导出／fresh／完整八项；不修改训练进程或用目录存在代替保存完成。固定规则是每节点只消费一次、最终整轨迹与早评receipt逐字段闭合、全部规定对照结束后选点；需要按模型／保存器重新核实提交顺序和实际字段。此处没有新增算法超参，有限调节仅是已授权训练／读出卡分配。6个真实技术节点CPU比对通过；正式k→o已完成8条500轨迹、32节点独导／完整八项，早评receipt与最终正式节点清单闭合。该证据限当前native格式，不证明任意HF checkpoint均可边写边读，不证明任务收益或具体加速比。[本批接线、验证与终态边界](../../records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md#cat-recovery)。

Linear输出实验在导出第4个模块时触发误差门限，旧runner此前没有保存整分片训练状态，导致后续60个Linear最终状态无法从日志恢复。校验还混合BF16训练、FP32融合与TF32部署路径。[导出故障记录](../../records/experiments/linear_output/RECOVERY_NOTES.md)

真实硬件审查还发现：finalization重新unpack已训练packed decoder、提前合并LoRA支路、保留FP32输出wrapper，以及已到max_steps的恢复额外多跑一步等问题。[硬件审查](../../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

<a id="checkpoint-lifecycle--边界"></a>

## 边界

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#边界)。

历史硬件审查只通过单卡/单rank链路，真实多卡为hardware-deferred。本次没有复测。结果报告里旧output_alignment_v6_partial192路径与后续distill_init交付名称不同；读取当前模型必须以实际路径和最新交付记录核对，不能照搬历史“当前状态”。[后续交付说明](../../records/experiments/linear_output/EXPERIMENT_SUMMARY.md)

<a id="checkpoint-lifecycle--层边界恢复须重建运行缓存2026-09-23"></a>

## 层边界恢复须重建运行缓存（2026-09-23）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#层边界恢复须重建运行缓存2026-09-23)。

**本配置实测支持**：Qwen3-8B单stage VAE逐层恢复中，block9/10各1步的连续训练与层边界恢复，在native码/decoder、输出、loss/LR、RNG和冻结状态上严格一致；注入7处过期grouped packed副本后，复用native refresh能恢复正确前向。[工程验证及原始证据](../../records/experiments/liftquant_recovery/2026-09-23_engineering_equivalence.md)

**动作**：边界原子保存仅完成块的native状态和必要身份/样例/RNG；恢复从原初始模型开始，验证完成列表是目标前缀，匹配实际初始权重、FP teacher权重、代码、配置与token内容。复制persistent状态后同步grouped packed并清除decoded缓存，再严格比较已完成块输出。teacher前缀重放后才在保存边界恢复RNG。

**边界**：新block原本就建立fresh optimizer，因此层边界可不保存前一块Adam；未完成block的连续proxy/Adam/scheduler不能仅从硬码恢复。本轮各1步只验证恢复路径，未证明长训练或下游收益。

<a id="checkpoint-lifecycle--e2e中途评估改变rng时的阶段恢复2026-09-24"></a>

## E2E中途评估改变RNG时的阶段恢复（2026-09-24）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#e2e中途评估改变rng时的阶段恢复2026-09-24)。

**证据状态：CPU精确对照通过、真实双卡流程已覆盖。** 当前lm-eval会重设随机状态，而Trainer的checkpoint原在评估前保存RNG；阶段暂停再恢复可能重放评估前状态，偏离连续训练。修复是在成功评估并恢复训练状态后，将各rank实际RNG写回本次checkpoint，再允许暂停；不改变连续训练原有语义。[验证及边界](../../records/e2e_0910/2026-09-24_rank8_search.md)

**当前实现约束**：启用eval_after_save时，save_steps和save_strategy属于不可变恢复契约；评估本身影响RNG，不能把降低评测频率当作无语义变化的续训加速。阶段预算只改stop_after_step，并保持其为原save_steps的整倍数；待评估完成、RNG写回并成功暂停后再恢复。当前恢复入口从checkpoint反推原run目录，`run_root_dir`不会将续训迁移到新目录；状态以resolved output为准，同一实验按step追加结果。依据为`v6_runtime_state.py`的evaluation_execution契约及`runtime_v6_pipeline.py`的阶段验证，具体运行见上述记录。

阶段预算应独立于scheduler总steps。真实CPU Trainer含dropout/累积的连续与恢复参数、Adam、scheduler逐位一致；真实GPU非确定运行在暂停前也有差异，不能据此宣称GPU逐位等价。最终导出核切换问题已独立修复并通过同路径验收，见下节；阶段恢复通过本身不能替代导出验证。

<a id="checkpoint-lifecycle--源码身份拒绝恢复时保留原运行版本2026-09-24"></a>

## 源码身份拒绝恢复时保留原运行版本（2026-09-24）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#源码身份拒绝恢复时保留原运行版本2026-09-24)。

**本配置实测支持**：LiftQuant全层恢复被SIGINT中断后，保存边界与当前源代码仅有通用E2E CLI六行差异，严格恢复因此拒绝。隔离源码副本还原这六行后，101个训练依赖文件SHA与边界逐个相同，初始权重、teacher、参数、token、runtime及保存张量检查通过。[中断与重启记录](../../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**动作**：先定位身份差异，不修改boundary或关闭检查，也不回滚其他活动实验使用的共享文件。已有原内容可精确重建时，在独立源码目录保留原版本，cwd/PYTHONPATH都指向它，模型/数据/产物路径显式指向原位置。运行必须持有明确源码副本或持续保护源码，不能仅依赖启动时已导入模块来保证后续子进程一致。

<a id="checkpoint-lifecycle--源身份的内容校验与分配空间观测"></a>

## 源身份的内容校验与分配空间观测

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#源身份的内容校验与分配空间观测)。

2026-10-26真实类别边界分叉补充：D独立保存及fresh strict已通过，末尾却因源权重`st_blocks`对应分配空间增加4096 bytes而失败；bytes/inode/device/mtime均未变，复核源与D全部2991持久张量的完整digest及metadata/runtime SHA仍一致。分配块数量只记录空间观测，不作为文件内容不变契约；稳定stat检查仍保留，但不能代替内容hash。保留原失败报告，在单独验证报告中绑定旧fresh strict及当前完整内容复核后接续既存D，不重建权重或重跑已完成C／G。该证据确认本次是allocation-only误判，不授权忽略mtime／内容变化。[原失败、限定修复及真实接续](../../../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_20261026/README.md#2026-10-26-2259-主体终态分叉修复与完整评测接续)。

<a id="checkpoint-lifecycle--信号中断与源码恢复分别取证"></a>

## 信号中断与源码恢复分别取证

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#信号中断与源码恢复分别取证)。

**边界**：本次CLI差异是重启阻断，不是SIGINT来源。只有KeyboardInterrupt/子进程signal退出时，不能认定OOM或算法失败，也不能推断信号发送者。无控制终端并断开stdin可隔离终端操作，但不能保证外部显式信号永不再次终止任务；仍保留正常的SIGINT/SIGTERM停止能力。本配置在独立session、stdin=/dev/null下仍第二次收到SIGINT，用户明确表示未主动停止；这证明先前终端隔离措施未解决未知发送源，不能将中断归因于SSH断线。最新边界的101个源码SHA及CPU结构校验仍通过，信号中断与源码恢复问题须分别判断。层边界恢复只保留已完成层，未完成层需从该层首步重做。

重复SIGINT中断而缺少发送者审计时，可在既有运行外层加入系统现有的信号记录工具。本次strace 5.16以seccomp过滤全部syscall，仅记录指定中断信号；真实CPU探针验证能给出原始发送者PID/UID，且保持原有SIGINT退出语义。Python最终可能再向自身发SIGINT以还原退出状态，溯源应看第一条外部SI_USER记录，不能把随后的self-signal误判为训练主动中断。此探针未查明此前两次发送者；工具只改善取证，不表示外部中断原因已经修复，也不改变训练协议。

<a id="checkpoint-lifecycle--持久冻结状态与可重建码缓存2026-10-18"></a>

## 持久冻结状态与可重建码缓存（2026-10-18）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#持久冻结状态与可重建码缓存2026-10-18)。

新BSQ主码更新短测在完成投影后被旧`frozen_stamp`拒绝；旧函数遍历`named_buffers`，混入了`persistent=False`的grouped packed及model indices缓存。真实C1模块按同一`install_banks`重建后，仅这两个缓存指针变化，13项持久状态的值精确不变。因此部署冻结契约应以`state_dict(keep_vars=True)`的持久键定义，不把允许重建的缓存对象身份纳入其中；同时保留未更新模块的缓存引用检查、全部冻结值和最终source restoration／fresh strict。隔离修正后两步实际翻码及3415冻结／3423保存重载通过，未放宽张量值、packed bits或schema合同。该边界针对允许改码的路径，不能借此省掉新码提交后的缓存失效／重建验证。[真实失败、CPU定位及GPU重跑](../../history/w25_transfer/2026-10-04_29_research_log.md#10月18日完整九项终态与下一轮验证)。

<a id="checkpoint-lifecycle--cpu账本审计与实际侧车状态"></a>

## CPU账本审计与实际侧车状态

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#cpu账本审计与实际侧车状态)。

2026-10-25，bestabs闭式rank1短测完成拟合后，因直接把GPU `state_dict` 传给只接受CPU strided张量的 `audit_state_dict` 而停止，尚未进入质量观察。修复使用已有CPU native loader，在真实CPU模型安装已保存因子，再收集metadata及4575字段计费；不以逐key设备复制假定storage共享关系不变，也不重新拟合。真实CPU审计通过，新增504个BF16 storage／43,646,976bits，其余组件保持，logical2.5989027853／packed2.6122774456；移除后原4071字段恢复。该证据仅确认内存状态账本，不能替代整模导出或任务质量；后续GPU消费者要核对同一因子和metadata。无CUDA可见时，直接强制FA2构造会失败，应复用已验证的CPU构造入口并恢复原推理policy，而非改正式注意力实现。[原失败、修复及实测证据](../../history/w25_transfer/2026-10-04_29_research_log.md)。


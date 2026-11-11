# 渐进式自适应残差 BSQ 压缩

建档：2026-10-30；更新：2026-11-08。**本页保存门控设计候选池、Qwen四阶初始化及早期恢复批次证据；跨模型第一阶段规则统一进入[4bit基线流程规范](../../docs/adaptive_residual_stage1_protocol.md)。** 用户已确认新规范沿SC-CAT初始化搜索，但取消INT8保护，所有目标投影权重均进入四阶、每阶K=D，之后才做自适应阶数训练。以下带保护Qwen参数与结果按历史实例解释，不作为新模型固定答案或无保护表示的验收证据。

已完成的原始无恢复S4、56更新比较、from-BF16前缀恢复、skip-up及后续局部恢复分开记录。75% model merge候选已完成同UUID完整九项并达到原带保护S4数值目标；**该模型不符合新无保护规格，冻结配方的BF16整链重放和跨模型泛化也尚未完成**。新无保护实验、门控／降阶均未启动。质量门槛按[通用规范](../../docs/adaptive_residual_stage1_protocol.md#2-固定表示与质量契约)相对各模型自身BF16计算；Qwen绝对阈值与原始批次设计保留在[历史阶段计划](PLAN.md)。

来源链见[从BF16的原完整候选](../s4_from_bf16_20261102/README.md#完整结果与本轮结论)、[只跳过up恢复](../s4_skip_up_20261103/README.md#完整结果与本轮结论)、[同父M-only／半LR定位](../s4_down_recovery_diagnosis_20261104/README.md)、[保护与目标配对](../s4_scope_objective_20261107/README.md#完整结果与本轮结论)、[合并](../s4_model_merge_20261108/README.md#完整结果与本轮结论)及[最新同模型完整九项](../s4_merge75_longbench_20261108/README.md#完整结果与本轮结论)。实时排程只见[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。以下早期条目按各自记录时点解释，不以旧“当前”限制替代新规范或授权。

阶段边界：无恢复基线完成不等于第一阶段增强配方定稿，也不构成停止已授权探索的理由。上游尚未确定时继续有界完整候选研究；选定配方验证后，其优质checkpoint可以反复作为第二阶段统一初始化。旧seq56暖启安排已撤回且未执行，旧成品只作外部对照；适用代码、合法静态BF16教师缓存和已有效验证只读复用。当前门控／降阶不启动。

先争取取得性能充分恢复的四阶训练基线。原BF16同协议MMLU73.00242131%、九项均分68.04714492%，是本阶段追赶的参照；四阶研究父不替代最终≤2.6bit成品。56更新的时机比较只说明小预算下的配方表现，不能当恢复上限。旧两臂各类实际decoder更新数56／48／40／32／24／16／8，完整七类联合仅8次；需要进一步检验联合优化、数据覆盖与监督目标。阶段反馈用于决定下一项改进，不能以训练loss下降承诺恢复到BF16。

[详细实验顺序与阶段汇报](PLAN.md) · [唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md) · [文献与经验](LITERATURE.md) · [实验记录流程](../../docs/experience/WORKFLOW.md)

## 已完成的低成本恢复比较

问题是在固定小总预算下，逐类及时恢复能否比末端集中恢复获得更好的质量／成本。只比较两个训练候选，复用现有无蒸馏基线；不因七个检查点给逐类臂七倍预算。两臂各七段×8更新，逐段prefix decoder集合、数据流及连续优化器规则相同；末端臂从完整S4开始，逐类臂按原BF16 factory与共享S4 bank推进。

固定协议、来源／冻结合同和实际验证／运行证据归[恢复比较子包](recovery_compare_20261101/README.md)，执行依赖、预算、负责人和实时GPU安排仍只维护于[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。目标为既有H96答案级静态TF混合目标，M使用原`.5F+.5gold`，QA使用gold-prefix全词表教师KL，不称全token纯教师蒸馏。原已测S4父保持独立，不把短验或恢复候选覆盖为原始基线。

## 目标与证据边界

研究一条从较高码率初态平滑进入极低位宽的基座生成路线：

**逐类四阶残差 BSQ、无类别蒸馏并观测损失 → 据损失选择必要蒸馏，形成高质量固定四阶父 → 端到端渐进收紧全局预算、每向量选择残差前缀 → 冻结门控、裁减并打包 → 完整质量与存储验收。**

- 保留 VAELLM／BSQ 权重压缩主体。早期批次只压既定q/k/v/o/gate/up/down七类投影的未保护向量并保留INT8保护；2026-11-08新规范已明确取消该保护，目标投影的全部权重均做四阶BSQ。embedding、head和骨干norm仍按既有范围处理，历史批次结果不改写。
- 研究希望是显著缩小当前约10个百分点的 MMLU 缺口，争取接近 BF16 约3个百分点的损失；约70分是当前 Qwen 的研究希望，不是新增单项验收门槛。
- 最终验收仍以唯一执行计划为准：投影完整逻辑位宽≤2.6、同模型同协议 BF16 九项平均绝对下降严格<0.9pp；packed 位宽和完整模型存储另报。跨系列证明仍是总目标的一部分。
- 四阶只是待研究的高保真起点，不等于已接近 BF16。渐进路径、可学习门控、数据覆盖与最终低位宽可表达性均待验证，不预设能追回约7个 MMLU 点。
- 本实验加入候选集合，不替换现有 dynamic／Rice、fresh C0/C1、Muon 或块预重建比较。实时资源、负责人和排程只维护于总计划。

## 与既有实验的关系

| 路线 | 已有内容 | 本专题要增加的内容 |
| --- | --- | --- |
| [现有自适应残差](../adaptive_class_20261030/README.md) | 真实残差 bank、逐向量前缀选择、局部预算、保存／消费者证据 | 路由与表征在端到端训练中共同适应；全局预算分阶段收紧 |
| [已有1—4阶原型](../adaptive_vector_rvq/README.md) | 局部码流、配额、退出下界和实包对照 | 不采用固定人数几何退出；允许预算内重新分配，不把早期退出永久锁死 |
| [原生混合表示](../adaptive_native_model_20261029/README.md) | 部分紧凑表示、路由费用、训练／导出基础 | 四阶门控计算图、训练态与部署态转换仍需另核，不能默认现接口已支持 |
| 完整块预重建→E2E候选 | 在完成的压缩基座上研究追加恢复顺序 | 本专题研究基座生成期间的码率路径，两者可分别比较，首批不同时组合 |

## 已确认核心与未定选择

| 已确认的研究思路 | 尚未选定或验证 |
| --- | --- |
| 初始化全部四阶，每阶K=D；Qwen实例gate D64K64、其余六类D32K32 | 新模型按通用规范重算和选择D等设置，不硬编码Qwen赢家 |
| 每向量独立选择保留1—4阶的前缀 | 已提名单重要性标量／有序硬阈值／平滑STE，温度和实现待定 |
| 随训练逐渐降低全局预算 | 预算单位、具体日程、λ／阈值更新频率与速度 |
| 剩余表征需要与路由共同适应 | 首轮提名门控＋decoder、码冻结；码／encoder更新保留为后续候选 |
| 最终门控固定并真正裁减存储 | 四阶表示、深度图编码、实际打包与消费者实现 |
| 用真实功能损失指导训练 | 块输出、整模型教师蒸馏及必要辅助项的选择 |

上表区分表示契约、Qwen实例和待验证门控候选。原无恢复及低成本恢复批次均已结束，后续恢复和最新达标候选按所属记录解释；当前整理通用第一阶段规范，实际运行授权仍只见总计划。

## 当前设计选择与来源缺口

此前讨论的后续候选现已暂缓：每向量单重要性标量，经有序阈值形成硬前缀、平滑STE反向，以BF16正向KL训练门控与decoder。它不是当前第一阶段的实现或执行配置，也未由文献证明其对本LLM有效；无类别蒸馏四阶基线已完成，新增实现仅为低成本恢复两臂。

按用户接续顺序，先完成四阶初始化观测及必要增强并报告，再结合实际结果共同讨论后续门控、可训练路径、数据与优化细节；上述首个降预算候选不因已建档就自动成为执行配置。初始化已选K=D／S4，成熟拟合设置先复用，完整费用按实际载荷核算。

首组降预算对照只比较预算日程。两臂共享同一个从明确BF16来源生成、已完成所选固定四阶适配的真实全四阶父，以及数据顺序、seed、总更新／总token、门控与参数范围、硬冻结时点及最终预算：一臂立即施加最终预算，另一臂渐进收紧。不另给渐进臂额外收尾更新；共同四阶生成与必要适配成本另计。候选门控或目标的其他变化留待该对照有证据后研究，不同时展开。

2026-10-31准备时的来源盘点（历史）为只读文件／metadata检查，没有加载权重或运行GPU：`/data1/cmcc/VAELLM/artifacts`下19份`checkpoint_meta.json`中，非空`converted_modules`均为S2；工作区`result`下1份native metadata也无S4，`/data1/cmcc/models`下无native metadata。旧Qwen的`reconstruction_s4_d32_block64_o3_down3`产物目录不存在；旧Llama gate S4目录仅余`run.sh`，不能作为可加载起点。旧`adaptive_vector_rvq`来源及其迁移映射均不存在，而且其历史active4只为活跃向量训练尾阶，本身也不是全向量四阶bank。此盘点只支持“已查迁移目录没有可直接使用的完整四阶父”，不声称其它归档绝无该产物。

原第一阶段及必要新路径验证已完成；本轮新增授权仅为低成本恢复两臂及必要验证。隔离实现复用原生S4与无类别恢复的保存路径，独立适配高码率／部分dense节点的真实费用与评测，不修改正式低位宽验收。实际资源衔接先核旧任务安排，验证、正式运行与旧结果分开；不能从S2成本承诺后续完整日程耗时，也不因已授权初始化自动启动门控或恢复长训。

## 四阶基线：先观测再确定蒸馏

2026-10-31用户确认先逐类四阶、暂不加类别蒸馏，观察每类压缩后的损失，再决定蒸馏位置与强度，以获得接近BF16、争取无掉点或更好的压缩训练父模型；并要求在对话中逐阶段报告，不能仅在全链结束后汇总。详细阶段、预算、比较与停止条件统一维护于[实验计划](PLAN.md)，本页维护设计候选与已核结果。

### 无类别蒸馏的四阶初始化

七类均从同一BF16生成全向量四阶，已选每阶K=D，未保护主码率4bit，保护／decoder等费用另计；先完成无类别蒸馏控制，质量下降先记录，不在同一观测链中途插入恢复。规格与拟合设置、计费见[准备阶段](PLAN.md#准备先确定什么哪些仍待定)，旧混合K高码率提案不用于本批。

### 类别节点需要回答什么

七个累计节点各报告同协议MMLU对BF16／前节点差值、固定功能读出、当前类前缀误差和费用；增量包含前缀交互，不直接当独立敏感度。完整读出量、复用及逐类汇报见[四阶观测阶段](PLAN.md#逐类四阶先得到无类别蒸馏的损失曲线)，不以训练loss代替下游结果。

### 据观测选择固定四阶增强

原始四阶损失已完整交付，当前只选择上文已授权的56更新两臂恢复比较。其它类别边界、块重建、多重蒸馏或额外位置搜索保留于PLAN折叠的[后续备忘](PLAN.md#固定四阶增强据损失选择蒸馏)，不作为本轮任务或预算。

## 表示、门控与压缩预算

令向量 i 属于投影 ℓ，d_{ℓ,s}(c_{i,s};θ_{ℓ,s}) 是第 s 阶在统一压缩坐标中的完整解码残差贡献。先按统一坐标求和，再沿该投影真实归一化／旋转／保护规则恢复原坐标：

\[
\hat{w}_i=\sum_{s=1}^{4}m_{i,s}d_{\ell,s}(c_{i,s};\theta_{\ell,s}),\qquad
1=m_{i,1}\ge m_{i,2}\ge m_{i,3}\ge m_{i,4},\quad m_{i,s}\in\{0,1\}.
\]

深度 k_i=Σ_s m_{i,s}，首轮限定1—4。0阶是另一个设计，不默默加入。固定长深度标签因此需2bit／向量；其熵压缩、索引和头部按实际编码另算。

**mask 作用于完整 stage 贡献。** bias、已融合 mean/std 等都必须随该阶一起关闭。BSQ 的全零 bit 串仍是有效码字；把 packed bits 设成0，或把 latent 乘0后继续解码，都不能代替关闭这一阶。

训练期间的门控可变，但本专题的部署对象是静态权重：最终每向量长度对所有输入固定。激活和梯度用于训练重要性，不在部署时为每个 token 重新生成一套权重 mask；门控预测器若不再需要，可以不随成品部署。

概念上的预算问题为：

\[
\min_{\theta,c,m}L_{\mathrm{func}}(\theta,c,m),\qquad
C_{\mathrm{logical}}(m,\theta,c)\le B(t),\qquad B(t)\downarrow B_{\mathrm{final}}.
\]

是否优化离散 c 由选定参数范围决定，并非默认所有变量都可导。最终 B_final 按投影原权重数和既定逻辑口径确定，不按平均阶数直接换算成“2.6bit”。

- 训练费用代理、目标预算 B(t) 和真实部署费用分别记录。λ惩罚或 mean(importance) 不是硬可行性的证明；每步硬预算还是阶段校准是候选选择。
- 费用包括真实保留码、decoder、深度图、保护值及其 indices/scales、旋转、路由等。共享 decoder 只要仍被任一向量使用，就完整计费。
- logical、含码padding的packed位宽、全模型tensor storage、实际权重文件和交付目录字节分别报告，不改变既有验收口径。
- 训练阶段保留完整候选 bank 及重新激活所需状态，最终冻结后再物化保留行。训练显存、计算量和临时权重不能冒充最终部署成本。
- 全局预算单调下降不要求每个 k_i 永久下降；早期可以把一阶从某向量转给另一向量。冻结时点须预先规定。

## 四阶初态与功能目标

首选研究的是从明确 BF16 来源重新生成全部目标的四阶初态。若改为给已有两阶成品增阶，它继承的恢复历史与初始缺口不同，须作为另一个初始化候选，不能混称同一 warm start。

四阶的 D/K、残差递归、stage 统计与累计精度须有真实来源。未测得四阶功能质量前，不将低重建误差或较多阶数写成“知识已保住”。

L_func 的首选讨论对象是合法 train 数据上的整模型 BF16 教师分布损失；块输出损失和已有 BSQ／重建辅助项可作候选。仅优化权重 MSE，不能宣称已经利用了最终任务敏感度。

在实际计算图接通时，连续门控的局部导数具有如下关系：

\[
\frac{\partial L_{\mathrm{func}}}{\partial m_{i,s}}
=\left\langle\frac{\partial L_{\mathrm{func}}}{\partial\hat w_i},d_{\ell,s}\right\rangle.
\]

它说明激活及下游梯度可以共同影响门控；真实二值选择仍不光滑，STE 是梯度代理，不是离散损失变化的精确值或全局最优证明。旁路收集统计却未影响门控，只能称统计评分。

数据仍限开源 train／auxiliary_train，记录版本、许可、实际token和重复曝光；不按正式测试逐题对错选训练样本。MMLU和完整九项只作评估与已声明的开发反馈，不能因此称盲测。

## 设计候选池

状态统一为**待验证**。文献事实见 [LITERATURE.md](LITERATURE.md)，本表只维护本实验的候选、用途与取舍。

| 设计维度 | 候选 | 借鉴依据与要回答的问题 | 主要待验边界 |
| --- | --- | --- | --- |
| 门控 | 单重要性标量→有序阈值→硬前缀，平滑STE反向 | VRVQ；能否用较少门控参数学出不同向量的阶数 | 原码率控制不等于硬预算；需核梯度偏差、饱和及内存 |
| 门控 | 有序条件L₀／hard-concrete | L₀稀疏结构学习；随机门控是否更易优化 | 独立开关不自动有序；训练可能有连续样本，需明确硬化规则 |
| 门控 | 边际功能收益→硬预算分配，交替更新表征 | BAMU及现有λ路由；能否直接保证预算可行 | 硬分配本身不可导；局部残差收益不等于任务收益 |
| 训练参数 | 门控＋现有decoder，码固定 | 先隔离门控与预算路径 | 不预设固定码足以重分配全部知识 |
| 训练参数 | 门控＋decoder＋明确的码／encoder更新 | PV-Tuning及既有BSQ经验 | 须保留真实encoder或定义离散重分配；不能假称已有导出含encoder |
| 前缀训练 | 随机前缀／quantizer dropout | SoundStream；低阶能否独立承担信息 | 不自动内容自适应；不能默认固定dropout比例有益 |
| 前缀训练 | 多前缀功能辅助损失 | 研究全阶与低阶能力是否能共同保持 | 是本实验候选，非上述论文已证；增加计算与目标权衡 |
| 预算日程 | 少量分段预算，段间重新分配并适配 | 便于识别骤降与恢复的影响 | 预算点和段长未定，不把4→3→2当所有模型通用规则 |
| 预算日程 | 连续收紧目标，λ／全局阈值跟踪 | 平滑率失真优化 | 代理预算可能超出真实费用，需明确校准策略 |
| 冻结 | 后期统一硬冻结，再在最终预算内适配 | 关闭软硬部署差 | 同对照共享总预算，不额外免费加收尾训练 |
| 尾阶维护 | 保留全阶训练窗口或其他尾阶监督 | 只在尾阶饥饿有实证时打开 | VRVQ的全阶样本消融不是通用成功经验 |
| 初态 | BF16新生成全四阶／已有低位宽父增阶 | 分开初始化价值与继承历史 | 两种来源不能当单变量预算日程对照 |

候选不是开关组合清单。首轮只选一个门控、一个功能目标和一个可训练范围；出现有决策价值的证据后，逐项比较其他选择。某候选有阴性结果，只关闭其已测试条件。

## 文献与项目经验给出的约束

1. **保留完整bank不等于尾阶仍在学习。** 被关阶的decoder／码本梯度可能为0；门控仍有STE梯度也不能消除这个问题。分别记录stage使用率、gate梯度和decoder更新。
2. **软系数幅度不代表已压缩。** 不能靠把gate变小、decoder放大抵消而少计位宽。正常训练使前阶承担更多信息则是预期适配，不应一概称作弊。
3. **码率代理不等于实包。** 深度标签压缩率和decoder启用费随路由变化，费用可能不可分；不称全局最优。
4. **音频时序不直接适用权重。** 不假设相邻权重向量适合相同深度、时间卷积或RLE平滑；若借用须另给依据。
5. **局部误差不能代替知识指标。** 既有退出下界、[输出损失反例](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)和[自适应边界](../../docs/experience/lessons/compression_choices.md#动态阶数的布局退出下界与同cap对照)均适用，不因加入梯度就预设消除这些限制。
6. **阶段贡献依赖真实累计路径。** [stage扩容与数值经验](../../docs/experience/lessons/compression_choices.md#原生stage扩容的累计与冻结契约)要求对齐真实stage输出、bias与归一化，不混用独立S1和多stage累计来制造残差目标。
7. **接口复用有边界。** 当前部分compact分配接口只覆盖最多三阶；四阶门控、反向、冻结、保存与独立重载需要新证据，不由旧两／三阶通过自动继承。

## 后续比较顺序与完成证据

本节保留早期执行顺序：原始四阶链→同bank两臂56更新→完整八项／有限补LB，均已按原批结束；不再作为当前启动指令。后续增强与通用阶段规则见[4bit基线规范](../../docs/adaptive_residual_stage1_protocol.md)，资源与授权只见唯一计划，门控／降阶尚未启动。

本批必须在聊天中报告每类结果、异常／补验及完整基线结果，不能仅更新文档或全链结束后告知。报告字段、后台等待及无法离线主动推送的边界见[阶段汇报约定](PLAN.md#必须在对话中出现的阶段汇报)。下方记录已完成的正式结果；短验／技术失败与正式指标分开。

## 后续记录与隔离边界

- 本目录只拥有后续新增实现及与其绑定的说明；不修改仍被正式实验依赖的源码、配置、初态和pins。需要复用组件时先核接口，输出独立保存。
- 设计定稿后才创建必要入口，不在本阶段添加空脚本、假配置、mock模型或无法执行的命令示例。
- 实际运行记录包含run_id、命令、代码SHA／差异、模型／数据身份、预算日程、路由、损失、bit账本、日志、PID／会话和退出；原始指标留结果目录，本页更新必要结论与链接。
- 模型、GPU与运行状态按原始记录核实，短验不充正式父；目录、论文或已授权本身不构成质量验证或实验完成。

## 第一阶段隔离实现与验证

`recipe.py` 固定本次七类S4配方，`fit.py` 只调用冻结N快照的原生拟合、安装和v6保存；`evaluation.py` 复用原HFLM/raw-sample、LongBench21 runner和评分，`run.py` 只编排既定类别链与读出。没有类别恢复、future LoRA、门控或后续压缩训练，公共源码及活动实验pins不改。

源模型为本机原始 `/data1/cmcc/models/Qwen/Qwen3-8B`；校准沿用 `vicgalle/alpaca-gpt4` revision `f7e3ded725cb81e8e564e32feb12860f376f2b51` 的 `train`，许可证按本机数据卡为 **CC-BY-NC-4.0**。保护校准为512×256／seed31；功能输入为固定4×256／seed2031，只计算BF16教师正向KL和下一token NLL/PPL，不用于优化。数据卡及实际parquet均绑定SHA，模型绑定原始HF头部／tokenizer身份，BF16完整评测沿已验证迁移结果复用。

每类一份累计v6边界：GPU6继续原生M36拟合，GPU7按q/k/v/o/gate/up/down顺序读取边界执行完整MMLU（B64、0-shot、无limit）。末节点复用同一MMLU，实核fit退出后GPU7执行标准LongBench分片1，GPU6先执行seven（B32）再分片2；按现成binder／merger核同UUID／config、21任务4750和原生成／评分协议。4bit研究父只报告质量及真实逻辑／packed／完整文件成本，不改变最终≤2.6bit验收。

观测分别保留原拟合器公共坐标FP32残差RMS，以及由同一硬码和融合decoder构造真实截断S1／S2／S3／S4拓扑、走各自原生BF16 `_decode_weight` 的1—4阶部署SSE/NMSE（含逆旋转与保护），两者不混称。S==1与S>1可有原生累计／舍入区别；这里未实现未来“始终保留S4 bank、mask尾阶”的门控图，不声称两种消费者等价。短验额外核四阶全量hard码、后续stage不改前stage载荷、每类非当前完整状态补集冻结，以及部分S4＋future BF16和全S4 fresh消费者。

2026-10-31 12:49:19启动的短验位于 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/short_s4_20261031_1250`。完整36层、M36、正式batch、七类四阶全量硬码及每类36个BF16截断前缀／冻结均已执行，每阶实际2更新，线性调度仍按10000步构建。七份真实累计边界均保存；末态 `after_down_proj` UUID为 `6fd3db67-18a9-498f-bbff-f6278ef6b82f`，权重文件6,046,045,505B。q部分S4＋216未来BF16的独立前向已exit0。

原短验在全S4 live前向之前的原生group8预热处自然exit1：原预热先解码全部252投影，最后才清各层展开码scratch，S4约50GiB展开码与BF16 decoded及当前组临时量累计，down第4组申请1.47GiB时OOM。原fit进程峰allocated83,546,853,888B、reserved83,839,942,656B；原始 `fit.json`／`run.json`／`run.exit` 保持实际失败，不能将拟合完成写成全验证通过。`short_live_logits.pt`未生成，末态MMLU／LB未执行，原fitter-live与fresh logits一致性尚无证据。两卡及全部自启子进程在13:20前已释放。

隔离 `prewarm.py` 只在原chunk返回后清该组临时展开码，保留原group8、同分组／顺序／核／dtype及全部decoded结果，不改公共或N快照源码。`released_scratch_bytes`只记累计清除引用的名义tensor字节，不当作实测显存释放或峰值节省。21:58补验从已保存末态开始，最大真实down group8原／及时清理8个输出exact、不与scratch别名，全252预热和真实前向已exit0；该进程20.534秒，峰allocated32,546,214,400B、reserved35,395,731,456B。第二次独立fresh的BF16 logits exact及末态MMLU每学科1题入口exit0，39.056秒；LB三个原代表请求也exit0，70.547秒，实际输入32704／32256／18047 tokens，生成64／512／128，覆盖真实KV解码主循环。补验没有重训28阶；两个fresh相等仍不补称丢失的原fitter-live证明，57题和三例不作为正式质量分数。

CPU核验已exit0，证据为短验目录 `formal_cpu_validation.json`：真实HF模型SDPA构造后走native v6，3927持久字段逐项exact；只查tensor，没有CPU前向或GPU数值声明。额外观察使用真实decoder构造，CPU RNG及下一真实M36 VAE初始化digest exact；scratch清理保持持久状态exact；原BF16 shard schema、21任务真实comparison键、标准两卡任务覆盖4750及实际pidfd／父shell退出发布接口已核。正式GPU仍保留原FlashAttention2／BF16／FP32 RoPE政策。原GPU拟合SHA、新CPU差异及未来GPU读出SHA分别绑定，不改旧报告伪装新SHA通过。

原固定接续于13:32:05挂载，driver `627761/start16211323`，目录 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/stage1_after_C0_Muon_20261031`。它将仅共享GPU的旧Muon误作成功依赖，旧controller／shell退出1后于16:46:59停止，`S4_formal_fit_started=false`；该失败／日志原样保留。S4不读取旧Muon产物，资源退出不应等于算法成功门槛，依据[既有经验](../../docs/experience/lessons/reproducibility.md#后台接续只绑定真实数据和算法依赖)人工独立恢复，不能改成通用失败自动继续。

21:58:25独立入口实际启动：tmux `S4_independent_stage1_20261031_2158`，driver `989077/start19249340`，目录 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157`，命令 `CUDA_VISIBLE_DEVICES="" python -B -u experiments/progressive_residual_bsq_20261030/readout.py --plan <该目录>/plan.json`。新plan SHA为 `12d47d95fc5e05a1cdd81c2cc5792a9b95abca3c16da2ff4f677858a1b83f121`，`after_control={}`；旧链各PID／代次已退役，原退出1如实记录，6／7启动前无compute进程。共享同一旧模型的GPU5独立读者属于工作区其它任务，不是本批或旧孤儿，不干扰。

这次只修改 `readout.py` 的显式人工资源释放控制；`run.py`、fitter、预热、加载及评测数学代码保持前次SHA。`control_cpu_validation.json` 精确绑定唯一控制文件差异与资源证据，3927字段／RNG等exact性质明确继承原CPU报告及其SHA，没有重跑全模型CPU。完整补验记录为 `remaining_validation/validation.json`。正式请求 `formal/request.json` 的 `after_control` 实核为 `{}`，不再等待旧失败链；fit `991810/start19263727` 在GPU6、eval控制器 `991809/start19263726` 为GPU7读出后继。22:02完成原BF16的512×256校准，22:03:34 q M36／S4第一阶已2400/10000，loss13.048、recon12.938、commit0.11043，约0.0274秒／步；没有残留2步限制，确认真实推进后停止主动监控。11月1日用户唤醒后核实整链完成，结果如下；未在等待期间新增其它研究或监控自动化。

短验原先为补验保留完整 `after_down_proj`，补验现已全部完成，后续用途已结束。此前六份短态 `pytorch_model.bin` 已清理80,107,130,699B（74.606GiB），另清理三份被替代且未运行的CPU预备目录与日志807,086B；这两笔不计入本次收尾释放量。本次保留与清理见下方精确记录。

## 第一阶段正式结果与交付

运行根目录 `R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157`。[整链状态](../../artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157/run.json)、`formal/run.json`、`formal/fit.json`、`formal/evaluation.json`及两份LongBench分片均exit0，`continuation.exit=0`。11月1日收尾只读复核1033个绑定源码SHA全部一致，七组均实际四阶×10000更新、M36、D32的B8192／gate D64的B4096，`recoveries=0`、`after_control={}`；没有类别蒸馏或未来权重更新。

七累计节点各57学科／14042题，原始与有效样本数一致，0-shot、B64、`limit=None`；各自57份原始JSONL均保留。BF16同协议10251/14042（73.00242131%）。固定功能读出来自同一4×256 train输入、1020个有效下一token；教师PPL4.67672557，未用于优化。节点增量包含此前压缩与当前类别的交互，不能作为独立类别敏感度。

| 新增类别 | 正确／14042 | MMLU% | 较前节点pp | 较BF16 pp | 教师正向KL | 功能PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q | 10234 | 72.88135593 | -0.12106538 | -0.12106538 | 0.00276152 | 4.66746682 |
| k | 10235 | 72.88847742 | +0.00712149 | -0.11394388 | 0.00389741 | 4.66262972 |
| v | 10162 | 72.36860846 | -0.51986896 | -0.63381285 | 0.01143717 | 4.67021020 |
| o | 10061 | 71.64933770 | -0.71927076 | -1.35308361 | 0.01683484 | 4.72235533 |
| gate | 10018 | 71.34311352 | -0.30622418 | -1.65930779 | 0.02545432 | 4.72282782 |
| up | 9963 | 70.95143142 | -0.39168210 | -2.05098989 | 0.03670711 | 4.76448409 |
| down | 9893 | 70.45292693 | -0.49850449 | -2.54949437 | 0.05438913 | 4.79466868 |

原始节点证据在 `R/formal/nodes/<category>/{mmlu,functional}.json`，实际BF16截断S1／S2／S3／S4部署误差与拟合器FP32残差分别保留于 `formal/fit.json`，未把它们改名互替。末节点MMLU直接复用至九项；seven和LongBench21／4750与同一最终UUID绑定，LongBench取21任务未舍入task宏平均，不取category宏平均。[完整结果](../../artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157/formal/result.json)如下：

| 项目 | 四阶基线% | BF16% | 差值pp |
| --- | ---: | ---: | ---: |
| MMLU | 70.45292693 | 73.00242131 | -2.54949437 |
| ARC Challenge | 54.52218430 | 56.56996587 | -2.04778157 |
| ARC Easy | 79.50336700 | 80.80808081 | -1.30471380 |
| BoolQ | 85.77981651 | 86.57492355 | -0.79510703 |
| OpenBookQA | 40.20000000 | 41.80000000 | -1.60000000 |
| PIQA | 76.87704026 | 77.52992383 | -0.65288357 |
| RTE | 77.25631769 | 77.97833935 | -0.72202166 |
| WinoGrande | 66.69297553 | 68.50828729 | -1.81531176 |
| LongBench21 | 48.67483515 | 49.65236225 | -0.97752710 |
| 九项等权均分 | **66.66216260** | **68.04714492** | **-1.38498232** |

唯一交付模型为 `R/formal/native/data1_cmcc_models_Qwen_Qwen3-8B_20261031_220056/after_down_proj`，UUID **`b12014e7-f931-4502-86e3-bca93085eaf3`**，metadata SHA `e816466632ef7c46cdccaf1b423f20c89a624d95670777fd22cd21dd0d9795ab`。252个目标投影全部四阶，未保护主码率4bit；包含decoder、INT8保护值／indices／scales和旋转的投影逻辑／packed均 **4.094031292459239bpw**，padding0。全模型tensor storage6,044,459,520B，权重文件6,046,045,505B，完整交付目录6,086,105,857B；embedding／head／骨干norm按原存储另计。该四阶父不满足最终≤2.6bit预算，也不满足九项下降严格<0.9pp的最终质量目标，本阶段仍按约定完成基线交付。

正式fit从10月31日22:00:56.133至11月1日00:30:55.913，8999.780秒（2.500小时，含校准／编码／保存）；formal整链从22:00:49.509至04:48:58.485，24488.976秒（6.803小时，拟合与节点评估部分重叠）。含130.14秒补验及控制准备的独立入口总墙钟24632.746秒（6.843小时）。这些是各流程墙钟，不是相加后的GPU小时。正式fit峰allocated16,572,725,760B／reserved16,592,666,624B；独立fresh预热的32,546,214,400B allocated峰与原fitter现场OOM峰属于不同驻留状态，不作相减后的净节省声明。原S4 PID均已退出，11月1日收尾实核GPU6／7各1MiB、无本批活动进程。

本次经验增量仅为同源S4/K=D无KD完整基线，以及[预热缓存生命周期的真实GPU证据](../../docs/experience/lessons/memory_precision.md#预热峰训练驻留与卸载测量)。这是单模型／单配方结果，不证明4bit普遍无损，不从累计节点推独立敏感度或自动选择后续增强。丢失的原fitter-live logits证明仍不存在；CPU持久字段exact、两个fresh logits exact与group8输出exact各保留原有合同边界。

### 从原BF16重新复现的准备入口

已结束运行的 `readout.py --plan R/plan.json` 仅作实际命令记录。默认CLI `--prepare` 仍绑定已退役旧控制器，且短态载荷已用完清理，不能将该旧命令称未来可直接重跑入口。复用本次已通过且源码pins一致的 `R/remaining_validation/validation.json`，可用现有程序化 `prepare(..., dependency={})` 从原BF16重新生成独立正式请求；不消费短态或旧Muon权重，不改冻结源码／CLI。下面仅提供复现说明，**本次收尾未执行、未创建新输出或新任务**：

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
which python
python -V
cd /home/cmcc/VAELLM
CUDA_VISIBLE_DEVICES="" python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/home/cmcc/VAELLM/experiments/progressive_residual_bsq_20261030')
from run import prepare
prepare(
    Path('/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/<unique_new_run>'),
    False,
    Path('/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157/remaining_validation/validation.json'),
    dependency={},
)
PY
```

需先确认GPU6／7释放，再由已激活同一bitvae的tmux／nohup调用者运行 `CUDA_VISIBLE_DEVICES="" python -B -u experiments/progressive_residual_bsq_20261030/run.py --request /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/<unique_new_run>/request.json`，stdout／stderr及退出码落在该新目录。执行入口还会检查源码pins与6／7占用，资源异常即停，不等待旧失败链或自动重试。若源码／协议变化，按真实变化范围补证据，不能借旧证明冒充新实现已验证。

### 第一阶段结束后的产物清理

删除前已核整链／所有自启S4进程退出，当前进程打开的文件中无下列候选；源码与计划未安排这些载荷的后继，原始BF16／数据与其它GPU0–5任务不在范围。保留唯一已测 `F/after_down_proj` 完整模型、各边界 `checkpoint_meta.json`／`config.json`、所有配置／源码pins／CPU与GPU验证、核心日志、退出状态、原始MMLU样本与LongBench预测及评分。

下表精确列明本次已删除路径和删除前大小；只删除列出的文件，不递归删除边界目录。`F=R/formal/native/data1_cmcc_models_Qwen_Qwen3-8B_20261031_220056`；`S=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/short_s4_20261031_1250/native/data1_cmcc_models_Qwen_Qwen3-8B_20261031_124926`。权重列对应该目录的 `pytorch_model.bin`，runtime列为 `cat_runtime_state.pt`，tokenizer列为六个精确文件 `special_tokens_map.json`、`merges.txt`、`added_tokens.json`、`tokenizer.json`、`vocab.json`、`tokenizer_config.json` 的合计。0表示该权重此前已经清理，本次不重复计数。

| 精确目录 | 本次权重B | runtime B | 六个tokenizer文件B |
| --- | ---: | ---: | ---: |
| `F/after_q_proj` | 15,483,383,550 | 22,729,058 | 15,882,422 |
| `F/after_k_proj` | 15,260,963,965 | 22,729,058 | 15,882,422 |
| `F/after_v_proj` | 15,038,544,493 | 22,729,058 | 15,882,422 |
| `F/after_o_proj` | 14,140,296,373 | 22,729,058 | 15,882,422 |
| `F/after_gate_proj` | 11,442,203,949 | 22,729,058 | 15,882,422 |
| `F/after_up_proj` | 8,741,738,369 | 22,729,058 | 15,882,422 |
| `S/after_q_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_k_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_v_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_o_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_gate_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_up_proj` | 0 | 22,729,058 | 15,882,422 |
| `S/after_down_proj` | 6,046,045,505 | 22,729,058 | 15,882,422 |
| `F/final_model` | 6,046,045,505 | 22,729,058 | 15,882,422 |

前六正式边界已完成完整评估且无后继；短态七边界已完成补验，其权重不能用作正式父；`F/final_model` 是原生额外自动保存的未选端点（UUID `fc737c8f-5251-4533-a528-f13b268976fa`），全部正式质量证据指向保留的 `after_down_proj`，不为未选副本增加另一份模型保留。以上无权重目录的runtime和tokenizer副本也无独立用途，原始来源与完整交付模型仍可提供对应配置。

下列临时logits已完成固定功能／两fresh一致性读出，摘要、输入、有效token、SHA与比较结果均保留，且本批没有续训或重评用途；它们不是原始评估指标或生成预测：

| 相对 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030` 的精确文件 | 字节 | 删除前SHA256 |
| --- | ---: | --- |
| `short_s4_20261031_1250/teacher_logits.pt` | 309,951,496 | `e63675fd005ccab71ab3c133a4a2c6d7fc5b10ba77e7c6e76dd6a8ac100faea4` |
| `independent_stage1_20261031_2157/formal/teacher_logits.pt` | 309,951,496 | `e63675fd005ccab71ab3c133a4a2c6d7fc5b10ba77e7c6e76dd6a8ac100faea4` |
| `independent_stage1_20261031_2157/remaining_validation/fresh_reference_logits.pt` | 77,794,930 | `a31e5cb45bd6bdca71cf9cab20b21387dc8940bbd8ffb7e6d2041be1f0626ecf` |

本次实际移除109个常规文件，逻辑字节 **93,437,480,351B（87.020435GiB）**，删除前本地独占分配块93,437,779,968B；逐项核目标已不存在，唯一完整交付模型的全部文件inode／大小／mtime保持不变，原始结果及七边界metadata仍可用。不将共享磁盘其它变化计入本次释放量。

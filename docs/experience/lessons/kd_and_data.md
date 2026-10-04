# KD 与数据边界：同名参数可能已有不同数学语义

## 问题与证据

早期prompt KD方案采用所有token的加权均值，后续改成响应区均值加权组合：L_response + w·L_prompt。两份历史审查都通过过各自版本，不能同时当成当前公式。[后续区域归一化审查](../history/reviews/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/final-review.md)

causal KD曾存在一位错位问题：位置t的logits对应t+1目标。硬件审查又发现Qwen3多轮ChatML无法依赖独立prefix tokenization定位真实响应边界。[mask修复计划](../history/plans/2026-08-04-fix-causal-kd-token-mask-alignment.md)、[硬件证据](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

## 下次怎么做

- 明确目标位置到logits的单次shift，最后位置、padding及prompt/response边界用具体token例子校验；避免双重shift。
- 改prompt权重时先写数学定义，确认是区域均值组合还是token加权均值。历史同名CLI不保证同样公式。
- CE+KD组合中CE只加入一次；hidden/pre-MLP的mask不能因KD改动而无意改变。
- dense/offload路径比较标量和学生梯度；空区域应有明确语义，不能用任意默认值掩盖问题。
- token统计按学生micro-batch计一次，不能因teacher、重计算或CPU staging重复计数；恢复后的首个日志窗口按实际观察步数计算。
- 历史错误mask训练得到的checkpoint不能当作“干净对照起点”；以新设计明确是否从同一原始模型重训。

## 边界

来源是历史计划、审查和有限硬件验证。本次未核验当前所有loss实现；CLI与支持的loss以当前源码及根README为准。审查通过不证明某个prompt权重能提高下游成绩。


## 2026-09-24 当前实现核查

当前 `train_utils/distill_loss_core.py` 的 prompt reduction 是所有有效token的加权均值 `(sum_response + w*sum_prompt)/(N_response + w*N_prompt)`，不是上文历史审查中的区域均值组合。`kl_top_partial` 是纯KD，`alpha`仅在 `kd*` 的CE/KD组合中生效；不能因命令含 `alpha=0.5` 就认为训练加入了真实答案CE。`kl_top_mass` 在teacher top-K之外加入tail桶，`kl_top`则是top-K条件分布。以上为当前源码核查，不是不同loss精度优劣的实测。[本轮设计与后续证据](../records/e2e_0910/2026-09-24_rank8_search.md)

2026-10-03补核当前E2E日志：`trainer._store_loss_parts`中的`distill_loss`是本次实际loss函数的完整返回值；`kd_top_mass`时包含`(1-alpha)*CE + alpha*KL`，不是纯KL，也未分别记录两项。该字段来自最后一次micro-batch，Trainer的`loss`则为日志窗口均值，不能直接相减求CE或把两者差异当异常。当前Qwen3 CE10与CE50/task100/response-only在前800步均训练3200条，但平均每条原始label统计的response token约143.28与11.26，混合目标权重、数据和mask不同，loss绝对值不可跨组直接排序；token数也不是梯度贡献。[实际日志及完整指标](../records/qwen3_w25/2026-09-30_goal.md)。

同日这两组完成2000步和每400步完整八任务验收：49e来源66.57623885%，CE10最终66.44208228%，CE50/task100/response-only则从首400的60.38795485%逐点升至66.89468320%，成为当时总体最佳；相同2.456988791bpw/无外置LoRA，冻结硬码与非decoder、3423保存/strict及固定实际64/32通过。首400大幅下降不能代替本例最终结果，也不授权无依据延长任何失败配方。CE50最终五项上升，但MMLU较49e低3.72454066个百分点、OBQA/PIQA亦下降，平均提升不代表全面改善；CE比例、数据及prompt mask联合改变，不能单独归因某一项，更不能把此正确训练数据/decoder-only/2000步结果与旧rank8/800样本实验混作同一对照。[十个完整保存点及逐项证据](../records/qwen3_w25/2026-09-30_goal.md)的CE10/CE50最终结果。

随后在这两个相同硬码/部署规格的端点间使用现有类别组装，CE50 attention+CE10 MLP完整均分67.13033275%，高于双方来源；反向互补仅66.65448072%。两者实际2.456988791bpw，3423保存/fresh strict与完整固定64/32通过，没有重训或增加分支。前者MMLU较CE50恢复299题，但仍低于CE10，且RTE下降；平均提升不代表所有任务改善。共享1911非decoder张量与504硬码冻结是这次比较的前提，不同几何/不同硬码来源的旧组合负结果不构成此新条件已经失败，来源指标也不能相加预测组合质量。该单次组合收益仍不能拆分CE、数据、prompt或各子类别的因果贡献。[来源审计、完整指标及细分对照](../records/qwen3_w25/2026-09-30_goal.md)的08:54同硬码decoder组合。

当前版本task混合生成器定义为七任务取train、MMLU取auxiliary_train；这仅描述生成器设计，不能证明训练实际加载的既有JSONL采用该版本。2026-09-24对实际alias所指文件的复核发现，其条数与内容结构强烈吻合较旧的MMLU dev+validation、附加LongBench生成方式；缺少逐行source与manifest，尚未逐条认证来源。当前MMLU评估使用test，现有证据不能写成已证明测试泄漏，也不能继续声称实际文件已经验证为auxiliary_train。此前由当前生成器推断实际artifact来源的表述在此修正，具体文件身份、历史版本及证据边界见[同轮数据核查](../records/e2e_0910/2026-09-24_rank8_search.md)。调整数据权重前区分样本占比和实际有效token贡献，本轮旧运行仍使用原文件，不原地改写其输入。


## CE混合配方的短程筛选（2026-09-24）

**当前配置实测，不能外推上限。** 固定0910、rank8、decoder冻结、B4acc1、LoRA LR1e-4、同数据及800样本，0.5CE+0.5partial KD比纯partial KD的完整八任务均分低约1pp，退化不限于RTE。因此这组混合权重退出本轮预算，不能把它推广为“CE整体无效”。混合同时改变KD权重，不能单独将结果归因于CE；CE监督来自混合训练文本，也不等于全部人工答案。后续如重试，需明确新的权重、数据或预算依据，先做同预算对照，不沿用损失名称猜实际公式。[配置、八项指标与分支去留](../records/e2e_0910/2026-09-24_rank8_search.md)


## 先核实实际数据文件，再解释混合比例（2026-09-24）

**证据状态：旧RTE内容错误已逐条验证，其余旧尾部来源仍有未认证部分。** 0910搜索实际task文件仅有messages，没有source/manifest；内容分段与旧生成器更吻合。随后逐条核对官方train确认旧2490条RTE全为空题干、答案全部颠倒，新版题干与正确标签映射全匹配。只读最新生成器不足以证明旧文件用其生成，也不能单凭路径名认定数据版本。[文件hash、分段计数、历史代码及后续方案](../records/e2e_0910/2026-09-24_rank8_search.md)

**动作。** 设计数据实验前沿运行alias确认真实路径、行数、字节数、SHA256和可用manifest，并对照实际生成代码版本与样本来源。新数据另存独立版本，保留旧文件及活动任务输入不变；验证来源、split和实际加载路径后，用匹配预算做仅data-version改变的对照，再考虑混合权重变化。本次暂不将来源尚待认证的旧task文件权重提高到0.9。

**统计边界。** prompt权重0.3作用于token归约；source样本比例、累计token分母比例、loss值比例和梯度贡献不是同一量。当前token telemetry统计原始labels，尚未做causal shift；用日志总数计算时必须注明未shift，不能直接冒充KD因果分母。实际encoder核实每条首token均为prompt后，causal prompt计数应减去样本数，response保持不变；没有逐样本证据时只能给条件估计。累计分母份额也不等于逐batch归约再平均后的训练贡献；具体loss/gradient还取决于逐token数值与导数，不能反推。

**已确认根因及修复边界。** RTE生成器曾把super_glue的premise/hypothesis误读为sentence1/sentence2，并把label0/1映射倒置；源码在2026-09-17已修正，但旧JSONL没有重新生成。因此检查源码“现在正确”不能证明训练数据正确。最小检查须对实际产物核实非空题干、官方字段与label names，并记录数据hash和生成版本；重新生成到独立路径，保护旧运行输入。新版176597条数据已生成并有manifest，RTE2490条已与官方train逐条核验；MMLU来源为固定revision的auxiliary_train，具体身份及全量生成验收见同轮记录。

**因果边界。** 新旧版本同时涉及RTE修复、MMLU来源/数量变化和去掉旧尾部，整体版本对照不能将效果归为MMLU单因素。旧数据下CE配方退化的观察仍真实，但不能推广到修复数据，也不能未经单独对照把退化全归因RTE。旧数据上mass K100与partial K1000在400步只较partial K100分别高约0.0044/0.0680pp，尚无显著提升证据；这同样不代表正确数据或更长预算下的上限。

**对照与格式。** 新版数据先用同task0.5混合的基准对旧版，再围绕新基准分别改变普通LoRA LR、task样本权重或CE/KD配方，避免把多个变化揉成单因素结论。新增数据alias保留旧alias不变，真实loader与tokenizer核对实际路径、题干/答案及长输入截断，不只验证JSON能解析。本轮训练仍采用原ChatML，评估仍lm-eval裸题；格式尚未对齐，不能把数据修复说成模板差异也已解决。CPU编码验证不能代替真实模型更新/评估短测，短测也不能证明下游增益。具体准备状态与后续实际运行见同轮记录。


## 生产路径重放token归因与模板审计（2026-09-24）

**本配置实测支持、范围有限。** 新数据J/L的CPU重放调用真实混合流、permutation、canonical encoder及collator，保持实际16 workers与B4；J首40条和两组前800条的20个日志窗口全匹配。不能按运行前normalized的workers=0重放：入口后续将其改为16，iterable路径也会关闭group_by_length。重放必须覆盖真实跳过无有效回答样本的逻辑，raw draw比例不能替代编码后训练样本。1600条是同配置确定性重放，后800在审计时尚未获实际日志逐窗覆盖，不能写为1600条都已实际训练。[统计原件、源码位置与范围](../records/e2e_0910/2026-09-24_rank8_search.md)

**观察与动作。** 已匹配前缀中，task样本权重0.5/0.8对应累计因果token分母份额仅约21.82%/48.88%；MMLU为18.61%/41.67%。长回答语料能显著改变样本权重在token归约中的份额，所以调整混合比例前先量化source级有效prompt/response，不能把task0.8当成优化贡献80%。本观察不是提高task权重有效的下游证明，也不授权改已运行配置。

**模板与内容分开。** MMLU auxiliary_train无subject时不能为凑评估description杜撰学科；Winogrande train的606条context差异经逐条比对仅为空格，目标与续写内容均正确，不能当作标签错配。ChatML空think与terminal EOS使已检RTE/MMLU的response目标包括包装token，不等于答案语义token数。格式差异的存在有源码/实际数据证据，对齐是否有收益仍待受控实验；保持真实内容，不用猜测填补缺失来源字段。


## 数据修复后的短程收益与去包装对照（2026-09-24）

**当前配置实测，不能当作上限。** 修正RTE、使用MMLU辅助训练集的J/K/L/M均完成400步与全量八任务评估；J基本持平初始，提高整个task权重到0.8没有收益，LR3e-4的K在200→400仍上升。纠正数据错误是保证比较有效性的必要条件，不保证该预算立即超过旧数据或达到69；旧大batch B与本轮同时存在数据、batch、预热和曝光差异，不能据其排名单独评价新数据。[全量指标、具体去留与保留用途](../records/e2e_0910/2026-09-24_rank8_search.md)

**已取消的数据对照的解释边界。** 原拟用原比例/每任务等概率与ChatML/裸续写组成2×2；用户随后固定本次实验的数据配置，因此该对照已停止，没有有效精度结论，不作为当前执行建议。裸续写保留原答案前导空格、使用评估pair分词，不附加EOS；取消包装会同时改变目标token、归约分母及截断后样本保留，不能单独归因于某个模板token。纯任务组相对半任务组同时移除通用数据并增加任务样本曝光，不能独立证明通用数据有害。MMLU缺subject与Question前缀、Winogrande空白差异仍保留；这些对照不等于完整模板对齐。CPU与短测只验证实现，效果以同预算全量指标判断。

## 0920的hidden与pre-MLP权重须分别对照（2026-10-01）

**搜索已完成，单seed按共同2500步比较。** 固定0920初始ckpt、rank8、数据和`kd_top_partial`类型，以相同优化配方为参照、固定pre-MLP=.01时，hidden=.1/0/.03的训练内八任务均分分别为67.883982%/67.805292%/67.757841%；不能据此把两种hidden对齐损失一起判为有益或有害。[配置、原始指标与结果](../records/e2e_0920/2026-09-24_rank8_search.md)

**分开改变权重，并复评实际交付模型。** 固定hidden=.1，同一参照下pre-MLP=0/.03/.01的2500步快照，经独立strict重载、全量八任务0-shot复评分别为68.024489%/67.859254%/67.742981%，与训练内排序一致，但分差有变化；最终保留pre-MLP=0的模型。这是本轮实测选择，不证明稳定显著优胜或所有正权重有害；两个权重各自的.03候选仅到2500步，不能将其排名推广到长程或其他配方。

**CE/KD混合结论限定于匹配条件。** 继承pre-MLP=0后，alpha=.99/1在2500步的训练内均分为67.688203%/67.708467%，均低于同预算alpha=.95的68.009998%；纯KD已到5000步，终点导出评测67.423667%，.99仅到2500步。因此本轮保留.95配方，不把短程与长程混比，也不将纯KD终点导出分冒充独立重载分。此单seed结果不证明CE普遍有益、纯KD整体无效或69不可达。


## Dense offload参数与实际显存边界（2026-09-30）

**源码核查和当前函数数值测试支持。** `e2e_common/dense_loss.py`的dense路径忽略`teacher_output_chunk_tokens`，将完整teacher logits转回student设备并转FP32，因此配置`teacher_output_offload=cpu, chunk8`不代表loss以8token分块。teacher前向输出存CPU与损失阶段GPU峰值是不同条件；启用mass/CE等分支时按真实B4/seq1024/accum4短测验收，不能从参数名推断峰值。当前mass/CE3/CE5在FP32/BF16的独立公式和学生梯度对照通过；CE同样受prompt权重作用。此观察不表示chunk配置改变了损失公式，也不提供精度提升结论。[真实流程和证据](../records/e2e_0910/2026-09-30_focused_b16_search.md)


2026-10-03沿49e原HF fresh七类remaining-prefix轨迹，只加入现有norm all、norm_lr1e-4和norm FP32训练，保持block32/D32/S2、每类500步、task50及可融合r12策略；最终完整固定64/32均分66.57623885→65.12216717%，实际252投影2.456988791bpw不变、无外置LoRA。七轮均日志记录308,224可训norm参数，原生保存145norm/BF16共616,448bytes，作为原骨干权重不增加存储；两份raw fresh strict、全24742题、46源码与实际normalized均核实。该具体配方退役，不能由参数增多推断更好恢复，也不否定其他norm子集/LR；不声称本次逐张量验证了所有norm更新。原生保存支持不能替代现有E2E每400步独立export/audit对norm none的限制。[实际条件、结果与证据](../records/qwen3_w25/2026-09-30_goal.md)的10:00 norm-all。

2026-10-04，同552282起点的正式2000步对照仅加入final norm训练：task50、纯mass-KL K100/prompt0.3、B4/max1024，decoder LR1e-5、final norm FP32 LR1e-4。每400步完整八任务均分依次65.81635610/65.48331180/66.13337070/66.78289462/66.82591641%，全部低于起点67.13033275%；相对同配方decoder-only控制，400步高0.145863pp，后四点低0.226740/0.097380/0.243896/0.205786pp。五点3423张量保存/fresh、1910冻结和504码不变均通过，final norm精确BF16投射，末点471/4096元素的更新保留到部署，实际2.456988791bpw不变。因此退役的是这个LR/task/预算组合，不能把首点小幅正差当作持续收益，也不能外推为norm训练无效；上述旧norm-none导出限制已由本轮隔离norm导出路径补齐。[正式五点、保存审计与收尾证据](../records/qwen3_w25/2026-09-30_goal.md)的finalnorm正式收尾。


2026-10-03，同552282起点新2000步CE50/task100/prompt0纯decoder蒸馏（fresh optimizer/cosine/warmup20）在1200/1600/2000完整八任务均分64.8401/66.3682/66.5887%，虽仍低于初始67.1303%，末段仍升，不能据此宣称步数已足或路线无效。用户明确要求考虑训练预算不足，后续另设同源8000步预算，warmup80与更长cosine时域是预算变化的组成部分；不要将它写成2000原轨迹exact-resume，也不要把不同task配比/CE系数的结果当成步数单变量。完整原始证据与新候选状态见[Qwen3专题](../records/qwen3_w25/2026-09-30_goal.md)的19:30/19:40记录。


同日对上述552282→CE2000实际日志补核：200个10步窗口实计8000入训序列、1,883,938个非padding输入token，其中未shift response89,776（含模板/EOS），仅相当176597源行数量的4.5301%。此single-source SFT无packing，seed0全量置换后由实际16worker分片并跳过无有效response样本；normalized写在数据构建前的workers0/group_by_length=true不代表真实16/false。无len IterableDataset下Trainer的epoch=1以max_steps作分母，不能据此声称完整遍历数据；预取/跳过和重复题目也使序列数与独立问题覆盖不同。8000计划32000序列而非已完成曝光，其token不能简单由旧结果乘4。判断预算不足应合看实际序列/token、调度与完整任务曲线。[同一专题21:17核查](../records/qwen3_w25/2026-09-30_goal.md)。

该旧CE1600→2000均分虽升0.220498pp，但RTE多6题贡献均分+0.270758pp，余七项平均−0.057440pp，MMLU少25题；末两个400步区间的窗口loss均值0.877865→0.879270，也非持续下降。这些现象支持继续验证而非提前宣称收敛或上限，不能用均分的单段斜率推断所有任务稳定恢复。

## 短答案的标签拟合与教师分布可能不一致（2026-10-04）

固定32条ChatML留出中，14条短响应包含98个包装token和26个答案token。五个真实BF16部署端点的观察中，L0 down联合decoder+code短测把这些短响应KL/token从0.60242降至0.21002，却将标签CE/token从1.03833升至4.24146；仅答案内容也出现KL降、CE升。这不是“短响应完全没学到”的证据，也不能由更低KL直接推断下游准确率更高；该joint完整八任务66.5958%，低于起点67.1303%。

核查了实际位置、模板和词表后，原HF teacher的直接预测显示另一种回答形式：标签C的位置偏好The，标签genes的位置偏好Answer或重述题干，空think结束位置偏好Okay继续思考。14个首答案top1均不同于直接标签，正确标签概率很低；这不证明teacher不知道正确答案，更不是完整八任务退化的唯一因果证明。重复EOS和换行也参与损失，不能将整段response CE冒充语义答案CE。mass-KL聚合top100之外的尾部质量，单个标签CE还不能独立确定其实际梯度方向。

该检查还区分了隔离诊断额外整模型BF16 cast与正式单卡原HF加载：前者将RoPE inv_freq转BF16，后者保留FP32。A/B真实前向确有数值差异，但短响应CE仍13.4658/13.4245，首答案top1一致，主要现象未消失。新诊断应保持实际teacher加载路径，先观察正确标签与top预测、按真实token边界区分答案和格式，再决定是否测试内容/模板对齐；不要仅因短样本token份额低就提高其KD权重。[原始五模型、teacher A/B和逐token证据](../records/qwen3_w25/2026-09-30_goal.md)的03:03及03:22记录；派生原件为`teacher_loading_ab/interpretation.json`。

随后同14条可恢复内容的teacher-only对照，保持后一加载路径，将ChatML改为`user + ASCII空格 + answer`，逐样本答案总NLL全部降低，均值28.5370→4.2221，最小下降15.6258；新分批相对旧B前向的最大样本NLL差0.5228不足以解释这组差值。它支持输入形式会影响直接答案的教师概率，不提供生成正确率或训练收益；裸格式首答案token仅7/14与标签一致。去包装和新增空格是共同变化，答案token从26变27，13个token融合分隔空格，不能逐token相减或称为纯模板单变量。未添加空格的literal分支因9/14存在跨边界融合、丢答案token等问题在CPU阶段取消；不能把按prefix长度切片能运行当作边界正确。后续短测须在两格式上共同验证完整题干、答案字符覆盖和未截断，保持同原始样本，分别报告损失分母及相对初始变化。该teacher-only检查不能判断训练收益，未改变主框架。[完整格式实验与边界证据](../records/qwen3_w25/2026-09-30_goal.md)的03:45记录；派生原件为`teacher_format_probe/interpretation.json`。

同日完成新的配对训练对照：同fresh552、原task训练集固定1024条+32条留出、L0 down、256步，纯mass-KL/prompt0.3；ChatML的decoder-only/joint完整八任务66.8725/66.6518%，bare_space为66.9499/66.8319%，四臂均低于初始67.1303%，同2.45698879bpw。裸格式在两模式下有小幅相对改善，但joint仍低于各自冻结码位控制，不能把+0.1028pp的描述性交互称为码位学习获益。新的32条与旧14条内容无交集，教师答案NLL仍逐条改善；这证明了概率现象的复现，却没有证明净精度恢复。ChatML joint的留出KL下降、32条答案NLL无一改善；bare joint避免了这一级别的ChatML退化，但其bare答案NLL仍略升。此证据将“输入形式冲突”保留为部分解释，不把它写成唯一根因或直接扩入主框架。[完整2×2及原始验收](../records/qwen3_w25/2026-09-30_goal.md)的04:39前后记录。

复用固定token archive测试已有其他loss配方时，训练器从labels现算分母，不代表外部诊断汇总也正确。上述archive缓存的是prompt0.3的分母；CE50/response-only短测须用同一production causal mask按实际prompt0重新计算训练遥测和heldout加权分母，保持archive不变。此时记录的Trainer loss是CE与KL混合总量，不能继续标作纯KL。两格式的response仍分别包含或排除ChatML格式/EOS，因此同样的样本数和alpha不等于同样的token监督；同时改变CE和prompt权重只能评价整体配方，不能拆出单因素归因。


同日同源/同固定数据/同L0 down/同256步补齐CE50/prompt0的完整2×2，ChatML decoder/joint为66.98004898%/67.23828170%，joint高0.25823272pp且高源552约0.10795pp；bare为67.04658171%/66.91719324%，joint仍低0.12938846pp。完整64叶24742题、固定64/32、实际2.45698879bpw及冻结/投射/码库保存审计均通过。这给出了现有联合训练在特定配方下的正向例子，不再笼统沿用“学码位必然较差”；单seed单层短程不证明普遍或显著优势，CE和prompt同时改变也不能认定只有CE是原因。教师在裸格式下更贴近标签，仍未预测训练后完整任务的优胜格式；目前优先沿匹配控制中有正向证据的ChatML CE50继续验证，而非把裸格式接入主框架。[完整对照及条件](../records/qwen3_w25/2026-09-30_goal.md)的05:21–05:40记录。

四臂原始初始状态、完整heldout0与teacher读数一致；同格式batch/token/response分母/LR逐步匹配。ChatML首次翻码41步、loss42步才分歧，joint在1–128/129–256在线训练loss分别低0.00699888/0.01159773。但终态同32条留出中，joint混合总loss高0.00590052（response加权也高0.00664757），答案序列NLL却低0.15104961，21/32改善。不是比较了不同留出数据，也不只是等batch归约：323个response token只有99个语义答案，另224为包装/EOS等；总目标含CE和KL，答案NLL仅答案CE。缺少分项forward记录时，不能反推具体由格式项或KL主导，也不能将在线批loss事后均值称为末态模型对整训练集的固定评估。

随后固定同552起点、ChatML完整输入、1024条训练/32条留出、CE50/prompt0和L0 down的256步预算，仅将训练labels限制到既有`answer_positions`，监督token从11911降至4743；留出labels保持原样。decoder-only/joint完整八任务为67.04741473%/67.10673513%，分别较匹配的完整response控制高0.067366pp、低0.131547pp，两者仍低于起点67.13033275%，同2.45698879bpw/rank0。留出答案序列NLL均改善，但完整response混合loss的等batch均值及token加权均值均上升，答案拟合改善没有稳定转为下游收益。这个单seed、单层、短预算反例不支持将新答案mask接入主框架，也不能认定包装监督是根因或外推其他预算；变化同时涉及监督覆盖与归约分母，并未删除输入中的包装。两个具体端点退役，保留[decoder完整验收](../../../result/qwen3_8b_w25_20260930/format_answer_chatml_decoder_s256/full8/eval/acceptance.json)、[joint完整验收](../../../result/qwen3_8b_w25_20260930/format_answer_chatml_joint_s256/full8/eval/acceptance.json)及相邻原始指标、导出审计和配置。

## 更换任务数据须同时核对实际监督量与采样路径（2026-10-04）

从新成品 69.312980%/MMLU61.024071% 出发，fresh optimizer、原 2000 步调度只执行 400 步，保持 L0 down 联合 decoder/code 和 CE50/prompt0，task100 ChatML 控制为 68.431619%/60.155249%；现有 MMLU auxiliary_train 的 LM 配方为 67.771653%/60.931491%。LM 相对控制 MMLU 高 0.776243 pp，但均分低 0.659966 pp，且两者都未超过来源，不能据专项相对增益直接延长该配方或宣布突破。

两组都处理 1600 个样本，但实际因果监督 token 分别为 18419 和 582250，差 31.6114 倍；LM 逐题全文 labels、没有 packing，还改变裸文本格式及长度分组采样，不能称为数据比例单变量。短程比较同时报告这些变化，并保留学习率时域。Trainer 默认吞吐以配置 max_steps2000 估计本次提前停止的 400 步，约高估 5 倍；实际吞吐需由真实更新数和耗时计算。结论限于本来源、范围、配方和预算，完整原始结果及 token/资源审计见[新阶段记录](../records/w25_transfer/2026-10-04_mmlu65_llama31.md)的 400 步对照及 `short400_comparison.json`。

同日补齐两种数据配方的冻结码对照后，MMLU LM decoder-only为68.683460%/61.422874%，task100 SFT decoder-only为69.268300%/61.066800%。LM的MMLU相对提高50题，但均分低0.584841 pp；这进一步支持专项指标与总体指标必须分开报告，仍不能把格式、监督token与采样差异归为数据比例单因素。LM自身decoder-only比匹配joint均分高0.911806 pp、MMLU多69题；两LM臂实际normalized只模式/输出不同、40个token窗口逐项相同，冻结504码库经实际导出验证。该400步证据支持在本配方先区分码位更新与数据效果，不证明联合码位或MMLU数据普遍无效；两条冻结码配方均未超过来源均分或达到新双阈值，不能据局部MMLU上升直接推广。

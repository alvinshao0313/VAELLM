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

# VAELLM 经验库

当前目录：`cmcc:/home/cmcc/VAELLM/docs/experience/`（2026-10-30 实测）。`star` 为旧服务器用户，旧服务器归档整理日期：2026-09-23。

[全项目入口](../README.md) · [文档管理规范](../MANAGEMENT.md)

SC-CAT通用流程已独立为项目级[压缩流程标准](../sc_cat_protocol.md)与[执行规范](../sc_cat_execution.md)。SC-CAT任务先读标准，再来本目录查技术经验和具体实验；本目录不维护第二份标准正文。

## 先查经验，再设计实验

设计或修改实验前，先按下表确定一级分类和二级专题，再按模块、损失、错误信息和评估目标搜索，阅读命中的经验条目及来源。设计中简要写清：参考了哪条经验、本次如何规避、哪些条件不同、还需要验证什么。没有命中时扩大到相关分类或全部经验并如实记录；需要重做已失败方案时说明改变的条件或新假设。

```bash
# 列出全部分类及专题；分类也接受下表中的中文名。
python tools/docs.py categories
python tools/docs.py categories --category category_recovery
python tools/docs.py search '学习率|步数' --category category_recovery
python tools/docs.py search 'Muon|预算' --category compression --topic fitting
python tools/docs.py search 'mask|截断' --category 数据与监督 --topic supervision
# 跨分类、具体证据和历史仍沿用原检索范围。
python tools/docs.py search '翻码|proxy' --scope lessons
python tools/docs.py search '逐类恢复' --scope records
# 从远程项目根目录执行；有 rg 时优先使用它。
rg -n -i '编码|STE|翻码|proxy' docs/experience/lessons/post_recovery
rg -n -i 'OOM|offload|显存|resume' docs/experience
# 服务器没有 rg 时，无需安装：
grep -RniE '编码|STE|翻码|proxy' docs/experience/lessons
```

## 经验与结果分开

- **lessons/**：可复用经验。每篇明确问题、证据、原因/解释、建议动作、适用边界和来源。
- **records/**：原始实验结果、实现验证、恢复报告和交接记录。配置、指标及复现证据归这里，不能只改名就称为经验。
- **history/**：历史方案和审查材料。计划、勾选项及“READY”不能代替实测；旧命令、授权、路径和“当前状态”仅代表当时。
- **[全项目目录](../INDEX.md)**：持续更新的分类与路径索引；[CATALOG.md](CATALOG.md) 保留首次归档的来源映射。旧路径仅保留跳转，不维护第二份正文。

实验原始 JSON、核心日志、配置、输入模型和正在运行的任务仍在各自结果目录；本文档整理不运行实验、不改变算法，也不更新任何实验的完成状态。

## 按问题查找

采用“一级分类目录 → 二级专题文件 → 三级经验小节”。例如 `lessons/category_recovery/optimization.md` 中的“逐类别恢复的本段损伤与整链净收益”。模型、方法、更新对象及证据状态写在条目条件中，不再平行建模型目录或成功／失败目录；一条经验只有一处正文，跨阶段通过链接关联。

| 一级分类（`--category`） | 二级专题（链接名即 `--topic`） |
| --- | --- |
| 压缩 `compression` | [representation](lessons/compression/representation.md)：表示与结构；[fitting](lessons/compression/fitting.md)：拟合超参与优化器；[losses](lessons/compression/losses.md)：重构／输出损失；[protection](lessons/compression/protection.md)：保护、旋转与补偿；[bit_allocation](lessons/compression/bit_allocation.md)：位宽／阶数分配 |
| 逐类别恢复 `category_recovery` | [sequence](lessons/category_recovery/sequence.md)：顺序、父源与统计；[scope](lessons/category_recovery/scope.md)：更新范围与类别适配；[optimization](lessons/category_recovery/optimization.md)：预算、优化与选点 |
| 压缩后恢复微调 `post_recovery` | [parameters](lessons/post_recovery/parameters.md)：初态、参数范围与保持；[objectives](lessons/post_recovery/objectives.md)：监督目标与能力取舍；[optimization](lessons/post_recovery/optimization.md)：优化器与预算；[codes](lessons/post_recovery/codes.md)：离散码阈值／坐标；[code_search](lessons/post_recovery/code_search.md)：候选与局部选码；[low_rank](lessons/post_recovery/low_rank.md)：低秩恢复；[combination](lessons/post_recovery/combination.md)：阶段组合与合并 |
| 数据与监督 `data` | [sources](lessons/data/sources.md)：来源、覆盖与曝光；[supervision](lessons/data/supervision.md)：模板、mask、归约与日志；[teachers](lessons/data/teachers.md)：教师目标、缓存与模型适配 |
| 评测与诊断 `evaluation` | [protocol](lessons/evaluation/protocol.md)：口径与完整任务；[diagnostics](lessons/evaluation/diagnostics.md)：损伤与能力诊断；[storage](lessons/evaluation/storage.md)：实际位宽与费用 |
| 搜索与对照设计 `experiment_design` | [comparison](lessons/experiment_design/comparison.md)：变量与公平对照；[search](lessons/experiment_design/search.md)：候选、预算与代理排序；[transfer](lessons/experiment_design/transfer.md)：跨模型／条件迁移 |
| 数值与运行性能 `runtime` | [precision](lessons/runtime/precision.md)：数值、梯度与舍入；[memory](lessons/runtime/memory.md)：显存、卸载与缓存；[performance](lessons/runtime/performance.md)：并行、算子与吞吐 |
| 保存与复现 `lifecycle` | [checkpoint](lessons/lifecycle/checkpoint.md)：状态与续训；[export](lessons/lifecycle/export.md)：转换、打包与重载；[reproducibility](lessons/lifecycle/reproducibility.md)：运行身份、并发接续、传输与收尾 |

逐类别恢复指压缩链内为接续后续类别而进行的恢复，可能更新整个已压缩前缀；压缩后恢复微调指从明确压缩基座独立开展的恢复。按所回答的问题归类，不能仅凭“全模型前向”“蒸馏”或方法名称判断。断点恢复归保存与复现，能力恢复归对应训练阶段。阶段专题中的通用机理可被其他阶段引用，但实测收益不能跨阶段直接外推。

`--category` 会直接搜索该分类目录；`--topic` 必须与分类同用并精确匹配专题名。两者用于 lessons，不与 records／history 范围混用；原有 `search`、`--scope`、正则表达式及 `--limit` 用法保留。旧跳转页不重复参与正文检索；若并行任务在跳转区前后补写，补写内容仍能被全经验搜索发现，随后按维护规则归并。

### 旧专题入口（保留原引用）

以下仅帮助从原问题名称追溯；旧文件保留全部标题锚点及新去向，新总结写入上方分类专题。

| 主题 | 经验文档 | 检索词 |
| --- | --- | --- |
| 基线、公平对照、评测口径 | [对照设计与结论边界](lessons/experiment_design.md) | baseline、seed、LR、step、全压、PPL |
| 输出感知目标和辅助损失尺度 | [输出损失与尺度](lessons/output_losses.md) | MSE、AMSE、Gram、block_output、BSQ |
| 残差 LoRA 与初始化/续训 | [残差 LoRA](lessons/residual_lora.md) | additive、replace、PEFT、rank |
| LiftQuant 式恢复与编码优化 | [编码坐标、STE 和更新预算](lessons/liftquant_recovery.md) | proxy、翻码、decoder、round、clamp |
| 打包、导出、重载、精确续训 | [Checkpoint 与数值路径](lessons/checkpoint_lifecycle.md) | v6、packed、BF16、TF32、finalization |
| 精度、显存、offload、多卡 | [资源与数值精度](lessons/memory_precision.md) | OOM、allocated、reserved、autocast |
| token mask、prompt KD、日志统计 | [KD 语义与数据边界](lessons/kd_and_data.md) | causal、padding、prompt、mask、telemetry |
| VAELLM本体超参、敏感条件、存储 | [参数作用与故障诊断](lessons/compression_choices.md#超参作用与故障诊断2026-10-29) | D/K/S、encoder/decoder、BSQ、batch、LR、optimizer、保护、旋转、bpw |
| Mixed-bit 候选、缓存与质量验证 | [Mixed-bit 工程及评测经验](lessons/mixed_bit_pipeline.md) | fingerprint、tokenizer、manifest、worker、KL、PPL |
| 旧配置、运行状态与收尾 | [复现及文档维护](lessons/reproducibility.md) | CLI、历史、清理、复现、日志 |

## 实验完成后的更新流程

分类、结果总结和经验更新采用 [轻量工作流程](WORKFLOW.md)。每次实验结束都检查经验增量；有新结论更新对应分类专题，没有新认识则注明原因。结果按研究主题/日期保存，经验按阶段或领域、问题专题、条目维护，二者双向引用。

## 维护规则

1. 新实验结果进入 `records/`；有可复用结论时，先定位一级分类、二级专题和已有问题条目，更新对应 `lessons/<分类>/<专题>.md`。独立新专题才新建文档并更新本表，不预建空目录或为每次运行新建经验文件，随后刷新全项目自动目录。
2. 先按各篇的“按条件选择动作”表进入问题小节，再按需读证据。条目模板见[实验流程](WORKFLOW.md#经验条目最小内容)，合并与归档规则见[文档规范](../MANAGEMENT.md#经验增量怎样写)；同规则补证据或修订条件，保留未验证解释的假设状态。
3. 不用训练 loss、单层 NMSE、机制冒烟或部分题目分数替代全量下游结论；不同模型、压缩范围、数据和预算不得直接排名。
4. 结论变化时补充日期和替代关系，保留必要证据。旧材料里的“待完成”不代表今天仍未完成；核实当前状态后另写更新。
5. 原始指标保留在结果目录；经验引用唯一来源，避免复制多份指标表、日志或模型。遵守 AGENTS.md 中的收尾授权。
6. 写入前重读最新正文和未提交差异，迁移前后核对内容、相对链接及旧锚点；发现并发新增总结，先保留并合并到适用专题，不能用旧快照覆盖，也不能因位于旧入口就丢弃。不要改写活动实验的状态、协议、代码或产物。整理证据见[本次分类记录](records/docs/2026-11-09_lesson_classification.md)。

## 研究与复现入口

- [当前低比特目标执行计划](records/w25_transfer/current_plan.md)：主目标、两条并行主线、阶段完成条件、当前子任务和资源安排统一维护于此；经验与历史作为依据，不另设当前计划副本。
- [自适应残差阶数训练的4bit基线规范](../adaptive_residual_stage1_protocol.md)：沿SC-CAT初始化搜索，取消INT8保护，所有目标投影权重均为四阶、每阶K=D，先生成良好基线再研究降阶。旧Qwen达标候选带保护，仅作历史参考，不能替代新表示验收。[专题设计与证据](../../experiments/progressive_residual_bsq_20261030/README.md)、[历史计划](../../experiments/progressive_residual_bsq_20261030/PLAN.md)及[论文整理](../../experiments/progressive_residual_bsq_20261030/LITERATURE.md)保存来源，新无保护实验、门控与降阶未启动。
- [Qwen3/Llama 低比特迁移专题](records/w25_transfer/2026-10-04_mmlu65_llama31.md)：按研究问题定位已有实验正文、迁移依据及完整历史；当前精度、位宽、数据边界与完成条件统一见执行计划。
- [Qwen3 8B 原八任务69目标专题](records/qwen3_w25/2026-09-30_goal.md)：原目标的成品复现、完整验收与历史依据；与当前九项跨模型目标分开阅读。
- [LiftQuant编码恢复经验](lessons/liftquant_recovery.md)、[硬码翻转贡献对照](records/experiments/liftquant_recovery/BIT_CONTRIBUTION_20260923.md)：前者按条件选动作，后者保留具体对照证据。

- [0910 checkpoint 端到端 rank8 搜索](records/e2e_0910/2026-09-24_rank8_search.md)；[七天固定数据自动搜索](records/e2e_0910/2026-09-24_seven_day_search.md)：运行标识、预算、验证和结果入口见各正文。
- [0920 checkpoint 的 rank8 DP 顺序搜索](records/e2e_0920/2026-09-24_rank8_search.md)：独立参数组学习率、辅助损失对照、中途最佳及最终导出重载结果；初始模型与0910专题不同。

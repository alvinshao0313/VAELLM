# 真实 C0 上的完整 BR36 与 BR→Muon32

唯一执行安排见[当前计划](../../docs/experience/records/w25_transfer/current_plan.md#完整块预重建后接端到端恢复)。本目录将已通过[混合相邻块短验](../block_reconstruction_cmcc_20261030/README.md#10月31日真实短验结果)的数学与生命周期接成正式入口。代码、真实数据与来源拒绝等CPU检查已完成；真实完整C0尚未生成，没有创建正式模型输出或启动训练／评测。2026-10-31已由[隔离固定接续](../block_reconstruction_sequence_cmcc_20261031/README.md)挂载C0成功事件等待，届时使用GPU3训练、GPU3/4评测；当前等待不占卡。接口可用不等于36层训练或质量已验证。

## 本批固定配方和边界

本批只固定一个可实施配方，不称最优，不提供参数网格。BR只接受新C0完整终态及其三类500恢复、live后缀两阶10K、独立保存审计；B、相邻块short、类别中间状态和改名的BR成品均不作C0替代。

| 项目 | 固定值或规则 |
| --- | --- |
| BR输入 | H96原1152个train身份，32个M32＋128个QA B1，共160个唯一entry；保留原32组引用，不截断或重采样 |
| BR范围 | 层0→35，每块完整七投影的现有decoder；其余码、路由、保护、旋转、norm等固定 |
| BR更新 | 每块原32组各一次，共1152更新；每块fresh AdamW，LR1e-6、betas(.9,.95)、eps1e-8、WD0、joint clip1.5，原500线性日程前32，无warmup |
| BR目标 | 九entry的有效token hidden SSE以FP32累加／归约，分母为本组有效token数×4096；不平均九个不同长度的MSE |
| BR前缀 | 原BF16教师独立推进；学生块D精确提交BF16后才推进自己的前缀，CPU两条hidden流原位更新 |
| 部署路径 | 保留已验完整同设备216native group8与36compact；先完整预热，再只驻留当前块，不新做peer优化 |
| E规则 | BR子UUID上fresh H96／Muon32，M1024一次、Q32/H96各两次、seed137/R0、原500日程前32；κ及QA学生轨迹／HF目标重算 |
| 新增质量预算 | BR中点与BR→E末点各一份完整九项，最多两份；C0及同父直接E已有结果复用 |

BR每块1,053,156有效input-token曝光，全36块37,913,616次**层级**token呈现；唯一文本池仍为708,845有效tokens，不能称37.9M独立训练tokens。按短验暂预留BR单卡约2–3小时工程窗口，另计E和完整九项；这不是完成时刻或质量承诺。完整池CPU双流约15.46GiB，实际正式长程成本仍须记录。

只从同一C0比较`C0→E`与`C0→BR→E`。BR中点掉分本身不取消E；模型、数据、冻结或执行错误才停止受影响依赖。若增加BR阶段最终有净收益，再另列相近实际成本的E-only对照，当前不称等算力优越或协同。E两臂同规则和文档呈现，在线轨迹及实际监督token分别报告，不假称逐值相同。

## 实现和真实来源

`training.py`原生遍历全部32组并记录32次实际更新，直接复用旧`entry_objective`、组／梯度／Adam／位移检查及`runtime`、`rolling`，不调用返回硬编码2步的旧short循环后改报告，也不修改旧`GROUPS`。当前一组实际有效token分母、Adam步数／LR和七投影梯度均记录。

`run.py`逐块写`blocks/layer_00.json`至`layer_35.json`，最终报告绑定36份原件SHA，避免每步复制整段流水。各块记录160项独立T/S前缀、BF16提交、冻结补集和真实费用。末态每层D逐叶SHA必须对应其实际FP32 master的BF16提交；最终fresh再比较全部实际持久字段，不只检查shape或success标记。不存未要求的optimizer／exact-resume侧车，也不设置自动续跑／重试。

`source.py`只认`BR36_H96_32_*`正式记录，核真实C0祖先、全36×32轨迹、数据／160条流关系、1728D实际所有权和1947非D冻结补集、geometry、账本及独立fresh。实际数量从metadata／模块推导，本Qwen拓扑数字不是放过缺失字段的替代。新BR UUID必须区别于C0，不能把short或父模型换名。

`recovery_*.py`使用独立`BR_E32_*` prepared／execution／provenance／fresh schema，保留BR UUID、完整BR record与祖先C0。更新直接调用原成熟`muon_compact_cmcc_20261030.fit.fit`及其optimizer／rollout／lifecycle；只有新进程的`mature.STEPS`设为原0..31，原文件及已挂载C0→E的1021项来源均未改。没有私有数学副本、旧schema冒充或旧学生轨迹复用。

正式读取用一次`configure_readers`核cmcc support并绑定本机HF路径，后续只复核绑定，不重置已安装的BR或BR→E `source/pins/command`。训练、fresh入口在CUDA分配前各检查指定GPU一次；完整评测沿原canonical资源检查。资源占用即停止，不抢占、不定时等待／重试。相关工程边界归入[读取上下文经验](../../docs/experience/lessons/checkpoint_lifecycle.md#cpu构造原权重读取与真实导入上下文)。

## 入口与顺序

调用者先激活`bitvae`，所有实际工作经tmux/nohup后台执行。`C0_DIR`、`C0_REPORT`指向实际完成C0及其报告；`BR_OUT`、`E_OUT`、评测输出必须是新目录。`GPU_ID`及评测卡列表由启动时的唯一计划和实际占用确定，以下是参数形式，**不是已提交作业或可直接复制的当前资源分配**。

| 顺序／依赖 | 模块与主要参数 |
| --- | --- |
| C0完整核证后运行BR | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.run --checkpoint C0_DIR --pipeline-report C0_REPORT --output BR_OUT --gpu GPU_ID` |
| BR结束后的独立读取 | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.fresh --directory BR_OUT --gpu GPU_ID` |
| BR中点完整九项 | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.evaluate --support SUPPORT --checkpoint BR_OUT/final_model --pipeline-report BR_OUT/report.json --output BR_EVAL --gpus GPU_LIST` |
| BR完整fresh后CPU准备E | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.recovery_prepare --checkpoint BR_OUT/final_model --br-report BR_OUT/report.json --output E_OUT` |
| E的32更新 | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.recovery_run --prepared E_OUT/prepared.json --gpu GPU_ID` |
| E独立读取 | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.recovery_fresh --prepared E_OUT/prepared.json --gpu GPU_ID` |
| E完整九项 | `python -B -u -m experiments.block_reconstruction_formal_cmcc_20261031.recovery_evaluate --support SUPPORT --checkpoint E_OUT/BR_E32/final_model --pipeline-report E_OUT/BR_E32/report.json --output E_EVAL --gpus GPU_LIST` |

训练／fresh进程的`CUDA_VISIBLE_DEVICES`须等于指定单卡编号；CPU prepare和评测driver须隐藏CUDA，由评测driver给各worker分配卡。`SUPPORT`为`/data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support`；评测结果须在`/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/`下。BR评测与E训练没有输入依赖，可在资源允许时并行；E评测只等自己的成品和指定资源。

两个评测adapter仅换来源、命令和源码绑定，保持官方full8的64叶／24742样本及完整LongBench v1的21任务／4750样本。验收仍为九项等权、logical≤2.6、相对同协议BF16绝对下降严格<0.9pp；quality `not_met`是有效结果，不是执行失败。

## 当前验证与未验证范围

- [BR CPU证据](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_formal_cmcc_20261031/cpu_v1/validation.json)：真实H96／160entries／32组、已完成混合GPU proof、实际metadata的D/冻结清单、原数值helper复用及拒绝实际B／short／未完成C0；CUDA未初始化。
- [BR→E CPU证据](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_formal_cmcc_20261031/recovery_cpu/validation.json)：真实H96曝光、Muon与BR既有proof、实际1728D／1008矩阵／1947冻结、拒绝short和缺失正式BR；没有构造完整BR成功fixture。
- [源码及资源顺序桥接](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_formal_cmcc_20261031/recovery_cpu_final/validation.json)：当前入口import/CLI、单次资源检查先于CUDA／输出，外部绑定不变；不重复已有数据核验。
- [本机读取与路由核证](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_formal_cmcc_20261031/readers_cpu/validation.json)：修复裸正向读取原先可能落到`/home/star`的问题；真实本机HF inventory252，BR／BR→E装好dispatch后7条实际拒绝路径及重复配置不改路由对象。当前14个新源码的最终SHA已绑定，未加载HF权重或重跑GPU。

独立源码审查已覆盖writer→source→prepare→run→fresh→evaluate；C0缺失时的拒绝不是正向完整流程成功。真正C0上的36层×32更新、BR→E、最终各自完整九项以及跨模型效果都仍未运行／未验证。相同已验mixed拓扑、关键形状和生命周期复用GPU证据；只在实际新增关键路径或异常时补相应检查，不因换UUID重复B短验。

代码、数据来源、预算与解释边界分别固定；模型迁移时重算模型结构／tokenizer／校准统计并重新核相应接口，本目录目前只覆盖Qwen3-8B。旧short的必要报告、配置及原始指标继续保留，已清理的short权重不得重新假称可用父模型。

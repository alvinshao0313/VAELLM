# 新共同父上的 C1 Rice

本入口实现fresh C1 Rice候选。**2026-11-01已完成原后缀、独立fresh及完整九项，controller与三个阶段exit均为0。** 本次同新共同父对照中，C1九项`64.1901653263311`比普通fixed2 C0高`0.60599937612826`个百分点；logical`2.5216473392818286`通过，最终质量仍未达。旧dynamic／Rice完整对照只作为本候选的选择依据，本次结果不再以旧Rice与旧B的混杂差值代替。

## 完整结果与来源核证

正式UUID为`744e3b98-8a09-4a1d-8b30-013db5d73d44`，成品为`/data1/cmcc/VAELLM/artifacts/fresh_rice_cmcc_20261031/formal/final_model`。训练`report.json`为`completed_fresh_C1_Rice_pipeline`／exit0，原三类各500恢复、两类自身live各S2×10K均完成，用时54171.36秒。`strict_reload.json`核3927持久字段及同输入BF16 logits exact；最终216 native＋36 Rice compact、0 dense、rank0。

| 同协议指标 | C0普通两阶 | C1 Rice |
| --- | ---: | ---: |
| 完整九项 | 63.58416595020284 | 64.1901653263311 |
| 完整八项 | 66.3144567034778 | 66.79837863504025 |
| MMLU | 62.39851872952571，8762／14042 | 61.707733941034036，8665／14042 |
| LongBench21 | 41.74183992400309 | 43.32445885665795 |
| logical bpw | 2.4987414816151494 | 2.5216473392818286 |
| packed bpw | 2.5121161419412363 | 2.5350219996079155 |

两臂实际`after_o`／`raw2` UUID、metadata/runtime身份相同；up与down两阶段的实际optimizer初始化记录（参数digest、形状、dtype和LR）、seed31及shuffle500031／600031完全匹配，各自live拟合目标digest不同。这是加入36gate Rice残差并重走自身后缀的配方增量，不是成品直接加patch，也不是同位宽比较：额外logical为0.0229058577bpw，全部Rice载荷已计入。C1完整tensor为4690895025字节。六项提高、ARC-E持平、MMLU与ARC-C下降；不能由综合领先称所有知识能力改善。相对同协议BF16九项68.04714491790848仍低3.8569795916个百分点。

原件为[完整九项与真实账本](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/evaluation/longbench/merged/nine_task_acceptance.json)、[评测终态](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/evaluation/evaluation.json)及[controller终态](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/run.json)。本轮重算C1的64份full8逐题／24742条和21份LB预测／4750条原件SHA，核MMLU正确数、九项汇总与共同评测输入身份；C0沿用此前核证，不重复其全量SHA。LB使用已完成官方scores，不冒称再次执行评分器。acceptance SHA256为`90348a9972bc8f929235cc6a7906906e2060c67ac7251455b724cc8ce43e998d`，实际metadata SHA256为`35eb57f00bcfbff92f57fb046ef3a30216ca07f0393b9cb175cb3f60ac195dae`。

[同成品Muon32](../rice_muon_cmcc_20261031/README.md#完整四格与恢复边界)亦已完成，九项64.89598790075989。C0／C1恢复前后四格支持保留Rice及该恢复配方的有限收益证据；不是跨seed／跨模型或最终达标证明。

只读复用[新共同前驱](../fresh_common_cmcc_20261030/README.md)的正式`after_o` UUID `7eeb82b4-3868-4cc5-9fa3-51870ee55894`及`raw2` UUID `f4669891-9028-4665-85a5-71c26ac45c33`。`sources.bind/recipe`复用[C0严格来源与正式配方](../fresh_fixed2_cmcc_20261030/README.md)，但C1不依赖完整C0结束。所有36gate都从本raw2复制同一compact BF16两阶decoder／packed codes／INT8保护／rotation与sign，再相对自身共同after_o的live gate重算abs005、INT8值、blocked256×256、Rice k=7载荷。保护列排除，完整矩阵ceil(abs005)每层251659项；实际raw2与after_o非gate补集以及替换后非gate字段必须exact。不会读取旧dynamic/Rice权重、层patch、第三阶bank或训练旧第三阶。

Rice消费者、构建器与冻结／训练保存回训／最终无dense／独立fresh路径直接复用`adaptive_class_20261030`和`cmcc_runtime_20261030`已验证实现。新增缺口仅为新来源接线：正式gate恢复前实际构建36份payload，真实保存、256token finite BF16前向、独立进程完整state和同输入BF16 logits exact，并断言CPU/CUDA RNG保持。此检查0训练更新，失败立即退出。正式入口仅在自身进程内将C0通用fresh-check命令绑定到新C1模块，不修改C0文件或仍被活动任务依赖的任何源码。

随后执行原gate500→自身live up S2×10K+500→自身live down S2×10K+500。七类正式配方、五源train、B4/max1024、512×256校准、原teacher／loss／未来dense rank12、FP32 decoder master／BF16计算与最终融合沿C0；future初始化seed31、shuffle500031／600031不变，记录两次真实stage optimizer初始参数digest。最终须为216 native＋36 Rice compact＋0 dense，rank0，计入Rice全部索引／值／scale／zero／header、主码、decoder、保护、rotation等的真实logical≤2.6；packed与全目录字节另外报告。

| 本支新增预算 | 固定数量 |
| --- | --- |
| gate重新拟合／第三阶拟合 | 0／0 |
| 重新生成Rice载荷 | 全36gate |
| 原逐类别恢复 | 3×500，共1500更新 |
| 自身live up/down拟合 | 2×S2×10K，共40000更新 |
| 新来源GPU接线验证 | 0更新、一次整模finite/save/独立fresh |
| 完整质量 | 同一最终UUID：full8的64 leaves／24742样本，LongBench21／4750 |
| 追加Muon32 | 本次0；新Rice来源的正式Muon32适配另行完成后接续 |

`evaluate.py`只接受本支`C1_Rice`正式完成身份和新来源，核36真实Rice载荷、原恢复与自身live后缀、零第三阶、最终独立fresh及真实账本。它一次初始化本地cmcc reader，后续source读取不重置自身dispatch，不退回`/home/star`的旧BASE。没有旧`source_raw_manifest`或虚填的`stage3_fit_actual_updates=10000`。其余完整评测与严格九项验收复用官方实现；合法`not_met`是已完成质量结果。

CPU验证读取真实正式common／36层spec／全七类recipe、恢复512×256 runtime，并核本地HF252库存、完整入口实际N命名空间、正向reader及dispatch保持；旧／short／未完成来源和非C1终态不得冒正式。它不加载模型权重、不初始化CUDA，也不代替新增GPU检查。活动C0的1003项与C0→Muon waiter的1021项pins在修改前后逐项核对，原值在[protected_pins_before.json](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/protected_pins_before.json)。依据[来源和实际live后缀边界](../../docs/experience/lessons/compression_choices.md#顺序先验live目标与逐类别旋转入口)及[真实导入上下文](../../docs/experience/lessons/checkpoint_lifecycle.md#cpu构造原权重读取与真实导入上下文)；不重跑旧两臂三类别2updates。

2026-10-31的[CPU证据](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/cpu/validation.json)及`cpu.exit=0`已完成上述实际检查，活动两份pins全部保持；[单次预审plan](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/plan.json)准备exit0。以下保留准备／启动时的历史验证事实；GPU新来源验证、正式后缀和完整质量现已由上节实际终态闭合。Python源码在prepare后冻结，后续改动必须重新绑定独立plan。

调用者激活已确认`bitvae`，核`which python`和`python -V`。CPU与正式任务均放tmux/nohup。准备本身只核CPU身份并写plan，不使用GPU：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_rice_cmcc_20261031.validate_cpu \
  --output result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/cpu

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_rice_cmcc_20261031.sequence \
  --prepare result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_rice_cmcc_20261031.sequence \
  --run result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/plan.json
```

最后一条已由主线程于2026-10-31 10:52在tmux `fresh_C1_Rice_20261031`启动，不能重复执行；driver502968/start15251008、worker503348/start15251666。启动前[真实CPU验证](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/cpu/validation.json)及`cpu.exit`、`prepare.exit`均为0；Astra审查通过，固定[plan](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/plan.json)绑定1019项来源，原活动1003／1021项未改。

10月31日10:58完成新36层载荷和零更新GPU接线验证：raw2 Rice UUID `ca2a6117-edb9-4ce2-9301-bfe147693f89`，3063个持久字段及同一256token输入BF16 logits独立fresh exact，CPU/CUDA RNG保持；证据在`artifacts/fresh_rice_cmcc_20261031/formal/04_gate_proj/raw2_strict_reload.json`及`formal/report.json`。随后原gate500恢复运行2步，第2步改变11,025,427个FP32 master元素，峰值allocated41,064,623,616 B／reserved41,603,301,376 B。确认实际更新后停止主动监控；此段只记录当时的新父接线及启动证明。文档索引当时已更新；全库check仍有既有历史迁移缺链，新入口及该次新增引用无报错。

2026-10-31 21:56的历史进度为gate／up各500已完成、down136／500、约39.9秒／更新；当时C1的1019项、C1→Muon的1040项和并行BR的1061项源码pins全部保持。此等待状态及约4小时估计已被11月1日完成记录替代，不需重启。用户分工限定本线程GPU0–5，GPU6/7留其它实验。

单次顺序固定为`run→独立CPU真实成品审计→full8+LB21`，历史训练GPU0、评测GPU0／1／2；各自启动点只核空闲一次，技术失败停依赖，不等待／轮询／重试／占卡。计划提前绑定全部来源，flock和单次运行记录避免重复启动。stdout/stderr、各子进程PID／start ticks／命令／退出码和总状态位于[结果目录](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/run.json)；模型输出独立位于`/data1/cmcc/VAELLM/artifacts/fresh_rice_cmcc_20261031/formal`，完整评测实际位于topic根的`evaluation/`，不是`single_20261031/evaluation/`。

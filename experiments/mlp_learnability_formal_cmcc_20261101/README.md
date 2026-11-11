# 新C0上的全MLP可学性对照

研究决定与资源分工以[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md#11月1日固定的c0两臂128预算)为准。本包承接[已有两步机制验证](../mlp_learnability_cmcc_20261030/README.md#两卡短验实测与取舍)，回答同一低比特父模型上，固定码decoder与放开非保护MLP权重分别能学到多少；free-dense是超预算诊断，不是目标成品。

当前实现已落盘，[实际C0／数据／CLI／派发CPU核证](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/cpu/validation.json)已exit0，真实128曝光、原500日程和reader回调保持通过，CUDA未初始化；必要新C0短链已闭合：两臂两次更新、step1/2独立fresh及单卡消费者全部exit0。11月1日正式两臂128训练、64／128四份独立fresh、64完整MMLU与128完整八项均已完成；两臂完整LongBench21／4750与严格九项已完成，外层固定链也已完成。后续按原始运行记录更新，不将代码或CPU通过写成模型质量通过。

## 固定比较与完成条件

共同父为C0 `cbb15c9f-1c1f-4394-817c-65c226d76ce7`，实际路径 `/data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/final_model`。两臂分别从该父fresh初始化，不续旧B或新短验权重。D更新108个MLP投影decoder，码与其余状态冻结；dense从同父实际BF16部署权重的非保护列初始化，保护、attention及非MLP骨干冻结。

两臂使用既有静态H96 TF输入／目标：M1024一次、Q32／H96各两次组成32组，M32＋QA8及原权重不变。groups0–31按原顺序连续四遍，固定128更新；每臂M4096＋QA1024次呈现，4,212,624非padding输入tokens、10,944答案位置。只使用原已核train来源，不引入正式测试训练样本。该对照与在线OPD／Muon配方分开解释。

AdamW betas(.9,.95)、eps1e-8、WD0、foreach=False、一次联合clip1.5，FP32 master／BF16计算、seed137；D基础LR1e-6，dense1e-5。沿原500步日程 `base*(1-(step-1)/500)`，不缩成128步退火。两个步长是固定诊断配方，不称已充分寻优，也不据此声称纯参数容量因果。

0／64／128观察完整固定训练池的分域KL、M gold NLL、正确数及相对父的修损；D另列零更新live／部署差异，dense初始化保持父部署函数。64保存独立BF16快照后沿同一master／Adam／LR／RNG继续训练，不按测试分数择优；128固定终止。训练结束后读64完整MMLU、128完整九项；末点MMLU在九项中复用，原C0完整控制不重评。

最终同时报告训练拟合、训练外MMLU、完整九项、实际位宽和成本。dense有训练外收益而D弱只支持本配方下的参数化／优化路径限制；两者拟合却不迁移时检查目标与数据；两者未充分拟合时不能判容量上限。质量预算结束后不因loss仍下降自动延长。

## 实现与验证边界

- `run.py`、`fit.py`、`readout.py`复用原两卡18／18层模型并行、静态目标、梯度／冻结检查及Adam数学，增加固定128与全池观察。
- `snapshot.py`单独构造部署对象，不替换活跃参数；整个构造、前向与保存过程保护Python／NumPy／CPU／CUDA RNG，并核活跃master、Adam、LR和模式保持。`fresh.py`独立重载实际快照，核完整状态、保护、冻结补集与同两卡BF16输入输出。
- `source.py`只接受本组真实C0来源、原配置、真实快照及fresh证据。D库存为216 native＋36 compact；dense为144 attention native＋108普通BF16 MLP，明确`budget_passed=false`及`final_budget_applicable=false`。祖先核验不重置已安装的评测派发。
- `evaluation_bindings.py`／`evaluate.py`复用官方HFLM、逐题原件、LongBench21和当前严格评分。64只读57科／14042题，不伪造八项；128才读完整64叶／24742题与LB21／4750。同协议BF16和≤2.6／严格<0.9pp阈值不改，dense在完整报告中显式标超预算。汇总返回值用`SystemExit`传播。

已有B的完整两卡机制与峰值证据复用。新C0必要short仅groups2／10两更新，在step1真实保存快照后继续step2，再独立fresh两个快照，覆盖36 compact与新中点路径；短验不充正式源。单卡消费者复用这份真实short末点：两臂都核strict／group8／完整fixture BF16有限前向和持久状态，MP与单卡差异测量后单列；D原单卡完整评测内核已有证据，不重复长文短验，dense新增消费者执行既有RTE两题及LongBench三个固定诊断。两题不称batch32峰值，局部验证不称完整质量。

## 固定运行入口

调用者在项目根目录激活`bitvae`；CPU controller隐藏CUDA，实际worker由固定参数分配卡号。旧MLP包与公共评测源码保持原样，本包显式`--gpus`不改变旧入口默认。所有模型及训练产物在 `/data1/cmcc/VAELLM/artifacts/mlp_learnability_formal_cmcc_20261101`，评测与控制记录在topic result根，避免把artifact目录误传为评测输出。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.mlp_learnability_formal_cmcc_20261101.sequence --prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.mlp_learnability_formal_cmcc_20261101.sequence --run
```

经tmux执行；已有运行记录后不能重复`--run`。`single_20261101/plan.json`绑定真实父、完整源码SHA、预算、实际命令和路径；`run.json`及各阶段日志／退出记录实际进程和状态。无守护、轮询、自动重试或质量分支。

| 阶段 | 固定资源与输出 |
| --- | --- |
| 实际C0两步／step1和step2独立fresh | D GPU0/1，dense GPU2/3；`short_decoder`／`short_dense` |
| 新单卡消费者 | D GPU4，dense GPU5；result中的`consumer_decoder`／`consumer_dense` |
| 两消费者通过后fresh原C0正式128／快照fresh | D GPU0/1，dense GPU2/3；`decoder128`／`dense128` |
| 固定64 MMLU和128完整九项 | D GPU4，dense GPU5；result中各自`midpoint`／`evaluate` |

每个快照位于该臂`snapshots/step_000064`或`step_000128`，包含独立`final_model`、实际`report.json`、状态／fixture及fresh证据。short对应000001／000002。一次短验上限14400秒、正式训练43200秒；更新预算仍仅2／128。旧B下D约3–4小时、dense约2.5–3小时只是前期粗估。新C0短验两组实测更新D为209.29／201.26秒，dense为47.53／28.46秒，D旧预估偏乐观；两组包含最长输入，不能直接线性外推全32组。短验边界读出＋独立构造／保存另约146–149秒。正式完整池观察、初始化、快照及评测另外计入，耗时待正式步速修正，不改变固定更新预算。

任一技术失败保留原件，停止其后继，不跳过失败、改小形状或自动重复训练。两个short消费者都真实通过后才启动本组正式配对；中点评分不门控末点。GPU6／7属于其它实验，本包不使用。

## 证据与当前状态

[消费者CPU语法／真实CLI检查](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/evaluation_cpu_v1.json)已通过，范围仅导入和CLI。来源、数据、实际短链、独立fresh与正式质量的状态以后续原始记录为准，当前没有新增下游效果结论。

[实际C0两臂必要短验汇总](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/short_validation_summary.json)已验证：D／dense两份快照分别3675／2127持久字段，保存、连续Adam、冻结及完整RNG保持，两卡fresh与单卡消费者均通过；两臂MP转单GPU同fixture实测exact，不外推为所有并行路径的逐位门槛。单卡consumer峰及两卡训练／保存区间峰分列，原件保留测量范围。D实际logical2.4987414816151494，dense13.110128651494565；后者只为超预算诊断。

真实C0 dense短验的两组目标均值为`.5240→.3537→.6182`，M训练正确数`52→56→59`、M KL下降，Qasper／Hotpot KL反弹；当步group10也从`.44391`升至`.52788`，不能概括为仅跨组遗忘。连续Adam、有限梯度、冻结和中点状态保持已过，部署W位移与B短验量级相近。Astra据这些[实际短验记录](../../artifacts/mlp_learnability_formal_cmcc_20261101/short_dense/report.json)确认尚无状态损坏或必须改变固定128诊断预算的依据；这也不证明原LR最优或长程稳定。维持预定LR／日程，由0／64／128全池分域读出和固定下游端点判断效果，不新增短测或网格。

同C0已有恢复对照的CPU终态已正确重发布，见[完整四端点](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/four_endpoints.json)；它不替代本组新训练。工程依据见[全MLP两卡资源与部署初态](../../docs/experience/lessons/memory_precision.md#全mlp非保护权重学习的资源与部署初态)，取舍边界见[局部改善与跨任务表现](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)。


固定链于11月1日10:00挂载：tmux `MLP_C0_D_dense_fixed128_20261101`，controller `1173282/start23578208`；D／dense短验worker分别1173872／1173873，start均23581505。[实际状态、命令及各阶段日志](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/single_20261101/run.json)与[固定plan](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/single_20261101/plan.json)记录全部路径；992项源码已冻结，plan SHA `19af7cdb3025d9411891b1581126dc9031ce7c50cfce6bbca9c80a6334413894`。必要短链八个阶段均真实exit0后，10:20自动接续正式128：D worker1179439、dense worker1179438，仍沿原卡号分工；没有重用短验权重或启动占卡进程。

11月2日实核原固定链为`completed_fixed_C0_learning_comparison`，两份完整九项均`complete_full8_full21_current_goal`，原64叶／24742题与LongBench21／4750齐全，合法质量not_met／acceptance exit1，GPU推理及外层执行成功。

| 端点 | MMLU | LongBench21 | 九项均分 | logical bpw |
| --- | ---: | ---: | ---: | ---: |
| 共同C0 | 62.39851872952571 | 41.74183992400309 | 63.58416595020284 | 2.4987414816151494 |
| 固定码D128 | 62.27745335422304 | 43.864596592189265 | 63.9548227506918 | 2.4987414816151494 |
| free-dense128 | 61.50121065375303 | 45.019811881185355 | 64.12712127749855 | 13.110128651494565 |

D128／dense128相对C0九项分别增加0.370656800488959／0.542955327295708pp，但MMLU分别少17／126题；均未满足严格质量目标，dense还超位宽，仅为机制诊断。D正式训练约7.65小时、dense约1.42小时；训练M正确数从811/1024到893／966，目标拟合明显改善不等于训练外知识恢复。64 MMLU分别62.48397664150406／58.83065090442957，64只观察、128固定终点，未按中点评分加步或换终点。

完整原件为[固定D九项](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/decoder128/evaluate/longbench/merged/nine_task_acceptance.json)、[dense九项](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/dense128/evaluate/longbench/merged/nine_task_acceptance.json)，同目录完整raw samples和各自`scores.json`绑定同一UUID。本次复用原严格汇总和状态核证，没有重训、GPU重评或重新逐字节审计全部预测文件。

D128较同C0 Muon32九项低0.19910545415515pp且成本更高，当前不延长该配方；D128末点原仅保留给M4096配对，该配对现已闭合，指标／元数据足够，不再保留其权重。dense比D高0.17229852680675pp但未出现知识净恢复，不能推出固定码D的严格容量上限，也不能将方法称完全无收益。两臂步长／参数化差异仍限定配方结论，不构成纯容量因果；同C0 D-M4096一次收束消融已结束，旧扩池负例并列，不新增dense4096／QA／目标网格。

保留与收尾：原C0继续保护；dense机制诊断已结束，无具体权重后继，原始配置／metadata／snapshot与fresh审计／原始指标／源码保留。先前已核用途后删除D64及dense64／128的三个权重文件，释放分配占用32,441,868,288B（约30.21GiB），见[原精确收尾记录](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_formal_cmcc_20261101/cleanup_20261102.json)。M4096配对闭合后再次核无活动读者，D128末点、旧short两臂四份权重及已用完fresh测试文件随M4096本组统一收尾；新增路径和释放量只计入[覆盖臂清理记录](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/cleanup_20261102.json)，未重复计旧三个权重。配置／metadata／审计／唯一逐步训练指标及原始评估结果保持，元数据中的旧权重身份不代表文件仍可加载。经验见[全MLP资源与部署初态](../../docs/experience/lessons/memory_precision.md#全mlp非保护权重学习的资源与部署初态)及[局部改善与跨任务表现](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)。

# Fresh C1 Rice → H96／Muon32

本包为[fresh C1 Rice](../fresh_rice_cmcc_20261031/README.md)建立隔离接续，不修改其1019项pins、C0的1003项pins或C0→Muon的1021项pins。**2026-11-01真实C1来源prepare、Rice short2／fresh、从原C1重新fresh的正式32／fresh及完整九项全部exit0，整链已完成。** 只读复用[成熟mixed Muon](../muon_compact_cmcc_20261030/README.md)的`fit/lifecycle/optimizer/rollout/inputs`和真实HF／学生loader；NS5、.95 Nesterov、κ校准、auxiliary Adam、联合clip1.5、seed137、H96、原500线性日程前32及OPD目标全部沿用。没有新算法或恢复网格。

## 完整四格与恢复边界

正式父UUID为`744e3b98-8a09-4a1d-8b30-013db5d73d44`，Muon成品为`3ebcc40c-e572-41db-96c9-1dc58be564a9`，位于`/data1/cmcc/VAELLM/artifacts/rice_muon_cmcc_20261031/C1_H96_Muon32/formal32/final_model`。short2另存为`92f895bd-7bbd-497d-80bf-c05e4ab5c3f7`，未用于正式初始化。两次独立fresh均核3927字段exact、2199非D冻结字段exact及全部Rice状态不变，同输入BF16 logits exact。正式训练／导出8047.48秒，报告内`fresh_pending`是训练发布时状态，已被后继`fresh_audit.json:verified`闭合。

| 同协议九项 | 基座 | H96／Muon32后 | 恢复增量 |
| --- | ---: | ---: | ---: |
| C0 fixed2 | 63.58416595020284 | 64.15392820484695 | +0.56976225464411 |
| C1 Rice | 64.1901653263311 | 64.89598790075989 | +0.70582257442879 |

C1→Muon完整八项67.51904359790477，MMLU61.57954707306651（8647／14042），LB21为43.91154232360084。相对自身C1，七项提高，但MMLU少18题、BoolQ少1题；RTE＋3.2490974729pp贡献九项增量中的0.3610108303pp。C0→Muon的MMLU也少1题，其九项主要增益之一是LB＋2.5665297968分。因此本组支持两种父上的综合恢复收益，不能写成Muon保证多任务保持或已追回知识缺口。

恢复后的Rice比恢复后的C0高0.7420596959pp，但MMLU低0.8118501638pp、ARC-C低1.6211604096pp、LB低0.3968273972分。两臂恢复增量之差＋0.1360603198pp只是单seed观察，不证明Rice与Muon具有稳定协同。C0与C1共享真实after_o／raw2及已匹配的后缀初始化，两次Muon使用相同原32组、seed137、原500日程、教师缓存和M输入／混合目标SHA；QA在线轨迹及κ按各自学生重新计算，不能称实际监督tokens逐值相同，也不能把这四格替代Muon对Adam的配对。

Rice恢复前后logical`2.5216473392818286`、packed`2.5350219996079155`、tensor字节4690895025均不变，216 native＋36 Rice compact、0 dense、rank0。成品九项相对同协议BF16仍低3.1511570171pp，严格<0.9质量目标未达；位宽通过。

原件为[完整九项](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/evaluation/longbench/merged/nine_task_acceptance.json)、[评测终态](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/evaluation/evaluation.json)、[整链终态](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/after_C1_complete/run.json)；C0控制使用[GPU5同成品补评](../../result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/C0_H96_Muon32_gpu5_20261031/longbench/merged/nine_task_acceptance.json)，不能读取旧失败目录当缺失控制。本轮核各自实际父UUID／metadata SHA、训练契约与完整评测协议；重算本成品64份full8逐题／24742条及LB21预测／4750条SHA，核MMLU正确数、九项汇总和C0／C1共同评测输入身份。acceptance SHA256为`6060a7b7416874eab0a395168dd4da2abbd333bc538efcd81959174212c16e48`，metadata SHA256为`2501dabf40e8b488e67eac57e1e71815fc032a6886597cfe525f75eda54f70ae`。本次不重复GPU训练或评测，取舍归[相同来源的结构与恢复证据](../../docs/experience/lessons/compression_choices.md#向量级位宽分配的待闭环问题2026-10-29)。

历史来源绑定为C1 driver`502968/start15251008`、初始formal worker`503348/start15251666`和[原控制plan](../../result/qwen3_8b_mmlu65_20261004/fresh_rice_cmcc_20261031/single_20261031/plan.json)。`sequence`已等整个C1 controller真实退出，再核原PID世代／命令／plan SHA、formal／audit／evaluation各实际exit0、C1独立成品审计以及同一final UUID的完整九项。合法质量`not_met`接受，技术失败或不完整停止依赖。提前出现formal final或退出文件不能跳过整个controller；本包只使用0／1／2，当前本线程资源范围0–5，GPU6／7属于用户独立任务。

固定顺序如下，不含调度器、定时器、轮询、自动重试或占卡：

1. 对真实完成的C1 final执行CPU prepare，严格调用C1自己的`evaluate.source`，保留其共同after_o／raw2 UUID链；不冒C0、旧Rice、short或中间类别为新父。
2. GPU0执行原组0／1两个真实完整OPD更新，然后独立进程fresh。它只补Rice subclass在Muon下的生成↔回训／缓存、第二次master／Parameter-storage／momentum／Adam／κ／LR往返、完整保存／冻结／BF16重载缺口，不重复旧B验证或重建已清理旧Rice。
3. 两步审计通过后，从同一个原始C1 final重新`checkpoint.load`，fresh seed137／FP32 masters／optimizer／κ／学生QA轨迹，在GPU0正式执行完整原组0…31。不会接着short2跑30步。
4. 正式32独立fresh后，GPU0／1／2执行新child UUID的官方full8与LB21／4750，并与C1自己的完整九项比较。

| 计算项 | 预算 |
| --- | --- |
| Rice工程生命周期验证 | 2更新＋独立fresh，单列；不计入正式32 |
| 正式H96恢复 | 32更新、原500日程前段、一次fresh原始C1起点 |
| 数据呈现 | M1024一次、Qasper32及Hotpot96各两次；每组M32＋QA8 |
| 目标 | M仅原`.5F＋.5gold`混合一次；QA按当前学生生成轨迹在线重算原始full-HF教师全词表KL |
| 完整质量 | 新child同UUID：64 leaves／24742样本＋LongBench21／4750；九项规则不改 |
| 额外C1类别训练／拟合 | 0 |

prepare从实际metadata和当前module归属推导全部decoder叶、native stage矩阵及compact二维矩阵，核1728 D叶／1008独立Muon矩阵的既有几何，实际参数量逐项记录。非D冻结清单由当前`state_dict`扣除D叶得到，实际为3927−1728＝2199，不套旧fixed2的1947。全部36层各8类Rice字段（active blocks、ptr、qvalues、scale、zero、header、offsets、stream）独立digest；码、INT8保护、rotation/signs和其他非D字段统一进入冻结proof。每层length隐式为2，length frame及decoded row／col／value缓存不得持久化；非持久缓存允许按成熟路径重建。

工程2和正式32分别保存真实新UUID，审计完整state、实际冻结集合／Rice状态、无dense的216 native＋36 compact拓扑、finite BF16同输入logits exact和真实账本。主码、decoder、Rice索引／值／scale／zero等计费保持；比较user／packed bits、原权重分母和全模型tensor bytes，完整目录字节另外实报。正式评测只接受本包`fresh_C1_Rice_Muon_*`身份、正式32、真实工程2证据和原C1 lineage；source／configure仅核已初始化本地reader，不能重置本包dispatch或退回旧`/home/star` BASE。

事件等待直接复用`fresh_muon.after_control`已验证的纯`process_identity/open_identity`工具，只等已绑定controller的一次pidfd退出。进程退出后其原子终态写入均已关闭，再读取完整成功发布；无需复制C0 formal-only的inotify逻辑。新包持有自己的固定身份、flock、独占创建的run记录、阶段命令、PID／start ticks、日志和退出码。

调用者先激活已确认`bitvae`，核`which python`和`python -V`。CPU测试及任何后续任务均置于tmux/nohup；prepare只写预审plan，不挂载等待器或用GPU：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.rice_muon_cmcc_20261031.validate_cpu \
  --output result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/cpu

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.rice_muon_cmcc_20261031.sequence --prepare

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.rice_muon_cmcc_20261031.sequence --run
```

最后一条已实际执行并完成，禁止重复启动。预审plan、总状态及阶段stdout/stderr／退出码位于`result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/after_C1_complete`；实际源prepare与`short2/formal32`独立模型目录位于`/data1/cmcc/VAELLM/artifacts/rice_muon_cmcc_20261031/C1_H96_Muon32`；完整评测位于topic根的`evaluation/`。每个GPU阶段启动点一次检查实际空闲，资源冲突停止，不等待重试。

准备时CPU仅核真实在途C1身份、36层Rice metadata、本地HF252库存、原H96数据及完整入口N命名空间／source dispatch，复用旧fixed2两步数学证据，拒绝未完成父和short假终态。它没有加载模型权重或伪造未来final／GPU成功；新增Rice生命周期、真实final布局与冻结数、正式32及完整质量随后按固定顺序完成，证据见上节。条件与证据边界依据[混合拓扑与Muon经验](../../docs/experience/lessons/checkpoint_lifecycle.md#compact依赖闭合与native前缀转换)，原字段证据以[成熟短验正文](../muon_compact_cmcc_20261030/README.md#实际结果2026-10-30已完成)为准。

2026-10-31的[真实CPU验证](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/cpu/validation.json)及`cpu.exit=0`完成以上准备范围；C1／C0／Muon的1019／1003／1021项活动pins全部保持。[固定plan](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/after_C1_complete/plan.json)绑定1040项源码，SHA为`0534f08587f77ba0857c46a41123d8ce00de8e09ff8bb3b14af7c15364eddfd6`，`prepare.exit=0`。10月31日11:11在tmux `C1_Rice_Muon32_after_complete_20261031`启动，PID`520956/start15367922`；当时`waiting_complete_C1_controller`及pidfd绑定`502968`是历史等待证据。现[整链状态](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/after_C1_complete/run.json)为`completed_actual_fresh_C1_Rice_Muon32_pair`／exit0。日志为同目录`continuation.log`，shell退出为`continuation.exit`；不据旧等待记录重复启动。

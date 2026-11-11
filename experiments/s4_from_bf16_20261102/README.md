# 从原始 BF16 完整拟合 S4 与逐类累计前缀恢复

11月3日用户更新的[第一阶段质量目标](../progressive_residual_bsq_20261030/PLAN.md#第一阶段当前质量目标11月3日确认)为约4bit主码率下九项均值相对BF16下降严格<0.4pp、MMLU最多掉1pp；本轮按该目标验收，保留启动时的原冻结预算／算法不变。

这是第一阶段配方探索的一次新完整候选，必要短验、从原始BF16新起的正式七类拟合／恢复及同UUID完整九项现均已完成、整链exit0；质量未达第一阶段新目标，不称已定稿或最优。模型从本机原始 `/data1/cmcc/models/Qwen/Qwen3-8B` 的 BF16 factory 构建，不读取旧 `b120`、sequential56 或 terminal56 的码、decoder 或权重作初始化。旧成品只作外部结果对照，第二阶段门控／降位暂缓。唯一实时计划仍在[当前执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。

七类按 q/k/v/o/gate/up/down 顺序执行本轮真实 M36、S4、每阶10000更新拟合，保留原 D/K、结构、原生 BSQ、保护、旋转、batch、优化器及10000步拟合日程。一次保护校准来自 `vicgalle/alpaca-gpt4` 固定 train／512×256／seed31；统计不随类别恢复刷新。未来类别始终是本次原始 BF16 权重。每类真实拟合／安装后，在原生类别边界保存前显式插入累计前缀恢复，再将真实 BF16 终态交给下一类。原生 dispatch 参数仍为 `none`，仅表示不用其仅支持 current-only 或其它原生配方的恢复分派；本候选实际有自定义 prefix 恢复，合同、日志和原生 checkpoint `distill_stage/history` 明确记录，不能按该参数宣称无恢复。

恢复只训练全部已压缩的 `36*j` 个原生 packed S4 decoder。段内 FP32 masters、BF16 compute、18／18 双卡映射，AdamW β=(.9,.95)、eps1e-8、WD0、foreach=False、joint clip1.5，LR `1e-6*(1-(step-1)/500)`。同一类别追加预算保持同一 live Adam；下一类才将已恢复 prefix 提交 BF16 并重新建 Adam。码、INT8保护、旋转、future dense、embedding/head/norm 等完整冻结补集均不更新。恢复／观察使用独立 seed137 随机流，结束后恢复 Python／NumPy／CPU／双 CUDA 的原拟合随机状态，避免恢复／评测意外扰动后继 VAE 初始化。

直接只读复用[已核证 M4096 静态 BF16 教师缓存](../mlp_m4096_cmcc_20261101/README.md)，不继承其 C0 学生权重。M 前1024已混合目标原样复用，新3072只混一次 `.5F+.5gold`；QA 使用原 Qasper32／Hotpot96 gold-prefix full-vocabulary KL，保留域／文档权重和实际归约，无 OPD。M 来源是 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` 的 `auxiliary_train/train`，MIT；QA 来源、版本、split、许可由同一实际 data contract 绑定。每128组为4096次M呈现和1024次QA呈现（128唯一QA各8次），输入4,223,364非padding tokens、10,944答案位置。循环采用 `(step-1)%128`，64只覆盖M前2048；256／384分别重放该128组两／三遍，不能称所有预算都将4096唯一M恰好用一次。

## 按本轮损失自适应类别预算

全部决定使用本轮完整57学科／14042题 MMLU 的 pooled 正确题数，q 的上一节点为同协议 BF16 正确10251题。令新增压缩损失 `D=max(0,previous_post_correct−pre_correct)`：D<36先64更新，36≤D<85先128，D≥85先256；down至少128。完整读出节点为64／128／256／384，初始预算内不因质量早停。到预算点，只有距压缩前仍缺至少15题、且最近相邻完整节点净增至少15题，才追加到下一节点，384为单类别硬上限；64与pre0比较，128与64、256与128、384与256比较。这是预先固定的开发启发式，不是显著性检验，不做逐点择优、回滚或全链最佳点拼接。

每类恢复前／后均完整 MMLU B64，各预算节点也完整 MMLU；真正当前终态进入下一类。最终同一 down 终态的 MMLU 复用到 seven B32＋LongBench21／4750完整九项。正式质量涨跌按已定预算规则处理，技术或协议失败立即停止受影响固定链，无自动重试或另开网格。固定 train 观察 panel 为 groups2／10／34／42，按旧／新增M两块报告 gold 正确数、NLL、KL与同身份修／损，QA分域；其128M和32次QA呈现／16唯一只用于方向观察，不称 heldout 或全4096池充分拟合证明。

## 新关键路径验证与入口

原生拟合、已有消费者和合法共享数据证据只读复用，不重做全 CPU 模型／旧缓存审计。新增 CPU 检查只涵盖真实 parser／导入／新源绑定和预算35／36、84／85、15、384等整数边界。真实 GPU short 保留完整36层、M36、正式batch、S4全量编码，每阶只2更新；每类prefix至少1次真实恢复，末全252 decoder执行 groups2／10／45，覆盖原长QA10593和新增M32×1222。q恢复后下一次k拟合核全部非当前字段exact，future dense逐项对本轮factory，恢复核完整冻结补集。

末段第2更新触发正式同一独立BF16快照路径，在真实训练驻留下独立fresh及HFLM短入口，随后恢复真正保存的 named FP32 masters／Adam／RNG并执行group45的第3更新。正式中点继续原liveAdam；短状态重载仅验证真实续训路径。读者记录实际forward batch／长度及自身峰值，调用方另记暂停训练进程实际驻留；不将配置B64或`limit=1`自动称为真实B64峰。短评分不驱动正式预算、不作质量成绩。新路径通过后才由同一固定后台链从原始BF16 fresh启动正式。

```bash
# 调用者先激活 bitvae，核 which python 与 python -V。
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_from_bf16_20261102.run --prepare /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/<unique_run>
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_from_bf16_20261102.run --plan /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/<unique_run>/plan.json
```

命令由已激活环境的 tmux／nohup 调用者运行、stdin断开、stdout/stderr与退出落盘。本包只用物理 GPU6／7，不改旧活动源码／配置／pins，不触碰0–5。本轮正式所有实际 argv、PID／代次、SHA、阶段预算／决策、实际曝光、训练／读出状态与成本写同一run目录；实际标识见下文。预算未定前的23–27小时是原固定7×128线性估计，已由下述分类别实测估时替代；后几类单步更慢，自适应更新量与中评次数也会改变成本，不能将前几类单步耗时线性套到全链。

`bindings.py` 固定来源与合同，`budget.py` 管整数规则，`fit.py` 只编排原 native fit 与一次类别hook，`recovery.py` 管prefix更新／快照／真实状态，`evaluate.py` 复用原评测API，`run.py` 管固定short→正式→完整九项。最终逻辑／packed位宽须由实际v6账本核4.094031292459239bpw，包含decoder、保护值／indices／scales和旋转；embedding/head/norm另报。它仍是4bit研究父，不满足最终≤2.6bit要求，最终质量以完整九项实测为准。

前6类pre载荷保留到该类别最后一个快照完成读取；中点评测权重与早期post完成读出且无后继引用后只留配置／metadata／expected tensor hashes／原始指标与样本／核心日志，按精确路径清理。运行中真实保存末段FP32／Adam／RNG／游标及其`down/pre`来源身份，状态读回证据保留；本轮结束后无续训用途的三份二进制状态已按下文清理，不再宣称可exact-resume。`down/pre`底座权重／runtime保留给已明确的新增路径夹具及fresh半LR诊断，最终成品保留交付／完整对照。旧正式模型不在本包清理范围。七类拟合／恢复、完整MMLU／seven及LongBench21／4750全部完成；末态九项与新目标结论如下。

## 完整结果与本轮结论

2026-11-03 17:02:01（北京时间）整链自然完成，`run.json=complete_fresh_BF16_S4_algorithm_trial`、`run.exit=0`、`continuation.exit=0`；正式fit、末态seven、两份LB预测／绑定及合并均正常退出。最终UUID **`b2cced0f-32c4-4969-a086-c380827eff31`**，完整MMLU57学科／14042题、seven及LongBench21任务／4750条绑定同一模型。1101个source/data pins收尾重算全部一致；旧无恢复控制与本轮BF16分数字典完全相同。[权威完整结果](../../artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/trial_20261102_1100/formal/result.json)与`formal/longbench/merged/nine_task_acceptance.json`内容一致；不能用源merge_plan保留的合并前status代替终态结果／退出。

| 指标 | 无恢复S4 | 本轮逐类恢复S4 | BF16 | 本轮−无恢复pp |
| --- | ---: | ---: | ---: | ---: |
| mmlu | 70.45292693 | 70.72354365 | 73.00242131 | +0.27061672 |
| arc_challenge | 54.52218430 | 54.77815700 | 56.56996587 | +0.25597270 |
| arc_easy | 79.50336700 | 79.37710438 | 80.80808081 | -0.12626263 |
| boolq | 85.77981651 | 86.54434251 | 86.57492355 | +0.76452599 |
| openbookqa | 40.20000000 | 41.40000000 | 41.80000000 | +1.20000000 |
| piqa | 76.87704026 | 77.42110990 | 77.52992383 | +0.54406964 |
| rte | 77.25631769 | 77.97833935 | 77.97833935 | +0.72202166 |
| winogrande | 66.69297553 | 69.61325967 | 68.50828729 | +2.92028414 |
| longbench | 48.67483515 | 49.13507796 | 49.65236225 | +0.46024282 |
| 九项均值 | 66.66216260 | 67.44121494 | 68.04714492 | +0.77905234 |

本轮九项较无恢复S4 **+0.7790523375262pp**，但仍比BF16低 **0.60592998215453pp**，按更新后的<0.4pp目标仍需再提高>0.20592998215453pp；MMLU仅净多38题，较BF16少320题／2.27887765275602pp，距第一阶段至少10111题仍差180题。八项中除ARC-E外均较无恢复提高，LongBench也提高；这是一条有完整综合收益的候选，但九项下降<0.4pp与M最多掉1pp两条件均未过，不把相对收益写成达标。两轮是独立完整累计流程，差值不分解为各类别恢复的独立因果，单seed不宣称稳定性。

实际投影logical／packed均 **4.094031292459239bpw**，含decoder、保护值／索引／scale、码与旋转，padding为0，完整tensor字节为6044459520；权重文件6046045505B。正式fit含段内读出 **91103.113秒／25.3064h**，从short开始到整链结束107969.194秒／29.9914h；最终fit退出后的nine尾段约3.15h。实际类别恢复64／64／128／64／64／128／128共640更新；各类decoder累计时钟640／576／512／384／320／256／128，完整252联合只128。总呈现M20480、Qasper1280、Hotpot3840，非padding输入21131536 tokens、有效答案位置54720；重放仍用同4096唯一M与128唯一QA，输入量不等于监督位置或知识覆盖。

本配置停止追加训练；保留它作为下一次only skip-up完整候选的主要质量控制。下一步只取消up后prefix恢复，gate、down算法／日程与其它条件保持，正式从原始BF16新起；完整方案／停止和转向条件见[阶段计划](../progressive_residual_bsq_20261030/PLAN.md#未达目标时的有界候选顺序)。末段64→128退化使后续同父步长诊断更有价值，本轮不同时改down而混两个变量。该下一候选已由用户授权推进，独立实现与新增路径验证见[only skip-up记录](../s4_skip_up_20261103/README.md)，实际启动／状态由唯一计划维护。

新增经验已归并至[预算、激活时钟与局部停止参照](../../docs/experience/lessons/experiment_design.md#更新曝光与学习率时域的预算充分性)和[完整综合收益与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。up／down负例仅限制这条prefix恢复轨迹，不识别“MLP不能恢复”、数据缺课、知识已丢失或优化器为唯一根因；更长预算与其它配方保持未验证。

## 当前实际运行

当前 `R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/trial_20261102_1100`，plan SHA **`bf2a9d24e7bb6e2e437bb9f3d77afc7f1921edab51bee606ac6b9323335de8b3`**。11:02在bitvae／Python3.11.16的tmux **`S4_from_BF16_adaptive_20261102_1102`** 启动`run --plan R/plan.json`，CPU driver **1642913/start32593296**、short fit **1643248/start32593974**，mask6,7；short已exit0，正式worker **1699709/start33142682** 于12:34启动。启动前两卡各1MiB、无compute进程，0–5任务未动。

首轮 `trial_20261102_1048` 的CPU prepare曾核1101pins、真实native parser／HFLM导入与预算整数边界，CUDAfalse；10:50启动后真实从HF校准及完成q的S4×2全量编码。其q/pre短M以每科1题实际只产生 `[57,293]` 一次forward，B57、reader peak allocated27,181,236,224B／reserved28,902,948,864B，虽然该评分exit0，却未覆盖正式B64的关键驻留条件。为修正明确缺证，已先停止自启driver1634820/start32520199，再停short1635188/start32520873及其compile子进程；`continuation.exit=143`、`external_interruption.json`记录实际中止，正式从未启动，不把旧`run.json`未完成状态改成通过。

新trial只将`evaluate.py`的short `lm_limit=1→2`，正式算法／评测代码路径不变。增量CPU proof用逆替换后SHA精确还原旧源码，核唯一变化、当前1101pins和真实API导入，复用未变parser／数据／预算证据，不重做CPU模型／旧缓存；新SHA为`ab32bfcd8c1e0260d948c1fec8a47150a60130af69bd9905d850b511d5ad1ab3`，CUDAfalse。新short从原始HF重新执行，后续按实际B64、reader峰和暂停父驻留核余量；不将两个版本合称一次通过。

旧q/pre模型、runtime和六个tokenizer副本已在进程退出、无后继引用后精确删除8文件，实际15,521,995,030B（14.456GiB）；其config／metadata／expected hashes、原始B57评分与样本、真实shape／峰、日志／退出／原source pins保留。共享数据与旧正式S4／56成品均未清理；新trial只按前述规则清理已完成且无后继引用的独立模型。在途产物与实际后继底座保持保护。实际命令／代次／阶段退出见新`R/run.json`，stdout/stderr为`R/continuation.log`及`R/short_fit.log`。只有当前short完整成功才从HF新起正式，不消费旧短态；正式参数／资源／更新推进另核后交付，技术失败停止受影响依赖、无自动重试。

本次有限核证已完成 q→k 关键衔接：新q/pre每科2题的读者实际为 `[64,327]`＋`[50,74]`，reader peak allocated29,750,254,592B／reserved31,484,542,976B，exit0。q累计prefix的首个真实恢复更新为step1／group2、36投影decoder梯度均有限且非零、LR1e-6、48.245195秒，训练峰allocated15,229,575,680／15,072,512,000B；更新后allocated7,970,531,328／7,970,539,520B。整个q恢复段191.406987秒（含固定train panel），完整冻结补集exact，BF16提交有真实decoder值改变；q/post短M也exit0。随后本轮k四阶×2及全量编码完成17.146614秒，安装后全部非当前持久字段exact，已恢复q BF16、future dense与骨干完整保持。q的pre／post独立模型已完成各自用途并按前述规则清理，配置／metadata／expected与raw读出保留；不将短2步bank的目标值或短M分数当质量结果。

截至2026-11-03 17:02核证：上述完整结果、整链exit0及1101pins一致已闭合；driver1642913/start32593296与正式worker1699709/start33142682均退役，GPU6／7空闲。实际命令、plan SHA、PID代次及原短验事实保留如下；它们是来源与生命周期证据，不作当前在途任务。

**两轮逐类结果对照（q至up为11:10核证；down终态于14:20补核）**

旧无恢复控制的[原始七节点](../../artifacts/progressive_residual_bsq_20261030/independent_stage1_20261031_2157/formal/nodes)与[本轮节点](../../artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/trial_20261102_1100/formal/nodes)分开保存。本轮每类的`pre/evaluation/mmlu.json`、`post/evaluation/mmlu.json`及`recovery/step_*/evaluation/mmlu.json`分别为恢复前、段末及节点读出，`recovery/budget.json`／`training.json`记录实际停点与曝光。本表全部分数来自完整57学科／14042题、0-shot、B64、无limit，原始与有效样本数均已复核。

| 累计节点新增类别 | 旧无恢复MMLU% | 本轮恢复前MMLU% | 本轮恢复后MMLU% | 本段恢复增益pp | 本段实际更新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q | 72.881356 | 72.881356 | 72.888477 | +0.007121 | 64 |
| k | 72.888477 | 72.788777 | 72.824384 | +0.035607 | 64 |
| v | 72.368608 | 72.432702 | 72.546646 | +0.113944 | 128 |
| o | 71.649338 | 72.511038 | 72.596496 | +0.085458 | 64 |
| gate | 71.343114 | 72.347244 | 72.325880 | -0.021364 | 64 |
| up | 70.951431 | 71.962683 | 71.407207 | -0.555476 | 128 |
| down | 70.452927 | 70.994160 | 70.723544 | -0.270617 | 128 |

BF16同协议73.00242130750605%。两列跨轮比较是各自累计链，包含前类恢复、重新拟合及其交互；本段恢复收益只用本轮post−pre，不把o较旧链高约.947159pp全归因于o本段蒸馏。q／k的1／5题变化及gate少3题不作显著性结论。

关键预算曲线：v为pre10171→64步10191→128步10187题（72.432702→72.575132→72.546646）；up为10105→10058→10027题（71.962683→71.627973→71.407207），128步比pre少78题／.555476pp，64→128再少31题。当前所有已完成类别实际仅64或128次更新，没有256步质量结果；不能称256／384充分恢复，更不能把这轮缺口单归因于步数少。按每次真实更新M32计，256步为8192次M呈现，只是batch32×2000的64000次的12.8%；本轮256还含2048次QA呈现与重复train池，不能等同旧训练目标／token／成本。

up固定train panel在64→128时，旧64题gold NLL .13261260→.18827295、正确60→58，新64题NLL .17761156→.20619539、正确60→59；Qasper KL .03625071→.02183785、Hotpot KL .00896344→.01115988，目标方向并不一致。该小观察面板只支持“目前不是所有监督持续共同改善”，不能据其宣布完整训练池充分拟合、优化已收敛或模型不可恢复。本轮已按原规则完整结束，用户中途担忧未被当作修改冻结参数的授权；后继按已确认的单变量候选推进。

**11月3日新目标下的停止边界**

本轮down64为10008题、上一up/post为10027题。按原规则，128要追加须同时比64多至少15题（≥10023）且距previous仍差至少15题（≤10012），条件无交集；本轮实际已在128结束、终态9931题。这不等于达到新目标所需至少10111题或充分恢复。后继优先候选及局部复用／完整整链边界见[当前阶段协议](../progressive_residual_bsq_20261030/PLAN.md#未达目标时的有界候选顺序)；本次未修改训练代码／参数或新增GPU任务。

**末态评测的决定与保留用途**

14:20保留两份LongBench的理由是为下一整链候选补齐综合对照；17:02现已全部闭合，无需补评旧中点。down曲线9969→10008→9931题，64→128少77、终态较pre少38；本配方不再追加训练或无关复评。最终模型作为本批唯一恢复阶段交付／完整对照保留；`down/pre`完整底座当前供新skip路径验证夹具及已明确的同父fresh半LR诊断保留；这些任务不读取旧optimizer状态。原始BF16／共享train缓存及旧b120／seq56／terminal56不在本次清理范围；本次没有把共享磁盘变化计作清理收益。

**本轮状态清理（11月3日收尾）**

执行代理已确认新skip路径只读取旧`formal/nodes/down_proj/pre/model`，不读旧训练状态。旧整链已exit0、PID代次已退，实际FD／maps无以下文件引用；已选配方不再续训，后续半LR是同父fresh Adam对照。故清理三份已用完的optimizer／FP32 master／RNG二进制状态，保留对应训练／恢复JSON、来源SHA、原始指标、核心日志和代码。删除前精确盘点如下，路径相对于本R：

| 路径 | logical bytes | allocated bytes | 删除前SHA256 |
| --- | ---: | ---: | --- |
| `formal/nodes/down_proj/recovery/training_state.pt` | 120496248 | 120500224 | `2fc6a95763eddc7e17052d46bb25286f2b12b3f461273213567a0398dfd4a01b` |
| `short/nodes/down_proj/recovery/training_state.pt` | 120495608 | 120496128 | `974688ec13bd5a7ead059732bdae32ffdc8b7109b8d8b01ff65d150b396f8c29` |
| `short/nodes/down_proj/recovery/step_000002/training_state.pt` | 120495544 | 120496128 | `536bd6e6b3a066216208ac2cd2a7061a1aac055d25df746e44a1e8663677b464` |

逻辑合计361487400B，实际本次回收allocated 361492480B（344.746MiB）；未计共享磁盘变化。旧pre夹具底座、final交付模型、原始数据／HF／共享教师缓存和全部结果证据受保护，保留状态见上文。

# S4末down同父恢复：M-only与半学习率

11月5日09:42核实：本批两臂128步训练、各自完整九项、父0补评和原control128诊断均已完成，全部正常退出；最晚后继于北京时间08:56:07结束。半LR综合最好，但九项均值下降0.41588459pp、MMLU下降1.77325167pp，仍未达本阶段两个目标。实际排程唯一维护于[执行计划](../../docs/experience/records/w25_transfer/current_plan.md)；[S4质量目标](../progressive_residual_bsq_20261030/PLAN.md#第一阶段当前质量目标11月3日确认)保持九项均值下降严格<0.4pp、MMLU最多下降1pp。未启动新的研究批次。

## 完整结果与本批结论

两臂从同一down/pre父fresh Adam新起，均实际128更新、groups0…127，M-only训练QA曝光确为0；冻结补集、3927字段独立fresh与BF16前向通过。三份新增九项均来自各自同一UUID，M＋seven为64叶24742题、LongBench为21任务4750条；原始逐行数量及SHA、LB原始与merged逐文件一致、配置／终态绑定均实读核对，单次attempt全部exit0，无pending。原joint完整九项直接复用。所有投影的实测logical／packed费用均为4.094031292459239bpw，含decoder／保护／旋转，embedding／head另列。

| 模型／固定128终点 | MMLU正确数／14042 | MMLU（%） | 九项均值（%） | 均值相对BF16（pp） | M正确数相对父0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 10251 | 73.00242131 | 68.04714492 | 0 | +244 |
| down压缩后、末段恢复前父0 | 10007 | 71.26477710 | 67.38954114 | −0.65760378 | 0 |
| 原联合恢复，LR1e-6 | 9971 | 71.00840336 | 67.53884896 | −0.50829596 | −36 |
| M-only，LR1e-6 | 9901 | 70.50989887 | 67.31670701 | −0.73043791 | −106 |
| 联合恢复，LR5e-7 | 10002 | 71.22916963 | 67.63126033 | −0.41588459 | −5 |

采用未舍入值判断：半LR相对父0九项+0.24171918794036173pp、相对原joint+0.09241137075234463pp；其均值距严格门槛还差超过0.0158845895079821pp，M距至少10111正确仍差109题。它减少了本轮恢复新增的M损伤，尚未修复压缩已有的知识缺口。M-only相对父0九项−0.07283413030793895pp、M少106题，相对原joint亦少70题，不能将QA训练直接定为伤害M的唯一原因。

64步只作观察：M-only为9950题／70.85885202%，半LR为10021题／71.36447799%；后者到128减少19题。64未评完整九项，不把64的M和128其余任务拼成成品，也不据此直接认定早停配方已验证。

固定诊断每组128题，括号内为gold NLL；训练早半／晚半与未参训同源题分开：

| 臂／更新 | 早半训练题 | 晚半训练题 | 未参训同源题 |
| --- | ---: | ---: | ---: |
| 共同父0 | 117（0.19503） | 112（0.38960） | 113（0.34120） |
| M-only64 | 120（0.13939） | 112（0.45504） | 112（0.39136） |
| M-only128 | 120（0.16566） | 121（0.25663） | 112（0.33819） |
| 半LR64 | 121（0.14650） | 111（0.43053） | 111（0.36225） |
| 半LR128 | 118（0.16178） | 117（0.30800） | 114（0.32916） |
| 原joint128（本批fresh读出） | 120（0.17421） | 121（0.25726） | 112（0.34492） |

这些读出说明在本配方下可改善训练样本拟合，但没有相应的完整M恢复；M-only的训练panel gold准确率／NLL改善仍对应更差正式M；其早半混合目标KL反而从0.0647707升至0.0872451，不能把gold指标改善写成原蒸馏目标整体拟合更好。晚半学入时早半部分收益回退，支持下一步讨论保留与迁移问题；128题不足以代表完整4096池或57学科，不足以证明数据覆盖、遗忘或表示容量中的某个单一根因。删除QA同时改变clip及Adam；新臂单卡、旧control双卡且仅单seed，优胜不能直接写成通用规律。

学科覆盖解释更正（11月5日用户指出后复核）：10月12日已核auxiliary_train来源及空subject，10月15日已做阅读偏重盲审、Med／Web知识来源配对，10月16日已建设4,288,142条57科V2材料池（55科direct、2科related）及实际曝光匹配，并已完成2000步完整恢复对照。不能将“当前S4尚未迁移验证这些配方”写成“此前未研究覆盖／需要从头盘点”。经验明确要求不把aux扩量代表学科扩张，原文见[已有覆盖与schema经验](../../docs/experience/lessons/kd_and_data.md#教师正确率真实知识覆盖与原始schema)、[已有57科重配与终态](../../docs/experience/lessons/kd_and_data.md#连续预算与来源重配的实际曝光)。已有广覆盖条件也未普遍改善九项，尚不能默认它们在S4必胜。

本批数据承接事实：11月1日逐类／末端比较复用成熟H96答案蒸馏，11月2日S4完整候选明确导入`mlp_m4096_cmcc_20261101.data`并复用静态教师缓存，随后skip-up及本批经`prior.data`继续继承；实际使用aux4096＋Qasper32／Hotpot96，没有接入上述57科V2池。这里是旧配方沿用，不是运行时偷偷换池；记录提供了固定对照与缓存复用的依据，没有证明aux4096优于既有知识匹配配方。本批能回答该固定配方下删除QA／减半LR的局部作用，不能代表最佳恢复数据探索或重验学科覆盖。研究衔接应先复用既有来源／作者规则／消费／完整结果，明确旧正文KL与当前四选项混合目标的接口差异，再讨论S4同父迁移；不重新创建已做过的aux内容盘点，不自动启动新数据或训练。旧`qwen_subject57_pool_20261016`、`qwen_c1_subject57_native_pair_20261016`的result README本机指定路径未找到，当前引用的是仍在的经验／历史正文；不能把历史材料视为当前已可直接加载。

当前只读统计仍有效：实际4096个subject全为空且为全池hash抽样；半LR相对父0为24科提高／22科下降／11科持平、净少5题，M-only为19／31／7、净少106题，其中moral_scenarios净少77／895题（−8.60335196pp）。它们不能证明某科训练内容缺席或作为按正式测试逐题选训练样本的依据。

本批按固定128结束。半LR作为本批综合最好的局部恢复候选保留；共同父及原joint控制继续保护。M-only固定点无收益且无后继，其最终model bin／runtime两文件已在原formal_training.log逐路径记录后清理，实释6068785152B（5.652GiB）；metadata／配置／全部原始指标和诊断保留。两臂64中点载荷已由原流程清理，本次未重复计入；未发现需另清的落盘optimizer／master／RNG状态。下一批先讨论目标／数据的迁移与更新保持如何区分，不自动续训、扩大LR网格或启动门控。上游固定前驱下的局部结论仍需最终入选流程从BF16整链验收。

原始结果：`R/m_only/formal/full_nine/result.json`（UUID `38f88bd0-d2a8-4444-9134-5a5628eef937`）、`R/joint_half_lr/formal/full_nine/result.json`（`8a874b00-a3ce-44f5-b869-7af1579cf988`）、`R/m_only/zero_parent/result.json`（父 `fa50bf40…`）；原control见上轮`P/formal/result.json`。三组诊断在各`formal/diagnostic_000000/000064/000128.json`，原control新端点在`R/m_only/control128_diagnostics/diagnostics.json`，原control64的late／未参训缺项保持缺失。下文给出R／P完整路径与复现条件。

M-only正式训练含诊断／中评／保存共4363.8495秒（1.2122h），半LR23617.4973秒（6.5604h）；纯更新均值25.3273／175.8843秒。两单卡controller从短验到全部后继分别47670.0760／46612.1791秒，拥有窗合26.1895153GPUh（含CPU衔接，非纯kernel计时），CPU首次准备失败另列。M-only训练11月4日21:05:32结束、半LR训练11月5日02:34:22结束；半LR九项／队列08:38:28结束，父补评及旧control诊断后GPU6队列08:56:07结束。对应PID代次均已退出，不保留占卡进程。M-only自身九项03:02:25落盘、父0九项08:54:34、control诊断08:56:05；这些为终态文件mtime，native原始时间仍在各绑定记录。正式training.json的`midpoint_readout_followed_by_real_update=False`由实现将该汇总字段固定赋为`short`所致，不能用它判断正式是否回训；实际64中评后step65…128完整，必要短验该字段为true，未改动冻结源码。

## 问题、共同父与控制

已有两轮S4的末段联合恢复都出现MMLU负差，训练小panel却能改善；它支持检验恢复目标／任务混合及更新步长，不说明数据或QA梯度是唯一根因。具体证据和边界见[恢复经验](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。

固定共同父为上一完整skip-up流程的 **down压缩完成、末段联合恢复尚未开始** 状态。前面类别已有恢复包含在父中；本批不重新压缩，也不把局部恢复称为从BF16完成的压缩整链。

`P=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/trial_20261103_1730`

| 角色 | checkpoint | 已验证结果与本批用途 |
| --- | --- | --- |
| 父0／共同初始化 | `P/formal/nodes/down_proj/pre/model`，UUID `fa50bf40-c682-4910-8e72-d024f6b678e4` | MMLU10007/14042、71.26477709727959%；原M复用，本批补seven及完整LB得到父0九项 |
| 原joint128控制 | `P/formal/nodes/down_proj/post/model`，UUID `a6a04164-2782-4b19-965f-410198c3ddde` | MMLU9971/14042、71.00840336134453%；九项67.53884895764816%、LB49.357977830947256；完整九项直接复用，仅补新的可用端点诊断 |

两端模型、runtime、config、tokenizer、原指标／样本均保留。旧joint64模型已按上一批规则清理，只剩真实已有early训练panel及完整M，不能补late／未训练诊断，也不重训第三臂补表。[原完整结果](../s4_skip_up_20261103/README.md#完整结果与本轮结论)保留算法、成本、来源和退出。

## 已确认的两臂

| 新臂 | 物理GPU | 相对原joint控制的预定干预 | 固定训练 |
| --- | --- | --- | --- |
| `m_only` | 6 | 真正删除QA的全部训练前后向；M原系数0.5保持，不重新归一到1 | fresh AdamW，初始LR1e-6，128更新 |
| `joint_half_lr` | 7 | 保留原M＋QA，只把初始LR从1e-6降至5e-7 | fresh AdamW，128更新 |

固定条件：全252个已安装S4投影的1512个decoder参数叶可训，其余码、保护、旋转、embedding／head／骨干等冻结；原生packed decoder、FP32 masters／grad／Adam moments、BF16计算与部署；AdamW betas(.9,.95)、eps1e-8、WD0、foreach=False、全局clip1.5、seed137。每步LR为`base_lr*(1-(step-1)/500)`，原500步线性时域、无warmup；不改为128步衰减到零。

M为固定官方auxiliary_train4096、原顺序groups0…127、每组32题，原四选条件概率目标`.5 teacher+.5 gold`只混一次。joint每组另8个QA文档，原gold-prefix full-vocabulary KL，每文档答案token平均后文档等权；M／QA域系数各.5。QA只有Qasper32／Hotpot96唯一文档，128更新中各重复8次。M-only训练曝光QA必须为0；为比较而做的QA诊断前向单独记账，不计训练曝光。

删除QA同时影响联合clip及Adam更新，故M-only是删除QA训练的综合干预，不能据结果直接认定梯度夹角冲突。新臂各单卡完整模型，原控制训练是双卡18/18；模型、batch、长度、目标及精度不变，设备放置差异如实记录，正常浮点差异不要求跨运行逐位相同。

## 诊断、完整评测与阶段反馈

新两臂均固定128终点，不按64最好分择优、回滚或自适应延长。64完整M仅观察，128同一最终UUID完整M／seven／LB21任务4750；父0已有M复用、其余补评。不得拼不同端点单项最高分。

0／64／128固定三组M读出，均不按正式测试逐题表现选题：

| 组 | 固定身份 | 用途 |
| --- | --- | --- |
| 早半训练题 | groups2/10/34/42，共128道唯一M | 保留旧panel，观察早期已见样本保持；旧QA panel另报 |
| 晚半训练题 | groups66/74/98/106，共128道唯一M | 区分尚未见／见过的新半池拟合 |
| 未参训同源题 | 原官方auxiliary_train中排除训练4096身份后，按已有稳定ID hash选128 | 观察同源训练外迁移；只读gold准确率／NLL，无需teacher缓存，不进入优化 |

未参训诊断不代表57学科的训练外覆盖；正式M仍完整14042题。记录各组准确率、gold NLL及相对父0的修对／弄错数量；不能以小panel或混合loss替代完整M／九项。原joint128模型仍在，可补这些端点诊断，原完整九项不重评；原joint64缺项明确保留缺失。

early沿原panel保留的`weighted_objective`是固定原M＋QA配方的只读诊断，即使在M-only臂也会读QA；它不代表M-only实际训练目标。实际优化只以`training.json`／逐更新loss和训练曝光为准，M-only不训练QA；不同臂训练loss不可直接按大小排名。

诊断的旧Arrow cache已在此前合法收尾中清理。本批从仍保留的同revision官方Parquet／capture构建独立缓存，核99842行及原训练4096文档身份后，按真实原row选未训练题；不下载新版本，不伪造原缓存inode／路径，source记录仍为`cais/mmlu`、revision `c30699e8356da336a370243923dbaf21066bb9fe`、config `auxiliary_train`、官方split `train`。实际源hash、排除集合、选择hash与分词签名随本批产物绑定。

汇报节点为：正式启动及真实推进；64完整M与三组诊断；128训练／fresh与完整M；完整九项／相对父及原控制差值。后台正常后停止主动轮询，用户唤起时核对应原件；日志记录节点不冒充已经向用户汇报。不建监控自动化。

## 必要验证、资源与边界

复用旧S4拟合、静态teacher目标、完整native保存／消费者和原任务协议证据，不重做七类短验或精确resume矩阵。新增验证只覆盖M-only真实删除、诊断合法来源与排除、单卡完整形状／峰值、真实更新、驻留B64中评后回训、保存及独立fresh。最小short拟用M-only组2/45两更新及joint组2/45/10三更新；保留真实M32、QA8及最长10593上下文，不缩模型／长度／参数范围掩盖资源问题。

GPU6／7各臂独立通过必要验证即接正式，无依赖的两臂不互等。GPU0–5不动。有限队列接各自完整九项，父0补评及原control新诊断接GPU6后继，不阻塞独立训练；具体顺序以实际controller和run记录为准，无自动retry／续期／占卡。

工作量固定为两条128步训练、2次64完整M观察、两新128终态完整九项，以及父0缺失评测和规定诊断。按旧双卡吞吐推算约30–36GPUh为工程估计，待单卡实测校准；不是用户指定的总额度或用于提前截断128的硬期限。短验、准备、失败尝试和评测成本另如实记录，不藏在正式训练均速里。

达标仍须九项未舍入均值>67.64714491790848%、M正确数≥10111；父0需净恢复104题。若本批有收益，仅形成此父条件下的局部恢复候选，后续入选变更再讨论从原始BF16的整链验证；新数据训练、额外LR／步数／seed网格、BR、门控或降阶均不在本批。

## 实际运行与产物保留

新源码位于本目录；实际独立输出：

`R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/diagnosis_20261104/trial_20261104_1940`

CPU准备`R/prepare.exit=0`、`CPU_validation.json`通过：同revision官方train99842行、原4096文档exact、128未训练题排除关系及原生选项token核实，留出题长度30–902、未超过已验证M1222；CUDA未初始化。实际train/run/evaluate CLI及ROOT-only终态caller环境核证通过。最终 **1115** 个源／数据pins，plan SHA **`a13dc4df19794bf93eb3e61a0f0ba24102da00273c78320ce7ed5c7d19d349c9`**。Astra窄审无阻断，单卡资源及实际GPU路径还须以short退出为准。

已于11月4日启动独立有限队列：tmux `S4_down_m_only_GPU6_20261104_1942`及`S4_down_joint_half_lr_GPU7_20261104_1942`。启动前6／7各1MiB且无其它compute占用，0–5原任务未动。CPU controller用隐藏CUDA，子训练仅暴露自己的物理卡；实际命令为（调用者先激活bitvae）：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_down_recovery_diagnosis_20261104.run --plan R/plan.json --arm m_only
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_down_recovery_diagnosis_20261104.run --plan R/plan.json --arm joint_half_lr
```

上面`R`替换为实际绝对trial路径，两条分别后台运行。每臂`run.json`记录controller／worker的PID、start_ticks、实际命令与阶段；stdout/stderr为各臂`continuation.log`、`validation_training.log`／`formal_training.log`，退出分别在对应`.exit`或`.outer.exit`；正式实际更新在`formal/updates/step_*.json`与训练日志。序列为short→128训练／fresh→自身完整九项；GPU6再接父0完整九项及原control128新增诊断，全部为有限既定后继。

M-only controller **2845489/start52987714**、原validation worker **2846119/start52988470**；joint controller **2845488/start52987713**、validation worker **2846115/start52988455**。两臂初态early128题的正确数、gold NLL及M／Qasper／Hotpot KL与原双卡同父读出差均0.0，case IDs完全一致；short late／未训练各32只证明技术路径，正式各128不变。

M-only short已完整exit0，655.237秒，实际两更新[2,45]、全252投影梯度finite且非零、QA训练0、完整冻结补集exact；step1驻留独立B64读出后实际完成step2，原生保存／独立fresh **3927** 字段exact且BF16 forward有限，logical／packed仍4.094031292459239bpw。最长M32×1222更新29.829秒，峰allocated37592818688B（37.593GB），两更新均值27.144秒；纯128更新约0.97h，诊断／完整中评及终评另计。其正式worker **2853331/start53054922** 已另从fa50父重新载入，并已核group0／step1／LR1e-6真实更新23.5405秒、峰26483360768B、QA训练0；该臂正常正式推进后停止主动监控。首步M loss 0.011612235568463802与原joint控制同父group0的M项差0.0、32个M身份同，删除QA没有改变M损失系数／目标。

joint short也已完整exit0，1131.483秒，实际[2,45,10]三更新（189.636／178.239／173.942秒），M96／Qasper8／Hotpot16，LR5e-7→4.98e-7。最长QA10593与M32×1222反向、驻留B64后真实回训、完整冻结、3927字段fresh／BF16有限及4.094bpw账本均通过；1115pins仍匹配，Astra最终审无阻断。平均180.606秒／步，纯128更新粗估6.42h、诊断及完整读出另计。正式worker **2859244/start53102512** 已独立从fa50父新起，首group0／step1／LR5e-7真实更新182.973秒、峰26483360768B、实际M32／Qasper8已核；正式三组初态均完整128、limited=False。其首组逐域M／QA loss与旧双卡同父控制差0.0，联合梯度norm差−0.00342941284，保留该设备放置数值边界，不宣称跨设备梯度或完整轨迹逐位相同。两臂PID代次／命令与请求一致，正常后台后均停止主动监控；固定后继继续。新short实际B64的最大长度327，未冒称完整M峰；同消费者已有完整B64／L1010的allocated64320849920B／reserved68847403008B，叠新驻留parent约9642656256B后分别73963506176B／78490059264B，小于实际单卡85899345920B。该既有峰与新的驻留B64回训证据合用，不重复完整M资源审计。 旧峰原件为`P/formal/nodes/down_proj/recovery/step_000064/evaluation/mmlu.json`及`step_000128/evaluation/mmlu.json`的`actual_max_batch`／`actual_max_sequence`／`reader_peak_allocated`／`reader_peak_reserved`字段，均仍保留。

首个CPU准备`trial_20261104_1937`在GPU／训练之前因capture绑定字典格式不匹配而退出1：旧记录含inode／mtime共五字段，本包通用binding为三字段，直接dict相等误判。原文件1442B、SHA `5302a15b8808c807bcba5f7fd05c2d67ea6b5fac8876395fb8e19c0683c98526`及inode／mtime均未改变。修复调用原M4096完整checker，保留原字段／SHA校验；没有改capture、依赖或数据。失败日志／退出保留，新1940trial独立冻结后再准备，无盲重试。

保护共同父、原joint控制、活动依赖、新最终待评模型、原始指标／逐题输出与核心日志。用完的short／64中间模型及无后继状态在证据归并和进程退出后按精确路径清理；配置、metadata、expected_state、source hash和最小复现记录保留。

本批新增short载荷收尾已完成：用完的M-only step1／post及joint step2／post各自原生model bin／runtime共8文件，逻辑24275098508B、实际分配 **24275140608B**（约22.61GiB），已核全部移除；精确删前／删后记录在各臂原`validation_training.log`。本次统计不含旧R1730／R1100任何清理。保留验证metadata／config／expected_state／fresh／原始诊断及核心日志，新formal、共同父／控制、独立aux缓存与heldout仍保护；1937失败证据保留。

启动时估计11月5日上午至中午收齐；实际全批08:56:07完成，完整结果与耗时见本文前部。两臂固定后继均已结束，无新增研究或主动监控。

# G09 并行恢复与阶段价值对照

**2026-10-30已收齐G02/direct/scale32/q64及Muon/G04完整终态，见本页末节。此前“尚未启动／等待／运行中”叙述为10月29日历史快照，已由最终结果替代。**

用户于2026-10-29要求在结构／动态阶数探索期间继续推进G09恢复，为下一代基座积累微调经验。本批次固定三次新训练、四份新完整九项评测；研究结果要形成阶段和选择规则，不能直接把不同基座的增益相加或挑历史模块拼装。

用户随后进一步明确两条并行主线：基座线研究保护／位宽／结构与类别压缩；本主题作为恢复线，先在已有明确压缩来源上探索有效增量，不等待新基座。新基座就绪后按同一恢复规则和约定预算复验，分别报告基座变化、恢复增量及最终九项，不直接继承旧父缓存或把旧增益相加。总体安排见[两线交接条件](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月29日两条并行主线与交接条件)。

## 来源与比较边界

`G9`对应G09 UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，是新重压路线经过分支恢复／组合及联合1200更新后的边界，已经包含恢复。它不是刚完成CAT的raw基座，也不是旧bestabs最终成品的同阶段来源。

- raw CAT边界：新G八项66.15221015、MMLU62.54807008；旧49e八项66.57623885、MMLU60.01281869。新G知识项较高、八项均分较低；旧49e缺匹配完整LongBench，不能据此判九项优胜。
- 对应09后处理边界：G09九项65.7842101001；旧f927为65.9751858205。G09的MMLU／LongBench较好，但九项低约0.1910分。两路线几何不同，此比较也不识别各阶段独立因果。
- bestabs九项67.0235120944属于更后面的最终成品参考；不能拿它与G09直接判断CAT结构优劣。

以上支持继续探索G09可恢复性，不构成“G09整体已超过旧最佳”的结论。原始来源见[重压尾部恢复](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/README.md)及[对齐恢复](../cat_aligned_recovery_20261028/README.md)。

## 固定首批与递进判断

| 比较对象 | 研究问题 | 起点与约束 |
| --- | --- | --- |
| 已有G02完整九项 | G09→QK2000→OPD32整条路线的当前质量 | G02 UUID `38a7461e-a2f1-4586-80e2-b99883ef2628`，全八项＋LongBench21，不以八项筛掉长文评测 |
| G09直接fresh OPD32 | 前置QK2000是否值得保留 | 与既有OPD32相同M1024/Q32/H96、full-F、R0、seed、Adam及32更新，只更换真实初态 |
| G02→D+scale32 | 加Med覆盖与保护scale恢复的完整配方价值 | 同G02父、该父新R896、原配方32组；不是只改变scale的单因素实验 |
| G02→D+q64 | 联合保护整数训练的完整配方价值 | 同G02父及R896、原aux/QA配方64更新；不能与scale32直接归因q对scale |

三份新训练末点全部完成独立原生导出后，各评一次完整九项。对照原G09／G01已验证完整结果，并分别记录相对父模型的任务变化、存储与成本。G01的九项下降不能提前证明跳过QK会更好；直接OPD必须实际检验。

若G02有净收益而保护配方损失其他能力，优先定位覆盖／目标保持与更新范围，不默认追加步数或继续旧拼接；几何校准使用真实同源pre/post对，作为另一个受控研究。若直接OPD优于绕经QK的路线，则将删减QK作为新基座上的复验候选；当前结果不能自动外推。若都无改善，保留完整负例与预算边界，把精力转回表示／结构，并只对有新依据的恢复缺口追加配对。

本批次不自动执行G04拼接、C支路、固定L18/L9修补或学习率网格。最终迁移的是数据规则、阶段定义、调参与停止程序；代表性层和旧端点不是最终算法的固定常数。

## 运行状态与复现入口

G02评测于10月29日15:20已核证完整终态，tmux原为 `G02_full9_GPU56_20261029`、初始shell PID365889，outer exit0。`evaluate_g02.py`绑定实际G02 UUID `38a7461e-a2f1-4586-80e2-b99883ef2628`，完整64叶／24742题与LongBench21／4750均通过同UUID和原样本核验。八项68.78660178433782、LongBench宏均42.905984213025924、九项 **65.9109776097476**，相对同协议BF16掉 **2.1361673081608683 pp**。`current_goal_result.json`完整，`transfer_acceptance.json`为`not_met`、`budget_passed=true`；位宽通过不等于精度达标。输出根为 `result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/`，核心 `G02_full9.log`、外退出 `G02_full9.exit`，详细终态在 `evaluation/G/02/evaluation.json`。本次复用主线程逐项核证的终态，不重跑完整样本。

原评测保留历史报告，末尾再调用独立 `experiments.w25_eval.summarize_transfer_nine_task` 生成 `evaluation/G/02/transfer_acceptance.json`，按当前严格掉点<0.9及逻辑≤2.6口径判断。它不单独认证可迁移性。

G02相对G01／G09的已核逐项变化如下，差值均按原0–100分标度；本轮归并现有完整结果，不重复评测。

| 指标 | G02−G01 | G02−G09 |
| --- | ---: | ---: |
| BoolQ | 0 | +0.030581 |
| RTE | −0.722022 | −1.083032 |
| Winogrande | +0.315706 | −0.236780 |
| ARC-E | +0.252525 | +0.378788 |
| ARC-C | +0.255973 | 0 |
| OBQA | +0.600000 | +1.000000 |
| PIQA | +0.326442 | +0.108814 |
| MMLU | −1.274747 | −0.847458 |
| LongBench21 | +2.020280 | +1.789995 |
| 九项 | +0.1971286211 | +0.1267675096 |

综合增益主要来自LongBench，同时存在知识项回退；现有对照不把它归因于Adam。直接OPD完整九项现已得到65.81553150530686，低于G02的65.9109776097476；本次单轨迹比较不支持删除QK前置阶段，但不是QK在任意新基座上都有效的证明。Muon仍为候选，当前未启动。

G09直接OPD32的新CPU准备已完成：实际G09与既有G01短验几何相同，147非投影张量与原full-F载体一致；复用已有原生OPD训练路径的GPU证据，未伪装为G01。新输出为 `/data/star/VAELLM/g09_recovery_20261029/direct_opd32/`，原tmux `G09_direct_OPD32_GPU2_20261029`，worker376618，核心 `run.log`。prepared SHA为`f4be0326affb447a0740cd4bb10f5509b4acc59941dfebc23addadc5c987a028`。完整32更新已完成，step32在elapsed4589.22秒落盘，独立native export也已verified；新成品UUID `eef5adb5-81e6-4177-ad26-0a63c5e9b018`，3423字段保存／fresh strict exact、1911冻结、504bank及原几何不变，训练和导出均exit0。首步训练峰allocated36.12GB、原生预热峰72.46GB；来源仍为同一个G09，不含QK2000步骤。原worker及导出进程均已自然退出，精确终态见 `execution.json`、`formal32/report.json` 和 `native_audit.json`。

G03首次短验的独立导出因252投影／504bank计数错误被拒绝，原异常处理又覆盖了完整训练报告，因此原两次运行保留失败，不补造成功字段。已归档失败证据、修正计数及异常证据保存；新CPU绑定和两臂真实两更新复验均已完整exit0，3423 fresh strict／504bank通过。GPU0／1已从相同G02 fresh启动正式32／64，分别为PID392288／392770，均确认连续两次更新。R896不重建，来源、损失与参数不变。尚无正式末点或新增九项；唯一实现／数据／失败与修复证据、日志和后继状态见[保护恢复正文](../cat_aligned_protection_20261028/README.md#10月29日修正嵌套bank计数后的独立复验)。

所有任务使用调用者激活的 `bitvae`，独立输出与后台日志。G09直达分支负责代码／验证／启动的子代理与结构、几何研究互不写对方目录；root负责统一来源、GPU安排和最后的九项结论。

直接OPD的完整评测由独立入口 `evaluate_direct.py --gpus <已分配GPU列表>` 执行；`--check-only`只核真实完成的32步、独立native审计与实际账本。入口在实际末点就绪前拒绝运行，不等待／重试／回退到G09父模型。它复用原完整八项、21任务分区与原始样本核验，单独输出 `evaluation/G/direct_opd32/`，最后再生成当前严格<0.9pp验收报告；没有改正在运行的 `evaluate_g02.py` 或其依赖。

直接OPD完整九项已于16:45:52自然完成：八项68.66730723412728、LongBench21为43.00132567474355、九项 **65.81553150530686**，较BF16下降 **2.231613412601616 pp**，当前严格验收未达。较G09增加0.0313214052pp，较G02下降0.0954461044pp；相对G02的LongBench增加0.0953414617分，未抵消八项均值下降0.1192945502分。所有完整64叶／24742题、21任务／4750预测及SHA、同UUID `eef5adb5-81e6-4177-ad26-0a63c5e9b018`、实际账本均已核；logical2.5295091711956523／packed2.6391813858695654、rank0预算通过。历史和新严格质量报告均为有效`not_met`，不把质量未达解释为执行失败。

原调用 `evaluate_direct.py --gpus 2,7`，tmux `G09_direct_full9_GPU27_20261029`，driver429141／start_ticks9329219及两native worker均已退出；MMLU／seven／两分区／两binder／merge均exit0，`direct_full9.exit=0`。核心 `direct_full9.log`，完整阶段／命令／来源在 `evaluation/G/direct_opd32/evaluation.json`；当前 `transfer_acceptance.json` SHA为`d897f59903add023c8590c61aef393f1cc9a06400ab4f8f308e1a62a865d4b4f`，`current_goal_result.json`完整。GPU2／7自然释放，本次没有重跑、补造末点或按八项提前取消LongBench。

14:06–14:08历史快照（其中G02状态已由上述15:20完整终态替代）：两份完整八项当时都已结束且各为64叶／24742题、同各自UUID、两子阶段exit0。G02八均68.78660178433782，直接OPD八均68.66730723412728；直接OPD的MMLU更高，但八均较低。按固定九项公式，直接OPD需完整LongBench宏均比G02高0.95435640168432分才追平九项；这是代数关系，不是预测或新增单项门槛。当时两份LongBench继续完整21，不据八项删除QK阶段或延长任一配方。现场原始预测分别2624／4750和232／4750，四个native worker及两driver均真实live；八卡无空余，最早新终态粗估仍超过30分钟，随后停止了主动轮询。

## G09同父Muon32对照（10月29日）

本节由恢复线独立推进，不等待基座线或G03全部结束。唯一新候选与已完成direct Adam32使用相同G09初态 UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`、M1024/Q32/H96 train、一次半full-F半gold混合M目标、当前学生native greedy轨迹的full-F QA、R0、seed137、全部1512个D叶和原500线性日程前32更新；保护参数及其他1911字段冻结。研究问题仅为矩阵更新能否改变原OPD配方的短任务代价，不预设Muon保知识，也不添加LR／steps网格或数据。

参照为direct Adam32九项`65.81553150530686`、G09父`65.78421010012953`及经QK2000的G02 `65.9109776097476`。direct相对父的LongBench改善与八项回退并存，不能因此将短题损失归因Adam。旧e3/N32的Muon完整九项`66.41057438274544`也不证明本G09／R0目标会获益；本臂必须在自身原生末点上完成同协议strict九项，并同时比较Adam和G09。

隔离代码位于[muon/](muon/)，不改direct或活动OPD／G03依赖。`optimizer.py`从实际metadata及packed参数推导每stage矩阵，不用旧K白名单或按ndim分配。实际504个linear_in/out矩阵叶拆为1008个独立stage矩阵，包含K34；矩阵4,552,704元素，全部D共4,762,368元素。1008个bias／norm叶使用辅助Adam，二维norm仍是affine。

数学沿用`qwen_optimizer_pair4_20261014/optimizer_math.py`：FP32 momentum .95/Nesterov，BF16 NS5系数`3.4445,-4.7750,2.0315`，NS后FP32乘`.2*sqrt(max(m,n))`；辅助AdamW `(.9,.95),eps1e-8,WD0`。每次实际M32＋QA8梯度只联合clip1.5一次。首个真实clip后的矩阵梯度在临时fresh Adam状态上匹配**未舍入FP32参数候选**总norm，得到单个κ，此后固定；正式fresh32重新计算κ，不借短验或CPU测试值。两组分别按`1e-6*(1-(step-1)/500)`及其κ倍数调度，不宣称BF16 decoder或decoded W步幅相同。

原native rollout的通用tensor-state守卫保留，对momentum buffer和aux Adam张量身份／值同样生效；外层补κ及两组LR／设置exact往返。校准额外核真实master与gradient对象／数值未变。decoded BF16位移只在已有初始和末态native cache各读一次全252原坐标W，CPU参考只驻RAM，结束释放，不加LM调用或落盘权重；报告末态净位移与NS耗时，不据候选norm推断部署效果。

CPU后台准备已正常退出0／12.0364秒，CUDA未初始化；真实布局、五类矩阵shape的stage扰动隔离／重排、norm分类、两步状态连续、momentum／辅助Adam／κ／双LR的内存save/load均通过。实际export签名、prepared路径及源文件stat也已检查。唯一prepared为`/data/star/VAELLM/g09_recovery_20261029/muon_opd32/prepared.json`，SHA `4c41511dbc6850bfa477bd11d87c51d90506eca6d39b76392dd17448998cf7e6`；CPU随机梯度κ仅数学测试，不复用于GPU。Astra窄审确认该数学及状态接口可进新Muon短验。

17:16:29在GPU7为0 MiB且无compute时，单次启动tmux `G09_Muon_short2_GPU7_20261029`，实际worker PID `543313`。原组0/1两次真实更新已完成，两个窗口均覆盖1008个非零stage矩阵梯度；第二次native生成时504 momentum＋1008 aux Adam、κ及双LR metadata、FP32 master／state对象和值往返exact。短验实际κ为`5.573023927214567`，fresh Adam／base Muon候选norm为`.0021305212364692862`／`.0003822917798836967`；两次NS wall为`.3985492`／`.4023199`秒。两步BF16 decoder位移分别`.0008020888`／`.0016335651`，全252原生BF16 W2−W0相对L2为`.00299551927`；这些不是质量收益或候选norm等价证明。

初链两次训练、末点保存／安装和源回滚成功，随后CPU导出因新启动脚本隐藏CUDA，被FA2构造检查拒绝。`map_location='cpu'`只决定张量位置，FA2构造仍需可见CUDA；没有改变FA2／数学或重新训练。保留可见GPU7后，用同源`export.py`单次补验成功：short UUID `d96ef95a-6f30-4ce2-ba15-6ac3a8c5447e`，3423保存／fresh strict、1911冻结、504bank与实际FP32→BF16对应均通过，native audit SHA `80e3b086651d32e14f037782eb555cc6815777a1d164206c30a2a7931c387e52`。

初链`short2_initial.exit=1`与`short2.log`原样保留，训练`short2/run.exit=0`，接续出口`short2_export_completion.exit=0`。[complete_export.py](muon/complete_export.py)逐项核实这些真实退出码、原report／endpoint／prepared SHA及实际strict audit后才写接续总完成码`short2.exit=0`；实际finalizer PID `556670`／exit0。完整旧命令／worker PID／日志绑定、新出口命令及audit SHA在`short2/completion.json`，不将原失败进程写成成功。原report／endpoint未被补改；原绑定脚本作为首次启动证据冻结，当前执行使用下面的显式入口。

调用者激活`bitvae`、在原snapshot cwd通过tmux后台运行；模型构造和导出均保留可见GPU7。CPU-only准备不加载模型，可隐藏CUDA：

```bash
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/g09_recovery_20261029/muon/prepare.py
CUDA_VISIBLE_DEVICES=7 python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/run.py --phase short2 --gpu 7 --timeout-seconds 7200
CUDA_VISIBLE_DEVICES=7 python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/export.py --report /data/star/VAELLM/g09_recovery_20261029/muon_opd32/short2/report.json --output-dir /data/star/VAELLM/g09_recovery_20261029/muon_opd32/short2/native_final/final_model
CUDA_VISIBLE_DEVICES=7 python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/run.py --phase formal32 --gpu 7 --timeout-seconds 43200
CUDA_VISIBLE_DEVICES=7 python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/export.py --report /data/star/VAELLM/g09_recovery_20261029/muon_opd32/formal32/report.json --output-dir /data/star/VAELLM/g09_recovery_20261029/muon_opd32/native_final/final_model
```

17:34:28在GPU7再次为0 MiB且无compute后，已fresh启动正式32，tmux `G09_Muon_formal32_GPU7_20261029`，shell `557588/557590`、实际worker `557591`。实际命令显式执行上面的formal run→可见CUDA export，环境保留原offline、`TOKENIZERS_PARALLELISM=false`及`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，不调用原隐藏CUDA出口的两份`.sh`。没有继承short参数／状态：首组生成前state为空且κ为None，首真实clip后新κ为`5.572230867430454`，fresh Adam／base Muon候选norm为`.002130524304888131`／`.0003823467396767408`。首更新矩阵LR `5.572230867430454e-6`、aux `1e-6`，NS wall `.3870847`秒，504／1008状态均有限且step1；随后第二组1512非空状态往返通过并完成M反传。独立运行不宣称与Adam或short位级同轨。

已确认正式主循环推进并停止主动监控；`muon_opd32/formal32.log`为唯一核心日志，整个训练→导出自然退出写`formal32.exit`，训练本身退出写`formal32/run.exit`。**正式32仍在运行，尚无末点或九项质量结果。** 末点将经独立CPU保存／fresh strict3423、504 bank／1911冻结、实际BF16对应与原位宽核证，终态在`muon_opd32/native_final/final_model`及`native_audit.json`。本臂不自动占用其它卡进行full9，完整九项资源由主线程另行协调；最终目标仍为同协议BF16九项等权综合严格掉点<0.9pp、logical≤2.6，不设单项门槛。

short权重用途在strict核证及正式fresh启动后结束。核无引用，已删除其`decoder_endpoint.pt`、native模型权重及六份tokenizer副本共8文件，实际移除分配空间`4,808,757,248 B`；保留原生metadata／config、核心日志、原始数值、初链失败与接续审计。精确路径／大小／原因和删除核查归`short2/completion.json`的cleanup字段，不将共享磁盘变化计入释放量。正式运行及其待完整九项末点继续保护。

18:56只读核实的正式终态替代上面的17:34在途快照：真实group0..31共32更新已完成，`formal32/run.exit=0`及含独立导出的`formal32.exit=0`，实际训练4694.879676216995秒。`formal32/report.json`为`G09_Muon_nativegreedy_OPD_formal32_complete`，真实最终FP32转BF16匹配、源恢复通过；源仍G09 `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，正式固定κ为`5.572230867430454`，没有借短验状态。

实际成品为`/data/star/VAELLM/g09_recovery_20261029/muon_opd32/native_final/final_model`，新UUID `1257125f-7c64-4650-80b0-04ec2deac99f`。`native_audit.json` verified：3423字段保存／fresh strict exact、1911冻结、504bank、原几何保持；来源及endpoint report SHA、metadata SHA／实际权重stat匹配。logical2.5295091711956523／packed2.6391813858695654 bpw、rank0，预算通过；audit SHA为`f8b6997e25a32458ef78e7c7504f7879b234d844040660faf10ae32b2e1c57d3`。原训练／导出进程已自然结束，未重训、重导出或据训练loss推断质量。

当前夜间`overnight_Muon_full9_20261029`的driver610001／start11083897已接受Muon数据依赖exit0，仅待q64释放评测资源；实际`pidfd`指向q64外层560500，没有占卡、重试或提前执行新评测。完整九项尚未产生，不能把原生终态通过写成Muon质量胜出；后继由已授权夜间固定入口完成，详见[夜间接续](../overnight_goal_20261029/README.md)。

## G03 scale32 末点与完整九项（10月29日）

G02→D+scale32已完成正式32步及独立native导出／fresh strict，成品为 `/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/scale32/native_final/final_model`，UUID `6a446110-a963-45a5-9847-8458f5d88c40`；`scale32/native_audit.json` SHA `29b7c6381aec24a045fb8e8ee643dc5a8a2251739cb773b7ade232121bc2df44`。实际logical2.5295091711956523／packed2.6391813858695654 bpw，正式32步、3423持久字段与504bank由原审计核证。本节替代上面启动快照中“scale32尚无正式末点”的状态；不推断本轮未检查的q64终态。

新增薄入口 `evaluate_scale32.py --gpus 5,6`，内部调用原 `source(profile='G',stage='03a',checkpoint=明确成品,native_audit=明确审计)`，只将输出改到本主题 `evaluation/G/scale32/` 并绑定新脚本及当前transfer验收源码。原full8、原64叶24742题／同UUID核验、完整LongBench21／4750分区／生成／评分、历史验收均复用，不改任何活动依赖，也不重复训练、保存或GPU短验。原完整九项结束后另生成 `transfer_acceptance.json` 和 `current_goal_result.json`，按当前严格BF16掉点<0.9pp、logical≤2.6判定，不以八项分数筛掉LongBench。

调用者使用`bitvae`、项目根cwd、CPU协调器隐藏CUDA；CPU `--check-only`已exit0，仅核成品身份与实际账本。15:31:01核GPU5／6均0MiB且无compute任务，随后正式启动唯一tmux `G_scale32_full9_GPU56_20261029`，outer shell466391、driver466397／start_ticks9909096；driver记录的start Unix为1793259095.7001734（15:31:35.700）。MMLU worker为466788，后续原七项与两卡完整LongBench21顺序接续。

实际命令如下；核心日志 `result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/scale32_full9.log`，自然结束后写同级`scale32_full9.exit`；详细阶段／命令／GPU资源／源码pins与成品绑定在 `evaluation/G/scale32/evaluation.json`，MMLU核心计算日志为其下`full8/mmlu.log`。15:33:25有限核实，实际MMLU loglikelihood已推进到2049／56168请求，随后停止主动监控；完整九项及最终退出尚未发生。CPU预检已归并本节，独立预检日志／退出散文件已清理，正式日志和结果均保留。

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/g09_recovery_20261029/evaluate_scale32.py --gpus 5,6
```

q64接续入口为`evaluate_q64.py --gpus <root届时分配的GPU列表>`，可加`--check-only`只核真实末点。它固定读取`G/03_protection/q64/{native_final/final_model,native_audit.json}`，复用原`source('G','03b',明确checkpoint/native_audit)`的formal64／原生审计与实际账本条件，并要求真实父UUID为G02 `38a7461e-a2f1-4586-80e2-b99883ef2628`；新成品UUID和audit SHA在运行时绑定，不借scale32。完整full8＋full21独立输出`evaluation/G/q64/`，末尾按当前strict<0.9pp／logical≤2.6生成transfer报告，拒缺源及已有输出，不等待／重试／回退。10月29日准备阶段的bitvae后台CPU导入、真实CLI解析／GPU列表检查和实际缺末点拒绝均通过，当时CUDA未初始化、评测输出未创建；这不是成品审计通过。当时尚无q64原生终态、未启动GPU评测，现由下节真实末点及启动记录替代。

## G03 q64 末点与完整九项（10月29日）

G02→D+q64已真实完成64次连续Adam更新，组0..31连续两遍，正式内外退出均0，实际15897.46920643399秒。所有1764叶获得非零梯度，1659字段冻结、full-F教师及源恢复合同通过；没有改变起点、数据、R896、32K条件或原500步日程。独立原生末点为`/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/q64/native_final/final_model`，新UUID `993ee918-c14a-4d3e-96f6-7766ba71b37c`，实际父仍G02 `38a7461e-a2f1-4586-80e2-b99883ef2628`。

原export首次隐藏CUDA时，在FA2模型构造阶段失败，未重训或写出成品；root恢复可见GPU0后用同一入口完成导出。`export.exit=0`，`native_audit.json` verified：3423字段保存／fresh strict exact、504bank、1659冻结及原几何保持，metadata SHA／权重stat与实际文件一致。audit SHA为`1dbd3da2defe495827f57caeedc8737e4151fbfd0dd8edb222488dd12f833fa1`，logical2.5295091711956523／packed2.6391813858695654 bpw、rank0。`q64/export_launch.json`已更新`complete_verified`，原首次失败`attempts[0].exit_code=1`及日志记录保持，没有把原失败改为成功。

当前真实来源经原`evaluate_q64.py --gpus 0,1,2 --check-only`单次CPU核查exit0；随后调用者激活`bitvae`、项目根cwd、协调器隐藏CUDA，17:37:57实核GPU0／1／2均0MiB且无compute后启动唯一tmux `G_q64_full9_GPU012_20261029`。shell560500／start_ticks10668273；driver560505／10668274，MMLU worker560923／10669433，源UUID与审计绑定均为本次实际q64成品。

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/star/VAELLM python -B -u experiments/g09_recovery_20261029/evaluate_q64.py --gpus 0,1,2
```

唯一启动记录为`result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/q64_full9_launch.json`，核心`q64_full9.log`，自然结束后写`q64_full9.exit`；结果及来源／源码pins／分区状态在`evaluation/G/q64/evaluation.json`，实际MMLU日志为其`full8/mmlu.log`。有限核实看到group8预热252／252、skipped／failed0（2.57秒），实际MMLU loglikelihood连续推进至5377／56168请求。此时尚无完整八项或九项，不把启动核证写成质量结果。

后继由原入口完成完整64叶／24742题八项，再沿成熟三卡分区完成LongBench21／4750及原未舍入九项，另写当前strict<0.9pp／logical≤2.6验收报告；不按八项分数停止LB。确认真实计算推进后已停止主动监控，不等待完整结束，不自动启动新任务／队列；GPU7的Muon、3／4的CAT及5／6的scale均未触碰。

## 10月30日完整结果与同父配对闭合

09:30实核相关任务均已自然退出，无tmux server。G02/direct/scale32/q64的外`*_full9.exit`均0；Muon/G04夜间job及其评测均0。每份完整八项都是64叶／24742题，LongBench都是21项／4750，MMLU、seven、原生分区、binder及merge均成功；当前transfer验收为有效`not_met`，不是运行失败。

| 成品／初态 | 九项（原报告未舍入值） | 相对指定对照的精确输入增量 |
| --- | ---: | --- |
| G09 | **65.78421010012953** | 共同父参考 |
| G02 | **65.9109776097476** | G02−G09 = +0.12676750961808866 |
| direct | **65.81553150530686** | direct−G09 = +0.031321405177341055 |
| Muon | **66.06887949148307** | Muon−direct = +0.25334798617620935 |
| scale32 | **66.05276503731372** | scale32−G02 = +0.14178742756611343 |
| q64 | **65.78621461843409** | q64−G02 = -0.12476299131351885 |
| G04 | **66.18850850359583** | G04−G02 = +0.27753089384822188 |

G09 UUID仍为`e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，G02仍为`38a7461e-a2f1-4586-80e2-b99883ef2628`。direct与Muon各从同G09 fresh正式32更新；scale32/q64各从同G02正式32/64更新；G04仅按事前固定规则以scale32整MLP＋q64整attention组装，不新增训练。各新UUID、对应audit、实际权重稳定stat、全九项未舍入分量与有理数差值统一保存在[终态与精确配对审计](../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/terminal_audit_20261030.json)，共2199个唯一来源／原样本文件SHA核对通过。G09作为已完成同协议控制读取原验收和评分，不新增GPU评测。

下表均为0–100标度差值，显示6位；选择与严格边界使用审计中的有理数，不使用本表显示舍入值。

| 指标 | direct−G09 | Muon−direct Adam | Muon−G09 | scale32−G02 | q64−G02 | G04−G02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BoolQ | +0.122324 | +0.183486 | +0.305810 | +0.122324 | -0.458716 | -0.214067 |
| RTE | -0.722022 | -0.722022 | -1.444043 | +0.000000 | -1.444043 | +0.361011 |
| Winogrande | -0.789266 | +0.473560 | -0.315706 | -0.552486 | -0.315706 | -0.157853 |
| ARC-E | +0.336700 | +0.420875 | +0.757576 | -0.210438 | +0.420875 | -0.294613 |
| ARC-C | -0.341297 | +0.682594 | +0.341297 | +0.255973 | +0.085324 | +0.767918 |
| OBQA | +0.400000 | -0.200000 | +0.200000 | +0.000000 | -1.600000 | +0.000000 |
| PIQA | -0.054407 | -0.108814 | -0.163221 | -0.163221 | -0.598477 | -0.489663 |
| MMLU | -0.555476 | +0.462897 | -0.092579 | +0.982766 | +1.118074 | +1.082467 |
| LongBench21 | +1.885336 | +1.087555 | +2.972891 | +0.841169 | +1.669801 | +1.442578 |
| 九项 | +0.031321 | +0.253348 | +0.284669 | +0.141787 | -0.124763 | +0.277531 |

| G04指定控制 | 九项增量（未预先舍入） | LongBench21增量 | MMLU增量 |
| --- | ---: | ---: | ---: |
| G04−scale32 | 0.13574346628210848 | 0.6014093911288162 | 0.09970089730807 |
| G04−q64 | 0.40229388516174075 | -0.22722280576450324 | -0.03560746332432 |
| G04−G02 | 0.2775308938482219 | 1.442578186762877 | 1.08246688505911 |

已核事实与边界：Muon同父同数据／R0／32更新的本轨迹九项高于同期direct Adam，且高于冻结G09；相对Adam仍有RTE、OBQA、PIQA下降，相对父的MMLU也未恢复到父分数。scale32相对G02有净收益；q64虽提高MMLU和LongBench，但OBQA/RTE/PIQA等回退使九项下降。scale32与q64数据覆盖和更新数不同，不能将配方差解释为scale对q的单因素因果。G04本次固定家族组合高于两父和G02，仍不证明逐层挑选、任意拼接或其它基座能复制该增量。

正式执行报告核实：direct32实际4613.212078258002秒，Muon32为4694.879676216995秒，均1512个D叶／4,762,368参数、教师不变、源恢复exact；Muon固定κ为5.572230867430454。scale32为7906.275684867011秒、q64为15897.46920643399秒，各1764个D＋保护叶均有非零梯度、教师不变、源恢复exact；两者真实内外退出均0。G04原组装3423字段fresh strict exact、504bank、144 attention完整替换，未逐题选donor。训练／组装报告路径及SHA在上述审计的`training_and_assembly`；q64早先隐藏CUDA导出失败仍保留历史，后续同入口可见GPU导出verified，未重训或改写原失败。

本节六个恢复／组合成品（G02/direct/Muon/scale32/q64/G04）真实账本均logical **2.5295091711956523**、packed **2.6391813858695654** bpw，rank0、252投影、504bank、3423字段；包含decoder、保护indices/scales/weights、rotation，packed额外计码padding。全模型tensor bytes为4,781,328,384，包含另计的embedding/head/norm，不含序列化容器额外开销；logical≤2.6通过，packed值单列。所有完整九项都未达到BF16基准68.04714491790848下严格掉点<0.9，即九项必须>67.14714491790848。

10月30日主线程与研究线确认：保留G09及Muon成品和direct控制证据，G04作为优于两父的候选保留，q64固定单支不延长；不据此否定所有整数训练或增加新的同分支网格。可复用事实及迁移限制见[同父恢复与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。

完整原证据分别为`result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/evaluation/G/{02,direct_opd32,scale32,q64}/`及`overnight_goal_20261029/evaluation/{Muon,G04}/`的`evaluation.json`、`full8/full8_summary.json`、`longbench/merged/scores.json`、`transfer_acceptance.json`和`current_goal_result.json`。原样本、预测、正式配置和核心日志保留；本次只读复算现有结果，未新增训练／完整评测或删除模型。已核事实可作为恢复选择的新证据，不能把阶段增量相加作新成品成绩；具体后继实验和模型用途由主线程当前计划统一维护。

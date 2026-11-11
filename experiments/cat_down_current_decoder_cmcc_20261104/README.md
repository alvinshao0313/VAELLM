# 原VAE-Muon raw：末down恢复范围36对252

本包只比较同一原D32 VAE-Muon raw `c5ccf174-0c56-4f90-9e29-7d4ba63d2e23` 的两种末down恢复范围。完整控制直接复用 [原all252 Adam500](../cat_down_vae_muon_closure_cmcc_20261103/README.md)，最终UUID `983dc190-d6ac-4a51-8225-83389b35011c`，九项64.30526846061926。新臂只训练全部36个native down decoder：216个packed decoder叶／767232元素；原控制是1728叶／5336064元素。码、保护、旋转、非down decoder及其余backbone冻结。不是已结束的新D16、gate或组合实验。

原控制完整八项已从raw66.3532649235提高到67.1002942859，没有证据证明all252损害质量。本实验是范围配方对照；选择变化同时影响全局clip库存和decoder执行路径，不能只归因于跨类别梯度冲突。

本批已批准总20GPUh，其中本包仅GPU0：新路径必要技术短验≤1GPUh，原500/fresh≤8GPUh，同最终UUID完整九项≤10GPUh。只固定500终点，不按250择终点；成功500之后无论中间指标如何都接完整九项，`not_met`是合法质量结果。九项≤64.30526846061926结束本点；微正只记单seed候选，不自动扩网格、延长、增加M门槛或运行新后缀。尾阶GPU2由另一包负责，GPU6/7不属于本线程。

## 实现与必要验证

原`recovery.selection`不读取`stage.mode`，总是开启全native和compact decoder。本包独立提供实际36down选择及薄`make_trainer`，公共活跃源码及其pins不改。训练、KD、教师、clip、保存消费及finalize复用原`MixedCatTrainer`和原核。冻结native只缓存解码权重，`F.linear(inputs, frozen_weight)`仍保留输入反传；不在冻结层forward外使用`no_grad`。真实两更新逐层检查全部36down获得非零有效梯度，第二次正LR使36层均实际更新。

原FP32 master、BF16前后向/导出、AdamW(.9,.999)/eps1e-8、decoderLR1e-5/WD0、cosine500/warmup5/clip1.5、save250/500、LoRA0全保持。原stage37/data31、原五源开源train及其加载、抽样、顺序重建、原BF16教师与KD目标、B4/GAS1/cap1024动态padding、lazy iterable/workers16/实际group_by_length=False保持。CPU用原真实Trainer数据消费者重建前500批，核留存首batch `[4,456]` 的token/mask identity并提取真实最大padding批；历史原500逐批hash未存，不能称与历史全500逐批exact。短验仅将该实际峰批重复两次覆盖正式资源/新范围，正式仍用原数据入口与原流，不沿用短验状态。

CPU合同记录实际选择、FP32、冻结补集、原optimizer所有组字段（仅参数列表按范围改变）、来源/代码SHA和数据身份。必要短验用原模型、B4及真实原流峰、原500日程两次更新，首LR0允许master不动但state前进；save1后继续step2，检查全部36down更新、冻结持久字段exact、独立fresh及位账。仅验证新增范围，不重复无关exact-resume。

完整九项沿原B64-M/B32其余、0shot/limitNone及原LBv1全21项4750题、32768窗口、精度/生成规则。一个绝对deadline传递至全部consumer组件，每个组件进入时再按剩余时间运行GNU timeout，LB native/outer也取同一deadline剩余量并保留20秒grace；不在边界后让detached runner继续。技术超时单列，不作为质量负例，不自动retry/resume或续期。

## 入口、产物与状态

调用者先激活`bitvae`，在项目根使用tmux/nohup后台执行。实验`.sh`仅设环境并调用Python，不激活conda。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_current_decoder_cmcc_20261104.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_current_decoder_cmcc_20261104.validate_deadline
timeout --signal=TERM --kill-after=20s 3580s bash experiments/cat_down_current_decoder_cmcc_20261104/short.sh
bash experiments/cat_down_current_decoder_cmcc_20261104/formal.sh
```

权重根：`/data1/cmcc/VAELLM/artifacts/cat_down_current_decoder_cmcc_20261104/`，short与formal独立。证据根：[result](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/)，`prepared.json`保存CPU来源/数据/optimizer/源码SHA，`short/report.json`及`strict_reload.json`保存必要短验，`run.json`保存正式命令、PID/start、退出、预算、日志及固定后继。未提交，不伪造commit ID。

经验复用[精度与资源](../../docs/experience/lessons/memory_precision.md)、[KD与数据](../../docs/experience/lessons/kd_and_data.md)、[checkpoint生命周期](../../docs/experience/lessons/checkpoint_lifecycle.md)及原闭合证据；完整质量结果与边界见末节。

## CPU与必要短验实测（11月4日）

[CPU合同](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/prepared.json)实际exit0、40.920555秒、未初始化CUDA：216叶／767232 FP32元素及冻结补集均核实；effective mode为`current_decoder`，原stage的`remaining_lora_prefix_decoder`只保留作完全相同原参数参照，不能据`original_stage_exact`称实际范围仍为all252。原Adam组除参数列表外全部字段与旧保存参数相同，fresh state0。原真实数据消费者重建500批／2000例，首批 `[4,456]` 对旧留存probe exact；最大padding批为真实第2批 `[4,1024]`，重新collate后所有token／mask／label exact。五源数据配置与source stats、全部新trace和1019个源码pins在prepared中；历史旧500逐批hash不存在，不扩大identity结论。

原保存参数实际是TRL `SFTConfig`，首检误传基础 `TrainingArguments`时拒绝其扩展字段，修复为原实际类后通过。随后一次空闲tmux shell重读bashrc将PATH改回base，导入torch前失败，未计算／未触CUDA；后续启动均显式传caller已激活PATH、`CONDA_DEFAULT_ENV=bitvae`及绝对bitvae Python。两次失败trace与源码SHA保存在 [CPU失败receipt](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/cpu_failure_receipt.json)，未吞错或安装依赖。

[真实评测plan检查](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/deadline_cpu.json)exit0：旧control实际config的全21／4750任务与协议保持，仅LB native runtime／outer timeout及对应argv三项时间字段改变；所有consumer命令传同一绝对deadline，过期deadline在consumer前拒绝。各组件启动时再按剩余deadline运行GNU timeout并留20秒grace，正式外层最多10h。Astra已复核选择、输入反传、save／finalize和三项新增证据，无核心阻断；未复审成熟KD或kernel。

GPU0真实短验tmux `DownScope_short_g0_20261104`、worker2465357，于10:53:55完整exit0；[报告](../../artifacts/cat_down_current_decoder_cmcc_20261104/short/report.json)内耗时135.262176秒，其中Trainer53.9091秒，只执行原500日程的前2更新。实际首批 `[4,1024]` 与CPU原流峰exact；全部36down的梯度finite／非零。首步实际应用LR0、master改变0、全部216个Adam state前进到1；step2实际LR2e-6，改变767227个FP32和99997个BF16元素，36层各自都有实际变化。save1后回训step2；两次Mixed保存均保留FP32 master及原HF optimizer／scheduler／RNG，保存前后对象和值不变，216状态叶。冻结持久补集exact，finalize只处理36native down。

短验最终UUID `4646e915-fdeb-43f5-9adc-c1874abb80ef`，[独立fresh](../../artifacts/cat_down_current_decoder_cmcc_20261104/short/strict_reload.json)核3675持久字段、同输入BF16 logits exact，group8 warm216／216且无失败。实测logical2.4987414816151494／packed2.5121161419412363bpw，full-model tensor4671007680B；含码、decoder、scale、保护、indices、rotation及padding，不把理论码率当存储。训练step2峰allocated42335092736B，短验全生命周期导出／预热峰57263232000B、reserved57870909440B，不能以训练峰替代启动／导出资源条件。

短验已无权重、optimizer或复评用途，正式从原raw fresh，未接续短验。[清理receipt](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/short_closure.json)记录删除前精确路径、大小／原因及删除后保护检查：移除short生成模型权重、trainer_state及用完logits夹具，实释4699037696B allocated；保留short report／strict／metadata、原始核心日志、CPU峰fixture及所有来源／退出／源码SHA。原raw与完整all252控制、正式目录及其它线程不受影响。

正式已独立启动tmux `DownScope_formal_g0_20261104`（pane shell2468395）。[启动核证](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/startup_verification.json)记录controller `2468398/start49826535`、timeout `2468847/start49828831`、真实worker `2468848/start49828832`，全1019个源码pins仍匹配。原raw fresh／原lazy iterable首批在真实forward hook中对CPU重建 `[4,456]` token／mask exact断言已过；该断言在首计算前执行，随后实际step2推进，完整inputs digest随终态formal report落盘。首步master改变0，第二步原LR2e-6改变766611个FP32和53170个BF16元素，36down均实际更新；第二批峰allocated42335092736B／reserved43620761600B，与必要峰短验相容。未加载short权重或optimizer。

下述为启动时状态记录：当时为`down500_running`、尚无最终质量结论；本批后续已按原链完整结束，见末节。固定500／fresh≤28800秒后直接接同最终UUID完整九项≤36000秒；实际命令／源SHA／PID／资源与退出均由 [run.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/run.json)记录，训练日志为同目录`down500.log`，后继为`full9.log`和`evaluation/`。完整终态比较对象为原all252的九项64.30526846061926；不补造raw完整九项、不按250选终点、不自动追加下一批。真实主循环已确认，交付后停止主动监控，后台固定有限链保留运行。


## 完整九项终态与同父范围取舍

本批19:04:48已完整结束，技术全exit0；九项`63.56846310013165`，较同父全252D控制-0.736805360488pp，按既定规则结束本固定点，不追加范围、训练或网格。

原`run.json`为`completed_current36_down_original_Adam500_full9`，sequence／down500／full9及M／seven／LB outer／bind／merge技术退出均0；LB native=`partition_complete, new_predictions=4750, already_present=0`。controller2468398和LB worker2574705均自然退出。九项`not_met/nine_task.exit=1`严格表示质量未达，区别于技术退出，未重跑或追加预算。唯一核证归[CPU终态receipt](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/closure_20261104.json)。

最终同UUID `63d38a00-41ef-4f32-b496-4a58acfe0ef8` 与原raw `c5ccf174…`／同父all252终态 `983dc190-d6ac-4a51-8225-83389b35011c` 完成核证：raw／两恢复端各64叶24742题所有原sample SHA、逐题doc_id／doc_hash／prompt_hash／target_hash完全一致；任务版本／configs、M64／其它32、0-shot／limitNone保持。两恢复端LB21／4750原预测SHA逐文件匹配，逐题dataset／row_index／_id／answers／input_ids_sha256相同，archive／官方代码数据revision／32768窗口／seed42／生成协议一致。只复核现存分数和均值，不重评分或推理。raw没有完整LB或九项，不能补造。

| 任务（分） | 原raw八项 | 同父all252 Adam500 | 当前36down Adam500 | 当前−all252（pp） |
| --- | ---: | ---: | ---: | ---: |
| mmlu | 62.2632103689 | 62.4768551488 | 62.4270047002 | -0.0498504487 |
| arc_challenge | 52.9863481229 | 53.7542662116 | 53.7542662116 | +0.0000000000 |
| arc_easy | 79.1245791246 | 79.2508417508 | 79.5033670034 | +0.2525252525 |
| boolq | 81.3761467890 | 83.1192660550 | 82.5382262997 | -0.5810397554 |
| openbookqa | 38.2000000000 | 39.0000000000 | 38.6000000000 | -0.4000000000 |
| piqa | 74.4831338411 | 75.6800870511 | 74.3199129489 | -1.3601741023 |
| rte | 76.1732851986 | 77.6173285199 | 73.6462093863 | -3.9711191336 |
| winogrande | 66.2194159432 | 65.9037095501 | 65.5880031571 | -0.3157063931 |
| longbench_v1 | 未评 | 41.9450618581 | 41.7391781941 | -0.2058836640 |
| 八项均分 | 66.3532649235 | 67.1002942859 | 66.2971237134 | -0.8031705725 |
| 九项均分 | 未评 | 64.3052684606 | 63.5684631001 | -0.7368053605 |

MMLU raw8743→当前8766/14042（+23题），相对all2528773少7题；当前八项较raw-0.056141210142pp。对all252九项7降、ARC-C相同、ARC-E上升；LB变化-0.205883663994分。九项相对BF16下降4.478681817777pp，原质量目标未达；LB及九项是复合分数，不视为统一accuracy。

实测投影logical `2.4987414816151494`／packed `2.5121161419412363` bpw，含code／decoder／保护值、indices、scales／rotation／route frame；packed另计92,897,280 padding bits。原raw、all252和当前预算相同，252投影＝216native＋36compact、无pending dense。全模型tensor 4671007680B，序列化权重 4672277077B，完整目录 4674060664B；budget通过与quality未达分开。3675字段／同输入BF16 fresh exact、group8 warm216/216及冻结补集原证据保持。CPU复核metadata SHA和权重size／mtime_ns／inode与评测绑定相同，复用原fresh／评测全SHA，没有重复加载模型或重hash4.67GB权重。

| 实际成本（秒） | 当前36down | 同父all252控制 |
| --- | ---: | ---: |
| Trainer500 | 7292.444400 | 19990.464700 |
| report含保存／fresh | 7372.651315 | 20069.395919 |
| 外层full9 | 21984.772721 | 21188.181666 |
| 固定序列总墙钟 | 29387.693765 | 41287.786860 |

当前down500外层7397.773463秒、evaluation内部21932.782261秒，二者与上表分别定义，不重复相加。另必要short报告135.262176秒；short＋固定序列保守按单GPU墙钟合8.200821095 GPUh，低于本卡≤19 GPUh。此口径含加载／CPU协调间隙而非利用率积分，不重复计已复用前驱／旧控制。训练段变短只说明本实现成本，完整质量更低，不能宣称等成本优胜或把不同阶段提速混为全链收益。

本条件不支持将原all252恢复改为仅36down；九项未胜按原规则闭合，不增加步数／网格或叠其它后缀。范围改变同时影响clip库存及decode执行路径，且历史500逐批hash未存，只证明原流入口／首批identity，不推导跨类梯度冲突，也不外推其它模型／几何／目标。本点已按原规则结束，正式checkpoint及训练状态无后继用途，已按持续授权收尾；raw／控制及全部原指标／预测／来源／核心日志保护。

收尾已完成：同一[closure receipt](../../result/qwen3_8b_mmlu65_20261004/cat_down_current_decoder_cmcc_20261104/closure_20261104.json)先记录解析后绝对路径／大小／原因与原SHA来源，再删除13个无用途文件，逻辑4698887157B、实际独占分配4698918912B（4.376209259GiB）。仅清当前点生成模型权重、optimizer／scheduler／RNG／training_model_state、tokenizer副本、用完部署logits夹具及空锁；检查nlink1、无目标打开FD，删除后全部不存在，125份受保护小记录SHA不变。实际分配来自精确文件st_blocks×512，不按共享磁盘变化推算；short前次已释放量不重复计入。保留一份当前配置／原训练参数、metadata、report／strict／保存审计、全部原始评测及最小核心日志／退出／源码SHA；本次未改正式源码、评测口径、新夜间包或其它活动任务。

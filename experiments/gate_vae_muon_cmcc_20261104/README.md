# Gate VAE-Muon 固定阶段对照

此包回答：在同一真实 after-o 父上，仅把 gate 编解码器拟合的矩阵优化器换为已验证 VAE-Muon，是否改善原生 gate raw2 的完整八项、固定 train 部署误差及成本。实际状态以结果目录 `/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/gate_vae_muon_cmcc_20261104` 和权重目录 `/data1/cmcc/VAELLM/artifacts/gate_vae_muon_cmcc_20261104` 的 JSON／日志为准；唯一当前计划仍由项目总计划维护。

共同父是 `7eeb82b4-3868-4cc5-9fa3-51870ee55894`，原控制 raw2 是 `f4669891-9028-4665-85a5-71c26ac45c33`。原生 native180＋future72dense，不能套用 down 完整252投影压缩消费者，也不能称为最终≤2.6产品。使用 `cat_geometry_cohort_cmcc_20261102/fit.py` 的真实 `run_cat_train`、resume/live重建和停止于真实 gate round_base 的流程；每个正式拟合仅一次seed31。

固定全36 gate、12288×4032未保护部分、非转置／1part／input64 INT8／block32、D64/K64/S2×10K、B4096、E/D128 asymmetric/R0/layer/swish、原所有loss／live stats／shuffle400031和原linear10K／.003／无warmup。成熟Muon数学从原VAE包独立加载，只将实际owner改为4×[36,128,64]/[36,64,128]，共144矩阵；1,211,904元素中1,179,648矩阵、32,256辅助。BF16参数／梯度／计算、不加FP32master，FP32动量／NS5／Nesterov／形状缩放和一次BF16回写保持；8个bias/norm叶沿原BF16 AdamW(.9,.95)/eps1e-8/WD0。每真实stage首梯度独立重算一个E+D共用κ，匹配克隆原BF16 Adam实际舍入位移，≤1%，最多2次扩界／12次二分，不用downκ或扫LR。

原控制没有两stage初始化digest，故不新增伪exact控制初态要求，不重训控制补历史；本次记录两个真实factory原flat顺序摘要。短验复用成熟每stage2步／linear2时域，只证明功能；正式从原after-o重新fresh、每stage10K，绝不接短验权重。正式直接读取原父，短验单独保存状态完全相同、仅短时域identity不同的父副本，减少一份大权重复制。

CPU `prepare.py`核来源、实际配置／owner／batch、原train512×256缓存及短验父全字段／恢复历史。GPU0 `run.py --phase short`总≤30分钟，执行真实M36/B4096/S2各2、每stageκ／更新／峰和独立CPU全2559字段／2091非gate冻结／实际账本，再独立GPU验证BF16 logits exact与group8 warm180/180。通过后单独 `--phase formal`：原父fresh拟合≤90分钟→fresh≤30分钟→完整八项≤90分钟→同train8×256部署读出≤30分钟。技术失败停本依赖，不自动retry。

GPU2 `run.py --phase control`只补原控制seven（B32／0-shot／limitNone）和CPU归并，合计≤60分钟；复用 `/data1/cmcc/VAELLM/artifacts/cat_hyperparam_cohort_cmcc_20261102/control/mmlu.json` 的原M9279/14042、`readout.json` 的原train8×256以及原 `control_fresh.json`，不重评M或readout。

所有命令由调用者先激活bitvae，在tmux/nohup中执行。controller使用`CUDA_VISIBLE_DEVICES=''`，由固定阶段明确分配GPU0或GPU2。修后正式入口为 `python -B -u -m experiments.gate_vae_muon_cmcc_20261104.repair --target run --phase formal`；实际argv／环境／PID／起止／source SHA由对应run记录。源码pins只包含相关Python，README可独立记录阶段结果，不成为活动依赖。

评价按完整八项、M与其他能力代价、部署误差／实际费用一起报告；完整八项不胜且output不改善时结束本点，有净益或取舍也只登记阶段候选。实际投影账本包含72个未来dense，编码器不部署；不自动启动500、up/down、full9、第二配置或占卡。完成本批后须汇报结果并与用户讨论下一批问题、变量、预算和完成条件。

## 实际 CPU／短验与隔离修复

11月4日原CPU60.0秒exit0，核实际1,211,904元素／144矩阵及短验父全字段与历史；实际short两stage各2步、编码和保存完成。κ分别5.466014168590715／5.446520777154589，BF16位移误差0.0053221603%／0.0420521342%，两E/D均有有限、非零更新。原short120.1946秒，在warm180/180后的诊断处失败：`fork.state_digest`只接受CPU，而调用传入GPU state；峰allocated31,526,853,120／reserved31,704,743,936B。原失败run／report／日志／原prepared及3个旧源快照保留，不改成成功。

经root限定授权，未重训两个stage，使用独立 `cpu_snapshot_repair` 结果／权重根复用已保存的短验UUID `029458d0-7934-42eb-aa57-ef00d74aaea6`。修代码仅为fit/evaluate显式`detach().cpu()`快照后沿原exact摘要、run抽出内部child command供隔离入口使用；旧源SHA与修后SHA及真实源快照归修后prepared。成熟Muon数学、实际owner／损失／数据／日程、原GPU1依赖未改。

修后CPU46.9794秒exit0，明确重绑源码／formal identity，保存权重CPU快照exact、独立2559全字段与2091非gate exact，以及对原控制180个gate保护q／scale／indices／旋转sign exact。第一个独立部署进程2437900核prewarm前后persistent全值exact、warm180/180并生成新的BF16期望；第二独立进程2438389重载warm180/180且新same-input BF16 logits exact。原训练live logits虽已计算，但因异常未落盘；本次新对照严格限于两个独立保存消费者，不称为保留的原live对照。正式拟合将在修后真实live warm/save后另行独立fresh，按原合同闭合。

修后权威CPU与发布证据为结果根 `cpu_snapshot_repair/{prepared.json,CPU_repair_verification.json,publication.exit}` 和权重根 `cpu_snapshot_repair/muon/{short_run.json,short/strict_reload.json,short/prewarm_reference.json,short/deployment_fresh.json}`。publication于10:07已exit0，未重训／重编码／重复成熟短验。正式tmux `GateMu_formal_g0_20261104` 于10:07:39启动修后独立链，实际原父resume及每stage10000配置已核；主循环状态见其 `formal_run.json`／`fit.log`，未提前宣称完成。

原控制仅seven补齐及CPU归并均exit0；64叶24742题完整八项68.2892696216683、M9279/14042，仍复用原readout。原件在结果根 `control/full8_summary.json` 与 `control/seven.json`，修后结果根仅以目录链接引用这唯一控制副本。

正式启动时的controller2439232/start49544396→fit timeout2439340→worker2439341身份保留于原run；首stage以B4096、D64/K64和10000时域重新fresh进入，未继承short权重或optimizer。该启动节点已由下述完整终态替代，不再处于等待正式结果状态。

修后源码绑定以 `cpu_snapshot_repair/prepared.json` 的完整pins为准，其中fit=`a4434fa1c7425a03f78c5e6cd359dddca86c9eb992be4198f6173ceef208865b`、evaluate=`f72aac7fcf8d7de1f4452278764154132aade5b6d5f1a217093665e8448e691e`、run=`8f6288a6727e0ea955cb3a877c6618e34474f9d82a293d94d6691802272ef7c7`、repair=`2ca3bbd7ab0278775823556a35c61a52e8990a56f4091755821aeee8a8d83c0e`。旧3源在同结果根 `source_before/`，对应原prepared的旧hash；README不在活动pins内。未提交，不伪造commit ID。

短验已完整核证且无后继reader后，按持续清理授权移除原权重根 `muon/short_initial_after_o/` 的10个生成文件及空目录（allocated13,924,286,464B），并移除 `muon/short/native/data1_cmcc_models_Qwen_Qwen3-8B_20261104_095641/training_rounds/0004_gate_proj/round_base/` 内pytorch_model.bin、cat_runtime_state.pt和6个tokenizer副本（allocated10,777,116,672B）。共释放本任务实际allocated24,701,403,136B，约23.005GiB。两大weight均单硬链接、无符号链接越界、当前formal未持有这些路径；删除后已核移除及原父／控制／活动formal／失败记录和核证仍在。原short checkpoint metadata、有效配置／核心日志、原始指标／摘要与新logits证据保留；不能再把已清理的short路径当作可用模型。正式使用原after-o，不依赖这些短验产物。

## 完整阶段终态与配对核证（11月4日）

11月04日10:31:27，[正式run](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/formal_run.json)自然完成`completed_gate_Muon_fit_fresh_full8_readout`／exit0。fit、CPU strict、GPU fresh、MMLU、seven、full8归并和train readout全部exit0，controller2439232及所有stage timeout／worker已退出；不是仅凭PID消失或程序启动判断完成。本批没有LongBench／九项、类别500或后续类别训练。

正式raw2 UUID `f9ac43d4-095e-4ef5-84a9-6905e25b9c2b`，控制仍为原`f4669891-9028-4665-85a5-71c26ac45c33`。本[完整八项](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/full8_summary.json)与[唯一控制八项](../../result/qwen3_8b_mmlu65_20261004/gate_vae_muon_cmcc_20261104/control/full8_summary.json)的双方64叶／24742条样本SHA、行数及doc_id／doc_hash／prompt_hash／target_hash逐条CPU配对全部一致；M57科14042、其它7叶10700齐全，task版本／metric keys、v6 strict、B64／B32、0-shot／limit=None、FA2／BF16／ropeFP32设置相同。均值只复算已有指标，M正确数只累加已有acc；未重评分或重跑模型。控制M继续复用原cohort，只有seven为本批补齐，不把控制旧评测算成本次重评。

| 完整任务（分） | 原gate raw2 AdamW | gate VAE-Muon raw2 | 本臂−控制（pp） |
| --- | ---: | ---: | ---: |
| mmlu | 66.0803304373 | 65.7883492380 | -0.2919811993 |
| arc_challenge | 53.7542662116 | 53.4129692833 | -0.3412969283 |
| arc_easy | 79.6296296296 | 79.2929292929 | -0.3367003367 |
| boolq | 86.1162079511 | 86.0244648318 | -0.0917431193 |
| openbookqa | 39.8000000000 | 41.0000000000 | +1.2000000000 |
| piqa | 76.9858541893 | 77.0402611534 | +0.0544069641 |
| rte | 74.7292418773 | 74.0072202166 | -0.7220216606 |
| winogrande | 69.2186266772 | 68.7450670876 | -0.4735595896 |
| 八项均分 | 68.2892696217 | 68.1639076380 | -0.1253619837 |

MMLU9279→9238/14042，少41题；八项精确68.16390763795663，对控制−0.12536198371167018pp。OpenBookQA／PIQA升，其余六项降；这里只记录本点配对事实，不作机制归因或后缀采用决定。

[本点部署读出](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/readout.json)与[原控制读出](../../artifacts/cat_hyperparam_cohort_cmcc_20261102/control/readout.json)的完整fixed_train_input相同：原Alpaca train前8×256、同after-o真实gate输入，token SHA `c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`。weight NMSE 0.1296141135714271→0.13216210300830264（相对+1.9658271516%），output NMSE 0.004733617204246657→0.004825175862398493（相对+1.9342218477%）；相应MSE同方向，分层原值保留在readout。该误差只是同train局部部署诊断，不作测试质量或完整恢复结论。

正式两stage各10000真实更新，原flat顺序初态digest分别`174693240ba6821bd114b57b9742702927ee8f0b609b392c9fefc4a55736754a`／`8a1b6f0d6388cacb2a61950e70e4e1bad8e13ba12709c6dc401397af0ec85e2b`。实际1211904个BF16参数、144独立E/D矩阵及8个aux叶由真实factory／owner记录；第1、2、10000步两侧都有有限非零BF16变化，FP32 Muon动量、BF16辅助Adam状态、无持久FP32 master保持。正式κ分别5.466014168590715／5.459220004138485，BF16实际舍入位移相对误差0.0053221603%／0.0248832985%，均≤1%；第二阶独立重算，不借short的κ。原控制没有历史stage初始化digest，未虚称控制初态逐位exact。

[正式CPU strict](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/formal/strict_reload.json)2559字段与真实保存state exact，2091非gate字段对原after-o exact；[正式独立GPU fresh](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/formal/deployment_fresh.json)与本次真实live warm/save的BF16同输入logits exact，warm180/180／失败0。终态核证还以CPU mmap直接逐张比较正式／控制的180个gate保护q／scales／indices／rotation字段，dtype／shape／值全exact，不构造新模型或触GPU。旧short原失败及三份source_before快照SHA继续保留，981个修后源码pins全匹配；short的“两独立保存消费者”边界不变，正式的“本次live→保存→独立fresh”是自身另有完整证据，未把旧short回写成原live已保存。

实际partial账本含native180＋future72dense：投影logical9.48590618631114／packed9.499280846637228bpw，其中未来72个BF16 dense共57982058496bits；gate本类logical／packed均2.107147216796875bpw。码、BF16 decoder、保护值／indices／scales、rotation全部计入，encoder不部署，packed另含92897280 padding bits。全模型tensor10737410304B，序列化权重10738482733B，保存时round_base全部文件实核10777948060B（包括runtime／metadata／tokenizer等）。用既有ledger对原控制实际CPU mmap存储复算，projection／gate各组件bits、padding和全模型tensor均与本点相同。此为未压完up/down的partial读出，不称最终≤2.6或预算验收通过；完整后缀仍须自己的最终账本与完整质量验收。

[正式fit报告](../../artifacts/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/muon/formal/report.json)含拟合／编码／live保存耗888.9606秒；各外层fit897.7812、CPU strict32.6533、部署fresh31.5254、M248.2600、seven178.0171、full8归并5.9658、readout31.6465秒。全正式有限链1426.2539秒（23.7709分钟），新增控制seven／full8链184.3234秒；内外层成本不能重复相加。formal峰allocated31526853120B／reserved31704743936B。早期原CPU／失败short、限定修复／保存消费者的费用见上文，未算作正式训练收益或静默省略原失败。

运行命令、PID/start、source SHA、原失败与真正退出都由既有原件保留；本批仅进行CPU核证、原位结果归并及下述已结束分支收尾，未新增实验或改冻结源码／配置。root已依原“八项不胜且output不改善”条件结束这个固定点，不自动接500／up／down／九项、重试或网格；该决定只覆盖本次配置，不称所有gate／Muon无效。root已确认本点无具体后继用途并完成下述生成载荷收尾；原after-o及控制raw2／补seven／旧M和readout保护。下一批先汇报讨论，主计划／common lessons／INDEX由root统一维护，本文不覆盖外部最新规划讨论草案。

## 已结束固定点的产物收尾（11月4日）

root已按本点“八项不胜且output不改善”的原条件正式结束，无500／live up-down／九项／复评／新诊断或交付用途。删除前复核原失败short的真实exit1、修后short／formal全链及各stage exit0、全部相关进程退役、候选路径无FD／mapping；外部用途也已核清。尾阶CPU源为原D16 raw02292fb6和after-up5a94d57f；规划草案的`current_decoder`父是D32 VAE-Muon raw c5ccf174，与本gate raw2不同，未把未授权草案变成启动或保留本模型的理由。

本次删除23个精确路径：formal round_base权重与生成runtime缓存／6份与原after-o逐SHA相同的tokenizer、short与formal两份用完same-input数值夹具，以及纯导入／预热／控制器wrapper重复日志。两个native logger副本的每条原行已核实在保留的原short_fit.log／formal fit.log中逐字存在，故只留完整核心日志；旧原失败short／controller及正式fit、M／seven日志继续保留。逻辑10777801194B，按删除前st_blocks实际释放**10777878528B（10.037682GiB）**；不重复计入此前short清理24,701,403,136B，也不混入其它三包／组合清理量。

精确路径／原因／size／st_blocks及删前用途检查归既有[CPU修复核证](../../result/qwen3_8b_mmlu65_20261004/gate_vae_muon_cmcc_20261104/cpu_snapshot_repair/CPU_repair_verification.json)的`ended_gate_fixed_point_cleanup_20261104`，未另建清单／归档体系。formal的`native/.../training_rounds/0004_gate_proj/round_base/`现只留checkpoint_meta.json和config.json，**不能加载模型或续训**；原short已是证据集合，也不能加载。上文10777948060B是保存时完整目录，不是当前metadata-only目录；partial tensor／bit账本和正式κ／两10K／2559／2091／180保护旋转及live→fresh证明仍按当时实测保留，删除payload不改写这些原始事实。

所有正式源码／helper、3份旧源快照及SHA、原失败报告、原prepared与修后有效prepared、CPU修复／两个保存消费者proof、完整64叶24742逐题文件／指标／同train readout／partial账本、原命令和真正退出均保留。readout中native `source_log`旧路径／SHA为清理前历史证明；原观测行在readout内及对应完整fit核心日志保留，不能把已删除重复路径声称当前仍可直接打开。少量launch日志的唯一probe／fresh PID已归收尾receipt，数值夹具删除不影响已有fresh证明，但不能再以它们重新执行验证。

删除后目标逐项不存在，其余受保护证据SHA、981个源码pins、三旧源快照、原after-o／控制raw2、补seven／旧M／readout及D32 MuV raw／最终、尾阶实际源等外部模型stat／小文件SHA均保持。未删原始数据、官方模型或其它线程产物，未启动GPU或下一批实验；本页只记录已结束固定点，不能推广为所有gate／Muon无效。

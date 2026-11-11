# 同父down D8/K12唯一邻点

问题是原同父D16 raw的局部权重/输出误差改善及完整八项正差，在更小一次向量划分时是否延续或出现拐点。唯一拟合父为原C0 after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a`：原BF16→common after_o→C0 gate→自身up S2×10K/500；直接复用已完成稳定前驱，**不读取D16/32 down权重作为拟合或恢复初态**。

主控制只复用 [D16原raw](../down_geometry_baseline_cmcc_20261103/README.md) `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`：八项67.01259272688102，logical2.497889643130095/packed2.511264303456182，tensor4,670,268,096B。D32原raw `ed507bcb-9b83-4e4c-9859-daaef44d80d2`仅参照。两个控制均不重训、不重评。D16仍有自身完整恢复/九项在途，本邻点不改变原四条链及S4安排。

## 固定变量、预算与停止

唯一变量族为D16/K24/B16384→**D8/K12/B32768**。S2各10K、M36 pooled BSQ、E128/D128/R0、transpose=True、input保护256、双侧block32旋转、原BF16 AdamW(.9,.95)/eps1e-8/WD0/.003 linear、全部损失、seed31/shuffle600031及live目标保持。`B×D=262144`只匹配标量曝光；K改变球面/commitment/熵尺度，向量数、初始化、FLOPs及实际存储也不同，不称纯D因果或等算力/等物理费用。未加入Muon、细尾K或辅助loss调参。

仅GPU1、成熟allocator `expandable_segments:True`。CPU来源/配置/bit/kernel diff证明完成后交Astra核心窄审，明确放行前不跑GPU；随后真实M36/B32768/S2各2更新＋独立fresh≤30分钟。通过才从原after-up fresh正式S2各10K→编码保存/独立fresh≤90分钟，完整raw八项≤90分钟，再同原Alpaca train8×256输入部署读出≤30分钟。完整八项保持64叶24742题、M B64/其它B32、0-shot/无限额；不重评控制、不降B/M、不关闭fused绕过新shape。

只有这一条有限阶段链；失败停该候选，不自动retry、占卡或缩正式batch。本次**没有500/九项预算，也不预留后缀**。若raw八项不高于D16且output误差也不改善，结束此邻点、不再缩D；即使净正或有明确取舍，也只保留单seed事实，待D16及两个Muon完整九项后再由研究审定，不能按M或局部NMSE自动延长。

## 原核缺口与独立有效消费者

冻结N snapshot的两个fused launcher原来都设`BLOCK_OUT=min(next_pow2(OUT),64)`；D8为8，而本机Triton NVIDIA BF16 dot的N最小16。独立 [effective_fused_multistage_decoder.py](effective_fused_multistage_decoder.py)逐字节复制原源，**仅把两个launcher表达式改成`max(16,min(next_pow2(OUT),64))`**。逻辑OUT仍8；现成`mask_out`保留真实w_out/b_out加载及y存储边界，步长、实际权重维度、归约、cast、stage累加和其余kernel体都不变，D≥16原受支持tile表达式结果保持。

[consumer.py](consumer.py)仅在本实验进程把已注册N模块的公开decode函数及`vae_linear`已导入的函数重绑到有效文件；N模块本身的身份/路径与原namespace检查保持。single消费者与group8 prewarm动态导入都取得同一有效函数，未修改snapshot/共享源。独立fresh、full8子进程、train readout均先安装并核该绑定。保存时在metadata `runtime.effective_consumer`与`provenance.effective_consumer`同时记录原/有效源绝对路径及SHA；原`native_code_sha256`仍只表示原base依赖，不能把它冒称未改fused实现。普通公共入口并未因此通用支持D8，复现需本包有效消费者与原N runtime同时存在。

新增验证只覆盖这一shape/consumer：CPU核真实36个12032×4096未保护转置矩阵可整除8，穷举4096个K12码字的原packer精确回读/高4位padding为零，配置及两行kernel精确diff。GPU短验检查两stage各2次真实前后向/有限非零更新与峰值、实际72份K12载荷全部bit回读/高4位为零、冻结补集/保护/旋转exact；用前8个真实D8 decoder、每层固定前257个真实code向量核single及group8对原同语义分解reference。BF16跨核不要求逐位一致，事前数值界为relative RMSE≤0.005且max_abs≤0.04×reference_RMS＋0.002，逐层记录实际值；异常需查根因，不能静默放宽。完整group8预热、保存重载3675个持久字段和同输入BF16 logits仍按原exact合同，未扩展exact-resume。

## 费用与迁移边界

K12每码字实际占2B，高4位为padding；同3bit主码不等于同packed费用。根据真实D16账本及矩阵推算：整体logical2.497463723887568、packed2.7662731668223506、full-model tensor4,891,672,128B，比D16多4.7407%。立项时这些仅为**推算**；下文短验已实测核得同数，正式已核自己的全组件、padding、tensor/完整目录及linear外embedding/head/norm账本，实测见下方正式结果。当前验收只限制logical≤2.6；packed>2.6本身不违规，也不能称与D16同物理存储。

迁移必须重新核原生矩阵、保护后flatten、D/K码字padding、decoder形状、实际group/peak和全账本；本masked OUT tile只处理本机原fused的输出tile约束，不证明任意D/K、其它硬件/模型/新kernel都合法。保留同父、同消费者与完整下游对照规则；最终入选仍须以原始BF16完成可复现整链及跨模型验收。E256阶段M正差而完整九项−0.5833121039pp的已结束反例，说明本阶段不凭局部误差或单M决定最终采用。

## 入口与真实状态

调用者激活`bitvae`，在`/home/cmcc/VAELLM`中用tmux/nohup后台执行。独立模型根为 `/data1/cmcc/VAELLM/artifacts/down_d8_neighbor_cmcc_20261103/`，证据根为 `result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/`。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d8_neighbor_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d8_neighbor_cmcc_20261103.sequence --phase short
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d8_neighbor_cmcc_20261103.sequence --phase formal
```

最小CPU已于11月3日17:11实际exit0，15.567706秒，CUDA未初始化。真实36down布局、live digest、全部原配置、4096个K12码字、原/有效kernel仅两行diff和D≥16 tile不变均通过；当前 [prepared.json](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/prepared.json) SHA为 `b1e6b4fa42ee3ba5a42cb5008f9aaf0dd52a64a334a21f52ece3bd9a471425bf`。首CPU暴露直接替换模块身份违反原N namespace检查，在加载拟合数据/运行GPU前停止；实现已改为保留N模块身份、仅重绑明确decode函数，复验通过，未绕过原检查或改snapshot。

Astra/ultra已针对上述prepared SHA明确`approved_GPU_short`、无必须修点，审查范围是两处BLOCK_OUT补16、逻辑OUT8/stride/mask/归一化/舍入不变，以及动态函数对象/co_filename/globals/full8 loader确绑独立effective；原 [core_review.json](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/core_review.json)记录来源和范围。该审查只放行真实GPU短验；后续已完成的GPU证据见下文。原kernel SHA `60d80c3d71b8bbe30bd69780a6c07a7fb546b5bf9af1170093140e8914cb3faf`，有效kernel SHA `b1b92efe27ef6c65440847114c37097074bbcfce383d48605447b6b993bc21aa`；精确diff存prepared。各phase的`*_run.json`/formal report保存命令/PID/start、日志、源码SHA、真实预算和退出状态。无提交则不伪造commit ID。

真实短验已全部exit0，worker整段268.0178秒，UUID `3a91d586-9009-41a8-8aab-7b9571eb1a2e`。原M36/B32768/S2各2的四次有限梯度/参数更新分别改变208552/199003、208462/198536个元素；72份实际K12载荷高4位为零且所有bit回读/repack exact。8个真实decoder各257行的single/group8与同语义分解reference，本次max_abs/relative RMSE均0；只记录此实测，不把跨核exact升为通用要求。原冻结补集/保护/旋转exact，3675字段及同输入BF16 logits独立fresh exact，完整group8 216/216、failed0。峰allocated57,772,397,568B、reserved58,053,361,664B；短验实测logical2.497463723887568/packed2.7662731668223506，tensor4,891,672,128B、保存时完整目录4,894,961,080B。详 [短验证明](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/short_verification.json)。

唯一正式阶段链从原after-up fresh执行，tmux `down_D8_formal_g1_20261103`、driver `2290502/start43548200`；原fit worker `2291223/start43551317`已于17:55完成两阶各10000更新、编码保存及独立fresh，未读取short权重。正式[report](../../artifacts/down_d8_neighbor_cmcc_20261103/formal/report.json)为`completed_D8_formal_raw`，1584.3205秒（含加载／拟合／编码／保存／fresh，不是纯kernel计时），峰allocated57,772,397,568B；raw UUID `a1c3ff10-1316-4c03-bc22-f33b0eb0e635`。[独立strict_reload](../../artifacts/down_d8_neighbor_cmcc_20261103/formal/strict_reload.json)为exit0，3675字段及同输入BF16 logits exact，实测logical2.497463723887568／packed2.7662731668223506、tensor4,891,672,128B，完整保存目录4,894,961,086B。两行effective消费者及原冻结合同保持。

[formal_run.json](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/formal_run.json)于11月3日18:04:51自然完成：`completed_D8_raw_eight_and_readout`／exit0，fit、完整raw八项、readout与外层sequence全部exit0；driver2290502及fit／评测／readout子进程均已退出。UUID `a1c3ff10-1316-4c03-bc22-f33b0eb0e635`贯穿真实formal、独立fresh、完整八项和train读出。来源仍是原after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a`，live dense digest `c2ffa5bbaffa477fd0f1b7e9be3a8efb4e32557ee000d8cb45fa4633eba26c05`；实际M36、B32768、S2各10000、seed31/shuffle600031及标量曝光262144，冻结完整补集／保护／旋转exact。两阶真实初始化摘要已记录，因shape不同不要求等D16或C0；72份K12实际载荷回读/repack exact且高4位为零。复用本包已通过短验和effective消费者审查，收尾只CPU读证据，不重训、重评或加载模型。

完整[八项原件](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/raw_full8/full8_summary.json)与[D16同父raw控制](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_full8/full8_summary.json)已配对核证：两臂各64份逐题文件的记录SHA全部匹配、24742行完整，doc_id/doc_hash/prompt_hash/target_hash逐条一致，MMLU57科14042题与其它七项10700题齐全。推理设置、B64/B32、0-shot／limit=None和各task metric一致，八项均分复算同记录。

| raw完整任务（分） | D16/K24 | D8/K12 | D8−D16（pp） |
| --- | ---: | ---: | ---: |
| mmlu | 62.9112662014 | 63.9153966671 | +1.0041304657 |
| arc_challenge | 51.6211604096 | 52.6450511945 | +1.0238907850 |
| arc_easy | 78.0723905724 | 78.4511784512 | +0.3787878788 |
| boolq | 83.9143730887 | 83.1192660550 | -0.7951070336 |
| openbookqa | 38.4000000000 | 37.8000000000 | -0.6000000000 |
| piqa | 75.1904243743 | 75.3536452666 | +0.1632208923 |
| rte | 79.0613718412 | 78.3393501805 | -0.7220216606 |
| winogrande | 66.9297553275 | 66.5351223362 | -0.3946329913 |
| 八项均分 | 67.0125927269 | 67.0198762689 | +0.0072835420 |

MMLU为8834→8975/14042，多141题；四项升、四项降。八项仅高0.007283542022946676pp，不能称稳定优胜或把单M收益直接换成最终九项收益。本次为D/K/B同标量曝光配方比较，实际shape、初始化、BSQ尺度、计算与packed费用有差异，不作纯D因果结论。

同原after-up、同Alpaca train前8×256 token的[部署读出](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/raw_readout.json)与[D16读出](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_readout.json)已核input SHA均为`c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`、36down全覆盖，weight/output参考分母及元素数相同：

| 真实BF16部署误差 | D16 | D8 | D8相对变化 |
| --- | ---: | ---: | ---: |
| weight MSE | 4.045173232548224e-5 | 3.738775945652523e-5 | −7.5743922270% |
| weight NMSE | 0.058529908762232255 | 0.05409662390246275 | −7.5743922270% |
| output MSE | 0.11114765897009377 | 0.12208081772033445 | +9.8366073128% |
| output NMSE | 0.013405357247111244 | 0.014723989598385437 | +9.8366073128% |

weight NMSE在36/36层改善，output在34/36层改善但总量恶化；L34 output NMSE由0.016392839253804692升至0.02923122059776242（+78.3170086962%），贡献汇总output MSE净增的161.42282635%，其它改善抵消部分增量。另L2为+2.8761579241%，贡献净增仅0.0008386763%。这是固定train局部功能误差，不能替代下游质量或据此逐层拼接；未发现新的实现／协议异常。

| 实际费用 | D16 raw | D8 raw |
| --- | ---: | ---: |
| formal report含拟合／编码／保存／fresh（秒） | 988.6184 | 1584.3205 |
| 完整raw full8内部计时（秒） | 477.6376 | 478.2460 |
| logical bpw | 2.497889643130095 | 2.497463723887568 |
| packed bpw | 2.511264303456182 | 2.7662731668223506 |
| 全模型tensor（B） | 4,670,268,096 | 4,891,672,128 |
| 序列化权重（B） | 4,671,537,493 | 4,892,941,733 |
| 保存时完整目录（B） | 4,673,279,416 | 4,894,961,086 |

D8本包外层fit1616.5391秒、raw八项502.3818秒、readout46.7130秒，完整阶段2166.4438秒（36.1074分钟），不含CPU／短验／固定前驱；外层费用与内部report计时分开，不能混称纯kernel性能。D8的decoder少369,792B，但K12 padding使实际code多221,773,824B，最终tensor净多221,404,032B（+4.7407135404%）；总padding为1,867,087,872bits。logical包含码、decoder、保护indices/values/scales、rotation及route frame，packed另含实际padding，tensor另计embedding/head及norm。formal峰allocated57,772,397,568B／reserved58,053,361,664B；目录实际文件大小已核，logical通过≤2.6，packed>2.6不违反当前验收约束，但物理费用更高。

固定阶段链已结束，现有证据是单seed下M/ARC-C等有收益、均分近零增量、局部weight改善而output和物理费用恶化的明确取舍；尚未测500或最终九项，不称全面被D16支配，也不默认延长。原卡“八项不高于D16且output不改善”的双条件未同时命中（均分微正），因此不据该条件宣称自动淘汰。Astra／root已审定此固定阶段结束、不采用、不接500／full9、不再缩D；保留其非支配单seed候选事实。D16、VAE_mu闭合及类别Muon三份既定九项已齐，唯一一次后继审定已结束，Astra／root决定不给500／full9，formal raw `a1c3ff10-1316-4c03-bc22-f33b0eb0e635`的唯一保留用途终止；已按下述receipt清理权重。metadata/config、report/strict及原指标／预测／源码、所有父/control继续保护，不据此预留GPU或启动新队列。

正式读取的短验证据仅为report/strict JSON，短验已结束且无reader/后续载荷用途。已清其raw权重、同输入临时载荷及3份已归并退出/导入散文件，共5个精确路径，按删除前st_blocks实释4,893,270,016B；精确路径/原因/大小及保护记录hash不变核查存启动核证。先前已解决的CPU namespace散日志2份另实释8192B，不计入该5文件数。保留short metadata/config、完整report/strict/数值与pack证明、核心short_fit.log、CPU/core证明及所有正式/源/control；short目录现仅为证据集合，不能再加载。

正式阶段收尾仅清4个已结束且无reader/后继读取的生成杂项：`formal/same_input_deployment.pt`及`formal_sequence.log`、`formal_raw_full8.log`、`formal_raw_readout.log`三份重复导入／预热日志。逻辑322,978B，按删除前st_blocks实释335,872B；精确路径、原因、大小、FD/mapping与保护核查归既有[启动核证](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/startup_verification.json)的`formal_stage_cleanup`，不与此前short清理重复计数。当时formal raw与metadata/config、完整report/strict、核心fit及原full8日志、原指标／预测／train readout、源码/pins及全部退出／绑定证据保留；该次仅删除fixture并未影响可加载性，后续正式权重清理以本次下文为准。

经验依据：[压缩参数与同曝光边界](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)、[新增shape生命周期峰值](../../docs/experience/lessons/memory_precision.md)、[保存/消费者合同](../../docs/experience/lessons/checkpoint_lifecycle.md)。本次新增阶段事实是weight／output误差及单项／均分排序不单调，当前证据不升级为最终质量结论；root持唯一执行计划、common lessons及索引，本文是本包实现/运行证据的唯一说明。

## 已结束分支的产物收尾（11月4日）

按Astra／root的既定结束决定，删除前盘点了本包生成物、外部后继、活动GPU1 D16-Muon的1024个源码pins和新gate的已确认父／控制依赖；候选路径无活跃FD／mapping，且新gate只复用成熟Muon源码、不读取三包载荷。精确路径／原因／逻辑bytes／st_blocks归既有[收尾原件](../../result/qwen3_8b_mmlu65_20261004/down_d8_neighbor_cmcc_20261103/startup_verification.json)的`ended_branch_cleanup_20261104`，本包本次删除1个文件，逻辑4892941733B、按删除前st_blocks实际释放4892946432B（4.556911GiB），不重复计入此前short或fixture清理。

`artifacts/down_d8_neighbor_cmcc_20261103/formal/raw_model`现为metadata/config证据集合，**不能再加载模型**；本次仅删其正式raw权重；之前short已为不可加载的证据集合。保留M+141题／ARC-C正差及高物理费用的非支配取舍事实，未把packed>2.6称作违反logical验收。所有原始指标／预测／train readout、corelog、metadata/config、report／strict、源码和exit均保留；少量重复wrapper／导入／warmup日志已归并后移除。删除后目标逐项不存在，其余受保护证据SHA精确保持；原C0 raw／控制、D16 raw／最终、VAE-Muon raw／最终、新组合活动模型stat及小文件SHA、GPU1源码pins均不变。此为已结束分支的收尾，不新增训练／评测或算法决定。

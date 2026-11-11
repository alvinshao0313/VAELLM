# 同C0 after-up的D16向量batch单点研究

唯一变量为实际每层向量batch `16384→8192`。保持原D16/K24、M36 pooled BSQ、S2各10000更新、symmetric E128/D128/R0、layer/swish、BF16 AdamW(.9,.95)/eps1e-8/WD0、LR.003 linear10K无warmup、全部loss、seed31/shuffle600031、transposeTrue、input保护256 INT8/block32旋转、同live统计与完整前缀。不补步数或LR。每层每更新标量呈现262144→131072，M36/S2/10K总呈现188743680000→94371840000。第二阶段残差随第一阶段反馈变化，这是固定更新预算的batch/曝光/pooled熵配方比较，不是等曝光或纯熵机制归因。

共同父为真正含36个dense down的C0 after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a`；来源、实际live dense digest、冻结前缀、原统计和原D16工程证据复用[原包](../down_geometry_baseline_cmcc_20261103/README.md)。直接引用原D16 raw `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`的[完整八项](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_full8/full8_summary.json)（67.01259272688102、M8834/14042）及[同train读出](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_readout.json)。不加载原D16 raw作为训练初态，不重训或重评控制。

`bindings.recipe`复用原D16真实配置，仅把有效runtime下down向量batch改为8192；`fit.execute`实际native调用显式取该batch。`prepare.py`在CPU绑定原完整证据与来源、原dtype和397728参数布局、真实stage0初始化；第二阶段不通过连续构造CPU裸VAE模拟。正式入口在真实factory/create_optimizer边界强核两份原stage初始化digest，首个真实训练输入强核BF16 `[8192,36,16]`，实际记录各stage的10000训练forward。原stage初始化digest为`021859dad9089b4d398099641618e146509f51729f17b0c5b207081dd854efcc`、`efd5e3f2a140e0a28d81f43b8d4a35204bae631fa1caee4f0f92daba65f78b12`。模型中间编码/extract_single的真实RNG消费完整保留。

仅GPU4，启动及阶段交界一次核空闲，冲突则失败停止，无轮询、重试、占卡或守护。固定有限链为正式fit/save/独立fresh≤90分钟→raw完整64叶24742八项（M B64、其它B32、0shot、limitNone）≤90分钟→同Alpaca train前8×256真实BF16部署W/output NMSE≤30分钟，然后无论质量如何结束。沿用成熟`expandable_segments:True`和原消费者；所有进程继承本stage timeout进程组，不启动detached LongBench。没有500/full9、其它batch/step/LR网格或自动采用。正差或明确质量成本取舍仅登记下次审定。

读出input SHA固定`c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`，仅train局部诊断。原校准为开源Alpaca train512×256，沿原runtime缓存与许可/来源记录，不新增数据或测试题选择。B变化不改部署布局，预计logical2.497889643130095/packed2.511264303456182、全模型tensor4670268096B；最终以真实save账本为准，含decoder、scale、保护、旋转、compact frame和padding，目录/序列化另报。

CPU准备与正式序列均由已激活`bitvae`的caller用nohup后台启动；脚本不激活conda。独立artifact根为`/data1/cmcc/VAELLM/artifacts/down_d16_batch8192_cmcc_20261103/`，证据根为`result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/`。`prepared.json`绑定代码SHA/原控制/来源/有效配置，`batch8192/run.json`记录实际命令、PID/startticks、timeout、日志、退出与完整耗时；fit报告/`effective_factory.json`记录有效输入与stage初态；`comparison.json`在全部完成后整理分项、成本与误差。正式进程依赖此包及原pins，运行中保持冻结。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d16_batch8192_cmcc_20261103.prepare
bash experiments/down_d16_batch8192_cmcc_20261103/run.sh
```

参考经验为[同码率结构、实际曝光与部署误差边界](../../docs/experience/lessons/compression_choices.md#当前先闭合同码率结构对照)及[实验流程](../../docs/experience/WORKFLOW.md)。复用成熟D16 shape/保存/fresh证据，更小batch不重复通用GPU short、500或exact-resume审计。完整阶段结果及边界见下节，不宣称第一阶段定稿、九项采用或跨模型适用。

## 完整阶段结果与核证（11月4日）

11月03日20:09:32自然结束，[run.json](../../result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/batch8192/run.json)为`completed_fit_fresh_full8_readout`／exit0，fit／raw_full8／raw_readout各外层均exit0；11月4日controller2397976、三个timeout及正式／fresh相关worker均已退出，完整产物与退出相互核对。固定链严格止于raw读出，没有500、LongBench或九项。

正式[fit报告](../../artifacts/down_d16_batch8192_cmcc_20261103/batch8192/formal/report.json)实核两个stage各10000个真实training forward，输入均BF16 `[8192,36,16]`，397728个参数，实际base LR.003；两个stage初态digest分别`021859dad9089b4d398099641618e146509f51729f17b0c5b207081dd854efcc`／`efd5e3f2a140e0a28d81f43b8d4a35204bae631fa1caee4f0f92daba65f78b12`，与原D16逐项相同。live down／冻结前缀／保护与旋转exact证据均沿真实fit报告，未用CPU连续裸构造模拟stage2。每层每更新曝光131072标量、全程94371840000，仅为控制一半，非等曝光机制对照。

最终raw UUID `94bd132d-c598-42ac-94c9-c40b071f1855`，3675持久字段独立fresh及同输入BF16 logits exact／exit0；[完整八项](../../result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/batch8192/raw_full8/full8_summary.json)与原D16 raw `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`的64叶／24742逐题SHA、行数及doc_id/doc_hash/prompt_hash/target_hash配对均已CPU核对；task版本／metric keys／M B64、其它B32／0-shot／limitNone保持，只复算已有指标、累加已有M acc。

| 完整任务（分） | 原B16384 D16 raw | B8192 raw | B8192−控制（pp） |
| --- | ---: | ---: | ---: |
| mmlu | 62.9112662014 | 62.8471727674 | -0.0640934340 |
| arc_challenge | 51.6211604096 | 52.5597269625 | +0.9385665529 |
| arc_easy | 78.0723905724 | 78.7457912458 | +0.6734006734 |
| boolq | 83.9143730887 | 82.5993883792 | -1.3149847095 |
| openbookqa | 38.4000000000 | 38.4000000000 | +0.0000000000 |
| piqa | 75.1904243743 | 74.9727965180 | -0.2176278564 |
| rte | 79.0613718412 | 78.3393501805 | -0.7220216606 |
| winogrande | 66.9297553275 | 67.9558011050 | +1.0260457774 |
| 八项均分 | 67.0125927269 | 67.0525033948 | +0.0399106679 |

MMLU8834→8825/14042（少9题），八项精确67.0525033947872，对控制+0.03991066790617026pp。Alpaca train前8×256读出与原控制的数据／indices／input IDs SHA完全相同（`c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`）；[部署误差](../../result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/batch8192/raw_readout.json)与[配对摘要](../../result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/batch8192/comparison.json)保留全部分层原值。weight NMSE 0.058529908762232255→0.058386422186784796（相对−0.2451508613%），output NMSE 0.013405357247111244→0.013339708270621508（相对−0.4897219468%）；这是同train局部诊断，不等于测试或恢复后收益。

实际fit含fresh757.4845秒、完整full8控制器476.6135秒、readout42.0541秒；外层全序列1328.5037秒（22.1417分钟，含启动／衔接），各外层fit/full8/readout为777.4693／499.8069／50.3297秒，不能重复相加内外成本。fit峰allocated57329259520B／reserved57948504064B。存储实测logical2.497889643130095／packed2.511264303456182bpw，全部252投影、216 native＋36 compact、pending dense0；包括码／decoder／INT8保护值、indices、scales／rotation／route frame及packed的92897280 padding bits；full-model tensor4670268096B含embedding/head/norm，权重序列化4671537493B、清理前完整目录4673279422B均实核。组件bits、文件bytes、源码和配置SHA以prepared／fit报告为权威；本次metadata SHA `339f5df710366e58424a7e7e3d44285ed9ee531fe9fc017887a401c66236dab4`。

原B16384 D16的同阶段fit报告为988.6184秒、峰allocated57440756736B／reserved60079210496B；本点fit为757.4845秒。此为两次实际墙钟记录，均含各自导出／fresh，未做独立重复计时，不把batch减半等同耗时减半。

本点是单seed、固定更新数下batch／曝光／pooled熵联合配方的阶段读出，不能归因为纯熵作用或宣布恢复后九项胜出；按预定有限链结束、不自动采用或追加500／九项／batch网格。经验由主线程统一归并，未增加机制解释。11月4日CPU核活动GPU1 D16-Muon prepared的1024个源码pins全部匹配，其中没有README；本次不改任何正式代码、配置或pins，不重训／重评；阶段结果整理后按下述receipt收尾。

## 已结束分支的产物收尾（11月4日）

按Astra／root的既定结束决定，删除前盘点了本包生成物、外部后继、活动GPU1 D16-Muon的1024个源码pins和新gate的已确认父／控制依赖；候选路径无活跃FD／mapping，且新gate只复用成熟Muon源码、不读取三包载荷。精确路径／原因／逻辑bytes／st_blocks归既有[收尾原件](../../result/qwen3_8b_mmlu65_20261004/down_d16_batch8192_cmcc_20261103/batch8192/run.json)的`ended_branch_cleanup_20261104`，本包本次删除6个文件，逻辑4671861540B、按删除前st_blocks实际释放4671885312B（4.351032GiB），不重复计入此前short或fixture清理。

`artifacts/down_d16_batch8192_cmcc_20261103/batch8192/formal/raw_model`现为metadata/config证据集合，**不能再加载模型**；正式raw权重及用完同输入fresh夹具已清；两个stage实际初始化／输入记录与完整fresh证明保留。所有原始指标／预测／train readout、corelog、metadata/config、report／strict、源码和exit均保留；少量重复wrapper／导入／warmup日志已归并后移除。删除后目标逐项不存在，其余受保护证据SHA精确保持；原C0 raw／控制、D16 raw／最终、VAE-Muon raw／最终、新组合活动模型stat及小文件SHA、GPU1源码pins均不变。此为已结束分支的收尾，不新增训练／评测或算法决定。

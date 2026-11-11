# Live down D16 的Adam学习率配对

用户11月4日要求使用GPU0–5快速补剩余搜索；本组只回答同C0 after-up5a94d57f…、D16/K24/B16384原AdamW的LR .003是否局部敏感。新点为.0015（GPU2）和.006（GPU4），各≤4GPUh，共≤8GPUh，与先前三点13.5h分开计费，不为失败短验续期。控制直接复用原D16 raw02292fb6…完整八项67.01259272688102及同train读出，双方完整拟合，不按训练loss／M单项提前停或扩.25／4倍。

保持全M36 pooled BSQ、S2各10000、原linear10000无warmup、BF16参数／moments、AdamW(.9,.95)/eps1e-8/WD0、D16/K24/B16384、E128/D128 asymmetric/R0、transposeTrue、input256 INT8、block32、seed31／shuffle600031、同真实父权重和校准。只改变当前down拟合的标量LR，历史冻结前缀保持原身份，不声称前六类使用新LR，不做Muon组合／恢复蒸馏／九项后缀。

实际拟合复用成熟down函数的原code object，以本包私有globals绑定当前输出／参数，不修改原源码或共享globals。wrapper对两stage真实D16控制初始tensor摘要强核；原函数记录的旧C0 D32初始化对比单列，不冒称D16已匹配。只读post-step hook在实际1／2／10000更新记录LR及有限BF16 Adam state，不改变参数、梯度、日程或RNG。常规标量超参复用已适用的M36／B16384／消费者证据，不重复GPU短验。

每臂独立同父fresh拟合→原生保存／独立fresh同输入BF16 logits→完整64叶24742八项→同train8×256部署误差及实际logical／packed／tensor／目录费用。绝对4h期限共享全部依赖，失败／超时清自启进程组，不重试／加臂／占卡。原控制保留，阶段小正只提名讨论，明显任务净退即使局部NMSE改善也不替换控制；完整九项及跨系列目标不变。

调用者先激活`bitvae`、隐藏CUDA，以tmux运行`python -B -u -m experiments.down_d16_lr_cmcc_20261104.prepare`；CPU与必要核心核证后，独立后台执行`python -B -u -m experiments.down_d16_lr_cmcc_20261104.sequence --arm half`和`--arm double`。实际PID／命令／源码及结果在`result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/`和独立artifacts目录。

两臂于14:36:29并行启动，均已完整结束；[half终态](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/half/sequence.json)为15:02:42.827、[double终态](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/double/sequence.json)为15:02:30.610，外层均 `completed_fit_fresh_full8_readout`／`exit0`，fit／独立fresh、完整八项及同train读出均退出0。half：tmux `DownLR_half_g2_20261104_1436`／controller2629331／fit2630005，终态 `3cc09c36-6868-4aaa-900b-8e8a5c6f342d`；double：tmux `DownLR_double_g4_20261104_1436`／controller2629334／fit2630006，终态 `ef2859ac-b790-4588-b9f3-478bf872d5c5`。共同父为 `5a94d57f-d3b9-4c97-adef-a099d659eb6a`，原控制为 `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`，控制未重训或复评。

完整八项均按各任务既定metric等权平均；两臂各64叶、24742条，逐份按JSONL物理行读取核到24742个合法JSON记录，未用Unicode字符串分行替代物理行。原[控制指标](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_full8/full8_summary.json)、[half指标与预测索引](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/half/raw_full8/full8_summary.json)、[double指标与预测索引](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/double/raw_full8/full8_summary.json)如下；分数单位为%，差值为百分点。

| 指标 | 控制LR .003 | half .0015 | half−控制 | double .006 | double−控制 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 八项均值 | 67.0125927269 | 66.5013922722 | −0.5112004547 | 66.8440408557 | −0.1685518712 |
| MMLU | 62.9112662014 | 62.7403503774 | −0.1709158240 | 62.6050420168 | −0.3062241846 |
| ARC Challenge | 51.6211604096 | 51.9624573379 | +0.3412969283 | 54.0955631399 | +2.4744027304 |
| ARC Easy | 78.0723905724 | 78.4511784512 | +0.3787878788 | 78.6616161616 | +0.5892255892 |
| BoolQ | 83.9143730887 | 82.0489296636 | −1.8654434251 | 82.1712538226 | −1.7431192661 |
| OpenBookQA | 38.4 | 39.4 | +1 | 38.4 | 0 |
| PIQA | 75.1904243743 | 75.0272034820 | −0.1632208923 | 75.2992383025 | +0.1088139282 |
| RTE | 79.0613718412 | 75.4512635379 | −3.6101083032 | 75.0902527076 | −3.9711191336 |
| WinoGrande | 66.9297553275 | 66.9297553275 | 0 | 68.4293606946 | +1.4996053670 |

任务间存在正负抵消：half的ARC／OpenBookQA增益未抵消BoolQ、RTE及其它下降；double的ARC Challenge +2.4744pp和WinoGrande +1.4996pp，仍未抵消RTE −3.9711pp、BoolQ −1.7431pp及MMLU下降。不能凭单项正差或double较小的净下降替换原控制。

同原父BF16参考、Alpaca `train` 前8×256部署读出，三份输入SHA均为 `c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`。原[控制读出](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_readout.json)、[half读出](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/half/raw_readout.json)、[double读出](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/double/raw_readout.json)的weight NMSE分别为 `0.058529908762232255`／`0.059933513333869544`／`0.06024252420134077`，half／double相对恶化 `2.3980980004%`／`2.9260517833%`；output NMSE为 `0.013405357247111244`／`0.013961007834788663`／`0.01384568300449754`，相对恶化 `4.1449890326%`／`3.2846999097%`。该固定train读出是局部诊断，完整任务判断以上述八项为据。

[half拟合报告](../../artifacts/down_d16_lr_cmcc_20261104/half/formal/report.json)与[double拟合报告](../../artifacts/down_d16_lr_cmcc_20261104/double/formal/report.json)确认两stage各10000次更新、`comparison_D16_initialization_match=[true,true]`、非down冻结字段及保护／sign／block32 spec exact。原函数的旧C0/D32初始化对比为false，不能误读为本包D16同初态对照失败。两stage实际1／2／10000更新的LR分别为half `.0015/.00149985/1.5e-7`、double `.006/.0059994/6e-7`，BF16 Adam moments各观测点有限；其余配方、数据与日程保持。[half独立fresh](../../artifacts/down_d16_lr_cmcc_20261104/half/formal/strict_reload.json)及[double独立fresh](../../artifacts/down_d16_lr_cmcc_20261104/double/formal/strict_reload.json)均核3675持久字段及同输入BF16 logits exact。

两臂实测logical均 `2.497889643130095`、packed均 `2.511264303456182` bpw，计入code、BF16 decoder/norm、保护indices/q/scales、rotation与route frame；完整tensor各 `4,670,268,096 B`。保存时目录half `4,673,279,437 B`、double `4,673,279,438 B`，另含非投影tensor、序列化和metadata/config，不能与投影logical口径混用。logical≤2.6通过，raw八项不替代完整九项或最终质量验收。

| 成本口径 | half／GPU2 | double／GPU4 |
| --- | ---: | ---: |
| 整链秒数 | 1573.4414958953857 | 1561.2208800315857 |
| 整链GPUh | 0.437067 | 0.433672 |
| fit报告含独立fresh秒数 | 991.960348367691 | 982.9729175567627 |
| 完整八项内部秒数 | 478.61693120002747 | 473.04742312431335 |

两卡整链合计约 `0.870740 GPUh`，各自在4GPUh硬上限内；并行墙钟不代替两卡费用总和，外层还含初始化／退出及读出。实际命令、源文件SHA、共同父／原控制、数据与种子见[prepared](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/prepared.json)、两份sequence、拟合报告及各自评测配置，未伪造提交版本。

本组已结束，Astra按完整八项及部署误差决定保留原 `.003`，不扩 `.25/4` 倍，不接恢复或复评。此证据仅支持当前父、D16/K24/B16384、固定seed与三点比较；不称全局最优，也不将gate旧负例或其它优化器／模型条件合并为通用LR规律。经验增量为本条件两侧倍率均未改善任务与误差，并保留任务间抵消的分项边界；已归并至[同日程有限LR比较经验](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)，实际接续由唯一计划维护。

已按授权清理两臂各自 `formal/raw_model/pytorch_model.bin` 和 `formal/same_input_deployment.pt`，共4文件，逻辑 `9,343,690,370 B`，实释分配空间 `9,343,713,280 B`（`8.702011 GiB`）；精确路径、删除前大小／FD/maps核查边界及保护核对见[本包receipt](../../result/qwen3_8b_mmlu65_20261004/down_d16_lr_cmcc_20261104/closure_20261104.json)。两臂metadata/config、report、strict核证、全部原指标和128份逐题预测、核心日志、prepared/sequence及源码保留；历史JSON模型路径在删除后仅作来源，不能作为可加载成品。原D16控制、共同after-up、数据、PUBLIC技术pilot和在途BF16 up profile均保护；不重复计入R64／K24／up其它批次的清理量。

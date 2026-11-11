# Live down D16 的旋转块对照

研究问题：在同一 C0 after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a` 上，将全部36个 down 的双侧 Hadamard block32 改为64，能否改善原 D16/K24/B16384 的 raw 完整八项与实际部署误差。

控制复用真实 D16 raw `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`，完整八项67.01259272688102及原 train8×256读出，不重训或重评控制。固定 S2各10K、M36 pooled BSQ、原 BF16 AdamW／LR .003／linear10K无warmup、E/D128 asymmetric/R0、input256 INT8、transpose=True、原父真实权重与校准、init31及shuffle600031。B×D=262144只匹配本点原曝光，不是等FLOPs声明。

原4K×12032未保护矩阵两轴均整除64；实际sign由module_name/seed/轴长决定，须与控制逐字段相等；只有rotation spec的block_size改变。冻结全部非down持久字段及保护indices／qvalues／scales／sign，核两stage原D16初始化摘要，保存后由独立真实消费者核完整字段／同输入BF16 logits／实际block64与全费用。

新包只复用成熟N消费者及拟合核心，不修改被GPU0／2／6／7依赖的原包。新增CPU核当前接线／resolved64／真实源，复用已适用M36/D16/B16384及block64证据，不重复常规超参搜索的GPU短验或恢复审计。单GPU4原raw拟合／fresh→完整64叶24742条八项→同父同train8×256读出，共同绝对上限4 GPUh；失败停止后继，无重试、500恢复或九项后缀。预计数十分钟至约一小时，硬上限不是预计耗时。

本点按完整同阶段八项、功能误差、logical／packed及tensor／目录费用报告；八项不胜且output不改善即结束固定点，有取舍或正证据仅提名后续讨论。最终≤2.6按实测logical审计；raw八项不能替代总目标九项或跨模型验证。它是既有live配方的有界缺口补齐，不冒充原BF16协议20槽已经完成。

调用者激活`bitvae`后，在CPU隐藏CUDA的tmux中运行`python -B -u -m experiments.down_d16_rotation64_cmcc_20261104.prepare`；CPU及必要核心核证通过后，以相同环境运行`python -B -u -m experiments.down_d16_rotation64_cmcc_20261104.sequence`。实际运行状态、源码SHA、命令、PID及退出由`result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/{prepared,sequence}.json`和独立权重输出保存。

CPU接线实测33.370秒／exit0、CUDA未初始化；实际父目标与补集、D16控制完整八项／读出、36个合法block64、resolved identity唯一差异及源码pins均核。首次入口仅因tmux server环境未继承已激活调用者的`CONDA_DEFAULT_ENV`而在计算前拒绝，失败日志`down_r64_cpu_20261104.log`保留；caller向tmux显式传入实际`CONDA_DEFAULT_ENV`／`CONDA_PREFIX`／`PATH`后通过，实际Python始终为bitvae。Astra必要窄审通过，并补齐自启进程组失败／绝对超时清理及同控制输入SHA断言。

正式于13:53:32后台启动，14:19:23完整结束：tmux `DownR64_formal_g4_20261104`，controller2603275、fit2603629；[外层终态](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/sequence.json)为 `completed_fit_fresh_full8_readout`／`exit0`，fit／独立fresh、raw完整八项和train读出各阶段均 `exit0`。新终态 `d05d2ed0-6834-4d0e-87f1-881a591fcd3d`，共同父及原block32控制与上文一致。

完整八项均为同终态64叶／24742条预测，按各任务既定metric等权平均，未加恢复或LongBench。原[block32指标](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_full8/full8_summary.json)与新[block64指标及预测索引](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/block64/raw_full8/full8_summary.json)对比如下；单位为%，差值为百分点。

| 指标 | block32控制 | block64 | block64−控制 |
| --- | ---: | ---: | ---: |
| 八项均值 | 67.0125927269 | 66.3219168842 | −0.6906758427 |
| MMLU | 62.9112662014 | 62.7189858994 | −0.1922803020 |
| ARC Challenge | 51.6211604096 | 52.4744027304 | +0.8532423208 |
| ARC Easy | 78.0723905724 | 77.9461279461 | −0.1262626263 |
| BoolQ | 83.9143730887 | 82.2935779817 | −1.6207951070 |
| OpenBookQA | 38.4 | 38.4 | 0 |
| PIQA | 75.1904243743 | 75.3536452666 | +0.1632208923 |
| RTE | 79.0613718412 | 75.0902527076 | −3.9711191336 |
| WinoGrande | 66.9297553275 | 66.2983425414 | −0.6314127861 |

同原父BF16参考、Alpaca `train` 前8×256读出，双方输入SHA均为 `c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`；[原读出](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_readout.json)和[新读出](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/block64/raw_readout.json)的weight NMSE分别为 `0.058529908762232255`／`0.05840037723234103`（相对−0.2213082724%），output NMSE为 `0.013405357247111244`／`0.012844172049922026`（相对−4.1862755825%）。局部部署误差改善，但完整八项下降；不把误差代理当成质量收益。

[拟合报告](../../artifacts/down_d16_rotation64_cmcc_20261104/block64/formal/report.json)确认两stage各10000次真实更新、两stage初始化摘要均匹配原D16、非down冻结字段exact、保护与sign持久字段exact，rotation spec仅block32→64；[独立消费者报告](../../artifacts/down_d16_rotation64_cmcc_20261104/block64/formal/strict_reload.json)确认3675持久字段与同输入BF16 logits exact。实测logical `2.497889643130095`／packed `2.511264303456182` bpw，含code、BF16 decoder/norm、保护indices/q/scales、rotation及route frame；完整tensor `4,670,268,096 B`，保存时模型目录 `4,673,279,420 B`，后者另含非投影tensor、序列化和metadata/config，不能当成同一口径bpw。logical≤2.6通过，不表示质量总目标通过。

整链GPU4实耗 `1550.5775108337402 s`（约 `0.430716 GPUh`，在4 GPUh上限内）；fit报告含独立fresh为 `980.4051868915558 s`，完整八项评测内部为 `470.0655429363251 s`。外层各阶段进程跨度还含初始化与退出开销，不把差额算作额外训练。实际命令、源文件SHA、种子／数据／配置分别保留于[prepared](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/prepared.json)、sequence、fit报告与[评测配置](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/block64/raw_full8/config.json)，未伪造提交版本。

本固定点已结束，保留block32；不接恢复，不展开rotation网格。Astra依据完整八项与成本审定此取舍：误差改善没有转成raw八项收益。本结果仅覆盖当前父、D16/K24/B16384及固定seed；旧不同父block64正例仍保留其条件边界，不据此否定双侧旋转或跨模型推广。经验增量是此条件下真实功能误差与完整任务排序不一致，共用经验归并由主线程维护。

已按授权清理本点无后继用途的 `formal/raw_model/pytorch_model.bin` 和已用完的 `formal/same_input_deployment.pt`，共2文件、逻辑 `4,671,845,185 B`，实释分配空间 `4,671,856,640 B`（`4.351006 GiB`）；精确路径、删除前大小／打开状态及保护核对见[本包收尾receipt](../../result/qwen3_8b_mmlu65_20261004/down_d16_rotation64_cmcc_20261104/closure_20261104.json)。模型metadata/config、report、strict报告、prepared/sequence、实际配置、原指标／逐题预测、核心日志及复现源码仍保留；JSON中的raw模型路径仅为历史来源，清理后不是可加载成品。原D16控制、共同after-up与其它在途包未改。

# 全36 down 的 K24 raw 阶段

本批由用户明确批准，仅GPU2、新增总上限4 GPUh。共同父为D16 raw `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`，真实live目标／一次capture来自after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a`。GPU0现有scope500及GPU6/7其它任务保持；本批无500、九项或自动续期／重试。唯一计划由root维护，此文只记录本包接线、状态与原件。

可复用的已审实现是原短卡的compact共同残差、独立BSQ、第三阶export、strict2→3 allocator、实际payload账本及单层artifact；没有修改原short或公共源码。新桥从完整raw只替换36个down为同`CompactRVQLinear`，严格得到180native＋72compact（36gate＋36down）＋0pending。非down全部持久字段和down前两阶code／decoder／normalization／decoder spec／geometry／保护／sign冻结；不保留旧native down的重复载荷。

K24第三阶固定D16、E/D128、R0/symmetric/layer/swish，原BF16参数／Adam动量、AdamW `.003/(.9,.95)/eps1e-8/WD0`、原loss和独立BSQ。按数字层号顺序12组 `(0,1,2)…(33,34,35)`，每层fresh init/shuffle2031，B16384／M3，固定10000更新、原linear10000／warmup0末点，无跨层clip；不从缺少encoder/optimizer/RNG的short部署包假续。首两步／末10000真实更新检查并入正式拟合，不重复2K短验。

原Alpaca train512×256前16一次捕获全部36个down输入，前8估h及路由，后8仅功能读出，仍为已见校准train池。保留原三层实际输入／h作exact身份比较。每层真实payload cap22,033,116B，严格正加权实际Δ、stabletie、最多4轮实际frame重算取最后可行轮，不借层间额度；完整logical≤2.6由真实全模型ledger审计，packed／容器／完整文件另报。

CPU准备只用真实完整compact2及原3层K24的局部fixture核来源、冻结、拓扑、alias／padding／frame账本和独立save-load；fixture明确无formal身份，full8来源入口须拒绝。正式control和K24各自完整保存／独立fresh同输入BF16 logits、180native group8 warm后，才允许完整full8。原native D16 raw八项67.01259272688102仅复用，分别报告native→同consumer compact2转换差及compact2→tail差。

`evaluate.py`仅在本包用通用`inspect(final=False)`后严格核180/72/0、UUID、存储SHA、自己的immutable raw/strict报告及完整36×10K；公共inspect原216/36约束不改。完整八项复用成熟mixed loading/HFLM/sample writer及汇总：64叶／24742题、MMLU B64／seven B32、0-shot／limitNone，控制和候选都评；不冒用原500／旧formal来源。只有候选严格超过compact2和native两者才讨论恢复，质量不足仍是有效完成。

入口为`python -B -u -m experiments.down_tail24_full_cmcc_20261104.sequence --output <新独立输出>`，由已激活bitvae的调用者在tmux后台运行，`CUDA_VISIBLE_DEVICES=2`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。实际环境在入口核对并随进程记录。单一driver从起点设共同4h硬deadline，固定所有子进程顺序；验证／save/fresh≤.5h、fit/encode/route≤2h（组内局部readout保守计入此项）、两full8≤1h、一次capture≤.5h。每子进程timeout取共同剩余与所属剩余预算的较小值，失败停止全部后继；不重开4h。nested fresh异常kill/wait自己的子进程，非零stage即使leader已退仍清理其新进程组，避免孤儿GPU。

CPU新桥接已通过约141秒，CUDA未初始化：两真实完整fixture的180/72/0、完整state／metadata冻结、实际ledger及独立fresh、评测拒绝fixture均通过。**这里只是CPU fixture账本**：compact2 logical2.497922772946565／packed2.511297433272652，full tensor4,670,296,860B；不是尚待完成的正式36×10K K24成品账本／质量。受影响CPU进程树检查已证实nested timeout child被kill/reap、leader已退时残子也清除；权威native八项原件由`allocator_repair/d16/raw_full8/full8_summary.json`的SHA/UUID/64叶24742绑定。最终158源码pins及6个short数据身份文件经Astra核实无差异，必要核心窄审已通过。证据为[prepared.json](../../result/qwen3_8b_mmlu65_20261004/down_tail24_full_cmcc_20261104/prepared.json)、同级`prepare.log/source.diff`。

正式链已在GPU2启动并确认真实拟合：tmux `Tail24_full_g2_20261104_1135`，controller `2497138/start50074204`。共同起点北京时间11:35:54，唯一硬截止15:35:54。独立结果根为[formal_20261104_1135](../../artifacts/down_tail24_full_cmcc_20261104/formal_20261104_1135/sequence.json)，其`sequence.json`记录每个实际worker的PID/start、命令、退出、分类费用及共同deadline；上层同名`driver.log`及各stage日志落盘。

实际capture `2497163/start50074269` 已exit0／46.128秒；formal compact2 control `2498022/start50078882` 及独立fresh `2499256/start50085876` 已exit0，合109.813秒。control UUID为 `5c4caa45-714e-496e-b864-7c6c5730e0d3`，真实180/72/0、3927持久字段、36个compact down的实际forward、180native groupwarm及同输入BF16 logits exact；这份正式控制自身证据与前面的CPU fixture分开存放于`control/report.json/fresh_report.json`。首组 `(0,1,2)` worker `2500041/start50089864` 已实核11:39:15推进到900/10000、LR `.00273`，首1/2更新的有限梯度、每层真实位移及BF16 Adam state检查通过。后续由已启动固定链继续；完整36×10K K24保存／fresh及两full8尚待，未预判质量，也不继续主动轮询整卡。

**已完成重叠层的预算读出：** 首组(0,1,2)完整10000更新、末Adam step10000与component exit0已核，实际拟合353.733986488秒，含编码355.444854053秒、组内全过程372.455087185秒、外层component380.582259417秒。与原三层短卡只有L0重叠；以下为同配方／同cap的2K与10K比较，不是固定route的纯decoder消融，也不是从原2K optimizer精确resume。

已核两次raw/live身份、target／prefix／residual／normalized摘要、mean／scale、normalization／geometry及compact2全部误差逐值一致；真实input／h exact。原独立M3按层重置init2031、独立shuffle2031、逐层BSQ／LayerNorm／Adam且无跨层clip，成员(0,18,35)改成(0,1,2)不改变L0目标或初始化。两者原linear10000／warmup0的step1、2和100、200…2000共22个L0 loss／LR记录点exact；这不证明每步参数／optimizer全状态逐位一致。两末点LR为.0024／0，每层向量呈现数32,768,000／163,840,000，末训练日志70.756974269／353.733088216秒约5倍。

| L0指标 | 2000更新 | 10000更新 | NMSE相对变化 |
| --- | ---: | ---: | ---: |
| 原坐标W NMSE | .043814658000338964 | .043918864762062 | +.2378353877% |
| 加权残差NMSE | .03627253440849039 | .03637280330065705 | +.2764319996% |
| 前8输出NMSE | .016280826768116863 | .016301096554979386 | +.1245009676% |
| 后8输出NMSE | .018225718311105987 | .018225089334071156 | −.0034510411% |

正号表示误差变大，不是任务精度百分点。两端同cap22,033,116B但均重新编码／路由：覆盖23.5736603433%→23.5846336852%，活跃726,114→726,452行，采用轮次均3；payload22,030,717→22,030,742B、frame270,839→269,850B、全部codes20,659,494→20,660,508B，decoder均33,120B／fixed均1,067,264B、code padding0。实际文件22,041,797→22,041,861B，容器／metadata11,080→11,119B。行数净增338不能解释为只有338个向量改路由；frame少989B、codes多1014B，payload净多25B，不能称完全等实际位宽。

原件为[短卡report](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/report.json)的`arms["24"].layers[0]`和[完整g00/report](../../artifacts/down_tail24_full_cmcc_20261104/formal_20261104_1135/groups/g00/report.json)的`layers[0]`；普通误差分别在该层`metrics`／`candidate.metrics`，加权误差分别`weighted_residual_NMSE`／`candidate.weighted_residual_NMSE`，费用均在该层`artifact`，路由分别`allocation`／`routing`。训练轨迹分别为短卡`arms["24"].training.rows`／g00的`training.rows`。两份L0部署文件SHA与报告相符，报告完整／稳定；input／h身份见本次`capture/report.json`。后8仍是既有校准train，只说明本层增加预算没有呈现实质局部部署增益；不证明2K已收敛／最优或发生过拟合。G缺dense-third weighted读出，不能再分解拟合／路由原因；K8长程排序、其它35层及完整任务仍未验证。**保持既定12组、候选fresh和两份full8，不因本读出提前停卡、扩K8或追加GPU。** 可迁移经验归[压缩选择](../../docs/experience/lessons/compression_choices.md#动态阶数的布局退出下界与同cap对照)。

## 全36层raw完整结果与取舍

11月4日实际[sequence](../../artifacts/down_tail24_full_cmcc_20261104/formal_20261104_1135/sequence.json)与[result](../../artifacts/down_tail24_full_cmcc_20261104/formal_20261104_1135/result.json)均为`completed_full36_raw_phase/exit0`：12组三层、全部36层各10000更新，完整候选保存与独立fresh、两份完整64叶24742条八项均完成。全链6708.866秒，约1.864 GPUh，低于新增4 GPUh硬上限；未执行恢复蒸馏或九项。

| 同阶段端点 | UUID | 完整八项均分 | logical / packed bpw |
| --- | --- | ---: | --- |
| 原native D16控制 | 02292fb6… | 67.01259272688102 | 2.497889643130095 / 2.511264303456182 |
| 同消费者compact2控制 | 5c4caa45… | 66.91295465455246 | 2.497922772946565 / 2.511297433272652 |
| K24第三阶候选 | c2063579… | 67.03221634113612 | 2.5998963008180334 / 2.6132709611441203 |

尾阶相对同消费者提高.1192616866pp，但原native转compact2自身下降.0996380723pp，因此相对原native净提升仅.0196236143pp。MMLU正确数由compact2的8828和native的8834变为8971／14042；RTE相对compact2下降2.52707581pp（少7题），相对native少8题，BoolQ也下降，不能称多任务全面提升。各项未舍入值与原始样本保留在result及两份full8目录。

同train前／后8条局部输出NMSE分别改善38.8291%／37.6080%，weight／weighted NMSE改善28.5826%／35.3317%，仍未转换成明显八项净收益；前后8均来自既有校准train，不能称盲测。K24相对compact2增加logical .1019735279bpw、tensor88,535,551B、完整目录88,706,339B，剩余硬位宽仅.0001036992bpw。候选全tensor4,758,832,411B、保存时完整目录4,762,116,169B，包含实际route/frame及全部部署组件，packed超2.6不等于logical门槛失败。

按原预设，严格胜两个raw控制的条件成立，保留这一单seed弱候选事实；相对native净增益很小且几乎耗尽剩余位宽，当前不优先追加500／九项或正式整合，不扩K／步数网格。该取舍不改变独立的三点VAE缺口搜索。固定可迁移规则仍是同消费者对照、全部实际费用计入、同父真实功能读出和完整任务回验；K24／10K及本路由阈值不能直接当跨模型默认。

源码pins不含README，progress/sequence与评测会pin的raw/strict完成报告分开，避免运行进度改变评测身份。原short K8已无后继用途，由其所属收尾按receipt清理3份载荷；历史SHA／CPU与GPU指标仍保留，不能将清理后的历史文件引用写成当前可加载。原K24／compact2三层及capture/h已完成本桥用途，生成载荷按实际后继引用核实后收尾，历史证据继续保留。

已结束且无读者／后继用途的两份CPU fixture生成载荷及重复预检文件，在完整证据并入prepared且Astra收口后清理；其历史尺寸／SHA／账本不代表文件仍可加载。精确路径、无活动FD、删除前尺寸／SHA及删除后保护核实在[cpu_fixture_cleanup.json](../../result/qwen3_8b_mmlu65_20261004/down_tail24_full_cmcc_20261104/cpu_fixture_cleanup.json)，实际释放9,354,715,136B allocated。immutable prepared未改，实际formal control、在途拟合及全部受保护short K24／compact2／inputs/h均保留。

完整raw批次结束后，当前没有已安排的恢复／复评或正式整合用途，按持续清理授权收尾本formal目录的43个生成tensor载荷：两份整模权重、36个第三阶bank、派生capture输入／h／tokens及两份用完的部署数值夹具。已核终态exit0、无活动FD／mapping、路径限定本包formal根；删除前精确路径／原因／大小、删除后保护SHA在[formal_cleanup.json](../../result/qwen3_8b_mmlu65_20261004/down_tail24_full_cmcc_20261104/formal_cleanup.json)。实际释放13,853,560,832B allocated（约12.90GiB），不重计原K8或CPU fixture。原始任务指标／逐题预测、全部训练／fresh／账本报告、配置与源码及核心日志保留，原D16／after-up及新up／block64和GPU0依赖均未删除。formal的模型、bank和capture路径现仅有历史证明，不能加载／续训或复验；单seed弱候选事实和保存时费用不因收尾改写。

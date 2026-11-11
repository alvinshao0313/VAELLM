# 原C0 live-down VAE拟合的Muon单点

本包回答优化器作用于**VAE编解码器拟合**时是否有收益。固定成熟C0 after-up UUID `5a94d57f-d3b9-4c97-adef-a099d659eb6a`，仅重拟合全36个live dense down；原控制是同源AdamW raw UUID `ed507bcb-9b83-4e4c-9859-daaef44d80d2`。复用其已完成raw完整八项65.97718259119465与固定train部署读出，不重训控制、不接D16几何或类别Muon恢复。

正式raw固定链已于11月3日12:24:47全部exit0，总1541.60秒，其中fit＋部署/fresh975.26秒。raw UUID为 `c5ccf174-0c56-4f90-9e29-7d4ba63d2e23`；完整64叶24742题八项66.3532649235，较原AdamW raw65.97718259119465高0.3760823323pp，MMLU8743/14042较原8787少44题。四升四降，增益集中ARC-E/RTE，其余六项均差−0.151218pp；不能称多任务全面保持或知识恢复。同固定train输入weight/output NMSE分别下降4.06776409%/2.24480239%。原件为本result的 [run.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/run.json)、[完整八项](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/raw_full8/full8_summary.json)及 [部署读出](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/raw_readout.json)。

实际logical2.4987414816151494/packed2.5121161419412363，full-model tensor4,671,007,680B，包含全部decoder、码、scale、保护、旋转、compact frame，packed另计padding。Astra已审定原预留的唯一一次原AdamW500＋完整九项闭合，独立在 [closure包](../cat_down_vae_muon_closure_cmcc_20261103/README.md)实施；仅以此raw作父，原raw权重按该具体用途保留。本包本身已结束，不重训VAE、不把raw增益与别臂相加。

## 固定配方与真实矩阵

D32/K48、M36、B8192、transpose=True、S2各10K、pooled BSQ、E/D128/R0/layer/swish、原所有损失与stage norm、input256 INT8保护、双侧block32旋转、seed31/shuffle600031及实际合法train校准全部保持。VAE参数、梯度与原前后向仍BF16，没有持久FP32参数/master；只在Muon内部保留FP32 momentum、执行既有BF16 NS5。原AdamW(.9,.95)/eps1e-8/WD0/.003继续处理全部bias与norm。

从真实 `MultiLayerVAE.model.encoder/decoder` 的 `ParallelLinear` owner读取36个group，不能把GroupConv packed tensor整体正交化，也不能按ndim将二维norm错分为矩阵。每stage有12个实际Parameter／767808元素，其中4个packed权重拆144矩阵、737280元素，其余8个bias/norm叶30528元素：

| 每Linear矩阵 | GroupConv实际存储 | Muon逻辑布局 |
| --- | --- | --- |
| encoder linear_in | 4608×32×1 | 36×128×32 |
| encoder linear_out | 1728×128×1 | 36×48×128 |
| decoder linear_in | 4608×48×1 | 36×128×48 |
| decoder linear_out | 1152×128×1 | 36×32×128 |

Muon沿已确认NS5系数 `(3.4445,-4.7750,2.0315)`、momentum.95/Nesterov、`.2*sqrt(max(rows,cols))`。每次从当前BF16 p与FP32方向计算 `p.float()−ηκU`，只回写一次BF16；FP32候选为临时计算，不跨步保留。外部标准Optimizer param_group LR及原linear10K/no warmup LambdaLR仍从.003开始，矩阵在step内部乘κ，辅助Adam不乘。

每residual stage只在第一个真实梯度校准一个共同κ，覆盖同stage全部E+D矩阵。克隆原BF16 p/grad，调用未替换的native BF16 AdamW一步，得到实际BF16矩阵位移norm A；fresh Muon U在临时零momentum上算一次。以κ0=A/(.003||U||)为起点，在 `[0,2κ0]` 上至多12次纯幅度二分，上端不足最多再翻倍2次，取全部已观察幅度中BF16位移norm最接近A者。相对差目标≤1%；A/U零或非有限、或有界最近值仍不满足时停止该路径交Astra，不换精度、不扩LR网格。校准不得改变真实p/grad/RNG/optimizer state；stage2用自身真实残差首梯度重算，短验κ不用于正式。

原初始化摘要按**原单组flat顺序**另算并须匹配C0两stage `f4f3533c…/4cc98b38…`。原observer看到的新两组顺序单列保留，不能假称原组序相同。E/D各自BF16位移及非零比例只作诊断，不再分别配平。

## 实际如何并行和成本边界

实际实现按上表四个真实`ParallelLinear` owner依次处理packed权重，每个owner将36个独立矩阵视为`[36,out,in]`，NS5用三维`@`做批量matmul，按最后两维逐矩阵归一化；没有逐144矩阵的Python串行循环。四种owner/形状仍依次执行，源码未显式调用`torch.bmm`，也没有跨owner融合或kernel profiling，不能据批量写法承诺加速。已有CPU排列/单矩阵扰动及batch内独立计算对照验证了矩阵隔离。

每stage首κ直接复用一次真实backward的梯度：在参数/梯度clone上做原BF16 Adam参考步，fresh Muon方向算好后供所有幅度候选复用。候选仅执行四份packed张量的逐元素缩放、一次BF16舍入与位移norm归约，缓存已试幅度；不重复模型forward/backward，也不为每个候选重算NS。校准后真实首step仍用真实momentum再算一次方向，因此不能称首步总共只算一遍NS。迁移时须重新核真实owner、group数M、E/D输入输出宽度、D/K与residual结构、packed reshape/norm维度及参数/存储账本；当前入口明确限定M36/R0，不是任意模型的通用适配器。各stage从自身真实首梯度重算κ，不能移用本Qwen κ或短验κ。

实测975.2612秒是`run.py`的整段正式耗时：包括恢复校准runtime/加载after-up、保护统计与S2各10K拟合及编码替换、部署group8 warmup/logits、保存raw及同输入证据、释放模型、独立进程fresh加载/全字段与BF16 logits核验、末端源码校验；不是纯optimizer或纯训练计时，也不含计时前的导入/来源绑定与recipe检查。完整raw固定链1541.5980秒另含完整八项和train部署读出。日志stage1/2末尾约0.0377/0.0379秒/步只是所记训练窗口；现有记录没有同一计时口径的纯Adam配对成本，不能据此比较两者速度优劣。

## 隔离接口与验证

`optimizer.py`仅在独立进程上下文捕获真实 `litebsq.llm_vae.MultiLayerVAE` 构造对象并替换 `native.create_optimizer`，`fit.py`调用原 `continuation.fit_next`，退出后恢复两个绑定。原数据采样、两stage构造/编码、逐类统计及替换算法不复制、不改活动模块。原VAE工厂只给params生成器，因此通过真实对象所有权建立布局。

`VAEMuon`是标准`torch.optim.Optimizer`，可接受原LambdaLR；aux Adam共享实际group/state。其自有schema/layout限定state保存读取。PyTorch默认加载可能把BF16 p对应的momentum cast成BF16，本adapter在super.load前保存原FP32 momentum原值，super后恢复这些原值，不能先降精再float冒exact；辅助Adam moments保留BF16。CPU通过真实BytesIO保存读取检查dtype与逐值，不扩展为训练exact-resume研究。

该规则回链[参数与优化器状态精度不同的重载](../../docs/experience/lessons/checkpoint_lifecycle.md#参数与优化器状态精度不同的重载)，本次只覆盖此新adapter自身schema/layout，没有旧Muon兼容实现。

CPU检查覆盖实际owner/四布局、每Linear扰动隔离与排列等变、aux与native BF16 AdamW两次更新exact、标准原scheduler、κ无副作用及自有optimizer state的dtype/逐值/κ往返。CPU随机梯度只作数学验证，不能用于GPU κ。新增路径短验保持完整8B/真实M36/B8192/S2各2次更新，检查两stage κ、BF16参数/梯度、FP32 momentum、aux BF16 moments、E/D真实非零更新、原flat初始化、全部非down/保护/旋转exact、真实保存/独立fresh/group8。成熟原父与消费者证据复用，不再短测类别恢复或控制重训。

原 `fit_next(short=True)` 的工程短验采用每stage **2步linear日程**（实际base LR .003→.0015），不是正式10K日程前缀；CPU另已验证标准10K LambdaLR，正式仍各10K。短验κ与权重均不用于正式。非零位移阻断仅限每stage前两步；正式第10000步仍记录finite/真实位移，原linear尾端若全部BF16舍入回原值，不将该现象冒作实现失败。

## 已完成验证与源码接续

初次CPU12.10秒exit0，CUDA未初始化；实际owner拆144矩阵，原flat stage0摘要为 `f4f3533c20d1e786be9193ea39f102a448fb6158724379e263a41084d2f26178`，aux原BF16 Adam两步exact，32个optimizer state字段dtype/逐值及κ/step往返exact。原CPU证据保存在 [cpu_before_tail_review.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/cpu_before_tail_review.json)，原SHA `1744300e224eb756d1316890b7c19f86681e4ada1f62cc4439225743faba5bed`。

真实GPU2 short耗时204.7267秒/exit0，UUID `242c366a-0cc7-4fff-97f8-ff850718c67a`；原两stage flat初始化摘要均exact，原非down补集/保护/rotation exact。BF16参数/grad、无持久FP32 p、FP32 momentum及aux BF16 moments均通过，全部四次更新实际非零：

| 工程短验stage | κ | 对native BF16 Adam实际位移norm相对差 | 两步E改变元素 | 两步D改变元素 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5.424683644120595 | 0.0127873173% | 350978 / 333253 | 347401 / 326545 |
| 1 | 5.403490476463650 | 0.0094706560% | 351181 / 333294 | 347485 / 326480 |

3675字段及同输入BF16 logits独立fresh exact，原group8 216/216、无失败；短验峰值allocated57,330,078,720B。实际logical2.4987414816151494/packed2.5121161419412363，tensor4,671,007,680B、短验保存时完整目录4,674,234,831B。原件为 `artifacts/cat_down_vae_muon_cmcc_20261103/short/{report,strict_reload}.json` 和本result的short.log/exit；它只证明新optimizer/部署合同，不证明质量。

short结束后才按Astra审查将正式尾端nonzero诊断限于前2步，optimizer数值更新/κ/精度逐字节保持。精确before/after SHA与三文件diff保存在 [tail_audit_review.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/tail_audit_review.json)：optimizer只改该审计条件，prepare增加一次真实第三步微小LR→BF16零位移边界，prepare/sequence明确绑定旧GPU short源digest。原short report及原source pins未改，不声称它测过新源码。

新版CPU `--after-short-review` 又实际执行12.30秒/exit0，32字段往返仍exact，并确认第三步真实更新在LR1e-30时E/D位移均为0且finite，无伪造正式步数。当前 [prepared.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/prepared.json) SHA为 `98ba6a42d91ca3669367e04090c159cb08877f05b87d43474edcbc8b01e3b831`；其逐文件校验仅允许该明确review变更，Astra已确认未受影响的原GPU short证据可复用，无需再跑GPU短验。

原raw full8保持64叶24742题、MMLU B64/其它B32、0-shot/limitNone。固定部署读出直接复用已核consumer：同after-up、Alpaca train前8×256 token，捕获同36down输入，原live W作参考，真实BF16原group8解码；这是固定train局部误差，不是盲测或最终九项。实测账本计全部decoder/码/scale/保护/旋转/compact frame和padding，另报tensor/完整目录字节。

## 入口与运行证据

调用者shell激活`bitvae`，所有测试/训练用tmux/nohup后台执行。独立模型根为 `/data1/cmcc/VAELLM/artifacts/cat_down_vae_muon_cmcc_20261103/`，原始证据根为 `result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/`。CPU、短验、正式各自分离，正式重新加载同after-up，不接short权重。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_vae_muon_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=31 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -B -u -m experiments.cat_down_vae_muon_cmcc_20261103.run --phase short
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_vae_muon_cmcc_20261103.sequence
```

短验总timeout30分钟；正式fit＋部署/fresh90分钟、完整raw8 90分钟、固定train readout30分钟。fit预计20–40分钟是设计估计，以实际run/report为准。固定链只等待自身依赖，不自动retry/resume/占卡或监控其它任务；失败停后继。CPU prepared、正式report/run均记录实际CLI、PID/start、源码SHA、原父绑定与控制证据；没有提交则不伪造commit ID。

历史启动证据：正式tmux为 `cat_down_vae_muon_GPU2_20261103`，controller `2069046/start41572761`，timeout `2069237/start41573422`，fit worker `2069238/start41573423`；实际CLI、环境、日志和退出保存在本result `run.json`及`sequence.log/exit`，正式模型输出独立在 `formal/`。12:01有限启动核已见stage1第1000/10000步，最近约.038秒/步；首κ为5.424683644120595、匹配误差.0127873173%。正式首两步base/aux LR为.003/.0029997，实际矩阵LR为.0162740509324/.0162724235273；E/D实际改变350978/347401、350912/347339元素，BF16参数/梯度、FP32 momentum及aux BF16 moments通过。最新终态以上方完整raw结果及原run/report为准。

正式已从原after-up真实推进，sequence仅消费short报告/源digest，不读取short权重。核short进程自然退出、`lsof`无reader、精确路径解析在批准范围内且无符号链接/硬链接共享后，删除 `/data1/cmcc/VAELLM/artifacts/cat_down_vae_muon_cmcc_20261103/short/raw_model/pytorch_model.bin`（4,672,277,077B）和同short目录 `same_input_deployment.pt`（307,692B），按删除前st_blocks实际释放4,672,593,920B。删除后保留short metadata/config、原report/strict/核心log/真实exit，以及原CPU、新CPU、精确diff与当前正式report；未碰原父/control、D16或其它任务。short现仅为证据目录，不能再次加载。

固定前驱与BF16分段复现来源见[原down同父包](../down_geometry_baseline_cmcc_20261103/README.md)、[C0完整入口](../fresh_fixed2_cmcc_20261030/README.md)及[common BF16入口](../fresh_common_cmcc_20261030/README.md)。本次复用稳定after-up，仅对down做optimizer比较；不能称七类均验证，最终入选仍须以原始BF16完成整链复现。跨类别/模型需按实际owner重建布局、每stage真实梯度重算κ和费用，不移用本次Qwen κ。

经验依据为[优化器阶段边界](../../docs/experience/lessons/compression_choices.md)与[Muon/Adam及真实位移口径](../../docs/experience/lessons/kd_and_data.md#muonpcgrad和同期adam控制)。既有端到端Muon32的FP32 master/未舍入κ和多任务收益不能替代本BF16 VAE路径；本次raw净正而M退化及局部误差改善并存，最终质量解释须等独立原Adam500闭合九项。

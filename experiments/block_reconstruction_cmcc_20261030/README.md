# 完整块预重建后接端到端恢复

2026-10-30设计，10月31日相邻块机制短验闭合。唯一当前安排见[执行计划](../../docs/experience/records/w25_transfer/current_plan.md#完整块预重建后接端到端恢复)。隔离实现和相邻两块机制短验已通过；32更新/块已据实测固定为本批单候选，正式入口与来源适配见[完整BR36／BR→E](../block_reconstruction_formal_cmcc_20261031/README.md)；真实C0尚未完成，未运行或排入正式36层训练。C0、现有dynamic/Rice及直接E2E继续原计划。

问题是在完整C0上，增加一次按层的block reconstruction（BR）是否改善随后相同E2E规则的最终收益。首组比较已有`C0→E`与新增`C0→BR→E`，保留`C0→BR`中点；E沿H96/Muon32，fresh初始化、各自重算κ和在线学生轨迹。BR中点即使下游下降，也不单凭中点取消已约定BR→E。四端点统一完整九项、真实位宽和全部成本；不把BR局部MSE当下游收益，不把额外训练收益称为等算力优势。

## 旧证据与可复用边界

| 条件及已有证据 | 本次采取的动作 | 来源 |
| --- | --- | --- |
| 旧无旋转/INT8、Alpaca64/max256、每块64步/LR1.25e-5，双流36块的训练与留出MSE均降，八项60.8495→53.1350；192步也未解决 | 不照抄这份数据/步数/LR/FP32 decoder compute，不新增局部MSE晋级门槛。本次变量是当前完整fresh父、已核H96完整输入、当前BF16 decoder路径及后接固定E | [完整负例及边界](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化) |
| 另一旧无旋转单stage、4096×2048、code+D联合逐层恢复使八项39.90→约59，仍未达标 | 可复用独立教师前缀和学生逐层推进原则；不将其收益或3968步预算迁到当前fixed-code新父 | [逐层恢复正例](../../docs/experience/lessons/liftquant_recovery.md#完整逐层恢复的下游收益2026-09-24完成2026-09-30核实) |
| 新native/compact、FP32 master/BF16 compute和原生导出已有完整模型生命周期证据；全MLP短验也已完成 | 复用实际consumer和冻结/保存规则；只验证本BR新增的局部七投影优化、BF16提交与双流衔接，不重做全MLP可学性短验 | [混合Muon接口](../muon_compact_cmcc_20261030/README.md)、[全MLP学习路径](../mlp_learnability_cmcc_20261030/README.md) |

源码参考范围：`qwen3_w25/sequential_block_calibration.py`的双流关系与真实block kwargs，`local_block.py`的`BlockBatch`、`_tree_to`和有效token平方误差公式，`block_calibration.py`的`_cpu_copy`。`_tree_to`会detach，只用于固定块输入，不能用于E2E跨层梯度。

旧`SequentialBlockCalibration`不原样套用：构造函数用单一rectangular池且硬编码逐篇B1，`_capture_inputs`不传显式`position_ids`；`prepare`还复制全部student/target/kwargs。旧`block_batching`只合并同长度与同kwargs形状，不会给不同长度自动padding。本实现改用下面160个既有实际执行批，无需新造ragged padding算法。

不调用旧`optimize_local_block/block_weight_graph`：它们只支持七个native VAELinear、旧统一几何，强制FP32 decoder重建并安装临时投影forward；每更新还遍历整个校准池。当前C0包含compact，BR须直接调用真实完整Transformer层、沿现有BF16可微解码，不把旧weight-leaf/VJP图冒充当前部署路径。旧driver的Alpaca、256截断、固定LR/步数、采样过滤、finalizer也不复用。

## 初态和可训练范围

正式唯一父为已完成新C0：原gate500、live up/down各S2×10K/500及终态保存全部闭合，绑定其真实UUID、pipeline报告、权重/metadata/runtime SHA和实际账本。预期216 native＋36 fixed2 compact，0 pending dense；源验证沿现有C0完整来源检查，不能从gate/up中间点开始。

BR按`0→num_hidden_layers−1`执行，Qwen本轮为0→35。每层训练attention的q/k/v/o与MLP的gate/up/down，含两个残差连接的**完整层输出**；目标在该层输出、下一层输入处，末层目标仍在最终norm之前。每次只更新当前层这七投影的现有decoder，按实际模块所有权派生参数，不把norm仅凭二维形状当矩阵，也不固定写42/48叶计数。全部码、路由/length、保护q/scales/indices、旋转、稀疏字段、norm、bias、embedding/head和其余层保持。无翻码、scale侧车、LoRA或教师权重替换。

当前可用的机制原型父是B原生成品`87571461-7310-46e5-bc2e-904baf1818ed`，路径`/data1/cmcc/VAELLM/artifacts/cat_structure_full_20261029/B/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/final_model`。为覆盖BR真实mixed路径，可只读B的36个gate实际载荷，经既有`adaptive_native_model_20261029.native_prefix.from_native`转fixed2：保留已融合norm/BSQ，不重拟合、不再次融合，所有非gate状态exact。形成216 native＋36 compact的独立**prototype_only**来源；记录原B、转换内容、真实消费者差异和状态摘要，不能称为新C0或质量对照。

原型可在内存中由B重建，不必另留一份永久完整转换父。输出新目录的BR短验成品必须自包含；它即使账本合格也仍是`validation_only`，不能成为正式BR/E初态。旧Muon临时compact父已经清理，不用不存在的路径替代来源。

## 数据、真实批次和mask

首候选复用已核H96静态TF输入：M1024、Qasper32、Hotpot96，1152个固定train身份，原seed137与32组顺序。来源、冻结revision、许可、完整输入与历史选择边界见[原数据记录](../../result/qwen3_8b_mmlu65_20261004/qwen_hotpot_recovery_20261012/README.md)及[来源汇总](../mlp_learnability_cmcc_20261030/README.md#固定输入教师目标与有效token)。M是`cais/mmlu auxiliary_train/train`；QA是官方train。保持原prompt/BOS/EOS、问题、上下文和gold-prefix输入，不重采、去重、按测试错误选样、另补学科标签或截到256/1024。

本阶段只复用输入，不消费旧QA/M logit目标。teacher hidden来自本次实际原始BF16教师的独立前缀；无需为了BR再读一整份旧全词表target payload。数据入口可复用HOT的实际dataset/manifest/源case验证，避免把`mlp_learnability.data.load()`最后的KL目标构造当成BR所需步骤。

执行批固定为：

- 32个M32 entry：直接复用现有`mmlu.batch_inputs`的左padding、attention mask与显式`position_ids`，不重做token化。32组的M身份互异，每题每块出现一次。
- 128个QA B1 entry：原始完整`input_ids=prompt+gold[:-1]`，attention mask全1。Qasper32/Hotpot96各自只存一个entry，原32组中各曝光两次，重复曝光引用同一缓存。
- 每个optimizer更新仍由原组的1个M32＋8个QA组成，先M后原QA顺序，共9次真实块前后向；一个完整周期32更新。teacher与student流每层仅推进**160个唯一entry各一次**，不能因QA曝光两遍将它向前推进两层。

`RollingStreams.capture(teacher, student, entries: EntryPool, device=...)`对每个entry调用真实Qwen3Model，仅在首层pre-hook捕获原`[B,L,H]`和实际kwargs后用专用异常停止；接受并传入全部真实`input_ids/attention_mask/position_ids`，不丢M32位置。两模型分别捕获，验证kwargs字段和值一致后共享一份不可变CPU kwargs。`valid_mask`来自实际输入attention mask，不从FA2可能为None的block mask猜测。捕获包含实际`position_embeddings`、`cache_position`、FA2 mask；之后沿所有层传递，不人为重算RoPE。

只移动捕获所需embedding到执行GPU，根`rotary_emb`可留CPU FP32：现Qwen3 forward自行将inv_freq device-only搬到hidden设备，避免移动registered buffer后遗漏`original_inv_freq`别名。教师按现有独立HF加载器以BF16参数/FA2加载、保留FP32 RoPE，禁止对整模型调用`.bfloat16()`；移动设备不改dtype。原始教师、Tokenizer及源非投影身份沿现有HF分片SHA和147字段对照规则核实。

块损失包含所有**有效输入token**，没有causal shift或答案only mask；所有hidden维都参与。对于组`g`：

```text
T_i : 第i层原始教师输入；S_i : 第i层当前学生输入
Y_T = teacher_block_i(T_i, actual_kwargs)
Y_S = student_block_i(S_i, actual_kwargs)
L_g = Σ_本组9个entry Σ_有效token,h (Y_S.float − Y_T.float)²
      / (H × 本组有效input_token总数)
```

每个entry算FP32平方和，除同一个组分母后backward，9次累积后只更新一次。padding既不计误差也不计分母；不得平均9个独立batch均值，不沿E的M/QA各`.5`域权，不对层做额外能量归一化。域token占比和teacher能量只作读数。由于组有效token总数不同，一周期的优化是32个**组内token平均目标**依序更新，不能称为整个池统一token加权的一次梯度。整池前后读数则可汇总SSE/总有效token，明确与逐组优化归约的区别。

当前不新抽dev；训练池上的前后MSE明确标为训练拟合，不能称独立泛化。BR本身不依据这些读数挑层、挑步、跳块或提前停。完整下游由最终四端点按原九项协议判断。

## CPU双流与GPU驻留

每个entry持有CPU BF16 `hS/hT`、valid mask和共享只读kwargs。`prepare_layer(i, teacher_block)`按entry将`T_i`经真实teacher block推进并**替换**为`T_{i+1}`；原teacher旧entry立刻可释放。训练保持`S_i`与`T_{i+1}`不变。当前学生块提交BF16后，`commit_and_advance(i, student_block)`按entry执行真实学生块，将`S_i`替换为`S_{i+1}`；该接口核已提交状态，实际BF16提交与完整group8预热由调用者完成。两个流都detach，无跨层图、周期重置、teacher hidden替换student输入或共同从HF重启。

首候选用一张GPU：teacher当前块上卡，160个entry目标更新完成即回CPU；student当前块上卡训练，提交并推进后回CPU，再进入下一层。训练期间其他35层和CPU流不在GPU常驻；但下述BF16部署预热会短暂迁入完整native投影集合，必须计其峰值。无需model parallel、全模型teacher logits、KV cache或磁盘36层activation目录。不新建prefetch/守护进程；在实测证明搬运成为主要成本前不加异步双GPU流水。

**BF16推进的数值路径固定为正式完整native group8。** 只传本层六个native，即使参数写`group_size=8`，每类别仍仅一个模块，实际会落到单模块解码；此前已见它与完整group8融合路径有差异。本轮不将这种差异隐含到逐层学生流，也不凭单层MSE下降判为无害。

首版选择简单的完整集合预热：在初态BF16读数及每次提交之后，把全部216个native投影临时迁到同一执行GPU，全部BF16/冻结、恢复各自源deployment runtime flags，再将完整名称集合交给原canonical `prime_named_vae_linear_cache(..., group_size=8)`。保留当前块六个native的实际cache；其余native先清decoded/runtime cache再回CPU，避免把全部解码W复制到主机。当前compact gate按既有BF16消费者显式清缓存后重建；无需为了它迁入其它35个compact。记录实际完整名单、共同设备、canonical来源、预热/搬运耗时和峰值。

native分组signature除类别/shape还含device、decoder `use_checkpoint`、param_dtype等，因此局部训练结束必须将`use_checkpoint`等部署分组相关flags恢复到该模块源值，不能让已训练前缀与未训练后缀因runtime残留分成不同组。正式最后保存/fresh沿同一策略，保持元数据与部署分组一致。预热后`commit_and_advance()`只消费已准备的native cache，**不得沿旧`_run_native_block`先清cache**再逐篇执行，否则又退回单模块路径。compact缓存也不能被BF16提交前的live读数复用。

完整native预热的单卡资源已由本次相邻块短验确认，实际峰与适用范围见下方实测；完整36块/32更新质量尚未运行。暂不实现“找当前模块所在group8 peers”优化，因为它需要正确复现signature、排序及chunk边界；只有完整预热的实测成本确实影响选择时再做，并以原canonical完整集合为参考，不手工按层号凑8个peer。

本次只读CPU对实际`dataset.pt`的算术（CUDA未初始化）：

| 完整32组候选 | 数量 |
| --- | ---: |
| unique有效输入tokens | 708,845 |
| M32 padding后的持久token slots | 668,992 |
| QA唯一完整token slots | 344,311 |
| 160 entry合计token slots | 1,013,303 |
| 一条BF16 hidden流，H4096 | 8,300,978,176 bytes |
| 两条hidden流 | **15.4617767334 GiB** |
| 一份共享cos/sin BF16 RoPE，head_dim128 | 518,811,136 bytes |
| 每块32更新实际有效输入曝光 | **1,053,156 tokens** |

另计mask/position、模型CPU权重、当前entry的新旧输出和训练临时量；这不是进程RSS或GPU实测。若每层保存两流会制造约0.56TiB重复activation，首版不这样做。不得用旧类`_cpu_copy`给整个S/T池再复制一份；`BlockBatch`仅作张量引用视图。更新同一entry时先算出新输出再替换，保证失败时不把半个层标为已完成。

前后读数尽量复用必需前向：teacher目标生成与教师能量统计同遍；学生BF16提交后的MSE与`S_{i+1}`推进同遍。需要BF16初态MSE时只多一遍当前学生块，输出即时释放，不保留第三条完整流。报告各段实际耗时、CPU驻留与GPU allocated/reserved峰，而不把缓存理论字节数当实测。

## 局部训练、BF16提交与预算候选

当前块先冻结所有参数，只开启其七投影decoder。native沿canonical trainable packed plan，FP32 master、`packed_vq_decoder_linear=True`、decoder checkpoint和BF16 compute；compact沿`enable_decoder_training()`，保护buffer不cast。实际block forward直接产生BF16输出，FP32归约损失，不安装旧临时weight forward，也不把teacher或前缀放入梯度图。

首候选为每块fresh AdamW，LR **1e-6**，betas(.9,.95)、eps1e-8、WD0、一次joint clip1.5；块内沿`base_lr*(1-(step-1)/500)`的前32更新，无warmup，块间重建optimizer，不继承另一块的m/v。采用现有数值稳定的decoder学习率作有界起点，不声称原MSE梯度尺度与答案KL相同，也不声称最优。记录raw/clip梯度、各投影梯度和BF16更新存活；仅有明确异常或没有有效更新的证据才调整一个变量，不铺LR×数据×步数网格。

直接调用Transformer block不会触发Qwen3Model负责的整层gradient checkpoint。首版仅保持decoder自身checkpoint；若真实资源需要整块checkpoint，须显式用nonreentrant包装真实block forward，不伪称`model.gradient_checkpointing_enable()`已经作用于这个直接调用路径。当前块规模小于整模型，不为假想峰值先增加offload体系。

每块固定末步后释放gradient和optimizer，将活动decoder准确cast回BF16、全块冻结eval；native `disable_trainable_decode()`、恢复源部署flags、安装BF16 runtime，**先显式清两类cache，再完成上述完整native group8预热**，之后才做BF16读数/推进。不能只依赖compact `_version`缓存检查：`.data`转换不保证版本变化。推进完成、块搬CPU前再清native/compact运行缓存。不能直接套完整模型`deployment_mode(block,...)`，因为block没有其完整模型API。只有BF16提交后的真实学生输出能成为下一层输入，live FP32-master decode的末输出不能直接进入CPU学生流。

每块保留before-deployed、before-live及after-deployed读数范围，区分native可微解码与部署算子正常差异；不加数值补偿或将实际MSE差都归于浮点。某层after-deployed MSE暂升不是自动停止条件，固定配置仍按下游终态判断；NaN、梯度链缺失、冻结字段改变、source/目标/mask违反才停止依赖。

已据短验成本固定本批候选：完整H96周期、**32更新/块×36块=1152更新**，每块1,053,156有效input-token曝光，36块合37,913,616次层级token呈现；不是37.9M独立训练tokens，唯一文本池仍708,845有效tokens。本批样本/完整长度、更新数及LR已在正式入口固定；只有真实C0与实际资源就绪才执行，本文数值不是已经提交的作业。短验只测两个组和两个块，不直接将其总时间乘比例当正式耗时；成本还包括160-entry目标前向/初态读数/BF16推进、搬运、最终导出、后接E及两份完整九项。

## 必要机制短验

建议一份真实B→fixed2混合原型，选择**相邻块0→1**，每块两更新，输入为原groups `[2,10]`。这不是质量层选择，两个块用于覆盖“提交→推进→下一块”新路径；其余34层保持真实完整模型和冻结状态。短池含2个M32＋16个QA共18个entry，两流约1.810546875GiB；实际90,328有效input tokens，保留最长QA10,593与M32×958，不缩模型、截长文或造hidden。

必须真实发生：原HF teacher和压缩student独立首层capture；块0的两次MSE FWBW/累积/Adam，第二次使用非空状态；当前七投影native+compact梯度有限且更新真实存活；BF16提交并推进S0→S1，同时教师T0→T1；块1消费这份S1/T1，继续两更新并再次提交；原保护/码/路由/旋转/非decoder与teacher不变。第二块输入的来源是流实际产生值，不能构造假的中间层fixture或重新执行原未更新学生前缀。

源转换消费者差异单列，短验不将B native→compact功能差称BR收益。短验前后完整checkpoint保持216 native＋36 compact，只有这两块七投影decoder可变；用现有mixed保存，独立进程只加载该成品，核完整state、冻结补集、真实账本和同输入BF16输出。保存/加载证明服务新mixed逐块提交路径，不额外做本轮未要求的exact-resume。短验生成的权重在证据归并且无明确用途后按持续授权清理。

若上述mixed拓扑、mask/长度、局部生命周期与保存路径通过，真正C0出现时复用适用证据，只核真实C0来源与实际拓扑差异；UUID/值变化本身不触发整套重复短验。若正式配置改变关键形状/执行路径，才补变化部分。GPU分配由主线程启动前一次核实；短验可优先GPU6，GPU7不为未定正式任务占用。没有轮询、自动重试或预先挂入原序列的BR长任务。

## BR端点与后接E

正式完成全部36块后，模型全部是已提交BF16的既有native/compact格式；保存原拓扑、完整冻结证明、逐块实际更新数/顺序、数据身份、源C0、代码SHA和真实logical/packed/文件字节账本。独立fresh确认同一最终BR端点；没有dense、外置LoRA或新部署字段，因此位宽变化应只来自已记录格式元数据，而非未计补偿。若账本不符，先查实际状态，不用“decoderonly”口头推定通过。

当前`fresh_muon_cmcc_20261030`只接受原C0来源，不能将BR权重改名冒充C0。后继薄来源适配须核真实C0→36块BR链、完整mixed终态及冻结证据，再复用同一H96/Muon32训练数学：M1024一次混合目标可复用，QA原生OPD轨迹/全词表教师目标和κ从BR父fresh重算，原500线性前32与可训练范围原则保持。直接E与BR→E的在线轨迹不同，等呈现/更新规则不等于逐值等监督token，实际曝光分别记录。

四端点C0、E、BR、BR→E用同一完整九项及各自实际存储；C0/E已完成的同模型同协议结果复用，新增最多BR与BR→E两份完整九项。不能在只完成BR中点时因下游下降取消BR→E，也不能把E旧增益加到BR分数。若BR→E相对E有净收益且影响采用决定，再单列相近实际成本更长E-only对照；未配平前只称增加BR阶段的整配方收益。若固定预算无净收益，闭合本配置，不因局部loss下降自动加步，也不否定所有块重建。

## 10月31日真实短验结果

权威证据：[训练与导出报告](../../artifacts/block_reconstruction_cmcc_20261030/adjacent_short_v1/report.json)、[独立fresh审计](../../artifacts/block_reconstruction_cmcc_20261030/adjacent_short_v1/fresh_audit.json)、[数据CPU核验](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_cmcc_20261030/data_cpu/validation.json)、[清理记录](../../artifacts/block_reconstruction_cmcc_20261030/adjacent_short_v1/cleanup.json)。真实命令为已激活bitvae环境下的`python -B -u -m experiments.block_reconstruction_cmcc_20261030.run_short --output /data1/cmcc/VAELLM/artifacts/block_reconstruction_cmcc_20261030/adjacent_short_v1 --timeout-seconds 14400`，`CUDA_VISIBLE_DEVICES=6`，成功后同一固定shell接`fresh --directory`同目录。tmux `BR_adjacent_short_20261031`、worker386604/start11451158，于00:18:43启动；run、fresh和外层均exit0，进程已退出，GPU6已释放。源码审查发现的`layers`重复keyword在GPU前统一为`layer_reports`，没有带着该问题训练或重跑。

本次输入是B真实36gate转换后的216native＋36fixed2完整模型，标记`prototype_only`；仅L0→L1各两个更新，原groups2/10、18个唯一entry、每块90,328有效input-token曝光。各块7投影、48个D参数叶／148,224个FP32元素；四次更新均为七投影梯度有限非零、48叶无零梯度、Adam在块内step1→2连续，块间fresh。两块末步分别有15,924／18,429个D元素改变且保留到BF16；不是完整36块收敛证明。

| 验证／费用 | 实测 |
| --- | --- |
| CPU两流推进 | 18个entry原位存储保持；teacher前缀独立；L0学生BF16提交输出＝L1真实输入，teacher相邻输入输出、各目标及训练前学生输入SHA均吻合 |
| 冻结与独立加载 | 3675持久字段strict exact，3579非训练补集exact，teacher完整状态exact；同64token实际输入末位置全词表BF16 logits exact，FP32 RoPE保持 |
| 训练＋导出／fresh | 200.050秒／25.222秒 |
| 四个真实更新 | 4.719／4.538／4.949／4.344秒；joint raw grad .0468–.0678，均未触clip1.5 |
| 完整216native预热 | 每次2.332–3.738秒；保留当前块缓存，非当前模块清缓存后回CPU，未按单模块解码冒充group8 |
| 全流程GPU峰 | allocated57,280,009,216B（53.346GiB），reserved57,887,686,656B（53.912GiB）；包含完整预热及末点全模读出 |
| F/B更新段峰／CPU HWM | 10,717,351,424B（9.981GiB）／30,902,902,784B（28.781GiB），分别计量 |
| 部署账本 | logical2.4987414816151494／packed2.5121161419412363；全部组件、tensor bytes与转换父不增 |
| 成品tensor／目录bytes（清理前） | 4,671,007,680／4,674,989,350 |

相同有效token分母的BF16部署MSE：L0两个组由`0.0013229815 / 0.0017303012`变为`0.0013174404 / 0.0017232363`；L1由`0.0024495501 / 0.0031962273`变为`0.0024433559 / 0.0031882978`。这些局部小幅下降及更新存活只闭合实现，不证明九项或后接E有效，不由此加LR/延长步数。BF16推进同一前向同时产生after-deployed读数，未另外重跑一遍学生推进。

本批采纳的是隔离实现、混合拓扑及单卡完整预热方案。现有证据不要求改变首候选H96、LR1e-6、32更新/块或再做B网格。4步速度折算1152步约83–95分钟仅为纯更新估计，另计全160-entry教师目标/读数/推进、搬运、导出；正式BR暂预留单卡2–3小时工程窗口，E及两份完整九项另计，不作交付时刻承诺。完整池两hidden约15.46GiB，比short多约13.65GiB，CPU总峰约43GiB只是粗估，未实测正式池。预热估计仅占全36层约3–4.5分钟，当前没有做peer抽取优化的收益依据。

薄36层driver、BR→E来源适配及本批固定预算已完成并CPU核验；下一步仍需真实完整C0和启动资源。相同混合拓扑/关键形状/生命周期复用本证据，只对实际变化补验证；不因换UUID原样重测。完整36层的跨层误差、BR中点及BR→E完整九项尚无结果，不能把本例升级为算法收益或跨模型结论。

短验转换父和末点均无后继用途，已仅删除两份`pytorch_model.bin`：各4,672,277,077逻辑字节，合计实际释放9,344,565,248分配字节。17份metadata/config/报告/字段哈希/fixture/命令/核心日志SHA保持，原B、HF和训练数据保留。`source_baseline`与`final_model`现为证据目录，不能再作为可加载模型。工程经验见[块重建的数值与衔接验证](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)。

## 最小实现职责与未定事项

| 隔离文件 | 职责及复用边界 |
| --- | --- |
| `bindings.py` | N namespace边界、真实C0/HF/H96来源；B→真实fixed2机制父标记；实际七投影decoder和冻结补集。共享源码/pins只读 |
| `data.py` | 验证现有H96输入，构造32M32＋128QA真实entry和原32组引用，记录mask/有效/padded token；不读无用logit目标或发明新采样 |
| `rolling.py` | thin真实首层capture、kwargs等值共享、160-entry CPU两流原位推进；提供`prepare_layer(i, teacher_block)`、`group(index)`、`commit_and_advance(i, student_block)`，检查顺序与entry只推进一次 |
| `runtime.py` | 同设备完整native group8预热、当前块缓存驻留、FP32 D训练开关／BF16提交、源deployment flags恢复、保RoPE alias的device-only搬运 |
| `fit.py` | 当前层七投影FP32 master/BF16 compute，9-entry有效token SSE归约、Adam与冻结审计；不调用旧FP32 weight graph |
| `run_short.py` | 真实mixed原型的0→1、每块2更新、实际峰/时长/双流接续与终态；没有正式36层选项暗门 |
| `export.py` / `fresh.py` | 复用mixed serializer及官方读出，核完整BF16状态、冻结、来源/位宽；新输出不覆盖已有成品 |
| 后续正式入口与BR→E source adapter | 短验后按一次定稿配置执行36块，明确BR provenance；后接已定同E规则，尚未实现/运行即不写已通过 |

输入/双流、局部生命周期、来源绑定、CPU检查及必要GPU短验均已完成。正式样本量/完整长度和每块步数已据成本固定，BR→E来源适配已实现；实际GPU时段及完整长程费用待真正运行记录。Astra/ultra负责设计与解释，sol/xhigh负责对应实现和验证，sol/high负责既定流程结果归并；主线程统一维护执行计划/lessons/导航。当前无新增恢复效果经验。

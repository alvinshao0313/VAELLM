# 同BF16 up三档profile有限批次

用户已批准本批总硬上限 **2.75 GPUh**，仅原 BF16 Qwen3-8B 的 `up_proj` 三档 profile，无完整八项、KD、第四码率或自动20槽。原 `cat_auto_protocol_cmcc_20261104` 源码、fit/code/consumer identity 完全冻结；本包只有具体请求生成和有限进程链，不适配或改写原入口。

共同来源为 revision `b968826d9c46dd6066d109eabc6255188de91218`，实际 `model_inventory_cpu.json` 与 `full_data_v0/data_plan.pt`（data identity `e675f901e68f0f56bc3ad1a641db52e15b4c14ba4f62bb15255f0db1f5d828bb`）。三档均 D32/S2/M36/B8192、E/D128/R0 asymmetric/layer/swish、BF16原AdamW .003/(.9,.95)/eps1e-8/WD0、linear10000/warmup0、TFalse/block32/inputINT8实际64列；semantic search seed31。每个正式stage完整10000更新，不按早期KL停。

| GPU | 配方 | 固定有限链 | 硬上限 |
| --- | --- | --- | --- |
| 1 | r2/K32 | 复用已完成K32真实技术receipt，直接fresh BF16正式 | 45分钟 |
| 3 | r2.5/K40 | 新latent/5byte技术2 → actual exit/fresh通过 → 新BF16 fresh正式 | 技术15＋正式45分钟，链总≤60 |
| 4 | r3/K48 | 新latent/6byte技术2 → actual exit/fresh通过 → 新BF16 fresh正式 | 技术15＋正式45分钟，链总≤60 |

两项技术测试互不等待，在不同卡同时开始；正式K32同时启动。技术模式仍为每stage原10000日程前2步、实际512×256校准及固定512 proposal，不缩M/B/数据。K40/K48正式registry只在本K真实完成receipt出现后生成，禁止借K32证据或复用技术权重。每次worker都由原入口完整执行fit/编码/packed bank/教师KL与逐层真实输入读出/native保存/实际partial账本/独立fresh；单链失败立即停止本链依赖，不重试或续期，不牵连其它独立链。

原入口的host策略已允许GPU1/3/4，所以直接用这些实际空闲卡，没有AST/FunctionType/假卡号。协调进程显式隐藏CUDA，实际worker复制调用者完整环境并设置其真实GPU；每stage一个自建PG由有限父进程按绝对deadline wait/kill/reap，无守护或额外占卡。CPU报告只核真实来源、K32正式证据、两新K必要技术请求及错误证据拒绝，不重跑成熟fit/planner。

启动者先激活实际 `bitvae`，再用tmux并显式继承 `CONDA_DEFAULT_ENV/CONDA_PREFIX/PATH`、绝对bitvae Python、allocator `expandable_segments:True`。准备入口 `python -m experiments.cat_up_rate_profiles_cmcc_20261104.prepare --batch-root <new-batch> --report <new-cpu-report>`；每卡入口 `python -m experiments.cat_up_rate_profiles_cmcc_20261104.run --label k32|k40|k48 --batch-root <same-new-batch>`。固定card/recipe在源码内，CLI不允许加第四rate或其它阶段。

批根 `batch.json` 是不可变来源／预算描述；每卡 `chain.json` 记录真实控制进程、各phase PID/命令/请求SHA/原始截止/退出。phase目录保存原worker日志、progress、原指标、真实packed bank/raw/fresh和exit；源码/registry与实际完成状态分开，不伪造GPU或已成证据。profile只生成本类费用及固定train读出，不将partial其余dense当压缩完成，也不以nominal r代替完整费用。

CPU来源／registry／反例／语法已实际exit0（7.068251秒，CUDA false），报告位于[result/prepared.json](../../result/qwen3_8b_mmlu65_20261004/cat_up_rate_profiles_cmcc_20261104/prepared.json)。Astra对最终有限链窄审无阻断；原core源码全部保持原pilot SHA。三tmux于15:40:20同时启动，批根为[batch_20261104_1538](../../artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538)，不可变`batch.json`及每卡真实registry/source/PID/exit都在该根。

| 链 | tmux | 控制PID | 初始worker PID | 原始截止 |
| --- | --- | ---: | ---: | --- |
| K32/GPU1 | `UpProfiles_K32_g1_20261104_1540` | 2678526 | 正式2678678 | 16:25:20 |
| K40/GPU3 | `UpProfiles_K40_g3_20261104_1540` | 2678530 | 技术2678679 | 技术15:55:26；全链16:40:20 |
| K48/GPU4 | `UpProfiles_K48_g4_20261104_1540` | 2678534 | 技术2678681 | 技术15:55:26；全链16:40:20 |

11月4日17:20终态核证：三条chain均`failed`、正式worker均`exit_code=-9`、外层退出1，原controller／worker已退出。K40／K48技术链各自完整exit0后，已独立重载BF16进入正式；三条正式都在原45分钟截止终止，未执行拟议handoff、扩额、续期或重启，**原批2.75 GPUh上限保持**。完整核证、原SHA、逐phase费用与收尾合并在单个[closure receipt](../../artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538/closure_20261104.json)。

| 链 | 技术phase实耗秒／退出 | 正式phase实耗秒／退出 | 原正式截止 | 最后落盘训练状态 |
| --- | ---: | ---: | --- | --- |
| K32/GPU1 | 复用原PUBLIC pilot，不计本批 | 2697.798474／−9 | 16:25:20 | stage1 10000；stage2 9000 |
| K40/GPU3 | 432.061242／0 | 2703.926397／−9 | 16:32:39 | stage1 10000；stage2 9000 |
| K48/GPU4 | 426.816253／0 | 2704.359332／−9 | 16:32:33 | stage1 10000；stage2 9000 |

phase实耗按每条worker启动至回收计算，含超时后的kill/reap开销，合8964.961699秒／2.490267139 GPUh；若连CPU协调器初始化和phase间隙一起保守计费，三链合8983.970000秒／2.495547222 GPUh。两口径均低于原上限；不是GPU利用率积分，也不并入旧PUBLIC pilot或本次CPU核证。正式链未走到proxy／local／save／fresh，不能虚构这些阶段费用。首两实步LR .003/.0029997、BF16 Adam state1/2及非零位移仍由原worker日志保留。

三条正式都没有完整`fit_report.json`、packed bank、raw、`completed.json`或encoder／optimizer／RNG checkpoint；stage1内存状态不等于可复用checkpoint，stage2确切被终止步数未知。**本批没有完整正式profile，不能比较三档质量，也不是质量负例。** 17:20时曾将K40／K48技术载荷与原PUBLIC pilot保留供新结果选择器／来源适配CPU核证；用途结束后的技术载荷清理见下文。两步技术结果不能替代正式初态或正式结果。后续仍由root／Astra依本批结果与预算讨论决定，不自动扩类别、full8/KD或整链。

本批失败formal无生成大载荷可删。已将准备日志和四个准备／外层exit散件的精确路径、内容、SHA与退出证据归并至上述closure，删除5件共565字节；原batch／请求／chain／worker核心日志／progress／phase process与exit／技术fit、原指标、metadata、源码全部保留。K32 process仍被现有CPU来源记录引用，未动；17:20时技术K40／K48共约28.13 GB的载荷及原PUBLIC pilot因上述具体用途保留；K40／K48后续清理与PUBLIC继续保护的区别见下文。未改冻结源码、在途任务、配置或预测。

只读性能诊断：K32多个1000步窗口为.1219/.1235/.1254/.1293/.1279/.1284/.1282/.1281/.1283秒/步，属于稳态约.125–.130；stage2同样约.129秒/步。当前适配固定`gpu_resident_data=False`，每步由`num_workers=0/pin_memory=False`的TensorDataset构造8192行并转GPU；旧同M36/B8192/D32K32 up运行实际开启`gpu_resident_data=True`，日志约.037–.038秒/步。两者AutoEncoder/ParallelLinear/DataLoader实现SHA相同，但旧父／优化器／完整pipeline来源有差异，不将比例作为严格归因。新审计每步有一次gradfinite标量同步，参数clone仅首2/末步；BSQ observer只stage1的5000/10000，不是每步CPU参数复制。只读证据不能把数据构造与同步的秒数单独分摊。

同源AutoEncoder的BF16 MSE×M日志在旧成熟up里也多次出现recon13.125（1000/5000/10000），不能据此认定新实现损坏；没有同批FP32读出，无法由该日志分辨细小损失变化与实际平台。按本批实测训练速度加完整前后处理，45分钟约不足以容纳完整链；期间未改任何在途参数、源文件、驻留flag或截止。


**技术载荷清理（2026-11-04T18:02:29+08:00）**

旧K40／K48两步技术链已结束，当前CPU来源／最新SC-CAT选择边界核证已完成（`cat_offline_execution_cmcc_20261104/current/`与`s8_selection_20261104_1717/`）；实际CPU worker适配读PUBLIC K32 pilot。新三链proposal／CPU／未来run只使用原BF16、inventory／DataPlan、PUBLIC K32 request／completed、旧closure及K32旧formal registry；不读取旧K40／K48 raw／bank，且各自必须重新取得本人新来源技术receipt。源码／脚本／当前结果无旧payload调用，旧controller／worker不存在，精确18个文件无活动打开FD。旧handoff草稿仅指已失败formal，不能构成技术载荷后继用途。

清理状态：`deleted_and_verified`。删除理由统一为：两步技术验证及本轮CPU来源核证用途结束，无已安排续训／复评／交付用途的生成权重、runtime及tokenizer副本。仅本表精确文件进入删除范围，不跟随符号链接。原checkpoint_meta.json／config.json、bank.json、result.json、fit_report.json、request／completed、corelog／progress／process／exit和batch／chain全部保留，用于复现配置、原指标、实际SHA及退出溯源。

表中P为本批解析后绝对根 `/data1/cmcc/VAELLM/artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538`；`P/`与各行后缀拼接为删除前核验的唯一绝对路径。

| P下精确文件后缀 | 文件字节 | 独占分配字节 |
| --- | ---: | ---: |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/pytorch_model.bin` | 13345625018 | 13345632256 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/cat_runtime_state.pt` | 3616410 | 3616768 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/tokenizer.json` | 11422654 | 11423744 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/tokenizer_config.json` | 9762 | 12288 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/vocab.json` | 2776833 | 2777088 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/merges.txt` | 1671853 | 1675264 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/added_tokens.json` | 707 | 4096 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/raw_model/special_tokens_map.json` | 613 | 4096 |
| `k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/category_bank/native_bank.pt` | 587851946 | 587857920 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/pytorch_model.bin` | 13457249210 | 13457256448 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/cat_runtime_state.pt` | 3616410 | 3616768 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/tokenizer.json` | 11422654 | 11423744 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/tokenizer_config.json` | 9762 | 12288 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/vocab.json` | 2776833 | 2777088 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/merges.txt` | 1671853 | 1675264 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/added_tokens.json` | 707 | 4096 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/raw_model/special_tokens_map.json` | 613 | 4096 |
| `k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/category_bank/native_bank.pt` | 699476138 | 699482112 |

合计18文件，逻辑28129199976字节（26.197358944 GiB），独占分配28129255424字节（26.197410583 GiB）。分配口径仅来自本表文件的st_blocks×512、nlink=1及无打开FD，不用共享磁盘空闲量差计算。

原实际SHA仅复用保留证据，不为删除重哈希大权重：

- k40：bank文件SHA256 `6c105afcd38c3631542ef3c5569aef4bd30ddfef0d856536c4c3eadbaedeb426`，来源`k40/technical_validation/a0137274be4e49f4498ba2e980f80c2ddf96057c664b0d4ab85589057202ab46/category_bank/bank.json:state_sha256`；raw权重原`fresh_evidence.weight_digest=e6329179057f6fe793b7bc4797fdf909bda0844981c298266f0a3479effffea3`、UUID `013d0dc3-36b6-4dc5-820f-df36193b79da`，来源同candidate的`result.json`及原`completed.json`，原field名与证据强度保持。
- k48：bank文件SHA256 `30684c662e8af9bebd4c157188fdcacffb00bc2cb72ba7434454fcba99ebc2a5`，来源`k48/technical_validation/98589283f3831b99e2bb9b83197063ca76161485e3ed2c41321153ecccce5ada/category_bank/bank.json:state_sha256`；raw权重原`fresh_evidence.weight_digest=1c71732ed4354b94169ad9abd76db9772fdddbc48aad87d9230244e3685d45c9`、UUID `6dd0e77f-96a3-4a48-82b1-ecb26a2ec9d9`，来源同candidate的`result.json`及原`completed.json`，原field名与证据强度保持。

受保护旧closure SHA256仍为`1b561819bd99303e84a50133f6a9a3b0a57d2ccaaaaaad4826d51485bd42f8e3`；其17:20保留用途是历史快照，本次只在本README归并后续清理，不改closure字节。PUBLIC K32 pilot保留供新三链来源／模型路径receipt及现有CPU入口；原BF16、inventory／DataPlan、K32静态registry、新proposal和相关源码均未更改。无新增算法经验，无GPU／安装／新监控。

删除后核证：表中18个精确文件全部不存在，保留最小证据及全部已冻结新proposal的SHA未变；本README全部相对链接目标存在。实际释放按上述独占分配字节计，原565字节散件收尾单列、不重复计入。

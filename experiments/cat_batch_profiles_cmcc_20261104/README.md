# 批量gather后的同BF16 up三档并行profile

**状态：三链技术、fresh BF16正式S2×10000、保存／独立fresh及外层全部exit0；三份正式CPU结果适配也全部exit0。** 本批只完成up三档profile，没有full8或质量采用结论。 实际授权绑定下述唯一proposal／root／卡及预算；本包prepare只生成未批准proposal，授权由root依据实际用户答复另存，原proposal保持不变。技术／正式／完整profile的完成状态以真实receipt及退出为准。原批已超时、保守实耗2.495547222163942 GPUh独立保留，以原closure SHA追溯，不清零、不挪为新额度，也不自动使用旧扩额建议。

固定原BF16 Qwen3-8B、原`model_inventory_cpu.json`／`full_data_v0/data_plan.pt`及原三recipe／semantic seed31；唯一拟合改动为路由至`cat_batch_gather_cmcc_20261104.search`与source-bound新fit identity（原7份＋fit/data2份，共9份）。原core、helper及旧profile包均不改。每份新request同时绑定原core全部Python、新helper全部Python、本包全部Python及实际依赖来源。

| label／GPU | 固定配方 | 本人独立有限链 | 硬上限 |
| --- | --- | --- | --- |
| k32／1 | r2、D32/K32/S2/M36/B8192 | 新入口技术2步 → 本人真实complete／fresh → fresh BF16正式各10000步 | 技术≤900秒，正式≤3600秒，整链≤4500秒 |
| k40／3 | r2.5、D32/K40/S2/M36/B8192 | 同上 | 同上 |
| k48／4 | r3、D32/K48/S2/M36/B8192 | 同上 | 同上 |

三链无跨卡屏障。每条自自身实际启动起固定4500秒绝对deadline，各phase取其900／3600秒与链剩余的较小者；任何失败或unknown receipt只停止本人链，不重试、不续期、不占卡。技术仍是原linear10000前2更新，保留实际模型／全类M36／B8192、calibration512×256及proposal512；正式另起进程重载原BF16，绝不复用技术权重或超时内存。K32也必须取得新来源身份的本人技术receipt，旧70pins GPU证明不能直接接本入口正式。60分钟正式上限参考旧CPU慢路径完整链约48–51分钟加余量，不根据CPU gather倍数承诺GPU速度。

prepare检查真实BF16／tokenizer文件SHA、原inventory与DataPlan、真实CPU calibration／proposal状态形状、三个完整recipe及新身份、预算算术、授权／错误来源／原技术receipt拒绝、语法与实际CLI；它只写`proposal.json`和CPU报告。proposal无授权开始／未来deadline、无request schema、无approved授权。source helper的loader／tiny fit／routing CPU证据不等于8B新GPU技术链通过。

调用者激活bitvae后，用tmux后台准备；输出根和报告必须全新：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u \
  -m experiments.cat_batch_profiles_cmcc_20261104.prepare \
  --batch-root NEW_UNAPPROVED_BATCH_ROOT --report NEW_CPU_REPORT
```

待用户批准后，root才可提供独立JSON授权：顶层`user_approved`必须为布尔true，`user_authorization`必须包含实际批准reference，并精确匹配proposal实际SHA256、proposal_identity、绝对batch_root、完整labels／cards、3.75 GPUh、900／3600／4500秒限制。没有该授权时run先拒绝，不创建chain或request。本包不提供生成approved授权的CLI；CPU反例没有创建有效授权文件。

经批准后的具体入口是`python -B -u -m experiments.cat_batch_profiles_cmcc_20261104.run --label k32|k40|k48 --batch-root ACTUAL_APPROVED_ROOT --authorization ACTUAL_USER_AUTHORIZATION`，调用者显式设置对应真实`CUDA_VISIBLE_DEVICES`。运行时重新核所有来源、授权和指定卡当时无既有compute进程；独占创建label目录，已有目录即拒绝再次运行，同一proposal／授权不能复制到另一root重跑。formal registry仅在本批本人新技术2步、全部source／UUID／实际bank／独立state与BF16 forward fresh证据通过后生成。

有限wait／PG kill/reap复用既有固定链做法，stdout/stderr落盘，stdin断开；每phase保存真实request／SHA、命令、PID、绝对deadline与退出。协调器隐藏CUDA，实际worker继承bitvae环境并使用真实GPU、新helper入口、`ACCELERATE_USE_CPU=false`与原allocator。没有新守护、跨卡调度、自动重试或占卡。

本批不含full8、KD、其它类别／第四rate或自动搜索。实际GPU技术2步／正式S2×10K与完整保存／fresh现已完成，真实终态及费用见下文；full8仍未执行。

实际最终[CPU报告](../../result/qwen3_8b_mmlu65_20261004/cat_batch_profiles_cmcc_20261104/prepared_cpu_final.json)为21.156262875秒／exit0、CUDA未初始化，所有BF16／tokenizer源文件均重hash；源码pins、真实数据形状、预算／身份／参数拒绝与CLI结果随报告保存。[不可变proposal](../../artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/proposal.json)的SHA256为`45bded322867df7e7dcdb99f7c2f507cdcc0a3089725672a82fa5702c44dc5ff`。Astra已窄审通过准备范围；该CPU运行时没有实际request／授权／worker，之后的实际启动另记如下。

本包不依赖旧K40／K48技术raw／bank权重；保留原BF16、DataPlan、PUBLIC K32原request／completed及旧closure的真实来源绑定即可。正式新GPU证明必须本批本人重新产生。

## 18:06已授权并行启动

用户对`call_2fc8aed4b0ac4bdeb84aaf7e5ba286dd`第0项明确答复“执行三条并行链（推荐）”。实际授权为同batch根下`user_authorization.json`，SHA256=`6f7e77936f79fc20fef063044d8a2828736ae4f0e7b82d55281d3942cd535a76`；proposal SHA保持上文值，三臂来源pins不再修改。

| label／GPU | tmux | controller／初始technical worker | 绝对整链截止（北京时间） |
| --- | --- | --- | --- |
| k32／1 | `Cat_gather_k32_g1_20261104_1806` | 2779395／2779705 | 19:21:19 |
| k40／3 | `Cat_gather_k40_g3_20261104_1806` | 2779398／2779706 | 19:21:19 |
| k48／4 | `Cat_gather_k48_g4_20261104_1806` | 2779401／2779704 | 19:21:19 |

启动前`nvidia-smi`确认1／3／4均无compute进程，GPU0原LB继续，6／7不使用。调用者已激活bitvae，tmux显式继承CONDA_DEFAULT_ENV／CONDA_PREFIX／PATH，实际命令为上述run入口、对应label／卡、唯一batch-root及实际authorization；OMP／MKL各4、OPENBLAS1。控制stdout／stderr为`<label>.controller.log`，真实shell退出为`<label>.controller.exit`，各臂`chain.json`记录阶段／PID／精确命令／deadline，真实worker日志为`<label>/technical_validation/worker.log`及后续`<label>/formal/worker.log`。stdin非交互，断开SSH不停止tmux。

每卡自行接续，技术未通过不会生成正式请求；不做已失败链的自动重试。三链均已建实际技术request／worker；有限核证确认三端真实教师计算分别37.5383／37.0612／37.3884秒完成、各GPU峰allocated22,613,389,312B，均进入本人fit／encode且真实进程持续计算。三份技术fresh与worker均已完成并正常退出；正式从各自新BF16进程启动，完成状态另记如下。

## 三技术终态与正式接续

三技术均完成真实S2各2更新／原10K日程、M36／B8192、完整512教师／两半local、native保存及独立fresh；完整持久state与同输入BF16 logits均exact，真实GPU证据资格通过。各`technical_validation/completed.json`、`exit.json`和`chain.json`为原件。实际worker phase耗时包含启动至退出，区别于内部progress分段计时；本批实耗继续按各phase和chain最终实测累计，不把短验费用清零。

| label | technical worker phase秒 | formal worker PID | formal截止（北京时间） |
| --- | ---: | ---: | --- |
| k32 | 392.567380 | 2784592 | 19:13:11 |
| k40 | 392.533999 | 2784609 | 19:13:12 |
| k48 | 411.042931 | 2784956 | 19:13:30 |

三份正式request均为`technical_steps=null`、完整S2各10000及自身新来源证据，原始BF16 fresh，未续用短验权重。原controller与tmux保持，正式日志／进度在各`formal/`。拟合、proxy、local、保存／fresh全部完成后才有正式profile；本批无八项或恢复训练预算。

正式主循环已核：三端第一阶均有真实step1／2更新及1000/10000日志，原LR .003／.0029997、Adam state步数保持。第一段1000更新的日志步时K32=.0372秒、K40=.0406秒、K48=.0442秒；旧K32同位置为.1219秒，新旧K32迭代耗时比3.276882。仅比较同配置日志窗口，不把它当全链或最终提速保证，也不从loss推断完整八项质量。真实log保留在各自formal目录，无额外运行或网格。


## 三档正式profile终态

11月4日18:37核证三链chain／completed／worker exit／外层退出齐全且全为0，三controller及formal worker已退出，均在原formal硬截止前自然完成。三份唯一formal/profile_result.json由现有results CLI在bitvae／CUDA隐藏tmux生成，显式传实际--completed、--candidate-id、--request、--request-sha256、--output，默认require_formal=True；没有使用technical-inspection。每臂profile_result.exit=0，完整命令及CPU日志在同目录profile_result.log。

| label | technical worker秒 | formal worker秒 | 整链秒 | formal结束（北京时间） | 正式UUID |
| --- | ---: | ---: | ---: | --- | --- |
| k32 | 392.567380 | 1152.259043 | 1576.775698 | 18:32:36 | `a7c00bde-f62c-43df-9c71-2354994049c4` |
| k40 | 392.533999 | 1182.915703 | 1608.954126 | 18:33:08 | `3f40c86f-9c2c-4137-be82-b3689f8004c6` |
| k48 | 411.042931 | 1238.636287 | 1682.459675 | 18:34:22 | `ca83eb1b-613f-443a-bb81-71c8bd829951` |

worker phase启动至退出合4769.955343秒／1.324987595 GPUh；含CPU协调初始化／phase间隙保守计为4868.189498秒／1.352274861 GPUh，低于新增3.75 GPUh。每phase一GPU，不除以并行卡数；旧超时批2.495547222 GPUh独立，不抵扣。以下正式内部stage_accounting是原分段计时，不与worker墙钟重复相加。

| label | source/data | pack | teacher | fit/encode | proxy | local | save | fresh | 峰allocated B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k32 | 16.261694 | 0.458838 | 37.703552 | 937.762831 | 37.333620 | 61.561688 | 26.644135 | 26.163479 | 23564461568 |
| k40 | 16.233323 | 0.474009 | 37.459828 | 966.013907 | 39.564358 | 61.124921 | 26.412934 | 26.852015 | 23792281088 |
| k48 | 16.270632 | 0.433776 | 37.506540 | 1020.468727 | 38.220889 | 61.698778 | 27.984113 | 27.558393 | 24020100608 |

三端同原BF16父、同DataPlan `e675f901e68f0f56bc3ad1a641db52e15b4c14ba4f62bb15255f0db1f5d828bb`、semantic search seed31、相同source9身份与PUBLIC consumer。正式mapper核原request／process绑定及当前全部source pins；S2各10000、冻结补集、36投影、同顺序512教师／proposal、每投影两半256同父真实输入local、831持久字段完整state及同输入BF16 logits独立fresh均齐全。原配方／seed／初始化摘要／费用组件及actualSHA由原completed与正式profile_result保存。

| label | nominal主码 | 本类logical bits | 本类packed bits | 本类logical/packed bpw | 本类tensor B | pooled512 KL | pooled local SSE/energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k32 | 2 | 3808548864 | 3808548864 | 2.101918538411/2.101918538411 | 476068608 | 0.195003572762 | 0.025729260469 |
| k40 | 5/2 | 4701542400 | 4701542400 | 2.594757080078/2.594757080078 | 587692800 | 0.135769527570 | 0.018387844558 |
| k48 | 3 | 5594535936 | 5594535936 | 3.087595621745/3.087595621745 | 699316992 | 0.102676854467 | 0.014049429899 |

本类分母N=1811939328，logical／packed均计实际code、decoder、保护INT8权重／indices／scales和rotation等部署组件，36份投影费用与本类组件／总费精确闭合。pooled local由36投影×两半的原SSE求和／target energy求和，不直接相加各层NMSE；KL／local仅固定train读出，不能代替完整八项质量。

| label | partial全252 logical/packed bpw | partial tensor B | raw权重文件 B | 原完整目录 B |
| --- | ---: | ---: | ---: | ---: |
| k32 | 12.374413531760/12.374413531760 | 13233660672 | 13234000634 | 13253676949 |
| k40 | 12.502980107846/12.502980107846 | 13345284864 | 13345625018 | 13365301333 |
| k48 | 12.631546683933/12.631546683933 | 13456909056 | 13457249210 | 13476925525 |

partial分母全252投影N=6945767424，其余216投影仍dense，不能算整模压缩完成或套用2.6bpw终态预算；tensor费用和包含metadata／tokenizer／序列化的目录／文件大小分列。

| label／stage1实际步 | commitment ratio/cosine | entropy ratio/cosine | extreme fraction |
| --- | ---: | ---: | ---: |
| k32/5000 | 0.027726476/-0.481254144 | 0.007127221/-0.432777355 | 0.0 |
| k32/10000 | 0.041021150/-0.708954423 | 0.012921338/-0.552893936 | 0.0 |
| k40/5000 | 0.016446080/-0.556614040 | 0.007524038/-0.465073259 | 0.0 |
| k40/10000 | 0.024370714/-0.818046347 | 0.011119548/-0.685013882 | 0.0 |
| k48/5000 | 0.012997599/-0.539199918 | 0.008017688/-0.458582161 | 0.0 |
| k48/10000 | 0.018736722/-0.795373771 | 0.011540327/-0.686638069 | 0.0 |

上表均为真实stage1 5000／10000 batch，gradient_accumulation_unchanged=true；不从本表自动派生BSQ调参规则。

正式结果入口：

- [k32 profile_result](../../artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/k32/formal/profile_result.json)：candidate `0a33176f35347018b1c0b0d6e3516354ffd8af74c3d90540c8d19d0f92eb4502`；本结果SHA256 `db1429fa593850888bf416d9367f3101674f84bfd86ac864315129bfc08d075c`。
- [k40 profile_result](../../artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/k40/formal/profile_result.json)：candidate `da91a8b53cf10a098659df328e140dc9505d983703ce941363b0528c8a092cf2`；本结果SHA256 `7264aed2ad4f87b076c05a2217fe8e101b44c84aae4b69f8c509935625db02c1`。
- [k48 profile_result](../../artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/k48/formal/profile_result.json)：candidate `fa99a58f385185fa910d760e91992a9db5dabcf11af48a1f70f960748e514131`；本结果SHA256 `88b7c6c6a34f443a113cba56066d0d16e68fc2ac3bfe8e76d70341b75aa7fd69`。

正式raw／bank均明确保留供下一批完整八项评测；本批没有full8／KD／恢复训练，也不自动启动后继或按train读出采用配置。三份技术2步payload在各自formal完整结束、确认无消费者后已清27文件，独占分配实释41,859,022,848B（38.984252930GiB）；91份小记录SHA及18个正式载荷stat不变，原技术request／completed／指标／配置／日志／退出与SHA保留。[精确清理记录](../../artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/technical_cleanup_20261104.json)另记边界，正式raw／bank保留给下方八项。不可变proposal、授权及旧closure字节全部保持。

<a id="下一批完整八项讨论卡"></a>

## 完整八项确认与终态

本批已单独获用户明确批准并完成：GPU1／3／4分别对上述K32／40／48真实formal UUID独立并行完整八项，各MMLU57科／14042题B64＋seven／10700题B32，共64叶24742条原始samples；同原BF16协议（FA2、BF16、FP32RoPE、max_length40960、BOSfalse、0-shot、无limit），复用已核参考70.34649275081789。BF16原参考未保存逐题sample，不声称逐题exact配对。三full8各自也是有界的首次PUBLIC FA2／group8／HFLM运行验证，不另跑重复子集、不等K48先行，也不重训／重评BF16／joint／LongBench。

实际授权每臂两分片共享900秒、合新增≤0.75GPUh；依据同机成熟完整八项外层约501–534.389秒，900秒约见过上限的1.68倍，是紧上限而非完成保证。各臂失败只记技术失败／无质量结论，不重试、不扩额、不停其它已启动臂；全部完整result及reader闭合后才比较。原拟合3.75GPUh与其实耗独立，不挪用剩额度。

[唯一未批准proposal](../../artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837/proposal.json)，SHA256=`44b8e7656990affdff2c1855612ff3589af56bd1c8a295b386dc001f3aad7706`，直接由三canonical profile_result及现有`evaluate.make_plan`生成。CPU准备6.616852秒／exit0／CUDAfalse，三计划authorization=None且实际启动guard拒绝；只有proposal，无有效授权、future deadline、plan.json或GPU运行目录。sol/xhigh只读核三原件／UUID／source9／72份代码pins／CLI／同协议及计数，Astra审核三卡并行和预算边界，均无阻断。用户随后明确选择三卡并行，实际授权／plan与启动时窗已冻结，三份完整八项及canonical关联均exit0。S8 K32／40／48为68.982639511619／68.680015218578／68.965810744804，BF16为70.346492750818；KL降低没有对应S8单调提升，M改善伴随RTE下降，不由此认定最终码率。

使用现有`python -m experiments.cat_offline_execution_cmcc_20261104.evaluate --component run --plan ACTUAL_PLAN`；CPU来源、plan.prepare、run及完整结果回接命令维护于[既有评测入口](../cat_offline_execution_cmcc_20261104/README.md#使用实际收据)，不另建调度／重试或probe体系。新认识归[真实profile费用](../../docs/experience/lessons/compression_choices.md#固定d的码率profile与真实费用)及[数据组织与实测吞吐](../../docs/experience/lessons/memory_precision.md#合批加速与缓存精度的数值对照)。

完整八项原件与实际成本唯一维护在[评测记录](../../artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837/README.md)。eval记录窗口.440718186GPUh不含record前verify_plan；按实际授权起点至shell写退出码给完整保守上界.480943702GPUh，低于.75，原字段／数值及其scope分开保留。正式raw／bank仍保留供后续联合分配。

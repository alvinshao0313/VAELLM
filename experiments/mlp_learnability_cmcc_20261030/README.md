# 全 MLP 固定码 decoder 与 free-dense 的可学性桥接

2026-10-30 设计与两卡原型实测。本文记录一个隔离对照的设计、实现边界和已验证范围；唯一当前安排仍在[执行计划](../../docs/experience/records/w25_transfer/current_plan.md#mmlu缺口的阶段定位与训练可学性)。设计审查、CPU验证和两卡两更新均已完成，未执行质量长训或下游评测；具体完成范围以原报告和下方实测小节为准。

11月1日的新C0接续已转入[固定128更新对照](../mlp_learnability_formal_cmcc_20261101/README.md)。本文的B短验、旧资源窗口和当时待定预算保留为原型记录；当前资源仅GPU0–5，正式预算和实时状态以执行计划及新包原始记录为准。

问题是：从**同一完整压缩部署函数**出发，在相同整模型教师目标和训练输入下，开放全部36层MLP非保护权重能否学回固定码decoder未学到的功能。它不把HF权重替换当学习，也不把单L9代表全MLP。free-dense仅为超预算诊断；最终目标仍是logical≤2.6、完整九项相对各自BF16下降严格<0.9pp及跨系列成立。约70分MMLU是研究希望，不新增正式单项门槛。

## 已有证据及本次新增条件

| 既有条件与结果 | 本次动作和解释边界 | 来源 |
| --- | --- | --- |
| f927全108MLP固定码D、知识aux-LM400：八项68.6948431，MMLU8469/14042；均低于父 | 不重新把全MLP/知识LM列为未试方向；改用相同部署初态下D对自由dense真实学习 | [完整负例](../../docs/experience/history/w25_transfer/2026-10-04_29_research_log.md#10月5日全mlp知识lm400结果与后续控制) |
| f927全108MLP码+D400：八项67.2051364，MMLU8518/14042；低于匹配D控制 | 不用再加STE或翻码作为默认修复；本原型无码位更新 | 同上正文 |
| 全MLP非保护HF干预曾到MMLU约69.8547；单层弱不能推定全族弱 | 只作为定位依据。HF干预撤销历史适配且超预算，不证明能训练到，也不是D严格上界；本机未重新核旧完整原件 | [当前父条件空间](../../docs/experience/lessons/liftquant_recovery.md#当前父源的单层全mlp与跨模型条件空间)、[归档边界](../../docs/experience/history/w25_transfer/2026-10-04_29_research_log.md#10月29日五方向子代理分工与接续边界) |
| 局部MLP/完整块训练与留出MSE改善仍可伴随下游退化；多轮小范围拟合也未解决MMLU | 使用真实整模型答案目标，并分开读训练拟合、完整MMLU和九项；不以局部误差晋级 | [损失经验](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)、[可学性与参数化预算](../../docs/experience/lessons/liftquant_recovery.md#定位范围的可学性与参数化预算) |
| aux1024没有subject标签；扩同源数量不等于补齐57科学科，旧Web/Med也有正负条件 | 首轮固定既有合法数据，不把新数据、优化器和参数化同时变化；负结果只约束该目标/分布/预算 | [数据边界](../../docs/experience/lessons/kd_and_data.md)、[对照经验](../../docs/experience/lessons/experiment_design.md) |

本次没有引用外部训练效果承诺，也不以UPQ等大token实验估计本小预算能追回7pp。

## 初态与两臂

正式判别唯一初态是新C0完整成品：必须完成原gate500、live up/down各S2×10K/500，使用真实pipeline终态及严格保存审计绑定UUID、metadata、权重、代码SHA。预计拓扑为216 native＋36 fixed2 compact、0 pending dense；不提前写死尚未生成的成品UUID，不从中间gate/up模型开始。

在C0完成前，可用已保留B完整原生成品做全模型资源/拓扑原型：UUID `87571461-7310-46e5-bc2e-904baf1818ed`，`/data1/cmcc/VAELLM/artifacts/cat_structure_full_20261029/B/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/final_model`。B的252 native不等于新C0 mixed拓扑；其结果标为`prototype_only`，只能复用未变机制/资源证据。旧Muon临时compact父已清理，不能引用为仍可用初态；若须提前补compact算子，用已保留真实载荷作限定机制检查，不能把它冒称正式C0。

| 项目 | 固定码D臂 | free-dense臂 |
| --- | --- | --- |
| 更新范围 | 所有36层gate/up/down现有decoder的实际可训练叶；按metadata与模块所有权派生 | 相同108投影的全部原坐标非保护输入列，FP32 master |
| 初始化 | C0自身部署decoder/码/几何；不扩宽、不重拟合 | 从C0每个投影的真实BF16部署矩阵取非保护列，再升FP32；不读取HF投影值 |
| 冻结 | 全部码、保护、路由/长度、旋转、稀疏与其余decoder，以及attention/norm/head/embedding/bias | 原保护列、attention及所有非MLP状态；初始压缩码和几何只作不可变来源 |
| 前向 | 已有native/compact可微decoder计算及BF16整模型 | 恢复完整BF16矩阵后一次`F.linear`，原bias冻结 |
| 产物 | 仍可按真实位宽判定低比特候选 | 144 native attention＋108 BF16 dense的诊断模型；明确超预算 |

free-dense只注册`W_nonprotected_fp32`为Parameter；保护索引及其原顺序、BF16保护值为buffer。每次由不可变保护模板和非保护BF16 cast按实际补集`index_copy`恢复原矩阵；不得把保护列也放入Parameter后只mask梯度，更不能给它weight decay。不要拆成保护/非保护两次GEMM相加，以免新增初态舍入差。master已在原坐标，不再次旋转。对B的gate/up每投影保护64输入列，down保护256；正式C0必须按实际metadata核对，不凭层号假定。

native初始化沿真实snapshot的部署group8预热取`_cached_weight`；compact沿`CompactRVQLinear.decode_weight()`，含真实BF16逐stage求和、transpose/逆旋转、保护回填。仅调用`_decode_weight(dtype=BF16)`不足以保证路径，因为`_decoder_compute_dtype`可覆盖它；`block_weight_graph.py`的FP32重建也不是此初态。若新父出现low-rank/sparse/protected-residual/output-protection等额外状态，先明确实际部署语义，不静默漏项。

转换前后应核完整W及保护列、整模型BF16读出；free-dense初态必须保留其来源函数。D可微训练decode与原部署cache可能有正常BF16差异，应在零更新状态单列实际loss/logit/候选排序差，不默认二者逐位同值，也不加补偿项伪造相等。若差异已大到能改变对照解释，先排查算子路径，质量训练暂不启动。

正式解释须分开共同BF16部署父、D零更新live、D训练末live和D的BF16部署端点；dense零更新需对齐共同部署父。训练拟合按各臂自身零更新live基线读，最终能力按共同部署父读。数据／教师／token可严格相同，实际可微前向、参数坐标和LR／优化几何仍有差异，因此这是“现有D学习路径与自由dense学习路径”的配方可学性比较，不能单归因于容量或证明D表达上限。logit相对差与W相对位移不是同量纲，少量top1不变也不证明质量无影响。


## 固定输入、教师目标与有效token

首轮建议复用现有 **H96静态teacher forcing**，不是让两臂独立在线生成OPD。如此同输入/同目标/token数可严格配平，同时teacher不常驻GPU。这与当前`C0→E`的在线Muon32不是同一个实验，不复用其结果作本D控制，也不把差值孤立解释成“只换优化器”。

数据权威入口为`result/qwen3_8b_mmlu65_20261004/qwen_hotpot_recovery_20261012/dataset_prepared.json`及`dataset.pt`（16,782,218 bytes，SHA256 `dd19ad88b2ba0829badb60515d9a1088ae40445e6514dd7a6fbf12c22164bb37`）。固定1152身份：M1024、Qasper32、Hotpot96；seed137，原32组顺序，每组M32＋QA8，M每题一次、QA每篇两次。保留完整原输入、答案query、mask、原生prompt/BOS/EOS；不重新下载、筛题、截断或补造学科标签。Qasper复用历史已固定的选择范围，不将其描述为本次无筛选随机样本；本次不追加文本去重或测试重叠过滤。

| 来源 | 版本和split | 许可及身份依据 |
| --- | --- | --- |
| `cais/mmlu` M1024 | revision `c30699e8356da336a370243923dbaf21066bb9fe`，config `auxiliary_train`，split `train`；全池99842行的原seed137 hash前1024 | 发布卡MIT；原Arrow SHA与case身份继承`qwen_teacher_scope_qa128_20261011/dataset_prepared.json:data_contract.mmlu`。[发布卡](https://huggingface.co/datasets/cais/mmlu/raw/main/README.md) |
| `allenai/qasper` Q32 | release v0.3，HF revision `fdc9d8214fbab5dd782958601db4d678e6934a54`，官方train；旧Q128的原顺序前32 | CC BY 4.0；上述prepared的`data_contract.qasper`绑定raw `qasper-train-v0.3.json` SHA `9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4`与实际case。[发布卡](https://huggingface.co/datasets/allenai/qasper/raw/main/README.md) |
| `hotpotqa/hotpot_qa` H96 | v1.1，revision `1908d6afbbead072334abe2965f91bd2709910ab`，distractor/train | CC BY-SA 4.0；`qwen_hotpot_train_cpu_20261011/prepared.json`与迁入manifest绑定两片原train和96身份。[作者主页](https://hotpotqa.github.io/) |

许可信息于本次读取发布者网页核实；main网页仅作许可说明，训练版本始终以上述冻结revision/原件SHA为准。历史manifest中的`/home/star`仅身份文字，实体经现有`cmcc_runtime_20261030.paths.resolve_legacy_path`映射；数据迁移的实际路径按本机已核对应关系，不机械重写原JSON使SHA失效。

教师为原始完整Qwen3-8B BF16/FA2、FP32 RoPE，使用原full7已认证缓存。`teacher_targets.json`组合原M/Q缓存和Hotpot缓存；按原input/query/support/target SHA检查。当前HF五shard/Tokenizer及C0非投影骨干身份复用`muon_compact_cmcc_20261030.inputs.teacher_identity`的核验原则。旧载体UUID不等于旧学生教师：原缓存恢复全部252投影、骨干已核同HF；身份条件保持时可复用，不必重新推理。

目标保留现有数学：M为四个原生候选上的`0.5*p_F + 0.5*one_hot(gold)`，T1，仅混一次；QA为完整真实prompt＋gold答案前缀上的全151936词表forward KL，先逐答案token均值，再每文档`.5/8`。M每题`.5/32`，两域总权各`.5`；R0，不新增prompt loss、CE项或动态权重。M的“1个监督位置”含四选支持；QA每位置全词表，不能把两者同数量视作同信息量。

2026-10-30 CPU实际读取数据（CUDA未初始化）与原`data_contract.token_counts`一致：

| 每32更新完整周期 | M | QA（Q32/H96各两遍） | 合计 |
| --- | ---: | ---: | ---: |
| 有效监督位置 | 1024 | 1712 | **2736** |
| 实际TF输入tokens（不含padding） | 364534 | 688622 | **1,053,156** |
| 最长输入 | 958（group10 M32） | 10593（group2 Qasper） | 保留原长输入 |

QA唯一答案token为856；Qasper唯一515、Hotpot唯一341。损失依旧按token→文档→域归约，不改成把2736位置统一等权。两臂应记录实际消费的case/target哈希、有效/填充tokens与归约，而非只报告名义步数。

隔离`data.py`可复用现有Muon loader的数据/cache验证，但须保留原静态QA目标；该loader目前刻意丢弃QA目标以供在线OPD，所以不能直接套其最终返回值。不得将旧学生生成轨迹放进本静态目标；也不调用会生成各自QA轨迹的旧`fit()`。

## 参数坐标、优化与正式预算决策

机制短验先用AdamW、FP32 master/gradient/m/v、betas(.9,.95)、eps1e-8、WD0；全部活动参数联合clip1.5一次，`foreach=False`避免整组额外scratch，非reentrant梯度checkpoint。D基准LR为`1e-6`，dense基准LR暂为`1e-5`，仅为机制验证初值。原500线性前缀为`lr(step)=base_lr*(1-(step-1)/500)`：首步为base，第二步为`.998*base`，**无warmup5、无首步零LR**。不因short2修改总时域。

不同坐标使用相同数字LR不代表公平；本组也不声称不同LR已最优。须按两臂记录未clip/clip范数、clip比例、每类非保护BF16有效W相对位移、更新存活比例、固定训练组前后loss/KL。dense巨大坐标的raw范数不能与D叶范数直接比较。首轮不加Muon、κ匹配、按层LR或dense→D梯度投影，以免把参数化比较再混成优化器组合。

正式质量预算尚未定稿、没有排入夜间队列。推荐待short2实测后一次固定 **128连续更新、64中点、128末点**：重复原32组4次，两臂各4096次M/1024次QA呈现，监督位置10944、TF输入4,212,624 tokens；64中点各为其一半。中点只读出，不重启优化器、不重置LR、不按分数择优停止；质量端点预先固定128。若两卡峰值/实际成本无法容纳此有界批次，主线程应在启动前明确缩到哪一完整32组周期及读出，不悄悄改为短测后直接声称128质量已授权。

此预算仅检验“已有静态目标在当前训练分布下是否可学及是否迁移”；M依旧只有1024唯一辅助题，四遍不等于4096独立知识样本，也不能代表全面57科覆盖。128步的负例不证明dense或D容量上限，更不排除其它目标/覆盖/优化器或长预算。若短验显示所有有效更新都被BF16吃掉、目标增加且步长显著失常等具体优化问题，先修实现或只调整一个有据LR，再以相同短验条件核受影响路径；不提前铺LR×预算×数据网格。正式LR和卡数必须连同预算在启动前落到配置，当前值不能被误读为已选定最优。

## 资源与最小两更新验证

按B真实metadata、H4096/I12288/L36计，全部MLP原权重5,435,817,984，非保护5,341,446,144，保护94,371,840。每层gate/up各非保护49,545,216，down49,283,072。以下为算术下界，**不是GPU实测**：

| 项目 | 总量 |
| --- | ---: |
| dense FP32 master | 19.8984375 GiB |
| dense FP32 gradient | 19.8984375 GiB |
| Adam两个FP32 moments | 39.796875 GiB |
| 上述静态合计 | **79.59375 GiB** |
| 两卡均分上述静态状态 | **39.796875 GiB/卡** |
| 当前实现全形状BF16保护模板（常驻） | 10.125 GiB |
| 全MLP显式BF16矩阵（若同时保留） | 另10.125 GiB |
| 冻结attention BF16缓存 / embedding与head | 约2.8125 / 2.3184 GiB |

当前`protection_template`虽仅94,371,840个保护值非零，仍分配完整矩阵；不能把其常驻费用按约0.176GiB非零值计算。还需激活、矩阵cast/index_copy、反向临时量、优化器scratch、通信和碎片。单80GB不够；两卡是候选，不承诺一定够。本次CPU只读内存为约1.08TB总量、1.018TB available（动态快照），不据此占满主机。

首选单进程两卡驻留layer MP：0–17层/embedding放逻辑cuda0，18–35层/norm/head放cuda1，两臂同切分便于比较。复用`compressed_e2e_fintuning/device_map.py:apply_layer_device_map`递归搬层args/kwargs和autograd `.to()`，只用boundary hooks不足。保持原FP32 RoPE与其`original_inv_freq`别名，不能对整个模型无条件`.bfloat16()`。旧M32 helper把options放embedding卡再对末卡logits gather有确定失配，隔离目标helper应按`logits.device`放options；QA helper本已有此语义。

teacher/cache驻CPU、按当前目标拷入，不常驻teacher模型。每次只保留一个实际训练学生，D与dense顺序短验；冻结attention缓存由`no_grad`创建，不能用`inference_mode`张量参与之后hidden反传。不要逐步复制全部20GiB master来算位移，可逐模块归约。现streaming offload移动Parameter但不自动同步Adam状态，不作为首个实现；两卡确实OOM时先记录分项峰值，主线程再决定等待可用四卡或设计明确CPU optimizer offload，不缩模型/范围/长度、静默改精度或安装依赖。

当前GPU6/7是短期窗口，C0完成后其完整评测已有安排；不能让尚未定预算的质量长训抢占。GPU0/1/5在途、2/3/4保留原链；启动时由主线程单次核资源与依赖，不轮询或占卡。

两更新短验采用原完整groups **[2,10]**：前者覆盖最长QA10593，后者覆盖最长M32输入958及Hotpot域；全部M32＋QA8保留。Hotpot最长2559是同QA算子的更短输入，不另外增加一轮峰测。两臂分别fresh起步，执行相同输入/目标，首轮两次均真实更新。

完成性质是：108投影/36层全范围、真实跨卡checkpoint重算与FW/BW/累积/统一clip/Adam；全部活动master梯度有限、108个投影均获得非零梯度、第二次非空m/v按连续step更新、非活动参数无grad；attention/保护/bias/norm/head/embedding逐字节冻结；更新有实测BF16存活。保存各卡allocated/reserved峰与两步结束后的常驻状态，区分首次Adam分配和第二次不当累积，不把“未OOM/能启动”当正确。

同时做一次完整初态功能对照和末态独立BF16导出/fresh加载同输入读出；不默认要求正常训练差异逐位一致。B短验只证明B路径，新C0出现后只补其实际compact/来源/拓扑变化，不能借B两步宣布C0已完成质量训练。没有正式resume需要时，不额外做exact-resume审计。

## 保存、读出与停止条件

free wrapper不能直接交现有`inventories()`；它应拒绝未知算子。读出时独立CPU部署对象将108个free算子物化为普通BF16`nn.Linear`，144 attention与非MLP照冻结源复制，复用mixed checkpoint保存/加载支持的`pending_dense`边界。独立manifest标记`diagnostic_overbudget=true`，如实算实际存储，不能把成品逻辑位宽写成通过。D保存仍用实际native/compact serializer及账本。

中点只复制必要BF16部署状态，不原地更换FP32 Parameter/storage或optimizer。若以后要求恢复训练，另存真实FP32 master、m/v、step/数据游标和CPU/各GPU RNG；单份dense master+m/v已约59.6953GiB，BF16部署save不能充当exact-resume。当前未要求该路径，不默认多留训练二进制。

正式读出建议：C0冻结控制复用其完整九项；两臂0/64/128测完整固定训练目标与M训练准确率/gold NLL，64/128各读完整正式MMLU；128末态两臂各完整九项。先固定这些端点，再看分数。中点不用于选测试逐题补训，属于开发观察，不能称盲测。自由dense须隔离source检查适配现有评测，不能放松原`final=True`对合格compact成品的库存断言；评分本身沿相同full8 64叶/24742与LB21/4750协议。

| 观察 | 可作的决定 | 不能推出 |
| --- | --- | --- |
| dense训练拟合强且完整MMLU明显修复，D弱 | 当前目标下存在跨层可学习方向，优先研究预算内参数化/优化/重编码如何承接 | D严格容量上限、任意7pp必能压回2.6bit、跨模型成立 |
| 两臂训练拟合明显改善而完整MMLU弱 | 先检查固定目标/数据与训练外迁移；记录QA/七任务代价 | 知识已擦除、dense无能力、数据是唯一根因 |
| 两臂训练功能也未学到 | 先核实际步长/BF16存活/梯度/预算；有新证据才增加预算或改一个因素 | 泛化失败或不可学习 |
| MMLU提升但九项不升 | 保留量级恢复机制证据与代价，成品仍不合格 | 总体验收完成或综合收益 |
| 固定配方预算结束无净收益 | 关闭这份配置的后续自动加码，最小保留证据 | 整个参数族/目标/所有训练预算无效 |

技术失败、冻结/数据/目标违反、实际源不完整或初始化失配会停止受影响依赖；质量`not_met`是有效结果。loss下降不触发无限延长，也不因BR或当前原链完成时间变化自动取消本设计。下一次同位宽码+D重编码、保护自由度、数据扩覆盖均须有本组新增证据后另定一个问题，不一次并入。

## 已完成的CPU证据与当前边界

数据验证实际exit0：`result/qwen3_8b_mmlu65_20261004/mlp_learnability_cmcc_20261030/data_cpu/validation.json`。完整H96的实际输入／目标合成SHA与原F_A_HOT记录相同；M只混一次，QA恢复原静态full-F目标。groups2/10覆盖上述实际形状，CUDA未初始化。跨GPU的M候选gather改动尚待真实两卡前向验证，不以数据核证代替它。

核心第一版CPU验证实际exit0、73.505秒：`artifacts/mlp_learnability_cmcc_20261030/core_cpu_v1/report.json`。真实B完整strict加载与108库存核证，native gate/down及从实际gate载荷构造的compact三个完整尺寸模块均通过初态部署矩阵／输出、输入梯度与非保护梯度对照；各两次FP32 Adam更新有BF16存活，保护和原source未变，独立进程重新从B派生后strict重载及同输入BF16输出一致。一次性state已清理。此处使用小batch机制输入，并未执行全36层整模型学习或两卡峰值；不能称正式条件短验已过。

算法审查要求全模型入口补齐：真正C0的完整pipeline身份绑定、一次完整部署group8预热后取各层W0、拒绝尚未覆盖的额外保护／稀疏拓扑。来源／真实group8缓存／额外拓扑边界已修，完整两卡入口经源码审查。独立CPU全D导出组装已exit0、51.94秒，报告在`artifacts/mlp_learnability_cmcc_20261030/export_cpu_v1/report.json`；该零更新检查未伪造两更新模型。另全108 free-dense安装／零更新普通Linear序列化CPU验证已exit0，744.779秒；`pipeline_cpu_v1/report.json`核5,341,446,144活动元素、2127持久字段独立strict、保护和非MLP exact，预算如实超限，CUDA未初始化。零更新测试模型13,886,029,126字节已清理。核心证据按实际代码范围复用，新GPU全路径仍待闭合。

真实C0入口的[只读CPU核证](../../result/qwen3_8b_mmlu65_20261004/mlp_learnability_cmcc_20261030/c0_interface_review_20261031/validation.json)已exit0：实际`run_short`／`fresh`导入与CLI、四个N namespace及其已导入子模块、本机HF路径的meta-only 252投影库存和实际未完成C0拒绝均通过；既有B两臂的源码、保存／独立fresh、保护与非MLP冻结证据匹配。五份活动计划的1003／1021／1019／1040／1061项pins全部保持，分别为C0、直接C0→Muon、Rice C1、Rice→Muon及BR固定链；本包`bindings.py`也在活动BR的pins内，源码未改。本次没有加载C0／B权重、重复B短验、启动GPU或等待器；完整C0尚未出现，真实compact全模型匹配short2仍未执行。

## 当前短验入口

首个decoder v1在GPU计算／训练更新前因公共设备map解析器将`cuda:0=0-17,cuda:1=18-35`误判为单device而exit1。仅隔离调用改用该API已支持的`0=0-17,1=18-35`；公共代码不变，CPU实际核36层／每卡18、CUDA未初始化，见`map_cpu_v1.log`。原v1失败、命令及SHA保留，不是学习负例。

修复后的decoder v2于22:59:17启动，tmux `mlp_decoder_short2_v2_20261030`，worker327198/start10974575；随后dense v1于23:09:24启动，tmux `mlp_dense_short2_20261030`，worker334702/start11035253。两臂各两更新、各独立fresh及外层全部exit0，均已退出，GPU6/7释放。日志及外层退出分别为`artifacts/mlp_learnability_cmcc_20261030/short2_decoder_v2.log/.exit`与`short2_dense_v1.log/.exit`，各自fresh日志同根`*_fresh.log`。原dynamic／Rice／C0链及其源码保持。

调用者激活bitvae，通过tmux/nohup执行；以下已使用的输出不能重复运行：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -B -u -m experiments.mlp_learnability_cmcc_20261030.run_short \
  --arm decoder \
  --output /data1/cmcc/VAELLM/artifacts/mlp_learnability_cmcc_20261030/short2_decoder_v2 \
  --timeout-seconds 14400

CUDA_VISIBLE_DEVICES=6,7 python -B -u -m experiments.mlp_learnability_cmcc_20261030.fresh \
  --directory /data1/cmcc/VAELLM/artifacts/mlp_learnability_cmcc_20261030/short2_decoder_v2
```

dense臂使用`--arm dense`及独立`short2_dense_v1`目录。两臂各有`execution.json`／核心日志／`report.json`／`fresh_audit.json`及真实退出，保留全部来源、训练统计、原始数值和实际账本。代码在短验运行中冻结；技术失败先核实终态再定位修复，不自动重试。短验成品和一次性CPU serializer模型在独立核证完成且无后续用途后清理；保留证据，不借其作为正式新C0模型。

**真实C0的后续调用尚未执行。** 必须等当前C0 v3的完整成品、pipeline终态及strict审计齐备，并由主线程确认GPU6/7实际空档；这两卡仍优先服务已安排的C0及直接Muon完整评测，不能因暂时空闲自行启动。调用者激活`bitvae`后，在tmux/nohup中沿上述入口运行两臂，保持`CUDA_VISIBLE_DEVICES=6,7`与`--timeout-seconds 14400`。每个`run_short`都必须显式添加`--checkpoint /data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/final_model`；省略该参数仍会使用默认B，不能得到C0匹配短验。

| 后续两臂 | `run_short --arm` | 新的`run_short --output`，亦为其后`fresh --directory` |
| --- | --- | --- |
| C0固定码D | `decoder` | `/data1/cmcc/VAELLM/artifacts/mlp_learnability_cmcc_20261030/C0_v3_decoder_short2` |
| C0 free-dense | `dense` | `/data1/cmcc/VAELLM/artifacts/mlp_learnability_cmcc_20261030/C0_v3_dense_short2` |

两臂各从同一原始C0重新fresh起步，输出各用全新目录；各自训练exit0后才调用同GPU6/7的独立`fresh`，不续用B或另一臂权重。仍仅执行原groups[2,10]的两更新及保存／fresh核验；没有128步训练、正式学习率决定或质量验收。

## 两卡短验实测与取舍

原始证据：[D训练与导出](../../artifacts/mlp_learnability_cmcc_20261030/short2_decoder_v2/report.json)、[D独立重载](../../artifacts/mlp_learnability_cmcc_20261030/short2_decoder_v2/fresh_audit.json)、[dense训练与导出](../../artifacts/mlp_learnability_cmcc_20261030/short2_dense_v1/report.json)、[dense独立重载](../../artifacts/mlp_learnability_cmcc_20261030/short2_dense_v1/fresh_audit.json)。相同输入／目标合成SHA为`35f476e2f43c597e2bc1e1b85f603da36ebec3fd2e2f4469debf76c21c147b1b`，均真实消费完整groups2/10，更新实际包含90,328输入tokens／200监督位置（额外只读观察另计），没有截断、缩范围或常驻GPU教师。

| 实测项 | 固定码D | free-dense |
| --- | ---: | ---: |
| FP32活动元素 | 2,598,912 | 5,341,446,144 |
| 训练／部署导出墙钟 | 491.664秒 | 492.448秒 |
| 独立fresh墙钟 | 28.667秒 | 56.919秒 |
| 单卡最大训练allocated | 18.087GiB | 54.442GiB |
| 单卡最大训练reserved | 32.496GiB | 62.816GiB |
| 两步后持久字段strict exact | 3423 | 2127 |
| logical／packed bpw | 2.4987315302／2.5121061906 | 13.1101286515／13.1235033118 |
| 整模型tensor bytes | 4,670,999,040 | 13,884,036,096 |
| 保存目录bytes（清理前） | 4,674,873,573 | 13,886,781,006 |

D部署初始化的allocated峰约30.09GiB／卡，高于其训练段；dense两次更新结束常驻约48GiB／卡，第二步峰54.44GiB，无观察到不当累积。两卡可行只在本Qwen3 B、18／18层、完整M32×958／QA10593、FP32 Adam与当前缓存／checkpoint条件下成立；不能外推单卡、在线教师常驻、C0 compact全部路径、其它长度／层数或更大batch。

两臂两步均有全部108投影有限非零梯度、连续Adam状态、冻结补集和保护列exact，全部108个部署投影出现实际BF16更新。dense的初态完整模型probe及M32四选分布与共同部署父exact，live末态→BF16部署也exact；D零更新live相对部署probe的relative-L2为0.0118452318，M32四选logp为0.0146532747、0／32 top1改变，训练末live→BF16部署为0.0126845573。这里不补偿、也不称差异无害；二者部署→fresh均在同一完整64token真实输入上全词表末位置BF16 exact，FP32 RoPE保持。

两个已训练组的静态目标分别为：D group2 `0.4854002343→0.4512541257`、group10 `0.2423744667→0.2273064286`；dense为`0.4902623557→0.2965167377`、`0.2450374335→0.1528333298`。gate/up/down的非保护BF16有效W相对位移，D约`.003097/.003417/.003442`，dense约`.000348/.000382/.000378`。这些只证明当前两组训练功能可优化及更新能部署，不证明MMLU、完整九项、低比特容量上界或跨模型收益；尤其不能把dense的13.11bpw超预算诊断称为压缩成品。

本批采纳的是已验证的隔离实现及两卡资源方案，保留两条学习路径供真实C0上的后续受控判别，不据两步选择最终方法。正式C0尚未生成；待其完成后，仅补新增36compact的全模型匹配两更新：D的gate decoder反向／部署，dense的实际compact W0／保护转换及共同初态，复用其余B证据。目前没有证据要求改LR／clip或重复B短验。随后按原设计定稿数据、LR和预算，128步仍是待定建议，本次没有自动启动质量长训。

两个GPU短成品无后续模型用途，独立核证后仅删除各自`final_model/pytorch_model.bin`，实际释放分配空间18,556,989,440字节；config、metadata、原始指标／full state哈希、fixture、代码／命令绑定与核心日志保留。上述目录现为证据，不能继续加载为模型；源B原件保留。CPU零更新serializer生成文件另记录13,886,029,126文件字节，未测分配块，因此不合并声称物理释放量。工程经验见[全MLP两卡资源与部署初态](../../docs/experience/lessons/memory_precision.md#全mlp非保护权重学习的资源与部署初态)。

## 最小文件职责与交接

| 隔离文件/职责 | 可复用入口 | 必要新增内容 |
| --- | --- | --- |
| `bindings.py` / source inventory | mixed `checkpoint.load`、cmcc路径映射、已有来源审计 | B prototype与新C0质量来源分开；实际108MLP及冻结补集、SHA |
| `free_dense.py` | 实际native部署cache、compact `decode_weight` | 非保护FP32 Parameter、BF16完整W与冻结保护；禁止HF初始化 |
| `data.py` / 目标读出 | HOT dataset/teacher validators、已有M mixture与answer KL | 保留静态QA cache；MP末卡gather；相同输入/目标/token契约 |
| `run_short.py`或等价薄入口 | layer MP、原Adam数学、既有非reentrant checkpoint | 两臂分别fresh的完整groups[2,10]，每卡峰值、全范围梯度/冻结/更新审计 |
| `export.py` / `fresh.py` | mixed save/load与官方评测计算协议 | 独立BF16诊断端点、strict reload与同输入读出；不伪造预算通过 |
| 后续`run.py`/正式配置 | 上述已通过路径 | 短验后一次固定LR、卡数、更新/token、64/128读出与成本；本轮不预先启动 |

实现及同路径验证由sol/xhigh负责，Astra/ultra审查参数化和结果解释，主线程统筹资源和正式预算。当前可以实现并CPU验证隔离模块；GPU短验及后续预算由主线程依据实际资源推进。设计阶段无新增实验效果经验；后续结果和清理应归本正文/原报告，主线程统一更新计划、lessons与导航，不复制第二份执行计划。

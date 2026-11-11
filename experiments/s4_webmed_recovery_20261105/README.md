# S4 down 后 Web／Med 数据替换恢复

2026-11-06核实：正式128步及同一checkpoint完整九项已于03:55结束，训练、评测和driver均exit0。九项67.59235905%、MMLU10034／14042，仍未达到S4质量目标；没有新增训练后继。唯一排程入口为[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)。本包只研究第一阶段四阶基线的恢复，不进入渐进门控训练。

## 问题、对照与已有依据

本批检验：在同一 down 压缩后、恢复前的 S4 父模型上，把原 auxiliary4096 换为两个已有研究依据的原生四选训练来源，是否改善知识迁移和完整九项。只新增一条恢复配方；父模型、QA、目标比例、优化器和更新预算与已完成半学习率臂固定。数据来源和题目内容同时改变，结果只能解释为这一数据配方的差异，不能证明学科覆盖是唯一原因。

指定父为 `fa50bf40-c682-4910-8e72-d024f6b678e4`，位于 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/trial_20261103_1730/formal/nodes/down_proj/pre/model`。各新运行从该父 fresh 开始，不接续已恢复成品。父完整九项67.38954114046014、MMLU10007/14042；已完成半LR控制 `8a874b00-a3ce-44f5-b869-7af1579cf988` 为67.6312603284005、10002/14042。控制完整指标直接复用，见[同父诊断批](../s4_down_recovery_diagnosis_20261104/README.md#完整结果与本批结论)。这属于用户明确指定父源的局部恢复研究，尚不构成 BF16 整链定稿。

依据来自[知识目标与数据经验](../../docs/experience/lessons/kd_and_data.md)：auxiliary_train 不保证57科学科覆盖，旧 Med／Web 原生选择题、12源储备及57科V2匹配均已研究；57科V2两条2000步完整对照未胜旧C1，不能把“匹配过学科”直接等同有效恢复。旧 Web 替换对某一 D-only 配方有小幅完整收益，但不同父、训练范围和预算限制其迁移性。本批复用旧来源与保守解析规则，验证新 S4 条件；不重复从头盘点 aux，也不复活已弃用的 validation 训练池或逐题测试匹配。

## 固定配方

| 项目 | 本批约定 |
| --- | --- |
| 知识数据 | WebInstruct 原生四选 train 2048＋MedMCQA train 2048；各唯一题呈现一次 |
| 每次更新 | Web16＋Med16＋原QA8，共128更新；QA原Qasper32／Hotpot96的身份、输入、target和顺序不变 |
| 知识目标 | 原始完整BF16教师的四选条件概率，与gold在概率空间混合一次：`q = 0.5 p_teacher + 0.5 one_hot(gold)` |
| 域权重 | 知识域0.5、QA域0.5；与上一行教师／gold比例是两个不同参数 |
| QA监督 | 原静态gold-prefix、完整词表教师KL；不切换在线OPD |
| 更新范围 | 全252个已安装S4投影decoder、1512叶；其余冻结，不增加LoRA、码更新或额外保持项 |
| 优化 | fresh AdamW，LR5e-7，原500步线性日程的前128步，无warmup；betas(.9,.95)、eps1e-8、weight_decay0、clip1.5 |
| 精度与种子 | FP32 master／梯度／moment，BF16计算和原生部署；seed137 |
| 长度与抽样 | 复用旧原生解析、固定hash排序；不按教师对错、测试表现或长度挑题，不静默截断超长输入 |
| 实测位宽条件 | 父logical／packed投影账本均4.094031292459239bpw，含decoder／保护／旋转；embedding／head不属于该投影账本 |

Web 固定 `TIGER-Lab/WebInstruct-verified@3e8a350b3a935d68fe70bdf379692500abc9ff51`，原train228736行，按旧规则识别原生四选。Med 固定 `openlifescienceai/medmcqa@91c6572c454088bf71b679ad90aa8dffcd0d5868`，原train182822行，保留原选项和`cop`答案。两份原始Parquet已恢复并核验历史SHA：Web为`e55a8e985e18e378e4cee9059442e94ed3e8d0f6280c4b3c6e0b2231982fc975`（43775058B），Med为`b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5`（85899025B）。实际读取各自固定revision的dataset card，均标注Apache-2.0；来源／版本／split／card SHA随材料记录。不使用正式测试或validation训练。旧原始数据未全部迁移，不能把历史清单当成本机已有数据。新M题重新计算原始BF16教师概率，QA直接复用原材料；旧aux混合缓存不得错接或再次混gold。

### 0.5 的证据边界

0.5不是已证最优值。[旧八格](../../docs/experience/history/w25_transfer/2026-10-04_29_research_log.md#10月11日-train-only-八组完整结果与下一轮覆盖对照)在a0e5／hybrid教师／32步条件下比较过纯教师与各半：M1024两对各半的九项+.119770／+.138691pp，M128两对则为负。[同C1 Muon32](../c1_m_target_muon_cmcc_20261102/README.md#完整结果成本与本批决定)比较过各半与纯gold，纯gold九项低.1527675556pp、M少31题。两组条件不同，不能拼成同组alpha=0／.5／1扫描；C1的`full_teacher`仍含一半gold，只改变支持集和教师执行路径，也不是alpha=1。

未找到teacher／gold=.25／.75等中间比例的已完成比较；当前S4／WebMed亦未比较。这里固定0.5用于隔离数据配方变量，不把它宣称为迁移最优。用户本次询问比例依据尚不构成追加网格授权；本批结果汇报后再讨论是否做同父、同数据的比例对照。

## 验证、评测、预算与汇报

新数据连接只补必要验证：固定来源和真实train身份、原始选项／gold与token化、一次混合数学、QA精确复用、新batch形状和实际最长输入、真实更新和冻结范围。复用适用的S4保存／fresh／64读出后继续训练等既有证据；只有新增路径或材料异常才补相应验证。CPU及新路径必要GPU验证通过后，正式独立目录从原父启动，不携带短测状态。

CPU prepare已exit0：Web2048题共208429输入tokens、最长969；Med2048题共105562 tokens、最长216。两源M合计313991 tokens，固定更新与题数不等于与旧aux等token预算，解释结果时保留这一差异。新M32最大pad形状为32×969，原QA峰10593且内容／目标／顺序不变，均在旧同父joint已验证M32×1222／QA10593条件内。Astra据实审定不重跑GPU short；先完成teacher4096、一次混合／QA digest和最终CPU plan，再直接fresh正式训练并核真实更新。原64驻留读出后回训、128导出及fresh证据复用，正式64／128评测保留。

| 节点 | 读出和用户汇报 |
| --- | --- |
| 启动 | 报实际数据清单、验证、GPU／会话／PID、日志与结果路径；确认真实更新后停止主动监控 |
| 0／64／128 | 每个节点固定early、late、untrained三组，各128题、每源64；训练组读gold与实际mixed KL，未参训组只读gold／NLL，不新增教师缓存 |
| 64 | 完整MMLU，仅阶段观察，不据此挑最优点；与父和已完成控制同口径比较 |
| 128 | 固定终点fresh及同UUID完整九项；MMLU57科14042题、LongBench21任务4750条，原协议不变 |
| 结束 | 报完整均值、MMLU和其它任务代价、实际费用与限制；再讨论下一批变量，不自动延长 |

使用当前授权GPU6／7，在实际空闲时安排教师缓存／必要验证和单条正式训练；不干扰GPU0–5的另一批任务。参考前批完整链约13GPUh，教师准备和必要验证另计；这是估计，不把CPU等待当实际GPU计算时间。固定128步，不自动增加候选、训练步数、比例扫描或新调度服务。

质量目标不变：相对BF16，九项均值下降严格小于0.4pp、MMLU最多下降1pp，即九项>67.64714491790848、MMLU至少10111/14042。未达标仍完整报告有效结果；技术失败停止受影响后继，不伪装成质量负例。父、已完成控制及本批最终部署模型按具体用途保留；128后未授权继续训练，不为假设用途另存长期optimizer／master／RNG。结束诊断权重按既有收尾规则清理，保留核心证据。

## 运行证据

固定源恢复、CPU原生数据组装及接口修复均已通过。GPU7新教师缓存`cache_fixed.exit=0`，4096条目标共334.17秒，原始HF身份、完整持久状态不变、未经预混合及教师完全释放已验证；一次混合与QA精确复用合同已生成，CPU plan exit0，正式后台流程随后启动。首次cache在加载教师前因新三字段binding与旧五字段checked格式不一致退出1，实际dataset SHA未变；新包改为本地同格式严格校验，不改旧包、不跳过哈希。真实4224个训练case／128个group／heldout128重新加载、CPU未初始化CUDA及binding往返已通过，原失败日志保留。CPU plan固定1137个源码／输入绑定，SHA为`1f3746b7d16678ea4211531652987ca05fd8149d0647d7e6045967f354f2d359`。当前没有本批质量结论。

产物根：`/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/webmed_20261105/trial_20261105_1223`。实际原生数据清单见[dataset_prepared.json](../../artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/webmed_20261105/trial_20261105_1223/data/dataset_prepared.json)，CPU校验与源恢复日志保留在同一根目录。已完成教师会话为`S4_WebMed_cache_fixed_g7_20261105`，worker3381759／start_ticks59656832，日志`cache_fixed.log`；这是已结束的教师准备进程，不能当作正式训练已启动。

正式会话：`S4_WebMed_formal_g6_20261105`，调用者激活`bitvae`后执行`bash experiments/s4_webmed_recovery_20261105/run.sh`。wrapper3387499，训练worker3387852／start_ticks59704797，2026-11-05北京时间14:21进入进程初始化。实际argv／状态见产物根`webmed/run.json`，控制日志`driver.log`、训练日志`webmed/formal_training.log`，正式请求`webmed/formal/request.json`。已核首个真实更新：group0的Web16／Med16／原Qasper8，LR5e-7，252投影梯度有限非零、联合clip1.5前范数107.19294；FP32改变9861033元素，BF16改变567413元素，步耗时200.875秒，峰allocated22547009024B。当前第二组计算中，未把它写成已完成更新。后续64完整M仅观察、128fresh及同UUID完整九项按固定流程继续；此刻没有质量结论。

[启动核证](../../artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/webmed_20261105/trial_20261105_1223/startup_verification.json)已完成：1137个源码／输入pins及原父／半LR控制绑定未变；driver3387520／start_ticks59704053、worker3387852／start_ticks59704797身份已核。证据记录完整首步1／128及第二组当时仍在计算，未声称训练完成；GPU7已释放。没有新增GPU short权重或待清短验副本；本次交付后停止主动监控，保留后台正式链。

**完整终态与比较边界（11月6日核证）。** 最终UUID为`69a31170-bff5-4700-aa67-9471675c3cd5`；原始完整九项位于产物根`webmed/formal/full_nine/result.json`，退出与成本见`webmed/run.json`和`webmed/formal/training.json`。九项均值67.59235904815581%，较共同父+.2028179077pp、较原aux4096半LR控制−.0389012802pp；MMLU10034／14042（71.45705739923088%），较父+27题、较半LR控制+32题。与最终门槛相比，九项仍低.0547858698pp，M仍少77题，两条件均未达。数据配方在本seed／128步下有M与综合分取舍，不能认定覆盖问题已解决；也不能以此128端点代替新范围实验的all64控制。

实际训练128步，均198.42483秒／步，训练／驻留中评／导出及fresh段26157.45秒（7.26596小时）、训练峰allocated33,491,408,384 B；正式全链48863.46秒（13.57318 GPU小时的单卡拥有窗，含训练、导出和完整九项），教师准备与必要验证不含在此数。最终模型作为既有控制、原父及数据继续保护。旧的“第二组／没有质量结论”等启动核证描述仅记录11月5日交付时点，当前终态以本段和原始结果为准。

完整评测实际包含原协议64叶24742条（MMLU57科14042、其余七项10700）及LongBench21任务4750条，逐题来源绑定已核。MMLU相对共同父修对253／改错226，净+27；原固定诊断early70→84、late81→86、heldout62→64，128终点的early和late都已消费，只有heldout未参训。两种训练范围在0步已有BF16核路径差异，不跨路径直接把诊断绝对差归因范围。

同一条全252D训练的完整MMLU三个节点为：恢复前10007／14042（71.26477710%）→64步10004（71.24341262%，较父−3）→128步10034（71.45705740%，较父+27、较64步+30）。64中评UUID为`84a0d452-c91b-4736-adda-783472380bd1`，来源`webmed/formal/step_000064/evaluation/mmlu.json`、exit0；这是原128链内观察点，不是attention分支，也不具备该点完整八项。64步微降没有延续到128终点，不据此推断未观测步数的走势或自动追加预算。

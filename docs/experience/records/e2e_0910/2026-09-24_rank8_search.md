# 0910 checkpoint 的 rank8 端到端微调搜索（2026-09-24）

## 当前状态

2026-09-24 23:37（北京时间）后续：[七天自动搜索](2026-09-24_seven_day_search.md)。下方旧九候选启动状态保留为历史，最新七天运行标识见后续记录。

**固定数据的长程超参搜索已于2026-09-24 11:33:25 UTC后台启动并接管GPU4–7；69+仍未达成。** 控制器wrapper3641607，四个既有实际训练进程已接入，9个候选按400→1000→2000→5000筛选。最佳已保留K400为65.6146%。本次固定修正后的v2数据、task0.5原混合比例与ChatML；取消的数据/模板对照已撤回并清理。数据冻结仅属于本次实验，不写入AGENTS.md。控制器状态、当前配置、最佳权重索引见结果根fixed_data_search/summary.json与best_metrics.json；本次交付后停止人工/代理主动监控，由远程脚本自行接续。

目标保持：初始 `result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model`，id `01457bf3-ef22-49e8-847f-dc721287c2d6`；rank≤8；原八任务全量0-shot均分≥69。没有声称目标已实现，也不保证有限搜索得到全局最优。69须以实际保存模型的同口径全量成绩确认，区分单次达标与稳定收益。

## 证据与差距

参考[实验设计](../../lessons/experiment_design.md)、[残差对照](../docs/exp_results/residual_lora_ab_20260923.md)、[残差经验](../../lessons/residual_lora.md)、[KD与数据](../../lessons/kd_and_data.md)、[checkpoint经验](../../lessons/checkpoint_lifecycle.md)。

- 历史普通LoRA 2000步66.4493，残差66.5645，残差+双hidden各0.1为66.9000；最后一项耗时约2.43倍，收益集中RTE。本轮不继续叠加残差/hidden。
- 旧decoder的step200均分66.35使用eval_limit256；400步训练后导出失败，没有有效全量终态。其普通LoRA LR也仅3e-6，同时改变了dropout、loss、K、prompt、norm/head，400步cosine末端接近零。不能据此否定decoder，也没有top1000优于top100的受控证据。
- 本轮重试依据：恢复普通LoRA的更新强度、保留norm/head，decoder独立小LR，FP32存储/BF16计算；先修复实际导出问题。
- 教师同口径全量均分 **70.19719892390623%**，HF revision `b968826d9c46dd6066d109eabc6255188de91218`，2026-09-24 01:55:28 UTC完成、exit0。它是参考而非学生严格上限。
- 修复后未微调0910均分 **65.37660397423905%**，02:11:57 UTC完成、exit0，距69为 **3.6234pp**。历史初始65.42是不同日期的四舍五入值，不能把约-0.04pp当严格核消融；四组统一以本次新基线计算训练增益。

| 任务 | 本次0910初始 % | 教师 % | 历史普通LoRA2000 % | 历史residual+hidden2000 % |
|---|---:|---:|---:|---:|
| boolq | 81.5902 | 86.6361 | 84.7401 | 84.9847 |
| rte | 71.4801 | 77.9783 | 72.2022 | 76.5343 |
| winogrande | 66.2983 | 67.7979 | 67.1665 | 67.1665 |
| arc_easy | 76.4310 | 80.7660 | 77.3569 | 77.0623 |
| arc_challenge | 49.4027 | 56.5700 | 50.8532 | 50.7679 |
| openbookqa | 39.2000 | 41.6000 | 39.4000 | 39.2000 |
| piqa | 76.3330 | 77.3123 | 77.1491 | 76.8226 |
| mmlu | 62.2775 | 72.9170 | 62.7261 | 62.6620 |

MMLU和两项ARC贡献当前最高均分配置与教师差距的约75%。仍按原八任务均分晋级，同时查看收益分布；只改善RTE难以补足69所需约2.10pp。

## 已取得的400步结果（2026-09-24）

A `lora_lr1e4`：八任务全量0-shot均分 **65.87574545218463%**，相对本轮初始 +0.49914148pp。分项%为boolq83.45565749、rte71.84115523、winogrande67.71902131、arc_easy76.80976431、arc_challenge48.89078498、openbookqa39.4、piqa76.60500544、mmlu62.28457485；八项齐全、指标键与基线一致。原始证据为结果根下 `lora_lr1e4/Qwen_Qwen3-8B_20260924_023506/lm_eval/lm_eval_results_step_400.json`。这个点可复用于后续小batch代理的校准，不代表长程上限或69达标。

B `lora_lr3e4`：八任务全量0-shot均分 **65.94692844283674%**，相对本轮初始 +0.57032447pp，仅比A高0.07118299pp。分项%为boolq83.57798165、rte71.48014440、winogrande66.92975533、arc_easy76.97811448、arc_challenge49.74402730、openbookqa40.2、piqa76.22415669、mmlu62.44124769；八项齐全、指标键与基线一致。原始证据为结果根下 `lora_lr3e4/Qwen_Qwen3-8B_20260924_023506/lm_eval/lm_eval_results_step_400.json`。两组成功暂停后队列当时分别进入C/D，400步权重曾为晋级/对照保留；该旧计划现已被新数据四格替代，A/B权重已清理，配置、原始指标及核心日志保留。不据这0.071pp差距确定冠军。

2026-09-24 04:39:45 UTC核查：C/D均约43步，约35–37秒/步；完成剩余357步约3.5小时，另加评测。最新global grad norm约11，仍在150步预热中；不能由此判定失败，也不能将裁剪比例直接解释成Adam有效学习率同比下降。HF提前暂停后的summary吞吐可能按配置max_steps计数；成本采用实际完成step及墙钟，不使用该summary的train_steps_per_second作400步吞吐。

## 首轮四组与固定条件

| 配置 | 普通LoRA LR | decoder | GPU/顺序 |
|---|---:|---|---|
| A `lora_lr1e4` | 1e-4 | 冻结 | 4,5先跑 |
| B `lora_lr3e4` | 3e-4 | 冻结 | 6,7先跑 |
| C `decoder_lora_lr1e4` | 1e-4 | LR3e-6 | 4,5接A |
| D `decoder_lora_lr3e4` | 3e-4 | LR3e-6 | 6,7接B |

四组使用同一修复后版本，从原0910开始，rank8/alpha16/dropout0.1，全部252压缩projection、0–35层。norm all与lm_head LoRA均LR1e-4，残差none，hidden/preMLP权重0。参数FP32、BF16计算，decoder精度设置仅在可训练时生效。保留离散压缩码与原位宽；不宣称额外LoRA/decoder存储免费，也未重算压缩率。

固定 `kl_top_partial`、K100、T1、prompt0.3、seq1024、seed/data_seed0、原数据混合（II7m .341、IIgen .067、Tulu .028、AM .064、task .5）。2卡DP、每卡batch8、累积2，有效batch32。实际iterable数据使group_by_length关闭；与历史4卡acc1归约形式相同，但样本分片/dropout流不同，本轮四组为主对照。全局grad clip1.5，结合grad_norm解释联合decoder的有效步长。

当前partial是纯KD，alpha0.5不会产生CE；prompt按 `(sum_response+w*sum_prompt)/(N_response+w*N_prompt)` 归约。partial缺少尾部项，mass包含tail桶。task .5是样本比例，不是token贡献50%。当前生成器定义七任务train与MMLU auxiliary_train，但后来对实际加载JSONL的核查强烈指向旧版生成结构，不能把当前生成器定义写成实际数据已经采用auxiliary_train；本处原先的推断已纠正，具体证据与尚未认证的边界见下方“实际task数据版本核查与中间结果”。不声称第三方语料已经逐样本去重。

正式脚本：`experiments/e2e_0910_search/run_{lora,decoder_lora}_lr{1e4,3e4}.sh`。总steps5000、warmup150、cosine cycles0.5，`stop_after_step400`仅限制本段执行；每400步保存、全八任务评估、正常暂停。不能改成steps400再换总步数冒充原轨迹续训。

## 原首轮晋级与预算（6/7卡已调整为下方E/F）

1. 400步按全量任务均分、分项收益、稳定性和实际GPU时间筛选，维护质量/成本上的非劣候选，最多2组晋级1200。其中150步仍warmup；分差接近时优先让两种训练模式各自最佳LR晋级，不把短程落后当作能力上限。
2. 同一数学配置、world/batch和5000调度恢复到1200，复用400/800/1200轨迹选主线，继续2000；仍有增益再到3200/5000。不能仅因尚未69就立即淘汰，也不重复重跑已完成步骤。
3. 若仍平台，固定胜出优化器设置，依次比较既有partial/mass、轻量CE混合，再考虑K100/1000及数据比例；每轮只回答一个问题，具体变更按项目授权边界执行，首轮不混改目标函数或采样。

原计划4–5跑A→C、6–7跑B→D，两条独立链不互相等待；失败停止该链，不自动重试。后续用户已授权停止D并改为E/F，原四组全部完成的预算估计不再代表当前安排。按2卡普通LoRA约15–16s/步、decoder短测约34–41s/步估计，首轮约6–7小时含评估/加载，实际以日志为准，不承诺400步达到69。

汇总复用各运行 `lm_eval/lm_eval_results_step_400.json` 和有效配置，不另建重复指标文件。现有日志均分会跳过N/A，因此晋级前必须确认同一步、全量0-shot、合并结果八项齐全且数值有限；不能拿partial rank或缺项均分排名。boolq/rte/winogrande/mmlu取 `acc,none`，ARC两项/openbookqa/piqa取 `acc_norm,none`，八项未舍入值等权求均分×100。分项增益减本轮初始JSON中对应任务分数，均分增益才减65.37660397423905。

GPU成本用每臂进入训练至该阶段成功评估并暂停的墙钟×实际卡数（A–D为2，E/F各为1），包含保存和评估中另一rank等待，不重复累加rank时间；冷启动/数据缓存准备另列，排队和结束后的显存保留不混入训练成本。仅比较已完成同一预算阶段的配置，不将未完成墙钟当最终成本。精度/成本非劣候选由这些实测点组成，不宣称统计显著或全局最优。

## 小batch快速筛选的设计依据（2026-09-24）

用户明确指单卡小batch、accumulation=1，以更低成本获取候选排序，而不是保持global batch32的accumulation=4。讨论后已获用户授权，具体执行E/F见下；其余缓存/子集等方向仍未实施。对照设计经验要求区分代理筛选与最终口径；既有低样本smoke误作正式基线的失败不意味着禁止使用低成本初筛，而是要求校准其淘汰和晋级依据。

可行方向：把batch也作为超参数，不预设最终必须B32；单卡B4/B8均是待实测选择，既有四臂B32结果用作重合配置的参考。先用少量覆盖低/高LR、冻结/联合decoder的锚点检查小batch短程与B32较长程排序，不能用三四点宣称全局高相关；特别保留接近候选和一个慢热/结构不同的候选。若排序翻转，缩小代理用途，不强制迁移或继续据此淘汰。

批量改变会同时改变梯度噪声和每步数据量，并与LR、warmup、Adam状态和衰减更新次数交互。相同步数并非同token预算，相同token预算也不能保持更新次数相同；记录两者及实际GPU成本，并预先固定各协议的调度和筛选节点。候选比较须覆盖预热后的更新，不能为省时让短段余弦末端归零或直接认定更少token已收敛。

更快比较还包括：同一固定开发题子集作快速下游观察（尤其覆盖MMLU和ARC），辅以未参与训练的验证loss；只用loss不决定晋级。子集与全量排序也用重合锚点检验，只将完整原八任务0-shot用于最终69判定。逐级增加预算，接近者继续、明显落后者才淘汰，空出的资源承接新候选。这个分配思路参考[Hyperband](https://jmlr.org/papers/v18/16-558.html)，不是本模型已证实收益；[batch与调参/预算的相互作用研究](https://jmlr.org/papers/v20/18-789.html)也不构成本轮排序可迁移的证明。

如果小batch优胜，可保留其原拓扑继续；若需要多卡，也可尝试保持小global batch，例如4卡×每卡1×acc1=4，实际加速须测。改变world size/batch不符合当前exact-resume契约；正式换拓扑从0910重新启动，并视为需复核的新运行。离线teacher缓存仅在重复token输入、teacher耗时占比和摊销收益已有证据时考虑，当前不先扩建缓存系统。

用户随后明确“按你的建议继续进行目标”，已授权保留C并停止D、切换6/7卡到小batch探索。D于2026-09-24 04:55:05 UTC经身份核对后向torchrun PID3052948发送SIGTERM，原worker3053015/3053016均退出；该rc1是主动资源重分配，不是数值故障。C的worker3049887/3049888保持运行。D未生成step checkpoint，只保留有效配置、核心日志和中止原因。A/B差距过小，单凭其排序不能验证代理迁移可靠性；尚不能淘汰decoder路线。

## 已批准的单卡低预算E/F（2026-09-24）

| 配置 | GPU | 目标函数 | 正式脚本（隔离worktree下） |
|---|---:|---|---|
| E `proxy_b4_kl` | 6 | `kl_top_partial`，K100 | `experiments/e2e_0910_search/run_proxy_b4_kl.sh` |
| F `proxy_b4_ce_kd` | 7 | `kd_top_partial`，alpha0.5，即0.5CE+0.5partial KD | `experiments/e2e_0910_search/run_proxy_b4_ce_kd.sh` |

两组共用原0910、rank8、LoRA LR1e-4、norm/head LR1e-4、decoder冻结、B4、acc1、world1、原数据/seed/seq1024/prompt0.3。5000步余弦轨迹、warmup20、stop200/save200，正式评估仍原八任务全量0-shot，无eval_limit。只改变训练预算和预热，先不引入小题集，以便直接观察下游方向；约800条样本的结果仍是短程筛选证据，不能与旧B32/warmup150的400步直接排名或认定上限。F若有希望，后续在旧正式协议复核，不能跨world size强行exact-resume。

E/F之间仅目标配方不同；F同时降低KD权重并加入CE，因此检验混合配方，不单独归因于“加CE”。CE针对训练文本，受原response/prompt mask与权重约束，不能称全部数据都是人工真实答案。总loss不能跨两臂直接比大小。

配置真实CLI解析、shell语法及diff检查通过。既有 `test_cpu_offload_all_dense_distillation_losses_backward[kd_top_partial]`通过，1 passed。初次CPU参数预检曾在torch导入后才修改CUDA可见设备，造成Invalid device id；改为在进程启动前设置CUDA_VISIBLE_DEVICES后正常，没有修改训练代码或环境。

两组真实单卡短测于04:55启动（wrapper PID E3112417/F3112418），同B4/acc1/seq1024及完整路径，仅stop/save4与eval_limit1用于验证更新、保存、单进程评估和暂停。实测前四步约14–15秒，首次teacher阶段峰值allocated E41,835,846,144/F42,650,240,000 bytes；不能据四步代表性不足的耗时承诺正式总时长。两组04:57均exit0；CPU读取状态确认global_step=4、全部Adam状态step4、651个参数张量有限、253个LoRA B张量非零且有限、动量有效，八任务smoke原始指标齐全。保存、评估后恢复1953个optimizer张量及teacher、写回RNG并暂停均完成。有效组件日志decoder=0，可训练LoRA/norm/head符合冻结decoder配置。单卡路径验证已通过，没有新增效果结论。


正式E/F已从原0910独立启动，没有resume和eval_limit残留。wrapper PID E3119111/F3119112，真实worker E3119117/F3119118；独立输出时间目录均为 `Qwen_Qwen3-8B_20260924_045835`。05:01:38 UTC验收：两组均至少80次optimizer更新、数值有限，step80 LR约9.99654e-5；显存进程占用E50,678MiB/F53,060MiB。初期耗时只能描述已走过的数据，尚无全200步质量或总成本结果，不据此预测69。保存200后原八任务全量评估并暂停，结束后各自60GiB有限保留最多12小时。完成正式启动验收后停止主动监控，等待用户唤起复盘。

正式脚本SHA256：
- `proxy_b4_kl`：`591fb3cc6ca1c5c5e75978e657c9c2c0bb562875145526af07d151ed009a879c`。
- `proxy_b4_ce_kd`：`f6401d6e4fdcfa21ae0c43017bc7eb4188862cfa46a07ab04309162ae6c3f37f`。

## 200步训练段成本与续训准备（2026-09-24 05:04 UTC）

E/F已分别于05:04:24/05:04:27完成200步训练并开始八任务完整评估，limit=None。此时尚无完整评测结果，不据训练loss选择赢家。每组见800条样本，按20个token统计窗口的样本数和四舍五入均长计算，约177,499个有效prompt+response token；两组统计一致。原A/B400为12,800条、约2,872,477个有效token，预算相差约16倍，不能直接把两阶段质量当等预算比较。

按核心日志相邻更新时间计，E第10–200步290.813秒（1.531秒/更新）、F293.864秒（1.547秒/更新），单卡各约4.85/4.90 GPU分钟。A第10–400步4880.240秒、B5022.049秒，双卡各约162.67/167.40 GPU分钟。计时终点取带loss的step400更新日志，排除评估后同step的train_runtime汇总行。这些是明确步数窗口内的训练成本，不包括首10步、加载、保存和全量评测，不是完整实验耗时，也不证明小batch质量更好。阶段完成后再按同一口径计完整质量/成本。

续训前提已从当前源码核实：`v6_runtime_state.py:191–203`在eval_after_save=true时将save_strategy/save_steps纳入不可变契约，:262–270严格比较；因此E/F续训必须保持save_steps=200，不能临时稀疏评测。`runtime_v6_pipeline.py:187–207`允许提高stop_after_step，但须大于checkpoint步数、小于5000且是200的整倍数。固定原world1/B4/acc1、5000步调度及其余数学配置，选择候选后设置resume_from_checkpoint和新的暂停节点；当前恢复入口从checkpoint反推原run目录，继续在该实验下按step追加，run_root_dir不改变恢复输出；现有脚本没有追加参数转发，不能用bash原脚本加额外参数假装覆盖。评估后RNG须已写回且正式进程成功暂停，不能从仍在评估的checkpoint并发启动恢复。不预先创建尚未选定的续训脚本。

## E/F完整200步结果及下一轮预算（2026-09-24）

两组全八任务、0-shot、limit=None、相同样本数和指标键，均成功保存、评估后写回RNG并pause，wrapper exit0。E纯partial KD均分 **65.72570229610588%**（相对初始+0.34909832pp），F的0.5CE+0.5partial KD均分 **64.72739303367939%**（相对初始-0.64921094pp），F比E低0.99830926pp。原始JSON位于各自 `Qwen_Qwen3-8B_20260924_045835/lm_eval/lm_eval_results_step_200.json`。

| 任务 | E纯KD % | F混合 % | F−E pp |
|---|---:|---:|---:|
| boolq | 81.926606 | 82.415902 | +0.489297 |
| rte | 71.119134 | 66.425993 | -4.693141 |
| winogrande | 67.245462 | 67.640095 | +0.394633 |
| arc_easy | 76.683502 | 75.968013 | -0.715488 |
| arc_challenge | 49.317406 | 48.208191 | -1.109215 |
| openbookqa | 40.600000 | 39.200000 | -1.400000 |
| piqa | 76.550598 | 76.060936 | -0.489663 |
| mmlu | 62.362911 | 61.900014 | -0.462897 |

F相对E六项下降，其中RTE贡献均分差约0.5866pp，其余任务净贡献约0.4117pp；本轮不追加F预算。这个结果是当前CE权重、数据和800样本预算下的实际退化，不证明CE整体无效或任何方法的长程上限。F同时减半KD权重，不能单独归因于加入CE。保持F的原始指标、有效配置、复现脚本和核心日志，权重无明确续训/成品用途，按持续授权清理。

E/F wrapper于04:58:29 UTC启动，分别05:13:08/05:13:09退出，包含加载的单卡总耗时14.65/14.67分钟。Trainer记录的真实elapsed为840.595/840.9398秒（含保存/评估/恢复，不用max_steps推算的summary吞吐）；完整评测耗时占比较大。成本结论限于已完成阶段，不能据此宣称相同样本预算或相同质量下快几十倍。

下一步先取得下一次可用比较，不预先给E锁定到800：GPU6将E的checkpoint200按同轨迹续到400；GPU7从原0910跑G `proxy_b4_kl_lr3e4` 到200，只有普通LoRA LR改为3e-4，norm/head仍1e-4。G与E的200步是匹配预算的LR对照，不能直接与E400比排名；A/B与E/G之间的warmup、样本/步数预算不同，只能作为跨协议排序观察，不能单独归因于batch×LR交互。C继续既定400步，不修改其依赖。

实际入口在隔离worktree：
- E续训：`bash experiments/e2e_0910_search/run_proxy_b4_kl_stage400.sh`，wrapper PID3147785，GPU6，日志 `proxy_b4_kl_stage400.launch.log`，实际继续写入 `proxy_b4_kl/Qwen_Qwen3-8B_20260924_045835/`。源为E的checkpoint200，仍steps5000、warmup20、save200、B4acc1world1、原评估口径。脚本SHA256 `f01a0f6dbfb2f823ca0e668b5754cc3689ecc08d3d3f0ae04e647ba702ee705c`。
- G：`bash experiments/e2e_0910_search/run_proxy_b4_kl_lr3e4.sh`，wrapper PID3147786，GPU7，日志 `proxy_b4_kl_lr3e4.launch.log`，独立实际输出 `proxy_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_051735/`。脚本SHA256 `e54e8dab442f01baf1f6656c17dfb0193768bd4dc57006d7ee02e7911cda84ee`。

复用现有单卡真实更新/保存/评估短测与阶段恢复证据；真实CLI解析确认E只改续训输入/输出/暂停节点，G只改LoRA LR/输出/GPU，shell语法和diff检查通过，无源码或环境变更。初始参考JSON采用合并schema，0-shot/无限样本由其config.json核实；初次读取时错误假定有顶层limit字段而KeyError，已按实际schema改正读法，没有改动指标或放宽正式条件。

## C/E/G复核及本轮宽度筛选（2026-09-24）

本节替代上文尚未取得C400/E400/G200成绩时的预算判断。均分使用八个未舍入分数等权平均×100；boolq/rte/winogrande/mmlu取`acc,none`，其余四项取`acc_norm,none`。

| 分支 | 实际训练条件 | 步数 | 完整八任务均分 % | Trainer真实耗时，含保存/评估/恢复 |
|---|---|---:|---:|---:|
| A | B32，普通LoRA LR1e-4 | 400 | 65.87574545218463 | 5423.5323秒 |
| B | B32，普通LoRA LR3e-4 | 400 | 65.94692844283674 | 5559.2907秒 |
| C | B32，LoRA LR1e-4+decoder LR3e-6 | 400 | 65.7582228864665 | 16792.1431秒 |
| E | B4acc1，普通LoRA LR1e-4 | 200 | 65.72570229610588 | 840.595秒 |
| E | 同轨迹从200恢复 | 400 | 65.63160883838877 | 新增809.6334秒，累计1650.2284秒 |
| G | B4acc1，普通LoRA LR3e-4 | 200 | 65.61246374016139 | 821.8331秒 |

新增原始指标路径以结果根为基准：C为`decoder_lora_lr1e4/Qwen_Qwen3-8B_20260924_041131/lm_eval/lm_eval_results_step_400.json`；E为`proxy_b4_kl/Qwen_Qwen3-8B_20260924_045835/lm_eval/lm_eval_results_step_400.json`；G为`proxy_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_051735/lm_eval/lm_eval_results_step_200.json`。各run内的`normalized_e2e_runtime_args.json`和`compressed_e2e_fintuning.log`保存生效配置与核心证据。

C/E400/G200均完成saved/evaluated pause并由wrapper退出0，最新checkpoint存在；E200已由checkpoint400替代，不能再从200恢复。E200/E400/G200原始JSON直接确认`num_fewshot=0`、`limit=null`，64个实际子任务（含57个MMLU学科）共24742个有效样本，`n_shot`全0且各项`effective=original`。A/B/C旧JSON不含这些顶层metadata，完整0-shot/无限样本依据生效配置`canonical_config.runtime.evaluation`及日志，八项指标齐全；不能把缺失字段写成已从JSON逐项验证。

- **C暂不晋级。** C与A的配置除输出位置外仅`train_mode`变化，均分低0.11752256571814154pp；MMLU低0.66229882pp、ARC-C高0.59726962pp。第10→400步分别13998.498/4880.240秒，C纯训练约2.87倍，含评估总Trainer时间约3.10倍。C一次评估及训练状态恢复约40.3分钟，A约6.6分钟。当前配置没有体现足以补偿成本的收益，因此本轮不追加预算；这不是decoder路线的长程上限结论，也不是裁剪比例导致收益差的已证实因果。
- **E/G均保留到匹配800步再比较。** G200比E200低0.11323855594449692pp，其中RTE低1.80505415pp，对八任务均分贡献约-0.22563pp；其余七项平均反而高0.12844939pp。不能据这个小差距确定1e-4优胜或淘汰G。E200→400低0.09409345771710931pp：winogrande/ARC-E/openbookqa分别下降0.6314/0.4630/0.6000pp，MMLU提高0.2136pp；它是短程平台/波动证据，不足证明最终上限。
- **跨batch排序尚未校准。** 大batch的B−A为+0.07118299pp，小batch的G200−E200为-0.11323856pp；差距都小，且预热、样本曝光和步数不同，不能宣称已证实排序反转，也不能将小batch优胜直接迁移到大batch。

成本取实际日志窗口：E10→200为290.813秒，E210→400为277.376秒，G10→200为289.117秒，约1.46–1.53秒/更新；每200步的完整评估及恢复约8.4分钟。E初段/续段与G200的wrapper全程分别14.65/14.37/14.52分钟，包括加载。按已有成本估计E400→800约28分钟、G200→800约42分钟，分别含2/3次完整评估；这不是新任务完成时间承诺。保持原save200不可变恢复契约，不以临时稀疏评估改变RNG轨迹。参考[对照设计经验](../../lessons/experiment_design.md)与[阶段恢复约束](../../lessons/checkpoint_lifecycle.md)。

本轮确定执行如下；表内状态是提交/验证状态，不将PID存在视为主循环推进证据：

| GPU | 配置与预算 | worktree下入口 | wrapper PID | 当前阶段 |
|---|---|---|---:|---|
| 6 | E400→800，同一原轨迹，save200 | `experiments/e2e_0910_search/run_proxy_b4_kl_stage800.sh` | 3471380 | nohup已提交，主任务核实推进中 |
| 7 | G200→800，同一原轨迹，save200 | `experiments/e2e_0910_search/run_proxy_b4_kl_lr3e4_stage800.sh` | 3471381 | nohup已提交，主任务核实推进中 |
| 4 | H：`kl_top_mass` K100，从0910到400，200/400全评 | `experiments/e2e_0910_search/run_proxy_b4_mass.sh` | 3474890（短测） | 真实4步短测中，正式未启动 |
| 5 | I：`kl_top_partial` K1000，从0910到400，200/400全评 | `experiments/e2e_0910_search/run_proxy_b4_k1000.sh` | 3474891（短测） | 真实4步短测中，正式未启动 |

H/I除所列变量外沿用E初始条件：原0910、rank8/alpha16/dropout0.1、LoRA与norm/head LR1e-4、冻结decoder、B4acc1world1、原数据/seed/seq1024/prompt0.3、warmup20和5000步调度。H相对E检验tail桶目标，I相对E检验K100→1000；H与I之间同时改变目标与K，不能把二者差异单独归因于K。历史没有受控top1000优于top100证据，本次是待检验假设。4步短测用于真实更新、保存、评估后恢复/暂停验证，使用独立输出与eval_limit1；正式从原0910重建，恢复全量八任务0-shot，满足验证条件才启动。

E/G正式脚本的真实CLI解析与token差分通过，仅恢复输入和暂停节点改变。H/I短测初次误将内部字段`eval_lm_limit`用作CLI参数，解析阶段exit2、模型尚未初始化；依据`train_utils/config/cli.py:537`修正为`--eval_limit 1`，重新解析确认mass/K100与partial/K1000、B4、stop/save4后，用表内新wrapper重新启动。没有修改训练实现或正式数学配置。现有CPU offload mass/partial反向测试2 passed、25 deselected；GPU短测结果仍待主任务验收，不将CPU通过当作GPU运行通过。

以上选择用两张卡延长LR对照、两张卡扩展目标/K的宽度筛选，暂不追加昂贵C；不会在任何终点无条件运行占卡脚本。本次只更新记录和经验，不删除权重，产物去留由主任务结合实际后续用途收尾。

## 实际task数据版本核查与中间结果（2026-09-24）

本节修正此前把当前生成器的MMLU auxiliary_train设计等同于实际训练文件来源的表述。只读核查沿`e2e_common/data.py:220`的alias定位到主源`data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl`，实际82021行、26119691 bytes，SHA256为`f42c3c51405472c4f63026dadc11a2a7a5aa7ef3891c109d37472dbd356383d0`。每条仅含messages，没有逐行source或生成manifest；本次没有改动该文件、运行alias或活动任务配置。

实际文件前76755行的任务分段条数符合七任务train规模，但不能称内容全部正确：后续逐条核查发现其中2490条RTE全为空题干且答案颠倒（具体证据见下）；其余七任务部分不应与这些已确认错误混为“全部匹配”。随后1816条具有57个MMLU学科、四选项及单字母答案结构，尾3450条为其他格式。历史`a22cbd2:tools/prepare_vaellm_task_mix.py:244–266`采用MMLU dev+validation，:324–328随后追加LongBench；到`e4a7547`（2026-09-20）才改为auxiliary_train并删除LongBench。文件分段强烈吻合旧生成器，但尚未逐条认证后1816/3450条的来源，不能把结构证据升级为逐样本来源证明。当前MMLU评估使用test；现有证据既不能支持“实际训练已确认auxiliary_train”，也不能据此宣称已证明测试泄漏。后续需以实际artifact身份和来源验证，不能只读最新生成器。

独立新数据版本已完成生成与manifest验收，保持上述旧文件及既有实验输入不变。后续采用相同初始模型、更新/样本预算和超参数下仅data-version变化的对照，训练实际启动状态另以正式交付为准；不能把数据生成完成当成训练完成。新版相对旧版同时修正RTE、改用MMLU auxiliary_train且不再附加旧版尾部，因此是数据版本整体对照，不是MMLU单因素实验。暂不把旧task权重提高到0.9，避免同时混改文件与采样比例。参考[数据版本与统计边界经验](../../lessons/kd_and_data.md)。

主任务新增的运行快照证据如下，两份JSON均已核实完整八任务、0-shot、无样本上限；这是运行中检查点结果，不代表800步任务完成：

| 分支 | 步数 | 八任务均分 % | 原始指标（结果根下） |
|---|---:|---:|---|
| E | 600 | 65.76912817173593 | `proxy_b4_kl/Qwen_Qwen3-8B_20260924_045835/lm_eval/lm_eval_results_step_600.json` |
| G | 400 | 65.90086076402191 | `proxy_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_051735/lm_eval/lm_eval_results_step_400.json` |

上述较早快照中四个正式worker均仍存活，H/I当时正在200步全评；这一运行状态现已被下方E800/H400/I400完成证据替代，不能继续当作当前状态。E600比E400高0.13751933pp，G400比匹配E400高0.26925193pp；E600与G400步数不同，不能用两者直接判LR胜负，仍按既定800步匹配终点复盘，不由本次快照修改运行配置。

E/G各自前400步均见1600条样本，日志聚合为prompt126984、response219326 token。后来核实`DistillTokenStatsAccumulator`统计原始labels/attention_mask、尚未做causal shift，因此`0.3×126984/(0.3×126984+219326)=14.7988%`只能标为未shift日志口径的粗统计，不能冒充实际KD因果分母份额。若这1600条的首token均为prompt，则需用prompt126984−1600，条件计算为`0.3×125384/(0.3×125384+219326)=14.6396%`；本次没有直接重放旧E/G证明该前提，故不将条件值写成精确实测。两种统计均不是loss/gradient占比或task数据贡献；不能据此推断提高task采样权重一定有效。实际source级核对与causal分母统计见下方新数据审计。

## E800/H400/I400完成结果与实际RTE错误（2026-09-24）

当次只读核实E/H/I三组核心日志、原始评估JSON及wrapper退出记录时，G800尚未完成；其随后完成结果已补在下方。E在10:11:14 UTC、H在10:15:43、I在10:14:30分别wrapper exit0，均已成功saved/evaluated pause，最新checkpoint800/400/400在核查时存在。所有下列JSON为原八任务完整0-shot、limit=None，64个子任务共24742样本，effective=original、n_shot全0，指标键符合既定口径。

| 分支/变量 | 步数 | 八任务未舍入均分 % | 相对同预算E的差值 pp |
|---|---:|---:|---:|
| E partial K100 | 800 | 65.7845923059528 | 不与400步H/I直接排名 |
| H mass K100 | 200 | 65.64120092230381 | -0.08450137380206918 |
| H mass K100 | 400 | 65.63604824312159 | +0.004439404732821395 |
| I partial K1000 | 200 | 65.7510425317455 | +0.025340235639615544 |
| I partial K1000 | 400 | 65.69960003022958 | +0.06799119184080216 |

原始指标位于结果根下E原run的`lm_eval/lm_eval_results_step_800.json`、`proxy_b4_mass/Qwen_Qwen3-8B_20260924_094654/lm_eval/lm_eval_results_step_{200,400}.json`、`proxy_b4_k1000/Qwen_Qwen3-8B_20260924_094654/lm_eval/lm_eval_results_step_{200,400}.json`。完整分项只以这些JSON为主源；相对E400的变化如下，E800列描述同一轨迹加预算后的变化，H/I列描述同400步对照：

| 任务 | E800−E400 pp | H400−E400 pp | I400−E400 pp |
|---|---:|---:|---:|
| boolq | -0.21406728 | +0.15290520 | +0.09174312 |
| rte | +1.08303249 | 0 | 0 |
| winogrande | -0.47355959 | +0.23677979 | +0.78926598 |
| arc_easy | +0.25252525 | +0.08417508 | +0.33670034 |
| arc_challenge | +0.08532423 | -0.25597270 | 0 |
| openbookqa | +0.60000000 | -0.40000000 | -0.60000000 |
| piqa | +0.05440696 | +0.21762786 | +0.05440696 |
| mmlu | -0.16379433 | 0 | -0.12818687 |

E800比E400高0.15298346756403777pp，其中RTE贡献约0.13538pp，MMLU反而下降；E600→800仅+0.01546413pp。H400几乎等于E400；I400小幅高0.06799pp但任务有升有降。这些单次、单预算结果尚不支持mass或K1000显著提升，也不能证明任何方法的长程上限。特别是在实际训练数据已发现错误后，当前效果只属于该旧数据版本，不应据此永久淘汰其他数据条件下的目标函数或CE。

成本按实际墙钟：E400→800的Trainer真实runtime1726.595秒（累计0→800为3376.8234秒）；H0→400为1709.0768秒，I0→400为1635.4694秒，均含保存、两次完整评估及恢复。对应本次wrapper从START至exit分别1812秒（30.20分钟）、1735秒（28.92分钟）、1662秒（27.70分钟），含加载，不含段间等待。纯训练窗口E410→600/610→800分别312.023/319.764秒，H10→200/210→400为300.823/289.718秒，I为292.657/276.936秒。不能用HF按max_steps推算的summary吞吐替代这些实测，也不能将E累计800步成本与H/I400步当成同预算。

### 已确认的RTE错误与修正后数据身份

独立只读审计逐条比较旧、新JSONL的1-based第74266–76755行，并直接读取官方缓存`/home/shaoyuantian/.cache/huggingface/datasets/super_glue/rte/0.0.0/3de24cf8022e94f4ee4b9d55a6f539891524d646/super_glue-train.arrow`及同目录`dataset_info.json`。2490条旧RTE的prompt全部相同，为`\nQuestion:  True or False?\nAnswer:`，没有premise/hypothesis；新版2490条题干均与官方train原始字段逐条一致。标签names为`['entailment','not_entailment']`，因此0→True、1→False；旧答案True→新版False共1241条，旧False→新版True共1249条，2490条全部颠倒，新版全正确。

历史commit `7ae658c9d87fc18c31af1c6ccdfb9d3d9ecc45e0`（2026-09-17）已把sentence1/sentence2修为premise/hypothesis，并把choices `[False,True]`改为`[True,False]`；当前`tools/prepare_vaellm_task_mix.py:181–187`正确，但旧JSONL没有随生成器修复重新生成。这是实际数据内容错误的直接证据，不再只是来源不明的怀疑；尚未实测修正对下游成绩的因果贡献，尤其不能把F的CE退化全部归因于RTE。

主任务在独立目录`data/edgerazor_qwen3/task_train_v2_20260924/`生成新文件，manifest已只读复核：

- 输出`task_vaellm_eval_instruct.jsonl`为176597行、191870906 bytes，输出文件SHA256 `894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1`；此SHA是数据文件身份，不是manifest文件SHA。
- 七任务76755条：arc_e2251、arc_c1119、boolq9427、piqa16113、winogrande40398、openbookqa4957、rte2490；加MMLU auxiliary_train99842条。与旧文件前76755条相比2490条messages不一致，正好是经独立审计确认修复的RTE。
- MMLU来源`cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe`，`auxiliary_train/train-00000-of-00001.parquet`，47518592 bytes、99842行，SHA256 `ae6662576ef989fed82a9289da1b89e950499c320f6849b052ec344dfcb709eb`。`split=train`指该auxiliary_train文件的本地加载split，不能混同test。
- 生成器SHA256 `ccf028621004b0ab1b4bb2f0ba6fcd2dfb8085457ffd8760a28aa452420bb8da`，formatter SHA256 `a066c01ac40dc584d027f5ccd393ab3f164802fa05ece298030f980cfa9cb83e`。manifest记载8条生成短测与全量生成均exit0、所有messages有效、MMLU答案均ABCD且条数等于官方输入；这些全量生成事实由主任务验证，独立审计另外验证RTE2490条，不混写验证职责。

七任务源revision、旧文件身份、完整命令及验证项以`data/edgerazor_qwen3/task_train_v2_20260924/manifest.json`为主源。旧文件及其运行输入保持不动。本次对旧尾部LongBench的来源仍只有结构和历史生成器吻合，未做逐条源匹配。新版整体同时修复RTE、替换MMLU训练来源/数量、去掉旧尾部，后续任何效果不能单独归因MMLU；先固定task混合权重和其他超参数比较数据版本，必要时再拆分因素。经验已合并[数据文件核查](../../lessons/kd_and_data.md)。本次文档工作没有改动任务、脚本或删除权重。

## G800收尾与新数据四格准备（2026-09-24）

G原始结果目录仍为`proxy_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_051735/`，`lm_eval/lm_eval_results_step_600.json`、`lm_eval_results_step_800.json`经独立只读核对：完整原八任务、0-shot、limit=None，指标键正确，七任务effective分别3270/277/1267/2376/1172/500/1838，57科MMLU共14042，未缩样本。

| G检查点 | 未舍入均分 % | 结论范围 |
|---|---:|---|
| 400 | 65.90086076402191 | 本支已观察到的最佳点 |
| 600 | 65.65847003270426 | 较400回落 |
| 800 | 65.56552801004902 | 比匹配E800低0.21906430pp |

G800分项%依boolq/rte/winogrande/arc_easy/arc_challenge/openbookqa/piqa/mmlu顺序为82.78287462/71.48014440/66.29834254/76.89393939/48.80546075/40.0/76.44178455/61.82167782。最后step800普通LoRA LR为0.0002822491388，仍约初始3e-4的94.08%，不能将回落解释为“到本段终点余弦已降到零”。这是旧数据版本下本支轨迹的观测，不证明单一LR或方法上限，也不单独归因旧RTE错误。

G的stage800 wrapper从09:41:02 UTC到10:24:24退出0，续训600步含加载/保存/三次全评共2602秒、单卡43.37分钟；核心日志成功saved/evaluated后paused，wrapper3471381已退出，checkpoint800在核实时存在。210→400、410→600、610→800各190步纯训练窗口286.746/313.993/311.354秒；三次评估与恢复513.451/513.002/515.627秒。后续权重去留由主任务按实际用途收尾，本节不预写已删除。

### 旧数据分支已完成的产物清理

主任务已删除A/B/E/H/I五个trainer_state、E/H/I三份已结束launch日志和8份重复summary表，实际释放**1,409,540,096 allocated bytes**。删除前所有目标均在本结果根内、无符号链接及活跃进程argv/fd引用；A/B没有脚本恢复到其checkpoint，所谓未来晋级/大batch对照只剩旧计划，没有当前具体执行用途。该用途已随转向新数据四格撤销，不影响历史精度记录。

五个被删除trainer_state分别属于`lora_lr1e4/Qwen_Qwen3-8B_20260924_023506`、`lora_lr3e4/Qwen_Qwen3-8B_20260924_023506`、`proxy_b4_kl/Qwen_Qwen3-8B_20260924_045835`、`proxy_b4_mass/Qwen_Qwen3-8B_20260924_094654`、`proxy_b4_k1000/Qwen_Qwen3-8B_20260924_094654`。E/H/I的START/FORMAL_EXIT已经归并各自核心日志，删除的是重复进度条启动日志；原始指标、normalized配置、核心日志、复现入口、旧数据和初始0910均保留。这次没有删除G；不把尚未执行的G清理计入释放量。

### J/K/L/M设计与验证状态

四格均使用新alias `vaellm_task_train_v2`，固定同一新数据版本、原0910初始checkpoint、rank8、decoder冻结、B4acc1world1、seq1024、seed/data_seed0、prompt0.3、norm/head LR1e-4、5000步调度与warmup20。均fresh from0910，正式stop400、save/eval200、原八任务全量0-shot。以下是准备中的明确配置，不是已启动声明：

| 配置 | GPU | task样本混合权重 | 普通LoRA LR | 损失 | 与J的研究变量 |
|---|---:|---:|---:|---|---|
| J `task_v2_b4_kl` | 4 | 0.5 | 1e-4 | `kl_top_partial` K100 | 新数据基准；与旧E比较数据版本整体 |
| K `task_v2_b4_kl_lr3e4` | 5 | 0.5 | 3e-4 | `kl_top_partial` K100 | 仅普通LoRA LR |
| L `task_v2_b4_task80` | 6 | 0.8 | 1e-4 | `kl_top_partial` K100 | 仅数据源混合权重 |
| M `task_v2_b4_ce10` | 7 | 0.5 | 1e-4 | `kd_top_partial` K100、alpha0.9，即0.1CE+0.9KD | 仅CE/KD混合配方 |

脚本为worktree中`experiments/e2e_0910_search/run_task_v2_b4_kl.sh`、`run_task_v2_b4_kl_lr3e4.sh`、`run_task_v2_b4_task80.sh`、`run_task_v2_b4_ce10.sh`。J相对E的0.5混合仅替换task alias；K/L/M分别以J为对照，不把L的样本权重0.8解释为token或gradient贡献80%，M同时降低KD权重，不能只归因为CE存在。旧F使用0.5CE且旧数据含错误RTE，与新M的0.1CE配方和数据都不同，因此重试有明确改变的条件。

训练数据入口的源码改动仅为worktree `e2e_common/data.py`新增alias，当前SHA256 `098fe413d1478f5db04db81d37694cbe2f22a40866f6a77ef68ca1d4bb82320c`；旧alias保持原路径，不原地替换旧文件。主任务已通过真实loader核对旧82021/新176597条与正确路径，生成器仍为前文`ccf028...`版本。CPU真实tokenizer的RTE、MMLU及长输入截断检查通过，表明所检样本在实际编码路径保持有效题干/回答边界；不将这一CPU检查写作GPU短测通过或下游增益。J/M两个4步真实模型短测已通过更新、保存、八任务limit1评估、teacher/optimizer恢复、RNG写回与暂停验收，10:27:53/54 UTC分别exit0；四格随后使用正式400步、全量评估配置启动，未沿用短测权重。

本轮仍保留原ChatML训练模板，尚未把训练格式改为lm-eval裸题格式。数据内容修复与模板对齐是不同问题；模板差异目前只是待验证因素，不能宣称已消除或已证明导致精度差，也不在本轮同时改模板破坏比较。

### 新数据四格的验证与后台交付（2026-09-24 10:31 UTC）

短测仅J纯KD与M轻CE两条实际损失路径，K的LR和L的数据权重为既有标量参数，复用适用证据，不逐配置重复GPU短测。两组真实B4、seq1024、原0910、原精度/单卡路径，stop/save4、eval_limit1；各651个可变张量有限、253个LoRA B非零，651组Adam step4及moment有限非零，scheduler.last_epoch4、八项指标齐全、Python/NumPy/CPU/CUDA RNG均保存。复现为对应正式脚本改独立verify输出、stop/save4、eval_limit1；核心证据已归并verify_task_v2_b4_kl和verify_task_v2_b4_ce10的Qwen_Qwen3-8B_20260924_102605目录。

CPU实际Qwen3 tokenizer revision b968826d9c46dd6066d109eabc6255188de91218：RTE第74266行42tokens、MMLU第76756行128tokens，各8个响应目标，首响应分别在位置34/120、对应logits33/119，答案与mask正确；真实长MMLU第94860行1380tokens，首响应1373，1024截断后返回None，不会仅用EOS训练。真实alias loader确认新176597/旧82021行且各自路径正确。此检查不证明训练模板与裸题评估完全一致。

| GPU | 入口名（experiments/e2e_0910_search/） | wrapper / Python主进程PID | 启动验收进度 |
|---|---|---|---:|
| 4 | run_task_v2_b4_kl.sh | 3541535 / 3541550 | 20 |
| 5 | run_task_v2_b4_kl_lr3e4.sh | 3541536 / 3541546 | 20 |
| 6 | run_task_v2_b4_task80.sh | 3541537 / 3541548 | 30 |
| 7 | run_task_v2_b4_ce10.sh | 3541538 / 3541549 | 20 |

在已激活bitvae的worktree执行nohup bash调用各脚本，标准输入关闭、stdout/stderr写结果根的同名task_v2_*.launch.log，退出写FORMAL_EXIT。各自输出目录为结果根下task_v2_b4_kl、task_v2_b4_kl_lr3e4、task_v2_b4_task80、task_v2_b4_ce10 / Qwen_Qwen3-8B_20260924_102952；核心日志compressed_e2e_fintuning.log，实际配置normalized_e2e_runtime_args.json。无eval_limit、无smoke resume；save/eval200、stop400、steps5000，无重试/结束后占卡。验收已观察有限loss/grad_norm和真实更新推进，随后停止主动监控。

正式入口SHA256分别为J=5170604f329b959cff8862f6e0c8031f1d60569b7e6ccfcfbb9b80dd4bb52e96，K=e1d6dfb2890cb48aa7b4a52e15d0cc0d600cd89e2ab8e5b2830ffb52c2979a27，L=c40913d3f26b12a910bc2207a4fa291621073ffa0eb0cee380488a67bbfd1bf1，M=464a152b984563abaa1d62ab947188b272087ca39fb997c2fa61af0ccec083df。四份bash语法及逐参数检查通过；数据alias与生成器diff检查通过，其余训练源码沿用已验证版本。

本轮后续清理G trainer_state、J/M短测trainer_state、重复表/启动日志及已用完的数据生成和状态检查临时文件，实际释放845799424 bytes，生命周期与短测证据已归并核心日志。加上前述A/B/E/H/I清理，两次共2255339520 bytes（约2.100GiB）；不是共享磁盘总量变化。G800权重没有具体续训/复评用途，已删除；完整指标与配置受保护。原始0910、新版训练数据/官方Parquet输入正在被后续工作使用，保留。Windows官方Parquet传输副本47,518,592 bytes的删除被自动审批审查拒绝，仅返回“blocked by policy”，暂保留，未绕过或冒称已清理。

## J/K/L/M的400步完成结果（2026-09-24）

独立只读核实四组`lm_eval_results_step_400.json`及200步对应文件、核心日志与wrapper退出记录：原八任务齐全且有限、指标键按既定协议，num_fewshot=0、limit=None，64个子任务共24742有效样本，effective=original、n_shot全0。四组均完成saved/evaluated pause、wrapper exit0，核查时checkpoint400存在。以下事实只认证当前400步终点，不表示5000步轨迹已完成或69+达成。

| 配置 | 200步均分 % | 400步均分 % | 400−200 pp | 400−初始0910 pp | 400−旧B400 pp |
|---|---:|---:|---:|---:|---:|
| J：task0.5、LR1e-4、纯KD | 65.28540244067685 | 65.404405410506 | +0.11900297 | +0.02780144 | -0.54252303 |
| K：普通LoRA LR3e-4 | 64.99758264463291 | 65.61459093822081 | +0.61700829 | +0.23798696 | -0.33233750 |
| L：task权重0.8 | 64.96636248046835 | 64.982715182728 | +0.01635270 | -0.39388879 | -0.96421326 |
| M：0.1CE+0.9KD | 65.08536328328961 | 65.31884429325547 | +0.23348101 | -0.05775968 | -0.62808415 |

初始值65.37660397423905%、旧B400值65.94692844283674%，均分均由原始八项未舍入值等权计算。旧B的数据版本、B32、warmup150及样本曝光与本轮B4/warmup20不同，表中差值仅是同口径评估成绩参照，不能把它当作单因素数据改动收益。J与旧E400的65.63160883838877%比较为-0.22720343pp，虽训练配置更接近，仍是RTE修复/MMLU替换/旧尾部移除的版本整体对照。

400步分项如下（单位%）：

| 任务 | J | K | L | M |
|---|---:|---:|---:|---:|
| boolq | 81.71253823 | 82.50764526 | 82.53822630 | 81.65137615 |
| rte | 71.84115523 | 70.03610108 | 68.23104693 | 68.95306859 |
| winogrande | 67.00868193 | 67.40331492 | 66.77190213 | 67.40331492 |
| arc_easy | 76.09427609 | 76.93602694 | 76.43097643 | 76.55723906 |
| arc_challenge | 48.97610922 | 49.23208191 | 49.23208191 | 49.23208191 |
| openbookqa | 39.20000000 | 40.00000000 | 38.40000000 | 39.80000000 |
| piqa | 76.06093580 | 76.65941240 | 76.11534276 | 76.60500544 |
| mmlu | 62.34154679 | 62.14214499 | 62.14214499 | 62.34866828 |

200→400变化中，K的六项上升、ARC-Challenge下降0.3413pp，其他主要上涨为RTE+1.8051、winogrande+1.2628、openbookqa+1.2pp，MMLU仅+0.1068pp；这支持它作为本轮优先增加有限预算的候选，而非已接近69。J的上涨主要由RTE+1.0830与winogrande+0.4736抵消其他下降。L仅+0.01635pp且仍低初始0.39389pp，没有显示提高整个task比例的收益。M较J均分低0.08556pp，但RTE低2.88809pp，其余七项平均反而高0.31480pp；不能根据微小总差宣布CE整体无效，也不能临时排除RTE改变主指标。

**已确定续训与权重用途：** J/L/M本轮不续训，只保留原始指标、有效配置、核心日志与复现入口；主任务已删除J/L/M的`trainer_state`；四组wrapper的START/FORMAL_EXIT生命周期记录先并入各自核心日志，再删除四份launch.log；各200/400重复summary.md也已删除。保留配置、核心日志、原始JSON与复现入口；本次实际释放846770176 allocated bytes（846648020 logical bytes），不计他人或共享磁盘变化。J的最小证据继续作为N/O/P包装与采样对照；这些新对照从0910开始，不依赖J权重。M分项信号保留在记录，不据此将CE整体判为无效。

K保留唯一`task_v2_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_102952/trainer_state/checkpoint-400`，具体用途已安排为N/O/P/Q这一轮后在GPU5按同一数据/目标函数/优化器/调度/评估间隔配置续至800，验证200→400的回升能否持续。主任务已新增续训入口`run_task_v2_b4_kl_lr3e4_stage800.sh`，SHA256为`92b8ec831afb1a7651b00da48863451dddd629e308424277d4ee3bc63fffd4ca`，GPU5、同一数据与5000步调度、resume400→stop800。该续训尚未启动，拟N/O/P/Q之后接续；保留是为这一明确任务，不是“以后可能有用”。K400比旧B400低0.33233750pp，但数据、batch、预热和样本曝光不同，仍不能作单因素归因，也不据此承诺K800会达69。

本轮修正了实际数据内容，但**采样与模板尚未同评估目标全面对齐**：task仍按行数混合、ARC覆盖较少；ChatML与裸题评估不同，MMLU auxiliary_train缺subject，不能杜撰description。结果不支持“正确数据必然提高均分”，也不能推断数据修复无价值或模板/均衡抽样一定有效。后续对照应保持已安排的独立变量，而非因400步未达69同时改变多处。

原始结果根为`/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/`。J/K/L/M实际子目录分别为`task_v2_b4_kl`、`task_v2_b4_kl_lr3e4`、`task_v2_b4_task80`、`task_v2_b4_ce10`下的`Qwen_Qwen3-8B_20260924_102952`；各自`lm_eval/lm_eval_results_step_400.json`为完整分项主源，`normalized_e2e_runtime_args.json`与`compressed_e2e_fintuning.log`保留实际配置/过程。四份复现脚本沿用上文已验证入口。

| 配置 | wrapper退出UTC / code | Trainer真实runtime，含评估 | wrapper全程 |
|---|---|---:|---:|
| J | 11:00:15 / 0 | 1765.976秒 | 1830秒 |
| K | 10:59:54 / 0 | 1745.0636秒 | 1809秒 |
| L | 10:58:23 / 0 | 1655.2077秒 | 1718秒 |
| M | 10:59:33 / 0 | 1724.9131秒 | 1788秒 |

四个wrapper均从10:29:45 UTC开始，耗时是对应单卡实际阶段成本，不采用HF按max_steps计算的summary吞吐。本段核查只读评估与运行证据，另据主任务已执行的清理结果补录上述释放量及保留用途；K续训和下一轮实际启动状态由主任务后续补录。

## J/K/L/M的200步中间结果（2026-09-24 10:46:58 UTC）

主任务已从四份原始JSON核实完整八任务、0-shot、limit=None、24742有效样本，未改变评估口径；这是200步检查点，四组仍按原定400步终点执行，不能写为最终实验已完成。本节复用该次核验，不另行轮询。

| 配置 | 200步八任务均分 % | 与同新版J差值 pp |
|---|---:|---:|
| J：task0.5、LR1e-4、纯KD | 65.28540244067685 | 0 |
| K：仅普通LoRA LR3e-4 | 64.99758264463291 | -0.28781980 |
| L：仅task混合权重0.8 | 64.96636248046835 | -0.31903996 |
| M：0.1CE+0.9KD | 65.08536328328961 | -0.20003916 |

各原始指标为对应`task_v2_b4_kl`、`task_v2_b4_kl_lr3e4`、`task_v2_b4_task80`、`task_v2_b4_ce10`下`Qwen_Qwen3-8B_20260924_102952/lm_eval/lm_eval_results_step_200.json`。J比旧E200的65.72570229610588%低0.44029986pp、比本轮初始65.37660397423905%低0.09120153pp；这是旧/新数据版本整体的200步观测，不能单独归因MMLU增加或RTE修复，也不证明更长预算下上限。K/L/M与J在本轮对应变量的匹配条件下均未显现均分优势，先保留既定400步检查点，不由这次中间结果擅自停止或改动运行。

结合source级token归因与模板审计，后续准备优先研究模板和答案监督分配，而非继续提高task样本比例。候选模块当前仅在独立文件中做CPU验证，尚未接入正式训练，没有新的效果证据；具体采用与否仍取决于验证和当前实验结果。

## 新数据source级token审计与模板边界（2026-09-24）

只读CPU诊断复用当前`_IndexedMixedRawStream`、真实permutation、canonical encoder、collator及token统计器，仅为诊断副本附加source/raw行号，不改正式数据、训练代码或运行状态。使用Qwen3 tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218`、seed0、B4、seq1024、world1；真实encoder返回None的样本按生产逻辑跳过，不用raw draw比例代替实际可训练样本。CPU退出0、CUDA未初始化。

J前40条重放prompt7559/response4758，精确匹配实际step10日志。进一步读取已写入的训练token telemetry而非评估输出，J/L各前800条的20个10步窗口全部匹配、0处不一致。完整1600条统计来自同一配置与迭代器语义的确定性重放，后800条在此次审计时尚无实际训练日志逐窗覆盖；不能写成400步已完成或1600条全部已经实测。这个证据比未编码的raw draw估计更强，但仍明确区分已匹配前缀与未观察到的后续。

normalized快照中的`dataloader_num_workers=0`和`group_by_length=true`是入口后续调整前的字段。`runtime_v6_pipeline.py:713–725`将workers≤0替换为`default_dataloader_num_workers()`，当前`e2e_common/lazy_datasets.py`返回16；iterable路径关闭group_by_length，并设dispatch_batches=false、split_batches=false。诊断据此实际使用16-worker DataLoader、drop_last=false，保持batch发出顺序；不能把normalized中的0当成正式实际单进程取数。以上来自对应源码路径与前缀token吻合，未以读取进程列表冒充逐worker追踪。

下表为**已匹配实际日志的前800条**。有效prompt/response按causal shift去掉每条首token后统计，分母为`0.3×有效prompt+有效response`的累计值；与未shift的日志计数分开：

| source | J样本数 | J有效prompt / response | J累计分母份额 % | L样本数 | L有效prompt / response | L累计分母份额 % |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 800 | 143965 / 109665 | 100 | 800 | 165239 / 54301 | 100 |
| task合计 | 420 | 95463 / 4720 | 21.823957 | 645 | 144930 / 7298 | 48.883874 |
| 其中MMLU | 253 | 88057 / 2024 | 18.606649 | 386 | 133989 / 3088 | 41.670911 |
| II7m | 247 | 28690 / 42603 | 33.502448 | 91 | 9912 / 16892 | 19.124948 |
| IIgen | 57 | 5891 / 32644 | 22.512455 | 19 | 2153 / 11229 | 11.432166 |
| Tulu | 29 | 6926 / 9047 | 7.278032 | 12 | 3237 / 3210 | 4.025215 |
| AM | 47 | 6995 / 20651 | 14.883108 | 33 | 5007 / 15672 | 16.533796 |

task合计包含MMLU，不能把这两行相加。J/L未shift的800条总计分别为prompt144765/response109665和prompt166039/response54301；确认这些样本首token都是prompt，causal prompt各减800。J/L累计因果分母分别152854.5/103872.7。配置task样本权重0.5/0.8并不等于分母份额50%/80%：当前前缀中非task长回答语料分别占累计分母78.176043%/51.116126%，说明样本权重被回答长度重新加权。它仍不是实际loss或gradient份额，也不等于逐batch归约再平均后的训练贡献；本诊断没有测量梯度或下游因果效果。

1600条**确定性重放**给出同方向观察：J/L的task样本828/1276，causal分母份额21.724186%/50.528893%；MMLU样本479/737，份额18.336085%/42.861373%。1600条总有效prompt/response为J273345/216317、L330915/101664，累计因果分母298320.5/200938.5。各source详细计数、未shift/shift统计、40个窗口、normalized配置SHA与运行参数保存在唯一[原始统计JSON](../../../../result/compressed_e2e_fintuning/e2e_0910_search_20260924/training_source_token_stats.json)，25315 bytes、SHA256 `f84dacaa1ffaae28ec598d41218d98f3d86a012dd12af9f4c8bcbe0102d9eda6`；归档前确认目标不存在，逐字节复制CPU结果，没有重写或舍入原始值。临时脚本/日志仍待主任务统一归并清理，本次不删除。

### 已核实的模板差异，不是效果结论

独立只读审计使用实际新版JSON、当前生成器与已安装lm-eval0.4.8源码，没有读取test题或改变训练/评估模板：

- 新版JSON第76756–176597行共99842条MMLU逐条统计均以`Question:`开始、0条`Subject:`、答案均为空格加A/B/C/D。`e2e_common/data.py:473–510`只能在来源含subject时生成可选Subject；auxiliary_train没有subject，不能假造评估学科description。安装包`tasks/mmlu/default/_default_template_yaml:7`使用裸题干与ABCD，各subject YAML另有英文description，`api/task.py:1066–1087`确认0-shot仍附description；`eval_utils.py:694–700`未启用chat，`evaluator.py:68`默认False。训练ChatML与该评估格式确实不同，但未实测对齐格式能否提高成绩。
- Winogrande仅用官方train缓存`winogrande_xl/0.0.0/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande-train.arrow`40398条与安装包`tasks/winogrande/preprocess_winogrande.py`真实函数逐条比较：目标索引与continuation suffix全部相同；606条context差异仅为空白处前2空格603条、3空格3条，被该函数:32–39的`rstrip`加单空格规范化。没有发现配对错误，不能将606条空格差异写成606条标签/内容错误。
- Qwen3 tokenizer该revision的`tokenizer_config.json`内chat_template第45行给最终assistant加入空think块；`train_utils/distill_data.py:113–148`再规范terminal EOS。此前真实CPU编码检查的RTE/MMLU各有8个response目标，包含包装/结束token，不能把它们全部视为答案语义token；这是模板及mask事实，不是这些包装导致质量下降的因果证明。

这些模板证据以实际文件和注明版本源码为主源，没有额外生成重复指标报告。经验已同步[数据、归约与模板边界](../../lessons/kd_and_data.md)，当前实验状态不因这次CPU诊断而更新。

## 已完成的必要验证

- 新增stage stop：成功保存/评估并恢复teacher/optimizer后，写回各rank实际RNG再暂停，不执行最终导出。修复原checkpoint在评估前保存RNG而lm-eval改变RNG的问题；总调度和数学resume契约不变。相关CPU69 passed，契约专项3 passed；真实CPU Trainer含dropout/累积的连续4步与2→4参数、Adam、scheduler逐位一致。
- 真实0910双卡、batch8×acc2、seq1024、decoder+LoRA共8步，连续与4→8恢复覆盖保存、八任务limit1评估、回训与阶段恢复。第8步scheduler及两rank Python/NumPy/CPU/CUDA RNG完全相同。参数不逐位相同：两轮第4步已有BF16/FlashAttention非确定差异，不能把第8步差异单独归因于resume，也不声称GPU exact参数轨迹。原始状态比较`verify_staged/stage_resume_validation.json`。
- 真实导出先复现旧core失败：训练解码packed路径在冻结后切到cached fused。旧fused的LN/SiLU、首层FP32 bias、末Linear舍入与当前decoder不一致。修复核6项CUDA回归通过，真实q/down采样前向逐元素一致；参数梯度一致或只有最大5.96e-8的FP32归约差异。整模块残余relL2约0.000223/0.000284。
- 同checkpoint8的完整模型分解证明：冻结保持原路径=0差异；仅切计算核max_abs0.1875/relL2 0.00794137；随后LoRA转换相对同路径=0差异。JSON `verify_continuous/candidate_model_stages.json`。因此结构校验统一部署路径，跨核误差单独记录，部署缓存及全部原门限不变；不是关缓存、删检查或放宽容差。
- 该pipeline候选CPU7 passed；真实最终导出及fresh strict重载exit0。core与runtime cleanup误差0；LM-head融合/整体结构max_abs0.125、relL2 0.000683423，满足原门限。原有FP32→BF16导出诊断0.1953125/0.00916364没有冒充零误差。
- fresh两rank均All keys matched successfully，同fast tokenizer、2卡八任务limit1的metrics及metric_keys逐项一致。只证明该配置的流程与保存重载可用，不是全量精度结论。核心验收日志 `verify_continuous/candidate_pipeline_finalize.log` 的 `ACCEPTANCE_SUMMARY`保留schema6、checkpoint id、完整runtime_audit与退出结果；原始final/fresh评估JSON保留。

## 版本、运行标识与资源

2026-09-24用户明确授权后创建 detached worktree `/home/shaoyuantian/program/VAELLM-e2e-0910-20260924`，基于HEAD `82993cfdb589934e4dc771797343701b36890393`；没有创建分支或提交。共享训练源码保持原样，阶段功能已在该HEAD。

已应用补丁 `experiments/e2e_0910_search/finalization_fix.patch`，SHA256 `3a17df1be0762d29e44b5e3558b7e7737b11fdcc6f11673e2984a3bda667c053`。落地kernel SHA256 `ac09ece5e46bb71463e6e63de0eb2d05107db4f2bf48029ae2ba4b94f9cd7abd`、pipeline SHA256 `0b278580b15c2dcb18b48e83af00c0fb62ba1358d4b41510c0508b2f7a82f8ef`均与通过真实验证的候选一致。`git diff --check`通过；真实CPU导入确认main/pipeline/fused都来自worktree，四脚本完整CLI解析及五个原始数据路径检查通过。复用上述数值/运行证据，没有重复短测。

`data`软链到原数据；初始模型、结果目录使用原绝对路径，不复制大数据或权重。启动shell已激活 `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`（Python3.11.13），使用nohup，两条链不互相等待。

| GPU | nohup队列PID | 实际命令（在worktree，顺序执行） | 队列日志 |
|---|---:|---|---|
| 4,5 | 2937476 | `bash experiments/e2e_0910_search/run_lora_lr1e4.sh` → `bash experiments/e2e_0910_search/run_decoder_lora_lr1e4.sh` | C核心日志（原lane45生命周期已归并） |
| 6 | 3147785 | `bash experiments/e2e_0910_search/run_proxy_b4_kl_stage400.sh` | E核心日志（原续训wrapper生命周期已归并） |
| 7 | 3147786 | `bash experiments/e2e_0910_search/run_proxy_b4_kl_lr3e4.sh` | G核心日志（原wrapper生命周期已归并） |

A/B本次实际运行子目录均为 `Qwen_Qwen3-8B_20260924_023506`，位于各自 `lora_lr1e4/`、`lora_lr3e4/` 内；核心训练日志为 `compressed_e2e_fintuning.log`，实际配置为 `normalized_e2e_runtime_args.json`。C实际子目录为 `decoder_lora_lr1e4/Qwen_Qwen3-8B_20260924_041131/`，D为 `decoder_lora_lr3e4/Qwen_Qwen3-8B_20260924_041336/`。旧6/7队列PID2937477及保留进程均已退出，唯一生命周期信息已归并B/D核心日志，旧lane67.log已清理。

旧60GiB保留进程4=2921847、5=2921848、6=2911406、7=2911407已正常释放并交接给真实任务。以下为历史安排，已被本日最新“无后续长程任务就释放资源”规则撤销：此前按尽量不空GPU4–7的要求，每条链成功跑完两组或任一组失败后，在原卡启动每卡60GiB的有限保留，最长12小时，SIGTERM立即释放，不空转计算、不自动重试训练。队列日志记录 `ARM_EXIT`、`LANE_TRAINING_EXIT`、保留PID和到期时间；因此队列PID存活不单独代表仍在训练。服务器无调度器排他预约，显存保留不保证其他用户不能同时使用剩余资源。

02:41:17 UTC启动验收：A/B均已完成2次optimizer更新，首步约19秒，前两步进度显示约15.7秒/步；真实worker PID为A=2937619/2937620、B=2937623/2937624，GPU4/6为64653MiB、5/7为65633MiB。数据缓存首次构建约5分钟，随后进入训练；没有修改正在运行的脚本/算法。已满足实际前反向与更新推进条件，交付后停止主动监控。400步为晋级检查点，不是69达标声明；阶段结束后由用户唤起复盘。继续实验前根据日志分辨训练、暂停、失败及资源保留状态，不重复启动。

## 最小保留与清理

结果根：`result/compressed_e2e_fintuning/e2e_0910_search_20260924/`。teacher_reference与initial_reference_fixed_decoder均保留实际配置/入口、核心日志和原始指标；复现代码为teacher_reference.py/run_teacher_reference.sh和evaluate_initial_reference.py，未来复跑改用独立输出目录。

短测所有生成模型已清理：staged trainer_state、continuous trainer_state与final_model合计实际磁盘占用 **5,493,882,880 bytes（约5.12GiB）**；初始0910受保护。这些仅8步的流程权重没有质量候选/续训/交付用途，验收结束就删除。另已清理完成的准备日志、重复指标表、退出码和一次性脚本；原始指标、有效配置、核心证据、长期测试保护。修复源码已落地worktree，审阅patch记录该未提交修复；初始参考复现入口的候选路径迁移到worktree中相同SHA256的源码，原始配置保留实际执行时的路径和哈希。

隔离落地后另清理3份/tmp旧候选、3份已结束预检/文档检查日志和2份已释放旧GPU保留日志，实际释放163,840 bytes；同哈希正式源码、复现入口、正在运行队列及其保留脚本仍受保护。

正式首段checkpoint当时按400→1200→2000候选晋级计划保留；该旧数据续训计划现已被从0910启动的新数据四格替代，不能继续作为A/B权重保留用途。A/B/E/H/I已按当前无后续用途清理，只保留最小证据；不因“以后可能有用”保留已停止投入的分支。

2026-09-24单卡E/F短测收尾：两份4步trainer_state、两份重复指标表、两份短测启动日志、已结束lane67.log和9份/tmp一次性预检/脚本/收尾检查日志，实际释放 **563,884,032 bytes（约537.76MiB）**。短测退出、参数更新/优化器有限性、流程验收和复现入口归并各自核心日志；每组仅留有效配置、核心日志与原始JSON，正式脚本加stop/save4、eval_limit1和独立输出可复现该流程。没有新增效果经验，新增的是单卡CE/KD真实流程通过证据。D未生成checkpoint，保留中止原因/配置/核心日志；在该次短测收尾时，A/B400权重用于当时计划的晋级，C/E/F及初始0910、有限保留脚本当时受保护；最新去留以下方新数据转向后的清理为准。

本次文档导航已刷新，docs.py check为errors=0；正式脚本bash语法与worktree diff空白检查通过。

E/F200步收尾：F全部trainer_state、两份重复指标表、两份旧启动日志、已归并的CPU准备及两份收尾日志共释放 **282,128,384 bytes（约269.06MiB）**。退出/资源交接已归并各自核心日志。E的checkpoint200作为正在使用的恢复输入保留；C、E续训、G和初始0910受保护。旧E/F wrapper已退出，两个60GiB保留进程已按身份核实后释放给新任务。

2026-09-24 05:20 UTC第二阶段启动验收：E真实worker3147791已从200推进超过240，G真实worker3147792已从原0910推进超过40，学习率/保存间隔/评估参数与正式配置一致，C的3049887/3049888仍运行。E的实际恢复输出由`runtime_v6.py:126`按checkpoint确定，继续使用原run目录；最初按新run_root寻找目录未找到，已根据源码与实际日志纠正记录，没有重新启动。当前normalized配置反映续训阶段，200步原始评测保持不变，原始入口脚本保存初段配置。E/G wrapper结束后仍各自有限保留60GiB最多12小时。本次只读核对和必要文档收尾完成后，交付并停止主动监控。

## 后续辅助脚本入口名称（2026-09-24）

2026-09-24最新要求将占卡辅助入口统一为隔离worktree内的 `experiments/e2e_0910_search/fake_kernel_test.py`。旧中文名 `解码器速度优化测试.py` 已重命名，不保留兼容副本；旧 `/tmp/e2e_0910_gpu_reservation_20260924.py` 已在09:17全部依赖退出后删除。首次统一入口时保持旧内容（965 bytes，SHA256 `eb3cc04ce9a2ec4e8d33dfb0e92357c51cd8667748eb12b5c4d27ffbed350867`），仅做有限显存保留；该版本现已被下述真实矩阵乘法版本替代。

此前“训练链完成或失败后继续保留显存”的安排已被用户撤销：仅在已安排的后续长程任务之间临时使用；没有后续任务，或任务暂停/取消时，不启动、续期或保留占卡进程，不在最后一项任务退出后无条件追加占卡。本规则已合并进项目 `AGENTS.md`。该次重命名交付时没有占卡进程，自动目标保持暂停。另清理已释放且无引用的两份旧guard日志（`/tmp/e2e_initial_ref_guard6_20260924.log`、`/tmp/e2e_initial_ref_guard7_20260924.log`），内容共254 bytes，实际释放8192 bytes。旧入口重命名不重复计入释放空间。

2026-09-24用户要求“先把任务停了”：仅对核实身份的4个保留进程发送SIGTERM（GPU4/5=3406978/3406979，GPU6/7=3167737/3167982），wrapper2937476/3147785/3147786随之正常退出。没有停止GPU0–3的进程，没有启动新任务或自动重试。三个已完成wrapper的生命周期归并到各自核心日志，清理其进度条日志、三份重复指标表及无引用旧临时入口，共释放1,081,344 bytes；原始指标和最近checkpoint仍存在。无新增效果经验，本次未比较新成绩。


用户随后要求helper实际执行GPU乘法，并继续最终目标。当前`fake_kernel_test.py`的SHA256为`da697809b064ef1299fff4258451a592b56b8443bb58533ac1e6670f53ea1e8a`，保留有限时长和SIGTERM释放语义；采用真实4096×4096 FP16矩阵乘法，不仅分配显存。主任务实测GPU7、1GiB、3秒完成4093次乘法，结果有限且抽样值2.0，exit0；GPU利用率曾达到98%。独立SIGTERM验证0.601秒退出，退出后显存0MiB。该验证只证明辅助脚本实际计算与释放，不是训练加速或精度证据。本轮未用helper长期占卡；E/G及验证通过后的H/I每项任务到终点直接退出。


本轮短测与C分支收尾：H/I真实4步均正常保存、评估、恢复optimizer/teacher并写回RNG后pause，各651个可变张量有限、253个LoRA B非零，651组Adam均step4且moment有限非零；scheduler=4，八任务limit1/0-shot指标齐全。验收和最终exit0已归并各自核心日志，保留normalized配置与原始JSON，删除两份短测trainer_state、重复指标表及临时入口/启动/CPU检查日志。C完整400步比匹配A略低且总耗时约3.10倍，本轮不再推进该配置，故删除无具体后续用途的C trainer_state，保留C配置、核心日志、原始指标和复现入口；这不证明decoder上限。本次合计实际释放902893568 bytes，其中C权重/训练状态339456000 bytes。该次C收尾时A/B400暂按旧大batch对照/候选晋级计划保留，E/G和原0910当时受保护；A/B这一用途现已被新数据方向替代，最新权重清理见下方。短测证据位置：verify_proxy_b4_mass/Qwen_Qwen3-8B_20260924_094253, verify_proxy_b4_k1000/Qwen_Qwen3-8B_20260924_094253。CPU相关反向检查2 passed、25 deselected，无新增效果经验。


## 历史后台交付（2026-09-24 09:48 UTC，现已结束）

四组均完成真实初始化并有optimizer更新推进。H/I无短测resume或eval_limit残留，初始checkpoint仍为0910/id01457bf3-ef22-49e8-847f-dc721287c2d6；rank8，原八任务全量0-shot。E/G仅改resume输入和stop节点，仍5000步总调度/save200/world1/B4acc1；H/I分别仅改变loss类型或K，与E的同200/400预算比较。所有新脚本均通过真实CLI解析、shell语法和逐参数差分；沿用既有恢复证据，无训练源码变更。

| GPU | 配置与终点 | wrapper / worker PID | 实际入口（worktree内） | 已验收进度 |
|---|---|---|---|---|
| 6 | E：1e-4，400→800 | 3471380 / 3471394 | bash experiments/e2e_0910_search/run_proxy_b4_kl_stage800.sh | 至少470 |
| 7 | G：3e-4，200→800 | 3471381 / 3471395 | bash experiments/e2e_0910_search/run_proxy_b4_kl_lr3e4_stage800.sh | 至少280 |
| 4 | H：mass K100，0→400 | 3481976 / 3481982 | bash experiments/e2e_0910_search/run_proxy_b4_mass.sh | 40 |
| 5 | I：partial K1000，0→400 | 3481977 / 3481983 | bash experiments/e2e_0910_search/run_proxy_b4_k1000.sh | 30 |

四组nohup调用均在已激活bitvae的隔离worktree执行，stdin关闭、stdout/stderr落盘，进程退出时写FORMAL_EXIT；无自动重试、无结束后占卡。启动日志位于结果根的proxy_b4_kl_stage800.launch.log、proxy_b4_kl_lr3e4_stage800.launch.log、proxy_b4_mass.launch.log、proxy_b4_k1000.launch.log。E/G继续原run目录；H/I正式目录分别为proxy_b4_mass/Qwen_Qwen3-8B_20260924_094654、proxy_b4_k1000/Qwen_Qwen3-8B_20260924_094654。各run核心日志为compressed_e2e_fintuning.log，保存的原始指标在lm_eval目录。

入口SHA256：
- run_proxy_b4_kl_stage800.sh：d11dc1452f4c8816cf0a1b48c6286e2d5b0103f76d9a1956b88ffb8a40d368b9
- run_proxy_b4_kl_lr3e4_stage800.sh：23378ca133e40c6671725282a653eed8c3268cc889ed47c65a85917903624733
- run_proxy_b4_mass.sh：f2b86ced03ce0dadb96356e0073f6f6ff4d348c3accfca541daadb4b251fbd65
- run_proxy_b4_k1000.sh：f5e01317e8d8800d193798704f966d4ef4f56b9d8fa327c3ad60d61f0e4b70a2

耗时仅按已完成E/G阶段估计：E本段约28分钟、G约42分钟；H/I的400步初筛估计各约30分钟，尚未实测完整时长，不作为承诺。提前pause时进度条分母仍5000，不代表本次会跑满5000。

本轮G最终评估至新数据任务之间，GPU4–6临时调用70GiB/1200秒的fake_kernel_test.py；仅为已安排的新数据长任务衔接，三个进程均已SIGTERM释放，未追加结束后保留。实际计算/释放记录：

- GPU4: MATMUL iterations=579943 elapsed_seconds=346.264939 shape=4096x4096 dtype=float16 finite=True sample=2.0
- GPU4: RELEASED
- GPU5: MATMUL iterations=975020 elapsed_seconds=573.269739 shape=4096x4096 dtype=float16 finite=True sample=2.0
- GPU5: RELEASED
- GPU6: MATMUL iterations=976118 elapsed_seconds=573.290776 shape=4096x4096 dtype=float16 finite=True sample=2.0
- GPU6: RELEASED

上述CPU检查与helper衔接原始临时脚本/日志在证据归并后另清理28672 bytes；本轮三批实际清理合计2255368192 bytes。

最终导航检查：269 documents + 206 aliases，canonical=270，relative_markdown_paths=1084，errors=0，exit0；检查临时日志归并后删除4096 bytes，本轮清理总计2255372288 bytes。

## 已取消的task裸续写与均衡抽样准备（2026-09-24历史过程）

在不修改当前J/K/L/M依赖的前提下，新增独立模块worktree train_utils/task_continuation.py，SHA256 617a62736a50097560ee83e1aaa3776d162a13aebf0769456609c690fdf99c7a。它严格接受两条user/assistant消息，保留题面和答案原空白，按照lm-eval0.4.8的causal pair边界分词，不添加ChatML包装、BOS或EOS；右截断到最大长度，没有可预测答案则跳过。本轮只准备编码函数，尚未注册dataset alias、修改canonical路由或用于GPU训练。

实际Qwen3 tokenizer的CPU参考检查通过：八任务各一条真实train样本与安装版HFLM/TemplateLM对**同一字符串**的pair分词逐token相同，首答案mask与causal shift正确；RTE/MMLU响应由现有ChatML的8目标变为1个答案token。长MMLU第94860行context1365，seq1024或仅容纳context时正确跳过，1366时保留首答案。v2无context尾空白，首次样本选择未满足该额外用例；随后仅读取现有EdgeRazor第43行真实尾空白样本，迁移规则一致。五种非法输入明确ValueError；10条检查最终exit0、CUDA未初始化。原始唯一结果：结果根task_continuation_cpu_validation.json。

这只证明候选编码与同一文本的评估分词规则一致，**不证明MMLU题面/学科description已对齐，也不证明精度提升**。下一轮模板对照先仅改变task源包装，保持题面、答案、原始行和其他SFT来源；不同包装长度可能改变截断后有效样本，比较时应记录，不能假称每条有效训练样本完全相同。当前四组先按既定400步完成，后续按完整结果选择续训及该分支的GPU短测/正式预算，不在活动实验中修改其依赖。

### 任务内均衡抽样的准备与下一轮比较

当前task整体是按各任务行数混合，MMLU/Winogrande规模较大，ARC较少。按已验证CPU重放，J前800条中ARC-Easy只有9条、ARC-Challenge只有3条，而固定初始模型与教师差距集中MMLU和ARC；因此仅提高整个task文件的权重不足以保证各目标任务有足够样本。下一轮检验任务内均衡抽样，收益尚属假设，不因发现比例失衡就宣称修复后必达69。

已将新版单文件按manifest原顺序/边界逐字节拆到data/edgerazor_qwen3/task_train_v2_20260924/by_task/，八文件顺序arc_e、arc_c、boolq、piqa、winogrande、openbookqa、rte、mmlu。各行数2251、1119、9427、16113、40398、4957、2490、99842；数据共191870906 bytes，manifest3929 bytes。CPU退出0、4.064秒，逐文件重读及按顺序拼接SHA与原894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1一致，原输入文件不变。by_task/manifest.json SHA256 61657ead9e009674138ffe02f36526347e9d627c1d5b2dc1f41e27a1fb0f356b。分文件用于即将执行的明确对照，保留；没有复制模型或修改活动输入。

以下四份脚本已准备并通过shell/标量参数检查，**尚未注册所需alias、未集成裸续写路由、未通过GPU短测、未启动**。待当前J/K/L/M完成后才修改其依赖、做必要短测并启动；不能把以下配置当成当前运行状态。均从0910开始，rank8、LoRA LR1e-4、其他固定条件与J相同、B4/seq1024/400步，保留原完整八任务评估。

| 配置与计划GPU | task包装 | task内部抽样 | 总task权重 | 作用 |
|---|---|---|---:|---|
| N task_v2_plain / 4 | plain continuation | 原比例 | 0.5 | 与J隔离包装变化 |
| O task_v2_balanced_chat / 5 | 原ChatML | 8任务等权 | 0.5 | 与J隔离任务内抽样 |
| P task_v2_balanced_plain / 6 | plain continuation | 8任务等权 | 0.5 | 与N/O共同看两个因素及交互 |
| Q task_v2_balanced_plain_only / 7 | plain continuation | 8任务等权 | 1.0 | 与P只改总task比例 |

N/O/P其他四源保持.341/.067/.028/.064；O/P每任务全局样本权重.0625；Q每任务.125。均衡抽样改变各任务随机流/有效样本，不假称仅损失权重变化。所有配置仍prompt0.3、纯partialK100；不在这轮同时改变CE、head或norm LR。对照目标是输入包装与任务样本覆盖，不以生成器名或CPU通过代替质量证据。

脚本位于worktree experiments/e2e_0910_search/run_{task_v2_plain,task_v2_balanced_chat,task_v2_balanced_plain,task_v2_balanced_plain_only}.sh，SHA依次2b94b760a45a33e1d8c6f1773cae489db8ee37a300a6c01a369a306073f365a0、f428da44db2232c18ddcecbaa9185a2bdbf97367a029baac49541f831edbc2e1、9c96f59684ec8d661f4004d3d1e68b22acf1db92a17745068e3089d1586eed29、63694b2f2cad2784a74ffb8366d742d425befece311c169b684bd0430cad04e9。

本轮CPU统计、编码验证及拆分数据的一次性脚本/临时日志和已归档重复JSON在证据归并后自动清理，实际释放98304 bytes；唯一结果JSON、模块、by_task数据及manifest保留用于上述具体下一轮工作。

本轮导航index/check通过（errors=0、exit0）；检查临时日志归并后另清理4096 bytes，CPU准备收尾合计102400 bytes。


## 本次实验的数据冻结与超参数比较（用户于2026-09-24明确）

**仅本次实验的约束。** 保持已修正的v2实际数据源及现有K数据配置，不再自行改变数据组成、抽样比例或输入模板。固定mix为`edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_task_train_v2=0.5`，任务内按原文件行分布，保留ChatML。此约束不是项目长期规范，不修改AGENTS.md。

此前N/O/P/Q的均衡采样、纯任务与裸续写是代理新增的比较，并非数据源修复必需；用户纠正后已对这四个本轮进程发送SIGTERM，wrapper3611318/3611319/3611320/3611321及worker3611330/3611333/3611331/3611332均退出。没有有效全量成绩，不对该方向作效果判断。撤回17个新增alias、plain路由、独立encoder及对应新增测试，仅保留修正后的v2 alias；原ChatML路由与原测试逐字节恢复HEAD，data.py恢复SHA098fe413d1478f5db04db81d37694cbe2f22a40866f6a77ef68ca1d4bb82320c，distill_data.py为0cf457f21b966c788b0d07bde9086703455f3941320162d32fa7e6ff46d455bb。相关准备段仅是历史，不能当当前执行计划。

保留K原配置与数据作为对照，另外三组均从原0910开始、rank8/alpha16/B4acc1/seq1024，5000步cosine调度、warmup20、保存/完整评测每200步、停400。每组仅改变表中一个训练参数，norm/head LR均保持1e-4，不引入eval_limit。K续到800只增加训练预算，不能将K800和新组400当等预算排名；新组先比较既有K400。

| GPU | 配置 | 相对K的唯一变化 | 入口 |
|---|---|---|---|
| 4 | R task_v2_b4_lr6e4 | 普通LoRA LR3e-4→6e-4 | run_task_v2_b4_lr6e4.sh |
| 5 | K续训 | step400→800，数据与超参数不变 | run_task_v2_b4_kl_lr3e4_stage800.sh |
| 6 | S task_v2_b4_prompt1 | prompt_loss_weight0.3→1.0 | run_task_v2_b4_prompt1.sh |
| 7 | T task_v2_b4_dropout0 | lora_dropout0.1→0 | run_task_v2_b4_dropout0.sh |

参考既有loss语义与短程经验：prompt1仍是token加权归约，不是改变task样本比例；dropout0检验小预算下正则影响；K已记录梯度范数0.25819–1.22043低于clip1.5，支持探索更高LoRA LR，但不是6e-4稳定性的既有证据。R先运行24步真实短测，保留warmup20以覆盖峰值LR更新；其余复用既有v2真实更新/保存/评测与exact-resume证据，不修改运行代码。shell及标量参数差分已确认数据mix完全相同，三个fresh只与K相差run_root及指定单参数。

新入口SHA256：R 6cf13ba03936ca3d96fa2644ca32dd3f520317cf4382c2aa8b074f1ef46c0a34；S b937da173d2b9d86fae723cef443c6a524740bf377c61ce8820bef890ae46fb1；T 68361df1fb429bca3d9ddcb975a0fa8f9a6cd63c8aad5dceacd9aa6fd148a712。K续训入口SHA92b8ec831afb1a7651b00da48863451dddd629e308424277d4ee3bc63fffd4ca。均位于隔离worktree experiments/e2e_0910_search。


### 可脱离本机运行的长程搜索（用户追加要求）

本次采用远程Python控制脚本`experiments/e2e_0910_search/run_fixed_data_search.py`及唯一配置`fixed_data_search_plan.json`，固定上述v2数据与评测口径。候选为K基准及8个单变量组：R LR6e-4、S prompt1、T dropout0、U LR1.5e-4、V prompt0.1、W topK1000、X temperature2、Y alpha0.9的kd_top_partial。Y为0.1CE+0.9KD，CE同样参与prompt0.3加权；X同时改变分布温度与T²尺度，不能只解释为更软的监督。

预定阶梯：400步取4，1000步取2，2000步取1，最后至5000步；每200步完整评测，所有阶段保持5000步总调度。400阶段统一使用对应400指标，K额外已跑到800的预算单列；如K晋级，从已有800接到1000，不重复训练。总预算约11000–11400更新、55–57次全量评估，按既有速度估计剩余13–14 GPU小时、墙钟约7–8小时；后段收缩至2卡/1卡，估计并非承诺。达到>=69的有效保存点后停止新增派发、已在跑的自然完成，保留实际最佳状态；预算用完仍未达标则明确budget_exhausted，不声明全局最优或任务完成。

当前先验验证：R实际24步短测跨过warmup20，峰值LR0.0006，24步loss/grad均有限，最大已记录grad1.47699213；651可变张量有限、253 LoRA B非零、651 Adam组step24且moment有限，保存与limit1评测/恢复成功、exit0。X温度2在实际A800、BF16、2×8×151936 logits上比较正式dense/offloaded teacher路径，loss同为0.02390030398964882、梯度最大差0，有限且非零；这是局部损失/反向验证，不是完整模型收益。其余复用对应v2、K1000、CE10与exact-resume证据。9种配置全部通过当前真实CLI解析。

长程状态写在结果根`fixed_data_search/`。K400最佳权重已先复制到该目录best_checkpoint并将round_base_ref重定位到同一个0910绝对路径，以免活动K的save_total_limit1轮转删除该实物；原base id不变，分数65.61459093822081与rawJSON一致。脚本仍在完成CPU校验/独立代码审阅，尚未宣称控制器已启动；当前四路训练是独立nohup，启动标识见后续验收。

取消组task_v2_plain：START 2026-09-24T11:13:52Z pid=3611318 script=experiments/e2e_0910_search/run_task_v2_plain.sh；FORMAL_EXIT 2026-09-24T11:14:49Z code=143；无保存checkpoint或全量指标。

取消组task_v2_balanced_chat：START 2026-09-24T11:13:52Z pid=3611319 script=experiments/e2e_0910_search/run_task_v2_balanced_chat.sh；FORMAL_EXIT 2026-09-24T11:14:49Z code=143；无保存checkpoint或全量指标。

取消组task_v2_balanced_plain：START 2026-09-24T11:13:52Z pid=3611320 script=experiments/e2e_0910_search/run_task_v2_balanced_plain.sh；FORMAL_EXIT 2026-09-24T11:14:49Z code=143；无保存checkpoint或全量指标。

取消组task_v2_balanced_plain_only：START 2026-09-24T11:13:52Z pid=3611321 script=experiments/e2e_0910_search/run_task_v2_balanced_plain_only.sh；FORMAL_EXIT 2026-09-24T11:14:49Z code=143；无保存checkpoint或全量指标。

本次有限衔接GPU4：MATMUL iterations=486580 elapsed_seconds=290.694295 shape=4096x4096 dtype=float16 finite=True sample=2.0；RELEASED。

本次有限衔接GPU5：MATMUL iterations=946275 elapsed_seconds=561.046221 shape=4096x4096 dtype=float16 finite=True sample=2.0；RELEASED。

本次有限衔接GPU6：MATMUL iterations=952751 elapsed_seconds=560.885489 shape=4096x4096 dtype=float16 finite=True sample=2.0；RELEASED。

本次有限衔接GPU7：MATMUL iterations=494003 elapsed_seconds=290.948539 shape=4096x4096 dtype=float16 finite=True sample=2.0；RELEASED。

取消的裸续写真实短测两组均4步exit0，651可变张量有限、253 LoRA B非零、651 Adam step4、scheduler4、RNG四项存在，八任务limit1齐全；仅机制通过，无效果结论。CPU唯一证据仍保留task_continuation_cpu_validation.json（包含该历史模块与集成结果），代码/派生分任务数据/取消的入口已撤回。

本次取消方向与已完成短测收尾释放allocated=1037217792 bytes、logical=1037066670 bytes；初始0910、原始v2单文件、活动K/R/S/T及已保护最佳checkpoint未删除。


### 长程控制器启动验收（2026-09-24 11:33 UTC）

生产--check由root再次执行并PASS：9配置、4个实际worker命令/GPU/normalized契约、同一0910与seed checkpoint均核验。控制器源码SHA256 39222420d10613dc474315cb4f14117317261811b675dee5dbf12a8786fa0dbb；计划SHA256 20e61723800034b3a1a6e2441c3ecc7db7ac9e527f0c4ded04aa2b9d10937fb6。独立代码审阅发现并修复同阶段比较/精确全量样本/中途达标停派/故障前有效候选保留/终态模型与指标对应问题；CPU重放使用实际K400 raw与metadata，没有伪造训练成功。新的后台控制器已进入run主循环并输出四条adopted，summary.status=running、active=4、trials=9。

实际命令：在bitvae和隔离worktree内以nohup调用`python -u experiments/e2e_0910_search/run_fixed_data_search.py --plan experiments/e2e_0910_search/fixed_data_search_plan.json`，stdin关闭、stdout/stderr写fixed_data_search/controller.log，wrapper3641607；实际控制器子进程：`3641610 python -u experiments/e2e_0910_search/run_fixed_data_search.py --plan experiments/e2e_0910_search/fixed_data_search_plan.json`。wrapper结束写SEARCH_EXIT。

- GPU5 K_base：wrapper3619141 / worker3619154，当前阶段终点800，run_dir=/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/task_v2_b4_kl_lr3e4/Qwen_Qwen3-8B_20260924_102952。
- GPU4 R_lr6e4：wrapper3631160 / worker3631163，当前阶段终点400，run_dir=/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/task_v2_b4_lr6e4/Qwen_Qwen3-8B_20260924_112559。
- GPU6 S_prompt1：wrapper3619142 / worker3619153，当前阶段终点400，run_dir=/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/task_v2_b4_prompt1/Qwen_Qwen3-8B_20260924_111835。
- GPU7 T_dropout0：wrapper3619143 / worker3619152，当前阶段终点400，run_dir=/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/task_v2_b4_dropout0/Qwen_Qwen3-8B_20260924_111835。

K/S/T已验证实际更新推进，R启动验收：`2026-09-24 11:32:02,909 - compressed_e2e_fintuning - INFO - E2E train: step=200 loss=0.2953 distill_loss=0.43282434344291687 learning_rate=0.0005980893688468733 grad_norm=0.6825549602508545 epoch=0.04`。脚本无自动重试；达到阈值不再派新任务，失败不会冒报完成，结束不追加占卡。只有下一阶段确实使用的卡在等待阶段衔接时才允许有限helper。末段无需4卡则释放不再使用的卡。

唯一最佳指标写best_metrics.json：中间训练checkpoint与最终final_model分别绑定实际评分；终态最佳只保留成品，不冒用final指标给训练态。故障前的有效优秀点若需保留则明确recoverable_failed_job，不视为该任务成功。控制器完成这轮预算不等于证明69可达或全局最优。

验证边界：真实R24峰值LR、已有exact-resume/保存评测、T2损失反向与CPU控制逻辑已验证；当前只确认接管和实际推进，尚未等到此控制器的首个400阶段结束或后续晋级，也未声称长程已完成。用户明确不希望等待，交付后由远程脚本继续，代理停止主动监控。

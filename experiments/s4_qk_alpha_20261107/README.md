# S4 固定QK范围的教师alpha对照

11月7日两臂均已完成64步、独立导出／fresh及完整八项，各阶段和外壳均exit0，GPU4／5已释放。外壳退出时间为北京时间09:27:43／09:27:59。原批准设计：知识域教师权重alpha=.25／.75，gold权重1−alpha；分别使用GPU4／5，从原down/pre `fa50bf40-c682-4910-8e72-d024f6b678e4` fresh AdamW执行64步和完整八项。固定QK72投影／432叶，其余180投影冻结但保留输入反传；全252投影均原生S4，logical／packed均4.094031292459239bpw。约2GPUh参考、非硬中断；不自动续128、LongBench、动态alpha、数据权重或其它批次。研究安排见[范围批后续](../s4_recovery_screen_20261105/README.md#后续候选与当前交付边界)，排程以[唯一当前计划](../../docs/experience/records/w25_transfer/current_plan.md)为准。

产物根：

```text
/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/qk_alpha_20261107/trial_20261107_alpha25_75
```

## 完整结果与本轮结论

固定同父、QK范围、64步、输入和优化条件，仅改变知识目标中的教师比例。下表均来自同一末点checkpoint的完整八项；原父和.5控制复用原始完整结果。

| 指标（%） | 本轮恢复前父模型 | 教师.25／gold.75 | 教师.5／gold.5控制 | 教师.75／gold.25 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 71.26477710 | 71.26477710 | 71.24341262 | 71.29326307 |
| ARC-Challenge | 54.94880546 | 54.77815700 | 54.77815700 | 54.60750853 |
| ARC-Easy | 78.70370370 | 78.78787879 | 78.87205387 | 78.91414141 |
| BoolQ | 86.33027523 | 86.36085627 | 86.33027523 | 86.45259939 |
| OpenBookQA | 41.00000000 | 42.40000000 | 42.60000000 | 42.40000000 |
| PIQA | 77.80195865 | 77.85636561 | 78.01958651 | 77.58433079 |
| RTE | 77.97833935 | 78.33935018 | 78.70036101 | 78.33935018 |
| WinoGrande | 69.37647987 | 69.37647987 | 69.92896606 | 69.45540647 |
| 八项均值 | 69.67554242 | 69.89548310 | 70.05910154 | 69.88082498 |
| MMLU正确数／14042 | 10007 | 10007 | 10004 | 10011 |

.25相对.5的M多3题（+.02136448pp），八项−.16361843pp，2项升／5项降／1项平；.75的M多7题（+.04985045pp），八项−.17827656pp，3项升／5项降。两条均未同时改善M和综合分。.5只是本三点、单seed、64步的最高八项，不能称全局最优比例。

| MMLU配对 | 修对 | 改错 | 净变化 |
| --- | ---: | ---: | ---: |
| 原父→.25 | 152 | 152 | 0 |
| 原父→.75 | 168 | 164 | +4 |
| QK .5→.25 | 65 | 62 | +3 |
| QK .5→.75 | 68 | 61 | +7 |

.75相对原父仅净增4题，仍比10111门槛少100题；.25少104题。.75损失最多的七项为PIQA、RTE和WinoGrande，不能仅按M的7题微增宣布新优胜。未评LongBench，最终九项>67.64714491790848、M≥10111及从BF16整链验证均未完成。

固定诊断每个面板128题，分别报告自身0→64；early已消费、late为尚未训练的后半、heldout独立未参训。

| 教师alpha | early命中／gold NLL | late命中／gold NLL | heldout命中／gold NLL |
| --- | --- | --- | --- |
| 0.25 | 70→73；1.128825→0.993596 | 81→82；1.076468→0.981067 | 60→60；1.429420→1.274305 |
| 0.75 | 70→75；1.128825→1.039499 | 81→83；1.076468→1.013435 | 60→59；1.429420→1.319880 |

两条各自early／late的混合目标损失下降，三面板gold NLL也下降，但完整M净收益很小，heldout命中未提高。不同alpha的mixed KL目标不同，绝对值不用于跨臂排序。当前证据支持“修复和新增错误大体抵消”的现象，不能据此判定数据覆盖、教师错误、参数容量或优化方式哪一项是根因。

| 臂／末点UUID | 训练内计时秒 | 正式全链秒／GPUh | 外壳退出时间（北京时间11月7日） |
| --- | ---: | ---: | --- |
| alpha25／`b735e725-cb79-44e6-81e8-ddfd94aecbb0` | 2231.769398 | 2809.685017／0.780468060 | 09:27:43 |
| alpha75／`7219f986-daf2-45b5-ad23-c76303e095c9` | 2244.004963 | 2825.712541／0.784920150 | 09:27:59 |

两臂正式链合计1.565388211GPUh，低于约2GPUh参考预留；每臂约47分钟。两臂训练峰allocated均35.39335251GiB（reserved38.30078125GiB），M／七项评测峰allocated分别59.90345955／44.79796934GiB。阶段进程近似耗时按train／export／fresh／M／seven计：.25为2258.859／64.133／38.081／256.307／188.827秒，.75为2271.131／66.207／38.475／257.949／188.513秒；由进程起点至exit文件时间估算，约1秒计时来源误差，不与训练内计时混用。CPU准备39.783936秒及代码准备另计，无本批新增GPU短验。正式链计时含训练、导出、fresh和八项，不能再累加训练内计时。实际每臂消费groups0…63、Web1024／Med1024各一次、QA128各四次，保持原500日程；未训练全部4096知识题。

每臂`result.json`保存八项、对父／.5的逐题修复损伤和原始样本哈希；`training/training.json`、`training/updates/`、`training/training_state.pt`记录64更新、cursor64、FP32 masters／Adam／RNG及数据来源；`post/export.json`与`post/evaluation/{fresh,mmlu,seven}.json`保存导出与同UUID评测。原记录显示完整冻结补集精确不变、fresh3927字段一致／前向有限，投影logical／packed均4.094031292459239bpw。根`completion_verification.json`归并本次CPU终态核证：64步／group／alpha和有限非零梯度、masters／Adam各432且全部step64、completed／next_cursor／next_group均64。独立质量审查核同UUID、原版本／0-shot／无limit／B64和B32以及已有逐题来源证据；本次未额外重跑全套原样本独立统计，完整逐题SHA／身份／修复数沿用已完成`result.py`的逐行核证，不宣称做过第二次全量复算。

本批评审结束，用户已批准下一批固定QK／alpha=.5的Web:Med源损失权重3:1／1:3两臂。两条alpha无具体后继，已按持续清理授权精确删除各自`post/model/pytorch_model.bin`、`post/model/cat_runtime_state.pt`及`training/training_state.pt`共6文件，实际释放12,198,690,816 B（11.360916GiB）；不是前两轮清理量的重复计数。删除前exit0、原PID退出且无reader，新源权重包不依赖这些载荷；删除后344份保留源码／配置／指标／样本／元数据哈希和11份共同父／控制／数据文件状态保持，原有completion_verification.json的terminal_cleanup_20261107保存精确路径／大小／状态。两个旧模型目录现在仅保留证据，不能加载或续训；QK .5、原父与数据继续保护。未改旧检查／源绑定或绕过依赖。不扩固定alpha网格，不由微小差值直接启用动态alpha、128或LongBench。经验归入[同父恢复的净收益与代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。

## 唯一变量与来源

只读复用原WebMed dataset／input／groups顺序、`raw_teacher.pt`及原QA目标，不重新抽样或计算教师。原manifest为`mixture_applications=0`。本包从原始logp用`logaddexp(raw_logp+log(alpha),onehot_logp+log(1-alpha))`混合一次，沿原四选支持和log-space数值路径，无clamp、额外softmax或归一化；不从旧.5目标再次混合。每臂自己的data_digest、data_contract、learning_contract、checkpoint provenance和诊断均记录实际alpha。

沿原groups0…63，每步Web16＋Med16＋QA8；Web1024／Med1024各一次，QA128各四遍。知识域总权重.5、QA域.5，QA仍为静态gold-prefix全词表KL。seed137、LR5e-7、betas(.9,.95)、eps1e-8、weight_decay0、foreach=False、joint clip1.5、原500步linear前64步不变；FP32 masters／grad／Adam、原BF16计算与部署。固定QK选择、packed forward、冻结缓存及`update`直接复用上一已验证路径，不重新进行GPU2/9短验或完整resume审计。

.5控制只读复用原QK64 UUID `40f76c27-8f58-4bf8-b448-08af2ea439c9`，M10004／14042，八项70.05910153709371，正式全链.778536GPUh；它是比较控制，不是两个新臂的初始化模型。

## CPU核证与完整输出

`prepare`核全部4096知识题的.5重建与旧FP32目标exact、128QA目标exact；.25／.75为有限detached FP32，概率公式绝对容限1e-7、归一化容限2e-6（沿原helper）。.75不要求gold为argmax，不截断或修正目标。全部input／身份／顺序／domain和document权重／QA／真实曝光保持；实际QK参数集合与.5控制相同、432叶／2,479,104元素，FP32初始masters来自原父，Adam state为空。请求固定64、无LongBench，旧代码／请求／数据合同不修改。

诊断保留各臂自身0→64的early／late／heldout面板：64步时early已消费，late尚未消费，heldout始终独立。每臂mixed KL使用自身alpha目标，跨alpha的KL绝对值不可用于收益排序；共同gold NLL／命中和同协议完整八项可比较。

controller顺序为64训练与完整可恢复态→进程退出→独立进程原生导出→strict fresh3927字段／finite／位宽→原M57科14042题B64及其余七项10700题B32，0-shot和原acc／acc_norm协议不变。汇总逐题对齐doc／prompt／target hash，对共同父及QK.5分别记录八项修复／损伤；不以部分八项或训练loss替代完整结果。新逐题汇总在CPU用真实24742条控制样本自配对检查。

两臂曾各保留一份正式64模型及完整masters／Adam／RNG／参数顺序／cursor64状态供批末讨论；评审结束后载荷已按上节清理，仅保留状态摘要与原指标，不再可加载／续训。无自动重试／轮询／占卡；正常更新确认后停止主动监控。

## 后台复现入口

调用者激活bitvae；命令应在tmux／nohup后台执行，脚本内不激活环境。`prepare`及正式入口拒绝覆盖已有attempt。

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_qk_alpha_20261107.prepare
bash experiments/s4_qk_alpha_20261107/run_alpha25.sh
# 独立tmux会话，GPU5：
bash experiments/s4_qk_alpha_20261107/run_alpha75.sh
```

CPU证据为根下`CPU_validation.json`，请求／数据合同位于`alpha25/`、`alpha75/`；各臂`run.json`记录精确命令和进程，`training.log`及`training/updates/`保存实际推进，`post/export.json`、`post/evaluation/`和`result.json`保存独立导出与完整八项事实。启动及完成须按这些原件区分，不把配置就绪写成已完成训练。

以下为已结束进程的历史启动证据，当前完整结果以上节为准。11月7日启动核证：CPU prepare用时39.783936秒、exit0、CUDA未初始化；.5全部4224目标exact重建、两新比例／全部输入QA权重／实际曝光／QK范围与freshAdam均通过。Astra独立审查新consumer、逐臂metadata及CPU／请求／源码pins通过。复用原QK GPU峰值／冻结／独导证据，本批未新增GPU短验。

| 臂 | tmux／wrapper／controller／worker | 正式step1／2秒数 |
| --- | --- | --- |
| 0.25 | `S4_QK_alpha25_g4_20261107`／203712／203735／204296 | 40.0685／42.2057 |
| 0.75 | `S4_QK_alpha75_g5_20261107`／203717／203737／204311 | 40.1497／42.3899 |

两条均实际消费groups0/1、原Web16＋Med16＋QA8、LR5e-7／4.99e-7；逐步原件记录真实alpha和一次混合digest，72投影梯度全部有限非零、首步早中晚FP32及BF16均真实变化。各臂`startup_verification.json`保精确command／PID／start_ticks／请求绑定／数据曝光。已停止主动轮询，controller继续约定64→独立导出→fresh→完整八项。此为启动证据，不称64训练或完整八项已完成。

准备时间单列：新包初建08:33:45至CPU plan完成08:38:59约314秒，其中CPU核证39.783936秒；此前文档／源码阅读未单独精准计时，不计入GPUh。Astra亦已独立核两臂原step1/2与0步诊断：各面板初态gold命中／NLL一致（early70、late81、heldout60），启动证据充分，无需额外GPU检查。

文档核查：本轮修改处diff检查通过，新结果及当前计划链接无新增错误；全库docs检查仍为1654条既有缺链，含经验正文8条既有缺链，不宣称全库检查通过。

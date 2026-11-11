# 同一 C1 的 M 目标与学习率恢复对照

本包固定使用完整 Rice C1 raw `744e3b98-8a09-4a1d-8b30-013db5d73d44`，各臂独立 fresh，不从已恢复 Muon32 接续。指定控制是[已完成 C1 Muon32](../rice_muon_cmcc_20261031/README.md)：完整九项64.89598790075989，原M teacher/gold各半、条件四项支持。四臂现均已完成32更新、独立fresh及同UUID完整九项；技术流程全部exit0，质量均`not_met`。`pure_gold`于GPU2空档接续，身份与来源见[所属记录](../c1_pure_gold_gpu2_cmcc_20261102/README.md)。

| 臂／GPU | M 域目标；域外系数仍 `.5` |
| --- | --- |
| `full_teacher`／3 | `.5 KL(full151936 F||S) + .5 CE(gold|native4)`；full词表目标与M32 teacher执行的完整配方 |
| `pure_gold`／2 | `CE(gold|native4)`；仅M去掉软教师项 |
| `conditional_lr_half`／4 | 原缓存条件四项混合目标；Muon／aux Adam实际scheduled LR整体×`.5` |
| `conditional_lr_double`／5 | 原缓存条件四项混合目标；Muon／aux Adam实际scheduled LR整体×`2` |

QA仍当前学生原生greedy轨迹、独立原始full-HF全词表KL，域系数`.5`。四臂均all252 D、R0、M1024/Qasper32/Hotpot96、原组0…31、seed137、原500线性日程前32。LR两臂直接消费原mixed缓存和原`mmlu.weighted_loss`，不从full cache反推替代目标，分别只改整体LR倍率。Muon NS5／.95 Nesterov、aux Adam、clip1.5不变；κ按各臂首个联合梯度重新校准并固定，LR倍率不进入κ校准基准。新的缓存／probe独立进程，不把其状态或RNG带入正式训练。目标两臂比较完整恢复配方，QA后续轨迹及κ可随目标改变，不称纯梯度几何机制证明。

原M缓存只有四项条件概率；GPU4独立补原M1024同最后query的完整151936词FP32 teacher cache（原始概率622329856B），核同原始HF及旧纯F四项条件概率差异，不拼接／修正教师分布。固定train首组32（不筛正确／错误）诊断teacher／student四项mass、full／conditional KL、gold CE和真实full-D backward量级；support mass与梯度幅度不作为质量门槛，不据此改配比或LR。旧mixed KL与`.5 KL_cond+.5 CE_cond`仅差目标常数，CPU验证梯度等价；新增真实backward覆盖新目标，Rice已有两更新／保存fresh证据复用，不重复短链。

上述[cache／probe](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/probe.log)均exit0，分别105.04／146.85秒。旧control／full_teacher／pure_gold首组M loss为.11800702／.15312326／.19549167，full-D梯度范数103.74963／102.62755／128.67898；probe无更新、原全参数未变、cache后teacher严格释放。新旧teacher四项概率差最大.174566，1024例中1例argmax改变，属于材料性差异；GPU3独立5次HF前向诊断26.78秒、exit0，当前serial逐值复现旧缓存，M32逐值复现新缓存，mask／position／options／FP32 RoPE／FA2一致。[诊断原件](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/hf_batch_diagnostic.json)支持本条件BF16 batch执行差异，不能把full_teacher视作纯support因果对照；不重复cache／probe或扩诊断。

产物根为`/data1/cmcc/VAELLM/artifacts/c1_m_target_muon_cmcc_20261102`，原始运行／完整评测在`result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102`；本次修复的准备、三臂成品和运行记录均在各根下独立`revision_group_fix_20261102/`，缓存及probe仍消费原根，不覆盖旧结果。每臂训练4h上限（原C1Mu32实测8047秒作成本参照），真实BF16成品随后独立strict／冻结Rice／同输入logits核证，再在自身GPU完整九项。技术失败只停止自己的后继，不自动重试／resume／选点／扩网格。现父只有BF16成品，无FP32 master／optimizer存档；本次是fresh新阶段，绝不称exact resume。

先由调用者激活`bitvae`，核`which python`和`python -V`。CPU、缓存／probe和正式controller全部从tmux运行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_cmcc_20261102.validate_cpu
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_cmcc_20261102.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_fill_20261102 --validate
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_fill_20261102
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_cmcc_20261102.sequence --arm conditional_lr_half --gpu 4
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_cmcc_20261102.sequence --arm conditional_lr_double --gpu 5
```

旧half／double已在首个真实`optimizer.step()`后因日志读`group['qa_group_index']`失败，worker及controller均exit1，无有效成品，不能计成功或续训。根因是LR倍率循环变量`group`遮蔽数据group；现仅改为`optimizer_group`，保持目标、κ校准和原schedule后全部group整体倍率。旧失败、旧prepared、旧GPU3 controller CPU证明均保留；[修复diff及旧源码归档](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/source_fix.diff)可追溯原版本。新revision重新绑定全部源码并fresh原C1，不自动重试原失败目录。

修复与启动证据：修复通过Astra窄审；revision CPU数学／导入／AST遮蔽回归、独立prepared（77.97秒）及GPU3 controller CPU绑定全部exit0。11月2日13:01:49，[三条tmux控制链](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/launch.json)启动；[实际首更新与下一组核证](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/startup_verification.json)已通过，源码pins未变、原C1 fresh来源一致。三臂均正确记录step1／group0／qa_group0，32M＋8QA联合更新、1728叶梯度与优化器状态有限、参数真实改变，随后进入step2／group1并保持master／optimizer／κ。以上是启动时证据；11月3日已核四臂终态完成，详下节，不再沿用“训练在途／fresh待做”的旧状态。

| GPU／tmux | controller／start ticks | formal worker／start ticks | 首步实际Muon LR／aux Adam LR | FP32位移L2／BF16改变元素 |
| --- | --- | --- | --- | --- |
| 3／`C1_full_teacher_g3_fix_20261102` | 1714908／33309794 | 1716255／33312099 | 5.56525277e-6／1e-6 | .00230641073／196818 |
| 4／`C1_lr_half_g4_fix_20261102` | 1714909／33309794 | 1717457／33320040 | 2.78306223e-6／5e-7 | .00115275319／98991 |
| 5／`C1_lr_double_g5_fix_20261102` | 1714910／33309794 | 1717469／33320078 | 1.11327426e-5／2e-6 | .00461276908／386405 |

首组更新分别255.87／256.37／254.11秒；half／double全部group实际LR精确为各自原schedule的`.5`／`2`倍，κ按首个真实joint gradient校准，未随倍率重标定。每臂完整命令、环境、日志及退出位置在各自`plan.json`／`run.json`与上述核证原件；后继已依次完成fresh／自身GPU完整九项，质量`not_met`作为有效结果。唯一执行安排见[当前计划](../../docs/experience/records/w25_transfer/current_plan.md#11月2日二阶段恢复目标对照)。本轮完整九项及配方比较已完成；阶段交付及已结束分支清理见下节。经验复用：[KD语义与数据边界](../../docs/experience/lessons/kd_and_data.md)要求原mixed缓存不重混，[来源和生命周期](../../docs/experience/lessons/checkpoint_lifecycle.md)要求各臂fresh来源和独立strict；实现修复与旧失败证据均保留；结果经验的适用边界见下节。

## 完整结果、成本与本批决定

11月3日核证四臂`run.json`、`formal32.exit`、`fresh.exit`、`full9.exit`、`run.exit`及外层controller均exit0，记录中的controller和阶段PID均已退出；评测为`complete_full8_full21_current_goal`。每臂完整九项内层`acceptance_exit_code=1`仅表示质量`not_met`。四份训练报告的`fresh_pending`为导出时状态，后继各自`fresh_audit.json:verified`已闭合，不能据旧字段记作未完成。

共同raw仍为`744e3b98…`（九项64.1901653263311），原1×条件目标Muon32仍为指定控制`3ebcc40c…`（64.89598790075989）。下表均为同父／同控制差值，单位pp；M为正式14042题，LB为21任务task macro。

| 配方 | 九项 | Δraw | Δ原Muon32 | MMLU | Δ控制M | LB | Δ控制LB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_teacher | 64.9479724422 | +0.7578071159 | +0.0519845415 | 61.5083321464 | −0.0712149266 | 43.9280942930 | +0.0165519694 |
| conditional_lr_half | 64.7449413929 | +0.5547760666 | −0.1510465078 | 61.3943882638 | −0.1851588093 | 43.7783839994 | −0.1331583242 |
| conditional_lr_double | 64.9349294209 | +0.7447640946 | +0.0389415201 | 61.8928927503 | +0.3133456773 | 43.5856349438 | −0.3259073798 |
| pure_gold | 64.7432203451 | +0.5530550188 | −0.1527675556 | 61.3587808005 | −0.2207662726 | 44.0132448810 | +0.1017025574 |

其余七项相对原Muon32的代价如下；各项原始分数以对应验收JSON为准，不新建重复汇总文件。

| Δ控制／pp | full_teacher | half | double | pure_gold |
| --- | ---: | ---: | ---: | ---: |
| BoolQ | −0.091743 | +0.244648 | −0.183486 | −0.061162 |
| RTE | +1.083032 | −0.361011 | +0.722022 | −0.361011 |
| WinoGrande | −0.157853 | −0.394633 | −0.710339 | −0.078927 |
| ARC-E | −0.505051 | −0.505051 | +0.042088 | −0.420875 |
| ARC-C | +0.085324 | −0.170648 | +0.255973 | −0.170648 |
| OBQA | 0 | +0.2 | +0.4 | 0 |
| PIQA | +0.108814 | −0.054407 | −0.163221 | −0.163221 |

| 配方／成品UUID | 训练／fresh／完整九项墙钟秒 | Qasper／Hotpot实际监督tokens |
| --- | --- | --- |
| full_teacher／`56ef38e0-dcfe-4993-a03c-36e6a491ffb2` | 8039.90／63.42／20751.53 | 1266／978 |
| half／`9bf27c21-58e2-4fb4-9695-41389d86ba48` | 8097.39／62.41／21352.73 | 1295／1001 |
| double／`495f23ff-7a20-4c77-b28b-90ca7bd44fc8` | 8118.18／62.86／20006.47 | 1174／996 |
| pure_gold／`36a6c91f-b4ef-4744-9c36-c24816e51dec` | 7995.37／62.86／21075.84 | 1327／973 |

四臂均实际32更新、M1024一次、Qasper64次呈现／32独立文档、Hotpot192次呈现／96文档，M及QA case顺序与原Muon32全同；训练case实核仅`cais_mmlu/auxiliary_train/train`、`qasper/train`、`hotpotqa/train`。原Muon32的Q/H监督tokens为1309／1002，各臂学生OPD轨迹随更新反馈变化，不宣称相同监督token曝光。训练约2.22–2.26h/臂、完整九项约5.56–5.93h/臂；各臂在自身GPU独立完成，不能将墙钟相加当本批并行耗时。

四臂均logical2.5216473392818286／packed2.5350219996079155，全模型tensor4690895025B；实际账本包含decoder、native code/保护/rotation以及Rice stream、indices、values、scales/zero-points等，embedding/head和norm另列。全部R0、源C1恢复前逐值还原、2199非D冻结和Rice状态exact；各自3927字段strict／同输入BF16 logits exact。每臂full8 64叶／24742条和LB21／4750条的原预测SHA、数量及scores/config SHA已重核，full8、LB、metadata、导出、fresh和pipeline绑定表中同一UUID；四臂与raw／原Muon32的full8 `(doc_id, doc_hash, prompt_hash, target_hash)`全同，LB protocol/archive/model policy/package versions全同。

原件：[full_teacher验收](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/full_teacher/evaluation/longbench/merged/nine_task_acceptance.json)、[half验收](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/conditional_lr_half/evaluation/longbench/merged/nine_task_acceptance.json)、[double验收](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/conditional_lr_double/evaluation/longbench/merged/nine_task_acceptance.json)、[pure_gold验收](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold/evaluation/longbench/merged/nine_task_acceptance.json)。实际argv、源码hash、训练trajectory、fresh及退出均保留在各臂`plan.json`／`run.json`及其引用的artifact报告，不伪造commit ID。

本批决定：half和pure_gold在本父／32步／既定预算结束，不追加。full_teacher保留为本批数值最高的综合候选，double保留“知识略升、综合近持平且付出LB代价”的探索事实；full与double仅差0.0130430213pp，单seed结果不构成稳健胜负，不为当前阶段报告追加复测或交叉网格。原1×条件目标继续作为控制，当前不正式替换、不延长同配方。full_teacher同时改变支持集与教师M32执行，不能把差值解释为纯支持集因果；double的M正差也不等于恢复路线已经定稿。四臂均距九项>67.1471449179质量边界有明显差距，跨父、跨模型和原始BF16整链仍未验证。阶段交付候选与后续工作统一由[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)安排；保留`full_teacher/final_model`一份为本批阶段交付成品（质量未达，不切换默认控制），不把该BF16终点当exact-resume源；half／double／pure_gold的完整评测与比较用途已结束，按下述精确清理收尾。

收尾：half／double／pure_gold各删除已用完的`final_model/pytorch_model.bin`与`same_input_output.pt`，共6文件，释放14077685760B（allocated）。full_teacher的一份`final_model`保留作本批阶段交付；原C1 raw、Muon32控制、共同C0和在途E全部保护，未删除shared数据／cache。删除前核E1007源码pins、E及6／7线程实际plan/request和选中产物FD/mapping；删除后E pins全同，753份配置/meta／源码绑定／核心日志／指标预测SHA及受保护权重stat均保持。精确路径、原因、大小、运行退出与前后核验见[清理receipt](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/cleanup_terminal_20261103.json)；历史run/acceptance不改，已清三臂的成品路径仅留证据，不能再称可加载权重。

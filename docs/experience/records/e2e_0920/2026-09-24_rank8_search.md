# 0920 checkpoint 的 rank8 DP 顺序超参数搜索（2026-09-24）

## 目标与当前状态

目标：从用户当前 `compressed_e2e_fintuning/scripts/e2e_decoder.sh` 指定的同一初始 checkpoint 出发，在用户明确的 rank≤8 约束内，用 rank8 LoRA 将既有八任务全量、0-shot 均分提高到至少 69%。用户已将当前三组学习率试验作为先行实验，授权其结束后自动接续七天顺序搜索，搜索范围为优化参数及已有损失权重；初始模型、数据、`loss_type`、DP规模和rank8保持不变。有限顺序搜索不保证找到全局最优或必然达到69。

**运行状态（2026-09-30 20:14，Asia/Shanghai）：先行三组已结束，长程搜索PID710033运行中，前5/6个坐标完成，11组已到5000步；最后一项KD/CE权重中，alpha=.99已到2500步，alpha=1.0仍在首段训练。结论状态：当前最高为step2500训练内全量八任务均分68.009998%，尚未达到69；该最佳点还没有对应的最终导出重载复评分数。** 本文更新为阶段性结果，不将整轮搜索写成已完成。最后一轮晋级、领先快照重载比较和最终清理仍由既有脚本继续执行。

历史启动说明：2026-09-24已按用户授权停止`only_lora/Qwen_Qwen3-8B_20260924_185517`早期试跑，保留证据；先行PID701847完成后，长程于2026-09-25 01:59自动接续，先导出并重载评估先行最佳可用1000步模型，再续训。新增路径的真实执行成功，见下文；没有因本次总结重新启动验证或改动训练。

执行工作区为当前 IDE 会话的 `/root/VAELLM`，环境为已确认的 `bitvae`。本机硬件为 4 张 A800-SXM4-80GB，环境来源见 [AGENTS.md](../../../../AGENTS.md)。旧 [0910 搜索记录](../e2e_0910/2026-09-24_rank8_search.md) 的初始模型和远程 GPU 4–7 均不适用于本轮。

先行实验入口为[run_search.py](../../../../experiments/e2e_0920_search/run_search.py)，后续自动搜索入口为[run_long_search.py](../../../../experiments/e2e_0920_search/run_long_search.py)，有限搜索规则见[search_policy.py](../../../../experiments/e2e_0920_search/search_policy.py)。输出分别为：

```text
/root/data/ckpts/result/compressed_e2e_fintuning/e2e_0920_search_20260924
/root/data/ckpts/result/compressed_e2e_fintuning/e2e_0920_long_search_20260924
```

先行实际后台命令为`python -u experiments/e2e_0920_search/run_search.py`，在已激活`bitvae`的shell中通过nohup启动；总日志为先行结果根目录下`search.log`。实际三组命令、源码SHA256和运行状态保存于`search_manifest.json`。后续实际命令如下，同样由bitvae调用者以nohup、断开stdin及独立session后台启动：

```bash
python -u experiments/e2e_0920_search/run_long_search.py --scope loss_weights --budget-days 7 --predecessor-pid 701847 --predecessor-start 192796583
```

后台PID710033，`/proc/PID/stat`启动时间为193972417；曾等待先行PID701847、启动时间192796583所标识的同一进程正常退出后接续，等待期间不占GPU。新结果根目录下`long_search.log`、`long_search_manifest.json`、`leaderboard.csv`保存阶段命令、源码hash、参数及指标。`best_result.json`截至本次核查仍是首个已重载基线66.554029%，并非搜索阶段最高68.009998%；最后重载比较后才更新最终交付模型。七天软截止为2026-10-02 01:59，可在有限计划完成后提前结束。

先行首组正式配置已确认rank8、5000步总调度、warmup150、stop1000、save500、eval_limit=None，无短测步数或题数限制残留。初次后台交付时仅检查至10步；当前结果来自用户再次唤起后对既有正式指标的只读核查。验证配置、核心日志和原始短测指标保存在先行根目录的`verification/continuous/Qwen_Qwen3-8B_20260924_190454/`，验证范围、数据路径/大小/mtime和历史清理结果归并在`verification/verification_summary.json`。

## 阶段性结果与经验（2026-09-30）

以下14组已完成首段的最高均分均在step2500。每个坐标只相对当时的基准配置改变该参数；跨行不得忽略此前已继承的其他参数。所有分数均为八任务全量、0-shot等权均值，单位%；5000步一列来自最终导出评测，与2500步训练内评测分开标注。

| 候选 | 本坐标取值 | step2500训练内均分 | step5000导出均分 | 已完成步数 |
| --- | --- | ---: | ---: | ---: |
| `pilot_lr1e4` | LoRA LR `1e-4` | 67.725621 | 67.483396 | 5000 |
| `pilot_lr3e5` | LoRA LR `3e-5` | 67.601846 | 67.329673 | 5000 |
| `pilot_lr3e4` | LoRA LR `3e-4` | 67.668557 | 67.324585 | 5000 |
| `c01_lm_head_lr_1` | head LR `3e-5` | 67.751340 | 67.330410 | 5000 |
| `c01_lm_head_lr_2` | head LR `3e-4` | 67.422665 | 67.275246 | 5000 |
| `c02_norm_lr_1` | norm LR `3e-5` | 67.611610 | 67.445731 | 5000 |
| `c02_norm_lr_2` | norm LR `3e-4` | 67.792827 | 67.594043 | 5000 |
| `c03_lora_dropout_1` | dropout `0` | 67.814147 | 67.596396 | 5000 |
| `c03_lora_dropout_2` | dropout `.05` | 67.883982 | 67.374186 | 5000 |
| `c04_hidden_loss_weight_1` | hidden `0` | 67.805292 | 67.462625 | 5000 |
| `c04_hidden_loss_weight_2` | hidden `.03` | 67.757841 | — | 2500 |
| `c05_pre_mlp_hidden_loss_weight_1` | pre-MLP `0` | **68.009998** | 67.470583 | 5000 |
| `c05_pre_mlp_hidden_loss_weight_2` | pre-MLP `.03` | 67.974884 | — | 2500 |
| `c06_alpha_1` | KD alpha `.99` | 67.688203 | — | 2500 |

未完成首段的`c06_alpha_2`为KD alpha=1.0，其已落盘step2000均分67.234925；不能把尚未完成的2500/5000步结果计入共同预算比较。三次额外晋级额度已用于head、norm和dropout坐标，其余未晋级分支不等于已经验证长期上限。

**当前最佳配方与用途。** `c05_pre_mlp_hidden_loss_weight_1`沿用rank8/LoRA alpha16、LoRA LR`1e-4`、norm LR`3e-4`、head linear LR`3e-5`、dropout`.05`、hidden`.1`、pre-MLP`0`、KD alpha`.95`，其余固定条件见下文。最佳权重位于新结果根目录的`trials/c05_pre_mlp_hidden_loss_weight_1/Qwen_Qwen3-8B_20260929_172447/trainer_state/checkpoint-2500`，保留用于收尾导出复评；原始分数为同run的`lm_eval/lm_eval_results_step_2500.json`。第二、第三名为pre-MLP`.03`的67.974884与dropout`.05`、pre-MLP`.01`的67.883982，各自最佳checkpoint亦仍保留。

**优化参数收益有限且非全任务同向。** 相比`pilot_lr1e4`，head/norm LR及dropout三坐标累计增益为0.158361个百分点，再关闭pre-MLP增加0.126016，合计0.284377。最终配方与原基准同时差四项，不能把总增益归给某一项。step2500逐任务变化（百分点）为BoolQ−0.611621、RTE+1.805054、WinoGrande+0.631413、ARC-Easy+0.168350、ARC-Challenge+0.938567、OpenBookQA−1.000000、PIQA+0.272035、MMLU+0.071215；六升两降，RTE对均分贡献+0.225632，占总增益的大部分。单seed的小差值尚不能证明稳定排序。

**保留中途最好点比延长训练更有证据。** 11组完成5000步的候选，其所有已观测训练内评测峰值均在2500步；2500→4500仍使用同一训练内评测路径，却全部回落0.061675–0.534569个百分点。最佳组4500步为67.532376，较2500步下降0.477622，因此回落不能只归因最终导出。新候选在2500暂停，但先行三组直接1000→5000、没有2500暂停，也在该点评测最高；不能仅凭峰值位置将退化归因于暂停恢复。机制尚未独立定位，不认定已过拟合、精确最优步数恒为2500或rank8的上限为68。后续可研究最佳区间和学习率衰减，但不因这条假设修改当前运行。

**两种hidden对齐应分开判断。** 相同2500预算下，hidden`.1`改为`0`或`.03`均未超过67.883982的参照；保持hidden`.1`而将pre-MLP`.01`改为`0`或`.03`则分别提高0.126016和0.090902个百分点。后两者只差0.035114，且`.03`没有5000步结果，不能断言所有正pre-MLP权重有害或完全关闭稳定更好。KD alpha`.99`截至2500步也未超过`.95`参照，纯KD仍运行；未完成的损失坐标不作最终结论。

**导出和续训已有真实证据，精度目标尚待交付验证。** 首次`pilot_lr3e5`的step1000训练内均分66.564804，独立导出、严格重载后为66.554029，相差−0.010775个百分点；对应`deployments/pilot_lr3e5_step1000/export.log`、`model/export_record.json`及`evaluation/lm_eval/lm_eval_results_reloaded.json`。现已有三组1000→5000和八组2500→5000真实续训完成，支持这些调用链可运行，但没有连续训练/恢复的逐位对照，不宣称已完成exact-resume等价审计。当前68.009998的最佳点仍待自己的导出重载，不以早期基线验证替代。

以上经验归入[对照设计](../../lessons/experiment_design.md)、[KD与数据边界](../../lessons/kd_and_data.md)、[Checkpoint生命周期](../../lessons/checkpoint_lifecycle.md)。全部配置、各任务分数、阶段退出码和源码hash以`long_search_manifest.json`、各run的配置/日志及原始JSON为证据；这是一轮固定模型/数据/单seed、尚未结束的搜索，不是69不可达的证明。

## 初始模型、已有结果与参考经验

固定初始模型：

```text
/root/data/ckpts/result/catlora/remaining_lora_mass/Qwen_Qwen3-8B_20260920_095822/final_model
checkpoint_id: 1a6fa98c-6685-4dab-a222-e03695919bfb
base_model: Qwen/Qwen3-8B
```

该模型 CAT 最终评估的八任务均分为 **65.56%**，距 69 约 3.44 个百分点；原始证据为相邻 `linear_by_category.log` 第 3209–3232 行的最终评估和保存记录。按脚本设定，初始模型、压缩范围和 packed bits 均不改变；本轮不重新压缩，不将名义位宽或参数量推算成实测压缩率。

仅据初始 `checkpoint_meta.json` 的shape/dtype核算，252个projection的非保护区域中，q/k/v/o/gate/up为 **2 bpw**，down_proj为 **4 bpw**；均为两stage，down每stage用64个bit解码32个权重，不能按stage数统一称为2bit模型。主packed码2,155,511,808 bytes，加INT8保护值97,910,784 bytes、BF16 scale39,168 bytes及INT64索引156,672 bytes，共 **2,253,618,432 bytes（2.098846 GiB）**；分母取这252个projection原有的6,945,767,424个权重，为 **2.595674 bpw**。核算只计`stage_vq_weights`一次，不重复累加其首stage别名`vq_weights`。这不是整体模型位宽或磁盘文件大小：未计decoder、未压缩的embedding/词表head/norm、序列化元数据，以及本轮新增可训练参数和optimizer等训练状态。

本轮全252个projection的rank8 LoRA参数量据原始输入/输出维度求和为 `8 × Σ(in_features + out_features) = 21,823,488`；FP32训练参数本身为 **87,293,952 bytes（83.25 MiB）**，不含梯度、optimizer、激活或临时buffer。`lm_head_train_mode=linear`另外添加4096×4096的post-norm无bias映射，共16,777,216参数、FP32为64 MiB；原词表输出头训练时冻结。该映射在最终化时可融合进原head，故训练期增加的参数字节数不能直接当作最终导出模型新增磁盘体积。以上来自metadata/config和当前 [head实现](../../../../e2e_common/post_norm_head.py)，未加载权重，也不代替真实显存测量。

用户打开的旧运行 `only_lora/Qwen_Qwen3-8B_20260919_011304` 实际来自 `remaining_lora_mass/Qwen_Qwen3-8B_20260910_180322/final_model`，并非上述 0920 checkpoint。其八任务均分在 step3000 达到 68.117150%，step4000–9000 为 67.72–67.86%；训练日志到 step10000，但没有对应最终评估或导出完成证据。该结果说明应观察中途最优点，不能仅凭增加步数预期达到 69；它不能替代本轮同初始模型基线。原始指标在该运行的 `lm_eval/lm_eval_results_step_*.json`。

同一 0920 初始模型的 `181434` 使用 residual `replace`，至230步总 loss 约6.87；`184425` 使用 residual `none`，至70步 loss 约0.42。两者均没有下游指标；replace 改变初始残差信息通路，不能把前者当普通 LoRA 的失败证据。`185517` 将 head 从 LoRA 改为 linear，与当前脚本一致；它已按本轮用户授权停止，其历史结果不与本轮混用运行标识。截至核查，`only_lora` 下使用当前 0920 checkpoint 的八个运行均未发现下游评估 JSON。

本次参考并遵守以下经验：

- [对照设计](../../lessons/experiment_design.md)：初始模型、数据、总调度、batch、seed 和评测口径固定；1000步是筛选点，不是收敛上限；训练 loss 不能替代下游准确率。
- [残差 LoRA](../../lessons/residual_lora.md)及[原始对照结果](../docs/exp_results/residual_lora_ab_20260923.md)：另一初始模型的 residual additive 加双 hidden 对齐仅有集中在 RTE 的小幅收益，不能照搬为本轮增益。首轮保持 residual none，不加入新残差模块。当前 hidden .1/.01 与旧 .1/.1 配方不同，不据旧结果断言当前设置无效。
- [KD 与数据边界](../../lessons/kd_and_data.md)：核对当前源码的loss和mask语义及实际数据身份；先行LR对照固定损失，后续已有权重逐项变化，避免把LR收益与训练目标变化混成单因素结论。旧0910/旧错误数据上的CE退化不能直接否定本轮已修复数据及不同权重下的候选。
- [Checkpoint 与数值路径](../../lessons/checkpoint_lifecycle.md)：区分保存、恢复、评估回训与导出的证据范围；本轮已覆盖部分真实路径，但不把未执行的GPU恢复对照报告为通过。

待验证假设：普通LoRA、head和norm的合适更新强度可能不同，已有hidden/pre-MLP及KD/CE权重也未必最优；用匹配训练预算、一次只变一个坐标的全量评测选择下一组基准。head模式、decoder联合更新、`loss_type`和数据版本/比例不在本轮搜索范围。七天只搜索一轮坐标，变量交互及不同seed的稳定性仍属于结论边界。

## 首轮配置：只搜索普通 LoRA LR

三组均从固定初始 checkpoint 新开训练，不从旧探索任务权重接续。每组使用本机 0–3 四卡 DP、每卡 batch8、梯度累积1，有效 batch32；由已启动入口依次运行，前一组失败即停止后续启动，不自动重试。

| 组别 | 普通 LoRA LR | norm LR | head linear LR | 首段步数 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | `1e-4` | `1e-4` | `1e-4` | 1000 |
| 较低 LR | `3e-5` | `1e-4` | `1e-4` | 1000 |
| 较高 LR | `3e-4` | `1e-4` | `1e-4` | 1000 |

只改变普通 LoRA 参数组的 LR；norm 和 head 保持当前脚本的独立 LR，不能将这一比较称为所有可训练参数统一 LR 的搜索。

三组固定条件：

- `train_mode=lora`，decoder 与 bits 冻结；0–35层、全部压缩 projection；普通 LoRA rank8、alpha16、dropout0.1；residual none；norm all；head linear。此处linear是hidden_size×hidden_size的post-norm可训练全秩映射，原词表head冻结；与旧head LoRA的训练范围不同。
- LoRA/norm/head 参数按当前 `distill_fp32_components=lora,lm_head,norm` 保持 FP32 存储，BF16 计算；不更改已有精度路径。
- `steps=5000`、cosine、warmup150；`stop_after_step=1000` 只结束首段，不将总调度改成1000步。weight decay .001、max grad norm1.5，seed/data_seed均为0。
- 序列上限1024、dynamic padding开启；训练入口按混合 iterable 数据的实际行为处理 `group_by_length`，不更换采样规则。
- 数据固定为`edgerazor_ii_7m=.341,edgerazor_ii_gen=.067,edgerazor_tulu=.028,edgerazor_am=.064,vaellm_eval_task=.500`，`dataset_task=lm`；沿用先行实验实际文件，task文件身份已核实如下。
- `kd_top_partial`、K100、T1、alpha .95、prompt weight .3、hidden .1、pre-MLP hidden .01、linear_depth；保留当前teacher/offload/checkpoint配置。
- 每500步保存并执行全量八任务0-shot评估；每组首段产生 step500、step1000 两个正式评估点，共六次。首段不使用短测题数限制，不因暂停改写同一步的评估口径。

当前实现中，`kd_top_partial` 是 `0.95 × KD + 0.05 × CE`。`dataset_task=lm` 使全部有效 token 具有可训练 label，KD 按 label 识别出的 prompt 区域为空，因此当前 `prompt_loss_weight=.3` 没有对问题部分降权；日志 prompt token 数为0与此一致。首轮保持该已在用语义，不将它误写成SFT或响应优先的目标。

本次只读核查实际task文件`/root/data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl`，191870906 bytes、176597行，mtime_ns为1789830247000000000，SHA256为`894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1`。SHA完整匹配[0910记录中的已修复数据](../e2e_0910/2026-09-24_rank8_search.md)，不是从行数或最新生成器推断版本。该记录有全部2490条RTE与官方train逐条核验，以及MMLU固定revision`c30699e8356da336a370243923dbaf21066bb9fe`的auxiliary_train来源证据；当前抽读第74266–74268行RTE题干非空，答案False/True/True符合`premise/hypothesis`与0→True、1→False规则。旧82021行文件的空题干/反标签问题不适用于当前文件。

边界：本机未找到上述生成manifest或官方Arrow原件，当前生成器hash也不同于记录中的生成器；本次仅以数据内容hash同一性复用既有来源验证，没有重新生成数据或开展全量来源/去重审计。没有已证实bad RTE或split问题阻断本轮搜索，也没有证据证明第三方EdgeRazor不存在测试重叠。`vaellm_eval_task=.5`是样本采样权重，不等于50%的有效训练token贡献，本轮不调整该比例。

## 七天顺序搜索与评测

主指标为 BoolQ、RTE、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA、PIQA、MMLU 八项分数的等权算术平均。BoolQ/RTE/WinoGrande/MMLU 使用 `acc,none`，其余四项使用 `acc_norm,none`；MMLU使用lm-eval聚合结果作为一项。固定0-shot、`eval_limit=None`、`eval_hif4_act=false`，不混用部分题目分数；本轮不额外引入PPL目标。

用户新授权替代此前“先行三组结束后等待人工选择”的安排。七天预算从先行实验全部成功结束、后续搜索开始计算，不包含排队等待。所有训练仍为本机0–3四卡DP、每卡batch8、累积1、有效batch32、seq1024和rank8；每500步全量评估八任务，保存中途最好点，不只比较最后一步。

1. 先从三组先行候选中选出截至1000步的一个最佳可用checkpoint导出，并用独立新进程严格重载后执行全量八任务评估。这是新导出/重载路径的首次真实任务，先行GPU任务结束前不抢卡短测；导出或重载失败即停止搜索，不继续消耗多天预算。
2. 三组先行候选均从各自1000步训练状态续训到5000步，保持原总5000步调度、optimizer/scheduler/RNG、world size、batch及seed，额外共12000步。按共同5000步预算内的最高均分选择当前基准，不用1000步短程排序直接淘汰某个LR。
3. 依次搜索下表六个坐标。每个坐标以当前基准配置为参照，两项新值各从同一0920初始模型训练到2500步；先选较优新候选推进到5000步，再与已有5000步基准比较，赢家作为下一坐标的参照。新候选继承此前选出的参数值，不继承其训练后模型；每次只改变该坐标。
4. 全搜索额外允许最多3个候选从2500推进到5000。与同坐标领先新候选的最佳均分差≤0.25pp，或差≤0.75pp且1500→2500步实际端点均分上升≥0.25pp时，可在剩余预算足够的情况下追加。阈值仅用于分配预算，不是统计显著性或长期上限判据。
5. 完成有限计划或到达预算收尾点后，将仍有checkpoint、按搜索阶段分数排名前三的快照重新导出、独立进程严格重载并执行同口径全量八任务评估；前三可以来自同一配置的不同step，5000步分数来自已有最终导出评测。已执行过相同checkpoint重载评估的结果可复用。最终在这些实际重载评估的模型中选择并保留一个最佳模型，写出全部候选比较和`best_result.json`。首次达69不提前取消已安排的有限坐标搜索；未达69也如实交付已执行范围内最好结果。

| 顺序 | 变化参数 | 先行值 | 两个新候选值 |
| --- | --- | ---: | --- |
| 1 | `lm_head_lr` | `1e-4` | `3e-5`、`3e-4` |
| 2 | `norm_lr` | `1e-4` | `3e-5`、`3e-4` |
| 3 | `lora_dropout` | `0.1` | `0`、`0.05` |
| 4 | `hidden_loss_weight` | `0.1` | `0`、`0.03` |
| 5 | `pre_mlp_hidden_loss_weight` | `0.01` | `0`、`0.03` |
| 6 | `alpha`（KD/CE权重） | `0.95` | `0.99`、`1.0` |

`loss_type=kd_top_partial`、K100、T1保持不变；最后一项分别为0.99KD+0.01CE和纯KD，hidden与pre-MLP仍由各自权重决定。LoRA alpha固定16，不能与KD的`alpha`混淆；weight decay .001、warmup150、prompt weight .3也不在这轮六坐标中变化。损失权重搜索属于本次明确授权，结果不能全部归因为优化器参数改善。

1000步接近的分数不足以宣称显著优劣；仍在上升的候选也不因尚未达到69而被解释为已到上限。若某配方出现已确认的数值失稳、训练/评估/保存失败，则停止依赖该运行的后续步骤并调查，不能降低正式配置绕过失败。

达到69以最终交付模型严格重载后的全量八任务均分为准；训练内、短测与导出重载结果分别记录。反复用同一八任务分数选择超参数存在对该评测集合的选择偏差，不能将最高值宣称为独立留出集上的泛化保证。本轮不改变既有评测口径，不以缩减题目数、改数据或临时放宽标准绕过失败。

## 已执行验证、限制与成本

2026-09-24启动前已运行的GPU短测使用真实0920全模型、四卡DP、batch8×acc1、seq1024、rank8/alpha16、head linear及当前训练功能，仅缩短步数、保存/评估触发周期并限制短测评估题数。当时的实际结果如下；后续真实续训和重载证据见前述阶段性结果：

- 连续4步路径完成中途评估、评估后回训、最终导出及八任务`limit=1`的流程，退出码0。该结果支持已执行流程可运行，不是正式精度结论。
- 分段路径完成到step2并以paused状态退出，退出码0。
- 当时未执行暂停后的resume，也未完成连续与恢复路径的GPU exact-resume比较；后来真实续训成功不等于补做了该等价性比较。
- CPU strict reload尝试因FlashAttention2要求CUDA而无法运行；没有切换注意力实现绕过。这是该CPU验证环境不适用，不能据此判断checkpoint损坏，也不能报告strict reload通过。

用户明确要求停止追加运行前验证后，先行实验保留以上证据及限制直接启动，不把未执行的GPU恢复比较写为通过。后续自动搜索也不重复GPU启动前短测；其真实新增导出/重载路径先在首项任务执行，实际续训状态由各阶段配置、日志和checkpoint核验。正式搜索使用全量评估，不带短测`limit=1`，短测指标不参与候选排名。

已完成 CPU 相关回归检查：在 `bitvae` 中执行 `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_e2e_stage_stop.py tests/test_e2e_mid_eval_fp32_wrapper.py tests/test_e2e_runtime_v6_finalization.py`，结果为 **12 passed，7.32秒**。这些检查覆盖相关已有测试，不等于四卡真实模型验证通过。

新增编排检查为`tests/test_e2e_long_search_policy.py`、`tests/test_e2e_long_search_artifacts.py`和`tests/test_e2e_long_search_workflow.py`，**20 passed，0.30秒，仅CPU，2026-09-24执行**。覆盖有限晋级、全量指标与checkpoint身份、精确清理范围、覆盖CLI参数及真实临时进程退出接续；这些CPU结果本身不证明GPU导出有效。当前bitvae的Python/libc没有`pidfd_open`封装，已使用本机Linux x86_64确认可用的内核调用完成真实退出接续测试。真实导出及fresh-load后来已在先行step1000模型执行成功，不再标记为尚待首次执行；当前领先快照仍须各自复评。

后续基础预算为`3×4000 + 6×(2×2500 + 2500) = 57000`额外训练步，不重复计算先行3000步。按含每500步评测和保存的约8.1秒/步粗估，基础计划约5.3天；最多3次额外晋级另加7500步、约17小时。预留2小时用于最后导出/重载评估和汇总；这些均为计划估算，hidden权重关闭、数据长度、加载/导出成本和系统负载会改变实际耗时，不能承诺准确结束时刻。

控制器按实际阶段耗时更新估算，开始新阶段前判断剩余预算是否足够并保留收尾时间；七天属于软截止，完成已经开始的阶段后汇总，不强杀正在训练/保存/评估的进程。若实际慢于估计，可以少执行后面的坐标；不得声称未运行配置已比较。失败时记录错误并停止，不自动重试、不静默降配。

先行及后续入口均通过nohup启动，stdout/stderr落盘并记录PID、先行进程身份和输出根目录。排队已于2026-09-25自动结束并接续正式训练；启动交付后停止主动监控，仅在用户再次询问/要求总结时读取现有证据，不等待七天搜索完成，不在最后一个任务完成或失败后追加占卡。

## 证据保留与收尾

先行和新候选输出独立，原始数据、初始checkpoint、既有`185517`结果及其他结果不覆盖。先行1000步训练状态明确用于自动续训至5000；搜索中保留需要续训的最新状态、每个候选最佳点及全局前三点用于最终重载比较，清理已无后续用途的其他checkpoint/重复导出。搜索正常收尾后只保留最终选择的一个交付模型，以及生效配置、核心日志、原始指标、复现入口和搜索汇总；删除其他已结束、无后续用途的生成权重，精确路径与字节数归入manifest。若失败则保护尚待诊断或已安排后续工作的产物，不把未执行清理报为完成。

已归并短测配置、核心日志、原始指标、parity与退出状态；删除连续短测的final_model和trainer_state、已结束的staged短测输出、一次性诊断脚本、重复launcher日志及两份对应W&B测试产物，共删除5,751,393,055 bytes（约5.36 GiB，按文件逻辑大小计）。未跟随W&B符号链接删除外部缓存；正式搜索、初始模型和原有试跑记录均保留。精确清单见verification_summary.json。

经验增量（替代2026-09-24“尚无新增精度经验”的启动结论）：已形成固定条件下分参数组更新强度、hidden/pre-MLP分别对照、中途最佳保留及训练内/重载分数分离的阶段性证据，见上文和三个经验主题。其收益有限、单seed且并非所有任务同向，未证明69不可达。数据hash核查仅复用既有RTE修复证据，不新增数据修复收益的因果结论。后台已按manifest清理无后续用途的中间状态，当前最好快照及后续续训状态仍保留；整轮尚未最终收尾，本次总结不额外删除运行产物。后续结果持续更新本文，不另建进度版或最终版。

文档检查：已按当前标题刷新`docs/INDEX.md`；本轮记录相对链接有效。全库历史/迁移缺失链接与本轮运行无关，未扩大范围修补；文档检查不启动GPU任务。

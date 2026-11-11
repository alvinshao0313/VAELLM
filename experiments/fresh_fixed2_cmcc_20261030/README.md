# 新 BF16 共同父上的普通两阶 C0

本入口实现下一批指定控制 C0：只消费 [fresh common](../fresh_common_cmcc_20261030/README.md) 本次正式 `after_o`／B gate `raw2`，把全部36个 gate 通过已有 `native_prefix.from_native` 转为普通 fixed2 compact，严格保留其余144 native＋72 live dense。随后完成原 gate500→自身 live up S2×10K／500→自身 live down S2×10K／500，最终真实保存216 native＋36 compact＋0 dense、实际逻辑位宽≤2.6；再执行该 UUID 自身的完整 full8 和 LongBench21／4750。

**当前结果（2026-10-31）：C0训练、独立fresh和完整九项均已完成，sequence／训练／评测exit均为0。九项63.5841659502，logical2.4987414816、packed2.5121161419；质量目标未达，位宽通过。** 正式成品与下游BR36的同父核对见下节；直接Muon是另一个端点，不计入本页C0分数。

本支是现有“至多两个基座×追加恢复开／关”预算中的一条基座后缀与一份完整九项。没有第三阶拟合、Rice稀疏载荷、额外 gate10K、追加32步Muon或其他恢复；不依赖旧dynamic／Rice的胜者。历史执行使用GPU5训练、GPU6／7评测；这是已完成任务的资源记录。最新授权下GPU6／7属于独立实验，本线程后继仅使用0–5，不能复用历史卡号启动新任务。原312 pins和共同任务源码保持只读；不占卡、不轮询、不重试。

## 完整结果与同父边界

正式UUID为`cbb15c9f-1c1f-4394-817c-65c226d76ce7`，模型位于`/data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/final_model`。父源是本次`after_o` `7eeb82b4-3868-4cc5-9fa3-51870ee55894`／`raw2` `f4669891-9028-4665-85a5-71c26ac45c33`；没有借用旧B或Rice后缀。pipeline完成gate／up／down各500恢复及自身两类S2×10K，后缀用时55433.20秒；最终独立fresh核3675字段和同输入BF16 logits exact。

| 指标 | C0正式结果 |
| --- | ---: |
| 完整九项 | 63.58416595020284 |
| 完整八项 | 66.3144567034778 |
| MMLU | 62.39851872952571%，8762／14042 |
| LongBench21 | 41.74183992400309 |
| logical／packed bpw | 2.4987414816151494／2.5121161419412363 |
| 相对同协议BF16九项68.04714491790848 | 下降4.462978967705648个百分点 |

252个压缩投影为216 native＋36 fixed2 compact、0 dense、rank0。账本包含实际decoder、scale／保护／旋转及持久frame；tensor共4671007680字节，完整目录4674030159字节。logical与packed不是全目录文件字节的等价换算。严格掉点<0.9未达，不将质量`not_met`记为技术失败。

原件为[完整九项及位宽账本](../../result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/single_20261030_v3/longbench/merged/nine_task_acceptance.json)、同评测目录`evaluation.json`和`full8/`；训练／fresh原件在上述模型父目录的`report.json`及`strict_reload.json`，controller记录在同级`single_20261030_v3/run.json`。本轮只读复核全部64份full8逐题文件（24742条）、21份LB预测（4750条）的SHA及样本数，重新汇总full8与九项；LB使用已核SHA的官方scores，不称重新执行评分器。与BR36逐题输入身份及协议完全配对。acceptance SHA256为`ab525a60b0bec491ba83ccd3d13a976afb4265c574da0203925bc678f314cb5f`；实际metadata SHA256为`de7436ffb9694089688ca9bb039097747cd057136c11c482c71e8a6a1ef67ee0`。

[同父BR36完整中点](../block_reconstruction_sequence_cmcc_20261031/README.md#正式br36结果与同父比较)九项增加0.2140486573个百分点，位宽不变；这是追加1152次块更新后的观测增量，未达质量目标，也不证明BR会帮助后接Muon。直接E与BR→E须各自完成九项后才能比较，不把尚未得到的收益相加。

## 来源、对照和预算

`sources.bind`必须看到同一目录正式 `common_prerequisite_complete`／exit0、原shell真实exit0、两原生phase exit0、新after_o／raw2事件、metadata/runtime SHA和实际权重stat；验证共同父没有gate恢复且与after_o的非gate补集exact。不会从旧archive、旧UUID或short权重重建正式父。原raw2完整strict加载后仅替换36gate，记录原非gate字段digest、实际raw权重SHA与三库存。

配方由未改的fresh B入口重新解析，用真实raw2 `cat_runtime_state.pt`核对七类别runtime identity并恢复本次512×256校准。五源train、B4/max1024、dynamic padding、gradient checkpointing、原teacher/top100/T1/prompt.3、FP32 D＋仅未来dense rank12、原500日程与250／500保存直接复用。临时dense LoRA沿原BF16融合规则结束；交付成品rank0。

未来up/down每类别在原N fitter前显式seed31，shuffle分别500031／600031，并记录两个stage实际optimizer创建时的参数digest、形状、dtype和LR。未来C1须遵相同规则并拟合自己的live权重；不会借C0后缀、父相关统计或轨迹。C0使用两个实际asymmetric D64/K64 gate decoder，length全部为2，完整持久frame按实际存储计入预算；没有Rice或第三decoder费用。最终完整logical／packed／全目录字节仍由未改的mixed checkpoint账本实测。

| 本支新增计算 | 固定预算 |
| --- | --- |
| gate额外拟合／第三阶／Rice | 0 |
| gate／up／down原恢复 | 各500更新，共1500 |
| 自身live up／down VAE | 各S2×10K，共40000更新 |
| 新父接线验证 | 0训练更新，一次真实模型finite／保存／独立fresh |
| 完整质量 | 一份full8，64 leaves／24742样本；一份官方21任务／4750样本 |

共同BF16前缀与raw2成本另由共同任务记录，两支共享时不得重复记为两次生成；C0新后缀和自身评测分别记录实际时间。完整C0质量结果已在上节核实。

## 验证范围和当前状态

复用[普通fixed2真实恢复、保存回训、独立fresh与live-up证据](../adaptive_native_model_20261029/README.md#nv2-混合恢复保存回训与真实-next-up-交接)，以及[dynamic／Rice已通过的最终无dense和消费者合同](../cmcc_runtime_20261030/README.md)。依据[同父与实际live后缀边界](../../docs/experience/lessons/compression_choices.md#顺序先验live目标与逐类别旋转入口)，不把旧后缀、局部误差或已运行声明当作本次完整结果；本次新缺口是新父加载、runtime和来源绑定。

启动前CPU核证见 [cpu_v2/validation.json](../../result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/cpu_v2/validation.json)：七类实际配置／恢复与已验证共同配方相同，runtime identity相同，实际short_v2及当时仍运行的formal均被拒绝作为可用正式父，保存的真实short metadata合法；CUDA未初始化。初次CPU检查只因JSON中的list与Python tuple直比而失败，规范化JSON表示后通过，没有改变算法或配置。short_v2权重此前已清理，本次没有重跑七类short。

10月30日20:37的首次正式子进程在任何训练更新及模型产物生成前退出1：`run_pipeline→continuation→artifact→sparse`延迟导入`train_utils.cat_train_residual_protection`时，helper此前把仓库S插入`sys.path`前端，namespace的动态`__path__`随之让该模块来自S，后续`recipe.parse→enable_snapshot`正确拒绝。旧CPU只导入`sources`，未覆盖这个完整顺序。原日志和`single_20261030_v2/{plan,run}.json`保留；[原顺序复现](../../result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/import_repair_20261030/original_failure.json)保存实际调用栈、唯一非N模块、namespace搜索路径及修改前SHA。原312pins没有本包文件，旧dynamic／Rice依赖与共同snapshot未改。

隔离修复沿用[已验证的namespace固定方式](../fresh_muon_cmcc_20261030/bindings.py)：在`sources`首次启用N后、任何旧helper导入前，显式把`train_utils`／`litebsq`／`rotation`／`e2e_common`的`__path__`固定到N，不改原模块、配方或算法。`validate_cpu`现先导入完整`run_pipeline`，消费已完成正式raw2 `f4669891-9028-4665-85a5-71c26ac45c33`，恢复原512×256 CPU校准并核七类配置／runtime；实际全部项目模块来自N，包括此前失败模块，CUDA未初始化，0训练更新、无模型权重加载或复制。22:06的[真实CPU证据](../../result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/import_repair_20261030/cpu_after/validation.json)及同目录上级`validation.log`／`validation.exit=0`已落盘，AST和源码pins包含新的`sequence.py`。旧plan已绑定修复前SHA，不能复用，须在独立输出重新prepare；新父GPU finite／保存／独立fresh及CPU／CUDA RNG契约仍由正式入口首次恢复前的零更新检查完成。本次CPU通过不冒称该GPU检查或C0恢复已通过。

**formal raw2及v3新父GPU验证均已完成。** `run_pipeline`在gate round_base真实保存后、任何恢复更新前，使用本父校准中的一个完整256token输入：整模预热、有限BF16输出、实际保存、独立新进程完整strict state与同输入BF16 logits exact。这只闭合新增来源接线，不做新的质量门槛或恢复短链。失败即退出，不开始500步；通过才使用该真实状态执行原后缀。每个恢复后保留原消费者独立fresh合同，最终才检查无dense和≤2.6。

启动前审查修正了检查本身的随机状态副作用：与已有比较入口一致，在真实模型重建后才设置seed31；新增readout包在CPU／CUDA RNG保存恢复上下文中，回到训练前实际断言两份RNG状态未变。真实已完成dynamic层载荷的CPU重建证明会消耗随机数，保护上下文后CPU状态exact，见[启动前审查证据](../../result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/prelaunch_review.json)。这项CPU证据不冒称新父GPU已验证；实际全模型CPU／CUDA检查在上述首次恢复前执行。单次runner还在取得flock后重新检查运行记录，防止并发启动覆盖；旧活动源码未改。

评测通过新隔离 `evaluate.py`适配真实 `C0_fixed2` provenance，核实际两阶length2、无Rice稀疏字段和0额外拟合，不填写旧`stage3_fit_actual_updates`。其余评测完整复用官方full8、full21、32K窗口及新机BF16绑定，不改严格<0.9／≤2.6验收。

## 复现入口

调用者激活`bitvae`，所有验证／实验通过tmux或nohup后台执行。共同任务已真实exit0；本次新plan使用`--completed-common`，直接核终态、原件和身份，不重新等待历史PID。旧`--common-shell-pid`仍用于尚在运行的共同任务，语义不变。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_fixed2_cmcc_20261030.sequence \
  --prepare /data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3 \
  --completed-common

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_fixed2_cmcc_20261030.sequence \
  --run /data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json
```

以上目录已实际使用，不能重复启动。旧v2已失败，原`plan.json`／`run.json`／核心日志／真实exit1保留；v3重新绑定当前1003项来源。准备命令已exit0，证据在`result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/import_repair_20261030/prepare_v3.log`及`.exit`。v3由tmux `fresh_C0_v3_20261030`执行，状态、日志及退出位于`artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3/`；模型输出为同级`single_20261030_v3_C0`，评测输出为`result/qwen3_8b_mmlu65_20261004/fresh_fixed2_cmcc_20261030/single_20261030_v3`。

runner启动时只核一次GPU5；完整成品经实际审计后只核一次GPU6／7，再执行本UUID的完整评测。资源占用或技术错误停止相应依赖；合法质量`not_met`作为完整结果记录。各子任务stdout/stderr和退出码落盘，不设守护、自动恢复或搜索。此次是查明故障、修复并验证后的人工新接续，不是失败自动重试。

以下为v3启动时的历史事实：真实GPU父源检查通过，fixed2 round_base UUID `55b6110c-84ac-4906-bb1c-62075a434610`，144 native＋36 compact＋72待压dense；独立进程完整strict state及256token同输入BF16 logits exact，CPU／CUDA RNG保持。原始证据为`single_20261030_v3_C0/04_gate_proj/raw2_strict_reload.json`及pipeline的`parent_loading_validation`。driver288378/start10671306、worker288496/start10671964；当时确认gate恢复到2／500，第2步改变11,025,571个master元素，峰值allocated40,881,532,928B，原500日程保持，随后停止主动监控。训练及完整九项现已完成，以本页正式结果为准。

正式共同after_o/raw2继续供C0／C1来源核对；C0成品保留给已安排的恢复配对及成品比较。修复经验归入[真实导入上下文](../../docs/experience/lessons/checkpoint_lifecycle.md#cpu构造原权重读取与真实导入上下文)；导入修复本身不构成质量收益证据，完整质量见本页结果。

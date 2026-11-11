# VAE-Muon raw的唯一原Adam500完整闭合

本包固定比较 `Mu_VAE/A_recovery` 与原控制 `A_VAE/A_recovery`。父仅 [VAE-Muon原raw](../cat_down_vae_muon_cmcc_20261103/README.md) UUID `c5ccf174-0c56-4f90-9e29-7d4ba63d2e23`，来自原C0 after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a` 后的D32/K48/M36/B8192/transpose=True/S2各10K。该raw完整八项66.3532649235，相对原Adam raw+0.3760823323pp，但M8743比原8787少44；局部weight/output NMSE分别下降4.06776409%/2.24480239%，均不能代替闭合结果。

唯一控制是原C0最终 `cbb15c9f-1c1f-4394-817c-65c226d76ce7`，原Adam500＋完整九项63.58416595020284。复用已完成上游，不再训练VAE、不接D16/Rice或类别Muon adapter，不将各臂差值相加。本次是一个预指定固定点：fresh原Adam500＋实际部署/独立fresh≤12h，随后同最终UUID完整九项≤8h；成功500后必须完成九项，不按loss、中评或M取消。失败停依赖，不自动retry/resume/占卡。九项≤C0即结束该固定点；微正也仅单seed候选，不自动扩七类或Mu+Mu，闭合后本分支新增预算为零。

## 原恢复合同与隔离

`run.py`直接调用成熟 `experiments.adaptive_class_20261030.continuation.recover`，该路径复用原 `MixedCatTrainer`。实际torch AdamW为betas(.9,.999)、eps1e-8、WD0；全252投影FP32 decoder master，共1728叶/5,336,064元素，LoRA0，BF16计算与部署。stage37/data31、原五源train/teacher/loss、B4/cap1024动态padding、clip1.5、decoderLR1e-5、原500 cosine/warmup.01、save250/500完整保持。optimizer fresh，不导入VAE optimizer state或类别Muon state。首step实际LR0会初始化Adam state；核实际第二步正LR更新后停止主动监控。

CPU仅验证新增父UUID/hash、真实252D拓扑、全部original_stage字段JSON归一后仅output_dir差别、原保存TrainingArguments及实际Adam构造kwargs/defaults/全部param_groups与原checkpoint500全字段相同。CPU构造原cosine scheduler后首LR0，因此与原500末端LR0的group字段可完整逐值比较；state仍空，不加载任何旧optimizer状态，也不执行训练更新。独立输出和reader只接本包formal最终UUID。成熟同拓扑证据复用，不追加GPU短验、exact-resume或保存审计。

`evaluate.py`复用原full8和LongBench消费者：64叶24742题，M B64/其它B32、0-shot/limitNone；LongBench原21项4750题/max32K，全部完成后同UUID九项。沿类别Muon已证明的绝对deadline方案，同时收紧detached LongBench native runtime与outer timeout；保留原20秒grace。CPU用原真实make_plan证明仅三项时间值变化，协议、数据、批量、终态绑定、评分口径不改。不能通过降低B64绕过资源问题。

原raw的logical2.4987414816151494/packed2.5121161419412363及full-model tensor4,671,007,680B在正式恢复前仅是本闭合的期望；下文已记录最终fresh后的实测全部组件与完整目录。账本包含decoder、码、scale、INT8保护、旋转、compact frame及packed padding，并单列linear之外embedding/head/norm，不以理论码率代替实测。

## 证据、入口与状态

CPU原件为 [prepared.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/prepared.json) 及同目录 `cpu.log/exit`。正式模型根独立为 `/data1/cmcc/VAELLM/artifacts/cat_down_vae_muon_closure_cmcc_20261103/formal/`，运行根为 `result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/`。唯一正式序列原件 [run.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/run.json) 记录实际命令、PID/start、GPU资源、源码SHA、日志、输出和退出状态；本次未提交，不伪造commit ID。

调用者shell激活bitvae，再在项目根执行；测试和正式均在nohup/tmux后台。GPU仅2，保留GPU0 D16、1 E256、3类别Muon、4 SciQ及6/7 S4原链及源码pins。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_vae_muon_closure_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_vae_muon_closure_cmcc_20261103.sequence
```

CPU已实际通过，29.698985秒/exit0，CUDA未初始化；严格新父UUID/hash/真实全252D、1728个FP32叶及5,336,064元素均核实。完整stage/data字段仅独立output_dir有差别，实际原Adam kwargs/defaults/全部param_groups无忽略字段、JSON归一后与原checkpoint500精确相同；fresh状态0，未导入两种Muon adapter。原C0真实config的21项4750题make_plan仅三项deadline时间字段变化，20秒grace保持。最初CPU在环境入口因tmux server未继承caller `CONDA_DEFAULT_ENV`即停，未加载模型/触CUDA；启动命令明确传caller bitvae后通过。失败环境散文件2份、1392B logical/8192B allocated已核无reader后清理，成功核心日志统一为 `cpu.log/exit`，prepared与全部正式证据保留。

启动时GPU2正常推进，tmux `VAE_Mu_Adam500_g2_20261103`，controller `2105990/start41901657`、timeout `2106532/start41903981`、worker `2106533/start41903981`。12:54:45真实Trainer确认B4/500/5,336,064个trainable参数，随后加载同原BF16教师；首LR0、master改变0，第二步实际应用LR2.0000000000000003e-6，已有5,333,524个FP32 master元素改变。原observer已实际核finite梯度与所有Adam state步数1/2，step2峰allocated35,737,044,992B；其后下一步LR4.000000000000001e-6。详 [启动核证](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/startup_verification.json)。以上为12:54启动节点证据；该节点不代表完成或质量收益，最新阶段如下。

11月3日18:29:04，[正式报告](../../artifacts/cat_down_vae_muon_closure_cmcc_20261103/formal/report.json)已自然完成为`completed_VAE_Muon_original_Adam500_and_fresh`／exit0，实际500/500及原500日程完整；外层down500于18:29:06 exit0。原worker2106533／timeout2106532及独立fresh2338879均已退出，controller2105990继续固定后继。来源严格保持VAE_mu raw `c5ccf174-0c56-4f90-9e29-7d4ba63d2e23`，raw metadata SHA及全部上游fit／fresh／raw八项／readout和C0控制绑定SHA均复核匹配，未重训VAE或导入其它臂状态。

正式`original_stage_exact=true`，原stage和contract与prepared逐值相同，`effective_optimizer`与原CPU构造证据完整相同、初始optimizer state为空；真实AdamW仍为(.9,.999)/eps1e-8，实际decoder group baseLR1e-5、WD0，scheduler初始LR0。1728叶／5,336,064 FP32 decoder参数，LoRA0；第1步master不动，第2步改变5,333,524个元素，末500步改变5,336,004个元素。冻结持久字段exact；原250/500保存审计均记录FP32 masters exact、optimizer对象／scheduler／RNG保存前后不变、1728状态叶。现存500 manifest核actual_global_step／actual_schedule_max_steps=500，round_base仍是该VAE_mu raw，所有保存文件大小与报告一致；250按原save_total_limit轮换，原审计仍保留。复用已生成合同证据，本次未新增保存／resume或GPU审计。

最终UUID `983dc190-d6ac-4a51-8225-83389b35011c`，[独立fresh](../../artifacts/cat_down_vae_muon_closure_cmcc_20261103/formal/strict_reload.json)exit0，3675持久字段、同输入BF16 logits exact，group8 warm216/216、无失败，digest `eacd8fa8cd1fded87d2c6d7c7fc3835798c024755b44570aaf98569fc45ddcfc`。最终metadata SHA与当前评测绑定相同，训练／fresh／评测UUID一致。252投影＝216 native＋36 compact、无pending dense；实测logical2.4987414816151494／packed2.5121161419412363bpw，含码、decoder、保护值／indices／scales、旋转、route frame及packed的92,897,280 padding bits。全模型tensor4,671,007,680B，含linear之外embedding/head和norm；序列化权重4,672,277,077B，完整目录4,674,084,141B（已核实际文件大小），不把目录费用混称投影bpw。

Trainer实际19,990.4647秒，formal report含加载／恢复／保存／fresh20,069.3959秒（5.5748小时），外层down50020,094.4545秒（5.5818小时），不含已复用VAE前驱。train_loss0.2869919540只记录训练结果；完整formal峰allocated57,263,232,000B／reserved57,889,783,808B，不用早期step2峰值代替整段峰值。

同GPU2完整九项于18:29:06自动接续：timeout `2339350/start43913458`→评测worker `2339351/start43913458`，统一绝对deadline按原8小时保持。[evaluation.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/evaluation/evaluation.json)当时仍running；MMLU timeout `2339969/start43917900`→worker `2339970/start43917901`，18:33:59实见loglikelihood53101/56168请求，确认真实主循环而非只初始化。有限核证时完整八项尚未产出，故本次不作C0完整八项或九项取舍，也不等待它们完成；八项/LB仍按既定链继续。本次只改所属说明，保留源／正式权重／退出／绑定证据，未重启、追加GPU验证或清理产物，交付后停止主动监控。

## 完整结果与核证（11月4日）

11月04日00:22:19固定序列自然结束，权威run为`completed_VAE_Muon_original_Adam500_full9`／exit0。11月4日CPU实核controller 2105990、正式训练／fresh和full9相关timeout／worker均已退出；正式报告、外层各stage和LB outer均exit0，未只凭PID消失判成功。`nine_task.exit=1`与`not_met`严格对应质量未达，非技术失败；完整九项已完成，不再等待。

最终同UUID `983dc190-d6ac-4a51-8225-83389b35011c` 的[完整八项](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/evaluation/full8/full8_summary.json)、[LongBench21](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/evaluation/longbench/merged/scores.json)、[九项验收](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/evaluation/longbench/merged/nine_task_acceptance.json)与原C0 `cbb15c9f-1c1f-4394-817c-65c226d76ce7`完成配对：双方full8各64叶／24742条原逐题文件SHA、行数全部匹配各自记录，doc_id/doc_hash/prompt_hash/target_hash逐条同身份，M57科14042及其它7叶10700完整；task版本、metric keys、B64/B32、0-shot／limit=None保持。LB双方21项／4750条预测SHA与行数逐项匹配各自记录，逐题_id／row_index／answers／input_ids_sha256、生成参数和截断后输入身份均一致，官方代码／数据revision／32768窗口／seed42／无采样协议及archive SHA相同。只复算已有八项、LB21 task macro和九项均值、累加M正确数，未重新评分或运行模型。

| 完整任务（分） | 原C0 AdamW500 | Mu_VAE/A_recovery | 本臂−C0（pp） |
| --- | ---: | ---: | ---: |
| mmlu | 62.3985187295 | 62.4768551488 | +0.0783364193 |
| arc_challenge | 53.9249146758 | 53.7542662116 | -0.1706484642 |
| arc_easy | 78.5774410774 | 79.2508417508 | +0.6734006734 |
| boolq | 82.3241590214 | 83.1192660550 | +0.7951070336 |
| openbookqa | 37.4000000000 | 39.0000000000 | +1.6000000000 |
| piqa | 75.1904243743 | 75.6800870511 | +0.4896626768 |
| rte | 74.0072202166 | 77.6173285199 | +3.6101083032 |
| winogrande | 66.6929755328 | 65.9037095501 | -0.7892659826 |
| longbench_v1 | 41.7418399240 | 41.9450618581 | +0.2032219341 |
| 八项均分 | 66.3144567035 | 67.1002942859 | +0.7858375825 |
| 九项均分 | 63.5841659502 | 64.3052684606 | +0.7211025104 |

MMLU为8762→8773/14042（+11题）。九项精确记录64.30526846061926，对C0 +0.721102510416422pp；相对BF16 68.04714491790848仍掉3.741876457289pp，未达原≤2.6bpw／九项掉分严格<0.9pp目标。LB为21任务未舍入宏平均，九项是复合分数，不作统一accuracy解释。

正式成本20069.3959秒，完整九项控制器21142.7731秒（5.8730小时）；外层固定序列41287.7869秒（11.4688小时），原命令／PID/start／日志见[运行原件](../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_closure_cmcc_20261103/run.json)。此成本不重复计入已复用上游或短验。最终metadata SHA `a30dc1b9a3e6af40c009e469a9bf9c92a5b82141ecc5107b7d996f81c8a2364f`；实际acceptance SHA `e3b0d3b1cfbbb5f2145309a17f6b192de6938212e12987f5300475c29cb1519a`与sequence质量摘要匹配，配置／metadata／正式report／strict小文件绑定SHA及权重size／mtime_ns／inode均与评测原件一致。本次未重复计算4.67GB权重SHA，原完整SHA与正式fresh证据沿用，未声称本次重加载验证。

实际存储沿前文最终账本：logical2.4987414816151494／packed2.5121161419412363bpw，码／decoder／保护值、indices、scales／rotation／route frame均计入，packed另含92897280 padding bits；full-model tensor4671007680B（含embedding/head/norm），序列化4672277077B，当前目录实核4674084141B。预算通过、质量未达，两者分别记录。完整组件bits及文件bytes保留于九项验收`bit_budget`，不将目录费用当投影位宽。

本固定点支持VAE-Muon→原Adam500的单seed配方正差，仍只是候选；不得与raw或类别Muon／二阶段Mu32增益相加，不自动扩七类或Mu+Mu，本支无新增预算。最终入选仍须以原BF16为根完成可复现整链。经验由主线程统一归并至[压缩选择](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)，本记录不新增算法机制解释。此次仅CPU核证和原位文档更新，原source/pins、GPU1活动链、权重和全部原始证据保持；未清权重、未新启动训练／评测。

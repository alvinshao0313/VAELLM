# G09 Muon：Hotpot 唯一覆盖与重复对照

2026-10-30，标签：G09／Muon／OPD／train-only／覆盖。运行状态：原机32步、cmcc独导及完整九项均已完成；首次full8汇总导入失败后，隔离接续复用完整八任务并完成LongBench21／4750，固定总链H192分支技术exit0。质量验收`not_met`、rank0及logical预算通过；与原Muon的完整配对见文末，方法取舍与经验判断由主线程维护。唯一当前执行计划由[主计划](../../docs/experience/records/w25_transfer/current_plan.md)维护。

本组检验有限呈现次数下，把 Hotpot96 两遍改为 Hotpot192 各一遍的完整配方效果。它不预设覆盖不足是原因，也不预设增加唯一输入有收益。完整九项与原控制配对后才决定采用或结束该配置；不凭训练 loss、MMLU 或部分 LongBench 提前判定。

## 指定控制与唯一改动

共同父是 G09 原生完整成品，UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，不是先从原 Muon 成品续训。控制复用已经完成的[同父 fresh Muon32](../g09_recovery_20261029/README.md#10月30日完整结果与同父配对闭合)，九项原值 **66.06887949148307**；本组只新训覆盖臂，不重跑控制。

固定 M1024 各一次、Qasper32 各两次、32 次更新、每步 M32＋QA8、M／QA 各 `.5`、R0、全部1512个 decoder 叶／4,762,368参数、seed137、原500步线性日程的前32步。M 沿原 `.5 F + .5 gold` 四选项目标且只混合一次；QA 沿当前学生 native greedy 轨迹及原 full-F 全词表在线 forward KL。不得使用既有 QA teacher-forcing targets。

0-based groups0..19 完全不变。仅 groups20..31 的 Hotpot8 槽按既有 hash 池 `H[96:192]` 替换，M 输入／顺序／权重和 Qasper 原值全部保持。成熟 fit 仍使用 `qa_group_index=index%16` 作为原域槽号；真实呈现32个 QA batch，其中28个不同 batch、224个唯一 QA 文档，不能把 modulo16 字段称为实际唯一组数。

Hotpot prompt 呈现总量 **277934**，控制 **278818**，差 **−884 tokens**；Qasper **408348**、M **364534**，二者保持。固定文档呈现和更新数不等于等 token 或等算力。`formal32/report.json` 的 `training.actual_OPD_supervision` 汇总各域真实生成监督 token；原逐步 `rollouts`／`QA_documents` 和 `formal32.log` 保留逐文档生成与监督量，生成轨迹由学生反馈决定，不保证与控制相同。

## 来源与实现边界

直接复用已有池 `/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_hot_qa512_s128_20261013/dataset.pt`，SHA256 `b954e547def4756033cd43e0286375e04d3112584d892ff6074740a588a9b9ba`；该池 H384 的前96篇和控制 H96 逐值一致。本组只取前192篇，不改已有池和 teacher cache。

HotpotQA release v1.1，HF revision `1908d6afbbead072334abe2965f91bd2709910ab`、`distractor/train`、90447行，[官方许可 CC BY-SA 4.0](https://hotpotqa.github.io/)。保持 `sha256('causal_output_fit/137/'+case_id)` 后按 case_id 打破并列的完整 train 排序，不细筛文本、跳样、替换或依据测试题选样。原始来源及 shard SHA 见[data manifest](../../data/hotpotqa_train_20261011/manifest.json)，既有选择／tokenization 见[QA512 manifest](../../result/qwen3_8b_mmlu65_20261004/qwen_hot_qa512_s128_20261013/dataset_prepared.json)。M／Qasper 的来源、版本、split 与原输入沿控制 prepared 和其传递绑定，不新建训练来源。

`bindings.py` 只增加显式新 dataset／contract，`run.py` 保留原1512叶生命周期并增加监督量汇总。stage-local Muon 数学、矩阵／bias-norm分组、NS5 BF16、FP32 master／momentum、Nesterov `.95`、NS 系数、`.2sqrt(max(m,n))`、aux Adam `(.9,.95)/eps1e-8/WD0`、joint clip1.5、native rollout、fit、motion、原生导出全部直接调用受保护的成熟 Muon 文件。κ仍由本次首个真实 joint clip 梯度计算且随后固定，不能复制控制／旧 short κ。

新输入长度低于原 QA 峰值，不新增资源或数学路径。复用原真实 Muon 两更新与 native export 验证，不重跑 GPU short，不重建 teacher targets。CPU 必须核验来源／原始行／签名／张量、前20组、末12组映射、曝光、M/Q及 M targets逐值不变，并核实际运行的 fit／optimizer／rollout／loader／export 绑定；通过后直接正式32。显式新 dataset 位于本输出目录，不改旧绑定源码、source 模型、direct Adam prepared 或旧 Muon pins。

参考了[同父恢复净收益与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)及[覆盖／预算比较边界](../../docs/experience/lessons/experiment_design.md)：只改一个域，固定父与目标，用完整九项判定，token 与真实成本另报。历史 QA512_128 相对 HOT128 的九项曾为 `+0.2471066161pp`，旧扩池并非全负；旧初态／预算／多个域变化不能代替本组配对证据。

## 运行与验收

本组独立根目录 `/data/star/VAELLM/g09_hotpot_coverage_20261030`。代码／脚本 SHA、原始来源和控制 bindings 均写 `prepared.json`；模型文件保持原 logical／packed 账本及 rank0，由真实导出审计检查。控制 logical2.5295091711956523、packed2.6391813858695654 bpw，计入 decoder、保护 indices/scales/weights、rotation，packed 另计码padding；本组只能在导出后称为实测相同。

调用者先激活 `bitvae`，不在实验 `.sh` 内激活。CPU 准备使用 tmux，隐藏 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/g09_hotpot_coverage_20261030/prepare.py
```

CPU 通过后在项目根目录，以 `bitvae` 的 tmux 执行一次性固定前驱 pipeline：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/g09_hotpot_coverage_20261030/pipeline.py
```

顺序为 GPU4 的 fresh formal32 → GPU4 可见 CUDA 的原生 export → 再核 GPU5/6/7 无 compute 占用 → 成品完整 full8＋LongBench21／4750。原 Muon 正式32实测约4695秒，作为本组约1.3小时训练的调度参考；不保证本组墙钟相同。正式训练超时上限43200秒，评测沿成熟入口的既有运行上限，不创建 daemon、重试或资源轮询。任一步失败停止依赖并保留原因；GPU5/6/7若届时占用，停止于评测启动前，不干扰既有任务。

核心证据：`prepared.json`、`dataset.pt`、`pipeline_launch.json`／`pipeline.log`／`pipeline.exit`、`formal32.log`／`formal32.exit`、`formal32/report.json`／`run.exit`、`native_audit.json`、`full9_launch.json`／`full9.log`／`full9.exit`。新成品在 `native_final/final_model`；评测在 `evaluation/Muon/`，`Muon`只是沿用成熟评测 profile，整个根目录专属于本覆盖臂。完整九项入口 `evaluate.py --gpus 5,6,7` 可独立执行，`--check-only`仅核真实成品，不等待／重试。

验收保持当前 BF16 **68.04714491790848**、九项等权未舍入绝对下降严格 `<0.9pp`，即九项 `>67.14714491790848`；logical≤2.6 bit、rank0。九项仍是 boolq、rte、winogrande、arc_easy、arc_challenge、openbookqa、piqa、mmlu、LongBench v1完整21任务宏均值，不设单项门槛，不按八项先出分取消LB。

## 10月30日实际验证与启动

`bitvae`／Python3.11.16 的 CPU preparation 实际48.0001秒、退出0、CUDA未初始化。完整90447行 train 身份 hash 排序和两份原始 shard SHA通过；前H96逐值一致、新H[96:192]原始行／完整context／冻结tokenization签名通过；原M/Q及M targets、前20组、后12组单域映射、权重／顺序／曝光及实际 fit／optimizer／rollout／loader／export 调用绑定全部通过。新显式 dataset 20,767,690B，SHA `28f353d82b59cb4cf1a2328c2eeae0e48642667ab7ac5e75d4aa67d30e7d4217`；prepared SHA `dedeac111e8deaf8c0d93b4b8eb46855d56760a09dd6b4f16342477f38ba5c8a`。实际H／Q最大prompt2678／10586，TF输入最大2682／10593，未增资源峰值路径。未重复GPU short。

完整九项真实 CLI 导入／参数解析通过；独立 `evaluate.py --gpus 5,6,7 --check-only`在真实末点尚不存在时，准确拒绝新输出根的缺成品前提、退出1，未创建评测目录。这是预期拒绝检查，不是成品审计或质量通过；日志合并入`prepare.log`。

09:46:50 启动 tmux `G09_H192_GPU4_full9_567_1030`，shell657558、pipeline657563、正式worker657569；pipeline source／argv／GPU UUID与启动占用在`pipeline_launch.json`。GPU4实核0MiB、无compute，正式入口为已绑定的`run_formal.sh`，训练及导出均保持GPU4可见。GPU5当前还分配给另一项有限验证；本组约1.3小时训练后，pipeline只做一次5/6/7资源检查，可用才执行完整九项，届时若占用就保留完整成品停止，不重试或占卡等待。

已有限核实真实主循环推进：elapsed246.3319秒完成第1次更新（该组148.2279秒），M32＋8QA真实反传、1512有限梯度、jointclip前norm122.5289383、504Mu／1008Adam连续state1通过；本次首真实梯度新κ为`5.572598053734668`，矩阵／aux LR分别`5.572598053734668e-6`／`1e-6`，188252个BF16 decoder元素已改变。首组生成75个监督tokens；elapsed272.6828秒第二组已生成65个，1512 master／optimizer逐值往返、504 momentum／1008 Adam状态与固定κ保持、连续step1通过。上述数值是本次实际运行，不声称与旧控制位级同轨。已确认推进并停止主动轮询；训练、导出及完整九项均尚未完成。

文档索引已刷新；全库链接检查当时149项历史／并发缺失，本目录README无缺失相对链接，未扩查或修改其它主题。原CPU、预期缺末点拒绝和链接检查输出归并`prepare.log`；7个无引用的一次性`/tmp`日志／退出文件已核精确路径及归并后删除（逻辑26970B、`/tmp`分配49152B，记录在核心日志），未删除模型／数据／配置。

尚无新增质量经验。新 dataset／endpoint 保留用途为本组正式恢复、独立导出和完整九项；源与控制均受保护。结论闭合后按最小证据与明确后续用途清理中间权重，不自动保留无用途副本。

## 迁移前暂停交接（10月30日）

用户要求完成本次安排后暂停推进、准备迁移。现已停止主动检查和新启动；保留既定后台 `formal32 → GPU4可见CUDA export → 一次检查GPU5/6/7 → 完整九项`，没有修改它的绑定源码。上节第1更新及第2组生成是暂停要求前最后核实的进度，不代表暂停时或迁移时的实时完成状态。后台会继续写产物。

最后已核身份：tmux `G09_H192_GPU4_full9_567_1030`；tmux shell657558、pipeline657563、训练worker657569。pipeline实际启动 `2026-10-30 09:46:50 Asia/Shanghai`，`pipeline_launch.json.started_unix=1793324810.4261627`；最后实核worker仍存活，随后未再查询。未采集proc start_ticks，不能只凭PID识别迁移后的进程。

当前活动写入根目录为 `/data/star/VAELLM/g09_hotpot_coverage_20261030`：`formal32.log`、`pipeline.log` 正由后台持有。后续预定新增写入包括 `formal32/{decoder_endpoint.pt,report.json,run.exit}`（失败则`failure.json`）、`native_final/final_model/`、根目录`native_audit.json`／`execution.json`／`formal32.exit`；资源可用后再写`full9_launch.json`／`full9.log`／`full9.exit`和`evaluation/Muon/`下完整full8／LongBench分片／合并／当前验收结果；正常收尾写`pipeline_terminal.json`／`pipeline.exit`。训练、导出和评测当前均未核完成。

运行期间必须保留原绝对路径和哈希绑定的源码／prepared／显式dataset／源模型／控制证据／原snapshot与数据路径，不对活动目录做最终迁移覆盖或清理。迁移清单需包含这些后续新增文件，或待后台真实终态后再做最终增量同步；当前静态文件清单不能视为完整交付。

若GPU5/6/7一次检查发现占用，pipeline将停止于已完成训练和导出的成品，原因留`pipeline.log`、外层`pipeline.exit`非0，完整九项不会自动启动或重试。用户恢复后先核`formal32.exit`、`native_audit.json`及实际成品身份，再核空卡；只需从本入口`evaluate.py --gpus 5,6,7`衔接尚未启动的完整九项，不重训或重导出。若已有部分评测输出或其它失败，先读真实退出与原因，不能覆盖输出或直接重启。

## cmcc 新机接续（2026-10-30）

旧机实际终态已由迁入原件核实：`formal32/run.exit=0`，32次更新完成，训练4691.3686秒；原生独导失败，所以外层`formal32.exit`和`pipeline.exit=1`。没有完整成品或九项评测，不能把训练完成写成整个pipeline完成。迁入原件位于`/data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_20261030`，保持原字节、旧路径和历史身份记录，不重训。

故障位于fresh source重建：快照的`rotation/weight_preprocess.py`调用了主仓库旧`rotation/hadamard_utils.py`，4032维被旧`n % 96`分支的幂二断言拒绝。完整S快照已从接续归档恢复并逐SHA核验；原已确认算法的4032维因子为252阶。新入口在任何会重排`sys.path`的项目helper之前固定导入完整快照的rotation／codec／loader，并断言各模块的真实路径和4032→252；没有改变rotation spec、保护、位宽、因子或跳过检查，也没有修改共用源码或历史快照。

本轮独立输出根为`/data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030`。`cmcc_export.py --prepare-only`已在`bitvae`下CPU退出0：实际父3423张量、端点1512个BF16叶／4,762,368参数全部有限，端点SHA与历史report一致；原dataset、目标contract、32步轨迹、源UUID和冻结声明通过。真实账本logical **2.5295091711956523**、packed **2.6391813858695654 bpw**，计入decoder、保护indices/scales/weights、rotation，packed另含码padding，rank0；覆盖臂端点替换未改变任何费用字段。另核1911个真实冻结张量的shape／dtype／字节SHA聚合，得到历史digest `681969eb1733259ba585322d1449f43e18fb489e22797645c1f4e1499d900461`精确一致。证据归并为新根`export_prepared.json`／`export.log`和`frozen_content_audit.json`。

新`export_prepared.json`按当前绝对路径、内容SHA和实际运行源码重新绑定；旧path／inode／mtime只保留作历史证据，不伪装成新机身份。独导仍用canonical v6原生装配／保存、完整3423保存比较和fresh strict、1911冻结、504码库、完整spec与真实账本；原HF路径显式指向`/data1/cmcc/models/Qwen/Qwen3-8B`。

为去掉独导前重复读取BF16原权重，隔离`cmcc_native_loader.py`复用已验证的`no_init_weights`／`AutoModelForCausalLM.from_config`构造：只读当前HF config／generation_config，以原BF16／FA2创建结构；随后直接交给S快照的`load_v6_full_checkpoint_into_model`完成metadata验证、canonical拓扑重建、`restore_parameter_dtypes`、strict全字段加载、runtime刷新。只支持本组Qwen3-8B完整252／rank0拓扑，不回退到其它算法。所有持久参数由实际v6覆盖；RoPE非持久buffer由原config构造，核为CPU FP32有限值，未整模转换dtype。新runtime及HF metadata独立绑定于`export_prepared.json`。

13:58:36在实核GPU4无compute后启动tmux `H192_cmcc_export_1030`（shell81450、start_ticks7730434）。实际约52.8秒自然退出0；新成品UUID **`74329fcf-c762-4498-9f41-218df9645b3f`**，路径`native_final/final_model`。`native_audit.json`确认保存3423／fresh strict3423全部exact、1911冻结和504码库、完整spec／拓扑／新UUID、logical2.5295091711956523／packed2.6391813858695654与rank0全部通过。装配／保存／比较在CPU执行，GPU4可见以满足原FA2构造前提，未做GPU前向。此时原HF五个shard尚不存在，实际完成证明完整checkpoint独导不需要先读原BF16；不代表在线teacher或正式来源校验也可省略。GPU4已释放；`export_launch.json`／`export.log`／`export.exit`保留实际命令、身份和退出证据。

本轮CPU准备与独导已完成，不重复创建prepared或覆盖成品。调用者先激活`bitvae`；独导实际命令为：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/cmcc/VAELLM python -B -u experiments/g09_hotpot_coverage_20261030/cmcc_export.py
```

完整九项使用共享新机BF16基线绑定和已验证新LongBench manifest，沿成熟完整full8→三卡LongBench21／4750→完整汇总流程执行。共享基线`/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b/migration_binding.json`已验证，SHA `c85f2f59177efb01a29adaf40bfc3143ab4bbb753774aab2e4266b8f9f7e9f71`。新旧inputs.u32／index的SHA精确相同，21份4750条预测逐行核输入／答案／顺序；基线副本只重绑实际路径／stat／对应身份SHA，原指标、任务配置和非身份逐题字段不变，原文件不回写。真实HF全5片原LFS SHA亦已通过。验收仍为本节之前已确认的BF16九项68.04714491790848、绝对下降严格`<0.9pp`、logical≤2.6和rank0。

完整CLI `cmcc_evaluate.py --gpus 5,6,7 --check-only`实际退出0；一次实核5／6／7均无compute后，14:09:57启动tmux `H192_cmcc_full9_567_1030`，shell84287／start_ticks7798511，CPU控制器84297／start_ticks7798547。入口直接使用上述既有成品，没有重训或重导出。其后MMLU与seven均实际退出0，但CPU full8汇总时`lm_eval`导入HF `evaluate`被脚本目录同名`evaluate.py`遮蔽，触发循环导入。`full9.exit=1`、`evaluation/Muon/evaluation.json.status=failed`，未执行LongBench；原PID均已退出。这是技术失败，不是质量`not_met`。原报告、日志和全部full8逐题文件保持原样，原绑定版本的`cmcc_evaluate.py`已按SHA `095caa1260e8e854509c8f57ca63ca8acf06ca8ee96f4fb9688fbe3cb02e01c7`保存在`failed_full9_source_20261030/`。

`run_cmcc.sh`保留最初独导→完整九项的调用示例，`bash -n`已通过。本轮独导和full8已经完成，当前有效接续命令以下节新模块为准。启动前核计划中的5／6／7空闲；评测内部仍做实际资源检查，技术失败停止本路且无自动重试。

### 完整八任务复用与固定总链接续

当前有效接续为`cmcc_continue.py`，从项目根目录以`-m`执行，包内使用完整包路径导入，并移除直接脚本调用产生的本实验目录搜索项，避免遮蔽已安装的HF `evaluate`；旧`evaluate.py`及原CLI参数不删除、不改名。新根为`full9_continuation_20261030/`，原失败目录及raw结果不覆盖。正式启动由固定总链唯一负责，本入口不自行排队、等待资源或重试。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/cmcc/VAELLM /home/cmcc/miniconda3/envs/bitvae/bin/python -B -u -m experiments.g09_hotpot_coverage_20261030.cmcc_continue --output /data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030/full9_continuation_20261030 --gpus 5,6,7
```

新根的CPU准备实际退出0：真实HF `evaluate.__file__`指向`bitvae/site-packages/evaluate/__init__.py`；原`recovery_eval.full8_summary`对实际两份完整结果及原raw样本执行版本／配置／模型身份／哈希检查，只将新摘要重定向到新根。64叶／24742样本全部通过，八任务均分 **68.71920867020135**；该值只是完整八项结果，不能替代九项。`prepared.json`含266条实际输入／源码SHA，SHA为`15a1c21a44ecc0d60eb3691a90000b0cb5bde0634cfa96f58cd5274342df5889`；`validated_config.json`和`validated_plan.json`检查原完整21／4750／三卡分组及原生模型协议。没有重新推理full8或变更旧报告。

正式接续复用这两份raw结果和摘要，只执行三卡完整LongBench→严格当前九项汇总→与固定Muon32控制的机械比较。隔离`cmcc_strict_summary.py`沿原`transfer_nine_task_acceptance.summarize`计算质量，再用真实`audit_checkpoint(max_bpw=2.6,max_lora_rank=0)`及无LoRA拓扑／存储检查声明rank0；不能仅把历史默认rank16字段改成0。真实旧Muon完整报告及其native权重已验证：score66.06887949148307不变，rank0实际账本通过；独立验证进程按`not_met`实际退出1，64／24742及九项字段保留。证据在新根`control_rank0_validation.json`、`control_rank0_process.log`／`.exit`。

总链接口：CLI `--output`指上述接续根；运行报告在其`evaluation/Muon/evaluation.json`，当前结果在`evaluation/Muon/current_goal_result.json`，严格九项报告在`evaluation/Muon/longbench/merged/nine_task_acceptance.json`，机械比较在根`muon_pair.json`。来源UUID仍为`74329fcf-c762-4498-9f41-218df9645b3f`。完整结果`not_met`属于成功完成的评测，汇总CLI的1由成熟merger接受；仅`invalid`／缺项／技术异常停止依赖。

**真实终态已核（2026-10-30）：** 固定sequence中的H192 worker `96551/start_ticks8005898`已自然退出且PID不存在；北京时间14:44:31.234133启动，16:27:24.453471结束，driver用时6173.219337940216秒，`h192_continuation.exit=0`，sequence分支`status=complete`。原两份full8均exit0，复用摘要及严格验收均记录64叶／24742样本；新增LongBench为完整21任务／4750样本，三份outer和bind均exit0，`merge_nine.exit=0`。`evaluation.status=complete_full8_full21_nine`，新评测段用时6148.208405256271秒；`nine_task.exit=1`只对应质量`not_met`，没有技术失败或缺项。本次核证复用既有严格汇总和原始完整性／SHA证据，不重新评测或重算全部逐题结果。

同一H192成品UUID `74329fcf-c762-4498-9f41-218df9645b3f`的九项原值为 **66.02252219344504**，同协议BF16为68.04714491790848，严格十进制／有理数验收的平均下降为2.0246227244634354个百分点，未满足严格小于0.9。原Muon控制UUID `1257125f-7c64-4650-80b0-04ec2deac99f`为66.06887949148307；机械配对差为 **−0.04635729803803201** 个百分点。各项如下，均为原报告未舍入数值，差为H192减原Muon：

| 项目 | H192分数 | 相对原Muon差（百分点） |
| --- | --- | --- |
| boolq | 84.58715596330275 | −0.3975535168195705 |
| rte | 79.78339350180505 | 0.0 |
| winogrande | 68.11365430149961 | +0.15785319652722762 |
| arc_easy | 81.56565656565657 | −0.042087542087543284 |
| arc_challenge | 54.60750853242321 | −0.5972696245733786 |
| openbookqa | 41.4 | 0.0 |
| piqa | 76.71381936887923 | +0.1088139281828262 |
| mmlu | 62.98248112804444 | −0.007121492664850848 |
| longbench_v1完整21项宏均值 | 44.44903037939455 | +0.3601493690930653 |

真实logical Linear为2.5295091711956523 bpw，packed为2.6391813858695654 bpw，与指定控制一致；logical计入全部码、decoder、保护indices／scales／weights及rotation，packed另含码padding。全模型tensor为4,781,328,384 B，实际单权重文件4,782,744,841 B，不能把packed或整文件口径改称logical。原native审计`verified`：3423保存／fresh strict、1911冻结、504码库及几何不变；新严格摘要用真实metadata及原生账本核rank0、≤2.6，`budget_passed=true`。本次复核pair绑定的H192报告SHA `8b574dc93d7d664e59e1f268892219845cc1d34db3970de4f7a0a376d59b5172`和控制SHA `3371eb8f3049bb23795ebe2bd1e6859fc63b7c0c8ef2f3a691a1e4077109acad`均匹配，audit／metadata绑定及实际权重stat也匹配。证据仍在上述接续根和[固定sequence状态](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/status.json)，旧技术失败原件保留。

有限资源核查时GPU5／6／7各1MiB、无compute进程；固定plan的余下dynamic／Rice训练仅用GPU0／1，评测仅用0／2／3及1／4，没有5／6／7后继。H192已结束，这三卡可供后续独立任务使用。原成品与源若仍被活动sequence终态校验绑定则继续保护，不能因为本分支结束而删除它们。

本固定32更新配方按预先规则保留H96，结束H192的追加预算；LongBench的正差不足以抵消其余任务代价。该单次完整配对不证明一般覆盖机制、统计稳健性或其它父源的优劣，也不把H96称为跨模型最优。

按两份原始`training.trajectory[].QA_documents`实际累计：H96／H192的Qasper均64次呈现，监督token分别1005／979；Hotpot均192次呈现，监督token1023／1035。QA总监督token为2028／2014，完整回放输入token为688938／688040；训练耗时4694.879676216995／4691.289199571998秒。这是等文档呈现和等更新预算，不能称等token；两个独立fresh运行的在线生成轨迹也并非逐值相同，细小均分差不作纯覆盖因果结论。后续新父按同一候选／曝光规则重新验证，重算κ和学生轨迹。方法边界已归入[同父恢复经验](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。

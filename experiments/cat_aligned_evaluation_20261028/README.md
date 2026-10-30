# C/G 对齐链原生3423完整九项评测

本目录只连接固定模型与成熟评测，不训练、不选参数或来源。普通原生节点为每个 profile 的01、03a、03b、04、06；09已有完整结果，不重评；02、05只保留其已有原生核证。07的Rice稀疏残差不适用这里的加载器，必须沿既有 `sparse_eval`／`sparse_runtime` 入口。

01的源只能来自 `/data/star/VAELLM/cat_aligned_recovery_20261028/{C,G}/01_qk_fineweb2000`：读实际 `training.json` 的 `training_result.saved_model_dir`，且要求真实2000更新、五个原生保存边界、成功退出、新UUID，以及 [audit01.py](../cat_aligned_opd_20261028/audit01.py) 生成的 `aligned_QK2000_terminal_native_v1` 审计。核对2991冻结字段、504码库、FP32末点转BF16精确读回、3423字段fresh strict、原生实际位宽、证据SHA与权重stat。源未就绪立即拒绝，不等待或借用旧09。

03a／03b／04／06要求显式实际 `--checkpoint` 和 `--native-audit`，只接受同profile新对齐目录与各自正式末点schema；03a32、03b64、C1完整投影拼接、B32的18可变／3405冻结字段均按现有核证字段检查。各节点独立运行，不自动启动后继，不因分数跳过LongBench。

固定协议：0-shot、无样本limit、fast tokenizer、BF16、group8；MMLU B64、其余七项B32，完整64叶／24742题。LongBench采用原官方21项／4750题，共享原输入archive，32K总窗口、各任务官方输出cap、非思考greedy、seed42。九项为 `(8×八任务均分 + LB21任务宏均)/9`，Qwen BF16为68.04714491790848，目标67.24714491790849；原生logical≤2.6／rank0，packed单列。`not_met` 是正常完整结果，九项子命令exit1由成熟合并器接受，总链正常退出，不触发重试。

实现复用：

- [recovery_eval.py](../night_sequence_20261026/recovery_eval.py) 的 `evaluate(full9=False)` 执行并核完整八项；本入口补核64叶原始样本、保留SHA。
- [原生LongBench加载器](../cat_recompression/longbench.py) 的 `load_model` 按文件装入，避免snapshot的 `experiments` 包遮蔽；`predict.py` 只替换原runner的此函数，不改生成。
- 既有gain-calibration `evaluation_adapter/full9.py` 的固定1–8卡分区策略；本轮每profile两卡，每分区86400秒。先在第一张卡完成八项，再同时启动两个LongBench分区。
- 既有 `qwen_mlp_night_eval/evaluate.py::start_group` 与 `llama_fullqk_qkonly2000_longbench21/{bind_terminal,merge_full21}.py`，核真实进程退出、SHA、UUID、生成参数与canonical完整覆盖后合并。
- 既有 `qwen_b32_mlp_s3_20261023/summarize_nine.py::summarize` 通用函数；新薄CLI不调用其硬绑旧S3源的 `main`。

调用者先激活 `bitvae`，在项目根目录执行。01真实审计产生、所列GPU空闲后，固定入口为：

```bash
bash experiments/cat_aligned_evaluation_20261028/run_C01.sh
bash experiments/cat_aligned_evaluation_20261028/run_G01.sh
```

实际长任务由调用者使用既有tmux/nohup后台执行，保留激活后的PATH、stdout/stderr及退出状态。也可显式启动同一命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/cat_aligned_evaluation_20261028/evaluate.py --profile C --stage 01 --gpus 2,3
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/cat_aligned_evaluation_20261028/evaluate.py --profile G --stage 01 --gpus 4,5
```

`--check-only` 只核源和真实账本，不检查GPU、不运行评测、不创建节点输出。后续节点保持同一CLI，例如 `--profile C --stage 03a --gpus 2,3 --checkpoint /data/.../C/03_protection/scale32/native_final/final_model --native-audit /data/.../C/03_protection/scale32/native_audit.json`，示例省略路径须替换为实际路径。

结果固定为 `result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/{C,G}/{01,03a,03b,04,06}`，拒绝覆盖。`evaluation.json` 记录源UUID、审计、账本、权重stat、实现SHA、进程身份、GPU与状态；完整八项在 `full8/`，两分区和归并计划在 `longbench/`，正式九项报告在 `longbench/merged/nine_task_acceptance.json`。LongBench只等待自己本次子进程退出，失败立即停止，无重试、续跑、占卡或新调度器。

CPU验证入口为 `validate_cpu.py`，只读原C/G09已有配置与pins验证加载/计划材料，明确不代表新01，也不重新推理或计分。验证覆盖实际import与CLI、1–8卡canonical唯一分区、非法GPU参数拒绝、新01无审计拒绝和原生2.6加载器／通用九项函数接线；证据输出到同结果根的 `cpu_validation.json`。成熟模型GPU与保存/导出路径已有适用证据，本次不重复运行。新01实际GPU评测尚待其真实审计及调用者启动；当前实现准备不等于质量验收完成。无新增算法经验。

2026-10-28 CPU实测exit0，22.0432秒，CUDA未初始化；另核原C/G09实际native2.6存储账本与64叶／24742题原始样本SHA，全部通过。当前证据为 [cpu_validation.json](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/cpu_validation.json) 与 [核心日志](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/cpu_validation.log)，旧首轮验证和退出散文件已归并删除。所有实际运行Python源码和两个shell入口均留SHA，shell语法检查通过；两份01审计当时均尚未产生，拒绝路径得到实际验证。

## 10月28日21:26：C01／G01完整九项已实际启动

两份01现已真实完成2000更新／exit0，`aligned_QK2000_terminal_native_v1`审计均verified：3423字段fresh strict、2991冻结字段和504主码bank契约通过。C UUID为`9a392147-20b2-418d-9cf6-25d2ffff3a5b`，G为`44ed6d15-abf0-46b0-a627-998a6406087a`；原`checkpoint-2000`、原生末态及全部审计输入保留。本次复用实际审计，没有重复GPU短测或重算大权重hash。

启动前实查GPU2／3／4／5均0MiB、0%且无compute进程；在已激活`bitvae`的项目根目录，由两个独立tmux各执行一次上述固定shell入口。CPU final验证SHA仍为`53b76a5793b51dabfedeb399e9fbc1a03bb309546485f88c95dd434db7178607`，已实际核对其中Python和shell源码pins。共享[launch_01_20261028.json](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/launch_01_20261028.json)保存完整tmux命令、激活与cwd、GPU UUID、shell／driver／native PID及start_ticks、日志和外层退出文件位置，没有覆盖已有节点或重启任何任务。

| 节点 | 固定GPU | tmux会话 | driver PID／start_ticks | MMLU native PID／start_ticks | 输出 |
| --- | --- | --- | --- | --- | --- |
| C01 | 2／3 | `cat_aligned_C01_full9_GPU23_20261028` | 320247／3396765 | 320838／3397920 | [C/01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/C/01) |
| G01 | 4／5 | `cat_aligned_G01_full9_GPU45_20261028` | 320319／3396799 | 320840／3397941 | [G/01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/G/01) |

有限启动核查确认两路原生加载和group8预热均为252／252、skipped／failed0（C2.23秒、G2.40秒）；MMLU完整loglikelihood请求均已连续推进至13061／56168。这证明八项中的实际推理已开始，尚无完整八项或九项质量结论。核心外层日志为[C01.log](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/C01.log)／[G01.log](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/G01.log)，实际MMLU日志在各节点`full8/mmlu.log`；外层`C01.exit`／`G01.exit`将在各固定链自然结束时落盘，启动确认时尚不存在，不补造完成状态。

后续按既定顺序完成第一张GPU的全八项，再以同profile两张GPU执行固定2100＋2650条LongBench分区并正常归并完整21／4750和九项，不按八项分数跳过LB。GPU0／1上的新OPD短测和GPU6／7上的Llama正式任务未触碰。已在上述实际计算确认后停止主动监控，不等低频LB或总完成，不追加占卡、任务、重试或配置修改。原生末态与01审计依赖均保持。

## 10月28日：01完整八项初报，九项尚待完整LongBench

以下替代21:26启动时尚无完整八项的状态。两路原完整八项已自然落盘，MMLU／seven实际exit0；CPU一次核对同01 UUID、两份原始结果SHA、64叶24742题和八项等权算术通过，没有重复模型推理或逐题评分。

| profile／原full8汇总 | 八项均分 | MMLU | 同源09八项 | 01−09八项差pp |
| --- | ---: | ---: | ---: | ---: |
| [C01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/C/01/full8/full8_summary.json) | 67.69115168947293 | 61.97835066229882 | 67.70762805758119 | −0.01647636810825759 |
| [G01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/G/01/full8/full8_summary.json) | 68.81736712449862 | 63.50947158524427 | 68.8677376676076 | −0.0503705431089827 |

C UUID仍为`9a392147-20b2-418d-9cf6-25d2ffff3a5b`、G为`44ed6d15-abf0-46b0-a627-998a6406087a`；09对照直接读取各自[C09](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/C/09_joint1200/evaluation_full9/full8_summary.json)／[G09](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/G/09_joint1200/evaluation_full9/full8_summary.json)原件。协议仍为0-shot／无limit、MMLU B64／14042题加其余B32／10700题，task metric选择与57科加权方式保持。

这是完整八项初报；固定两卡LongBench21／4750仍由已启动后台链执行，本次不检查新GPU进度或部分预测。尚无本次核实的完整九项结果，不能用八项的小幅下降提前判定QK恢复九项收益、跳过LB或修改后继来源。01训练核证和02短验／正式启动分别归[恢复主入口](../cat_aligned_recovery_20261028/README.md#10月28日01完成核证02正式恢复已启动)与[OPD正文](../cat_aligned_opd_20261028/README.md#10月28日short2完成与fresh正式32启动)。无新增算法经验。

## 10月29日：C01／G01完整九项终态

本节替代上面的在途状态。两路于10月29日00:20:39／00:19:18（Asia/Shanghai）自然完整结束；`C01.exit`／`G01.exit`均0，`evaluation.json.status=complete_full8_full21_nine`，每路两个LongBench分区、bind和merge均exit0。原九项子命令exit1表示质量`not_met`，不表示程序失败。原报告仍沿运行时≤0.8pp阈值；[10月29日当前目标](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月29日当前目标与完成条件)改为严格<0.9pp，以下两成品同样未达，不改历史判定。

| 成品／原九项报告 | 八项均分 | LongBench21 | 九项综合分 | 比同源09九项变化pp | 比BF16下降pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| [C01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/C/01/longbench/merged/nine_task_acceptance.json) | 67.69115168947293 | 40.84491372650079 | 64.7082363602538 | −0.005156345388485306 | 3.338908557654674 |
| [G01](../../result/qwen3_8b_mmlu65_20261004/cat_aligned_evaluation_20261028/G/01/longbench/merged/nine_task_acceptance.json) | 68.81736712449862 | 40.88570390153926 | 65.71384898861425 | −0.07036111151528246 | 2.3332959292942355 |

同源C09／G09九项分别64.71339270564229／65.78421010012953。01相对09，C的MMLU／LongBench变化为＋0.007121492664865059／＋0.08540383636970716pp；G为＋0.4272895598917543／−0.23028565876567342pp。QK专项2000步在这两个固定来源上没有提高完整九项，不能由G的MMLU提高推出综合改善，也不能据此否定其他起点／预算的QK恢复。

本次仅CPU只读核证：两份外层退出、同01 UUID、审计SHA和实际权重stat一致；实际full8原始样本64叶／24742行，两个merged结果均官方21项／4750行，逐预测文件SHA及各任务样本数一致，任务宏均及九项未舍入算术复核通过。C／G logical／packed仍分别2.497367527173913／2.5107421875和2.5295091711956523／2.6391813858695654，含码、decoder、保护和旋转等实际部署字段，rank0；G的packed超2.6仍单列。没有重新推理、训练或评分，也没有新启动／续跑；01模型与审计继续作为已完成02和后续链的受保护父来源。02已完成与独立导出见[OPD终态](../cat_aligned_opd_20261028/README.md#10月29日正式32与独立原生导出完成)。

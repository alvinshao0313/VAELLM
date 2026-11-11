# 已完成末端臂的评分修复与缺失逐类臂接续

2026-11-02唤醒后实核，本包已补齐末端评分和缺失逐类56，全部阶段、run及continuation均exit0，两臂比较完成：逐类MMLU70.79475858%、八项69.53589734%，末端70.73778664%／69.08149851%。[完整终态与结论](#两臂56完成与本配置结论)见下方。原 `pair_20261101_1200` 在末端56及fresh后、首次评分前因缺少`sample_artifacts`导入上下文退出1；本包只修调用者环境，原比较全部Python、1062pins及失败退出／日志保持不动。

已完成末端模型UUID `b85c0689-0d16-40a5-84e4-3b4c6a595c81`，训练／导出4625.804349秒，56/56、fresh3927字段exact。沿用原参数、数据、码／保护／旋转和数值路径，不重训末端，不重复short、教师缓存、CPU模型加载或GPU保存审计。

实际修复为worker `PYTHONPATH=SAMPLE:REPO:N_snapshot`。CPU在隐藏CUDA的真实旧evaluate导入顺序中执行原 `eval_utils_with_samples` 与 `sample_artifacts`，核实际路径／SHA及API callable；这是导入证据，不冒称GPU数值验证。新结果目录内terminal只读view的`report.json`、`fresh.json`和`final_model`链接到原成功产物，评分JSON／样本／日志／退出写新view，原mmlu／seven的exit1和failure证据不覆盖。

有限链只做：已完成末端的首次完整MMLU B64＋seven B32→两片成功后，从原b120 bank新起缺失sequential56→其原fresh→同协议MMLU／seven→汇总一旧一新真实UUID。56更新、七段prefix集合、连续Adam、原500线性前缀、H96静态混合目标、seed与曝光不变，不消费末端或short权重作为逐类父。不增加第三臂、LongBench、完整九项、LoRA、OPD或门控；只用GPU6／7，不碰0–5。任一技术失败停止，无守护或自动重试。

## 当前运行与证据

新根目录 `R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/resume_20261101_1438`，plan SHA `363e357ae3b4f99bea021ca62af5e80ee80886e6c942ad05d652a481691b8fd2`，CPU prepare.exit0。`R/CPU_validation.json`记录真实API/support导入成功、CUDAfalse、9个旧controller／worker代次均退役，旧1062pins未变；新包三份Python另外绑定，不把新增caller环境证明冒称数学或GPU短验重跑。

tmux **`S4_recovery_resume_20261101_1438`**，shell1322300，driver **1322642/start25245615**。首次补MMLU worker1322964、seven worker1322965（均start25246312），分别GPU6／7；14:38:29已越过原失败点，实际HFLM正在建立57学科contexts。准备时两卡各1MiB，无其它compute进程，GPU0–5不在本包调度范围。实际pane已激活bitvae／Python3.11.16并设置：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH='/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/lm_eval_sample_artifacts_20261010:/home/cmcc/VAELLM:/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
python -B -u -m experiments.s4_recovery_resume_20261101.run --plan /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/resume_20261101_1438/plan.json
```

每个GPU worker的实际mask／PYTHONPATH／PID／代次／命令均落`R/run.json`，stdout／stderr为`R/<stage>.log`。terminal新结果及原始样本写`R/terminal_readout/evaluation/`，两个新exit写该view；`R/terminal_result.json`只在两片成功后产生。控制器／shell最终退出为`R/run.exit`／`R/continuation.exit`，原pair的exit1／failure不变。缺失逐类臂未来输出为`R/formal/sequential/`，实际开始与第一更新另核，不能将当前补评写成已训练第二臂。

## 已完成末端的首次完整八项

14:42:02完整MMLU首次保存，两片均exit0，MMLU57叶／14042题、seven7叶／10700题，0-shot、B64／B32、limit=None，任务config／版本匹配原BF16协议。两片绑定同一已完成terminal UUID `b85c0689-0d16-40a5-84e4-3b4c6a595c81` 与56更新；原始样本和指标保留在新view，汇总为`R/terminal_result.json`。本次并行补评从启动到两片reap的墙钟约249.3秒，未重复训练。

| 项目 | terminal56% | 较原无恢复S4 pp |
| --- | ---: | ---: |
| MMLU | 70.73778664 | +0.28485971 |
| ARC Challenge | 54.69283276 | +0.17064846 |
| ARC Easy | 79.58754209 | +0.08417508 |
| BoolQ | 86.23853211 | +0.45871560 |
| OpenBookQA | 38.60000000 | -1.60000000 |
| PIQA | 77.63873776 | +0.76169750 |
| RTE | 78.70036101 | +1.44404332 |
| WinoGrande | 66.45619574 | -0.23677979 |
| 八项等权均分 | **69.08149851** | **+0.17091998** |

MMLU9933/14042，较原S4的9893题净多40题；相对BF16的73.00242131%仍低2.26463467pp。原S4八项为68.91057853%，末端本预算小幅改善均分，但OBQA与WinoGrande退步；不能称充分恢复、全面改善、完整九项收益或两臂胜负。位宽仍4.094031292459239bpw，不是最终≤2.6bit验收。

## 缺失逐类正式已实际推进

两片评分和原失败证据不变检查通过后，14:42:11.976启动原`run --arm sequential --mode formal`，worker **1326290/start25271253**，物理GPU6／7，planned_updates56、validation_only=false，来源仍为原b120 bank，未消费terminal或short权重。首真实更新已核：stage1／step1／group0，LR1e-6，q类Adam有效step1、有限非零梯度，37.542982秒，objective0.084902609587159；GPU6／7峰allocated14,185,991,168／14,052,304,384B。单组loss不与末端初值作质量排名。

上述为11月1日确认首更新时的交付状态；随后固定后台链完成，无自动重试。11月2日唤醒后核实如下真实终态。

## 两臂56完成与本配置结论

`R/run.json`为`complete_original_S4_placement_comparison_tail`，`run.exit=0`、`continuation.exit=0`；[原始两臂结果](../../artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/resume_20261101_1438/result.json)同时绑定两个真实终态。逐类训练／导出4625.066954秒，56更新、各类Adam有效step56／48／40／32／24／16／8；fresh3927持久字段exact。两臂均为完整57科／14042题MMLU及seven7叶／10700题、原0-shot B64／B32、limit=None。1062旧pins及3个接续Python于本次核证全部一致。

| 项目 | 原无恢复S4% | terminal56% | sequential56% |
| --- | ---: | ---: | ---: |
| MMLU | 70.45292693 | 70.73778664 | 70.79475858 |
| ARC Challenge | 54.52218430 | 54.69283276 | 55.88737201 |
| ARC Easy | 79.50336700 | 79.58754209 | 80.30303030 |
| BoolQ | 85.77981651 | 86.23853211 | 86.23853211 |
| OpenBookQA | 40.20000000 | 38.60000000 | 40.20000000 |
| PIQA | 76.87704026 | 77.63873776 | 77.31229597 |
| RTE | 77.25631769 | 78.70036101 | 78.70036101 |
| WinoGrande | 66.69297553 | 66.45619574 | 66.85082873 |
| 八项等权均分 | 68.91057853 | 69.08149851 | **69.53589734** |

逐类UUID **`8a6eb8bb-342f-40cf-b315-86ddc2ada97b`**，实际模型为`R/formal/sequential/final_model`；末端UUID **`b85c0689-0d16-40a5-84e4-3b4c6a595c81`**，实际模型仍在原pair的`formal/terminal/final_model`。本配置下逐类较原S4八项+0.62531880pp、MMLU+48题／0.34183165pp；较末端八项+0.45439883pp、MMLU+8题／0.05697194pp。两臂是同seed、同数据和56更新的恢复配方比较，有限差值不证明一般机制或稳健性。逐类相对原BF16 MMLU仍低2.20766273pp；没有LongBench、完整九项或最终≤2.6bit验收结果，位宽仍4.094031292459239bpw。

两份部署模型只保存BF16模型及CAT激活／布局runtime；现场连续Adam虽在训练期间核证，**没有序列化FP32 masters、Adam moments／step、scheduler或RNG**。`cat_runtime_state.pt`的实际顶层仅`format`、`version`、`runtime_identity`、`activation_runtime`、`global_adaptive_plan`，不能当作训练checkpoint。后续由任一成品开始恢复必须说明BF16暖启动与新优化器，不称exact-resume。原父、逐类和末端模型暂为后继性能恢复核证保留，本次没有清理权重或改动旧失败证据。

经验增量限于本S4／H96／56预算下的配方结果与明确的续训状态边界；不从小幅收益直接归因预算不足，也不据本批否定更长全decoder恢复。后继性能恢复的决定及资源安排由唯一执行计划维护，本包原固定比较已结束。

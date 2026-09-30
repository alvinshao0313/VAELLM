# 全层恢复的两组 decoder 学习率对照（2026-09-24）

运行状态：两组均于2026-09-24完成全部28层恢复和完整八任务评测，退出码0；2026-09-30核实。GPU0组59.3750%、GPU1组59.1009%，同路径初始化基线39.9029%。结果已验证，不扩大实验。

## 问题与依据

以同一个初始 `result/linear_output/distill_init` 做完整 LiftQuant 式逐层恢复，判断是否改善相同下游任务。唯一训练变量是 decoder LR：`1.25e-5` 与 `6.25e-6`。前者有此前384步 joint 对照支持，后者是假设更保守更新可缓和 decoder 过冲的半值候选，尚无更优实测结论。

参考[编码恢复经验](../../../lessons/liftquant_recovery.md)及[工程等价性验证](2026-09-23_engineering_equivalence.md)。此前128-token单block局部收益不能外推到全层或下游，旧2e-4过冲说明不应盲目增大学习率。本次保持敏感度码坐标、所有code与decoder可学、官方实际调度序列、teacher align=1、whole-block MSE以及其余冻结约束。

## 固定配置和比较口径

- 初始 checkpoint ID `c9880629-6b54-4482-8086-857ba185d162`，Qwen3-8B，无旋转、单 residual stage，无 LoRA；192 compressed Linear + 60 dense。
- 训练 blocks：0、9至35，共28块。每块独立优化，teacher prefix 始终为原始FP模型；未训练的dense块仍参与teacher传播。
- 官方完整 RedPajama，固定数据 revision `4b6d76ca56b821e4f2110204943b7a67927b565c`，11原始Arrow共930,514行。数据来源及SHA见数据目录source_manifest。
- `nsamples=4096, holdout=128, seqlen=2048, batch_size=2, epochs=2, seed=42`，每块3968次更新，两组均从原始checkpoint新加载。
- code LR上限均为`2e-5`，实际仍取 `min(cap, proxy.std()/50)`；decoder LR是本次唯一变量。
- 物理 GPU 0/1，各64GiB allocator上限；bitvae。主机实测约1.4TiB available，双任务hidden+target共256GiB，另计模型/保存样例；磁盘约856GiB空闲，运行前再核对。
- payload为压缩目标每32权重64bit，即2bit/weight；不包括decoder等开销，不宣称全模型2bit。
- 评测沿用 `train_utils.eval_utils.run_lm_eval`：boolq/rte/winogrande/mmlu用acc，arc_easy/arc_challenge/openbookqa/piqa用acc_norm；0-shot、batch1、无limit、八任务等权均值。MMLU遵守既有按大小聚合。
- 初始模型与两个成品均用同一packed BF16全驻留路径。旧历史初始基线走whole-decoder fused，不能直接当本次同路径对照；GPU0完成训练后重评初始和成品，GPU1评自身成品，共用GPU0初始基线。

主指标为相对初始模型的八任务均值，辅看逐任务与逐block留出MSE；训练/留出MSE下降本身不等于下游提升。两组不是官方LiftQuant压缩表示的逐项复刻，code/decoder是既有VAELLM适配。

## 入口、验证和运行位置

复现入口 `experiments/liftquant_recovery/run_full_decoder_1p25e5.sh` 与 `run_full_decoder_6p25e6.sh`，须从已激活bitvae的项目根目录运行。薄Python工作流 `run_experiment.py` 串联原恢复入口与相同评測入口，失败即停止，无自动重试。

正式结果根目录 `.result/liftquant_recovery/full_layers_lr_20260924_01/`，两个子目录 `decoder_1p25e5`、`decoder_6p25e6`；每份 `run.json`记录实际命令/状态，`recovery/manifest.json`记录代码与初始payload身份、参数，`recovery/block_metrics.json`为原始逐块指标，`evaluation/`保留原始下游结果。

先执行正式形状短测：同初始模型、完整Arrow，blocks0/9，8样本、2留出、2048token、batch2、2epochs，每块6步。覆盖partial/full块、两轮数据、块间推进、原子boundary两次、完整native导出和严格重载。只缩短样本与步数，不能用128token测试替代。另核对全模型streamed/resident输出和真实评测数据入口。

短测结果根 `.result/liftquant_recovery/formal_shape_20260924_01/`。恢复入口的旧 `final_experiment_run` 字段仅区分API-page数据，因此Arrow短测会标true；本轮实际用途以外层 `run.json:purpose=short_validation` 和本记录为准，不当作正式训练。

当前适用的14decoder数学、7梯度分支、新旧训练轨迹和层边界精确恢复证据直接复用，不扩大回归矩阵。仅待正式形状与新驻留评测路径通过后启动两组；确认真实训练推进后交付并停止主动监控。

## 经验和产物

目前无新增质量经验，候选优劣等待正式结果。完整校准数据用于本次两组和具体复现；初始checkpoint始终保护。正式成品保留用于已安排的下游评测；完成后按评测结论与既有清理规则决定留存。短测权重仅用于链路验证，结束并记录后清理，保留配置、核心日志和原始指标。

## 本轮实测与正式启动

2026-09-24：两组正式形状短测均退出0。每组block0、9各6步、两轮数据，整体与留出MSE均下降，码位翻转0（此步数只验证链路）；两组token SHA完全相同。全部冻结状态SHA不变、hard→native与完整保存重载输出严格相等。训练allocated峰值8.0097GiB、reserved峰值9.7168GiB。八任务limit1全部读取并计算成功，这些少量样本分数不用于候选排名。

全驻留评测验证退出0：同一初始模型、真实2×2048输入，logits误差0、NLL差0、argmax一致率100%、192个BF16缓存未重建；allocated峰值19.5061GiB。单批逐层搬运6.646s、驻留准备加前向2.301s，非全量benchmark。新适用经验已回写上述经验页。

原始证据：短测目录的summary.json、合并后的validation.log、两个recovery/block_metrics.json及evaluation/B.json；驻留目录 `.result/liftquant_recovery/eval_residency_20260924_01/gpu/summary.json`；原数据source_manifest.json记录11份上游hash及4096×2048校准token SHA `6b5fa59d5e9dc29c0c907e3f5da127e34de28329e6055efaca8bb357a4084c02`。

正式启动标识：

| GPU | decoder LR | tmux session | 工作流PID | 成品目录 |
| --- | --- | --- | --- | --- |
| 0 | 1.25e-5 | lq_full_a_20260924 | 2831293 | decoder_1p25e5 |
| 1 | 6.25e-6 | lq_full_b_20260924 | 2831297 | decoder_6p25e6 |

日志分别为正式根目录a.log、b.log；工作流状态在各子目录run.json，最终进程退出码为a.exit/b.exit。实际调用为激活bitvae后 `bash experiments/liftquant_recovery/run_full_decoder_1p25e5.sh` 和 `bash experiments/liftquant_recovery/run_full_decoder_6p25e6.sh`。有限启动检查已确认GPU0至少完成block0第288步、GPU1至少第384步，loss有限；两组正常推进后立即结束启动检查，不等待完整训练。

短测全部已结束，无后续权重用途，删除两份完整短测模型及两个boundary共14,853,037,668字节（约13.83GiB），保留真实配置、核心日志、原始指标和用于驻留回归的token输入。初始模型、完整数据及正式任务不受影响。

用户补充允许在有依据时利用空闲显存增大batch。本次保留已验证的batch2：固定样本与epoch时，batch8会使每块更新从3968降至992，连同码阈值跨越和调度发生变化，现有证据不足以判断更优。两臂继续只比较decoder LR；该决定仅针对本次对照，不形成后续batch限制。

启动核验完成时间（UTC）：2026-09-24T01:37:26.703845+00:00。正式两组4096×2048实际输入SHA与准备阶段一致。一次性数据验证token副本已删除67110213字节，实际正式运行token仍保留；11原始Arrow与初始checkpoint受保护。短测CLI/退出码/清理结果已合并到summary及单份validation.log，已用完的一次性脚本和重复小文件已清理。本轮未提交Git、未改环境，未以短测指标声称下游改善。

## GPU0中断核查（2026-09-24）

用户询问GPU0是否完成后，核实a.exit=1，run.json状态failed、stage=recovery、子进程exit_code=-2，结束时间2026-09-24T05:16:04.348Z（北京时间13:16）。日志为SIGINT/KeyboardInterrupt，停在block17反向传播，最后打印2336/3968步；日志无法确定信号发送者，不将其归因为算法或显存错误。

已完成block0及9–16，共9/28个可训练block；block_metrics.json和最后的约967MiB原子latest_boundary.pt仍在。尚未产生最终导出或启动下游评测。边界仅保存完整层，后续若恢复须重做block17，不能续接未保存的本层第2336步。原始进程已不存在；GPU1的原工作流和恢复进程仍在运行，本次只作有限状态查询。

未自动重启，未修改活动实验依赖。无新增算法经验；中断来源未知且本次目标未完成，保留用于该任务恢复的层边界、配置、原始指标和核心日志，不据此淘汰配置或删除恢复状态。本次无新增产物清理，释放0字节；未新增文档或改变导航路径。

## 用户授权恢复GPU0（2026-09-24）

用户明确要求查因、解决问题后重启。本次只读查因发现：训练源码无主动SIGINT/超时终止路径，原异常是恢复子进程signal -2、外层工作流记录CalledProcessError。可访问的同一时间窗口journal及针对PID/session的shell历史检索均未给出信号发送者；无可用信号审计记录。信号来源未查明，不归因为OOM、数值异常或SSH断线。

重启前另发现独立阻断：`train_utils/config/cli.py` 在原任务启动后增加E2E `stop_after_step` 六行，导致CPU原样检查报training_code不匹配。初始payload、FP teacher、正式参数、校准token及runtime均无差异。未篡改边界或放宽检查；在 `.result/liftquant_recovery/full_layers_lr_20260924_01/restart_source_01` 保存136个必要Python文件（2,699,039字节），只在副本还原六行CLI，101个训练文件SHA与保存身份完全一致。复核413个native状态张量、354个浮点张量有限，9个完成块均3968步；实际parser/import通过。证据 `restart_check_20260924/report.json`、`check.log`；一次性诊断已清理，副本和旧断点保留供当前恢复进程使用。对应[保存/恢复经验](../../../lessons/checkpoint_lifecycle.md)已更新。

新运行使用原全部训练/评测参数，新输出子目录 `decoder_1p25e5_resume_01`，重放原teacher前缀后从block17首步继续；已完成block0、9–16不重训。自动接原八任务初始baseline和成品评测，不覆盖旧结果。复现脚本 `experiments/liftquant_recovery/resume_full_decoder_1p25e5.sh`，caller激活bitvae后cd上述源码副本；脚本参数使用原模型/数据/输出的绝对路径，PYTHONPATH仅指向副本。

本次以 `nohup setsid --fork --wait env --default-signal=INT,QUIT,TERM bash -lc ... </dev/null` 运行。独立session/进程组为3178749，无控制终端，stdin=/dev/null；保留正常INT/QUIT/TERM处理，仅nohup忽略HUP，不安装守护或自动重试。nohup launcher PID3178748，工作流PID3178776，恢复PID3178777。日志正式根目录 `a_resume_01.log`，最终退出码 `a_resume_01.exit`，实际命令/当前状态见新输出run.json。GPU1共享代码与原任务未修改。有限启动核验已通过（UTC 2026-09-24T05:48:28.042274+00:00）：正式入口严格载入9个完成层并进行真实GPU计算，日志确认block0的teacher input与native output均与保存样例逐元素一致。该次有限启动核验时仍在重放FP teacher前缀，随后从block17首步续训，未声称当时已开始该层更新或完成训练。核验后停止主动监控，保持后台任务运行。

本轮文档index/check均退出0，relative_markdown_paths=1080、errors=0；检查结果已并入restart_check的单份report.json/check.log。已清理的一次性诊断文件及重复日志共19,553字节，源码副本、原断点和当前输出均有正在使用的明确用途，予以保留。

第二次中断核查：用户反馈GPU0似乎再次停止，并明确表示没有主动停止该任务。`decoder_1p25e5_resume_01/run.json`记录结束时间2026-09-24T09:18:58.231532Z（北京时间17:18:58）、status=failed、stage=recovery、子进程exit_code=-2。`a_resume_01.log`显示block25最后打印3648/3968步，随后在训练参数有限性检查处收到KeyboardInterrupt，外层记录子进程SIGINT；异常落点本身不代表参数非有限。该进程已使用独立session、无控制终端和stdin=/dev/null，仍发生SIGINT，先前隔离措施未解决未知的信号发送源，不能归因SSH断线。信号发送者仍未知；GPU1原实验继续运行，未修改或停止。

最新可恢复边界为正式根目录下 `decoder_1p25e5_resume_01/recovery/latest_boundary.pt`，大小1,952,972,384字节（约1.82GiB），mtime为2026-09-24T08:59:57.033253Z（北京时间16:59:57）。其中完整保存block0、9–24，共17/28层，每层3968步、805个允许修改的状态张量，以及17层重载输入/输出样例；原始block_metrics的完成层集合一致。bitvae CPU mmap加载及实际`_validate`的结构、manifest身份、可变状态集合和完整训练预算检查通过；`restart_source_01`的101个训练文件SHA仍全部匹配manifest和边界，无须再修源码。本次没有GPU计算、未重新计算初始/教师全量权重哈希，也未执行新一轮GPU恢复，正式入口仍须保留原身份与重放一致性校验。继续时应从这一最新边界重放教师前缀后重做block25全部3968步，不能续接未保存的第3648步；0、9–24无需重训。原边界和最新边界均保留供本任务恢复，配置、核心日志、原始指标及源码副本受保护。本次核查未清理产物，也未因中断改变算法或实验参数。

第二次恢复沿用原算法、配置和`restart_source_01`，从上述17层边界启动，新输出为`decoder_1p25e5_resume_02`，复现入口`experiments/liftquant_recovery/resume_full_decoder_1p25e5_02.sh`。与resume01脚本只差新输出路径和最新边界路径。调用者仍激活bitvae、进入隔离源码副本，PYTHONPATH仅指向它；原训练/评测参数不变、GPU0单卡、64GiB上限，GPU1任务不变。

本次在原nohup/setsid外层调用中加入现有`strace -f -qq --seccomp-bpf -e trace=none -e signal=SIGINT,SIGTERM,SIGHUP -ttt -o .../a_resume_02.signals.log`，只记录信号，不忽略INT、不捕获后继续训练、不自动重试。CPU自启临时进程的真实SIGINT验证通过，记录到已知发送者PID/UID（SI_USER），退出仍为-2；seccomp过滤确实启用。50万次getpid用时无跟踪0.190秒、跟踪0.194秒，仅为本次CPU过滤有效性证据，不当作GPU训练吞吐结论。已并入`restart_check_20260924/report.json`和单份`check.log`；临时信号日志261字节已删除。

实际运行标识：nohup launcher PID3450834，独立session/进程组3450835，信号记录PID3450860，工作流PID3450865，恢复PID3450893。恢复进程stdin=/dev/null、TracerPid=3450860、Seccomp=2/filters=1，CUDA all-bit STE实算检查PASS。核心日志`a_resume_02.log`、信号记录`a_resume_02.signals.log`、最终退出码`a_resume_02.exit`；实际命令/状态在新输出run.json。有限启动核验已通过（UTC 2026-09-24T09:33:48.743027+00:00）：正式入口严格载入17个完成层，原输入/参数/模型/teacher/源码身份校验通过；GPU0已进入真实teacher前缀输入计算（进程RSS约40.4GiB并继续增长，GPU约1.76GiB/4%利用率），GPU数值检查PASS。复用前一轮相同源码的native逐元素回放证据，不重等完整前缀；此时尚未开始block25的新更新。到此交付并停止主动监控。


本轮收尾：原9层boundary的全部413状态、9层输入输出样例及指标已由17层boundary完整同值覆盖；新进程只引用17层boundary，正式严格载入已通过。删除原9层boundary（1,013,668,112字节）、原始/resume01两份不被恢复入口读取的重复calibration_ids（各67,110,084字节）、已归并的两个失败退出码散文件（各2字节），合计1,147,888,284字节（约1.07GiB）。旧resume01脚本已注明被02替代，其已清理边界不再是当前恢复入口。核心日志、失败run.json、原指标、实际配置/源码身份仍保留；17层边界、原始数据、初始模型、当前源码副本和resume02输出受保护。信号探针临时日志另清理261字节，总计本轮清理1,147,888,545字节。文档index/check已通过（1079个相对路径、errors=0）。

## 最终结果（2026-09-24完成，2026-09-30核实）

两组均完成28层、每层3968步、共111104次有效优化更新；冻结状态SHA不变，完整native保存及严格重载通过，28层保存样例max_abs均0。GPU1于北京时间2026-09-24 19:42:12完成，GPU0于23:05:03完成，run.json均completed/exit_code=0，评测summary均PASS。最后恢复运行的信号日志为空，未再记录指定中断；此前两次SIGINT来源仍未知，不能据此声称查明根因。

评测为同一packed BF16驻留路径、0-shot、batch1、无样本截断，八任务等权平均，MMLU沿用原group汇总。初始模型为本次重新评测，未混用历史fused基线。

| 任务 | 初始化 | decoder LR 1.25e-5 | decoder LR 6.25e-6 |
| --- | ---: | ---: | ---: |
| boolq | 49.4495 | 82.1407 | 81.8043 |
| rte | 48.7365 | 71.1191 | 68.9531 |
| winogrande | 52.4073 | 61.7206 | 62.5888 |
| arc_easy | 35.5219 | 65.1936 | 65.3620 |
| arc_challenge | 24.1468 | 41.2116 | 40.5290 |
| openbookqa | 27.8000 | 34.0000 | 34.6000 |
| piqa | 58.2155 | 70.3482 | 70.7291 |
| mmlu | 22.9454 | 49.2665 | 48.2410 |
| 八任务等权均值 | 39.9029 | 59.3750 | 59.1009 |

两组分别较初始化增加19.4722、19.1980个百分点；高decoder LR组高0.2741个百分点。支持本次VAE初始化后的联合code/decoder逐层恢复有明显下游收益；单seed两组不足证明学习率稳定优胜，未达到69%目标，也没有同条件官方LiftQuant基线，不能声称超过官方方法。

原始结果：正式根目录`decoder_1p25e5_resume_02/evaluation/A.json`为初始基线，两组各自`evaluation/B.json`为成品。原run.json、训练/评测summary、freeze_check及逐层指标为验收证据。对应经验已更新；本次只读验收与归档，没有新增训练或评测。

本轮收尾盘点：两组最终模型选为本次A/B交付成品保留，初始模型、原始指标、核心日志及隔离源码保留。完整训练/导出/评测均已成功，无具体续训任务；以下无活动文件持有者的中间边界与可再生校准副本已不需要，按持续授权清理：

- `/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5_resume_01/recovery/latest_boundary.pt`：1952972384字节。
- `/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5_resume_02/recovery/latest_boundary.pt`：3244301968字节。
- `/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_6p25e6/recovery/latest_boundary.pt`：3244290384字节。
- `/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5_resume_02/recovery/calibration_ids.pt`：67110084字节。
- `/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_6p25e6/recovery/calibration_ids.pt`：67110084字节。

实际释放8575784904字节（7.99GiB）；两份交付成品及全部原始评测结果仍保留。历史resume入口依赖的中间边界已结束用途，不再作为当前入口。

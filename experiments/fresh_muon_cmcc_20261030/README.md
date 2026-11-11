# 新鲜 C0 上的 H96／Muon32 入口

本入口执行已经约定的`C0→E`：从实际完成的新C0普通fixed2成品fresh启动，沿H96数据与原500线性日程前32更新，只训练现有decoder，不使用历史B转换父或short权重替代新C0。正式32更新、独立fresh及GPU5完整九项均已完成；九项64.15392820484695，同父净增0.5697622546个百分点，质量仍`not_met`。输出路径和CPU退出传播的两次原失败均保留；11月1日仅重做CPU验收发布及严格配对核证，没有重训／GPU重评或修改原源码pins。

唯一当前安排仍在[执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。参数数学和完整8B两更新证据复用[已完成混合Muon验证](../muon_compact_cmcc_20261030/README.md)，不复制训练loss、优化器或生成／回训实现。新进程只将成熟fit的group索引绑定为0..31；NS5、FP32 momentum／master、BF16计算、joint clip1.5、fresh首梯度κ、aux Adam及LR序列保持。源必须通过现有C0完整来源、500／10K、保存、冻结和实际预算核验；新父布局须与已经验证的fixed2算子路径一致。

M1024各一次、Qasper32各两次、Hotpot96各两次，32组各M32＋QA8；R0、seed137、M／QA权重与既有目标保持。仅1024份once-mixed F-M目标可复用，学生轨迹和对应全词表QA教师目标全部重算。QA真实监督token逐步记录并汇总，不将等呈现／更新预算写成等token。新父的147个非投影字段、教师与Tokenizer身份、实际矩阵布局、冻结补集和κ都从本次原件核实。

`prepare.py`只在真实C0成品与pipeline报告齐备后生成本次prepared；`run.py`调用成熟混合Mu fit完成32组并保存BF16；`fresh.py`独立进程严格核全部字段、冻结补集、账本和同输入BF16输出；`evaluate.py`只接受这份真实32步成品，沿官方full8 64叶／24742和LongBench21／4750生成严格<0.9pp、logical≤2.6的完整九项。接口成功或质量`not_met`不等于目标完成。

新增导入边界在CPU检查中发现：旧数据helper会把S加入`sys.path`，从而让尚未导入的namespace子模块落到另一快照。新进程显式将四个项目namespace的子模块搜索固定到已选择的完整N，并保留实际导入来源断言；没有改任何冻结快照或放松检查。两份`rotation.model_utils.py`字节相同，但路径身份仍明确绑定。初次两份导入失败日志保留在本题验证根，正式训练未启动。

## C0训练结束后的单次接续

[after_control.py](after_control.py)只绑定已经运行的C0 v3：controller288378/start10671306、formal worker288496/start10671964及其实际argv／父子关系，源plan为`artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json`。它先安装目录事件监听，再结合两份pidfd等待worker退出及控制进程关闭退出码文件；同worker的原子终态记录也可证明退出码已发布。只看到文件出现、空文件或进程消失都不触发恢复，父进程死亡且无成功发布则停止。等待没有定时轮询，不重试、不重新挂起、不选新分支。

真实CPU内核协议的9项测试已全部通过，见[验证报告](../../result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/after_control_cpu_v2/validation.json)及同级`after_control_cpu_v2.log/.exit`。包含退出码延迟发布、关闭前不能启动实际CPU消费者、控制进程后续仍活跃时返回、非零退出不触发、父死／身份变化／重复claim等边界；没有导入torch、构造C0或调用GPU/pipeline。原17个Mu/C0/成熟helper源码SHA不变，训练数学和预算仍是原32更新＋一份完整九项。

本次实际命令由已激活bitvae的调用shell执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_muon_cmcc_20261030.after_control --prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_muon_cmcc_20261030.after_control --run
```

固定[执行计划](../../artifacts/fresh_muon_cmcc_20261030/after_control_C0_v3/plan.json)保留原1021项源码／输入绑定。`--run`在tmux `C0_Muon32_after_formal_20261031`中于01:05:35启动，PID420498/start11732332；挂载时实核内核poll阻塞、1个inotify及绑定288496／288378的2个pidfd。它已接真实C0完成事件，执行prepare／32更新／fresh后，因后述full9路径错误退出，不再活动。状态与日志为`artifacts/fresh_muon_cmcc_20261030/after_control_C0_v3/{run.json,continuation.log,run.exit,continuation.exit}`；子pipeline日志与退出在同目录`pipeline.log/.exit`。已使用的prepare/run不得重复执行。

成功事件只把控制移交既有pipeline；真实C0完整来源、实际布局、冻结状态、位宽和启动时GPU资源仍由原入口核验，不凭等待器“通过”跳过。恢复目标固定为`artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32`。原GPU5恢复与C0的GPU6/7评测已分别完成；本次续评按用户最新资源安排只使用GPU5，6/7留给其他实验。合法quality `not_met`是有效结果，不是执行失败，仍须等待完整结果后判断质量。

这次等待期间，本包和计划列出的1021项来源保持冻结；后续BR等适配放在独立目录，不覆盖本链代码或修改既有C0。无需安装新依赖、占卡或新增监控服务。

## 当前验证与使用

CPU原始证据位于`result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/cpu_final/validation.json`及同级`cpu_final.log/.exit`。检查复用了真实short2完成报告、独立fresh审计和成熟helper源码SHA，实际读取并验证全部H96 train输入／M目标、32组顺序及曝光；真实未完成C0被拒绝，CUDA未初始化。没有构造假父或模拟GPU通过。其后的真实32步／导出／独立fresh已完成，完整评测按下节续接。

调用者先激活`bitvae`。下列是原单次pipeline的复现命令，已经由上面的接续器执行，不能再次调用或重训同一预算；这里只保留真实命令。该入口不等待模型文件出现，不轮询、不自动重试：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_muon_cmcc_20261030.pipeline \
  --control-plan /data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json \
  --output /data1/cmcc/VAELLM/artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32 \
  --gpu 5
```

原pipeline的输出、资源安排及失败记录保留；本次只续接未执行的评测，不修改它的argv／状态或活动源码pins。资源占用、来源变化、技术失败或控制结果不完整会停止受影响依赖并保留现场，不自动安装、改环境、占卡或重跑。

## 已完成Muon32与GPU5完整续评

`artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32/{prepare,train32,fresh}.exit`均为0。真实C0父UUID为`cbb15c9f-1c1f-4394-817c-65c226d76ce7`，32步成品为`48c254f7-6a73-46f9-b42e-3f13b4083a9d`；独立[fresh审计](../../artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32/recovery/formal32/fresh_audit.json)核全部持久字段、冻结状态及同输入BF16 exact，实测logical `2.4987414816151494`。训练墙钟8024.704秒，fresh32.745秒；这些不是完整质量结果。

原`full9.exit=1`的根因是pipeline将`--output`设在`/data1/.../C0_H96_Muon32/evaluation`，而官方driver要求新目录在topic result根下。它在创建评测目录及GPU推理前被拒，没有成功full8／LongBench可复用；旧`full9.log`及失败`run.json`保持原样，不将此故障算作质量负例。现有CLI支持单卡，`--gpus 5`使full8两shard串行，LongBench官方21任务／4750样本完整分给同一GPU5；样本、生成参数、BF16参照及严格验收不变，只改变调度。

唯一续评命令在tmux `C0_Muon32_full9_gpu5_20261031`于10月31日21:54启动，driver `985518/start19225669`、评测worker `985523/start19225714`；现已完成full8 64叶／24742及LongBench21／4750，实际评测子进程exit0，19431.24秒约5.40小时。调用者已激活`bitvae`，CPU driver隐藏CUDA，再由原入口为worker分配GPU5：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.fresh_muon_cmcc_20261030.evaluate \
  --support /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support \
  --checkpoint /data1/cmcc/VAELLM/artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32/recovery/formal32/final_model \
  --pipeline-report /data1/cmcc/VAELLM/artifacts/fresh_muon_cmcc_20261030/C0_H96_Muon32/recovery/formal32/report.json \
  --output /home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/C0_H96_Muon32_gpu5_20261031 \
  --gpus 5
```

实际argv、PID/start、来源与退出状态写在[launch记录](../../result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/C0_H96_Muon32_gpu5_20261031.launch.json)，同级`.log/.exit`保存本次完整续评日志／退出；评测根内`evaluation.json`与各阶段日志记录进展。此输出已使用，不得重复启动。

原`summarize_pair_once.py`的首启缺`PYTHONPATH`错误及其后CPU配对失败均留在`pair.log/pair.run.json`，原launch记录也保持失败。第二个根因是本目录`evaluate.py`的入口只调用`main()`，未把summarize返回值传给`SystemExit`；完整`not_met`因此伴随真实`nine_task.exit=0`，被严格validator拒绝。这是CPU终态契约错误，不是完整预测失败。未改旧wrapper或已绑定源码；隔离入口正确传播原summarize返回值，在新目录发布真实exit1的相同acceptance，原严格校验和foreign-UUID拒绝均通过。

权威[同父配对](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/C0_vs_Muon32.json)及[四端点比较](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/four_endpoints.json)已完成；[CPU验证](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/validation.json)exit0，直接E整份新acceptance与原件完全相同，SHA为`164f567279631b78874bc398e52cf089c04d19ba04e42a502e99d40761b949ba`。九项由C0的63.58416595020284升至64.15392820484695，LongBench由41.74183992400309升至44.30836972082521；MMLU为62.39139723686085，较C0少1题。logical2.4987414816151494通过，但相对BF16九项仍低3.8932167131个百分点，不满足严格<0.9目标；不证明最优预算或跨模型迁移。

原单卡粗估5.69小时／预留6–7小时依据真实C0分区GPU时间求和，计算保留在`pair.run.json:timing_estimate`；实际评测为5.40小时，不外推为其它模型／生成长度保证。所有相关进程已结束。

本目录只核直接C0来源，不能冒称BR／dynamic／Rice通用；其它父源沿各自独立适配和生命周期证据。当前没有新增质量或可迁移性结论。

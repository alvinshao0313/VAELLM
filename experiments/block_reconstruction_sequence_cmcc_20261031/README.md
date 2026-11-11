# C0→BR36→Muon32 固定单次接续

本目录只接线[已定稿正式 BR36／BR→E](../block_reconstruction_formal_cmcc_20261031/README.md)。唯一研究安排、配方与预算见[当前计划](../../docs/experience/records/w25_transfer/current_plan.md#完整块预重建后接端到端恢复)。本次不改正式数学、活动 C0／Rice／直接 Muon 入口或其 pins。

`sequence --prepare`只写可审查的固定计划；`sequence --run`挂载实际 C0 事件等待并执行。2026-10-31 11:11挂载接续是历史启动事实；**BR36、两次fresh及BR／BR→E完整九项均已exit0，旧controller只在末尾CPU比较失败；11月1日已独立修复验收终态发布并完成同父四端点比较。** BR九项63.7982146076，BR→E为64.2484399018，比同父直接E高0.0945116970个百分点，均未达完整质量目标。模型／训练产物分别隔离在 `/data1/cmcc/VAELLM/artifacts/block_reconstruction_sequence_cmcc_20261031/BR36` 和同根 `BR_E32`，计划、日志、评测和汇总保留在 `result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/`。权重沿迁移后的 artifacts 集中布局落在 `/data1`，不会占用系统盘上的 result 空间或覆盖旧结果。

## 正式BR36结果与同父比较

BR成品UUID为`5f063691-e165-4884-b6a5-05c55593046b`，直接父源为已完成[C0](../fresh_fixed2_cmcc_20261030/README.md#完整结果与同父边界)的`cbb15c9f-1c1f-4394-817c-65c226d76ce7`。实际来源绑定中的C0 UUID、metadata SHA及权重SHA记录与C0评测一致；本轮直接重算metadata及评测原件SHA，未重新读取4.67GB权重计算SHA。BR独立fresh已核3675字段exact，其中1728个D字段为训练集合、1947个冻结字段exact，教师冻结，同输入BF16 logits exact。训练`report.json`中`fresh_pending`是训练发布时状态，由其后独立`fresh_audit.json`的`verified`闭合。

| 同协议指标 | C0 | BR36 | BR−C0 |
| --- | ---: | ---: | ---: |
| 九项 | 63.58416595020284 | 63.79821460755214 | +0.2140486573 |
| 八项 | 66.3144567034778 | 66.41505607260325 | +0.1005993691 |
| MMLU | 62.39851872952571 | 62.89702321606608 | +0.4985044865 |
| LongBench21 | 41.74183992400309 | 42.86348288714317 | +1.1216429631 |
| logical bpw | 2.4987414816151494 | 2.4987414816151494 | 0 |
| packed bpw | 2.5121161419412363 | 2.5121161419412363 | 0 |

MMLU从8762增至8832／14042题。九项中五项提高、三项下降、RTE持平；下降为BoolQ−0.3058103976、Winogrande−0.9471191792和ARC-C−0.1706484642个百分点。BR相对同协议BF16九项68.04714491790848仍低4.2489303104个百分点，严格掉点<0.9未达；logical≤2.6通过，rank0。两臂均为216 native＋36 compact、0 dense，tensor字节4671007680不变；完整目录BR为4674491313字节，较C0增加的是非tensor文件开销，不能混同packed位宽。

BR实际完成36层×32＝1152更新，用时9081.03秒（约2.52小时），另有导出27.01秒。该同父对照支持这份H96／LR1e-6／原500日程前32更新的BR中点小幅综合提升；它包含额外计算和训练暴露，不证明等算力优越、跨父／跨种子稳定性或BR→E必然获益。两份完整结果仍不满足最终质量目标，不能因此把实测增量抹为无收益。

原件：[BR九项及账本](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/single_C0_v3/evaluation_BR/longbench/merged/nine_task_acceptance.json)、[真实C0来源](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/single_C0_v3/C0_source.json)、[BR子源](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/single_C0_v3/BR_source.json)；BR训练及fresh为上述`BR36/{report,fresh_audit}.json`。BR acceptance SHA256为`9d32782ab43158ff51bd674d0017cbdf4e1bef1dd4c591edbe5fc61440b89305`，metadata SHA256为`b7a141c22e05cab85ad10669509c86c553ac6a3d65064b06ed6c33b63441fe0f`。

本轮逐一复核两臂64份full8逐题JSONL／24742条、21份LongBench预测／4750条的SHA和数量；full8正确数与九项汇总重算相符。两臂full8的doc／prompt／target身份及LB输入token身份、答案与生成预算完全配对，版本、九项协议和BF16基线相同。LB核对已有官方scores的SHA与宏平均，未重新执行逐题评分器。各评测子阶段和BR_full9均exit0；质量判定内部`nine_task.exit=1`对应合法`not_met`，不等于完整评测技术失败。

同父[四端点权威比较](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/four_endpoints.json)已闭合：C0／E／BR／BR→E九项分别63.58416595020284／64.15392820484695／63.79821460755214／64.24843990182039；`E−C0=+0.5697622546`，`BR→E−BR=+0.4502252943`，`BR→E−E=+0.0945116970`个百分点。BR→E子UUID为`c65929af-c37e-45c7-a85b-a0d2a99bf0a0`，直接E显式使用GPU5补评原件；四端点同C0、完整协议及logical2.4987414816151494。额外BR约2.52小时只带来本条件下的小幅末点增量，不称等算力优越或跨父稳定收益；没有重复训练／GPU评测。

旧comparison失败与直接E的CPU收尾失败同源：`recovery_evaluate.py`只调用`main()`而未传播返回值，使完整`not_met`报告对应真实exit0，原严格validator拒绝。旧summary还绑定直接E原错误目录。保留所有旧失败和原pins；隔离CPU入口正确传播原summarize返回值，只在新目录重发E／BR→E acceptance真实exit1，整份报告与原件逐字段相同，再由原validator严格通过并拒绝foreign UUID。见[核证记录](../../result/qwen3_8b_mmlu65_20261004/completed_comparison_cmcc_20261101/cpu/validation.json)；BR→E acceptance SHA为`c758b697c46444033d915db2a8a32f73d36a5676bcd5aeca5d67296731432e75`。原controller的失败状态不改写成成功，质量仍为`not_met`。

## 固定依赖、卡号和预算

唯一父源是 `/data1/cmcc/VAELLM/artifacts/fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json` 的真实 `C0_fixed2`：controller `288378/start10671306`，worker `288496/start10671964`。prepare 复核既有真实进程绑定、实际 argv、父子关系与 plan SHA；运行直接调用只读 `fresh_muon_cmcc_20261030.after_control.wait_formal`，等待该 worker 的 pidfd 退出加关闭成功发布，或同 worker 的正式 terminal record。单凭文件存在、PID 消失或部分模型不接续；没有轮询、定时器或第二套事件等待协议。

| 顺序 | 固定动作 | 资源 |
| --- | --- | --- |
| 1 | 真实 C0 成功事件后，用正式 reader 做 CPU 来源核证 | CPU |
| 2 | 全 H96 原160 entries／32组，层0→35各32更新，fresh AdamW LR1e-6／原500日程前缀 | GPU3 |
| 3 | BR 独立 fresh 全状态／同输入 BF16 核验 | GPU3 |
| 4 | 实际 BR 来源核证，原正式 recovery_prepare 验证父身份／数据与 fresh 恢复输入 | CPU |
| 5 | 实际 BR 子 UUID 上原 H96／Muon32 原32组，fresh optimizer、重算在线轨迹 | GPU3 |
| 6 | BR→E 独立 fresh | GPU3 |
| 7 | BR 中点官方 full8 64叶／24742及 LB21／4750 | GPU3/4 |
| 8 | BR→E 末点相同完整九项 | GPU3/4 |
| 9 | 同父 C0／直接 E／BR／BR→E 机械汇总 | CPU |

新增仅一次1152更新BR、一次32更新E、最多两份完整九项。BR单卡原预留2–3小时，现实际9081.03秒；E／九项另计。每个真实子阶段 exit0 后才启动下一阶段；官方完整九项的合法质量 `not_met`仍为完成。BR中点质量不门控E；本顺序先训练E再评中点，更不会由中点评分取消E。

本接续仅GPU3/4；GPU0/1/2留给既有C1链，直接C0→Muon当前续评仅GPU5，GPU6/7已属于用户独立实验。本页资源记录不扩大授权。正式入口在各启动点一次核卡，占用即技术停止，不等待资源、不抢占、不自动重试、不占卡。GPU3/4不在CPU prepare或事件等待期间分配。

## 准备、审查与单次运行

调用者在项目根目录激活 `bitvae`，核对 `which python` 与 `python -V`；命令经tmux/nohup后台执行。CPU driver设 `CUDA_VISIBLE_DEVICES=''`，每个训练/fresh子进程固定GPU3，评测driver隐藏CUDA后将3/4交官方worker。`plan.json`完整记录全部阶段argv、环境、输入和source SHA、实际父绑定及预算，不通过shell动态拼装CLI。

```bash
python -B -u -m experiments.block_reconstruction_sequence_cmcc_20261031.validate_cpu --output result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/cpu
python -B -u -m experiments.block_reconstruction_sequence_cmcc_20261031.sequence --prepare
```

prepare产物固定为 `result/.../block_reconstruction_sequence_cmcc_20261031/single_C0_v3/plan.json`。审核后只运行一次：

```bash
python -B -u -m experiments.block_reconstruction_sequence_cmcc_20261031.sequence --run
```

上述`--run`由主线程在tmux `C0_BR36_Muon32_after_formal_20261031`启动，PID`521982/start15374060`；启动时为`waiting_bound_C0_formal_publication`，当时实核一个inotify与指向C0 worker/controller的两个pidfd，无GPU占用。这是历史等待记录；[原状态](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/single_C0_v3/run.json)保留末尾comparison失败，前述训练／评测阶段已各自完成。日志为同目录`continuation.log`，shell退出为`continuation.exit`；固定plan SHA为`7cd7a93ec13085f97930728a1e0c6433666f90e372ece273c6cf7c64d053fa00`。运行复用原 `claim_once` 的 `flock(LOCK_EX|LOCK_NB)`及独占run记录发布，禁止并发／重复启动和自动续跑；失败保留现场。记录含主进程／各阶段 PID-start、实际argv、环境、日志、exit与时间。stdout/stderr合并落盘，各阶段`.exit`与最终`run.exit`区分技术成功与质量状态。原Python源码及其pins保持不变。

最终汇总只读一次已有C0和直接E完整结果；未完成的控制端点列为pending，保留本链两个完整端点，不轮询其他链、不重跑评测。四格完整后才计算 `BR→E−E`配方收益；尚不称等算力优越。两臂在线学生轨迹和监督tokens独立记录，不假称逐值相同。

## 验证边界

沿用[真实9项内核事件证据](../../result/qwen3_8b_mmlu65_20261004/fresh_muon_cmcc_20261030/after_control_cpu_v2/validation.json)及[正式BR已有证据](../block_reconstruction_formal_cmcc_20261031/README.md#当前验证与未验证范围)，不重复GPU短验。[新增CPU证据](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/cpu_final/validation.json)及同级`.log/.exit`已通过：固定命令／环境、实际短生命周期子进程的串行依赖与失败传播、新模块CLI、真实未完成C0拒绝及1003／1021／1019活动pins全部保持。没有构造完整C0/BR成功fixture。

真实prepare已exit0，固定plan含1061项source SHA及11项实际输入SHA；[prepared审计](../../result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/prepared_audit.json)记录准备时源码SHA、全计划读回及当时未启动状态。CPU测试后的源码改动限于pins合并冲突显式拒绝断言和 BR/E 输出改到 `/data1`；由真实prepare与针对路径／argv／跨盘／活动pins的读回覆盖，保留原CPU有效证据。旧未执行计划仅保留被替代SHA，不保留重复计划。没有重跑已过数学／事件／GPU测试。真实36层BR、BR→E、两次fresh和两份完整九项均已完成；11月1日的独立CPU验收重发布及四端点比较亦已严格核证。

经验参考为[真实读取上下文与dispatch](../../docs/experience/lessons/checkpoint_lifecycle.md#cpu构造原权重读取与真实导入上下文)和原等待器的关闭发布协议；本轮同父完整BR中点的证据及外推边界已原位归入[局部误差与跨任务结果](../../docs/experience/lessons/output_losses.md#局部误差改善与整模型跨任务退化)。

文档收尾已在`bitvae`的tmux `docs_closeout_20261031`执行：`tools/docs.py index` exit0，登记295份文档及18个别名；`git diff --check` exit0。全库`tools/docs.py check` exit1，仍有1655项历史／迁移缺链，本次两份新接续README、正式BR README及current_plan无缺链。原始日志及退出保留在本实验result根目录的`docs_index.*`、`docs_check.*`与`diff_check.*`；不将全库检查写成通过。

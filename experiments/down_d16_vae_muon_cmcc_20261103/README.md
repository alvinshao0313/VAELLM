# D16＋VAE-Muon唯一完整缺失格

同原C0 after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a` 的真实36个dense down fresh拟合。固定D16/K24/B16384/M36/S2各10K、E128/D128/R0、transpose=True、input保护256/block32旋转、原pooled BSQ/全部损失、seed31/shuffle600031，唯一拟合变量是矩阵optimizer由原BF16 AdamW变为原已审Muon；不加载D16或D32-Muon的down权重/恢复态作为初态。

本点使用root/Astra今晚明确给出的唯一预算，提前回答缺失格，不等待原三份九项作为启动前提，也不修改它们。复用C0原Adam拟合/恢复、D16原Adam拟合/恢复、D32-Muon拟合＋原Adam恢复三个控制；类别Muon500属于另一轴，不混入本组合。每个控制的完整UUID/原件SHA记录在prepared，尚在途的九项只读、不重训重评。不相加旧阶段增益预测新分数，不接双Muon、外置LoRA或参数网格。

## 实现与新shape验证

`native_fit.py`原样复制成熟down几何入口，实际命令式参数明确`batch_size=settings['batch']=16384`；不调用内置B8192的原`fit_next`冒充D16。`optimizer.py`原样复用已审MuV数学/辅助Adam/状态代码，仅改本adapter schema和真实owner总数767808→397728。四个真实owner分别按`[36,128,16]`、`[36,24,128]`、`[36,128,24]`、`[36,16,128]`做批量NS，共144独立矩阵；矩阵368640元素，8个bias/norm叶29088元素。四owner依次执行，没有逐144矩阵Python循环、不改pooled熵、不加新kernel；D16原生消费者可用，不导入D8 masked tile扩展。

参数/梯度/计算/NS保持BF16，无持久FP32参数master；momentum FP32，NS5、μ.95/Nesterov及原`.2*sqrt(maxshape)`，只一次BF16回写。aux原BF16 AdamW(.9,.95)/eps1e-8/WD0/base.003及原linear10K保持。每stage真实首梯度重算一个E+D共用κ，克隆原BF16 Adam实际舍入位移作参考，沿原最多2扩界/12二分/相对误差≤1%；候选复用方向，不重复FW/BW或候选NS，不搬D32κ、不分开配平E/D。

CPU只核新owner/初始化、显式batch、原父/runtime/配置及后缀绑定，成熟NS/aux/state审计按源码逐字节相同复用。stage0可直接CPU实核；真实stage间编码/extract_single会构造decoder，所以第二个裸CPU VAE并不是stage2 RNG路径。CPU绑定D16原stage2摘要，实际short与正式的两个factory都强制匹配D16原两stage flat摘要，不能用D32摘要或人为重设RNG去凑。

唯一short保持真实M36/B16384、两stage各2更新、原完整编码和保护/旋转，核两stage初始化/κ、BF16参数/grad、FP32 momentum及aux BF16 moments、真实更新、冻结补集、72份K24原pack精确读回、实际峰及原group8/独立fresh/BF16 logits。它是2步工程日程，不是正式10K前缀，short权重/optimizer/κ不用于正式。未新增通用exact-resume、旧控制评测或恢复500短验。

## 固定有限后台链

GPU仅1，调用者激活bitvae；tmux后台、所有stdout/stderr落盘、stdin断开。GPU1启动前核真实空闲，allocator保持`expandable_segments:True`；GPU0/2/3原九项及4/5/6/7安排不动。不占卡、不建监控守护、不等其它父PID、不重试或缩B/M绕过失败。

1. 实际short/保存/独立fresh≤30分钟。
2. 通过后从原after-up fresh正式S2各10K/编码保存/独立fresh≤90分钟。
3. 只接本点自己的raw UUID，原`flow.recover`/`MixedCatTrainer` fresh原Adam500/保存/独立fresh≤12小时。
4. 同最终UUID完整九项≤8小时，原64叶24742题/M B64/其它B32/0-shot无限额，LongBench原21项4750/max32K；全部detached LB子链继承同一绝对deadline、保留原grace。

后缀保持原五源train、BF16教师/loss、all252 FP32 D（D16约4966272元素）/LoRA0、B4/cap1024动态padding、stage37/data31、clip1.5、AdamW(.9,.999)/eps1e-8/WD0、decoderLR1e-5、500 cosine/warmup.01/save250/500。VAE optimizer state不导入恢复；没有额外raw八项或train readout。技术失败停依赖，不按loss、单项或中评取消已排定九项；quality `not_met`仍是合法完整结果。九项严格高于三个控制最好值才保留本父单seed组合候选，否则结束本点，不扩seed/LR/步数；若控制尚未出齐则只记录比较待补，不阻断本点完整评测。

部署结构预计同D16：logical2.497889643130095/packed2.511264303456182、tensor4,670,268,096B，必须以自身short/正式/final实际保存账本为准。全部decoder/码/scale/保护/旋转/compact frame及padding计入，单列embedding/head/norm和目录开销；不以理论码率冒实测。原同父和原生D16消费者规则可迁移，模型/类别变化时重新核owner、D/K/B、初始化、每stageκ、峰和完整费用。本点不是七类均改Muon或从BF16整链/跨模型已验证。

## 入口与状态

独立模型根 `/data1/cmcc/VAELLM/artifacts/down_d16_vae_muon_cmcc_20261103/`；证据根 `result/qwen3_8b_mmlu65_20261004/down_d16_vae_muon_cmcc_20261103/`。从项目根、已激活bitvae的caller运行；所有执行放入tmux/nohup：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d16_vae_muon_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d16_vae_muon_cmcc_20261103.sequence --phase short
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_d16_vae_muon_cmcc_20261103.sequence --phase formal
```

11月4日原正式fit／独立fresh及原Adam500／独立fresh均exit0，实际两段分别1181.9157／21494.0255秒；最终UUID `ce4759de-af8a-46d5-a487-1425c43ae566`。原full8已完整64叶／24742题、均分66.90161754238876%。原LongBench在native28050秒预算后自然结束为`partial_runtime_budget`／exit0，20项4250题及`repobench-p`前340题合计4590条；原full9和sequence因完整覆盖不足真实exit1，原绝对deadline与失败报告保留，不改成功、不对partial评分；4590条原始预测由隔离接续的完整4750条逐字节承接，旧重复文件已按下文无损去重。

root明确批准仅闭合该在途批次的必要缺失：独立[终态接续根](../../result/qwen3_8b_mmlu65_20261004/down_d16_muon_terminal_completion_cmcc_20261104/run.json)逐字节复制config、原4590预测及原native／outer证据，原`bind_terminal.py`先核对全部ID／连续前缀、实际输入token切片和生成参数。10:03在GPU3一次核空闲后，后台tmux `combo_lb_resume_1104`调用原`runner.run(resume=True)`，只补`repobench-p`原序列剩160条，10:13:05自然完成。底层已有resume语义由[隔离入口](../down_d16_muon_terminal_completion_cmcc_20261104/complete.py)调用，原已完成样本不再生成，未修改活动源码或协议。按原预测耗时p95=6.181565336秒计算native1170／outer1350秒，新增GPU总上限30分钟，实际新增native581.0751秒；没有第三次attempt、retry或延长。

11月4日完成原binder第二次连续性绑定、原merger／summary和同UUID validator；`completion.exit`、两binder、merge及comparison均0，`nine_task.exit=1`是有效`not_met`。完整64叶／24742＋LongBench21／4750齐全，foreign UUID拒绝、闭合时原输入／来源SHA及4590条旧prefix字节均不变，GPU3进程已自然退出。实际九项64.25774532381078%、MMLU63.13203247400655%、LongBench43.106767575186865%；相对BF16九项−3.789399594097702pp。自身成品实测logical2.497889643130095／packed2.511264303456182bpw，tensor4,670,268,096B、清理前完整目录4,673,344,612B；费用包含码、BF16 decoder、保护q／scale／index、旋转和compact frame，padding单列，非投影与序列化／目录开销另报。权威[验收](../../result/qwen3_8b_mmlu65_20261004/down_d16_muon_terminal_completion_cmcc_20261104/longbench/merged/nine_task_acceptance.json)、[完整LB原分数](../../result/qwen3_8b_mmlu65_20261004/down_d16_muon_terminal_completion_cmcc_20261104/longbench/merged/scores.json)及[三控制比较／实际成本](../../result/qwen3_8b_mmlu65_20261004/down_d16_muon_terminal_completion_cmcc_20261104/comparison.json)保留。

| 同父完整控制 | 控制九项 | 本点差值（pp） |
| --- | ---: | ---: |
| C0原Adam拟合／原Adam500 | 63.58416595020284 | +0.673579373607943 |
| D16原Adam拟合／原Adam500 | 64.27858659314296 | −0.020841269332180445 |
| D32 VAE-Muon拟合／原Adam500 | 64.30526846061926 | −0.047523136808479194 |

按预定“严格超过三控制最好值”规则，本固定组合点结束，不扩seed／LR／步数。相对MuV的M／LB分别+.6551773252／+1.1617057171pp，但RTE−2.8880866426、ARC易−.9259259259、ARC难−1.5358361775pp等使九项低于控制，不能只凭M／LB正差选择组合。拟合／fresh与原Adam500／fresh成本如上；原不完整full9耗28606.7974秒，必要接续含CPU741.2024秒，成本分开记录，不把预算缺口当质量失败或把旧失败回写成功。本结论只覆盖同父单seed的D16×VAE-Muon缺失格，不是七类／BF16整链或跨模型验证。

`prepared.json`及各phase的`*_run.json`、artifact report保留真实源／三控制UUID、D16 owner／初始化和batch、成熟Muon两处diff、原500全字段合同、实际命令／PID／start、源码SHA和退出。旧D16控制原评测路径缺终态时，独立比较引用其[同UUID CPU publication](../../result/qwen3_8b_mmlu65_20261004/down_d16_terminal_recovery_cmcc_20261104/publication/validation.json)，不回写原prepared或旧失败路径。无提交则不伪造commit ID。

经验依据：[optimizer必须区分阶段](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)、[新shape峰值范围](../../docs/experience/lessons/memory_precision.md)、[BF16参数与FP32动量重载](../../docs/experience/lessons/checkpoint_lifecycle.md#参数与优化器状态精度不同的重载)。本点为“既有单变量正差不可直接相加、M／LB改善不代替完整九项”的同父完整补证；唯一执行计划／common lessons／INDEX由root维护。root已按预定规则结束本固定点，生成载荷已在下述引用核实后清理；完整原始指标／4750预测／配置／核心日志／源码及真正退出始终保护。

## 已结束固定点的产物收尾（11月4日）

本点九项未严格胜过三个控制最好值，root依原预定规则结束，不安排新500／九项／复评或交付用途。删除前检查真实fit／Adam500／fresh均0、原full9／sequence失败1、独立160接续与CPU完整比较0，核活动FD／mapping及后继依赖；没有当前读者。gate活动修复版prepared的981个源码pins全部匹配，两份prepared原文件SHA不变；原prepared的三处旧source SHA差异严格对应其既有repair记录，未把历史旧pins冒称当前全同。尾阶CPU的实际源仍是原D16 raw02292fb6和C0 live after-up5a94d57f，均未用本组合载荷。原C0 raw／控制、D16 raw／最终、D32 VAE-Muon raw／最终及相关gate／尾阶来源保护。

本次删除36个精确文件：formal raw和recovery final的两份权重、两份用完same-input fresh载荷、checkpoint-500的master／optimizer／scheduler／RNG及与原C0逐SHA相同的6份tokenizer副本、21份旧4590重复预测、一个退役lock。逻辑9433344576B，按删除前st_blocks实际释放**9433452544B（8.785587GiB）**。此前三包13.317558GiB及本包旧short清理均不计入本次。精确路径／原因／size／st_blocks、旧预测替代路径／前缀bytes／原SHA及行身份digest归既有[启动核证](../../result/qwen3_8b_mmlu65_20261004/down_d16_vae_muon_cmcc_20261103/startup_verification.json)的`ended_fixed_point_cleanup_20261104`。

`artifacts/down_d16_vae_muon_cmcc_20261103/formal/raw_model`、`recovery/final_model`及原short目录现仅为metadata/config证据，**不可加载模型**；checkpoint-500只保留实际training_args.bin、trainer_state.json、mixed_training_meta.json和save audits等证据，不能续训。所有源码、完整full8原逐题／指标、完整4750预测及最终九项、比较／原config／两attempt原run、原partial／failed报告、真正退出与核心fit／恢复／失败日志保留；未删除原失败full9／sequence日志来掩盖预算缺口。

旧21份4590文件已逐项证明是保留的[完整4750预测](../../result/qwen3_8b_mmlu65_20261004/down_d16_muon_terminal_completion_cmcc_20261104/longbench/groups/part_01/predictions)的完整字节前缀；4590行的ID／输入token hash／生成选项和旧attempt时间／退出等唯一内容均承接或保留，新160条与全部最终预测未改。completion run的`original_inputs_sha256`及plan的`terminal_completion.preserved_original_sha256`仍保留清理前历史证明，**不再表示已删除旧预测路径可直接重跑check_hashes**；按receipt记录的replacement_path／retained_prefix_bytes可无损核回旧SHA与行身份。完整两attempt和原config／native/outer现场继续保护，没有为去重另建压缩归档。

删除后各目标已不存在，全部其余受保护证据SHA、gate活动pins／两prepared SHA、尾阶实际源及外部保护模型stat／小文件SHA均与删除前一致。本次仅有界收尾，无GPU、训练、评测、安装或Git操作。

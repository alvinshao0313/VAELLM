# 只跳过 up 后恢复的完整 S4 候选

本候选从原始 `/data1/cmcc/models/Qwen/Qwen3-8B` BF16 重新执行完整七类 S4 拟合，预先声明的算法变化是 up 类压缩后的累计 prefix 恢复为 **0 更新**。q/k/v/o/gate/down 的数据、恢复范围、AdamW、LR500、整数预算规则、RNG 隔离、BF16 提交、保护及拟合参数均继承[原完整候选](../s4_from_bf16_20261102/README.md)，旧源码及1101项来源冻结。新增路径CPU／GPU验证已通过，正式于11月4日18:22完成同一最终UUID完整九项，整链exit0；本轮已结束，第一阶段质量目标仍未通过。

原 trial_1100 已完成同一 UUID 的完整九项：MMLU9931／14042、九项67.4412149358%、LongBench49.1350779649%，仍未达到[当前第一阶段目标](../progressive_residual_bsq_20261030/PLAN.md#第一阶段当前质量目标11月3日确认)。11月3日用户已将九项均值验收改为相对BF16下降严格<0.4pp，MMLU最多掉1pp保持；仅更新验收目标，本轮冻结的训练预算／停止规则保持。其中 up 后恢复是整个216投影 prefix 的128更新，MMLU10105→10027；这支持检验该插入点，并不说明 up 权重本身有问题，也不预期跳过就机械追回78题。down64→128的负差不在本轮同时修改，避免两个干预。实时安排归[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。

## 完整结果与本轮结论

最终模型UUID **`a6a04164-2782-4b19-965f-410198c3ddde`**，路径`R/formal/nodes/down_proj/post/model`。`run.exit`、`continuation.exit`、`formal/fit.exit`及MMLU／seven／两分组LongBench预测、绑定和合并退出均0，最终评分来自同一模型。权威原件为`R/formal/result.json`及`formal/longbench/merged/nine_task_acceptance.json`，原始64叶／24742条短任务样本和LongBench21任务／4750条预测保留。

| 指标（%） | 本轮skip-up | 原trial_1100 | BF16 |
| --- | --- | --- | --- |
| MMLU | 71.00840336 | 70.72354365 | 73.00242131 |
| ARC-C | 54.77815700 | 54.77815700 | 56.56996587 |
| ARC-E | 79.33501684 | 79.37710438 | 80.80808081 |
| BoolQ | 86.33027523 | 86.54434251 | 86.57492355 |
| OpenBookQA | 40.80000000 | 41.40000000 | 41.80000000 |
| PIQA | 77.80195865 | 77.42110990 | 77.52992383 |
| RTE | 79.06137184 | 77.97833935 | 77.97833935 |
| WinoGrande | 69.37647987 | 69.61325967 | 68.50828729 |
| LongBench | 49.35797783 | 49.13507796 | 49.65236225 |
| 九项等权均值 | **67.53884896** | **67.44121494** | **68.04714492** |

未舍入九项均值 **67.53884895764816%**，较原trial_1100高 **0.09763402189420844pp**、较无恢复S4高 **0.87668635942041pp**；但较BF16仍低 **0.5082959602603268pp**，离严格<0.4pp目标还需提高 **>0.10829596026032pp**。MMLU **9971／14042**，比原trial_1100多40题，但比BF16少280题／1.9940179461615202pp，距至少10111题还差140题。终态八项均值 **69.81145784848576%**；联合恢复前只有完整MMLU，没有八项均值，不能据此判断末段九项净损益。

末down恢复的本次同轨迹MMLU为 **10007→10002→9971**（0／64／128更新），128净损36题／0.25637373593506pp；原规则在128结束，没有回滚或追加。这个重复出现的末段负差支持下一批先讨论末段恢复的必要性／配方，尚未定位步长、目标或范围的唯一根因。q..gate在跳过up以前已与原trial_1100有数值差异，gate实际128更新而旧轮64，所以整体小幅正差不能作为skip-up独立因果或稳定优胜的证明。

最终252目标投影全部S4，未保护主码率4bit；含decoder、INT8保护indices／scales／values与旋转的logical／packed均 **4.094031292459239bpw**，padding0、无账本违规。全模型tensor6044459520B，权重文件6046045505B；embedding／head／骨干norm按其实际存储计。正式拟合／逐段恢复与读出于11月4日15:10:01结束，76042.528696秒（21.1229h）；含新增验证和最终评测的完整driver共89253.433084秒（24.7926h），18:22:33结束。两卡6／7运行，墙钟不冒充实测GPU活跃时数。

本轮只交付候选和证据，未宣布S4配方定稿，不启动门控／降阶。用户11月4日在结果与定位方案讨论后，已批准从下述down/pre父开展M-only与joint半LR各128更新的同父实验、父0九项补评及早／晚／未训练三组诊断；实际新包与运行状态见唯一执行计划。它们回答固定父的恢复问题，不改变本轮已结束合同，也不把新的局部收益冒称BF16整链结果。BR等其它条件候选仍未提名。

**已保留的恢复前对照父：** `R/formal/nodes/down_proj/pre/model`，UUID `fa50bf40-c682-4910-8e72-d024f6b678e4`，原生权重6046045505B，runtime／config／tokenizer完整。它是本轮七类都压完、末段联合恢复尚未开始的S4模型，前面类别的既有恢复已包含其中；完整MMLU10007／14042。用户11月4日追问时已再次实核文件存在；随后已批准的恢复前能力／同父恢复对照使用此父，最终post作为原joint控制与本轮交付候选同时保护。

## 算法与复用边界

正式入口要求所有252投影从原始HF dense开始，禁止任何fixture或旧压缩checkpoint作为正式父。原 q/k/v/o/gate/up/down、M36、每类S4×10000拟合、D/K、vector batch、AdamW拟合配方、HF512×256保护统计原样保持。每个正常恢复段仍训练完整累计prefix、fresh AdamW、FP32 masters/BF16 compute、18／18双卡、LR1e-6与原500线性时域；64／128／256／384的初始分档与15题边际规则不变。

up 的新分支在读取训练组或创建Adam之前返回。完整持久字段、参数身份／版本／requires_grad、模型模式和Python／NumPy／CPU／双CUDA RNG均核前后exact；actual_updates、optimizer_created、训练组与token曝光均为0，原生history记录`did_train=False`、`new_Adam=False`。up新pre／post只有在完整state、config和converted topology核同后才复用该次新up/pre实际M读出。post仅写`mmlu_reuse.json`，保留原评分／样本的真实源UUID与路径，不伪造新的post推理、退出或改原raw指标UUID。down的previous_post取这个本轮up真实读出，不使用旧10027或pre-gate分数。

新包只保留需要改变语义的类别driver和支持单组增量验证的恢复loop；其余panel、snapshot、训练state、原数学、budget和消费者只读复用。私有模块实例只绑定新合同，不改旧canonical模块全局。CPU会证明：去除up早返回和validation单组入口两处后，恢复loop AST与原函数完全相等；旧合同全部已有字段保持一致。

## 最小新增验证

只验证真实 **up 新S4拟合→skip0→down新S4拟合→一次真实恢复→保存／fresh**，不再重做原七类短验、峰组、B64、中评回训或resume矩阵。那些已有效的证据直接绑定 trial_1100 的真实short/source。

局部验证夹具从旧 `formal/nodes/down_proj/pre/model`（UUID `54f2452f-59b1-4249-b4cb-45586915966f`）取 q..gate 的180个冻结S4投影，再把up/down全部72个投影替换为原HF真实BF16。这个prefix可能包含旧后续恢复历史，只是结构／接口夹具，**不是新候选的前五类训练轨迹或正式初始化**。fixture、请求、metadata和日志均标`validation_only`／`incremental_validation_fixture`，旧源只读；正式请求不含fixture并另从HF新起。通过原native category-resume，在fixture之后只实际拟合up/down，仍完整M36、S4、正式batch和全部编码，每阶减少为2更新，down只执行group0的一次真实更新；模型规模和算法未缩。

CPU仅做新增合同／绑定／AST、真实夹具strict重载、180旧prefix／72HF边界、原生resume首待up与runtime identity检查。CUDA隐藏时原FA2构造器在载权重前拒绝CPU；因此只有CPU夹具张量构建显式使用SDPA且不执行前向，保存前恢复FA2 policy。GPU验证和正式仍使用原冻结native FA2工厂／消费者，CPU构造差异不进入GPU计算。

## 入口与记录

源码在 `experiments/s4_skip_up_20261103/`。为复用成熟通用消费者的受检目录范围，新产物使用独立子根：

`/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/<unique_trial>`。

```bash
# 调用者激活bitvae，核which python与python -V；实验脚本不自行激活环境。
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_skip_up_20261103.run --prepare <unique_trial>
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_skip_up_20261103.run --plan <unique_trial>/plan.json
```

均由tmux／nohup后台执行、stdin断开、stdout/stderr与退出落盘，只用物理GPU6／7。prepare建立CPU fixture与独立validation／formal请求；固定链只有新增路径通过才从HF启动正式，无自动重试／监控自动化／占卡。正式各节点按原规则观察与决策，最终同UUID完整MMLU／seven／LongBench21／4750，逻辑与packed必须实核4.094031292459239bpw（含码、decoder、保护值／indices／scales和旋转，embedding/head/norm另计）。该4bit研究父仍不代表最终≤2.6bit达标。

旧首轮结果、源码、保护的pre／final模型由主线程维护；本包不清理旧产物。自己的局部fixture、validation状态及无后继模型，在必要证据和指标归并、进程退出后按精确路径清理。正式末pre、真实训练state与final按明确后继／完整验收用途保留。若down128，整链22–26小时双卡仅为旧吞吐粗估；down预算变化会改变成本，不是完成承诺。

首个CPU准备 `trial_20261103_1724` 在任何模型载入／GPU运行前因原FA2构造器拒绝CPU而exit1，日志、精确异常与来源边界保留；没有正式或GPU任务开始。修复仅限上述新CPU夹具构造。

## 当前实际运行

`R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/trial_20261103_1730`。CPU准备及终态caller增量核证均已exit0：1108个源码／数据绑定、原合同已有字段不变、恢复loop归一AST相等、实际fixture strict读回／前5完成且首待up／runtime identity与冻结snapshot导入路径通过；CUDA未初始化。终态只替换summary调用模块，真实旧bind/merge plan schema保持，实际新CLI `--request/--summarize --help`退出0，没有重评旧模型或指标。最终plan SHA **`57c860153acd106dc6233c7f700fa3ab3202f1c9257a4e318d4ed4bbdb35e958`**。

17:35在已核bitvae／Python3.11.16的tmux **`S4_skip_up_20261103_1735`** 启动上述`run --plan R/plan.json`，CPU driver **2296635/start43588090**，mask6,7。启动前两卡各1MiB且无compute进程；0–5既有任务未改。增量validation worker **2296996/start43588764** 已退出0，`R/validation/validation.json`为`verified_incremental_up_skip_down`，不代表候选质量通过。

真实up新S4拟合197.096秒，进入skip后实际0更新、无Adam／训练组消费，完整持久state、参数身份／版本与全部RNG exact。原生history的`actual_updates=0`、`did_train=False`、`new_Adam=False`。本轮up/pre UUID **`9c32dd10-2052-4e1c-b9da-9a2587450305`** 与post UUID **`e6dc7f89-5285-4f38-ac48-e96d9df33dde`** 在完整state／config／topology核同后复用pre的实际114题短读出，真实来源写在`validation/nodes/up_proj/post/mmlu_reuse.json`，没有伪造post新推理／raw指标。此短读出不触发正式自适应预算；正式down按代码取本轮up真实完整M读出。

随后真实down新S4拟合264.758秒，拟合前后完整非当前字段exact，验证up0到下一类fit衔接。down执行group0、LR1e-6的1次真实恢复，更新耗时227.086秒，252投影decoder梯度finite且非零，完整冻结补集exact。原生BF16保存后fresh实际读回 **3927** 持久字段exact，真实forward有限，逻辑／packed均为 **4.094031292459239bpw**。证据分别为`R/validation/validation.json`、`validation/nodes/down_proj/recovery/training.json`与`validation/nodes/down_proj/post/evaluation/fresh.json`。原七类短验、峰／B64、中评回训及resume证据只读复用；Astra终审确认1108 pins与新增证据匹配、无阻断，无需扩验。

固定链通过上述新增路径后，已启动正式worker **2318969/start43751460**：

```bash
CUDA_VISIBLE_DEVICES=6,7 /home/cmcc/miniconda3/envs/bitvae/bin/python -B -u -m experiments.s4_skip_up_20261103.fit --request R/formal/request.json
```

此处`R`须替换为上面的绝对trial路径。`formal/fit.json`记录`complete_fresh_HF_candidate`、`old_compressed_inputs=[]`；`formal/factory.json`证实原始HF的399持久字段／252普通dense投影、`validation_only=False`。18:07有限核证q stage1已至少 **5700／10000** 实际更新（loss13.053／recon12.938／commit0.11567，约0.0280秒／更新），无2步限制残留。它没有读取局部夹具或旧压缩产物。

实际PID／代次／命令／退出在`R/run.json`；正式stdout/stderr为`R/formal_fit.log`，固定链日志`R/continuation.log`，新增验证核心日志`R/validation_fit.log`。CPU证据在`R/CPU_validation.json`及`R/fixture/fixture.json`。

11月4日结束实核：七类新S4×10000拟合均完成，q／k／v／o／gate／up／down恢复分别64／64／128／64／128／0／128更新，总576。原driver2296635/start43588090与formal2318969/start43751460均已正常退出，`formal/fit.json`为`complete_fresh_BF16_S4_prefix_recovery`，`run.json`为`complete_HF_skip_up_candidate`，最终九项齐全；没有继续占卡或启动后继。

| 累计节点 | 新压缩后／恢复前MMLU正确数 | 本类恢复后正确数 | 实际恢复更新 |
| --- | --- | --- | --- |
| q | 10234 | 10243 | 64 |
| k | 10232 | 10215 | 64 |
| v | 10159 | 10194 | 128 |
| o | 10177 | 10227 | 64 |
| gate | 10183 | 10171 | 128 |
| up | 10076 | 10076（同state复用） | 0 |
| down | 10007 | 9971 | 128 |

表中MMLU均已核57学科／14042题、完整无limit、非validation；up/post保留真实pre评分来源，不生成假推理结果。它们是本轮累计流程节点，不当作各类独立敏感度。down恢复前为71.26477710%，与终态71.00840336%分开报告。q..gate本轮读数已与旧trial_1100有差异，终态差值不能全部归为单独跳过up的确定因果。

原算法／预算／数据／LR均未变更，末down实际128更新后按既定规则退出。没有因阶段MMLU负差取消后继评测，也不将正常结束写成质量达标。后续只执行已讨论确认的批次，不添加轮询／重试／占卡。

本次用完的局部fixture、up/down验证pre／post模型、runtime、tokenizer副本已按各自完成节点清理；验证末`validation/nodes/down_proj/recovery/training_state.pt`（120497016字节，SHA `0b28dabfd632280ff2afce21cbb2e2c35a8a96047bca208927298aa715a13482`）在worker退出0且无续训用途后删除，精确前后记录追加到`validation_fit.log`。本trial共删除35个生成文件，逻辑 **41315444011字节**、实际分配 **41315553280字节**，均核不存在；只统计本轮删除。保留源码／合同／hash、config与metadata、expected_state、原始指标和核心日志。旧源pre／final未动；正式末pre／state／final按后续读出和明确用途保护。

11月4日正式结束收尾：1108 pins与plan SHA仍全匹配，64份八项raw及21份LB预测SHA／行数核证通过；33份正式exit全0，旧进程代次均已退。只新增删除无后继精确续训用途的`R/formal/nodes/down_proj/recovery/training_state.pt`，逻辑120497656B、实际分配 **120500224B**（约114.92MiB），SHA `282cd2c6217a3ce12f74a9e57f078defff2bd92f9f3d3d5b6a6ca87cd1a1b101`；精确删前／删后记录在原`continuation.log`，已核移除。down/pre与最终down/post及其25个保护文件stat未变，原始指标／配置／metadata／expected_state／日志继续保留。删除后不再提供本轮末点的optimizer/RNG精确续训状态；同父fresh恢复与模型评测不受影响。上段35文件清理属于启动验证阶段，本次仅新增计1文件，未重复统计历史释放量。

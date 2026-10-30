# Softmax 数值隔离诊断

仅诊断 [A Fix for Softmax and Cross-Entropy](https://jonbarron.info/softmaxfix/) 的 Eq. 5、6、9 与当前原生 KD 数值路径。不正式整合，不训练，不改全局 softmax、FlashAttention、温度、CE 混合、mask 或归约，不用 loss 变化声称九项收益。

第一步由 Astra 审定为 G02 的 BoolQ `train` 原顺序前 4 文档、全部 8 候选分支及完整 continuation；一次加载冻结原生 BF16 G02，先按缓存生成的逐分支完整 prefix 前向，再复用正式 `nqa_retention.task_batch` 的左填充整批前向。无需新 HF 教师或 QA 生成，不新增 FP32 decoder/live 路径。G02 与 R896 身份来自各自已验证 manifest；不重建或修改原缓存。

四类差量分别报告：

1. 同一个学生 raw tensor、同一个既存 FP32 logp 目标下，当前 FP32 KL 对保留目标数值的 FP64 参照的误差。
2. `p=exp(cached_logp)` 的总质量、缓存对 serial replay、固定 raw logits 下 FP32 logp 表示对 FP64 教师归一化概率的差。缓存对 replay 还含历史前向/核差，不将它全称量化误差；本次不分离 BF16 head logits 自身的量化误差。
3. 同一个 BF16 raw tensor 克隆师生的标准路径残差与 Eq. 9 恒等控制。缓存精确目标本身的非零梯度另列，不能都算伪梯度。
4. 同 G02 的逐分支与 padded batch 两条真实前向的差及其 FP64 梯度；这不是不同模型实验，也不覆盖训练 FP32 master 与部署 BF16 的差异。

当前缓存 KL 的精确梯度为 `a*q-p`，其中 `a=sum(p)`；FP64 参照保留 `a`。把缓存当 raw teacher logits 套 Eq. 9 会归一化成 `p/a`，改变目标及梯度规模。论文 Eq. 9 在本目录只用于明确的 raw teacher logits，不覆盖现有训练调用。报告绝对误差、参照范数和真实路径梯度尺度，不用近零分母的相对误差判断收益。

`numerics.py` 是 PyTorch 算子翻译；`test_math.py` 仅检验 CPU 数学性质，包括高置信 CE、恒等师生、近教师梯度、非归一缓存质量和 FP64 autograd。合成算子测试不证明整模型链路正确。

主线程分配 GPU1。调用者先激活 `bitvae`，核对该物理 GPU 无计算进程且空闲，在原 snapshot cwd 用 tmux 后台执行：

```bash
CUDA_VISIBLE_DEVICES=1 python -B /home/star/VAELLM/experiments/softmax_numerics_20261029/capture_r.py --gpu 1 --output /data/star/VAELLM/softmax_numerics_20261029/R_boolq_first4
```

输出统一在 `/data/star/VAELLM/softmax_numerics_20261029/`。`R_boolq_first4/capture.pt` 保存选中输入、serial/batch BF16 raw logits、原 FP32 缓存和重播 FP32 logp、源身份及实现 SHA；`audit_cuda.json` 为同卡真实算子 FP32/FP64 报告。`run.log`、`run.exit` 区分运行与完成。冷加载复用既有原生 loader，已有 R cache 实测峰值 allocated 72.53 GB／reserved 76.18 GB，需要一张完整空闲 A800 80GB；本任务不降低正式输入或切核来规避资源。

已查 [KD 与数据边界](../../docs/experience/lessons/kd_and_data.md)、[Checkpoint 与数值路径](../../docs/experience/lessons/checkpoint_lifecycle.md)、[资源与精度](../../docs/experience/lessons/memory_precision.md)。既有经验要求分离 live/native 参数相关路径差、目标/归约与实际任务收益。未发现已完成的同类 softmax Eq. 9 原生数值审计；本次提供该隔离证据，不扩大为完整质量实验。

## 10月29日首轮结果

CPU 5 项数学检查及真实捕获入口导入通过。GPU1 的 tmux `softmax_G02_Rboolq_GPU1_20261029`、PID `346844` 已正常结束，`run.exit=0`，捕获耗时 26.7253 秒，结束后 GPU1 为 0 MiB／无计算进程。模型冷加载峰值 allocated 72,534,277,120 B／reserved 76,174,852,096 B。G02 UUID `38a7461e-a2f1-4586-80e2-b99883ef2628` 的全部 3423 持久张量前后摘要均为 `3401c96dcbd83bdfb354272d17aedb3ee13d997cac028ade3626af78aafd8b3a`；运行后重核源码及源模型 audit／R cache manifest SHA 未变。

实际 BoolQ train ID 为 `1668,2041,4723,85`，完整 8 分支共 8 个 continuation token；正式左填充 batch shape 为 `8×239`。完整原始数值、实际 argv／Python／cwd／物理 GPU／代码 SHA 与模型不变证据在 `/data/star/VAELLM/softmax_numerics_20261029/R_boolq_first4/audit_cuda.json`，raw tensor 复算输入在同目录 `capture.pt`。

| 观察量 | 实测 |
| --- | ---: |
| 保留原 R 归约的 FP32 loss | 7.73674764786847e-5 |
| 相同缓存目标 FP64 loss | 7.74213814408083e-5 |
| FP32 公式梯度误差 L2 | 6.2184685429e-9 |
| 相同缓存 FP64 参照梯度 L2 | 5.5409655304e-4 |
| raw serial 对 padded batch 的 FP64 梯度尺度 L2 | 5.5409618594e-4 |
| 反传至 BF16 logits 边界后的公式梯度差 L2 | 3.4928342306e-10 |
| 同一 BF16 边界参照梯度 L2 | 5.5430585132e-4 |
| 原缓存与 serial FP32 replay 完全相同 | 8／8 分支 |
| FP64 读取缓存的最大概率总质量偏差 | 8.2852456296e-7 |
| raw identity Eq. 9 非零梯度元素 | 0 |
| raw identity 标准路径各分支加权梯度 L2 | 8.16e-10～2.70e-9 |
| 同 identity 缓存精确目标的 FP64 梯度 L2 | 5.93e-11～1.76e-10 |
| 本窗口最大 argmax 对其余 LSE 的 margin | 1.4172502377 |

本窗口 FP32 公式误差约为相同目标参照梯度范数的 `1.12e-5`，远小于这里两条真实原生前向路径的差。serial／padded batch 的最大 logits 差为 `.140625～.25`；这是执行路径差的尺度定位，不能只因同模型非零 KL 称缓存 bug、要求改缓存或引入同批教师。8／8 重播一致只排除了这 8 分支的缓存内容漂移。没有 token 达到 FP32 的约 `16.6` margin 阈值。

经 Astra 审查，本轮证据足以停止扩大 R 采样或人为降低精度放大误差；不据此正式整合或训练。结论仅限本 BoolQ4doc、native BF16 serial／leftpad、相同 cached 目标的 **logit 梯度**。未测参数 Jacobian、Adam 更新、全部 R、M／QA 真实模型、FP32 decoder/live 或九项质量。M 目前只有合成四类算子数学控制，不能冒充真实 G02 M 梯度。后续若研究 M／QA，须先说明其独立决策价值。

本轮可复用增量是 raw logits 与缓存质量语义必须分开，以及应先将数值噪声与实际运行路径差同尺度比较；不新增质量改进结论。没有生成模型权重或一次性测试夹具；保留上述最小 raw tensor 证据、唯一数值报告、核心日志、退出码及 CPU 数学日志供复算。

## 10月29日 M／QA 研究与唯一后续窗口

此节由独立研究分工核查，**只完成一手机制映射、实际缓存CPU筛查和下一轮设计，没有运行新模型前后向**。R首轮结论与输入保持，不延伸R采样；当前不正式替换loss，不申请GPU，不启动数值修复训练。

一手依据为作者[项目页](https://jonbarron.info/softmaxfix/)与[论文v1](https://arxiv.org/html/2610.31642v1)：Eq.5／6避免高margin主类梯度消失；Eq.9用raw师生logits差处理接近教师时的消减。论文蒸馏例子是同权初始化的小型模运算模型，T=4；其Adam漂移不能外推为Qwen恢复已经命中，也不能把logit噪声直接与参数Adam的epsilon比较。

### 实际调用链与精度

| 路径 | 本项目实际目标与归约 | 对论文机制的约束 |
| --- | --- | --- |
| M | 原`qwen_mmlu_target_mix32/targets.py`以`logaddexp`将full-F教师与gold各半**混合一次**，保存FP32 logp；`qwen_aux_mmlu_batch_probe_20261011/mmlu.py`以完整原生M32左填充、末query四个ABCD续写token执行KL，逐文档权重`.5/32` | 不是运行时两个独立CE／KD项。近full-F不等于近混合目标；不能重新混合或把Eq.6另加一次。 |
| QA | `cat_aligned_opd_20261028/rollout.py`先暂用native BF16学生生成，恢复原FP32 decoder master，再用当前学生轨迹给full-F打分；`answer_core.py`保持完整prefix、真实答案query、151936词表，先答案token均值再文档`.5/8` | 不能改成gold轨迹／截断prefix／top-k来诊断。教师是full-F，学生是压缩模型，无同权近恒等前提。 |
| R | 原学生原生serial缓存对当前left-pad学生，原分支／文档归约 | 已有首轮量级定位，当前不重做。 |

M与QA的head logits均显式要求BF16，但在`log_softmax`前均调用`.float()`；本研究关注的公式cliff是**FP32约16.6**，不能拿BF16约5.5判当前loss已collapse。反传仍会跨回BF16 logits边界，须单独报告梯度量化；它不同于head前向舍入，更不同于live decoder与native decoded cache的路径差。

FP32 master/live不是可省略的最终验证：若问题只到固定logits的公式，离线FP64对照足够；若要判断恢复更新，必须在真实live decoder图上验证。这里的live仍有BF16解码运算与head，不能称全模型FP32。依据还包括[KD经验](../../docs/experience/lessons/kd_and_data.md)中10月14日实测的参数相关live/native差；该差曾改变有限步loss方向，不能统一扣除。

### 原首组M32的CPU结果

复算入口为[screen_m_cache.py](screen_m_cache.py)，输出 `/data/star/VAELLM/softmax_numerics_20261029/M_group0_cache_screen/report.json`。实际`bitvae` Python3.11.16／PyTorch2.6.0+cu124，tmux `softmax_Mcache_CPU_20261029`结束，`run.exit=0`、`cuda_initialized=false`。读取原`qwen_hotpot_recovery_20261012`数据与full-F缓存，核对数据／缓存文件SHA、每条teacher张量hash、input signature、四候选及原`.5/32`；选择正式group0原顺序32条`cais_mmlu/auxiliary_train/train`，不按正式测试表现选样。来源版本与数据协议沿用绑定的原dataset/cache manifest，未新建训练池。

| 观察量 | 实测 |
| --- | ---: |
| 混合目标logp所隐含的gold margin范围 | 0.0001921241～14.2812791324 |
| 上述margin>5.5／>16.6 | 25／32；0／32 |
| 原教师缓存`exp(logp)`总质量最大偏差，FP64读取 | 1.0431996333e-7 |
| 实际一次混合后缓存总质量最大偏差，FP64读取 | 6.8821627841e-8 |
| FP32混合logp表示相对同原缓存值的FP64半教师半gold混合，最大概率误差 | 3.4302496399e-8 |
| G09 directOPD真实step1加权M／QA loss | 0.1465826035／0.1919601748 |
| 同step1联合参数梯度norm，clip前 | 122.5473709106 |
| G09 directOPD真实step2加权M／QA loss | 0.0553351566／0.2436308591 |
| 同step2联合参数梯度norm，clip前 | 101.4266967773 |

目标margin不是学生margin；上述3.43e-8只分离**混合运算及其logp表示**，没有raw教师logits，不能称原教师log-softmax误差已分离。原始日志前两条完成更新的行SHA保存在报告中；它们仅表明本次窗口不是整体零loss，不排除某个token／参数方向接近零。联合norm不能推导逐域梯度，更不能据此断言数值误差对Adam无影响。CPU诊断初版因训练日志含非JSON启动行而退出1，已改为只解析目标JSON event并成功重跑；没有触及活动训练源码／日志／状态。

### 下一轮只选G09初态的正式group0 M32

选择M而暂不做QA：原M目标含高置信样本，支持集小、原始缓存完整且可复算；可以一次确定FP32公式误差是否穿过BF16边界，成本低于重新加载full-F并收集QA轨迹。选择预先固定group0而非挑最大误差样本。G09初态UUID为`e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，源与完整参数绑定复用`/data/star/VAELLM/g09_recovery_20261029/direct_opd32/prepared.json`；不得把已有G02当作G09。以下为执行前固定的补测方案，实际运行结果见下一节：

1. 独立加载同G09初态，复用正式all252／1512 FP32 decoder叶、双层checkpoint、训练mode及实际M32输入。绑定原32条完整prefix、left-pad mask／position_ids、四个续写ID、原混合缓存、T=1和`.5/32`。不优化参数，不加载QA教师；保存原生BF16末query选中logits及其计算图对应的来源／代码SHA。唯一输出使用新目录，GPU由root安排，预计需要一张完整空闲A800；不修改运行中的fit或loader。
2. 固定该raw张量与同一个缓存`lp`，原FP32 KL和FP64同目标参照分别计算loss与logit梯度。令`p=exp(lp)`、`a=sum(p)`，参照为`(.5/32)*(a*q-p)`，用Eq.5的FP64稳定log-softmax和autograd交叉核对；报告每行margin、真梯度norm、误差norm／max、零项及错号的绝对量级。不得用`p/a`替代实际目标，也不把原缓存当raw full-F调用Eq.9。
3. 把两套梯度分别转换到**同一BF16 logits反传边界**，当前FP32梯度转BF16，参照FP64梯度先转FP32再跨相同BF16边界，记录保留差量。该比较仅识别反传边界舍入；本轮没有未舍入head accumulator，不声称测得纯head前向量化误差，也不另换FP32 head／内核制造对照。
4. 若边界完全一致，本窗口到此停止；否则用同一次真实live计算图作两次VJP，分别注入当前／参照边界梯度，比较全部实际1512叶的参数梯度及两次结果之差。保持原文档权重，记录全体及每叶的绝对误差、信号范数、有限性；不假设有限精度下单独反传差量与两次完整VJP之差逐位相等，不以logit结果冒充参数结果。只需这一窗口，不拓成层／任务穷举。
5. **本窗口不产生Adam收益结论。** M单域VJP不能替代M32＋QA8联合梯度、原全局clip1.5、AdamW历史矩及epsilon1e-8。只有上一项显示值得继续的参数差量，下一轮才考虑复用一次完整原组及真实优化器状态做同一步提议比较；比较后BF16部署位移，仍不得用局部步差冒充九项质量。当前不安排该扩展。

停止／继续是研究资源判断，不是新增精度验收线：若没有FP32 collapse，公式误差及边界后误差均低于相应非近零梯度norm的`1e-3`，而实际参数VJP也低于`1e-3`且没有由数值误差占主导的整叶，停止本分支、不扩QA；这是将不足0.1%的局部梯度扰动降优先级，**不证明Adam或九项逐位等价**。整叶近零时同时报告绝对误差，不用相对量单独判异常。若高margin collapse或近目标消减的误差穿过BF16边界且参数差量达到上述量级，才将同状态完整一步对照列入下轮预算；零散微小坐标错号不自动触发长程训练。任一候选最终仍须同预算完整九项、投影logical≤2.6，且九项等权综合分相对同协议BF16严格掉点<0.9pp；不设单项门槛。论文或局部数值改善均不能替代。

设计时决定：只安排上述一次M32补测，暂不整合softmax修复、延长恢复或扩QA审计；当时结论仅为实际精度／混合目标与CPU量级边界。未新增权重，CPU证据保留唯一report／log／exit与复算入口。

## G09初态group0 M32真实live结果

实现为 [capture_m.py](capture_m.py) 和 [m_math.py](m_math.py)，未改已有R算子／诊断或活动G09源码。CPU `--prepare-only` 已实际完成，`prepare.exit=0`：正式group0原顺序32条auxiliary train、原生四续写token、一次混合缓存与独立CPU筛查逐值exact；复用正式`batch_inputs`，真实left-pad／mask／position批次为`32×509`。CPU算子交叉检查使用缓存logp坐标，并非学生capture，其作用和结果在`cpu.json`单列。

主线程随后分配GPU7，启动前0 MiB且无计算进程。tmux `softmax_G09_M32_live_GPU7_20261029`、PID `397647` 已正常结束，`run.exit=0`，约35.84秒完成并释放GPU7。真实G09初态先经原strict native loader，再切换到正式全部252投影／1512 FP32 decoder master、BF16计算及BF16 head、原full-model与decoder两级非重入checkpoint；实际M32前向**不使用native整权缓存**。没有加载QA教师或生成轨迹，没有创建optimizer。

调用者激活`bitvae`，在原snapshot cwd运行的入口为：

```bash
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/softmax_numerics_20261029/capture_m.py --prepare-only
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -B -u /home/star/VAELLM/experiments/softmax_numerics_20261029/capture_m.py --gpu 7
```

唯一输出目录为`/data/star/VAELLM/softmax_numerics_20261029/M_group0_live/`。`report.json`记录32行原始数值、源UUID／正式prepared绑定、对应代码SHA、精度／资源及无更新证据；`logits.pt`保留真实BF16四项logits、原FP32混合logp、两种梯度及对应BF16边界值，`inputs.pt`保留原M32输入供复算。`prepare.log/prepare.exit`与`run.log/run.exit`分别记录CPU及GPU终态。

| 观察量 | 实测 |
| --- | ---: |
| 当前原FP32加权M loss | 0.14658260345458984 |
| 保留同一cached目标的FP64加权loss | 0.14658261362416478 |
| FP64 logit梯度参照L2 | 0.031862987376705 |
| 当前FP32公式梯度误差L2／最大绝对误差 | 4.6962541991e-9／1.7406946370e-9 |
| 公式误差／参照范数 | 1.4738901107e-7 |
| 各文档最大全梯度相对误差 | 3.2398696081e-5 |
| FP64解析式对Eq.5 autograd最大绝对差 | 5.2041704279e-18 |
| 实际学生argmax对其余LSE的margin范围 | 0.0466796290～6.3186433933 |
| 学生margin>16.6／梯度零项／非零错号 | 0／0／0 |
| 原缓存FP64概率总质量最大偏差 | 6.8821627730e-8 |
| 两套梯度跨同一BF16 logits边界后的不同元素 | **0／128** |
| 同BF16边界梯度差L2／最大绝对差 | **0／0** |
| 同BF16边界梯度参照L2 | 0.031886913245425 |

FP64参照严格保留`a=sum exp(cached_logp)`，没有重新归一化混合目标、重混gold或套用raw teacher Eq.9。真实学生margin与教师／目标margin分别记录，未用CPU目标margin代替学生。按照预先固定方案，**BF16边界完全一致时停止**，因此本轮没有执行参数VJP；条件VJP分支也不声称已经GPU验证。没有参数Adam或完整联合M＋QA对照数据，不从此结果外推参数更新或九项等价。

状态核查：1512 live master前后值摘要均为`8d422145fbe5813a681fc0fc10c830903a2f64cbb96b80209439b475f44159bb`，Parameter身份／版本不变；原1512 BF16 decoder与1911冻结字段恢复exact，没有累计`.grad`或优化步骤。冷加载峰值allocated 72,402,078,720 B／reserved 72,917,975,040 B，live前向与算子审计峰值allocated 15,629,293,568 B；reserved保留了冷加载分配器池，不能当作live激活用量。函数退出后模型／图均销毁，PyTorch allocated为0，进程结束后GPU7为0 MiB。

本窗口未见FP32 collapse，公式误差低于既定`1e-3`研究资源阈值，且边界后差量为零，故结束此M32分支，不自动扩QA、增加样本或启动修复训练。结论限于G09该初态／group0 M32／现有混合目标与live路径；不排除其它初态、QA或优化历史条件中的不同结果，更不改变九项综合掉点<0.9pp的最终验收。未生成模型权重，约756 KiB最小输入／数值／日志证据保留供复算。

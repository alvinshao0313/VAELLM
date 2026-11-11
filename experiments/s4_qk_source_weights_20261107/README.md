# S4 QK固定64步的Web／Med源损失权重对照

11月7日后继已完成：[QK1:1与偏Web3:1各64→128完整结果](../s4_qk_source128_20261107/README.md#完整128步结果与本轮结论)显示，两臂M分别10004→9988、10019→9998，八项均值分别下降.15334054／.09536908pp；18:18全部正常退出，GPU4／5释放，新增1.546800012GPUh。128未优于自身64，不自动延长；下一批目标方案待讨论。下文64步事实保持。

11月7日核实：两条正式臂均已完成64步、独立导出／fresh及同一末点完整八项，各阶段／controller／wrapper均exit0，原进程退出，GPU4／5已释放。北京时间15:38:40／15:38:42完成；偏Web的M有小幅净增，八项均值仍低于1:1控制，最终目标未达。两个独立fresh AdamW臂都从原down/pre UUID `fa50bf40-c682-4910-8e72-d024f6b678e4` 初始化，只训QK72投影、432叶／2,479,104参数。其余180投影冻结但保留input backward；全252投影仍原生S4，投影logical／packed均4.094031292459239bpw。沿用[原QK验证路径](../s4_scope_refine_20261106/README.md)的安装、冻结、FP32状态、独立原生导出及完整评测；本包不依赖已结束alpha25／75模型或训练态。排程以[唯一当前计划](../../docs/experience/records/w25_transfer/current_plan.md)为准。

产物根：

```text
/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/qk_source_weights_20261107/trial_20261107_web3_med3
```

## 完整64步结果与本轮结论

本批只改变Web／Med源均值损失权重，模型、teacher／gold=.5/.5、样本流、更新预算和优化条件匹配。以下八项均来自各自同一个64步checkpoint；没有拼接不同更新数结果。

| 指标（%） | 本轮恢复前父模型 | 原1:1控制 | 偏Web 3:1 | 偏Med 1:3 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 71.26477710 | 71.24341262 | 71.35023501 | 71.23629113 |
| ARC-Challenge | 54.94880546 | 54.77815700 | 55.11945392 | 54.77815700 |
| ARC-Easy | 78.70370370 | 78.87205387 | 78.95622896 | 79.04040404 |
| BoolQ | 86.33027523 | 86.33027523 | 86.20795107 | 86.26911315 |
| OpenBookQA | 41.00000000 | 42.60000000 | 42.20000000 | 42.00000000 |
| PIQA | 77.80195865 | 78.01958651 | 78.12840044 | 77.85636561 |
| RTE | 77.97833935 | 78.70036101 | 77.97833935 | 78.70036101 |
| WinoGrande | 69.37647987 | 69.92896606 | 69.69218627 | 69.61325967 |
| 八项均值 | 69.67554242 | 70.05910154 | 69.95409938 | 69.93674395 |
| MMLU正确数／14042 | 10007 | 10004 | 10019 | 10003 |

偏Web较原1:1多15道M题（+.10682239pp），八项低.10500216pp，4项升／4项降：ARC-C、ARC-E、PIQA及M提高，BoolQ、OBQA、RTE和Wino下降。它相对原父净增12题（+.08545791pp），距M≥10111仍差92题，可保留为本次M／综合分取舍候选，尚不支持全面替换1:1控制。按各自对共同父统计，1:1修162坏165，偏Web修163坏151，净收益差主要表现为损伤计数少14、修复计数多1；这些计数不代表两臂改变的是同一组题，也不证明稳定减少遗忘。偏Med相对1:1少1道M题（−.00712149pp），八项低.12235759pp，1项升／5项降／2项平；本配方没有超过匹配控制。三点中1:1仍为最高八项。

| MMLU配对 | 修对 | 改错 | 净变化 |
| --- | ---: | ---: | ---: |
| 原父→偏Web | 163 | 151 | +12 |
| 1:1控制→偏Web | 59 | 44 | +15 |
| 原父→偏Med | 152 | 156 | -4 |
| 1:1控制→偏Med | 63 | 64 | -1 |

57学科均有完整分数与逐题修复／损伤，原始明细在各臂`result.json`的`paired_parent/paired_half.subject_diagnostics`。相对1:1，偏Web为17科净增／13科净减／27科持平，偏Med为16／18／23。此为完整测试后的诊断，未据此挑选训练样本；不能由本组微小整体变化判断医学覆盖充分或某一来源普遍更优。

固定面板每组128题，按自身0→64比较；early已训练，late为尚未消费的后半，heldout独立未参训。

| 配方 | early命中／gold NLL | late命中／gold NLL | heldout命中／gold NLL |
| --- | --- | --- | --- |
| 偏Web | 70→76；1.128825→1.014878 | 81→82；1.076468→0.997469 | 60→60；1.429420→1.305576 |
| 偏Med | 70→73；1.128825→1.008890 | 81→82；1.076468→0.980586 | 60→61；1.429420→1.275434 |

两臂各源early／late未加权KL、三面板gold NLL均下降，完整M与八项仍呈取舍。偏Med的heldout多1题同时完整M少4题，说明小面板和更低NLL不能代替完整测试。聚合objective按各配方真实权重计算，跨臂不按其绝对值排名。

| 配方／UUID | 训练内计时秒 | 正式全链秒／GPUh | 外壳退出时间（北京时间11月7日） |
| --- | ---: | ---: | --- |
| 偏Web／`93deb5f1-7847-4546-9f2c-f6dce6585029` | 2240.386462 | 2805.945388／0.779429274 | 15:38:40 |
| 偏Med／`f2935920-a5e3-4350-b4f2-fd08ecb40bcb` | 2244.197546 | 2808.179569／0.780049880 | 15:38:42 |

本批正式全链合计1.559479155GPUh，低于约2GPUh参考，每臂约46.8分钟；含编排CPU开销而非纯kernel时间，不再累加训练内计时。两臂训练峰allocated均35.39335203GiB，reserved38.30468750GiB；M／七项评测峰allocated为59.90345955／44.79796934GiB。两次准备CPU67.834322秒及代码准备368.988秒按下文原记录另计，CPU已包含在该准备窗口内，不重复加算。实际每臂64更新、Web1024／Med1024各一次、QA128各四次；未消费全部4096知识题。

终态CPU核证归本根`completion_verification.json`：两臂64份真实更新与源权重／有效梯度、完整state的masters／Adam各432且step64／cursor64、实际学习合同与data_digest一致；独立导出与fresh3927字段exact／BF16有限、冻结补集exact、投影logical／packed4.094031292459239bpw。两臂M／seven与导出为同一UUID，完整64叶24742题，其中M57科14042题；本次重新核256份唯一原始样本文件SHA及57科汇总一致性，逐题doc／prompt／target身份与修复损伤沿用已完成的`result.py`逐行核证，未宣称完成第二套独立全量重评分。原始指标在每臂`post/evaluation/{mmlu,seven}.json`，完整逐题配对及版本来源在`result.json`。

本批交付结束，无新GPU任务。两臂各保留一份64模型和完整可恢复状态用于本次结果讨论，原1:1控制／父／数据保护，本轮未清理产物。单seed、64步及八项结果不证明稳定排序或恢复上限；没有LongBench／完整九项，也未作新配方从BF16整链重放，九项>67.64714491790848与M≥10111最终目标未达。下一批先讨论偏Web是否按同配方接续128并补匹配1:1控制，或在固定预算下研究减少正确题损伤的目标函数；每批只选一个问题，公式、对照及预算另行确定，不自动延长、接LongBench或启动新实验。[相关经验](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。

## 已完成批次的设计与复现入口
| 臂 | GPU | 实际源均值KL的权重 | 每case权重（Web／Med／QA） |
| --- | --- | --- | --- |
| web3 | 4 | .375 Web＋.125 Med＋.5 QA | 3/128、1/128、1/16 |
| med3 | 5 | .125 Web＋.375 Med＋.5 QA | 1/128、3/128、1/16 |

这是损失权重，非抽样比例。两臂每步都保持原Web16＋Med16＋QA8、groups0…63；实际曝光Web1024／Med1024各一次、QA128各四遍。知识总权重.5、QA总权重.5；知识仍使用原teacher／gold=.5/.5目标，QA仍为静态gold-prefix全词表KL。数据不重采样、不重新教师推理、不改变目标混合。源级归约与对照边界复用[同父恢复经验](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。直接只读复用[原WebMed数据合同及来源](../s4_webmed_recovery_20261105/README.md#固定配方)：WebInstruct-verified revision `3e8a350b3a935d68fe70bdf379692500abc9ff51`、MedMCQA revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`均为Apache-2.0／train；原QA来源、版本、许可与train split继续以其原始合同为准。不使用测试集训练，不据测试逐题选样。

seed137、LR5e-7、betas(.9,.95)、eps1e-8、weight_decay0、foreach=False、联合clip1.5、原500步linear前64、无warmup保持。FP32 masters／grad／Adam，原生BF16计算与部署。旧知识helper的总值硬编码1:1，本包`objective.knowledge_loss`保留其同M32 forward与逐例KL计算，显式将group case weights乘入后求和；新`train.update`实际调用本包consumer，再累积原QA、仅联合clip一次，无额外除32／40或二次乘知识.5。

只读对照为原QK64 .5 UUID `40f76c27-8f58-4bf8-b448-08af2ea439c9`、M10004／14042、八项70.05910153709371；它不是新臂初始化。原父与新臂完整同版本0-shot评测：M57学科14042题B64，七项10700题B32，acc／acc_norm按原协议。每臂固定64→完整可恢复状态（cursor64／masters／Adam／RNG／names／input-target-weight-code bindings）→训练进程退出→独立native导出→fresh3927字段exact／finite／bpw→完整八项→对父及QK.5配对→退出释放GPU。

新增57学科分数、修复／损伤只作结果诊断。early为本批已训、late为groups64以后尚未训、heldout为独立train保留面板。各源未加权KL、共同gold NLL／命中可比较；每臂按自身真实源权重聚合知识目标，聚合weighted objective绝对值不能直接跨配方排名。heldout没有教师KL，不伪造目标值。请求、data_digest／data_contract、learning_contract、named state、checkpoint provenance、更新和诊断都记录真实源权重。

CPU `prepare`实际加载原4224目标／全部输入顺序与QA，检查仅源权重变化、128组源权重和曝光、scope及fresh Adam；知识consumer逐组核原1:1总值exact、groups0／63梯度exact，两新臂核显式逐源mean公式和梯度。测试输入仅用真实缓存目标组成有限CPU数值输入，不替代正式模型或训练数据。逐题reader使用真实24742控制样本自配对，覆盖57科新增诊断。复用既有QK GPU峰组2／9、冻结、保存／恢复及独立导出证据，不额外重复GPU短测。

约2GPUh参考预算、非硬中断；原同路径每臂训练加八项约47分钟，代码准备和CPU另计。按实际64终点完成，预计超参考时说明原因；不自动128、LongBench、动态alpha、新数据或下一批。每臂保留一份64模型、一份可恢复训练态及最小配置／核心日志／原始指标供批末评审；原父／QK.5及共享数据保护，本包不执行权重清理。

调用者激活bitvae，CPU及正式入口用tmux／nohup后台执行，拒绝覆盖已存在attempt：

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_qk_source_weights_20261107.prepare
bash experiments/s4_qk_source_weights_20261107/run_web3.sh
# 另一独立会话：
bash experiments/s4_qk_source_weights_20261107/run_med3.sh
```

`CPU_validation.json`与`plan.json`绑定两份请求和源码SHA；每臂`run.json`记录实际argv、PID／start_ticks及退出，`training.log`／`training/updates/`记录源权重、各源未加权KL、实际加权贡献、有效梯度和更新。正式前两步确认后停止主动轮询，后台完成约定链；启动不能称训练或完整八项已完成。实际启动标识及首两步证据见文末。

最终CPU核证34.213902858秒、exit0、CUDA未初始化；首次CPU核证33.620419248秒，统计实现整理后最终版本重核，两次CPU合计67.834322106秒，原日志同存`CPU_prepare.log`。旧1:1 objective与代表组梯度exact，两新臂梯度对源均值公式差0；总值归约最大绝对差4.768371582e-7，在rtol1e-6、atol1e-7组合容限内。Astra逐个核本包14份py／sh SHA与最终plan／requests／proof一致后准予启动，未新增GPU冒烟。

以下为历史启动记录，原进程现已退出，终态以上节为准。正式tmux为`S4_QK_web3_g4_20261107`／`S4_QK_med3_g5_20261107`；wrapper403148／403151、controller403173／403172、training403881／403874。所有实际argv及start_ticks以各臂`run.json`为准，各臂`startup_verification.json`已保存前两次真实更新、请求绑定、完整argv与进程start_ticks，现已停止主动轮询。

两臂正式前两步均消费groups0／1，LR5e-7／4.99e-7，实际Web／Med／QA总权重分别.375／.125／.5与.125／.375／.5；分源原始KL、实际加权贡献及QA均有限。72个投影梯度全部有限非零，联合clip前范数有限非零，两个更新的early／middle／late FP32 masters和BF16部署值都实际变化。两臂首步各源未加权KL exact相同，0步early／late／heldout所有逐题gold诊断及各源KL exact相同；仅加权聚合按配方不同。聚合公式在真实步骤记录中核验通过，不能把两臂的加权loss绝对值直接排名。

| 臂 | 正式step1／2秒数 | 启动状态 |
| --- | --- | --- |
| web3 | 40.297268581／42.595544395 | 前两步真实更新已核，后台继续 |
| med3 | 40.412063024／42.371640788 | 前两步真实更新已核，后台继续 |

确认推进后已停止主动轮询；后继仅为本批约定的64→独立导出→fresh→完整八项及配对汇总→退出释放GPU，启动交付当时尚未核完成64或八项；本次完整终态及质量取舍见上方结果节。

准备窗口：新包目录创建（14:44:37.622）至最终CPU plan写出（14:50:46.610）约368.988秒，包含上述两次CPU共67.834秒；此前文档／源码阅读未单独精准计时，准备时间不计GPUh。

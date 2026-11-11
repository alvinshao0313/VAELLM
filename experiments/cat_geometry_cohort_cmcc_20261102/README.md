# 同父原生 gate：向量规格与 encoder 容量两点研究

本批继续第一阶段 CAT 配方研究，不把未定稿上游停工。共同父为 fresh_common 正式 after_o `7eeb82b4-3868-4cc5-9fa3-51870ee55894`；共享控制复用原生 raw2 `f4669891-9028-4665-85a5-71c26ac45c33` 及[已完成的同阶段完整 MMLU／部署读出](../cat_hyperparam_cohort_cmcc_20261102/README.md#完整结果与取舍)，不重训或重评控制。

| 臂 | GPU | gate D/K | encoder/decoder | 每 Linear 向量 batch | 各 stage 更新 |
| --- | --- | --- | --- | --- | --- |
| control | 已有证据 | 64/64 | 128/128 | 4096 | 已完成 10K×2 |
| encoder256 | 1 | 64/64 | 256/128 | 4096 | fresh 10K×2 |
| vector128 | 2 | 128/128 | 128/128 | 2048 | fresh 10K×2 |

固定原 N native M36 pooled BSQ、S2、asymmetric/R0/layer/swish、seed31/shuffle400031、LR .003/linear/无 warmup、commitment .25/lfq2.5/entropy.01/gamma0=gamma=zeta1/inv_temperature100/new_quant。保护仍 input64/int8、双侧旋转 block32、方向／512×256 train 校准及未来 dense 补集不变。两臂只执行完整 36gate 拟合，到真实 native round_base 停止；gate 恢复0、180压缩＋72未来dense，不实施后缀或转换 compact。

encoder256 检查原 C 在 D32/K32 背景的完整收益是否能迁到 D64；vector128 补一个同主码率向量规格点。D/K 同增会改变 BSQ 维数／幅度／辅助尺度、向量池、采样映射和初始化，不能归因为纯 D。`B×D=262144` 只匹配每 Linear 每更新标量曝光，不能称等 FLOPs／等实际成本。encoder 不部署；decoder、码、保护值／scale／索引、旋转及 future dense 均按真实保存状态计费。

真实 control 元数据是非转置 `12288×4032` 未保护矩阵、parallel_parts1。原生 materialize/reshape 按每 part 全部元素重排，不要求逐行整除；D128 为每 gate 387072 向量，无权重 padding。K64／128 都按字节完整打包。旧 `checkpoint_bpw.audit_state_dict` 不接受 `round_base`，并另将保护通道数非 D 倍数列为违反政策；D128 的保护64不符合后者。本包保留真实 kind／保护／旋转，以单独 partial 账本测实际 unique tensor storage，并如实列这个稳定审计缺口。真实 native 路径能运行不等于完整后缀已有正式验收支持，后者需先解决此缺口；本批不作最终≤2.6或九项成品声明。

参考[本体超参／结构经验](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)、[stage 损失尺度](../../docs/experience/lessons/output_losses.md)和[共同前驱](../fresh_common_cmcc_20261030/README.md)。旧 LR／commitment／20K 三点均负，不扩原网格。原 C 局部与完整排序已有反例，因此两 raw 结果返回研究审定，没有自动 winner／后缀：容量迁移优先保留一个完整闭合名额，不因 raw M 微负或 NMSE 局部较差机械取消；后续必须另安排同真实父、同消费者和匹配后缀控制，不能拿旧父完整 B 与本次 partial 混排。

CPU 准备严格加载真实共同父，四份独立 formal/short fork 的全部状态和恢复历史逐项核对，代表 fork 独立进程 strict；真实 control 的既有 strict／MMLU／readout 绑定 SHA。新形状／容量各做原 native M36、正式向量 batch、S2 各2更新，观察实际有限非零梯度与参数变化，真实 raw2 导出、独立 strict、非 gate 2091字段对父 exact、真实 group8 BF16部署读出通过后，另进程从从未更新 formal fork 开始10K×2。short 不接正式，不重复 resume 或 short 完整 MMLU。

正式固定链是 fit → 独立2559字段 strict／partial账本 → 完整 MMLU57科／14042题（0-shot、BF16/FA2/FP32-RoPE、group8预热、batch64、无limit）→ 真实部署误差。误差使用真实 group8 产生的 BF16 全权重缓存及同父 dense gate；输出读出用父 activation cache 固定首8条 Alpaca train、256 token，在真实 after_o 上捕获各 gate 输入。没有 test 选样。原 stage 归一重构日志／融合前残差 RMS 与部署 weight/output MSE、NMSE 分列，低 MSE 不等于下游胜。

调用者先激活 `bitvae`，所有执行以 nohup/tmux 后台运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -B -u -m experiments.cat_geometry_cohort_cmcc_20261102.prepare
python -B -u -m experiments.cat_geometry_cohort_cmcc_20261102.run --arm encoder256
python -B -u -m experiments.cat_geometry_cohort_cmcc_20261102.run --arm vector128
```

原始配置、源 SHA、CPU核证：`result/qwen3_8b_mmlu65_20261004/cat_geometry_cohort_cmcc_20261102/prepared.json`。每臂实际命令／PID／start_ticks／资源／阶段退出与路径：`artifacts/cat_geometry_cohort_cmcc_20261102/<arm>/run.json`；真实 fit 配置与初始化摘要为 `<phase>/report.json`，strict／账本为 `<phase>/strict_reload.json`，完整评测为 `mmlu.json`／`mmlu_logs/`，部署读出为 `readout.json`。每阶段使用固定有界 timeout；拟合90分钟，strict30分钟，每次 MMLU90分钟，readout30分钟，short各组件30分钟。失败停止该臂依赖，不自动重试／占卡／调度。旧 D64 同规模拟合约11分钟仅供估时，新点不保证同速。

## 实际验证与后台运行

2026-11-02 13:03前后（北京时间）核证：原 CPU 准备真实 `cpu_verified/exit0`，耗时165.15秒；四份独立分叉全部完整状态／恢复历史相等，代表 formal 分叉独立2127字段 strict 通过。未重复准备。原准备记录的 `new_short_required=False` 为旧文案，已改为 `True`，并在 recipe 显式要求保护64、拒绝自动 round；原完成记录保存在 `prepared_initial_cpu.json`，共974项源码绑定中除这两个明确改动文件外均核一致，四份 fork 身份／实际 recipe 不变后重绑定新 SHA。

两臂真实 M36／正式向量 batch／S2各2更新均完成，四次更新均为有限非零梯度且实际参数改变；各自真实 raw2 独立2559字段 strict 与2091非gate对共同父 exact 均通过。后续真实 BF16/FA2/FP32-RoPE、group8预热均180/180成功、无failed；36gate固定 train 输入部署读出均完成。短测用于新路径正确性，不作收敛或质量比较。

| 臂 | short fit耗时／peak allocated | 正式未更新父UUID | 持久会话／controller PID | 正式fit timeout／worker PID |
| --- | --- | --- | --- | --- |
| encoder256 | 117.64秒／14,419,576,320B | `d87e899f-ee2b-47f2-849d-e6da396723bc` | `CAT_geometry_E256_GPU1_20261102_v2`／1712320 | 1717453／1717454 |
| vector128 | 123.95秒／14,424,335,872B | `37493e76-f750-46d9-8064-0dfa10b0b9a6` | `CAT_geometry_D128_GPU2_20261102_v2`／1712319 | 1717524／1717525 |

两 controller 的 `start_ticks=33297622`；各自从未更新 formal 父 fresh 启动，每stage仍为完整10K、原linear日程。13:04:38／40实核E256第一阶600/10000、D128第一阶500/10000，loss有限、约0.027秒/更新；此后停止主动监控，尚无正式MMLU／最终部署结果。实际启动为激活 `bitvae` 后执行 `python -B -u -m experiments.cat_geometry_cohort_cmcc_20261102.run --arm <arm>`，阶段命令、PID、资源与实际退出写入各 `run.json`。外层日志为各arm的 `controller_v2.log`，外层退出为 `launch_v2.exit`；阶段日志／退出为 `<stage>.log/.exit`，尚在运行者无退出码。固定后继为 strict／partial账本→完整MMLU→部署读出，不自动完整后缀。

12:58的首轮 tmux 单命令入口会话已退出，两个 `controller.log` 为空、无 `run.json`、无阶段产物或更新，退出原因无可用证据；12:59改为保留外层退出的 tmux 壳后才进入真实链。首轮空日志的状态已归并下述清理记录，不能把首轮会话创建算作实验已启动。无新增算法效果结论。

短验收尾已完成：实核formal及后继strict／MMLU／readout只读正式父／正式输出，short路径仅为JSON历史证明；533个可访问进程的cmdline／fd／maps无短权重引用，受保护认证及其它UID进程的访问边界如实记录。仅删除两臂short raw与short初始fork的4份权重、配套runtime／tokenizer副本、完全重复native日志、已归并exit0文件及空bootstrap日志，共42文件，逻辑49,403,699,964B、实际释放49,403,817,984B（46.0109GiB）。各短验核心日志／配置／metadata、全部数值和2559＋2091＋group8证明、原／新prepared及formal必需fork均已复核保留；未改活动代码／配置，未停止进程或等待正式结束。精确删除路径、大小、用途和保留核证见[本组cleanup.json](../../result/qwen3_8b_mmlu65_20261004/cat_geometry_cohort_cmcc_20261102/cleanup.json)；短验阶段exit0保留在该记录与各 `run.json`，无需已删除的独立 `.exit` 文件。原common／control raw2及其它分支未动。

## 完整阶段结果与取舍

2026-11-02 17:21重新核对实际终态：两臂fit、独立strict、完整MMLU及部署读出均exit0，外层`completed_fit_fresh_full_mmlu_readout/exit0`。encoder256于13:19:56、vector128于13:19:49结束，非仍在评测。两份正式2559字段strict及2091非gate对共同after_o exact均通过；未来72个dense投影仍保留，结论范围止于本次raw2节点。

| 同父raw2配置 | 完整MMLU（14042题） | 部署weight NMSE | 固定train输入output NMSE | 正式fit耗时 |
| --- | --- | --- | --- | --- |
| 共享控制D64/K64、E128/D128 | 9279／66.0803304373% | .1296141135714271 | .004733617204246657 | 复用原控制 |
| D64/K64、E256/D128 | 9302／66.2441247686% | .12914523377989018 | .004715559600623871 | 659.23秒 |
| D128/K128、E128/D128 | 8823／62.8329297821% | .173074473435163 | .006380481413049228 | 650.22秒 |

原始分数、UUID、协议及样本在各arm的`mmlu.json/mmlu_logs/`，误差在`readout.json`；控制原件为[既有control MMLU](../../artifacts/cat_hyperparam_cohort_cmcc_20261102/control/mmlu.json)和[部署读出](../../artifacts/cat_hyperparam_cohort_cmcc_20261102/control/readout.json)。三者读出使用同一after_o、8条既有Alpaca train／256 token及同一输入SHA，不能据此替代下游或跨模型收益。

E256较控制多23题（+.1637943313pp），output NMSE改善约.3815%，单seed微小差不足称稳定最优；结合旧D32背景完整九项+.580037pp的容量配方正例，Astra/ultra审定使用既定一次完整闭合名额。后继只复用本E256正式raw2，与新C0同消费者、同后缀初始化规则完成gate恢复及自身live up/down，不借旧父后缀、不叠Rice或新恢复目标；完整九项与真实位宽结果未产出前不采用为新最终基座。

D128本配置少456题（−3.2474006552pp），output NMSE恶化约34.79%，本预算下不追加完整后缀。此点同时改变D/K、BSQ尺度、向量池和采样；不证明所有长向量无效、已收敛或D的独立因果。实际全模型tensor bytes分别为E256的10,737,410,304及D128的10,739,778,816，后者多2,368,512B；两者均含72 future dense，原始账本保留在`formal/strict_reload.json`，不是最终≤2.6成品位宽。D128还有上文稳定审计政策缺口，不能凭原生运行通过跳过。

经验增量归入[本体超参与症状分流](../../docs/experience/lessons/compression_choices.md#按症状分流)。本组固定的方法是同父／单变量族／有限预算／真实部署读出／完整后继验证；需按模型重算矩阵布局、BSQ归约尺度、真实费用及live输入；D/K64与128、encoder128与256仅为本条件有限探索点，不作跨模型推荐。E256正式raw2保留供已安排的完整链；实施方已确认新链不读D128或四fork绑定。D128两份无后继生成权重、runtime及tokenizer副本共16文件已清，实际释放24,703,053,824B（约23.0065GiB），原始数据、共同父、control、E256权重均未动。全部D128核心配置／日志／指标／metadata及原始MMLU样本SHA、E256证据及权重stat均核不变，精确范围、活动引用检查及保留核证见[终态清理记录](../../result/qwen3_8b_mmlu65_20261004/cat_geometry_cohort_cmcc_20261102/cleanup_vector128_20261102.json)。与早前short清理分开计费，不重复计入。

# 0910 rank8 八卡 B16 定向探索（2026-09-30）

## 状态与目标

运行状态：已启动，运行中；八组正式任务均已通过真实推进验收。控制器PID114973，2026-09-30 19:43:35北京时间启动。结论：尚无本轮新精度结果。

用户本次明确授权GPU0–7继续端到端探索。目标仍为固定0910初始checkpoint、LoRA rank≤8、既定八任务全量0-shot均分争取69+。不改v2训练数据、原混合比例、自然任务比例、ChatML或data_seed0；不新增均衡采样。当前参考为[上一轮已核实冠军67.663915%](2026-09-24_seven_day_search.md)，距69为1.336085个百分点；该冠军尚缺相同B16条件的种子复验。

## 依据与首轮设计

参考[搜索覆盖与确认经验](../../lessons/experiment_design.md)、[KD与数据经验](../../lessons/kd_and_data.md)及[checkpoint经验](../../lessons/checkpoint_lifecycle.md)。前轮列出的mass/CE没有实际覆盖；小batch筛选排序不保证迁移；确认应针对最后产生的实际配方。旧数据短程mass/大CE效果弱，因此本轮仅以正确v2、T2、B16、完整5000步测试新条件，不宣称已有提升。

C2为现冠军配方：普通LoRA LR3e-4、rank8/alpha16/dropout.05、norm LR3e-5、head LR1e-4、prompt .6、T2、K100、clip3、partial KL。E87为后期B4强配置E087迁移至B16，相对C2改dropout .2、prompt2、norm LR1e-4；这是组合迁移，不能把所有收益归因于单个改动。

统一单卡microbatch4×accum4，有效B16，seq1024，5000步cosine、warmup20、weight_decay.001、每200步全量八任务评测。decoder冻结，普通LoRA+norm+head可训练；残差none、两个hidden损失为0。除F01外model seed0，data_seed全部0。

| ID | GPU首轮计划 | 改动与用途 |
|---|---:|---|
| F01_C2_seed1 | 0 | C2仅改seed1，确认真实B16配方 |
| F02_E87_b16 | 1 | E087迁移B16 |
| F03_C2_mass | 2 | C2仅改kl_top_mass |
| F04_C2_ce3 | 3 | C2改kd_top_partial、alpha .97 |
| F05_E87_mass | 4 | E87仅改mass，与F02对照 |
| F06_E87_ce3 | 5 | E87改3%CE，与F02对照 |
| F07_C2_prompt2 | 6 | C2仅改prompt2，拆分组合贡献 |
| F08_C2_lrhalf | 7 | C2仅改LoRA LR1.5e-4，检查B16最优LR迁移 |

以上GPU是全部空闲时的派发次序，实际以summary.stage_history为准；不停止或抢占其他用户任务。CE系数3%不是3%梯度份额，CE同样监督按prompt权重归约的混合文本，不是仅人工答案CE。所有正式任务fresh0910，无smoke权重续训。

## 有界后续与停止

复用既有Executor，不新建守护或自动重试系统。首轮8组都以5000为固定终点，不用400步判定上限。所有首轮候选具有共同3200步评估（或明确终止）后，按历史最高八任务均分与最佳相邻两次均分的平均选择最多两个seed0/B16配方，允许首轮慢任务完成期间空出的卡开始第二轮，减少无意义等待。

第二轮最多8组：各中心LoRA LR×.75/1.25、prompt中间值1.0（若已为1则1.5）、5%CE partialKD；全部单轴修改，唯纯mass中心转CE5同时改变loss结构和CE系数，应作为配方候选，不能声称只检验CE比例。已有相同配置自动跳过，不用重复候选填满名额。

所有探索完成后重新选最终两个seed0配方，各补同B16/5000调度的seed1/2，最多4组；已执行同配置种子不重复。最终确认在探索结束后选择，避免旧搜索提前锁定冠军。排序用于分配预算，主指标始终原八任务均值，不删除RTE或重加权。

最多20个新任务、总墙钟上限72小时；预计约两天完成，实际以八卡并发耗时为准，不保证用满上限。单阶段按实测与保守估算不足剩余预算则不派发；全部完成即退出，不追加占卡程序。任何候选失败则停止新派发，让已运行同伴正常结束，等待原因核查；失败不自动重试。正式推进验收后停止主动监控，远程进程独立运行。

## 入口、数据与保存

工作区：`/home/shaoyuantian/program/VAELLM-e2e-0910-20260924`，HEAD `82993cfdb589934e4dc771797343701b36890393`，已有未提交改动。训练核心本轮不改；执行器扩展GPU授权列表、读取上一轮best.json、记录当前源码hash，并将失败终态明确标为search_failed而非正常耗尽。

入口：`experiments/e2e_0910_search/run_focused_search.py`；策略`focused_strategy.py`；计划`focused_search_plan.json`。共用执行器`seven_day_runner.py`和原冻结helper。实际文件hash、计划、各任务命令/种子/状态写入输出，不伪造提交。

结果根：`/home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_focused_20260930`。`summary.json`记录进度/命令/退出状态；`best.json`指向独立保存的最佳副本；`controller.log`为控制日志；`trials/`保存有效配置、核心日志、原始评估JSON。先复制验证旧冠军作为保底，绝不改写旧根；只留全局最佳权重，已结束非优胜权重自动清理，配置与原始指标保留。

固定数据路径、191870906字节与SHA894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1已在只读preflight核实；初始ID01457bf3-ef22-49e8-847f-dc721287c2d6。八任务按原acc/acc_norm、64任务科目、24742题、0-shot、limit=None严格检查。

调用shell先激活bitvae，再后台执行：

```bash
nohup python -u experiments/e2e_0910_search/run_focused_search.py --plan experiments/e2e_0910_search/focused_search_plan.json > /home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_focused_20260930/controller.log 2>&1 </dev/null &
```

## 验证与边界

CPU损失6组FP32/BF16×mass/CE3/CE5的实际函数与独立概率公式及学生梯度对照通过，最大loss差2.38e-7，BF16 mass梯度相对L2误差4.33e-5；padding/末token梯度0，prompt有梯度，teacher detach。已有7项损失回归通过。执行器9项CPU测试通过，含真实保留冠军导入/源文件保护、8卡列表、进程组退出、记录实际hash。

新损失真实短测使用E87的mass及CE3两配方：真实0910、B4×acc4、seq1024、T2、FP32可训练参数/BF16计算、CPU teacher offload、4次更新，step2中间评估后回训、step4最终导出。仅缩短步数、warmup0和eval_limit1；只能证明流程，不提供精度结论。此前已验证的acc4 exact resume可复用，本次没有再次声称做恢复验证。

实现核查发现dense loss忽略teacher_output_chunk_tokens，并将完整teacher logits搬回设备转FP32；因此不能用配置chunk8推断显存上限，本轮以真实峰值和重复更新验收。流程短测结果、权重数值检查及清理随后归并至输出根validation.json。暂无新增精度经验。


### 本次真实验证与收尾

两条4步真实流程均正常完成、exit0，各约201.9秒；step2评估后回训、step4导出及八任务limit1评估均完成。每条651个mutable张量均有限且step2到4全部改变，253个LoRA B全非零，651份Adam状态均有限并到step4。teacher首步日志峰值分别43,994,278,912 / 45,197,636,608字节；这是该日志测点，不冒充整段训练最高显存。最终执行器＋策略CPU回归15 passed / 2.57s，正式8配置与当前源码hash再次核对通过。全部正式配置steps5000、acc4、完整评估，无smoke上限或续训路径。

验证证据归并至正式结果根validation.json。清理已结束短测的全部trainer_state/final_model、重复指标Markdown和一次性CPU测试目录，主线程本次实删分配空间11,319,828,480字节（约10.54GiB）；保留各短测有效配置、core日志、原始指标及唯一run元数据。受保护记录、0910、旧冠军、新根保底副本和八个正式进程均核实存在。


### 正式启动验收与交付

2026-09-30 19:47北京时间，GPU0–7八组均已实际推进至少10次optimizer更新，loss、学习率与梯度日志有限，完整有效参数与计划一致。不是只凭PID或排队判断成功。控制器PID114973；训练PID按GPU0–7分别为115024、115027、115028、115097、115161、115226、115296、115380。运行身份、实际目录和各组日志原文已归并validation.json。

当前没有新完整下游分数；best.json中的67.663915%是已核实复制的上一轮冠军保底，不冒充本轮新结果。训练和策略依赖自此保护不改；停止主动监控，由远程nohup任务自行完成探索与最后的种子确认。用户关机不影响远程任务。首轮历史B16耗时约13–14小时，本次八卡并发可能受共享CPU/磁盘影响；整体预计约两天，上限72小时。

# CAT批量取样与原生随机流对照

研究问题：原BF16 up三档profile在CPU逐批取样路径约.125–.130秒／更新，45分钟不足以完成整个S2×10K及读出链。优先减少逐8192行取样／拼接开销，保留当前取样序列和精度，不直接切换到另一条shuffle流。原批终态和实际成本见[三档记录](../cat_up_rate_profiles_cmcc_20261104/README.md)。

`data.py`调用真实`train_utils.cat_train_data.build_block_data_loaders`，复用它创建的batch sampler和generator，仅以`TensorDataset.__getitems__`一次`index_select`读取整个batch，配合直接返回已成批数据的collate。输出仍是原生`[x, index]`列表、CPU BF16与原索引类型；原DataLoader每次迭代的base seed、RandomSampler的epoch消费及尾批都保留。当前适配只接受原单进程、非pinned CPU路径，无共享模块或原fit源码修改。

`fit.py`提供可直接调用的隔离`fit_category`入口，以原函数真实code及独立namespace只替换私有loader，原namespace保持不变。新fit identity显式绑定原7份源码和本包`fit.py/data.py`；原formal registry、DataPlan、消费者及CPU不能升级为formal的检查全部保留。`search.py`在独立namespace中把原真实worker的一处fit import路由到这个入口，保留原bank安装、其余relative import、有限process-group超时与日志。原search源码及PUBLIC pilot／技术验证的70份source pins保持原样，原GPU证据不升级为新入口验证。

## 实际CPU证据

[cpu.json](../../result/qwen3_8b_mmlu65_20261004/cat_batch_gather_cmcc_20261104/cpu.json)及同目录核心log／exit：`bitvae`、Python3.11.16、PyTorch2.6.0+cu124，OMP／MKL各4线程，CUDA未初始化，4.282597秒／exit0。

五个case覆盖整除／尾批、None／显式shuffle seed、非连续源张量，以及真实M36／B8192／D32的BF16 batch形状。每case跨3个epoch并交错完整／部分eval，原生与新loader的每批索引、值、输出layout、全局及私有generator状态全部exact。这里验证数据组织，不把CPU tensor fixture称为8B模型实验。

| CPU数据形状 | 两轮原生取样秒数 | 两轮批量取样秒数 | 实际batch数 |
| --- | ---: | ---: | ---: |
| 16384×36×32 | .224802 | .011555 | 4 |
| 16401×36×32 | .236559 | .013221 | 6（含两次尾批） |

计时只含CPU sampler／gather；未包含CPU→GPU搬运、VAE前后向、编码或完整模型，不据此推算正式GPU加速倍数或放宽时限。全部源码SHA写入原report。

同一report另归并了独立的原生索引CPU诊断：直接读取实际DataLoader并从当前native源码提取驻留索引函数，在N/B=19/8、24/8、1548288/8192各3epoch比较，同seed顺序均不同。DataLoader的base_seed／RandomSampler空tail与eval全局RNG消费有别；两者分布同属无放回、尾批保留，不能当作同轨迹。原诊断log／一次性脚本的SHA及输出已归并，临时文件已清；这份独立诊断不计入批量gather测试的4.282597秒。

[fit_cpu.json](../../result/qwen3_8b_mmlu65_20261004/cat_batch_gather_cmcc_20261104/fit_cpu.json)是随后实际native拟合对照，3.939050秒／exit0／CUDAfalse。原入口与新入口使用同一真实tiny Qwen3父、同扩展来源身份及S2各2更新／原linear10000前缀；完整持久state、packed bank、两stage初始化／seed、真实optimizer更新记录与host RNG全部exact，冻结补集通过，缺新wrapper／helper来源会拒绝。Astra已窄审。这是实际小模型CPU调用链，不是8B形状、正式10K或GPU吞吐；临时模型／bank随临时目录退出清理。

[entry_cpu.json](../../result/qwen3_8b_mmlu65_20261004/cat_batch_gather_cmcc_20261104/entry_cpu.json)验证真实worker编译后的import及实际CLI，8.106735秒／exit0／CUDAfalse：新fit恰路由一次，原bank installer与relative package保持，共享namespace不变，旧fit身份被拒绝。没有运行GPU／模型worker或伪造授权。三份CPU检查分别覆盖数据、拟合与入口，不把CLI通过称为正式模型运行。

调用者先激活`bitvae`，后台CPU复现入口：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -B \
  -m experiments.cat_batch_gather_cmcc_20261104.validate_cpu --output NEW_CPU_REPORT.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m experiments.cat_batch_gather_cmcc_20261104.validate_fit_cpu --output NEW_FIT_CPU_REPORT.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m experiments.cat_batch_gather_cmcc_20261104.validate_entry_cpu --output NEW_ENTRY_CPU_REPORT.json
```

## 使用边界与接续

固定方法是先保留原抽样、数值和调用时序，仅优化等价的数据组织，再测实际受影响路径。按模型重算N／M／D和batch内存；B8192、M36仅为本次形状，不是通用最优。当前不改VAE结构、目标、优化器或训练预算。

新runner调用为`python -m experiments.cat_batch_gather_cmcc_20261104.search --request NEW_ACTUALLY_AUTHORIZED_REQUEST.json`；request须实际获准、绑定全部新源码，正式请求还须有本入口自己的真实模型技术证据。下一步以讨论后的明确预算验证正式M/B/D、两stage衔接与真实吞吐／峰值，各卡独立技术通过即接本卡正式拟合。用户随后明确批准的新三档均已通过真实Qwen3-8B S2各2更新及完整保存／独立fresh，state与同输入BF16 logits exact；实际worker phase分别392.567380／392.533999／411.042931秒，原模型形状／512数据及原10K日程保持，已各自接fresh BF16正式。详情见[本批实际记录](../cat_batch_profiles_cmcc_20261104/README.md#三技术终态与正式接续)。这只将新路径升级为已验证GPU技术链，三正式均已推进到第一阶1000/10000：K32／40／48分别.0372／.0406／.0442秒每步；旧K32同1000日志为.1219秒，迭代耗时比约3.28。此为实测日志窗口；随后三正式S2各10K、编码／读出／保存／独立fresh及正式结果adapter均exit0，formal worker为1152.259043／1182.915703／1238.636287秒，含技术与协调的整批实耗1.352274861GPUh。完整profile已齐，下游精度尚待单独评测；旧批未完整结束，不能给完整全链加速比。新预算与已结束旧批成本独立。

可迁移经验归[资源与数值路径](../../docs/experience/lessons/memory_precision.md#合批加速与缓存精度的数值对照)；当前任务与预算仍以[唯一计划](../../docs/experience/records/w25_transfer/current_plan.md)为准。

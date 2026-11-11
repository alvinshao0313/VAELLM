# 同父 native gate CAT 三点超参 cohort

本组按2026-11-02用户加速CAT研究的授权执行。指定共同父是 fresh_common 正式 after_o `7eeb82b4-3868-4cc5-9fa3-51870ee55894`；控制直接复用现成 B raw2 `f4669891-9028-4665-85a5-71c26ac45c33`，不重训或转 compact。全组比较原生180压缩＋72未来dense的同一 partial 节点，gate恢复0，不进行gate/up/down的500恢复或全后缀，不声明最终九项或≤2.6成品。

| 臂 | GPU | 唯一改变 | 各stage更新 |
| --- | --- | --- | --- |
| control | 2顺序补评 | 现存原生raw2 | 已完成10K×2 |
| lr_low | 2 | VAE LR .003→.0015 | 10K×2 |
| budget20k | 4 | gate各stage10K→20K | fresh20K×2 |
| commit_double | 5 | commitment .25→.5 | 10K×2 |

共同固定36gate、D64/K64/S2、encoder128/decoder128/R0 asymmetric、原M36 pooled BSQ/B4096、seed31/shuffle400031、保护/旋转/方向/校准与未来dense补集。其余lfq2.5/entropy.01/gamma0=gamma=zeta1/inv_temperature100/new_quant保持。总新增80K stage更新，只有三新点，不扩网格或自动选winner。20K是fresh线性20K时域、无warmup，既不是原10K续训，也不是只增加同一LR轨迹步数。

CPU准备严格加载真实after_o，三份独立原生fork只改实际identity中对应超参（shared LR/commit记录所有类别，但本入口实际只执行gate），核全部父状态、恢复历史及runtime数据；三份保存payload均核完整state digest，并在独立进程核代表fork的strict重载。现存控制也真实CPU strict全2559字段重载；控制及每个新raw2的CPU严格核证均逐字段检查2091个非gate字段对共同after_o的dtype/shape/value exact，消费者要求该证据。CPU构造用官方`no_init_weights()`略过随后被真实checkpoint完整覆盖的无用随机初始化；原生strict加载和完整state digest仍核真实恢复值，不改变正式启动。正式fit完整复用原native.run_cat_train入口、HF/resume重建顺序与入口一次seed31，fit前不增加采样、校准或reseed；两阶optimizer初值只读记录，不声称与未留init digest的历史控制已逐位对照。N的train_utils/litebsq/rotation/e2e_common namespace固定在canonical snapshot，既有C/D及其pins不修改。

三臂完成后，各自在新进程CPU strict核原生raw2的全部2559字段，再用同一个native loader、BF16/FA2/FP32-RoPE/group8预热、固定64、0-shot执行完整MMLU57科/14042题。GPU2在lr_low拟合结束后顺序补唯一控制MMLU及读出，不阻塞三臂拟合。每个端点部署误差使用本次group8预热实际生成的原生BF16全权重缓存（包含保护、旋转与舍入）对同父dense gate，并用parent activation cache中固定首8条Alpaca **train** 256-token输入在真实after_o模型上捕获的dense gate输入读出输出MSE/NMSE。没有test筛样；原stage归一重建日志/融合前残差RMS与部署误差分列，低MSE不等同下游胜。

参考[超参和敏感条件](../../docs/experience/lessons/compression_choices.md)、[BSQ尺度与fresh日程解释](../../docs/experience/lessons/output_losses.md)、[共同前驱实证](../fresh_common_cmcc_20261030/README.md)及原C/D native fork/消费者路径。沿用既有真实M36/D64/B4096/native导出的适用证据；只更改常规fit超参，按规则不重复short2、精确resume或完整导出审计。本次CPU核证覆盖新fork与partial消费者实际结构，正式fit失败即停止该臂依赖，不重试或降配置。

调用者先激活bitvae；全部执行通过tmux/nohup后台。CPU入口：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -B -u -m experiments.cat_hyperparam_cohort_cmcc_20261102.prepare
python -B -u -m experiments.cat_hyperparam_cohort_cmcc_20261102.run --arm lr_low
python -B -u -m experiments.cat_hyperparam_cohort_cmcc_20261102.run --arm budget20k
python -B -u -m experiments.cat_hyperparam_cohort_cmcc_20261102.run --arm commit_double
```

原始配置/代码SHA/资源/PID/start/命令/退出在 `result/qwen3_8b_mmlu65_20261004/cat_hyperparam_cohort_cmcc_20261102/prepared.json` 与 `artifacts/cat_hyperparam_cohort_cmcc_20261102/<arm>/run.json`。拟合每臂5400秒、fresh1800秒、每次完整MMLU5400秒、train读出1800秒有界超时。10K×2旧参考约11–25分钟，20K臂更长，工程预算每臂90分钟外加每次MMLU约1小时；估计不是保证。

## 完整结果与取舍

CPU实核于北京时间2026-11-02 10:13完成，`prepared.json`为`cpu_verified/exit0`，耗时181.369秒；三fork完整2127字段、13,884,036,096 tensor bytes对父digest一致，代表fork独立fresh strict通过；现成控制2559字段严格加载及2091个非gate字段对父exact通过。首次CPU加载尚未生成任何fork时，为纳入核心审查补充的实际部署缓存／非gate补集核证停止本轮临时进程；修正后从原共同父重做，未复用不完整产物。原入口已由Astra复核，无新增GPU短验需求。

正式链通过调用者bitvae中的`nohup setsid python -B -u -m experiments.cat_hyperparam_cohort_cmcc_20261102.run --arm <arm>`启动；`setsid`使会话脱离本CLI调用shell。首次普通nohup的三个Python随调用shell退出，日志为空、未建立run.json或执行拟合；没有已训练状态重启。实际driver分别为1605648／1605649／1605650，fit worker分别为1606456／1606448／1606452，全部最终退出0。

2026-11-02核实际JSON：三臂均为`completed_fit_fresh_full_mmlu_readout`，fit、独立strict、完整MMLU及train读出全部exit0。控制补评也全部exit0；每个端点180压缩＋72dense，group8实际预热180/180，MMLU57科／14042题、0-shot／batch64／无limit一致。

| 臂 | 完整MMLU正确数／14042 | MMLU % | 相对控制题数 | 部署weight NMSE | 固定train输入output NMSE | 正式fit秒 |
| --- | --- | --- | --- | --- | --- | --- |
| control | 9279 | 66.08033043725965 | 0 | .1296141135714271 | .004733617204246657 | 复用原raw2 |
| lr_low | 9244 | 65.83107819398946 | -35 | .1314309135388211 | .0048117084994171435 | 670.1385 |
| budget20k | 9273 | 66.03760148127047 | -6 | .12889295294834832 | .004703486562393107 | 1221.4772 |
| commit_double | 9199 | 65.51061102407064 | -80 | .12960382165918402 | .004730965294161163 | 666.0739 |

三个新raw2 UUID依次为`e6223128-5624-4a45-9c3c-9d749c4395f7`、`b87a9106-f8ac-40bc-9e2a-61ac431cb5ae`、`0a42a877-8127-4790-a0f9-8dc54d6ec70f`。三新臂两个stage的实际optimizer初始参数digest一致；旧控制没有该摘要，不能称与控制逐位匹配。各新端点2559字段strict及2091非gate补集exact均通过。实际argv、源SHA、完整配置／状态、UUID、阶段退出及初始化摘要分别保存在各臂[原始目录](../../artifacts/cat_hyperparam_cohort_cmcc_20261102/)的`run.json`、`formal/report.json`和`formal/strict_reload.json`；原始任务／逐题指标为`mmlu.json`及`mmlu_logs/`，真实部署读出为`readout.json`。

本批三点都不提名完整恢复后缀，三个配置结束。fresh20K让真实group8 output NMSE仅降低0.636525%，MMLU反少6题，不能据此继续无条件翻倍预算；commitment翻倍的局部误差几乎持平，MMLU仍少80题。该结论只针对本Qwen父源、36gate D64/K64/S2、既定数据／seed与三个点，不表示全部CAT超参不敏感，也不证明gate恢复后或最终九项排序。这里没有全压位宽与完整九项成品结论。可复用经验已归入[压缩超参](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)和[预算／部署误差边界](../../docs/experience/lessons/output_losses.md#覆盖率后续节点与延长vae预算)。

共同after_o及原raw2控制保留给既定后续；三臂无后继的独立fork、生成raw2及其附带运行杂项已按持续授权清理，六目录共66文件／74,106,867,764字节（69.017GiB），精确路径与核证见[清理记录](../../result/qwen3_8b_mmlu65_20261004/cat_hyperparam_cohort_cmcc_20261102/cleanup_20261102.json)。配置、源SHA、核心fit日志、原始评测与部署读出均保留。

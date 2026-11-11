# S4 attention64→128 精确接续

本包执行用户11月6日批准的接续臂，仅使用GPU7。从原attention64完整训练状态继续groups64…127；原64模型UUID为`40b00612-b701-45ea-bf30-e922b8c1876e`，共同down/pre父为`fa50bf40-c682-4910-8e72-d024f6b678e4`。训练完由独立进程导出原生BF16，strict fresh及有限前向通过后完成原协议八项；不运行LongBench或其它后继。实际状态见产物根`run.json`，启动不等于完成。

## 完整128步结果与本轮结论

北京时间11月7日00:04:43完整结束，train／export／fresh／MMLU／seven／run与外层wrapper全部exit0，原PID已退出，GPU7释放。终态UUID为`e3d6cce3-cfd9-43cb-b494-5ae3ad3d46d3`；同一模型完成57科14042题MMLU及其余七项10700题。此表中64→128是同轨迹的更新及数据曝光变化，范围比较只使用同128控制，不与QK／VO64跨预算排名。

| 指标（%） | attention64 | attention128 | 同128控制all128 |
| --- | ---: | ---: | ---: |
| MMLU | 71.24341262 | 71.22204814 | 71.45705740 |
| ARC-C | 54.77815700 | 55.11945392 | 54.52218430 |
| ARC-E | 79.33501684 | 79.29292929 | 79.50336700 |
| BoolQ | 86.54434251 | 86.60550459 | 86.36085627 |
| OpenBookQA | 41.80000000 | 41.80000000 | 42.00000000 |
| PIQA | 78.12840044 | 77.74755169 | 77.69314472 |
| RTE | 77.97833935 | 77.61732852 | 77.97833935 |
| Winogrande | 69.85003946 | 70.16574586 | 69.69218627 |
| 八项均值 | 69.95721353 | 69.94632025 | 69.90089191 |
| MMLU正确数／14042 | 10004 | 10001 | 10034 |

自身64→128八项−.0108932748pp、M−3题：M修119／坏122；相对原父修240／坏246、净−6。同128对all八项+.0454283372pp但M少33题（修152／坏185），八项为4升4降取舍；不能以均值微高掩盖M代价。原all128对父M净+27，此attention128对父净−6，此前attention64的综合／成本优势未证明长程知识恢复更好。

父、attention64、attention128、all128共256个原始样本文件SHA及每份64叶／24742条完整性已核；逐题doc_id及doc／prompt／target hash、版本、提示、0-shot、B64／B32、limit及acc／acc_norm均匹配，逐题重算8项与result一致。证据为本根`result.json`、`post/evaluation/`及raw samples；对照原件见[原64批](../s4_recovery_screen_20261105/README.md)和[旧WebMed128](../s4_webmed_recovery_20261105/README.md)。

| 固定128题面板 | 自身64→128正确数 | 修对／改错 | gold NLL 64→128 | 128时曝光 |
| --- | --- | --- | --- | --- |
| early | 82→80 | 3／5 | .920822→.905021 | 128／128均已训练 |
| late | 82→82 | 3／3 | .939973→.860531 | 128／128均已训练 |
| heldout | 60→62 | 3／1 | 1.208470→1.176394 | 0／128，独立未参训 |

三个面板gold NLL、early／late混合目标loss均下降，完整M及八项却未净改善；不能以小heldout多2题宣称泛化获胜。新曝光标签已按实际case身份核实，原64的late仍是未训练，128才变为已训练。新增Web1024＋Med1024各一次、QA128各4遍，共1,531,547输入tokens／5472答案位置；累计知识4096题各一次、QA128各8遍，不能将更新与独立知识曝光的共同变化解释为单独时长效应。

| 本次追加64＋完整八项成本 | 实测 |
| --- | ---: |
| 训练进程内部秒数／GPUh | 3515.545609／.976540447 |
| 实际64更新总秒数 | 3418.403895 |
| 训练峰allocated GiB | 27.401010 |
| export／fresh阶段墙钟包络秒数 | 65.104002／39.140001 |
| MMLU／七项阶段墙钟包络秒数 | 257.324006／190.524005 |
| MMLU／七项峰allocated GiB | 59.903460／44.797969 |
| 本次正式整链秒数／GPUh | 4079.126591／1.133090720 |
| 原attention64＋本次累计正式GPUh | 2.526493756 |

分项墙钟包络由stage日志birth至exit文件mtime重建，含进程启停，不与训练内部计时混称；全链以`run.elapsed_seconds`为准，不重复相加。旧all128费用包含其原中评与LongBench，不能直接拿本次追加64＋八项的1.13309GPUh宣称全128成本比。全批QK64／VO64／本次attention三正式共2.608004642GPUh，加本次范围短验.150996706合2.759001348GPUh，低于约5–7GPUh参考；代码／CPU准备及旧重复验证不计入。

原1150个pins、新11个源码pins、原请求及接续绑定均核一致；FP32 masters／完整Adam／RNG／游标及冻结补集均精确恢复，最终864个master共4,958,208参数，Adam step128、cursor128、next_group=None。独立导出与fresh3927字段精确／有限前向通过，全252投影账本logical=packed4.094031292459239bpw保持。原native模型6,046,045,569B及训练态60,852,722B在评审后已无后继用途，于11月7日连同runtime载荷清理；原attention64载荷也已结束用途并清理，原父／数据／QK控制保护。四结束臂合计12文件释放22.8620224GiB，精确清单与原状态schema／拓扑／SHA、全部评测／源码／元数据证据见范围批产物根completion_verification.json的terminal_cleanup_20261107。目录内meta/config不代表模型仍可加载；原attention64 state已删除，本文历史接续入口不能原样重放，旧checks／bindings未绕过或改写。

本次接续没有改善自身M或八项，不自动延长256、不补LongBench或改数据。M距最终10111门槛仍差110题，八项不是九项达标。下一批建议与QK／VO一并讨论，见[研究正文](../s4_recovery_screen_20261105/README.md#后续候选与当前交付边界)；本臂无后继；用户现已批准独立的QK alpha比较，状态见当前计划，完整算法最终仍须BF16整链。

## 隔离边界与复现

只新增本包，不改[原64批次](../s4_recovery_screen_20261105/README.md)的源码、请求、source pins、验证证据或产物。原request所有字段保留不变，新增`continuation`记录新合同、新包哈希、旧request／训练态／结果绑定。直接复用原`train.build/update`及`state.restore`：恢复864个FP32 masters、完整AdamW step64、Python／NumPy／CPU／单卡CUDA RNG及next_cursor64，不从BF16成品重新训练。

128保存沿用`S4_scope_named_training_state_v1`及原identity，并额外保存`continuation`元数据。原`source_sha256`不改写成新源码。`saved_execution_policy`明确区分原first_batch_stop64及本次stop128，附接续合同。独立导出通过本包`state.restore`调用原严格恢复算子，检查masters／Adam／RNG／冻结补集精确一致；本包恢复消费者同时支持已绑定的原64与本次128状态。

学习算法、原data_digest／教师／QA／seed137／alpha.5／域各.5保持；AdamW LR5e-7、betas(.9,.95)、eps1e-8、wd0、foreach=False、clip1.5，继续原500步线性日程。step65 LR4.36e-7、step128 LR3.73e-7。新增Web1024＋Med1024各一次、QA128各4遍；累计Web2048＋Med2048各一次、QA128各8遍。全252投影均为S4，只有144个attention投影／864叶更新，投影logical／packed均4.094031292459239bpw，含decoder／保护／旋转。

CPU准备检查真实64载荷、原identity、Adam64及实际数据曝光，并检查64时early已训练／late未训练、128时二者均训练、heldout始终未参训。新面板用`trained_by_cursor`和实际case identity交集记录，保留旧64报告语义，不把旧`training_by64`字段换名解释成128。

复用旧GPU2+1的真实形状、梯度、冻结补集及恢复验证，不重复整套GPU短验。正式step65／66继续使用旧真实update并记录早／中／晚FP32和BF16位移，补新增接续入口的实际推进证据。最终独立导出、fresh、完整八项属于本臂必需后继。

调用者先激活`bitvae`，CPU准备和正式入口分别以tmux/nohup执行。脚本不激活环境：

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_attention128_20261106.prepare
bash experiments/s4_attention128_20261106/run.sh
```

唯一产物根为：

```text
/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/attention128_20261106/trial_20261106_attention64_to128
```

准备与训练均拒绝覆盖已有尝试，不自动重试。controller串行等待真实训练退出后执行独立export／fresh／MMLU／seven；同一UUID的57科14042题MMLU用B64，其余七项10700题用B32。保留128训练态及一份native模型供评审。只与旧all128作同128配置控制，并单独报告本臂64→128变化，不与QK／VO64混排。

经验依据为[checkpoint生命周期](../../docs/experience/lessons/checkpoint_lifecycle.md)、[同父恢复与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)及[原批次后继边界](../s4_recovery_screen_20261105/README.md#后续候选与当前交付边界)。本次完整结果已在上方核证；当前计划由主线程统一维护。

## 实测启动与当前边界

11月6日22:56正式启动，tmux为`S4_attention128_g7_20261106`，wrapper PID4031866，controller4031873／start_ticks71438487，training worker4032197／start_ticks71439189。原入口及子命令在`run.json`，总日志为产物父目录`attention128_driver.log`，训练日志为本根`training.log`。CPU准备30.9884秒通过，1150个旧pins与11个新入口pins均按原绑定核对；10个Python模块语法／LF和本README相对路径检查通过。Astra已只读审查学习／恢复／新绑定及实际65／66更新，无阻断；不追加GPU短验。

`training/resume_verified.json`记录原64的FP32 masters、完整Adam、RNG、游标及冻结补集精确恢复。第65／66步分别实际消费group64／65，LR4.36e-7／4.35e-7，耗时59.79494／64.07157秒；两步的144投影梯度均有限非零，原更新算子逐叶断言Adam步号65／66。早／中／晚层两步合计BF16变化分别67106／64343／68562个，FP32各段亦真实改变；两步峰allocated27,661,346,304 B。原始逐步记录与启动汇总为`training/updates/step_000065.json`、`step_000066.json`及`startup_verification.json`。

确认连续真实更新后已停止主动监控，未创建监控自动化。本段为历史启动记录；后继128、冻结补集终检、独立导出／fresh及完整八项现已全部完成，当前终态见上方结果章节。`result.json`中的`contract`保留原64历史身份，实际本次执行策略看`continuation.contract`与128保存态中的`stop_step`；不能只摘历史合同将新端点误报为64。

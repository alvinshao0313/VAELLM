# C09／G09 对齐恢复：在线 OPD32

这是[同恢复链对照](../cat_aligned_recovery_20261028/README.md)的第02阶段，独立使用各自真实完成的01 QK FineWeb2000末点。旧最佳模型的祖先链和阶段贡献以[主实验记录](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月25日当前最佳的实际祖先链与迁移边界)为准；不把G09直接与已完成全部后恢复的bestabs作等条件算法比较。

## 固定训练与来源

复用旧 `qwen_train_only_opd_hotpot_20261012` 入选配方：seed137，MMLU auxiliary_train 1024例、Qasper train32篇、HotpotQA train96篇，32组。M目标只混一次 `.5×full-F + .5×gold`；QA在每组中由**当前学生**原生greedy生成，再由全精度教师在相同完整prefix上提供全词表KL。M和QA各权重.5，QA先token均值再文档均值；无R保留项，不使用旧学生答案、旧学生logits或旧QA teacher-forcing目标。

全252个投影、1512个decoder叶采用FP32 master／BF16部署计算，AdamW(.9,.95)、eps1e-8、WD0、clip1.5、D LR1e-6，保留原500步linear调度的前32步。正式fresh从组0到31；短验独立用组2和14，覆盖最长Qasper／Hotpot输入及非空Adam往返，不继承短验状态。码、保护值／scale、norm/head等1911个字段冻结，无外置LoRA；实际decoder元素数由新源几何读取，不能硬套旧源或把G的gate34改成32。

full-F教师独立读取旧载体并将全部252个投影恢复为同一原HF权重。复用缓存前核对HF／tokenizer／输入和非投影骨干身份；旧载体不是新学生、也不是R教师。实际学生保留full-RHT snapshot、原生v6、FA2／FP32 RoPE／group8路径，按本目标logical≤2.6和rank0审计。

## 实现与执行顺序

- `audit01.py`：01确实完成2000步后，核432个FP32末点转BF16、2991冻结字段／504码库、3423字段独立strict重载及真实位宽。它会实际构造原生模型，执行时须有可用GPU上下文；没有前向训练。
- `prepare.py`：只绑定已通过上述核证的新01末点；未就绪立即拒绝。`--materials-only`仅核来源独立的数据与教师材料，不能替代新学生绑定。
- `run.py`、`fit.py`、`rollout.py`：真实独立short2／fresh formal32，检查参数与缓存切换、冻结状态和原生端点。正式入口要求同一prepared的short2完整exit0。
- `export.py`：把实际formal32的1512个BF16 decoder装入对应新01，保存新UUID；核原生3423字段、1911冻结、504码库、几何与位宽，并输出 `02_online_opd32/native_audit.json`。

调用者先激活 `bitvae`，工作目录为现有 `result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01`。对C或G先设置 `VAELLM_ALIGNED_PROFILE`，在该源训练完成且所用GPU已释放后执行：

```bash
VAELLM_ALIGNED_PROFILE=G CUDA_VISIBLE_DEVICES=0 python -B /home/star/VAELLM/experiments/cat_aligned_opd_20261028/audit01.py
VAELLM_ALIGNED_PROFILE=G CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_opd_20261028/prepare.py
VAELLM_ALIGNED_PROFILE=G CUDA_VISIBLE_DEVICES=0 python -B -u /home/star/VAELLM/experiments/cat_aligned_opd_20261028/run.py \
  --prepared /data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/prepared.json \
  --phase short2 --gpu 0 --timeout-seconds 7200
```

短验通过后，G用 `run_G.sh`／GPU0，C用 `run_C.sh`／GPU1。正式32步完成后，`export.py --report <02目录>/formal32/report.json --output-dir <02目录>/native_final/final_model`执行独立原生导出。每次GPU执行均用tmux及独立日志，保留真实 `run.exit`；这里没有自动等前驱、轮询、重试或下一阶段启动器。以上为真实入口说明，尚未执行的部分不称已通过。

## 10月28日 CPU证据与未完成项

来源独立材料已在真实bitvae入口完成exit0，证据为 `/data/star/VAELLM/cat_aligned_recovery_20261028/opd_cpu_materials.json` 与 `opd_cpu_materials.log`：1152例实际train来源、M一次／QA两次、M混合只一次、128个QA生成配置及384个EOS／长度／完整prefix重放检查通过；实际导入、AST／shell语法、未就绪源拒绝和4032→252旋转因子通过，未初始化CUDA、无FW／BW。原C09／G09的几何和147个非投影张量仅用于可行性核查，不冒充新的01末点。

新代码的关键路径已只读审查，无阻断发现。材料准备当时，**01仍在训练；实际01末态审计、新学生prepared、GPU short2、formal32和独立导出均未运行**。以下真实结果替代该准备状态；CPU材料和代码审查本身不能证明GPU结果或九项精度提升。后续03分支只有从各自真实02末点重新绑定后才可执行。

## 10月28日：short2完成与fresh正式32启动

两份01均已2000更新／exit0并通过独立`aligned_QK2000_terminal_native_v1`核证，实际UUID／冻结范围和账本归[01终态](../cat_aligned_recovery_20261028/README.md#10月28日01完成核证02正式恢复已启动)。新学生prepared分别绑定各自真实01，短验独立执行原group2→14；两次真实更新、保存／读取／安装BF16端点及恢复全部1512源decoder和1911冻结字段均通过。

| profile／短验原报告 | 实耗秒 | FP32可训练叶／元素 | student FW/BW／teacher QA FW | BF16改变元素 |
| --- | ---: | ---: | ---: | ---: |
| [C short2](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/short2/report.json) | 395.2830814879999 | 1512／4743936 | 18／16 | 356724 |
| [G short2](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/short2/report.json) | 400.9277953749988 | 1512／4762368 | 18／16 | 375131 |

两份`short2/run.exit`均0，所有1512叶实际非零FP32更新；第二次原生OPD前后1512叶非空Adam、FP32 master／Parameter身份和值往返exact，1911冻结字段前后SHA相同、教师不变、全部源恢复exact。G的gate34使decoder元素多于C，未硬套旧源大小。这里证明真实训练与状态切换可用，不证明32步质量、收敛或完整九项改善。

正式均从对应01 fresh重载、group0..31和新Adam开始，没有继承短验patch或状态；随后只导出本臂真实formal32端点。唯一[C formal_launch](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/formal_launch.json)／[G formal_launch](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/formal_launch.json)绑定短报告SHA、prepared／源码、完整验收checks、精确tmux命令、日志与退出位置。C prepared SHA为`33bfa1efeb7481ce33bfaffe1210c438ddb75c4be3548ade1624b556593cba51`，G为`5411ec450a088b6ae52239e655320d8daf9ab899d7810878c7340ad8e03cb8c1`。

| profile | GPU | tmux | 正式worker PID／start_ticks | 21:48已确认状态 |
| --- | --- | --- | --- | --- |
| C | 1 | `cat_aligned_C_OPD32_GPU1_20261028` | 322803／3468699 | 实际3／32连续更新 |
| G | 0 | `cat_aligned_G_OPD32_GPU0_20261028` | 322798／3468684 | 同上 |

该进度来自root已写入`startup_validation`的一次有限核查；本次未读取新训练轨迹或主动监控GPU。两正式链保留后台执行，正式32和独立导出尚无本次核实的完成结论。01完整八项与待完成LongBench归[独立评测正文](../cat_aligned_evaluation_20261028/README.md#10月28日01完整八项初报九项尚待完整longbench)，不据八项改变既定链。

## 10月28日：仅两份已用完短patch收尾

用途核查确认正式`run.py:51–56`仅读取short报告／exit／prepared SHA，重新载入真实01；`export.py:21–36`强制消费`formal32/decoder_endpoint.pt`，不读取短patch。aligned阶段入口／计划无短patch消费者，prepared未绑定短patch；一次同用户`/proc`核查无short2进程、候选文件参数引用或打开FD，正式worker身份与既有PID／start_ticks匹配，未轮询训练进度。短验没有续训／复评／交付用途，因此仅删除以下两文件。

| 已删除精确路径 | 文件bytes | 实际allocated释放bytes |
| --- | ---: | ---: |
| `/data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/short2/decoder_endpoint.pt` | 10042926 | 10043392 |
| `/data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/short2/decoder_endpoint.pt` | 10079790 | 10080256 |

删除前已在各`formal_launch.json.short_patch_cleanup`记精确路径／bytes／allocated／SHA／inode、原因、代码依赖及活动进程检查；删除后确认两文件消失、protected小文件SHA和01 `checkpoint-2000`／原生末态保持。C短patch SHA为`716fa95853618d397ded458fea99825e49fbeec7bfdc068ccb9b916768fc1976`，G为`25e8395bc0d77bd05b909c39f93043b7aa84cad3a7f780157e954433a48473ce`。合计文件20122716B、实际allocated释放20123648B，不计共享磁盘变化。

短报告及其原endpoint哈希／路径作为历史证据保持，当前短patch已不再可加载；报告本身未修改，formal_launch所绑两份short报告SHA继续匹配。prepared、short exit、01 checkpoint-2000／native_audit、父模型和全部正式输入保留；未删除其它产物、修改正式代码或新增算法经验。

## 10月29日：正式32与独立原生导出完成

两臂已自然完成实际group0..31共32次更新，并各自完成独立原生导出；本节替代此前“正式32及独立导出尚未核实”的状态。`formal32/run.exit`和`formal_pipeline.exit`两层均0，正式报告状态均为`aligned_nativegreedy_OPD_formal32_complete`，没有续用short2状态或自动重试。

| profile／原正式报告 | 正式训练秒 | 实际02成品UUID | 原生审计 |
| --- | ---: | --- | --- |
| [C formal32](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/formal32/report.json) | 4548.399762087996 | `564dc617-b16e-47bf-a5b6-2945c52ed23e` | [C verified](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/native_audit.json) |
| [G formal32](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/formal32/report.json) | 4621.758292819002 | `38a7461e-a2f1-4586-80e2-b99883ef2628` | [G verified](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/native_audit.json) |

成品均在各自`02_online_opd32/native_final/final_model`，核心日志为各自`formal_pipeline.log`。两份`aligned_online_OPD_native_audit_v1`均verified：序列化及fresh strict各3423字段exact、1911冻结字段和504主码bank保持，实际几何不变，源仍分别为本profile真实01。实际logical／packed与01一致：C为2.497367527173913／2.5107421875，G为2.5295091711956523／2.6391813858695654，rank0；G的packed不称≤2.6。

本次CPU只读核实真实32组、两层退出、report SHA、成品metadata SHA／UUID与权重size／mtime／inode，全部与审计绑定一致；没有重复GPU验证。两份02及其真实01父来源保留，已满足03绑定的父模型条件。03仍需各自由此02生成新的R896，再CPU准备、必要GPU接线短验，不能把旧02／旧R或来源独立CPU材料当作新学生证据。当前没有02自身完整九项，训练完成不等于质量改善；验收目标以[10月29日当前目标](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月29日当前目标与完成条件)为准，历史报告不回写新阈值。

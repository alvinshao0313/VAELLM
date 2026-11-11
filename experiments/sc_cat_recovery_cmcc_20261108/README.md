# 新版 SC-CAT 逐类恢复：独立实现

唯一批次记录：[Qwen3-8B / 2026-11-08_172625_s31](../../docs/experience/records/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31_results.md)。用户11月9日19:31要求停止当前探索，控制器和GPU4–7任务已退出，四卡释放。K已完成并核验，o搜索未完成选优；累计39.58900760359234GPUh，其中第二批13.751448631220395。`search.json`状态为`stopped_by_user_for_protocol_revision`，终止证据为同run的`user_stop_20261109.json`；exit1来自主动SIGTERM及依赖取消，不当作质量失败。首类真实边界、已有指标和保留状态保护，原脚本不可再次运行到已有目录。新增数据E按主协议与原知识候选同轮比较，新探索未启动。

## 入口与状态

`handoff.py`从原132份合法完整结果重建Top2；`settings.py`／`search.py`／`selection.py`实现确认的候选、逐项校准及全节点选择。`mutable.py`／`protection.py`／`optimizer.py`／`builder.py`／`trainer.py`接原生KD和完整训练态；`boundary.py`按每份实际LoRA及P设置导出。`raw.py`在BF16上安装首类合规bank，后类接口只接受本新链真实前缀。

`worker.py`及`evaluate.py`执行独立GPU拥有窗。`processes.py`由成熟旧批有限执行器作隔离适配，共用4张卡而不固定训练／读出分区；失败停、不重试、不驻留占卡。`gpu_validation.py`只做新关键路径真实验证。`first_batch.py`在显式预算及真实GPU验证后，完成首类默认比较、单赢家逐项搜索、校准、条件1000及最终边界，随后退出等待第二批预算。节点保存成功的`node_ready.json`才允许并行读出；动态报告只更新唯一结果文档的恢复区块。

原514次CPU检查及全部GPU验证（含250→500）已通过；存储接续另外完成真实375→500的500次CPU更新、全状态与数据位置检查。新首类结果已完成并核验；后六类CPU编排与范围差异检查已通过；真实GPU执行、末类双端点九项、最终位宽与BF16全链重放仍未完成。不能将独立worker、状态接口或首类控制器当成完整七类流程已经通过。

## CPU复现

由调用者先激活环境：

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
which python
python -V
```

在tmux后台、隐藏CUDA后运行。输出必须使用尚不存在的新文件，不覆盖证据：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -B -u -m experiments.sc_cat_recovery_cmcc_20261108.validate_cpu --resume --output /absolute/new/output.json
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -B -u -m experiments.sc_cat_recovery_cmcc_20261108.audit_cpu --handoff artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31/selected_settings.json --output /absolute/new/audit.json
```

真实tiny Qwen检查D/P/L全部七组合、联合q/scale、可变LoRA原生保存/导出/fresh、四组Muon及固定κ的学习率配对，并从真实250保存态续至500。CPU合成分数仅用于选择规则反例，明确不具正式资格。实际文件审计重新hash原BF16全部登记字节及k两份bank，同时检查首类状态决策和有限费用、四卡分配；不运行GPU。

## 首类GPU执行准备

具体上限为62.5GPUh，分项、任务数和估算依据见批次`budget_proposal.json`；原提案保留`confirmed=false`。用户明确回复“确认”后，同根`authorization.json`已记录实际批准；当前`authorization_import_fix.json`保留该批准并绑定修复后manifest SHA、先前费用和唯一可复用技术raw，仅k类别、GPU4–7及相同分项和计数不变。原授权和失败代码绑定保留；后续类别未授权。

确认后在已激活bitvae的调用者shell中用tmux运行`run_first.sh`，stdout/stderr与退出状态落盘至本批根；先验证后正式。脚本不存在额外授权或自动扩预算逻辑：

```bash
bash experiments/sc_cat_recovery_cmcc_20261108/run_first.sh
```

实际调用按顺序为`gpu_validation --authorization .../authorization_import_fix.json`和`first_batch --authorization .../authorization_import_fix.json --technical-receipt .../technical_import_fix/validation.json`。日志为`controller_import_fix.log`，退出状态为`controller_import_fix.exit`。验证不是完整八项质量对照；其全部实际拥有窗由正式费用表继承，各budget.json不能相加。正式每条轨迹的训练保存、各节点export/fresh、每份完整八项分别记账，主进程只收集本批子进程，不处理其它实验。

最大同时4个GPU任务，原生完整节点可在训练继续时由其它空卡读出。磁盘限制是最多3份导出/评测中的大副本；同raw完成评测的较差副本可清理，完整轨迹结束后只保留其最佳master及完整配置、原指标和元数据。250短筛的待晋级状态不适用该清理；本首类控制器全部500，后续短筛使用独立恢复接口。已完成技术载荷不进入正式训练，并在验证后按明确用途清理。原bank、原始BF16、数据、旧实验产物均不删除。

## 复用与第二批

首类复用要求原父、模型/数据、拟合代码、消费者、初始化与resolved配方完全适用；运行时还核bank字节、实际目标权重和非目标补集。第二名旧raw缺失仅重建raw，不重拟合。后类必须从本run的已选边界调用真实source9 fitter；`sc_cat_run_id`和completed-prefix校验拒绝旧链父，不能拼旧后缀。

首类最终写入`formal_kappa_fix/first_boundary.json`、`first_selected_recipe.json`及根`search.json`，保留独立短筛状态、全模型已用1000类别和累计费用，状态为`first_class_complete_awaiting_second_budget`。第二批从该实际来源继续，同一run和结果文档，不重置配额或费用；后六类编排和CPU检查已完成，真实raw出现后执行必要GPU差异验证。最终还需末类双九项、完整账本和原BF16固定配方重放，不以首类完成结项。

代码来源：本包的builder/mutable/boundary及有限拥有窗由成熟原生适配器隔离改造；KD/data/native v6、原evaluator、Muon方向及旧exact-resume观察器只读复用。manifest绑定实际依赖和本包SHA；没有修改依赖、安装库、创建分支或提交。

## 人工核查后的存储接续

11月8日存储接续使用`run_continue.sh`，对应`authorization_storage_fix.json`、`manifest_storage_fix.json`及冻结的`continuation_storage_fix.json`。原`run_first.sh`保留初始复现来源，不可在既有目录再次启动。旧正式费用表只读，`formal_storage_fix/budget.json`继承其所有实际窗与预约计数；失败窗原样保留，新窗另计，不能相加两个账本。原GPU证明按未变的训练数学复用，375入口沿相同原生恢复机制，实际运行仍核对masters、optimizer、scheduler、RNG及后续输入哈希。

`continuation.py`限定人工核查的两条默认轨迹，不形成自动重试系统。缓存复用核原request与结果身份；先保存首个失败，再取消本批其他worker。存储迁移已核全文件checksum，源/data1路径符号链接到/data2，checkpoint和数据身份不变；同批未完成结果不包装为完成。当前日志`controller_storage_fix.log`，正常退出/失败写`controller_storage_fix.exit`。

## 固定κ检查与已完成搜索状态接续

当前授权`authorization_kappa_fix.json`绑定`manifest_kappa_final.json`及`continuation_kappa_fix.json`。`hpo_recovery.py`恢复真实完整轨迹、原最高部署、原κ和已冻结pending轮；`first_batch.py`跳过已完成轮次，保留校准状态，只重开明确列出的两条失败训练。已结算失败窗不重置为预约，新的诊断拥有窗仅计一次，所有候选／费用上限不变。

参数SHA、实际分组、raw／数据／种子与非倍率设置保持精确身份；梯度SHA记录而不要求逐字相等。更新候选范数及比值在当前同源实测支持的0.2%界内时保留原κ，检查结果随优化器保存；超界立即停止。CPU及真实P倍率.5、LoRA LR半各两步GPU证据见本批`kappa_repair_20261109`、`technical_kappa_fix`。这些技术状态不进入正式训练；首类最终核验已完成，原最高节点仍为入选点；这不构成整链验收。

## 剩余全部类别与最终验收

用户11月9日要求按规范做完全部类别，执行范围已确认；`remaining.py`从首类真实边界接o→gate→q→v→down→up，`terminal.py`处理末类两端点完整九项和固定recipe的原BF16独立重放。新CPU检查`validate_remaining_cpu.py`已完成129次真实更新及范围／末类导出和fresh，250 exact-resume沿未变训练组件复用首类CPU／GPU证明；两种真实范围与末类无L的必要GPU验证随各自新raw执行，技术状态不成为正式初态。

第二批`budget_proposal_second.json`的新增硬上限689GPUh，估计220–360；包括所有规定拟合、训练／续训、导出、完整八项、末类LB和独立重放，首类已花25.8375589724GPUh另外继承。用户已确认此新增上限；`run_remaining.sh`使用新的`authorization_second.json`及`manifest_second_authorized.json`，原提案保留草案记录。主日志为同run的`controller_remaining.log`，正常或异常退出均写`controller_remaining.exit`，进度／费用持续写入`search.json`及`remaining/budget.json`。无需重跑离线或首类搜索；全部来源仍在同run记录。CPU证据、资源快照及分项估算见唯一结果文档的“剩余全部类别与验收执行准备”。

重放始终保留原schedule_max，在每类冻结selected_step停止；不重新搜参或重选末类端点。完成训练／评测只写已执行状态，数值复现需按实际差异另核，不能把exit0或位宽通过当作全目标通过。

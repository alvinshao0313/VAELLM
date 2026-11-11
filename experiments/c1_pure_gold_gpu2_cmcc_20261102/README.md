# C1 pure_gold 的 GPU2 空档接续

本次直接复用[冻结恢复包](../c1_m_target_muon_cmcc_20261102/README.md)的原 `sequence --arm pure_gold --gpu 2`，不新增 controller，不修改仍被 GPU3／4／5 正式九项依赖的源码、prepared 或 pins。完整安排统一见[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)。本页仅保存本次资源连接、运行身份与证据。

同一完整 Rice C1 raw `744e3b98-8a09-4a1d-8b30-013db5d73d44`，指定控制为已完成 C1 Muon32 的完整九项 `64.89598790075989`。唯一目标改动是 M 条件四选改为纯 gold CE，M／QA 域系数仍 `.5/.5`；QA 保持当前学生原生 greedy OPD 和原始 full-HF 全词表目标。all252 D、R0、seed137、原组0–31、M1024一次／Q32和H96两次、Muon NS5／.95 Nesterov、aux Adam、clip1.5、base `1e-6`、整体 LR `1×` 与原500线性日程前32保持；κ按该臂首个真实联合梯度重算。

复用已完成的 pure_gold 固定首 M32 真实 backward、Rice 两更新／保存／fresh 生命周期及修复 revision 的 CPU 证据，不重复 cache、probe 或 GPU short。已有[数据与目标经验](../../docs/experience/lessons/kd_and_data.md)要求原目标不重复混合、按真实轨迹报告监督；[生命周期经验](../../docs/experience/lessons/checkpoint_lifecycle.md)要求原父 fresh 与独立 strict。这是目标配方比较；κ及后续 QA 轨迹可随目标变化，不解释为纯梯度几何机制证据。本臂现已完成完整九项，取舍归[共同对照结果](../c1_m_target_muon_cmcc_20261102/README.md#完整结果成本与本批决定)。

冻结 revision 为 `revision_group_fix_20261102`。训练预算 `14400` 秒；成功导出后依序独立 fresh，再用同一 GPU2 完整八项64叶／24742样本与 LongBench21／4750样本，读取同 UUID 的真实位宽。技术失败停止后继，不自动 retry、resume、占卡或扩网格。

调用者 shell 激活 `bitvae` 并核对 `which python`、`python -V` 后，在 tmux 中执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_m_target_muon_cmcc_20261102.sequence --arm pure_gold --gpu 2
```

原入口将训练和 fresh 暴露物理 GPU2 为逻辑 `cuda:0`，完整九项也只分配 GPU2。产物不覆盖其他臂：

- 权重／报告：`/data1/cmcc/VAELLM/artifacts/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold/`。
- 日志／完整评测：[revision 下 pure_gold](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold/)。
- CPU 来源／真实 dispatch／输出未使用／GPU2 空闲核证：[pure_gold_allocation_cpu.json](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold_allocation_cpu.json)。

启动时证据：CPU 只读核证与[首更新／下一组核证](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold_startup_verification.json)均 exit0。CPU 未初始化 CUDA；原 revision prepared SHA256 为 `28d11c1eb0ca636332e603d82ecb55c463b389b85ef4da4bfd828076d8b910d3`，原 sequence SHA256 为 `879a05ec44fc471a9bb2b8639f08e97c810cca5bc19740db0d4c8de2a9bdcd92`。核证同时检查三条活动臂的 pins 未变、pure_gold 两处输出未使用、GPU2 UUID `GPU-b90a573f-fdff-5641-10e6-00361f1f7e5d` 无其它计算占用。

11月2日17:33:49（北京时间）正式 tmux `C1_pure_gold_g2_20261102` 启动，shell PID `1906573`；[controller 日志](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold_controller.log)与同名 `.exit` 分别记录输出及最终退出。正式计划和运行身份分别见本臂 `plan.json`／`run.json`；启动时身份如下（11月3日已核均退出）：

| 进程 | PID／start ticks |
| --- | --- |
| controller | `1906585`／`34941795` |
| formal worker | `1907507`／`34948079` |

首个真实 step1／group0／qa_group0 已完成 M32＋QA8 联合更新，1728 个 D 叶 FP32 梯度及优化器状态有限，clip前梯度范数 `253.3813934326172`，κ `5.563966655557068`。实际 Muon LR `5.563966655557068e-6`、aux Adam LR `1e-6`，精确沿原 schedule 的 `1×`；M／QA loss 分别 `.19549167156219482`／`.22282872581854463`。FP32 位移 L2 `.0023064149633110313`，BF16 decoder 改变 `196789` 个元素。首组更新 `253.64990586199565` 秒，训练显存峰 allocated `36568554496` B、reserved `58307117056` B。

随后step2／group1正常推进，启动时master／optimizer连续、κ保持，源码pins未变。11月3日已核本臂训练、fresh、完整九项和外层全部exit0，记录PID均已退出；技术流程已完成，质量为`not_met`。共同对照与取舍已归[冻结恢复包结果](../c1_m_target_muon_cmcc_20261102/README.md#完整结果成本与本批决定)，本页不复制全部指标。复现仍直接使用原冻结入口，不复制旧父训练或重做成熟短链。

一次性 CPU／启动核证 stdout 和退出散文件的唯一信息已归入上述两份 JSON 与本页，完成后删除4文件共2934 B；正式 controller／训练日志、计划、运行状态及源绑定保留。

完整九项64.74322034513447，相对同C1 raw +0.55305501880338pp、相对原Muon32 −0.15276755562542pp；MMLU61.35878080045577（8616/14042）、LB44.01324488099547。成品UUID`36a6c91f-b4ef-4744-9c36-c24816e51dec`的logical2.5216473392818286／packed2.5350219996079155，预算通过，BF16九项掉点3.30392457277400pp仍未达严格小于0.9pp。实际训练7995.37秒、fresh62.86秒、完整评测21075.84秒；32更新，Q/H监督tokens1327／973。3927字段strict、2199非D冻结、Rice状态及同输入BF16 logits exact均通过；full8 64叶／24742条、LB21／4750条原预测SHA和数量已核，与共同父控制评测身份一致。[验收原件](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/pure_gold/evaluation/longbench/merged/nine_task_acceptance.json)记录完整分项、同UUID、位宽与pipeline来源。

本父／32步／既定预算的pure_gold配置结束，不追加预算或混合比例网格；此结论不推广为所有纯gold监督无效。本臂完整评测用途已结束；核引用／活动FD及E1007源码pins后，已清`final_model/pytorch_model.bin`和`same_input_output.pt`两文件，释放4692561920B（allocated），保留配置/meta、报告、核心日志及全部原指标／预测。精确路径和前后核验见[共同清理receipt](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/cleanup_terminal_20261103.json)，历史run/acceptance不改；现有final_model目录只留证据，不作为可加载权重或精确续训源。旧启动和修复证据保留，未改活动源码或6／7线程。

# 原C0 up的两个独立有界点

两候选完整结束，formal／独立fresh／完整八项／同train部署读出全部exit0。Muon于15:04:05结束、D16于15:03:47结束；Astra按本批结果审定两固定点均不采用、不接500恢复／后缀／新网格。Muon局部误差微降未迁移为完整八项增益；D16少592128B但八项和误差均恶化。该取舍只限本父、本配方、单seed，不概括所有up优化器／向量长度。

| 实测项 | 原Adam控制 | Muon | D16 Adam |
| --- | ---: | ---: | ---: |
| 完整八项% | 67.6733108398 | 67.5094458717 | 66.5937095312 |
| 对控制差pp | — | −0.1638649681 | −1.0796013086 |
| weight NMSE | .132470148715 | .131302650309（−.881329%） | .136422281114（+2.983414%） |
| output NMSE | .0247768502751 | .0244926063141（−1.147216%） | .0256639264938（+3.580262%） |
| partial logical bpw | 5.860329669455 | 5.860329669455 | 5.859647667926 |
| partial packed bpw | 5.873704329781 | 5.873704329781 | 5.873022328252 |
| 实际全部tensor B | 7589608896 | 7589608896 | 7589016768 |
| 含失败的本GPU外层wall上界h | .161954901 | .541215126 | .539876361 |

| 原协议任务% | 控制 | Muon | D16 Adam |
| --- | ---: | ---: | ---: |
| MMLU | 65.8097137160 | 64.9836205669 | 64.2216208517 |
| ARC-C | 53.4129692833 | 51.2798634812 | 51.5358361775 |
| ARC-E | 79.3771043771 | 77.0622895623 | 78.0303030303 |
| BoolQ | 84.0978593272 | 84.1896024465 | 82.0489296636 |
| OBQA | 38.0000000000 | 40.4000000000 | 38.4000000000 |
| PIQA | 75.4080522307 | 75.5712731230 | 74.7551686616 |
| RTE | 79.0613718412 | 79.4223826715 | 77.6173285199 |
| WinoGrande | 66.2194159432 | 67.1665351223 | 66.1404893449 |

Muon最终raw UUID `61cb2582-a193-4a54-bfb9-59b1d2195bab`，D16为`9a2026c5-ae05-421c-86d4-d640a56cc8fb`。两者formal保持S2各10000、各自fresh原父；Muon两stage原C0初始化摘要均匹配，D16按真实新shape记录。冻结前缀／保护／旋转exact及独立fresh完整状态／同输入BF16 logits均通过。partial拓扑180 native＋36 compact＋36 future dense的费用含所有当前部署组件和future dense，不适用最终≤2.6成品判定。

实际v2 sequence耗时Muon1763.960秒、D161745.968秒；其中formal（含保存／fresh）973.183／945.008秒，完整八项530.637／534.389秒。加原两次失败及控制实际完整八项／读出，总外层wall保守计费约1.2430464 GPUh，含CPU加载／保存／wrapper等待，非纯kernel时间；授权上限仍为9.5 GPUh。无追加实验。

终态CPU核证[closure_20261104.json](../../result/qwen3_8b_mmlu65_20261004/up_vae_muon_cmcc_20261104/closure_20261104.json)在75.769秒内只读实际weights／metadata SHA和完整partial账本、两formal／fresh、三端原64叶24742条及原始样本SHA，核BF16版本和task configs；三端每叶有序`doc_id/doc_hash/prompt_hash/target_hash`完全一致，CUDA未初始化。原指标权威为控制`control/raw_full8/full8_summary.json`及两候选`runtime_repair_v2/{arm}/raw_full8/full8_summary.json`，读出为对应`runtime_repair_v2/{arm}/raw_readout.json`；不重做GPU评分。第一次临时CPU读出器误用`splitlines()`切开JSON字符串中的Unicode换行，已改逐物理行读取，原样本SHA未变、失败日志保留于`closure_reader_first.log`。

按研究取舍及实际引用核证，已清理本包两臂short／formal的4份`raw_model/pytorch_model.bin`和4份已用完`same_input_deployment.pt`，实际释放allocated **30363033600B（28.27778 GiB）**，不是共享磁盘差值。原metadata／配置、formal／short report／strict、全部原始full8 shard和24742条样本／指标、核心日志及initial fail／runtime_repair1／v2源码证据保留；原BF16／C0父／原控制及其它活动链未触及。精确路径和317份保留文件hash复核见[cleanup_20261104.json](../../result/qwen3_8b_mmlu65_20261004/up_vae_muon_cmcc_20261104/cleanup_20261104.json)。3份用完的一次性`/tmp`CPU脚本另已删除，实际诊断源码已在结果目录保留。候选模型载荷已删除，不再可直接load／续训；复现需按保留入口重训，不能把保留报告当仍有权重。

共同父为原C0 `04_gate_proj/after_recovery`（`dafee7a6-2301-47a6-9412-d6dfaa1b61ed`），原Adam raw-up控制为 `05_up_proj/round_base`（`44378f2a-a34a-457a-8384-3dab3fcef13b`）。直接读实际权重，不复制父模型、不借另一后缀。

| 点 | 唯一变量族 | 资源上限 |
| --- | --- | --- |
| control | 原D32/K32/B8192 Adam raw，只补完整八项及读出 | GPU5 ≤1.5 GPUh |
| muon | 原D32/K32/B8192的VAE optimizer换为已验证数学的BF16 Muon | GPU1 ≤4 GPUh，含新owner短验、fresh正式、完整八项和读出 |
| d16 | 原Adam，D16/K16/B16384 | GPU3 ≤4 GPUh，含新shape短验及完整阶段；与control独立并行 |

均保留M36 pooled BSQ、S2各10000、symmetric E/D128/R0、layer/swish、input64 INT8、block32、transpose=False、seed31/shuffle500031。`B×D=262144`只匹配单更新标量曝光，不称等FLOPs或纯D因果。没有Muon×D16、500恢复、down后缀或九项预算。

新包仅在独立worker中适配实际调用。Muon私有导入成熟down optimizer源码，按实际symmetric up owner覆盖布局和本包schema；NS5、BF16参数/梯度/NS、FP32矩阵momentum和原BF16辅助Adam(.9,.95)数学不变。每stage首真实梯度重新以原Adam .003实际BF16位移匹配κ，要求相对差≤1%，最多2次扩界／12次二分。原CPU κ不用于GPU。formal两个stage原flat-order初始化摘要必须匹配真实控制；D16只记录自身实际初始化。

短验为真实M36、正式B、两stage各2更新，保留linear10000前缀；两处scheduler hook仅存在于短验worker并在finally恢复。随后各自fresh原父重做完整10000×2，不续短验。保存完整partial端点并独进程fresh，持久字段及同输入BF16 logits核证；180 native＋36 compact＋36 future dense实际收费，不称全模型≤2.6成品。

完整八项沿原64叶／24742题、MMLU B64／其余B32、0-shot／无limit。三个点的读出均使用原common校准流前8×256 train块，CPU预先冻结输入SHA；捕获同after-gate的up输入，比较实际group8 BF16部署权重及输出。它是训练集局部诊断，不是留出或盲测。

调用者先激活`bitvae`，项目根目录用tmux/nohup执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_vae_muon_cmcc_20261104.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_vae_muon_cmcc_20261104.sequence --arm muon
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_vae_muon_cmcc_20261104.sequence --arm control
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_vae_muon_cmcc_20261104.sequence --arm d16
```

`bindings.py`维护实际路径；CPU生效证据为`result/.../up_vae_muon_cmcc_20261104/prepared.json`，运行报告／核心日志位于同目录各arm，模型／fit/fresh位于`artifacts/up_vae_muon_cmcc_20261104/{muon,d16}/{short,formal}`。每arm从第一阶段起使用一个绝对deadline，不因fresh或中评重置；每个实际子任务有独立session/PG，finally清理本PG并reap。失败、非有限、冻结／数据／保存消费合同异常或超时停止相关后继，无自动retry，不将技术失败记作质量负例。

CPU30.5146秒已通过原up stage0初始化、真实619776/144矩阵布局、aux原BF16 Adam两步和32字段state精度往返；D16 factory实测323712参数。首次CPU只因校验字典误带`.weight`后缀被拒，已按原ref.name修正，失败日志保留。CPU机制通过不代表真实GPU更新、阶段质量或迁移通过；真实短验／正式状态以各arm原报告为准。

两候选首次真实短验在任何VAE更新前因包装器误取`native.transformers`属性失败；实际N fitter在函数内部导入`transformers`。第一次隔离修复实际完成两stage各2更新，但top级引用指向旧`_LazyModule`，实际N使用另一同路径对象，scheduler没有命中，第二更新LR为`.0015`；因此未保存／fresh／启动正式。两次失败及首份修复源码保留于`runtime_repair/`，不将失败记作质量负例。

当前隔离入口仍为`experiments/up_runtime_repair_cmcc_20261104.py`，在上下文ENTER时从`importlib`取得当前真实模块，actual fitter调用前核module/getter未替换。`actual_N_scheduler_repaired.json`为真实N CPU tinyfit证据：S2各2更新、两个scheduler命中、LR均`[.003,.0029997]`，函数恢复且CUDA未初始化；该CPU机制诊断不是M36 GPU形状或部署证明。CPU实际执行源码先备份为`scheduler_repaired_actual_CPU_source.py`，最终prepared分别绑定该源码、负／正诊断JSON和实际新入口SHA。后续只修summary返回值消费，不改VAE算法或原8份源码。

原控制已实际完成64叶／24742题并写盘完整summary，但wrapper从不含UUID的API返回值读取`checkpoint_id`而失败。隔离入口只读原summary／两shard／原样本SHA核证，不重做GPU评分；原失败记录保留，GPU5仅补同train读出。候选新full8从真实写盘summary核UUID。新候选报告／模型位于上述根的`runtime_repair_v2/`，沿用首次启动的18:04:59绝对deadline；控制仍为15:34:59，不重置预算。

原tmux为`upMuon_v2_g1_20261104_1434`、`upD16_v2_g3_20261104_1434`及`upControl_v2_g5_20261104_1434`，均已结束，终态以`runtime_repair_v2/{arm}/run.json`为准。原controller PID2626807／2626806及formal timeout PG2631968／2632204已退出，GPU1／3／5无本包活动读取者；历史启动／失败日志保留来源，不再列在途。

修复接续只调用以下入口，不能重复原启动命令覆盖旧输出：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_runtime_repair_cmcc_20261104 --component prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_runtime_repair_cmcc_20261104 --component control_full8_cpu
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_runtime_repair_cmcc_20261104 --component sequence --arm muon
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_runtime_repair_cmcc_20261104 --component sequence --arm d16
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.up_runtime_repair_cmcc_20261104 --component sequence --arm control
```

本批结果不相加两个点收益，不自动采用；可迁移方法与当前类别边界已归并至[优化器与结构经验](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)。主协议的BF16整链、完整九项及跨模型验收仍是后续独立任务。

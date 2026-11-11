# 显式 bank 联合导出入口

本包只补一份外部明确指定的完整 bank 组合：原始 BF16 → 原 `install_category_bank` → 新原生 v6 `round_base` → 独立 strict fresh／实际账本 → 既有 PUBLIC 完整八项。它不生成提名、搜索候选、类别顺序、赢家或 cap，不调用旧 KL-only proposal／ranking，不训练或计算新的512教师分数。所有旧活动源码保持不动。

`prepare` 复用活动包外的 coverage reader，重读真实 canonical／request／completed／metadata／full8 plan、summary、evaluation、两 shard 等小文件，核相同原 BF16／模型／数据／原 fit source9／PUBLIC consumer／seed／512和完整八项协议。bank 与样本大字节摘要仅引用原生产证据，明确输出 `payload_bytes_rehashed=false,sample_bytes_reverified=false,formal_eligible=false`。新 producer 的 source pins 独立保存，不改原 bank parent/code identity。实际 inventory 决定类别、名称、层数及分母，没有固定36／252层数假设。

调用者显式提供 `inventory_receipt,baseline_receipt,run_identity,profiles,rate_vector,output_root,allowed_physical_GPUs,limits`。每类 `profiles[category]` 为 `canonical:{path,sha256},checkpoint_id,full8:{summary:{path,sha256},evaluation:{path,sha256}}`。`rate_vector` 按类别名排序，与这些 bank 的真实 recipe 一致。`limits` 必须显式给 `export_seconds,full8_seconds,total_seconds,max_logical_bits`；前两者之和不超过 total，逻辑上限不超过 `floor(2.6N)`。这是整份工作集合的限制，不是每类 cap。packed 仅据实报告。

由调用者先激活 `bitvae`，以下命令在 tmux／nohup 后台执行，CPU 控制器显式隐藏 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_joint_export_cmcc_20261104.producer --component prepare --input explicit_spec.json --output proposal_UNAPPROVED.json
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_joint_export_cmcc_20261104.producer --component run --proposal proposal_UNAPPROVED.json --authorization actual_authorization.json
```

第一条只写一个未批准 proposal，不创建执行目录、模型、有效授权、请求或未来期限。第二条只供另有真实完整八项授权时使用，尚未执行。授权格式沿一次性现有合同：`user_approved=true,authorization_reference,execution_scope,proposal_sha256,preparation_identity,proposal_id,output_root,allowed_physical_GPUs,limits,physical_GPU,authorized_GPU_hours,authorized_start_epoch,absolute_deadline_epoch`，均与实际 proposal／文件 SHA／唯一输出绑定。`execution_scope` 必须区分 `export_and_full8` 和 `export_fresh_ledger_only`，request复制并核同一scope；科学 `run` 明确拒绝技术only。授权到期、总窗口或资源不符即拒绝。此次准备 spec 的资源集合仅0–5，6／7受保护；卡号集合属于这次实际授权，不是跨模型算法常量。启动前仍调用真实 `check_resources`，不抢卡。

实际运行先授权再做大 I/O；原 BF16 文件仅在 worker 开头完整核字节一次，随后调用已有 `worker_result(require_formal=True)` 核本人正式 profile／bank。原 BF16 加载为基底，逐 bank 安装并核所有 owned state／spec／digest及非全部目标补集，保 norm／embed／head。原生导出生成新 UUID，`pending_dense_targets,skip_targets,completed_categories` 均为空，明确不是已完成 CAT 恢复边界。完整实际账本必须与各 bank 本类费用逐项一致、future dense 为零、logical≤2.6且不超过显式工作集合上限；同时报告 packed、全模型 tensor 字节、state 序列化字节和模型目录字节。

fresh 使用原独立完整 strict state／packed精确合同；真实运行另验同输入、同 BF16／SDPA／FP32 RoPE、group8路径与GPU证据。logits 要求有限，并沿 `torch.testing.assert_close` 的 BF16 容限 `rtol=.016,atol=1e-5`，记录实际差值，不把正常浮点差异升级为逐bit条件。通过后才生成可供 `evaluate.make_plan` 消费的正式 endpoint；质量仍为 awaiting。full8 直接复用原 `evaluate.run` 的 FA2、B64／B32、完整样本及子进程清理，无子集或协议变更。

导出 worker 复用现有 `wait_child` 的有限 PG timeout／finally kill／reap；full8 使用 `min(总绝对deadline,实际开始+full8_seconds)`，不重置总授权窗口。唯一输出目录排斥再次运行，任一阶段失败停止后继，不重试／降级。最终 `run.json` 分列内部窗口与授权开始至结束的保守 GPU 窗口，内部／evaluation窗口不另相加。成功时 `joint_full8_result.json.selection_proposal` 可直接作为 `selection.freeze_joint` 的一项；提案全集与 cap 仍须外部明确决定。

CPU 验证入口：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m experiments.cat_joint_export_cmcc_20261104.validate_cpu --output NEW_DIRECTORY/validation.json
```

它使用当前真实七类 K32 小文件准备正例及拒绝反例，另以现成 genuine tiny Qwen3 fixture 的七类 bank（实际 M2、S2各2步）执行正逆安装、原生保存、新 UUID、独立 strict fresh、非目标补集、同路径 logits 与实际账本闭合。tiny decoder 开销超过2.6时如实记录并验证正式预算拒绝；CPU／technical 不能变为正式 endpoint／八项证据。临时模型在测试结束自动清理。

实际 CPU 证据：

- [完整报告](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_final/validation.json)／[核心日志](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_final/validation.log)：exit0、CUDAfalse、15.303927035秒。真实七类252投影的小文件与 `freeze_joint` schema 接线通过；genuine tiny 七类14投影的完整安装／保存／strict fresh／补集通过，正逆安装 state 与实际 logits exact，误差0。实际 tiny logical／packed 均55.375bpw，正式预算明确拒绝。
- [末次受影响补验](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_affected/validation.json)／[核心日志](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_affected/validation.log)：exit0、CUDAfalse、14.569042608秒，绑定最终78项源码 pins；重冻当前 prepare、授权拒绝／CLI／schema接线及实际 dtype 审计，显式引用上述完整报告，未重复 fit／安装／fresh。
- [当前显式 spec](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_affected/explicit_spec_UNAPPROVED.json)／[未授权 proposal](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_affected/proposal_UNAPPROVED.json)仅为 CPU 技术正例，给出的900＋900／1800秒是合同形状示例，不是已批预算；K32不是已提名组合。执行根不存在，没有实际联合模型、有效授权或GPU运行。

首轮验证将混合小 JSON／code 来源中的 `.py` 当 JSON，已改为核原 SHA 后重建语义，保留[失败日志](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_20261104_2120/validation.log)。第二轮 tiny fixture 原先整模 `.to(bfloat16)` 把非持久 RoPE 也转为 BF16，与 fresh 的 FP32 RoPE 不同，见[失败日志](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/cpu_20261104_2050/validation.log)；修正测试准备为原 canonical BF16 factory＋严格装回原 persistent state，保原权重／身份且不放宽容限。两轮无效重复输入及负例文件7份已清，实际释放319,488B allocated；失败核心日志／退出和完整、受影响正例证据保留。

上述CPU交付时尚未验证真实8B联合安装／导出／fresh；其后必要技术验证见下节。FA2完整八项、实际联合质量及最终SC-CAT整链仍未验证，须按另行明确安排的研究批推进，不能由CPU／技术通过推定已获研究预算。完整八项可直接复用本入口，无需新增子集probe或重训已冻结bank。

## 必要真实模型技术验证

21:03 root将前述“本轮仅CPU”的范围更新为一次必要GPU验证，依据AGENTS既有任务内真实模型GPU验证自主权限及当前0–5资源授权，仅用已空闲GPU1、≤900秒／.25GPUh。该上限是root具体限定，不声称用户逐字新批了该数字；不属于下一研究批。新的[scope CPU补验](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/technical_scope_cpu/validation.json)／[日志](../../result/qwen3_8b_mmlu65_20261004/cat_joint_export_cmcc_20261104/technical_scope_cpu/validation.log)为exit0、CUDAfalse、11.161923964秒，复用完整tiny证据，核技术scope／phase、CPU不能冒真实模型技术、技术不能进正式八项及当前源码pins；Astra终审通过。

技术入口仍需独立实际授权，使用 `--component validate-gpu` 而非 `run`；它只启动同一个 `export-worker`，全union／fresh／实际账本路径保持，另保存 `phase=technical_validation,formal_eligible=false,quality_status=not_evaluated_technical_only`。实际源bank仍必须是原BF16的真实完整正式profile；不能用CPU或短权重替代。技术入口不生成full8 plan、质量分数、selection proposal、赢家或cap。授权的 `cleanup_reserve_seconds` 为同一绝对窗口内的清理保留，未使用的full8配置上限不构成full8授权。

本次实际tmux `Cat_joint_tech_g1_20261104_2116`，driver2911942、worker／PG2912157；[独立实际授权](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/actual_authorization.json)起点21:14:56.648、总绝对deadline21:29:56.648，worker最晚21:29:26.648，保留30秒清理。先用此实际auth调用科学run已[立即拒绝](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/actual_scope_rejection.json)，随后启动技术only。[控制器日志](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/controller.log)／[worker日志](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/execution/worker.log)及终态原件在同目录。

[真实技术结果](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/execution/joint_result.json)／[控制器终态](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/execution/technical_run.json)／[worker退出](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/execution/exit.json)均exit0，UUID `b583ada7-9e33-47c5-b9af-c75c59966dc8`。原BF16及七bank真实字节来源核验后，实际252投影并集、每bank完整键／packed digest、非目标补集exact；3423持久张量独立strict fresh exact。live／fresh两次GPU group8各252／252、failed0，参数BF16／RoPE FP32／SDPA同路径 logits 的最大绝对差及relative L2均0。全部数值、逐类／component账本及来源保留在上述原结果中。

实际 `N=6,945,767,424`、logical＝packed `14,635,431,936` bits、均为 `2.107100777004076bpw`，future dense0，逐类费用与各来源本类实测全部相同。完整tensor `4,319,364,864B`、state序列化 `4,320,773,301B`、保存时模型目录 `4,337,785,868B`；峰allocated `23,098,403,328B`、reserved `23,362,273,280B`。授权开始至控制器结束373.175958秒／.103659988GPUh是含大I/O的子窗口，最终成本还须延伸到收尾结束，不与它相加。

当前实际源码：`producer.py` SHA `1f8bf9b06a97dcf5c069cd6bba988098459fa468306fb3d3271370faca2612d4`，`validate_cpu.py` SHA `9b22f4dfb16b7ff5acd0afe73994b12776a476b6a3cf730989560f16e4f3a12a`。技术endpoint始终 `formal_eligible=false`，未产生full8目录、质量分数或selection proposal；这次通过只覆盖真实8B联合安装／保存／fresh／账本及短输入GPU计算，不覆盖FA2／正式批量／完整任务质量或SC-CAT搜索。

Astra已核真实证据通过，随后依持续收尾授权删除本次新生成的 `pytorch_model.bin` 与六份tokenizer副本，7文件实际释放 `4,336,676,864B allocated`，原七类bank／raw不动。保留原meta／config、完整数值／实际账本、source SHA、命令、日志与退出。[最终清理／费用 closure](../../artifacts/cat_joint_export_cmcc_20261104/gpu_technical_20261104_2116/cleanup_closure.json) SHA `0008569659208b11d114c16af88f420699d81f491fa9b2c557a76acf9a42404c`：授权开始至实际cleanup结束 `600.922811747秒／.166923003263GPUh`，含373.175958秒子窗且不相加，低于900秒／.25GPUh；21:24:57.571完成，driver／worker已退出，无占卡或后继。当前仅保留技术metadata证据，生成的联合模型不能加载／复评／续训；本次不提名K32，也不改变21项profile的科学覆盖状态。停止监控。

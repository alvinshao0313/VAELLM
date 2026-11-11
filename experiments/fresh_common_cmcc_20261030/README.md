# cmcc 共用 BF16 前缀与 B gate raw2

本入口只生成下一批 C0/C1 共用前驱：原 HF → q/k/v/o 各 S2×10K＋500 恢复 → 真实 `after_o_proj` 保存 → 新进程 seed31 严格重载 → B gate S2×10K → 原生 `round_base` 保存后停止。始终声明全部七类别，保留原未来 dense LoRA 范围；不进入 gate500、不预选 C1、不运行 up/down 或评测。

复用 [fresh 配方](../cat_structure_fresh_20261029/README.md) 与原 N snapshot；仅在本隔离目录绑定 cmcc Python、模型根及阶段边界，旧入口、运行中的 312 个 pins 及原始 metadata/runtime 均不修改。`run.py` 依次启动两个真实原生进程，不设等待占卡或重试；保存后核对 UUID、runtime identity 和 raw2 非 gate 补集与本次 after_o exact。相同原生逐类恢复期间的浮点行为沿原算法。

调用者激活 `bitvae`，以 tmux/nohup 后台执行。先用 `validate_cpu.py --output NEW_CPU_DIR` 核真实 HF safetensors 头、当前 Alpaca train token、五源 train 路径、原 B 实际配置及已迁入真实 after_o/raw2 metadata。旧 fresh 的原 template 与 CPU 输出未迁入，不声称重跑全部 16 项；本次复用 14 个可直接运行的原合同用例并对照原 B 实际配置。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u experiments/fresh_common_cmcc_20261030/validate_cpu.py \
  --output result/qwen3_8b_mmlu65_20261004/fresh_common_cmcc_20261030/cpu

python -B -u experiments/fresh_common_cmcc_20261030/run.py \
  --output_dir /data1/cmcc/VAELLM/artifacts/fresh_common_cmcc_20261030/short_v2 \
  --gpu 5 --validation-only

python -B -u experiments/fresh_common_cmcc_20261030/run.py \
  --output_dir /data1/cmcc/VAELLM/artifacts/fresh_common_cmcc_20261030/formal \
  --gpu 5 \
  --validation-report /data1/cmcc/VAELLM/artifacts/fresh_common_cmcc_20261030/short_v2/run.json
```

短验保留真实模型、全 512×256 校准、M36 batch、七类别和原 500 日程；每个拟合 stage 只执行 2 次更新，q/k/v/o 恢复各执行原日程前 2 次更新。metadata 明确 `validation_only`、actual2/planned500；短验状态不作为正式父模型。两次原生保存及独立接续完成后，CPU 检查 actual raw2 与 after_o 非 gate 补集、runtime identity 和真实更新；正式入口必须绑定同代码 SHA 的成功短验，正式重走全新 BF16，不继承短验权重或 optimizer。

`request.json` / `run.json` 记录实际命令、环境合同、源码 SHA、HF 身份、子进程 PID/start、退出；`prefix.json` / `gate_raw2.json` 记录拟合时间、校准 token SHA、真实保存与峰值；stdout/stderr 为同名 `.log`，真实退出为 `.exit`。`boundary_verification.json` 只证明这两个原生边界和共享补集，不证明 C1 消费者、位宽或九项质量。完成后保留正式 after_o 与 raw2，供已安排的 C0/C1 共享生成；短验权重按项目收尾规则清理，保留核心证据。

后续 C0/C1 必须显式绑定本次新父身份，不调用历史 UUID 绑定的旧 sources。优先两支共同 compact 两阶前缀；各自 gate500 与 live up/down，以及每类初始化匹配和父相关统计／OPD 重算由下一阶段入口负责，本入口不提前实现。

## 10月30日新机实测与启动

CPU 检查在 [validation.json](../../result/qwen3_8b_mmlu65_20261004/fresh_common_cmcc_20261030/cpu/validation.json) 完成，7.45 秒，14 项适用旧合同通过；新 B 的七类 resolved spec／recovery 与迁入原实际 argv 相同。原 HF 五分片已有 [官方 manifest 完整 SHA256 验证](../../result/migration_audit_20261030/qwen_direct_download.json)，本轮真实加载成功；399 个 tensor／16,381,470,720 tensor bytes，Alpaca train fingerprint 与原证据相同，新 token SHA 仍为 `6f6f73ca44838dbd61e01bcd240d5a03c2a45266b63102fc582045f174d55475`。

第一次 `short` 在新鲜校准和 q 拟合阶段主动停止（真实 exit143），用于修正短验历史标注，未产生恢复节点或正式权重。审查发现复制的原生累积 history 只有末次条目带标注；隔离 saver 现为每项已完成历史统一标记 actual2／planned500／validation_only，raw2 当前 `distill_stage` 仍为 `None`。这只修正证据描述，不改变训练语义。

`short_v2` 已完成／exit0，总计 580.50 秒。真实 prefix 进程 174462 与 raw2 进程 179942 分离；after_o UUID 为 `d877cbc7-1102-4566-97c9-abee166ef797`，raw2 为 `fe60602e-d5fc-4e83-8f45-24ee8b541495`。四恢复均完成两次真实更新，第二次 decoder master 实际变化；包含真实 B4×1024，q/k/v/o 第二步分别 5.98／5.89／6.60／8.18 秒。prefix 峰值 allocated 44,848,142,336 B、reserved 45,267,025,920 B；raw2 单独进程峰值 allocated 14,408,326,656 B。独立 CPU 检查核对 2,091 个非 gate 字段 dtype／shape／value exact，七类别 runtime identity 一致，raw2 共 2,559 个 tensor、gate 恢复更新为 0。核心证据在 `artifacts/fresh_common_cmcc_20261030/short_v2/{run,prefix,gate_raw2,boundary_verification}.json` 和同名日志；只是路径和保存接续验证，不是质量结论。

正式共同前驱于北京时间16:53:10启动，20:37:08完成，两个phase及外层均exit0，总13437.184128761292秒；prefix为12740.58秒、gate raw2为683.72秒。原driver PID／start_ticks为183022／8777845，prefix worker183027，独立raw2 worker279235，均已退出。输出为`artifacts/fresh_common_cmcc_20261030/formal`，实际命令与源码pins在其中`run.json`／`request.json`，无短验权重继承。

正式after_o UUID为`7eeb82b4-3868-4cc5-9fa3-51870ee55894`，共享raw2为`f4669891-9028-4665-85a5-71c26ac45c33`。原始`boundary_verification.json`核对2091个非gate字段exact、runtime identity一致、raw2共2559字段且gate恢复0；q/k/v/o各S2×10K及500恢复、B gate S2×10K均有原始完成记录。这是共同前驱完成证据，尚非全压缩成品或质量结果；两边界及runtime继续保留给C0/C1。首次C0消费在其导入边界退出，未更新这些原件，也不要求重跑本前驱，见[C0修复与接续](../fresh_fixed2_cmcc_20261030/README.md)。

短验已按无后续用途规则收尾：删除 74 个无用权重、runtime／tokenizer 重复及原生日志／参数副本，实际释放 98,988,285,952 B（约 92.2 GiB）。两个真实边界 metadata、核心 JSON／日志和正式绑定的成功 proof 保持 SHA 不变，清理记录在 [cleanup.log](../../result/qwen3_8b_mmlu65_20261004/fresh_common_cmcc_20261030/cleanup.log)。短验现不再可加载；正式任务只绑定其证据而从原 HF 全新训练。无新增算法或质量经验。

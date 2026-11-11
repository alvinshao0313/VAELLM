# 整类 dynamic／Rice 的完整九项评测

本入口仅适配 `vaellm_adaptive_native_model_v2`，直接调用真实 `checkpoint.load`；不生成 dense 模型、不改名为 v6，也不构造原生审计字段。最终端点必须是 216 个 native＋36 个 compact gate＋0 个 pending dense，来自同一臂实际完成的 gate／up／down 流程。`completed_whole_class_pipeline`、formal 500 恢复／10000 拟合预算、原始 B manifest、最终 UUID、独立 strict／同输入 BF16 logits 证据及当前模型文件 SHA 均绑定。

完整八项复用已验证的 HFLM 和原始逐题保存实现：MMLU batch64，其他七项 batch32，0-shot，不设 limit，不改 tokenizer／长度语义；检查全部 64 个叶任务、24742 题及与原 BF16 相同的任务配置／版本。LongBench 复用原统一分区、runner、官方生成设置、terminal binder、merger 和 scorer，全部 21 项／4750 条覆盖后评分；每分区只跑一次，无自动续跑、重试或部分评分。

最终公式直接调用现行 `transfer_nine_task_acceptance.combine_scores`：九项等权、未舍入分数的平均下降严格 `<0.9pp`，logical `≤2.6bpw`。packed 位宽、完整目录字节和序列化开销另报。源码绑定覆盖实际评测／消费者依赖；历史训练源码哈希保留为 provenance，不要求无关在开发训练入口字节不变。预热工具使用 N runtime 的 `tools/cat_eval.py` 绝对路径，实际路径和 SHA 写入加载审计。

在项目根目录、调用者已激活 `bitvae` 的 shell 运行。CPU driver 隐藏 CUDA，实际子进程按 `--gpus` 分配设备；正式运行用 tmux／nohup，stdout/stderr 落盘。以下是准备好的正式命令，只有各自真实 formal 末点完成后执行；不能把 short 成品套到正式入口。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.adaptive_class_evaluation_20261030.evaluate \
  --checkpoint /data/star/VAELLM/adaptive_class_20261030/dynamic/final_model \
  --pipeline-report /data/star/VAELLM/adaptive_class_20261030/dynamic/report.json \
  --output result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/dynamic \
  --gpus 5,6,7

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.adaptive_class_evaluation_20261030.evaluate \
  --checkpoint /data/star/VAELLM/adaptive_class_20261030/rice/final_model \
  --pipeline-report /data/star/VAELLM/adaptive_class_20261030/rice/report.json \
  --output result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/rice \
  --gpus 5,6,7
```

`--check-only` 只检查实际 final 格式、存储、来源及协议，不启动推理。两条命令的设备只是显式用法示例；启动前按当时占用分配，不同时复用同一设备。完整输出为 `evaluation.json`、`full8/` 原始逐题文件、`longbench/merged/scores.json` 和 `longbench/merged/nine_task_acceptance.json`，汇总在 `current_goal_result.json`。质量未达仍算完整评测完成，只有无效证据退出才阻断汇总。

CPU 格式／协议验证和真实 GPU 推理验证分别使用独立目录：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.adaptive_class_evaluation_20261030.validate_cpu \
  --checkpoint /data/star/VAELLM/adaptive_class_20261030/short_dynamic/04_gate_proj/round_base \
  --output result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/cpu02

CUDA_VISIBLE_DEVICES=5 python -B -u -m experiments.adaptive_class_evaluation_20261030.validate_gpu \
  --checkpoint /data/star/VAELLM/adaptive_class_20261030/short_dynamic/04_gate_proj/after_recovery \
  --output result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/gpu_dynamic01
```

GPU 短验覆盖真实有限 logits、2 个 RTE likelihood 样本、所有 compact 的实际解码缓存，以及原 LongBench 三个按输入几何选择的最长／512-token 摘要／SAMSum 多步生成。Rice 臂用自己的新路径和输出重复同一关键消费者检查。短验允许真实 partial mixed 边界，明确不产生完整八项／九项或 final2.6 结论。

2026-10-30 迁移交接时停止新增任务及主动轮询，已启动进程自然运行，未停止训练／评测。当前只确认 CLI 导入／解析 exit0；`cpu01` 正确拒绝已清理 `pytorch_model.bin` 的旧证据目录，未恢复旧端点。`cpu02` 已实测拒绝把真实 144＋36＋72 边界当 final，随后因准备过程中本模块 `bindings.py` 改变而按源码 SHA 契约退出1；未放松检查。代码稳定后的 `cpu03` 和动态 GPU 短验已启动，交接时尚未读取终态，**不能称通过**；Rice GPU 短验未启动，两臂正式完整九项均未启动。

| 已启动任务 | 运行标识（启动时实读） | 输入和输出 |
| --- | --- | --- |
| CPU 格式／协议核验 | tmux `adaptive_eval_cpu03_1030`；Python PID `662072` | 输入 `short_dynamic/04_gate_proj/round_base`；结果根下 `cpu03/`、`cpu03.log`、`cpu03.exit` |
| GPU5 动态必要推理 | tmux `adaptive_eval_dynamic_gpu5_1030`；Python PID `662071`；物理 GPU5 | 输入 `short_dynamic/04_gate_proj/after_recovery`，UUID `5a02cf39-561b-4f35-bbd8-c248da26dc16`；结果根下 `gpu_dynamic01/`、`gpu_dynamic01.log`、`gpu_dynamic01.exit` |

恢复前先核进程身份及日志／退出文件，不按本表认定仍运行或成功。动态输入的独立 `strict_reload.json`（3063字段、同输入 BF16 logits exact）由训练方已完成，本评测复用该证明；本 GPU 短验自身的 HFLM／长输入／KV／缓存／生成结果尚待终态核验。Rice 必须用其自身实际消费者和新边界补关键推理验证；final216＋36＋0 的来源绑定及完整九项质量闭环仍待真实 formal 末点，不能用本 partial 诊断替代。

迁移需要本模块全部源码和 README，以及结果根 `result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/` 的最小证据。模型与训练输入在 `/data/star/VAELLM/adaptive_class_20261030/`，原始 B 归档在 `/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/B/`；本模块未复制或修改这些权重。运行还依赖 `bindings.py` 中明确指定的 N runtime、原 HFLM sample writer、LongBench 统一工具及 BF16 基线／输入 archive。未改 `current_plan.md`，未执行 Git 操作或安装依赖；本次新增文档索引／链接检查尚未运行，交由迁移后接续。

结果及源绑定保存在 `result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/`。经验复用：[checkpoint 运行依赖与格式边界](../../docs/experience/lessons/checkpoint_lifecycle.md)、[完整逐题与 LongBench 口径](../../docs/experience/lessons/experiment_design.md)、[混合预算消费者](../../docs/experience/lessons/mixed_bit_pipeline.md)。当前无新增质量经验；短验不能证明方法收益或迁移性。

## 10月30日新机接续

迁移后补读真实终态：旧`cpu03`为`verified_cpu_format_and_protocol`／exit0，旧`gpu_dynamic01`为`verified_gpu_mixed_inference_diagnostic`，其LongBench为`diagnostic_complete`／exit0。动态的原有限logits、RTE两例及三个32K／KV／摘要生成case已通过，不重做同一诊断；Rice原GPU诊断未启动，仍须用其自身实际消费者补必要验证。旧证据不当作新机最终216＋36＋0成品或完整九项结果。

新入口`experiments.cmcc_runtime_20261030.adaptive_evaluation`分别准备新机支持配置，并将evaluate／full8／predict／merge／summarize子进程显式路由到同一支持目录。该目录新建原canonical32K/BF16协议的config及template，绑定新机HF文件stat、现有`result/prepare_eval_20261030/longbench/manifest.json`和共享BF16基线迁移身份；历史config、template、预测和metadata不改。仍使用原mixed loader、strict状态与真实预算核对、64叶／24742题、21项／4750条、九项等权及下降严格`<0.9pp`。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cmcc_runtime_20261030.adaptive_evaluation \
  --prepare /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support

CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cmcc_runtime_20261030.adaptive_evaluation \
  --support /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support \
  --checkpoint /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/dynamic/final_model \
  --pipeline-report /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/dynamic/report.json \
  --output result/qwen3_8b_mmlu65_20261004/adaptive_class_cmcc_20261030/dynamic \
  --gpus 5,6,7
```

支持配置准备已在tmux`adaptive_cmcc_eval_support_20261030`真实exit0，状态`verified_cmcc_mixed_evaluation_support`；绑定完整Qwen权重与共享`cmcc_baselines_20261030/qwen3_8b/migration_binding.json`真实verified身份。随后从本轮`short_rice/04_gate_proj/round_base`完成CPU格式／协议验证和Rice必要GPU推理，分别为tmux`adaptive_cmcc_eval_cpu_20261030`、`adaptive_cmcc_rice_diagnostic_gpu2_20261030`（GPU2，shell85667／worker85669）；均exit0，UUID`d61b6ea0-ed72-415c-b687-c2f9fbaa3e6b`。原始证据为`result/qwen3_8b_mmlu65_20261004/adaptive_class_cmcc_20261030/{cpu_rice,gpu_rice}/report.json`，核心日志和exit位于artifact根`adaptive_class_cmcc_20261030/evaluation_{cpu,gpu}_rice.{log,exit}`。

CPU实际核对完整64叶／24742题与21项／4750条协议，并拒绝partial当final。GPU原生144模块group8预热、所有36个Rice compact实际解码、finite真实logits、RTE两例和逐题原件通过；三个原定生成case分别为repobench-p输入32704／生成64、gov_report输入32256／生成512、samsum输入18047／生成35。有限题量不提供方法质量结论，不将原生预热或这份partial模型预算写成最终216＋36验收。

这份round_base仅为144native＋36Rice＋72dense；恢复因`trl`缺失在首次更新前停止，独立推理诊断不依赖该训练库，也不证明训练或down末态。正式评测仍依赖本臂formal末点及同UUID strict证据，当前未启动完整九项。

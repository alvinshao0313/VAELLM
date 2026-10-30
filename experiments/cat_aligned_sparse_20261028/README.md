# C/G 对齐链：新06末态的 abs005／Rice7

本目录是[对齐参考链](../cat_aligned_recovery_20261028/README.md)的07。只接受本profile实际完成并verified的 `06_B32/native_final/final_model` 和审计，核05父、04-R、正式32步、report／prepared／源码SHA、3423字段、3405冻结及504 bank；不以09、raw C/G或旧B32代替新06。03、05／06、成熟稀疏源码及普通评测模块均保持冻结。

## 固定构建、保存和预算

直接只读调用 `experiments/night_sequence_20261026/sparse_build.py::build/export`，不用其硬绑定raw C/G来源的 `prepare/run`。新窄 `sparse07_prepare.py`仅在真实06审计存在后，创建该源独立prepared；没有新源时拒绝，不生成未来UUID或候选manifest。

108个MLP各 `ceil(4096*12288*.005)=251659` 项，总27,179,172项。每个新源都从实际新06的原生BF16／group8部署W重新计算 `HF_BF16−W`，排除保护输入列后按abs选址；保留原256×256块、INT8值、FP32 scale／zero、BF16解码后BF16相加语义。Rice7只无损改坐标表示，保留原支持和值；新源的支持、q／scale／zero、码流长度和实际总开销必须重新生成和测量。

成熟export保存自包含的 `base/` 原生v6＋`rice7_payload.pt`＋本地 `rice7.py`，新UUID、3423原字段冻结、864 Rice字段、fresh strict共4287字段，504 bank不变；逐108模块核保护列、Rice坐标及BF16 patch一致，不持久化展开local indices。格式标签沿成熟 `vaellm_raw_abs005_rice7_bundle_v1`，这是消费者格式名，实际 `source_checkpoint_id` 必须是新06。

实际ledger包括主码、decoder、保护q／index／scale、旋转及全部Rice q／scale／zero／ptr／block IDs／stream／offset／header，要求logical≤2.6、rank0。packed padding与文件容器另列；不能用raw G的旧2.598等数字估算为新成品实测。如果唯一问题是实际固定密度超过2.6，保留合法 `over_budget` 构建／严格回读记录，不进入该候选质量评测，不自动改密度、值位宽或选址规则。

## 完整九项使用真正的Rice消费者

构建／保存／实际加载直接复用成熟raw C/G的 `sparse_bundle.py`／`sparse_runtime.py`；独立consumer仍核108个新稀疏cache和一次真实auxiliary_train32前向、全保护列与持久状态。这个新源必要核证不能由旧raw C/G报告代替，也不重复无关机制测试。

已追溯旧0075 Rice的真实多卡完整九项：`qwen_sparse_mlp0075_rice7_probe_20261024/evaluation_staged/longbench/merged/`。其 `evaluation_adapter/endpoint.py`硬绑定旧B32 UUID、旧载荷字节数及七卡映射，不能直接调用；0075生成权重已退役。本目录沿其实际下层流程，窄接两卡 canonical `task_groups/start_group/binder/merge`，使用当前abs005 raw C/G loader和collector。`gpu_list/task_groups/check_resources`与已冻结普通评测入口的定义AST一致，不修改该模块。

`sparse07_evaluate.py`先在第一张卡完成full8，随后两卡同时运行各自完整任务分区。full8直接调用成熟 `sparse_runtime.py full8`，0-shot、MMLU B64／七项B32、fast tokenizer、64叶／24742题；不二次预热覆盖已安装的Rice patch。`predict07.py`只将原LongBench runner连接到成熟 `sparse_runtime.load_model`，逐分片核4287字段／504 bank／108 cache及实际完整ledger。官方21项／4750题、32K总窗口、官方cap、非思考greedy／seed42不变；分区各86400秒，只等待本轮自己的子进程，不等资源、重试、续跑或部分计分。

完整原始样本与两个分区都正常结束后，由成熟binder／merger完成canonical覆盖、SHA、UUID及参数核证。`summary07.py`只适配实际Rice候选和成熟Rice collector，复用原九项公式，BF16为68.04714491790848，目标67.24714491790849、logical≤2.6／rank0；`not_met`和nine exit1是合法完整结果，不能据八项分数跳过LongBench。

独审发现成熟raw Rice的load proof没有binder必读的 `model_policy`，现存raw C/G报告也证实该缺字段。`predict07.load_model`在成熟loader之后，实际核model_type、RoPE theta／scaling、attention实现和embedding BF16，并要求canonical policy与输入config一致，才把已核policy加入proof；不盲填配置，也不修改binder／成熟loader。CPU只核旧实物证明及canonical配置契约，新源实际模型检查仍待GPU加载。

## 精确入口与输出

调用者先激活 `bitvae`并核Python，cwd为：

```text
/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
```

各次运行使用tmux／nohup及独立落盘日志。以下为后续执行入口说明，本轮未启动GPU；GPU变量须由调用者设为实际新分配资源。profile用环境变量显式选择，子进程继承，CPU合并计划也记录该profile。

```bash
export VAELLM_ALIGNED_PROFILE=G
# 来源独立CPU材料；不创建07候选目录或正式prepared。
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_prepare.py --materials-only

# 只有真06审计齐全后才可执行。
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_prepare.py
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_worker.py build --gpu "$VAELLM_ALIGNED_GPU"
CUDA_VISIBLE_DEVICES='' python -B -u /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_worker.py export

# verified且预算通过才运行；over_budget在export处收束。
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_worker.py consume --gpu "$VAELLM_ALIGNED_GPU"
CUDA_VISIBLE_DEVICES='' python -B -u /home/star/VAELLM/experiments/cat_aligned_sparse_20261028/sparse07_evaluate.py --gpus 2,3
```

最后示例假定GPU2／3已另行分配且空闲；本入口不会等待占用者。`--check-only`只核当前真实bundle和预算，不检查资源或启动评测。`over_budget`会正常返回记录，不创建full9。

输出固定在 `/data/star/VAELLM/cat_aligned_recovery_20261028/{C,G}/07_sparse005/`：输入 `prepared.json`；真实构建 `build.json`、`artifacts/payload.pt`；native `artifacts/final_model/`、`audit.json`；独立 `gpu_validation.json`；最终 `full9/mmlu.json`、`seven.json`、原始样本、`longbench/groups/part_{01,02}`及`longbench/merged/scores.json`、`nine_task_acceptance.json`。每一步显式调用，拒绝覆盖；没有自动接续07、通用调度器或占卡进程。

## 2026-10-28 CPU材料与边界

来源独立材料为 `/data/star/VAELLM/cat_aligned_recovery_20261028/sparse07_cpu_materials.json`，核心日志为同目录 `sparse07_cpu_materials.log`。真实bitvae／tmux CPU入口已exit0、CUDA未初始化，核成熟raw C/G已完成build／4287 fresh strict／独立108消费者／完整九项证据、旧Rice多分片真实终态、原4032/full-RHT路径、C/G09的108-MLP几何、原aux32 train身份、真实06未就绪拒绝、显式构建接口／全部新入口import与AST、两卡21项唯一分区及原分组函数AST一致。

这些是来源独立材料和已验证codec／profile几何；**当前新06仍无，正式prepared、07支持／值／码流、实际新预算、bundle、独立新源consumer及完整九项均未执行。** 不把旧raw C/G载荷或旧0075 UUID写成新候选。新源到齐后，仍须按build→真实保存／readback／fresh strict→新源真实消费者→完整full8／两片LB21／九项完成；源码与CPU准备不代表新精度结果。参考[稀疏／Rice存储经验](../../docs/experience/lessons/compression_choices.md)及[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)；本次暂无新增技术经验，不改主record／INDEX。

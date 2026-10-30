# G09 Muon：Hotpot 唯一覆盖与重复对照

2026-10-30，标签：G09／Muon／OPD／train-only／覆盖。运行状态：CPU 已验证，正式训练已启动，尚未完成；质量结论：证据不足。唯一当前执行计划由[主计划](../../docs/experience/records/w25_transfer/current_plan.md)维护。

本组检验有限呈现次数下，把 Hotpot96 两遍改为 Hotpot192 各一遍的完整配方效果。它不预设覆盖不足是原因，也不预设增加唯一输入有收益。完整九项与原控制配对后才决定采用或结束该配置；不凭训练 loss、MMLU 或部分 LongBench 提前判定。

## 指定控制与唯一改动

共同父是 G09 原生完整成品，UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，不是先从原 Muon 成品续训。控制复用已经完成的[同父 fresh Muon32](../g09_recovery_20261029/README.md#10月30日完整结果与同父配对闭合)，九项原值 **66.06887949148307**；本组只新训覆盖臂，不重跑控制。

固定 M1024 各一次、Qasper32 各两次、32 次更新、每步 M32＋QA8、M／QA 各 `.5`、R0、全部1512个 decoder 叶／4,762,368参数、seed137、原500步线性日程的前32步。M 沿原 `.5 F + .5 gold` 四选项目标且只混合一次；QA 沿当前学生 native greedy 轨迹及原 full-F 全词表在线 forward KL。不得使用既有 QA teacher-forcing targets。

0-based groups0..19 完全不变。仅 groups20..31 的 Hotpot8 槽按既有 hash 池 `H[96:192]` 替换，M 输入／顺序／权重和 Qasper 原值全部保持。成熟 fit 仍使用 `qa_group_index=index%16` 作为原域槽号；真实呈现32个 QA batch，其中28个不同 batch、224个唯一 QA 文档，不能把 modulo16 字段称为实际唯一组数。

Hotpot prompt 呈现总量 **277934**，控制 **278818**，差 **−884 tokens**；Qasper **408348**、M **364534**，二者保持。固定文档呈现和更新数不等于等 token 或等算力。`formal32/report.json` 的 `training.actual_OPD_supervision` 汇总各域真实生成监督 token；原逐步 `rollouts`／`QA_documents` 和 `formal32.log` 保留逐文档生成与监督量，生成轨迹由学生反馈决定，不保证与控制相同。

## 来源与实现边界

直接复用已有池 `/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_hot_qa512_s128_20261013/dataset.pt`，SHA256 `b954e547def4756033cd43e0286375e04d3112584d892ff6074740a588a9b9ba`；该池 H384 的前96篇和控制 H96 逐值一致。本组只取前192篇，不改已有池和 teacher cache。

HotpotQA release v1.1，HF revision `1908d6afbbead072334abe2965f91bd2709910ab`、`distractor/train`、90447行，[官方许可 CC BY-SA 4.0](https://hotpotqa.github.io/)。保持 `sha256('causal_output_fit/137/'+case_id)` 后按 case_id 打破并列的完整 train 排序，不细筛文本、跳样、替换或依据测试题选样。原始来源及 shard SHA 见[data manifest](../../data/hotpotqa_train_20261011/manifest.json)，既有选择／tokenization 见[QA512 manifest](../../result/qwen3_8b_mmlu65_20261004/qwen_hot_qa512_s128_20261013/dataset_prepared.json)。M／Qasper 的来源、版本、split 与原输入沿控制 prepared 和其传递绑定，不新建训练来源。

`bindings.py` 只增加显式新 dataset／contract，`run.py` 保留原1512叶生命周期并增加监督量汇总。stage-local Muon 数学、矩阵／bias-norm分组、NS5 BF16、FP32 master／momentum、Nesterov `.95`、NS 系数、`.2sqrt(max(m,n))`、aux Adam `(.9,.95)/eps1e-8/WD0`、joint clip1.5、native rollout、fit、motion、原生导出全部直接调用受保护的成熟 Muon 文件。κ仍由本次首个真实 joint clip 梯度计算且随后固定，不能复制控制／旧 short κ。

新输入长度低于原 QA 峰值，不新增资源或数学路径。复用原真实 Muon 两更新与 native export 验证，不重跑 GPU short，不重建 teacher targets。CPU 必须核验来源／原始行／签名／张量、前20组、末12组映射、曝光、M/Q及 M targets逐值不变，并核实际运行的 fit／optimizer／rollout／loader／export 绑定；通过后直接正式32。显式新 dataset 位于本输出目录，不改旧绑定源码、source 模型、direct Adam prepared 或旧 Muon pins。

参考了[同父恢复净收益与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)及[覆盖／预算比较边界](../../docs/experience/lessons/experiment_design.md)：只改一个域，固定父与目标，用完整九项判定，token 与真实成本另报。历史 QA512_128 相对 HOT128 的九项曾为 `+0.2471066161pp`，旧扩池并非全负；旧初态／预算／多个域变化不能代替本组配对证据。

## 运行与验收

本组独立根目录 `/data/star/VAELLM/g09_hotpot_coverage_20261030`。代码／脚本 SHA、原始来源和控制 bindings 均写 `prepared.json`；模型文件保持原 logical／packed 账本及 rank0，由真实导出审计检查。控制 logical2.5295091711956523、packed2.6391813858695654 bpw，计入 decoder、保护 indices/scales/weights、rotation，packed 另计码padding；本组只能在导出后称为实测相同。

调用者先激活 `bitvae`，不在实验 `.sh` 内激活。CPU 准备使用 tmux，隐藏 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/g09_hotpot_coverage_20261030/prepare.py
```

CPU 通过后在项目根目录，以 `bitvae` 的 tmux 执行一次性固定前驱 pipeline：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/star/VAELLM python -B -u experiments/g09_hotpot_coverage_20261030/pipeline.py
```

顺序为 GPU4 的 fresh formal32 → GPU4 可见 CUDA 的原生 export → 再核 GPU5/6/7 无 compute 占用 → 成品完整 full8＋LongBench21／4750。原 Muon 正式32实测约4695秒，作为本组约1.3小时训练的调度参考；不保证本组墙钟相同。正式训练超时上限43200秒，评测沿成熟入口的既有运行上限，不创建 daemon、重试或资源轮询。任一步失败停止依赖并保留原因；GPU5/6/7若届时占用，停止于评测启动前，不干扰既有任务。

核心证据：`prepared.json`、`dataset.pt`、`pipeline_launch.json`／`pipeline.log`／`pipeline.exit`、`formal32.log`／`formal32.exit`、`formal32/report.json`／`run.exit`、`native_audit.json`、`full9_launch.json`／`full9.log`／`full9.exit`。新成品在 `native_final/final_model`；评测在 `evaluation/Muon/`，`Muon`只是沿用成熟评测 profile，整个根目录专属于本覆盖臂。完整九项入口 `evaluate.py --gpus 5,6,7` 可独立执行，`--check-only`仅核真实成品，不等待／重试。

验收保持当前 BF16 **68.04714491790848**、九项等权未舍入绝对下降严格 `<0.9pp`，即九项 `>67.14714491790848`；logical≤2.6 bit、rank0。九项仍是 boolq、rte、winogrande、arc_easy、arc_challenge、openbookqa、piqa、mmlu、LongBench v1完整21任务宏均值，不设单项门槛，不按八项先出分取消LB。

## 10月30日实际验证与启动

`bitvae`／Python3.11.16 的 CPU preparation 实际48.0001秒、退出0、CUDA未初始化。完整90447行 train 身份 hash 排序和两份原始 shard SHA通过；前H96逐值一致、新H[96:192]原始行／完整context／冻结tokenization签名通过；原M/Q及M targets、前20组、后12组单域映射、权重／顺序／曝光及实际 fit／optimizer／rollout／loader／export 调用绑定全部通过。新显式 dataset 20,767,690B，SHA `28f353d82b59cb4cf1a2328c2eeae0e48642667ab7ac5e75d4aa67d30e7d4217`；prepared SHA `dedeac111e8deaf8c0d93b4b8eb46855d56760a09dd6b4f16342477f38ba5c8a`。实际H／Q最大prompt2678／10586，TF输入最大2682／10593，未增资源峰值路径。未重复GPU short。

完整九项真实 CLI 导入／参数解析通过；独立 `evaluate.py --gpus 5,6,7 --check-only`在真实末点尚不存在时，准确拒绝新输出根的缺成品前提、退出1，未创建评测目录。这是预期拒绝检查，不是成品审计或质量通过；日志合并入`prepare.log`。

09:46:50 启动 tmux `G09_H192_GPU4_full9_567_1030`，shell657558、pipeline657563、正式worker657569；pipeline source／argv／GPU UUID与启动占用在`pipeline_launch.json`。GPU4实核0MiB、无compute，正式入口为已绑定的`run_formal.sh`，训练及导出均保持GPU4可见。GPU5当前还分配给另一项有限验证；本组约1.3小时训练后，pipeline只做一次5/6/7资源检查，可用才执行完整九项，届时若占用就保留完整成品停止，不重试或占卡等待。

已有限核实真实主循环推进：elapsed246.3319秒完成第1次更新（该组148.2279秒），M32＋8QA真实反传、1512有限梯度、jointclip前norm122.5289383、504Mu／1008Adam连续state1通过；本次首真实梯度新κ为`5.572598053734668`，矩阵／aux LR分别`5.572598053734668e-6`／`1e-6`，188252个BF16 decoder元素已改变。首组生成75个监督tokens；elapsed272.6828秒第二组已生成65个，1512 master／optimizer逐值往返、504 momentum／1008 Adam状态与固定κ保持、连续step1通过。上述数值是本次实际运行，不声称与旧控制位级同轨。已确认推进并停止主动轮询；训练、导出及完整九项均尚未完成。

文档索引已刷新；全库链接检查当时149项历史／并发缺失，本目录README无缺失相对链接，未扩查或修改其它主题。原CPU、预期缺末点拒绝和链接检查输出归并`prepare.log`；7个无引用的一次性`/tmp`日志／退出文件已核精确路径及归并后删除（逻辑26970B、`/tmp`分配49152B，记录在核心日志），未删除模型／数据／配置。

尚无新增质量经验。新 dataset／endpoint 保留用途为本组正式恢复、独立导出和完整九项；源与控制均受保护。结论闭合后按最小证据与明确后续用途清理中间权重，不自动保留无用途副本。

## 迁移前暂停交接（10月30日）

用户要求完成本次安排后暂停推进、准备迁移。现已停止主动检查和新启动；保留既定后台 `formal32 → GPU4可见CUDA export → 一次检查GPU5/6/7 → 完整九项`，没有修改它的绑定源码。上节第1更新及第2组生成是暂停要求前最后核实的进度，不代表暂停时或迁移时的实时完成状态。后台会继续写产物。

最后已核身份：tmux `G09_H192_GPU4_full9_567_1030`；tmux shell657558、pipeline657563、训练worker657569。pipeline实际启动 `2026-10-30 09:46:50 Asia/Shanghai`，`pipeline_launch.json.started_unix=1793324810.4261627`；最后实核worker仍存活，随后未再查询。未采集proc start_ticks，不能只凭PID识别迁移后的进程。

当前活动写入根目录为 `/data/star/VAELLM/g09_hotpot_coverage_20261030`：`formal32.log`、`pipeline.log` 正由后台持有。后续预定新增写入包括 `formal32/{decoder_endpoint.pt,report.json,run.exit}`（失败则`failure.json`）、`native_final/final_model/`、根目录`native_audit.json`／`execution.json`／`formal32.exit`；资源可用后再写`full9_launch.json`／`full9.log`／`full9.exit`和`evaluation/Muon/`下完整full8／LongBench分片／合并／当前验收结果；正常收尾写`pipeline_terminal.json`／`pipeline.exit`。训练、导出和评测当前均未核完成。

运行期间必须保留原绝对路径和哈希绑定的源码／prepared／显式dataset／源模型／控制证据／原snapshot与数据路径，不对活动目录做最终迁移覆盖或清理。迁移清单需包含这些后续新增文件，或待后台真实终态后再做最终增量同步；当前静态文件清单不能视为完整交付。

若GPU5/6/7一次检查发现占用，pipeline将停止于已完成训练和导出的成品，原因留`pipeline.log`、外层`pipeline.exit`非0，完整九项不会自动启动或重试。用户恢复后先核`formal32.exit`、`native_audit.json`及实际成品身份，再核空卡；只需从本入口`evaluate.py --gpus 5,6,7`衔接尚未启动的完整九项，不重训或重导出。若已有部分评测输出或其它失败，先读真实退出与原因，不能覆盖输出或直接重启。

# G01→G02 几何校准：原生可达性与能力保持

2026-10-29。**当前状态：CPU＋六臂原生拟合＋有界定幅均已完成，尚无任务收益证据。** 两个K目标的有限定幅候选均未改善；down SPACE单步降低表示误差，均不能外推为能力保持或精度提升。本支短测结束，不进入正式链、不扩252／功能面板／新LR；GPU任务均已退出。下文保留最初研究设计的历史语境，实际证据与限制见后面的两轮GPU实测。G01/G02、活动运行源码、评测及依赖保持只读。

## 已核一手来源与准确语义

[论文 v1](https://arxiv.org/html/2610.18634v1) §4、§5.2、Appendix C.5，以及[作者实现 core.py](https://raw.githubusercontent.com/jianzhubao/SPACE/main/src/space_calibration/core.py)已阅读。定义实际线性算子 `W ∈ R[out,in]`，`Δ=Wpost−Wpre`。对 `Wpre=UΣVᵀ`，取平方奇异值累计能量首次达到 `ρ=.5` 的最小 `k`，令 `P=V_k V_kᵀ`，默认 `W*=Wpost−ΔP`。这是恢复输入主子空间上的父模型映射，保持其正交补上的恢复更新；50% 指父权重能量，**不是更新能量的一半**。[作者 API](https://github.com/jianzhubao/SPACE)默认全部二维权重，SVD/差分/投影 FP32、最后转 post dtype；只支持完整非量化浮点 checkpoint。论文的训练期投影损失和在线反复投影是独立对照，不能当作本方法的默认实现。论文任务为 SFT；压缩补偿 OPD 的效果属于待验证迁移。

本轮几何参考必须是**恢复训练实际父模型 G01**，post 必须是其实际 G02。BF16 仅是独立质量参照；不能用 BF16→G02 的差代替本次恢复更新，也不能拿 G09/历史 donor 拼成不存在的训练 pair。

| 角色 | 原生 UUID | 来源 |
| --- | --- | --- |
| pre / G01 | `44ed6d15-abf0-46b0-a627-998a6406087a` | `/data/star/VAELLM/cat_aligned_recovery_20261028/G/01_qk_fineweb2000/` 的 `training.json`、`native_audit.json` |
| post / G02 | `38a7461e-a2f1-4586-80e2-b99883ef2628` | `/data/star/VAELLM/cat_aligned_recovery_20261028/G/02_online_opd32/native_final/final_model`，上级 `native_audit.json` |

已有 verified 审计绑定：G02 从 G01 fresh OPD32，仅 252 投影的 1512 decoder 叶变化；1911 其余持久字段、504 BSQ bank、保护 q/scale、norm/head 固定，3423 字段 fresh strict 一致；无 LoRA。logical `2.5295091711956523` bpw、packed `2.6391813858695654` bpw；前者含投影范围 decoder/保护等，后者另含码填充，不含 embedding/head/model norm。模型 tensor 总字节 `4781328384`，两份实际 `.bin` 各 `4782744841` 字节，不能把磁盘文件大小混写成 logical bpw。

因此论文默认“全部变化二维权重”在这里落实到 252 个**解码后的原坐标线性算子**。不可对小 decoder 的二维参数矩阵直接套 SPACE：该空间与输入 hidden 方向不同，非线性解码也不保持矩阵投影等价。所有未变化的 embedding/head 以及非二维 norm 保持 G02。`Δ` 必须先在同一原生部署解码路径得到 BF16 `Wpre/Wpost`，精确提升 FP32 后相减；不能先相减 BF16，不能混用训练 FP32 master 和部署端点。投影目标在 FP32 保留，转 BF16 的误差单列。

## 已完成真实 CPU 诊断

[入口](diagnose_pair.py)、[原始指标](cpu_diagnostic.json)、[核心日志](cpu_diagnostic.log)。沿冻结 snapshot 的 `_rebuild_converted_modules`、严格加载、`refresh_vae_linear_runtime_after_state_load`、`VAELinear._decode_weight`，仅重建一个真实完整投影；没有模型前向或替代解码实现。选择规则预先固定为中间深度的完整 K 矩阵，作为小矩阵成本探针，不是最终层选择器，也不据其结果推断全 MLP。

运行 `bitvae` Python 3.11.16、PyTorch 2.6.0+cu124、CPU 4 线程，tmux `geometric_pair_cpu_20261029` 已结束、日志 `exit_status=0`、CUDA 未初始化。mmap 两个原件，每个只触达该投影 13 个张量、`1,398,976` 字节；核审计 size/mtime/inode 和同源 UUID，未遍历全模型做 hash。代码与实际使用的两个 snapshot 源码 hash 在 JSON。

| 实测量 | 值 |
| --- | ---: |
| 完整矩阵 | `model.layers.18.self_attn.k_proj`, `1024×4096` |
| decoder 参数量 / 固定输入列 | 18,752 / 64 |
| 两次 CPU 原生参考 BF16 解码 | .277 / .248 秒 |
| FP32 完整 SVD＋投影 | .752 秒 |
| 总诊断时间 | 2.161 秒（不含解释器 import） |
| `ρ=.5` 的 `k` / 实际父权重能量 | 153 / .50005025 |
| `‖Δ‖/‖Wpre‖` | .01408025 |
| 移除更新能量 `‖ΔP‖²/‖Δ‖²` | .04939226 |
| 与保留更新等范数的回缩系数 `β` | .97499113 |
| FP32 投影正交残差 `‖(Δ−ΔP)V_k‖/‖Δ‖` | 2.77e−7 |
| 固定保护列不可达能量 / 理想校准位移能量 | .01851098 |
| 固定保护列不可达能量 / 目标保留更新能量 | .00096180 |
| BF16 舍入误差能量 / 理想校准位移能量 | .05612565 |
| BF16 实际校准位移能量 / FP32 理想值 | .96943648 |
| dense SPACE 转 BF16 后被改变的保护元素 | 20,761 |

CPU 使用真实原生参考算子，**不是 CUDA packed kernel 的逐位复放**；下一轮若执行 GPU 筛选，需在实际部署路径重新生成该 pair 的矩阵。这些数字不能当作整网精度结论，也不能将单个小 K 的时间直接外推全252。实际更大的 Q/O 为 `4096×4096`、MLP 为 `12288×4096` 或转置；完整精确 SVD 应逐投影流式执行，避开同时物化252个FP32矩阵及显式 `in×in` 投影矩阵。

附带读出的相对 BF16 权重平方误差：G01 `.11625840`、G02 `.11659372`、SPACE BF16 `.11649466`、等范数回缩 BF16 `.11658194`。这仅是权重距离；没有任何能力保持结论。“更靠近父模型”由构造保证，也不是实验收益。

## 三种实现位置不能混为一个方法

### 1. Dense 理想干预

每个投影原坐标精确计算 `W*`，以 BF16 临时权重进行同协议前向。可用于判断几何方向的功能效果，额外权重只用于隔离诊断；不能将其宣称为≤2.6bpw的交付模型。生成目标应使用 `C=(ΔV_k)V_kᵀ`，不构造大 `P`。目标只由真实 pair 得到，不引入训练/测试标签。

### 2. 固定码 decoder 重拟合，以及更昂贵的重新编码

本轮选择固定码 decoder 路线。令原生解码器为 `f(D,z,q,s,R)`，从 G02 的 `D2` 初始化，固定原码 `z`、保护 `q/s`、rotation 和拓扑，拟合上述 dense 目标 `W*`；每个投影独立，因此可以流式处理，并保持原尺寸/位宽。优化目标是**post-hoc 表示拟合**，不是向 OPD/SFT 增加几何 loss。

必须报告 `E=Wnative−W*`，包括 `‖E‖²/‖C‖²`、相对 G02 到目标的拟合改善，以及校准方向存活 `⟨Wnative−Wpost,−C⟩/‖C‖²`。只给 `‖E‖²/‖W*‖²` 会被大基准权重掩盖，不能据此宣称投影保留。FP32 master 拟合、BF16导出、保存/fresh 后分别测量，最终以原生部署结果为准。

保护输入列给出严格下界：G01/G02 的 `Δ[:,J]=0`，但一般 `C[:,J]≠0`；固定保护的任何 decoder/主码重拟合都满足 `Wnative[:,J]=Wpost[:,J]`，所以 `‖E‖²≥‖C[:,J]‖²`。本例实际非零，不能用“删掉保护列的校准”冒充原论文；它是另一个受约束算法。下界大小本身不决定任务好坏，也不据1.85%自动否决路线。其余非保护方向还受共享非线性 decoder 容量限制，18,752个参数不能保证任意419万维矩阵修正可达。

重新编码可放开 `z` 降低非保护拟合误差，但仍不能解除固定保护下界。原生 checkpoint 无 encoder 状态，不能假称可直接一次 encode：需接 CAT 的对应 encoder/两级残差递归拟合，或已有严格定义的硬码投影算法，另计编码成本、码变更、原生保存和位宽。若还更新保护 q/scale，必须单列该新变量。首轮不同时启用这些自由度，以免结果无法归因。

### 3. CAT 尚未压缩 dense 的临时 LoRA

现有 CAT `remaining_lora_prefix_decoder` 在每类后把尚未压缩 dense 的 plain LoRA 融合，后续才压缩；这比已压缩 decoder 更容易精确实现输入侧投影。实际 PEFT 约定 `Δ=s B A`，`B:[out,r]`、`A:[r,in]`，令 `A'=A−(AV_k)V_kᵀ`，保持 rank 与 `B/s`。如果文档改写 `Δ=AB` 且 `A:[out,r]`，应改右因子 `B'=B(I−P)`；不能混用名字。

`pre` 须固定为该 CAT 恢复阶段开始时真实 dense 父矩阵，不能用最终 BF16原模型替代。需要在 `_merge_dense_peft_lora_into_base` 前的隔离回调计算参考谱、校准右因子、再走原融合；明确原生 BF16 乘加顺序与理想 FP32目标的差。不能改活动 finalizer，不能对已经压缩的代理套同样融合并新增部署 LoRA。后续 CAT 重新压缩还会产生第二次投影误差，须在最终原生模型检验。这是可行的另一阶段研究，本轮**不选作首测**。

## 唯一建议首测：G01→G02 post-hoc 校准＋固定码原生拟合

以下是初次CPU研究时提出的后续设计，保留其对照定义，**不是当前已执行范围或进一步执行授权**。当时提议范围为该 pair 的所有252投影，`ρ=.5、α=1`，不扫描阈值、不按测试选层、不训练在线新loss；先以小范围表示诊断确认实现，再考虑全投影与短train-only功能筛选。实际只完成后述三完整投影六臂和三个单步定幅case；尚未执行全252或功能筛选，本支已结束。

1. **四个状态，原始读出齐全。** 复用 G01、G02 的完整9及BF16同口径基线（G02正在其他任务评测，不能重复启动）。额外两候选为 SPACE 和逐投影等保留更新范数回缩：`β_l=‖Δ_l(I−P_l)‖/‖Δ_l‖`，`Wshrink_l=Wpre_l+β_lΔ_l`；Δ为零时不变。该控制匹配“距父模型的保留更新范数”，并非匹配“距G02的干预步长”，二者不能混称，也不使用全模型统一β。BF16后两范数实际偏差另报，不暗调系数。所有状态统一解码/舍入/输入/后端。
2. **同时保留 dense 理想和原生拟合读出。** 对 SPACE、回缩使用同一D2初态、固定码/保护/拓扑、固定更新预算与批序，先在少量按矩阵形状/类别和相对深度预先选定的完整投影上核目标可达性，再决定是否对全252执行。不得只挑拟合效果好的层拼成全模。未校准目标 `Wpost` 做一次同实现零更新/初态重构核查；若出现误差，先修接线，不能让原生对照混入另一种解码精度。候选拟合预算与优化器需在执行前固定并记录，不按任务分数早停择点。
3. **短功能面板只用既有 train/auxiliary_train。** 复用本 pair 已验证的 G02 数据清单/教师身份/模板/生成语义；用预先固定的两个观察半组，同时保留 OPD目标域和已有Web/短题保持域的读出。先核实际 case ID、split、来源/版本/许可；不得从正式9测试集抽训练或按错题选样。若复用历史观察组，明确这是已接触的开发观察，不能称独立泛化。G01、G02、dense SPACE、dense回缩、原生SPACE、原生回缩共用同一批输入；记录教师KL、gold NLL、原生自由生成和各域变化，不能只看联合均值或离父模型更近。没有此面板的源码绑定前不启动任务。
4. **完成条件只允许导向下一步。** 几何方向本身未优于等范数回缩，且两观察半组无一致功能改善，则停止本pair/本ρ配置，不扫描更多rho掩盖阴性。dense有益而原生收益消失，同时`E`大/方向存活差，则定位为表示损失，只有这时才值得单独研究重新编码或CAT dense位置。原生拟合保留方向、短面板无明显退化且有净收益，才运行一个原生候选与回缩控制的完整9；短面板不作为正式验收或替代完整9。最终仍按原目标logical≤2.6、相对同口径BF16绝对下降严格<.9pp，且需在另一系列复用规则验证。

这一路径的价值在于把“几何规则是否有用”和“低比特能否表达”分开。若原生SPACE只是回退到G01、撤销G02已获得的能力，不算成功。完整9无净提升时结束当前配置，记录候选数和开发接触；不把结论扩大成所有几何校准无效。

## 下一步明确需要的实现

这是CPU诊断阶段的实施缺口记录；当时只有 `diagnose_pair.py` 已执行。现在下述目标与拟合链路已在三个完整投影实现/验证，整模功能比较与完整原生模型导出尚未实施；完整范围及结果见后续小节。

- 隔离 `build_targets.py`：以原生GPU消费者读同源252投影，流式FP32 exact SVD、SPACE/回缩目标、保护不可达下界、BF16误差；逐模块记录shape/rank/来源，避免复制整个HF模型或永久保存全部dense目标。
- 隔离 `fit_native.py`：复用真实 VAELinear 可导 decoder 和现有原生训练精度，固定所有非decoder持久字段；用原始坐标目标与一致采样顺序重拟合，报告上述方向/误差/成本。记录FP32 master和实际BF16端点，不能只保存master或只比较decoder参数距离。
- 隔离 `compare.py`：复用当前数据/教师/原生自由生成接口，支持内存dense诊断和native候选切换，记录无干预复放及源恢复；活动G02评测脚本保持冻结。所有正式评测仍用既有完整9消费者。
- 原生导出接成熟 v6 saver、fresh strict 和现有位宽审计：新UUID、原schema、1512预期可变/1911冻结/504码不变、logical/packed分别复核。至少真实一次前向验证保存前后功能；不能只用CPU state equality声称GPU功能一致。

相关既有经验已查：[对照与完整能力边界](../../docs/experience/lessons/experiment_design.md)、[编码可达性/保护与恢复](../../docs/experience/lessons/liftquant_recovery.md)、[低秩空间与能力不等价](../../docs/experience/lessons/residual_lora.md)、[精度/原生导出契约](../../docs/experience/lessons/checkpoint_lifecycle.md)。本次新认识仅为该真实pair的几何/表示证据；没有新增能力收益经验。中央入口/index 由根线程归并，避免并行覆盖。

复现已完成CPU诊断（调用者先激活 `bitvae`，后台执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B experiments/geometric_calibration_20261029/diagnose_pair.py
```

该命令更新本目录的诊断JSON，保持模型只读。已结束诊断没有中间权重、optimizer、GPU占位或待清理大产物。

## 10月29日隔离表示原型：实现与固定预算

本节为后续增量，不替代上面的研究定义。已新增 [真实 pair 载入](native_pair.py)、[原坐标目标](build_targets.py)、[固定码拟合](fit_native.py)、[三投影入口](run_native.py) 和 [CPU 契约检查](verify_cpu.py)，仅本目录代码；原诊断及其证据未修改。运行源码来自原 CPU 诊断相同冻结 snapshot，绝不修改活动训练、原 checkpoint 或依赖。

代表矩阵预先固定为中间相对深度 L18 的完整 K `1024×4096`、gate `12288×4096`、down `4096×12288`。这里只诊断三类表示，不是最终层选择器。每投影重新从实际 CUDA 原生部署路径读 BF16 pre/post，提升 FP32 再差分；完整 FP32 SVD (`gesvd`)、`ρ=.5/α=1`，用 `(ΔV_k)V_kᵀ`，不显式物化大 P、不去掉保护列。逐投影处理，不保存大 dense 目标副本。

根线程已在 GPU 执行前审定以下预算：SPACE / 等保留更新范数回缩各从同一 G02 decoder fresh 初始化，各 **500 次完整矩阵更新**，AdamW `lr=1e-5, betas=(.9,.95), eps=1e-8, wd=0`，linear500、无 warmup；FP32 decoder master / Adam，BF16 原生可导解码、stage 累加、逆旋转和保护回填。两臂都以该投影固定 `mean(Wpost²)` 归一化原坐标 FP32 MSE，每步完整矩阵，因此同批序及标量曝光。只保留固定500末点，不按曲线挑点；没有文本、OPD loss、encoder 或码更新。

优化器依据为现有 `joint_reconstruction_probe` / `full_weight_bits_probe` / `decoder_budget_probe` 的固定码完整权重 NMSE 路径，并非套用 OPD 学习率。旧 Qwen L18 gate full-matrix 单步约 `.053s`、两臂2000步总 `216.672s`，本轮六臂500步＋三次完整 SVD 估计约3–8分钟，**不是当前形状的已测耗时**；总15分钟到期记失败，不降低形状、目标或预算。目标比旧 HF 重构近，若末点拟合不佳，只能归于此预算下优化／表示混合限制；严格不可达结论仅来自固定保护列下界。

无干预控制分两层：同 G02 状态 fresh 原生部署复放须 exact；升级 FP32 master、选择可导路径后的零更新 W 偏移单列其能量／校准位移能量及方向存活，允许正常 BF16 核差。若偏移能量达到目标干预能量，暂停该依赖拟合，报告量级交根线程审查。不能将路径差计成 SPACE 收益。

每臂报告 FP32 master/BF16计算的 live 末态、BF16部署以及保存/fresh后的实际部署误差和方向存活，并核 fixed codes／保护 q-scale／rotation／schema 不变。小文件 `*.decoder.pt` 只含六个 decoder 的 FP32 master 与 BF16部署值及来源契约，叫做**单投影隔离载荷**，不是完整原生模型或2.6bpw成品。保存后用源固定字段＋该 BF16 decoder strict 重建，再核同部署路径的 W exact；没有整模前向、全252拟合或任务面板。

CPU 最小验证已于2026-10-29完成，tmux `geometric_native_cpu_20261029` 自然退出、核心日志 `exit_status=0`，耗时 `3.942s`、CUDA未初始化。三投影 strict 状态逐值一致、父子非decoder固定；真实 K 复现原 `rank=153` / β，未干预目标相对误差1、方向存活0；SPACE保护列下界为正，回缩保护位移为0。原始 [CPU报告](/data/star/VAELLM/geometric_calibration_20261029/cpu_contract_v1/report.json)、[CPU日志](/data/star/VAELLM/geometric_calibration_20261029/cpu_contract_v1.log)。这些不覆盖 CUDA、反向、500步拟合与 GPU 保存重载；后续 GPU 实测见下节。

复现入口，调用者先激活 `bitvae`，使用 tmux 后台；输出目录须不存在：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B experiments/geometric_calibration_20261029/verify_cpu.py \
  --output /data/star/VAELLM/geometric_calibration_20261029/cpu_contract_v1

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B experiments/geometric_calibration_20261029/run_native.py \
  --output /data/star/VAELLM/geometric_calibration_20261029/native_three_v1
```

## 10月29日三完整投影 GPU 实测结果

状态：**六臂均完成固定500末点并通过持久状态与原生重载检查**。2026-10-29 13:01:41 起，GPU7／A800，tmux `geometric_native_GPU7_20261029`、pane PID400309、worker PID400312，`167.274s` 自然完成，核心日志 `exit_status=0`，900秒 timeout 未触发；会话和 worker 均退出，末查 GPU7 `0MiB/0%`。实际入口即上面的 `run_native.py` 命令；软件环境 `bitvae` Python3.11.16/PyTorch2.6.0+cu124，FP32 matmul关闭TF32。原始 [GPU报告](/data/star/VAELLM/geometric_calibration_20261029/native_three_v1/report.json)、[核心日志](/data/star/VAELLM/geometric_calibration_20261029/native_three_v1.log) 包含完整命令、两源 UUID/stat、运行源码与原生 snapshot hash、全部3000次更新、原始 pre/post、dense控制和各端点指标。源码未提交，无伪造commit ID。

实际 CUDA 部署解码与原 CPU 参考存在正常低精度路径差，本轮**重新计算了所有目标**，没有借用 CPU Δ。以下是完整 SVD 的真实 GPU 几何；时间包含投影和同步统计，不能据此外推全252。

| 投影 | 完整shape | rank | β | 固定保护下界 / SPACE步长能量 | SVD＋投影秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| K | 1024×4096 | 153 | .97487557 | .01856461 | .540 |
| gate | 12288×4096 | 918 | .85483205 | .00841862 | 6.733 |
| down | 4096×12288 | 966 | .45962460 | .00340723 | 6.998 |

令每臂自己的 `T` 为 FP32 SPACE／回缩目标，`M=T−Wpost`。表中相对误差为 `‖W−T‖²/‖M‖²`，G02无干预恒为1，**不是**对大权重范数归一化。方向存活为 `⟨W−Wpost,M⟩/‖M‖²`，回缩列使用其自身M；大于1也可能伴随很大的正交误差，不能单独称保留成功。master列仍是FP32参数、BF16原生计算，不能写成纯FP32推理。

| 投影／臂 | live初态路径误差 / ‖M‖² | FP32 master末态相对误差 | 最终原生BF16相对误差 | 最终方向存活 |
| --- | ---: | ---: | ---: | ---: |
| K SPACE | .01230946 | 2.121916 | 2.008719 | .199428 |
| K 回缩 | .96754352 | 86.148166 | 73.939701 | 1.895144 |
| gate SPACE | .00008199 | .668414 | .665972 | .409867 |
| gate 回缩 | .00104755 | .870522 | .828479 | 1.074024 |
| down SPACE | .00004180 | .229526 | .228516 | .816546 |
| down 回缩 | .00011290 | .118149 | .114850 | 1.022312 |

各臂 fresh G02原生部署初态 exact，只有六个packed decoder叶可训练；500次真实前向／反向／Adam更新均有限，master、梯度、moments为FP32。每臂7个非decoder字段（码、保护q/scale、rotation等）逐值固定、schema/dtype不变，保护输入列实际部署位移严格0。两臂初始化、优化器／分母／预算及完整矩阵标量曝光相等；K每臂曝光2,097,152,000个权重标量，gate/down每臂25,165,824,000。保存BF16 decoder后，以源冻结字段fresh strict重建，六臂实际 CUDA 部署矩阵均exact一致。峰allocated K `.462GiB`、gate `4.977GiB`、down `4.961GiB`；只含本原型，无整模内存外推。

dense目标转BF16本身的舍入误差／‖M‖²，SPACE／回缩依次为：K `.056144/.159039`、gate `.008376/.026081`、down `.007252/.019825`。两臂理想保留更新范数按β匹配，BF16实际范数未暗调：SPACE／回缩 K `.696109/.696822`、gate `4.451814/4.459571`、down `1.782719/1.738823`。原始无干预与dense控制齐全，不能把这些密集临时算子当原生低比特交付。

这次只说明：该固定预算对 gate/down 的表示误差有部分改善，而 K 两臂未拟合到目标；不能以此宣称严格容量上限。K 首次更新的NMSE从约 `9.93e-6/2.43e-7` 跃升到 `.00170/.00174`，且回缩live初态偏移本就接近目标步长能量；这是现有LR／BF16路径在小目标下的重要条件边界，未据曲线换LR或挑点。SPACE保护列下界是严格代数限制，其余残差仍混合优化预算、非线性表示和低精度路径影响。gate/down的局部MSE改善也不证明能力保持、整模收益或达到当前≤2.6bpw／九项验收；没有启动任务面板、全252、重新编码或正式训练。新经验由根线程归并，此目录仅保留原始证据与实现。

收尾：六个已验证单投影生成载荷无已安排后续消费者，已按持续授权清理 `735,420B`；FP32/BF16各叶hash、载荷hash、精确路径/字节数、冻结与fresh核验和全部原始指标归入同一GPU报告，源G01/G02保持只读。一次性汇总文本 `2,200B` 已归并并删除，合计本次实际释放 `737,620B`，删除后逐目标核实不存在。现仅保留CPU/GPU各一份原始报告和核心日志、本目录源码及原研究证据；没有中间checkpoint、optimizer、dense目标副本或占卡进程。需要新研究时复用实现与原源，使用新输出目录，不自动重跑。

## 唯一追加短验：同一首步方向的有界定幅

根线程／Astra审查首批后，仅批准一次 [step_calibration.py](step_calibration.py)：K SPACE、K回缩、down SPACE 各从真实G02 fresh，完整目标和原loss不变，恰一次真实 backward＋AdamW首步。保留实际 FP32 参数位移 `D1−D0`；先测 `t0=1e-5` 持久BF16 decoder的实际部署位移 `d`，用 `e=T−Wdeploy(D0)` 提名 `a=clip(<e,d>/‖d‖²,0,1)`。候选逐次fresh strict装同初态，再赋 `D0+a(D1−D0)`，不累加更新／Adam时钟；a不改善实际部署误差时最多再4次折半，首次实际改善即停止该case。`d=0`或内积非正时直接记方向失败，不扩网格。完整记录零／原步长／所有候选，不能改写首批500末点或声称训练曲线选点。

已于2026-10-29 13:12:09启动并完成，GPU7启动前再次为空，tmux `geometric_amplitude_GPU7_20261029`、pane PID408231／worker PID408235，耗时 `12.788s`、`exit_status=0`，180秒界限未触发；GPU7自然释放至 `0MiB/0%`。原六臂代码和指标未修改；旧六载荷已按上节收尾，新验证没有依赖它们。原始 [定幅报告](/data/star/VAELLM/geometric_calibration_20261029/step_amplitude_v1/report.json)、[核心日志](/data/star/VAELLM/geometric_calibration_20261029/step_amplitude_v1.log) 保存源/代码hash、0/t0/候选FP32/BF16叶hash、梯度hash、一次真实Adam提议、全部实际部署误差与冻结检查。

以下仍按每个case自己的 `‖T−Wpost‖²` 归一化，零步长原生控制均为1且exact G02。每行FP32 master列仍是BF16可导计算；以持久decoder转BF16后的真实原生部署作为改善依据。

| case | t0部署误差 | 提名a | 所有a/折半后的部署误差（顺序） | 结果 |
| --- | ---: | ---: | --- | --- |
| K SPACE | 172.185655 | .02503936 | 1.200436 → 1.103533 → 1.051335 → 1.025360 → 1.010752 | 五候选均未改善，结束 |
| K 回缩 | 13897.340967 | .00640925 | 5.064763 → 3.067099 → 1.864083 → 1.439082 → 1.132801 | 五候选均未改善，结束 |
| down SPACE | 2.422731 | .32712036 | .784350 | 首试改善，未再折半 |

K SPACE候选live误差依次 `1.629083/1.630789/1.626810/1.622640/1.573972`，部署BF16实际变化参数元素 `394/193/92/40/20`；K回缩候选live误差 `51.737017/51.388477/47.442688/38.580456/22.654915`，变化 `93/40/20/9/4`。因此这两例有界失败**不是全部候选参数被BF16抹为零**，也不能由一次方向推断容量不可达。live/native初态差仍另列：K两目标分别为目标步长能量的 `.012309/.967544`，没有将路径差归入几何收益。

down SPACE提名候选真实LR `3.2712036e-6`，live误差 `.781173`、原生BF16误差 `.784350`（仅对此目标误差较G02下降21.565%），BF16变化2993参数元素、方向存活 `.162310`；该候选没有继续训练。每case总1次backward／1次Adam.step，所有候选本身0次优化更新；codes／保护／rotation／schema不变，实际保护列位移0，零步长decoder变化0。没有生成权重文件或可清理的大张量；只保留一份报告和核心日志。结论仅限这三个case的首步定幅：down可消除该次已观测过冲并降低表示误差，两个K的有界候选未成功；不声称SPACE能力收益，不安排更多方向／全252／任务面板。

未解决的是两件独立问题：dense理想SPACE相对逐投影等保留更新范数回缩是否有真实功能价值，以及低比特decoder能否可靠表达该有价值方向。两个K的有限候选失败不证明几何校准普遍无效，也不证明固定码容量上限；down的局部改善同样不证明任务精度提升。若以后重新启动本方向，须先以同源pair及既定train-only观察证明dense理想干预的功能价值，再决定是否值得继续表示研究；此前不因局部MSE继续扫描LR、方向、层或扩252。当前没有新训练、常驻GPU或待自动执行步骤。

复现时调用者先激活 `bitvae`、后台执行，使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B experiments/geometric_calibration_20261029/step_calibration.py \
  --output /data/star/VAELLM/geometric_calibration_20261029/step_amplitude_v1
```

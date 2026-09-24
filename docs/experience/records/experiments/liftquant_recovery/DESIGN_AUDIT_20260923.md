> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`experiments/liftquant_recovery/DESIGN_AUDIT_20260923.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# LiftQuant 第二阶段设计核查（2026-09-23）

结论：当前实现是 VAE 表示下的逐 block 恢复原型，不是官方 LiftQuant 第二阶段的严格复现。学习率数字部分取自官方示例，但所作用的参数、代理坐标和更新尺度不等价。通路冒烟通过，不代表恢复有效或设计已经对齐，当前不能据此启动正式实验。

本次只做源码核查、CPU 调度复算和已有 A/B checkpoint 比较；未启动新 GPU 训练，未安装依赖、删除数据或提交 Git。训练代码只纠正解释性注释，没有修改算法。

## 核查依据

- 官方固定提交 `72b3875c770e4579639931fed89dc95e4067edac`；本次查询到的 main 也是此提交。
- 对照 Stage2、FWTLinear、变换矩阵、优化器工具、README、CLI 默认值与依赖文件。
- 远程工作区：`iaaccn74:/home/shaoyuantian/program/VAELLM`。
- 实际运行源码以 `.result/liftquant_recovery/from_distill_init_20260923_02/code_snapshot/` 为准，配置见同目录 manifest 和运行日志。
- CPU 程序：`experiments/liftquant_recovery/design_audit.py`；结果：`.result/liftquant_recovery/design_audit_20260923/audit.json`，退出码 0，`cuda_used=false`。
- 输入：`result/linear_output/distill_init`，无旋转、单 residual stage、native v6，未重新初始化。

官方来源：

1. [Stage2 参数组与循环](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/quantize/liftq.py#L495)
2. [代理初始化](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/quantize/tmplinear.py#L134)及[硬前向](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/quantize/tmplinear.py#L320)
3. [变换矩阵参数化](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/trans_utils.py#L399)
4. [CLI](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/main.py#L155)与[README 示例](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/README.md)
5. [GradScaler](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/utils.py#L27)与[依赖](https://github.com/Heliulu/LiftQuant/blob/72b3875c770e4579639931fed89dc95e4067edac/requirements.txt)

## 可学习参数与模式

官方比较基准为 dense attention block、开启 training_trans 与 finetuning_weights、关闭 pvtuning 的示例路线。

| 项目 | 官方 Stage2 | 当前 VAE 恢复 | 判断 |
| --- | --- | --- | --- |
| 编码 | 七个 FWTLinear 的连续 weight 代理 | 七个已有 VAELinear 的全部 FP32 bit 代理 | 全编码训练意图相同，参数化不同 |
| 解码 | Trans.linear_left/right | 完整 VAE decoder | 架构适配，非等价替换 |
| 独立尺度 | scale、a1、a2 | native VAE 无这些独立参数 | 不能一一对应 |
| decoder norm/bias | 无 VAE decoder | decoder 自身 LayerNorm 和线性 bias 全部训练 | 当前范围不只是解码矩阵 |
| 原 LLM norm/dense/head/embedding | 不在上述 Stage2 可学习组 | 全部冻结 | 冻结原则一致 |
| LoRA / VAE encoder | 不在上述训练组 | 不加入；native ckpt 无 encoder | 符合任务约束 |
| 联合/交替 | 最后 ALL 组联合训练；前面子组 epochs=0，用于转换 | 同 block 全部 code 与 decoder 同时更新 | 与关闭 pvtuning 的联合模式对齐 |

官方变换类的 to_buffer() 实际注册 nn.Parameter，linear_left/right 仍可训练，不能根据函数名认为它们冻结。官方该路线有 7 个 weight 组加 scale、linear_、a1/a2 三组，共 10 组；当前为 7 个 code 组加 7 个 decoder 组，共 14 组。

输入 checkpoint 有 192 个压缩 Linear、60 个 dense Linear。block 0 仅 q/k/v 压缩；block 1–8 全 dense；block 9–35 每个 block 七个 Linear 压缩。当前仅恢复已有压缩目标，不重新旋转或把 dense 转成 FWT。

## 编码坐标与 STE：实质差异

官方二值代理初值为 ±l2，l2 为变换域权重的行 RMS；scale=2*l2。默认 groupsize=-1 的训练前向对 weight+0.5 做 round STE 和 clamp，再乘尺度及做逆变换，二值边界接近 0，饱和区由外层 clamp 截断梯度。这里按源码描述：其训练表达式没有先除 scale。

当前代理直接初始化为 0 或 1，阈值 0.5，反向是没有饱和门控的 identity STE，硬前向复用 packed kernel。坐标、距阈值距离、decoder 雅可比、梯度门控均不同。复用 2e-5 不意味着相同更新尺度。

实际冒烟的代理最大移动约 2.002716e-5，初始距阈值为 0.5；14 个目标的 packed byte 变化均为 0。浮点代理移动不能当成编码已有效更新；也不能据此断言以后永远不会翻转，或用强制翻转制造通过结果。

已通过的 CUDA 梯度测试覆盖 packed 第一层线性算子与打包边界，并未完整证明整个非线性 decoder/block 的梯度和优化步骤正确。

## 学习率及超参数

| 项目 | 官方 CLI 默认 / README 示例 | 当前实际冒烟 | 判断 |
| --- | --- | --- | --- |
| 编码 lr | lw_lr=2e-5；取 min(lw_lr, FWT代理.std()/50) | min(2e-5, 原始 FP teacher 权重.std()/50) | std 对象不同 |
| 变换/decoder lr | CLI lt_lr=1e-3；示例 2e-4，用于变换矩阵 | 完整 decoder 统一 2e-4 | 借用示例数字，非官方 decoder 配置 |
| scale lr | lw_lr/5，即 4e-6 | 无独立 scale 组 | 表示不同 |
| a1/a2 lr | CLI la_lr=1e-3；示例 2e-3 | 无对应组 | 表示不同 |
| AdamW | wd=0；默认 betas=(0.9,0.999)、eps=1e-8 | 同数值；另固定 foreach=False | 核心设定一致 |
| 调度 | 独立 CosineAnnealingLR；下限初始 lr/20；step 后读 get_lr | LambdaLR 标准余弦；同下限 | 实际序列不同，见下节 |
| 裁剪/累积/预热 | 这条 Stage2 路线均未启用 | 均未启用 | 一致 |
| 数据 | RedPajama 完整语料随机文档/窗口 | 同源前 32 文档中取 4 个窗口 | 仅用于通路冒烟 |
| 样本/长度/batch/epochs | CLI 128/2048/1/10；示例 4096/2048/2/2 | 4/64/2/1，每 block 1 步 | 有意缩减 |
| 留出 | nsamples2 等于总样本时减去 1/32 | 4 条显式留 2 条 | 50% 非 1/32；官方 4//32=0 |
| seed | CLI 42 | 0 | 无必要的差异 |
| 模式 | 基础模型 eval，参数仍可学习 | student.train，teacher.eval | 不同；本 Qwen3 dropout=0，当前无该随机性 |
| 精度 | block FP32；默认 FP16 autocast + GradScaler，可选 BF16 | 代理/decoder FP32，冻结 LLM 部分 BF16；BF16 autocast，无 GradScaler | 不同 |
| attention / transformers | 默认 eager；requirements 5.9.0 | FlashAttention2；4.51.0 | 不同；本次未升级环境 |

README 仍用 epochs/nsamples 老参数，当前 CLI 已拆分 1/2，示例参数存在兼容问题；不能把示例意图说成当前可直接执行的默认配置。

示例 expc=20to8 是 2.5 payload bit/weight；本 VAE 压缩目标为 64 bit/32 weight，即 2 payload bit/weight。均不能忽略 decoder/变换/scale/dense/head 等存储而称为整个模型实测平均位宽。

全量脚本计划 4096 样本、长度 2048、batch 2、2 epochs，留 128、训 3968，即每 block 3968 次更新。尚未执行，所需 11 个原始 Arrow shard 尚未下载；准备好脚本不代表设计审查通过。

## 调度复算结果

官方在 dummy optimizer 的 scheduler.step 后读取 get_lr，赋给真实 optimizer。当前为标准余弦闭式公式。在 PyTorch 2.6.0 CPU 按实际调用复算：

| 4 步示例的更新序号 | 官方使用 lr，初始 2e-4 | 当前使用 lr |
| --- | --- | --- |
| 1 | 0.0002000000 | 0.0002000000 |
| 2 | 0.0001484251 | 0.0001721751 |
| 3 | 0.0000656497 | 0.0001050000 |
| 4 | 0.0000181497 | 0.0000378249 |

T=3968 时最大绝对差约 7.5155e-8，较小但不为零。T=1 时唯一更新的 lr 相同，故此差异不能解释本次一步冒烟的 MSE 上升。已纠正“当前公式是官方调度的闭式等价”这一注释。

## 已对齐的训练逻辑与边界

- align=1 语义：student 与 teacher 使用同一 FP teacher 前缀输入；对整个 block 最终 hidden state 做 MSE。
- 当前 block 全部已有压缩 Linear 联合训练，其余受保护参数冻结。
- 按 block 顺序恢复，其余模型在 CPU，仅当前 block 驻留 GPU。
- 采用 packed 硬前向；恢复 native 类型、保存、strict 重载后的输出要求逐元素相等，未放宽容差。

官方 target 收集与层间传播还受 epochs1>0 条件控制，不能简单设 epochs1=0 就得到独立 Stage B。当前独立 teacher 实现是必要的接入适配。

本框架没有直接运行官方 FWTLinear 或完整训练函数；upstream 保存的原文件仅作对照。保留已有 VAE decoder、无旋转 checkpoint 与原生格式，参数集合就无法与 FWT 完全相同。能严格复用的是外围训练逻辑，表示适配必须单独说明。

## 冒烟结果及 MSE

第二轮完整退出码 0，恢复 PASS，冻结 hash 一致，v6 strict 重载成功，两 block 重载输出 max_abs=0。最小同设置 A/B 评测 PASS，但各只做 1 道 PIQA，两边均答错；只证明评测通路，不证明质量相等。

CPU 比较 A/B 的全部 1551 个 state 条目：schema 相同，所有 packed code 与非 decoder 状态逐元素相同；只变化了 block 9/10 各 42 个 decoder tensor，共 84 个。每 block 有 88,928 个 decoder 参数、385,875,968 个浮点编码代理。

| MSE | block 9 | block 10 |
| --- | --- | --- |
| 全 4 条，更新前 | 0.07333381 | 0.07736746 |
| 全 4 条，更新后 | 0.80694306 | 0.16667563 |
| 训练 2 条，更新前首步 loss | 0.07141407 | 0.07627089 |
| 训练 2 条，更新后（由均值反推） | 1.17409277 | 0.17841533 |
| 留出 2 条，更新后 | 0.43979335 | 0.15493594 |

更新后训练 MSE 由 2×after_all−after_holdout 得到，两组样本/token 数相同；不是新增前向测量。训练集自身误差也上升，不能只解释为留出噪声或小样本。编码未变说明此次函数变化来自 decoder，但根因不能直接断言为 lr 偏大；梯度实现、精度、Adam 步长和 decoder 参数化仍需定位。

第一轮默认 fused decoder 预热与训练分步 BF16 舍入不同，导致失败。当前测量、重载与 A/B 统一采用原生 packed-u8 BF16 预热，严格输出检查已通过。其他工具默认 fused 路径仍可能不同，不能直接沿用 A 的历史评测指标。

## 修正优先级

1. 规定 VAE 编码代理坐标、尺度、硬判决与 STE 门控，明确与 FWT 的关系；0/1 代理直接套官方 lr 不能称等价。
2. 明确 decoder 参数组与学习率依据；不能把 lt_lr 自动扩展到整个 decoder，也不能无依据添加旋转/scale/a1/a2 改变现有表示。
3. 严格提取可复用的官方外围逻辑：参数组驱动的 optimizer/实际 scheduler、联合模式、teacher 输入、抽样与 seed。精度与 attention 的必要偏离逐项记录。
4. 验证完整 decoder/block 梯度和单步行为，再做有限 smoke；分开报告“通路/冻结/保存正确”和“恢复有效”。

本次完成的是审查与证据固定，尚未实施这些算法修改，也未新增调参或正式 A/B 实验。

## 文件与磁盘

输入 checkpoint 与新 B 各约 6.8 GiB，保留用于复现。CPU 核查输出、源码快照和官方 reference 为小文件，保留作为证据。旧 2026-09-22 冒烟目录约 46 GiB、120 GiB，本次未修改或删除；删除仍需明确授权。

收尾实测：远程剩余 959 GiB（使用率 98%）；整个实验代码目录含样本文本/reference 约 2.6 MiB；本次 CPU 审查结果目录 24 KiB、日志 8 KiB；第一轮失败目录 24 KiB。全部保留作为复现/审查依据，本次没有删除。与已运行的 code_snapshot 比较，block_train.py 仅注释变化。

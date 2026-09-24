> 本地历史交接记录（2026-09-23）；状态与授权均为当时语境。归档时未重新查询实验结果；后续操作遵循最新项目规范。

# distill_init 第二阶段恢复交接

## 2026-09-23 编码冻结修复

见 [编码代理修复](../experiments/liftquant_recovery/PROXY_FIX_20260923.md)。已实现训练期 decoder 敏感度尺度，p=s(b−0.5)，硬码round/clamp，STE含1/s及饱和门控；初始硬码和推理表示不变。14个真实decoder的完整数值对照通过，标量量化器回归证实旧配置不翻码、新尺度可跨阈值，但不能冒充真实模型质量结论。

两个相邻block、各16更新冒烟已完成，退出0，冻结/strict重载及最小A/B评测通过；全样本MSE为block9 0.07358347→0.06956137、block10 0.08025201→0.07706711，硬码翻转均0，下降来自decoder。进一步的单block9、384步、4×64token跨阈值检查已启动，tmux `lq_code_crossing_20260923_01`，输出 `.result/liftquant_recovery/code_crossing_smoke_20260923_01/`，最终结果待核实。正式实验未运行。

## 2026-09-23 历史：冻结修复之前的设计修正

用户要求确定合理一致范围并改善蒸馏效果后，新增 [一致范围与恢复设计](../experiments/liftquant_recovery/RECOVERY_DESIGN_20260923.md)，优先于下方历史记录。已修正 seed42、eval、官方 AdamW/实际调度、proxy std 的对象和 code VJP 的权重舍入。完整真实 decoder BF16 数值对照完成：前向与编码梯度逐元素一致，decoder梯度最大相对误差2.2433e-8。

数学上界表明旧0/1代理在3968步、lr上限2e-5下不可能跨过0.5阈值（精确算术）；需要重设坐标尺度，不能直接启动全量。该适配尚待实现。

已保存更新方向的纯前向诊断完成：block9 原更新量1/16时，train MSE=0.07021061、holdout=0.07424438，均低于原A；完整步量使二者上升。下一次有限smoke的decoder lr候选1.25e-5，尚未进行该配置的新训练。本轮诊断退出0，未写新checkpoint；所有后台诊断均已完成。

## 2026-09-23 后续核查更新（优先于下方历史状态）

第二轮完整任务已完成，退出码 0；恢复与最小 A/B 评测均 PASS。冻结 hash 不变，native v6 strict 重载通过，两个 block 重载输出 max_abs=0。A/B 各仅 1 道 PIQA，两边均答错，不能推断下游质量。

用户要求逐项核对官方设计后发现：当前 VAE 适配并非严格 LiftQuant 复现。编码代理坐标/STE、std 学习率上限对象、decoder 参数组映射、调度实际序列、seed、精度和模式等存在差异。见 [完整核查报告](../experiments/liftquant_recovery/DESIGN_AUDIT_20260923.md)。正式实验仍未启动，不能把通路 PASS 当成恢复有效。

新增 CPU 核查程序 design_audit.py 已完成，结果位于远程 `.result/liftquant_recovery/design_audit_20260923/audit.json`，退出码 0。比较全部 1551 个 state 条目，所有 packed codes 与非 decoder 状态完全相同，只有两个 block 共 84 个 decoder tensor 改变。训练集自身的更新后 MSE 也上升，不能只解释为留出集波动。此次没有新增 GPU 训练或修改算法，只纠正文档和注释。

## 之前的搭建与启动记录

用户最新要求优先于早期重新初始化要求：直接使用远程 `result/linear_output/distill_init`，只搭建与冒烟，不运行最终实验。

工作目录：`iaaccn74:/home/shaoyuantian/program/VAELLM`。
新增框架：`experiments/liftquant_recovery/`，详细文档 `CHECKPOINT_RECOVERY.md`。
入口 `recover.py`；当前启动脚本 `run_checkpoint_smoke.sh`；统一评测入口 `evaluate_pair.py`。
旧 `smoke.py` 和 `run_smoke.sh` 未使用，用户已有暂存改动未覆盖。未执行 Git 提交。

输入包含 192 个压缩 Linear 和 60 个 dense Linear；单 stage、无旋转。block 0 只有 q/k/v 压缩，block 9–35 全七个 Linear 压缩。保持实际范围。

实现覆盖：严格 v6 加载、FP teacher 前缀、逐 block 整体输出 MSE、全部 FP32 编码代理 + decoder 更新、原生 packed 硬前向、冻结状态 hash、完整 v6 导出与严格重载。优化器/调度参照 LiftQuant commit `72b3875c770e4579639931fed89dc95e4067edac` Stage2；VAE 参数组属于明确记录的适配。

当前 GPU 4 冒烟：两个相邻 block 9、10；4 条 64-token RedPajama 文本，2 条训练/2 条保留，每个 block 1 次更新；12 GiB allocator 上限。

已验证：

- 新模块语法检查通过。
- CUDA packed 编码梯度与 dense 数学参考比较通过，打包阈值跨越检查通过。
- 两个真实 block 的编码代理和 decoder 均有有限非零梯度并实际更新。
- 两个 block 的训练态硬前向与原生 packed 缓存推理逐元素一致。

首轮因默认 whole-decoder fused 推理和训练时分步 packed 解码的舍入差异而失败；未放宽断言。修复后，训练前后测量、重载与 A/B 评测统一使用原生 packed-u8 BF16 预热策略，不修改共享源码。默认其他推理工具的 fused 路径仍可能给出不同舍入结果。

第二轮最后一次有限检查时，两 block 已完成，任务仍在后台。之后的冻结检查、导出、重载和 A/B 各 1 道 PIQA 尚未读取最终结果，不应报告全部通过。

- tmux：`lq_distill_init_smoke_20260923_02`，启动 shell PID `1326598`。
- 输出：`.result/liftquant_recovery/from_distill_init_20260923_02/`。
- 日志：`.result/liftquant_recovery/from_distill_init_20260923_02.log`。
- 整体退出码：`.result/liftquant_recovery/from_distill_init_20260923_02.exit`。
- 恢复检查：输出中的 `summary.json`。
- 最小 A/B 评测：输出中的 `eval_smoke/summary.json`。

按项目规范，确认启动后不持续轮询或等待整个实验；用户通知完成后再检查上述结果。

小样本 MSE：block 9 从 0.07333381 到 0.8069431，block 10 从 0.07736746 到 0.1666756。只做 1 步的通路冒烟，不能说明恢复有效；没有为追求下降而调整配置或新增实验。

正式运行所需全量 RedPajama 尚未下载。入口显式支持官方镜像的 11 个原始 Arrow shard（共 930514 行，约 5.3 GB）；当前 smoke 仅保存同一数据集的 32 条真实文本，带 revision/source/hash，不能用于正式抽样。

收尾盘点：本次源码及小样本文本约 2 MiB，setup 记录约 1.7 MiB，首轮失败输出约 24 KiB，第二轮当时约 112 KiB（保存完整模型后预计另增约 6.8 GiB）。磁盘剩余约 966 GiB。旧冒烟目录 `smoke_20260922_104101` 约 46 GiB、`smoke_20260922_114204` 约 120 GiB，保留作为候选清理项；删除须另获明确授权。本次没有删除文件、安装依赖、覆盖源 checkpoint 或旧结果。

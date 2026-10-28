# SparseBit 选择性重锚隔离对照

本目录复用当前 bestabs 原生4071字段成品、L18 down 的两个完整码库和六个 decoder，以及既有32组严格 train-only M／QA／C1-R 目标。原始 blocked 稀疏残差保持并真实参与前向。不修改 CAT 快照或公共训练代码，不把翻码减少或局部 loss 下降解释为任务质量提升。

共同轨迹运行128次真实更新，按原32组顺序循环。风险筛选只使用65–128更新：逐位窗口翻转数 `F>=2` 且 `F/sum(abs(delta(p/s)))>=4`。在每个 bank 的64个预定 `abs(p/s)` 距离箱内，风险臂与非风险重锚臂均取可匹配的共同数量，记录覆盖率及实际距离分布。只有共同128点干预一次；后续继续至256。其余控制为原 joint、bit LR 减半、冻结当前码/score/bit Adam而继续 decoder。匹配为空时仅原轨迹延至256。

bit Adam 保留 `.9/.999`、epsilon `1e-8` 和原生 score 投影边界 `[-s/2,s/2]`，初始 LR `4e-5`；不裁剪 score 梯度。decoder AdamW 为 `.9/.95`、epsilon `1e-8`、WD0、LR `1e-6*(1-(step-1)/500)`，只对六个 decoder 做 clip1.5。隔离实例将轮长和稳定结束阈值都设为257，所有臂均禁用普通换轮。标度 s 固定，重锚保留两套优化器矩。

`training.jsonl` 每次真实更新记录原始16个损失、原生 packed 翻转计数、逐位自然 score 行程汇总、固定样本和短间隔 flipback。`F`、净奇偶和往返对数按64步窗口统计；`last_flip_step` 保留跨窗口历史，因此短间隔 flipback 可跨窗口。人为重锚单独记录行程并刷新观察快照，不计入自然行程或翻转。

入口在调用者激活 `bitvae` 后执行，统一单卡 `CUDA_VISIBLE_DEVICES`、`FLASH_ATTENTION_DETERMINISTIC=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，联网库设 offline。`run.py` 必须指定有限 `--max-seconds`，支持：

- `--phase short --output-root <fresh>`：原组0/1的两次完整更新，含32561-token prompt；第一步原生保存往返后继续第二步，核对真实码、score、decoder及两套 Adam。
- `--phase anchor --output-root <fresh>`：共同128锚点；有匹配风险则写 `anchor128.pt`，否则仅原轨迹到256。
- `--phase arm --arm baseline|risk|matched|half_lr|freeze_codes --anchor <anchor128.pt> --output-root <fresh>`：从同一真实优化器/RNG锚点继续至256。

128、192、256端点仅导出原生八个可变字段；192为 compact 留存，不额外完整评测。`export.py` 和 `inference.py verify` 分别完成 CPU 整模保存/严格重载与 fresh GPU 原生权重核验。`run_chain.py` 是一次有限的既定五臂与完整九项链，不自动重试、不等待抢卡；具体授权、资源及运行状态以唯一实验记录为准。

核心 CPU 合同测试：`python -B -m unittest experiments.sparse_bit_deoscillation.test_controller`。当前真实短验及导出证据位于 `result/qwen3_8b_mmlu65_20261004/qwen_deoscillation_20261026/validation/`；短验通过不等于质量收益。

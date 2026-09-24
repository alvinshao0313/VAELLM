> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`experiments/liftquant_recovery/CHECKPOINT_RECOVERY.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# 从 distill_init 开始的 LiftQuant 式第二阶段

当前实现与验证状态见 [PROXY_FIX_20260923.md](PROXY_FIX_20260923.md)。DESIGN_AUDIT_20260923.md 和 RECOVERY_DESIGN_20260923.md 为同日较早历史。入口 recover.py 直接加载用户指定的初始化 checkpoint；旧 smoke.py/run_smoke.sh 是早期初始化草稿，本次不使用。正式实验尚未运行，没有 Git 提交。

## 输入与范围

输入 result/linear_output/distill_init，Qwen3-8B native v6；192/252目标Linear压缩、60个为BF16 dense；block0仅q/k/v压缩、block1–8全dense、block9–35每层七个Linear压缩。单residual stage、单part、无旋转。每32个权重对应64个二值码，即压缩目标的code payload为2 bit/weight，不含decoder和其他dense/embedding/head，不能称整个模型2bit。

只联合训练当前block已有压缩Linear的全部连续编码代理与原decoder全部参数；无LoRA，不新增推理参数。原LLM norm/bias/head、dense与其他block冻结。输入checkpoint受保护，输出路径必须是新目录。初始化来源见原ckpt的TRAINING_PROVENANCE.json，本次没有重新训练Stage A。

## 与官方源码的对应及明确适配

参考 [LiftQuant官方源码](https://github.com/Heliulu/LiftQuant/tree/72b3875c770e4579639931fed89dc95e4067edac)，upstream/*.reference保留所读原文件。liftquant_optimizer.py 提取其AdamW与实际scheduler.step()+get_lr()序列，已通过序列对照。

- Stage2 ALL组、align=1：相同FP teacher前缀输入送入teacher/student block，对齐整个block最终hidden state，所有非padding token的MSE。每次只驻留一个block，teacher独立向下传播。
- seed42、block.eval；AdamW betas=(0.9,0.999)、eps=1e-8、wd=0；代理、decoder与Adam状态FP32；BF16原生packed解码。精度/attention实现依赖本项目环境，不能称逐位等价复现。
- VAE没有官方FWT和独立量化scale，不能照抄其变量含义。现有decoder敏感度固定尺度s，p0=s(b0−0.5)，硬码clamp(round(p/s+0.5),0,1)，STE含1/s与clamp门控。原初始化硬码不变；s只存在训练runtime和日志，不进入推理结构。
- code lr=min(2e-5, actual_proxy.std()/50)；相对码空间的有效步长约lr/s，此为明确的VAE适配，不是官方FWT的等价重参数化。
- 全decoder lr=1.25e-5，依据本项目历史方向诊断与有限多步冒烟选择，是待继续验证的配置，不能宣称最优或与官方独立scale lr等价。
- 各组使用官方实际余弦调用序列降到初始lr的1/20；正式配置4096样本留128、batch2、2epoch，每个压缩block3968次更新。正式配置尚未运行。

## 实现与检查

recover.py处理CLI、严格加载、逐层调度、冻结hash、v6保存与strict重载；block_train.py处理联合蒸馏及训练/留出MSE；proxy_coordinates.py定义敏感度坐标与位移预算；all_bits.py在当前实例上接入硬码前向与STE，导出前恢复native类型；recovery_runtime.py负责输入捕获、显存搬运和hash；recovery_data.py复用官方随机文档/窗口采样规则。verify_recovery.py、verify_decoder_gradients.py、verify_proxy_coordinates.py是长期数值回归入口。

每步记录真实packed bit变化、代理位移与阈值距离、饱和比例、梯度和decoder更新。有限非零梯度不等于硬码已经变化。训练硬前向、native前向、保存重载后的输出必须逐元素相等，冻结state必须hash不变。

VAELinear默认whole-decoder fused缓存与训练packed-u8分步BF16路径舍入不同。本实验统一用现有packed-u8路径预热缓存（prime_packed_cache），不改共享源码或schema、不放宽相等断言。其他默认缓存工具的旧A分数不能直接作为本次A/B基线。

## 有限冒烟

已激活bitvae的远程shell在项目根目录通过tmux/nohup调用。GPU4共享、PyTorch allocator上限12GiB；确认启动后不持续监控。

    bash experiments/liftquant_recovery/run_checkpoint_smoke.sh .result/liftquant_recovery/<新目录>

两相邻block9/10、4×64token、2训练/2留出、各16步，验证训练/冻结/保存链路；已完成PASS，但硬码翻转0。

    bash experiments/liftquant_recovery/run_code_crossing_smoke.sh .result/liftquant_recovery/<新目录>

单block9、同样4×64token、384步，直接检验自然梯度是否推动实际翻码。当前任务状态见PROXY_FIX_20260923.md，不以数学可达性代替真实翻码结果。

smoke文本是同一RedPajama镜像第一个Arrow shard的前32条真实文本，带source/revision/hash，仅用于机制冒烟。prepare_smoke_data.py复现此小样本，不等价于正式数据抽样。

## 同下游评测

    CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u -m experiments.liftquant_recovery.evaluate_pair --a result/linear_output/distill_init --b .result/liftquant_recovery/<恢复目录>/recovered_model --output .result/liftquant_recovery/<新评测目录> --tasks piqa --limit 1

evaluate_pair.py复用train_utils.eval_utils.run_lm_eval；A/B相同加载、任务、缓存策略；0-shot、batch1，无chat template和激活量化，单block驻留。A/B各1道PIQA只验证通路。正式八任务评测尚未运行。

## 输出与正式运行前提

manifest.json记录配置、输入指纹、源码hash和环境；calibration记录采样来源和token；block_metrics记录逐步指标；freeze_check和summary记录检查结果。完成后的探索模型若没有具体后续用途，按项目规则清理，保留最小复现与指标记录；不删除初始checkpoint或活动任务产物。

正式RedPajama镜像为11个原始Arrow shard和dataset_info.json，共930514行、约5.3GB。本次未下载，不能以smoke前缀代替：

    bash experiments/liftquant_recovery/run_checkpoint_recovery.sh <新输出目录> <RedPajama的11-shard目录>

此入口仅准备好，需真实机制验证、完整数据及正式运行授权，不自动执行。

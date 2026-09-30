# 将敏感度代理坐标接入 Sparse Bit（2026-09-30）

运行状态：实现与必要验证已完成；没有启动正式训练。结论：坐标、更新和保存恢复机制通过，Sparse Bit 下游收益尚未验证。

## 范围与设计

本次将逐层恢复中的固定 decoder 敏感度坐标接入服务器主仓库的 Sparse Bit。使用入口见 [E2E 模块说明](../../../../../compressed_e2e_fintuning/README.md#sparse-bit-敏感度代理坐标)，参考 [编码坐标与更新预算经验](../../../lessons/liftquant_recovery.md)。原来的 Sparse Bit 是 FP16 ±1 score、零阈值，auto LR 为 RMS-SGD 0.05 / Adam(W) 0.02，本来可以翻码；不能把旧 LiftQuant 0/1 代理与 2e-5 预算的冻结结论直接套用。

新增 `--bit_proxy_coordinates decoder_sensitivity`，必须显式给正数 `--bit_lr`。初始 `p=s(b−0.5)`，score/grad FP32，硬位保持 `p>=0`（零归1），STE 梯度除以 s，更新后限制在 ±s/2。保留现有稀疏采样、换轮、optimizer、损失、数据及可训练组件选择。原 unit 默认、旧 CLI 和 unit 精确恢复契约不变。

每 bank 最多256行、全 latent bit 单独翻转，使用抽取出的完整单 bank decoder eval 输出的 RMS 改变量作为 s。`measure_scale` 原样抽取到 `litebsq/bit_sensitivity.py` 与逐层恢复共用，AST 比较确认函数数学未变。标定只执行一次，保护模型、offload 驻留和 RNG；s 跨采样轮次固定。grouped decoder 抽成单 bank 后允许正常低精度算子顺序差异；BatchNorm 使用初始 eval running stats，未定义训练批统计敏感度。

E2E 按 AMP 或无 AMP 时的 embedding dtype 传入校准精度，避免 FP32 decoder 参数被误当作 BF16 输入的计算精度。新模式编码 VJP 使用实际前向精度舍入后的首层 decoder weight；原 unit 数值路径不变。

新模式 exact/coverage 内部状态版本2，保存固定 s 与 FP32 score；精确恢复先加载状态，不按训练后的 decoder 重标定。外部 sidecar 容器版本不变。最终模型仅保留提交后的原生 packed bits 和既有模型组件，无此功能新增推理开销。训练时每 active bit 的 score+grad 比旧模式多4字节，Adam两个FP32动量不增加。

## 验证环境与结果

服务器 iaaccn74，bitvae Python 3.11.13 / PyTorch 2.6.0+cu124 / A800 80GB，物理 GPU0，单进程、OMP_NUM_THREADS=2，PyTorch 分配器上限为设备容量5%。所有测试通过 nohup 后台执行。使用现有真实 Decoder/VAELinear/Trainer 的小形状回归，没有加载用户初始 checkpoint、修改数据协议或运行模型质量评估。LoRA 组合检查的 allocated 峰值为18,117,120字节；该值只代表该小测试，不能外推正式训练显存。

- CPU 配置/CLI：81项通过。
- CUDA packed 初始化/投影、dense STE、decoder 梯度、RMS-SGD/Adam/AdamW更新及 AMP：22项通过。
- checkpoint、streaming offload、grouped/非线性 decoder 标定、初始状态/RNG、纯 Sparse Bit 与 decoder+Sparse Bit Trainer、layout 和 LoRA+Sparse Bit：其余30个不重复回归项通过（其中3项layout为CPU计算）。总计133个不同回归项通过。
- 真实 Trainer 在第2步保存中断，第4步完成；敏感度模式用2e-5、梯度累积2、轮长3，覆盖换轮。decoder、score、采样、Adam、scheduler 完整状态与连续训练逐元素相等，保存的固定 s 恢复后保持一致。unit 原条件也通过。
- 独立小矩阵机制测试：冻结正交 decoder，BF16权重幅值1e-3、Adam LR2e-5、active ratio0.5、训练种子71，使用真实 packed 前后向，未注入梯度。敏感度模式第9步翻码，MSE从9.9890167e-7降至4.4395631e-7（约55.6%）；相同预算的unit在12步内无翻码、MSE不变。这只证明受控问题上的阈值可达性和更新有效，不证明大模型收益。

首轮校准参考测试有1项超出原设的2e-4相对容限：packed FP32首层使用TF32，dense参考使用FP32，观测相对差3.686e-4。依据TF32的10位尾数，将该跨算子参考容限设为2e-3后通过；不改训练算法，不放宽 packed bits、RNG、exact-resume 等严格相等契约。其余受影响精度/Trainer路径复验通过。

## 复现与证据

[核心日志](../../../../../.result/sparse_bit_proxy_coordinates_20260930/validation.log)保留命令、通过摘要、首次失败和修正证据；[配置与文件哈希](../../../../../.result/sparse_bit_proxy_coordinates_20260930/validation.json)列出测试文件、运行环境、Git基础版本及本次未提交文件SHA256。首次实现与验证时未创建Git提交。重跑时在bitvae中按JSON中的文件列表调用 `python -m pytest -q`；CUDA列表设 `CUDA_VISIBLE_DEVICES=0`，本次真实调用还通过 `torch.cuda.set_per_process_memory_fraction(0.05,0)` 限制分配器。

原先8张卡的任务从独立源码副本 `/home/shaoyuantian/program/VAELLM-e2e-0910-20260924` 启动，本次没有修改其代码或配置。用户初始ckpt及正式结果未改动。

## 决定与收尾

交付可选新坐标模式，原默认配置保留；2e-5仅作机制验证起点，不宣称最优。Sparse Bit 每轮会重选active集合并重置代理/动量，轮长影响翻码预算；正式效果需同预算对照，不能搬用逐层恢复的质量结论。本次未跑全模型、下游评测、多卡通信或性能基准。

保留实现、长期回归、核心日志、配置/哈希及本文；一次性编辑/收集日志归并后清理，本轮临时小模型和优化器断点已无复用用途，核对路径后清理。经验增量已归并到原编码坐标主题。

本轮主代理核实并清理 `/tmp/pytest-of-shaoyuantian/pytest-237`、`pytest-238` 的一次性测试权重/状态及已归并日志，共释放 691,561 字节；正式结果、初始ckpt与保护记录均保留。


## 用户要求的再次复核（2026-09-30）

23个相关源码/测试文件的SHA256与首次验证一致，未发现后续覆盖。独立复查数学/计算核和CLI/Trainer/精确恢复/导出两条链路，没有发现需要修改的问题；没有改动训练源码。

重新执行敏感度与启用边界相关测试，`27 passed, 36 deselected`，9.49秒、退出0。覆盖CLI显式LR与模式契约、grouped及BF16前向/FP32参数的梯度、三种optimizer、实际翻码、换轮、streaming offload、LoRA组合和真实Trainer精确恢复。受控测试再次在第9步翻码，MSE为9.9890167e-7→4.4395631e-7；unit对应12步未翻码。GPU0测试allocated峰值18,136,064字节，不作为正式训练资源估计。[此次核心日志与执行参数](../../../../../.result/sparse_bit_proxy_coordinates_20260930/recheck_01/validation.log)

使用边界再次确认：仅设置坐标参数不会开启编码训练，还需`train_mode`包含`sparse_bit`、显式正数`bit_lr`。复核时运行中的8个任务均在独立目录`/home/shaoyuantian/program/VAELLM-e2e-0910-20260924`使用`--train_mode lora`，没有启用此功能。未更改或重启既有任务。相关测试临时权重无后续用途，清理后只保留本次核心日志；已有下游收益限制不变，本次无新增算法经验。

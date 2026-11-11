<!-- lesson-redirect -->

# 显存与精度：分清驻留对象、数值存储和验证范围

本专题已按“一级分类 → 二级专题 → 经验条目”拆分；以下保留旧标题锚点和完整去向，原经验正文只在目标专题维护。新总结请写入[经验分类入口](../README.md)对应专题。若并行任务仍向本文件补写，须先保留增量、核对来源后归并，不能覆盖或直接删除。

<a id="按条件选择动作"></a>

- **按条件选择动作** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--按条件选择动作)

<a id="下次怎么做"></a>

- **下次怎么做** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--下次怎么做)

<a id="长文本生命周期的资源边界"></a>

- **长文本生命周期的资源边界** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--长文本生命周期的资源边界)

<a id="动态padding的长度上限与实际批形状"></a>

- **动态padding的长度上限与实际批形状** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--动态padding的长度上限与实际批形状)

<a id="问题与证据"></a>

- **问题与证据** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--问题与证据)

<a id="固定κ复用与梯度数值身份"></a>

- **固定κ复用与梯度数值身份** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--固定κ复用与梯度数值身份)

<a id="全mlp非保护权重学习的资源与部署初态"></a>

- **全MLP非保护权重学习的资源与部署初态** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--全mlp非保护权重学习的资源与部署初态)

<a id="只读探针autograd与强引用释放"></a>

- **只读探针、autograd与强引用释放** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--只读探针autograd与强引用释放)

<a id="主缓存稀疏辅助缓存与恢复stamp"></a>

- **主缓存、稀疏辅助缓存与恢复stamp** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--主缓存稀疏辅助缓存与恢复stamp)

<a id="gpu身份表示与effective-dtype"></a>

- **GPU身份表示与effective dtype** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--gpu身份表示与effective-dtype)

<a id="原缓存外部引用与无必要重复预热"></a>

- **原缓存外部引用与无必要重复预热** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--原缓存外部引用与无必要重复预热)

<a id="teacher驻留搬运成本与真实峰值"></a>

- **teacher驻留、搬运成本与真实峰值** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--teacher驻留搬运成本与真实峰值)

<a id="rope原始精度与norm部署跨格预算"></a>

- **RoPE原始精度与norm部署跨格预算** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--rope原始精度与norm部署跨格预算)

<a id="单模块精度与历史mid-eval证据边界"></a>

- **单模块精度与历史mid-eval证据边界** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--单模块精度与历史mid-eval证据边界)

<a id="新入口的allocator与cpu线程"></a>

- **新入口的allocator与CPU线程** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--新入口的allocator与cpu线程)

<a id="cpu-only构造与gpu消费者的初始化边界"></a>

- **CPU-only构造与GPU消费者的初始化边界** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--cpu-only构造与gpu消费者的初始化边界)

<a id="精度切换后的缓存初始化和同步d2h观察"></a>

- **精度切换后的缓存初始化和同步D2H观察** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--精度切换后的缓存初始化和同步d2h观察)

<a id="合批加速与缓存精度的数值对照"></a>

- **合批加速与缓存精度的数值对照** → [数值与运行性能经验 / 并行、算子、评测与吞吐](runtime/performance.md#memory-precision--合批加速与缓存精度的数值对照)

<a id="独立目标分组拟合的效率边界"></a>

- **独立目标分组拟合的效率边界** → [数值与运行性能经验 / 并行、算子、评测与吞吐](runtime/performance.md#memory-precision--独立目标分组拟合的效率边界)

<a id="实际激活dtypemaster存储与部署缓存"></a>

- **实际激活dtype、master存储与部署缓存** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--实际激活dtypemaster存储与部署缓存)

<a id="pure-cat全fp32与独立down完整结果"></a>

- **pure CAT全FP32与独立down完整结果** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--pure-cat全fp32与独立down完整结果)

<a id="评测阶段耗时与离线数据访问"></a>

- **评测阶段耗时与离线数据访问** → [数值与运行性能经验 / 并行、算子、评测与吞吐](runtime/performance.md#memory-precision--评测阶段耗时与离线数据访问)

<a id="opd往返完整词表峰值与triton设备"></a>

- **OPD往返、完整词表峰值与Triton设备** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--opd往返完整词表峰值与triton设备)

<a id="边界"></a>

- **边界** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--边界)

<a id="长度与cpu卸载的正式终态边界"></a>

- **长度与CPU卸载的正式终态边界** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--长度与cpu卸载的正式终态边界)

<a id="预热峰训练驻留与卸载测量"></a>

- **预热峰、训练驻留与卸载测量** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--预热峰训练驻留与卸载测量)

<a id="kernel数值检查的dtype和形状范围"></a>

- **kernel数值检查的dtype和形状范围** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--kernel数值检查的dtype和形状范围)

<a id="完整启动环境与碎片化归因边界"></a>

- **完整启动环境与碎片化归因边界** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--完整启动环境与碎片化归因边界)

<a id="同bf16计算下的fp32叶参数对照"></a>

- **同BF16计算下的FP32叶参数对照** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#memory-precision--同bf16计算下的fp32叶参数对照)

<a id="cuda-allocator初始化与测量顺序"></a>

- **CUDA allocator初始化与测量顺序** → [数值与运行性能经验 / 显存、卸载与缓存生命周期](runtime/memory.md#memory-precision--cuda-allocator初始化与测量顺序)

<!-- /lesson-redirect -->

# 自适应残差码流的整模型恢复桥接草稿

用户于10月29日要求先说明编解码器高效并行实现方案；已完成说明及[真实分组拟合配对](../adaptive_grouped_fit_20261029/README.md)。本桥接现已完整通过真实GPU短测，仅证明以下单层两更新与保存契约。效率设计唯一正文见[高效并行实现方案](../adaptive_vector_rvq/README.md#10月29日高效并行实现方案与讨论边界)。

拟验证同一原始HF Qwen3-8B内的真实L18 down，将现成`fixed2＋INT8／abs005`或`weighted_dynamic3`紧凑文件装入，使用既有auxiliary_train M32第0／1组及原来一次混合的教师／gold目标，只做两次decoder更新。L18只是已有浅中深证据中的代表层，不是最终选层规则。其它HF权重冻结；这不是全压≤2.6bit模型、完整M/QA/R恢复或九项质量实验。

`consumer.py`直接读已有两种紧凑文件，保留实际短码流、路由与冻结保护／旋转／稀疏字段，持久FP32 decoder参数通过既有精度运行层进行BF16计算；仅活动行的阶段输出按BF16顺序相加。该版本逐chunk解码用于核正确性，尚无吞吐或可扩展性结论。公共VAELinear、group8、v6格式及旧原型不变。

`inputs.py`复用已验证数据加载及损失；`run_bridge.py`检查同文件初态、真实8B反传／更新、缓存失效、冻结状态、实际紧凑保存及fresh模型重载。稀疏对照源文件位于`/data/star/VAELLM/adaptive_sparse_layers_20261028/layer_18/sparse_abs005.avrqsp`；动态臂直接使用本次并行拟合产物`/data/star/VAELLM/adaptive_grouped_fit_20261029/grouped/endpoint_18/weighted_dynamic3.avrvqz`，避免以后重复验证并行产物接线。原模型为`/home/star/models/Qwen/Qwen3-8B`。研究产物依赖完整HF骨干，不能仅以一个约20MB的层文件计成整模型存储。

首轮`paired_M2`在训练前核HF/full-F骨干时，直接比较CUDA实际张量和CPU载体触发device mismatch，尚未构造消费者或执行优化。已改为实际147非投影张量搬CPU、保持dtype后严格比较，没有改权重、目标或放宽相等。旧`run.log`及失败报告保留。

修正版在GPU2、caller bitvae、项目根目录的tmux `adaptive_bridge_GPU2_fix_20261029`中执行，shell PID350266，输出为`/data/star/VAELLM/adaptive_native_bridge_20261029/paired_M2_runtime_fix`，核心日志`runtime_fix.log`、外层退出`runtime_fix.exit`。首次失败未进入消费者／训练，因此新独立运行同时采用上面的真实grouped动态输入，没有重启或改写活跃任务。

## 10月29日实际终态

真实两臂均完成2次M32整模型更新，根报告`verified_bridge_only`／exit0，实际145.8924秒。fixed2_sparse有12个实际decoder叶，dynamic3有18个；梯度有限且非零，FP32 master真实更新，两次更新后的cache_builds均由1到2。其余HF权重及短码、长度、保护、旋转和稀疏字段冻结检查通过。

同文件初态BF16权重对照完全一致；更新后部署权重分别有25,244,366／25,440,925个元素变化。实际保存文件分别20,373,157／20,372,850B，均保持各自源文件大小；fresh紧凑文件的冻结字段／当前BF16 decoder逐项核对通过，再加载一份完整原HF模型并装入新层，同输入输出完全一致。结束后又核两文件实际大小与SHA，GPU2已释放。

唯一原始证据为输出根`paired_M2_runtime_fix/report.json`及各臂`report.json`；包括来源、数据／目标、实际命令、代码SHA、冻结摘要、每次更新与保存信息。共享进程观测的CUDA allocated峰值约23.59GB、reserved约26.89GB，不用累积峰值对两臂作显存排名。两步使用不同M组，不能把两步loss数值差当作精度改善。

这次闭合了“并行拟合生成的真实短码流→完整8B前后向→decoder更新→保存／fresh重载”链路。仍只有一个压缩down，其余HF投影是dense，不证明全类峰值、正式多域恢复、全模型位宽或九项达标。用户随后要求充分研究向量划分与编解码器容量；下一步先做同预算结构设计，不据此直接启动36个down正式重压。

## 10月29日：完成后最小产物收尾

已确认成功桥接完整结束、无本组活动进程或候选文件打开FD。保留`paired_M2_runtime_fix/dynamic3/layer.avrvqz`作为后续compact consumer规格扩展时，D32／H128真实decoder恢复后保存／重载的回归输入：20,372,850 B，SHA `579318687cf25069c8cc480144f5e16623f6868f5c4f61818ae7ad6bac7d601a`。其grouped L18初态由[分组拟合正文](../adaptive_grouped_fit_20261029/README.md#10月29日完成后最小产物收尾)保留；原HF骨干和旧sparse／dynamic bank等输入均保护。

训练后`paired_M2_runtime_fix/fixed2_sparse/layer.avrqsp`没有后继续训、复评或交付用途，已删除；它现不可再加载，原始稀疏输入、该臂原报告和全部数值／冻结／保存契约证据仍保留。唯一清理增量位于既有`paired_M2_runtime_fix/report.json.cleanup`，记录该精确路径及SHA、原大小／inode、原因和实际allocated释放。`runtime_fix.exit=0`及原首轮`run.exit=1`已归入该cleanup后删除散文件；原失败`run.log`、`paired_M2/report.json`、成功核心`runtime_fix.log`、根／两臂报告、代码版本及配置证据保留，没有改写失败事实。

本组共删除3文件、逻辑20,373,161 B、实际allocated释放 **20,381,696 B**。删除后保留dynamic端点和必要证据SHA均保持。与groupedfit同次收尾合计29文件、实际allocated释放 **84,000,768 B**，按各文件实际blocks及最后硬链接计，不采用共享磁盘空闲变化；没有修改代码、运行训练或新增精度结论。

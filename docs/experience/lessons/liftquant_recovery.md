<!-- lesson-redirect -->

# 编码恢复：先验证阈值可达性和真实数值路径

本专题已按“一级分类 → 二级专题 → 经验条目”拆分；以下保留旧标题锚点和完整去向，原经验正文只在目标专题维护。新总结请写入[经验分类入口](../README.md)对应专题。若并行任务仍向本文件补写，须先保留增量、核对来源后归并，不能覆盖或直接删除。

<a id="按条件选择动作"></a>

- **按条件选择动作** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--按条件选择动作)

<a id="下次怎么做"></a>

- **下次怎么做** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--下次怎么做)

<a id="问题与证据"></a>

- **问题与证据** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--问题与证据)

<a id="适配后选码排名反转不等于选择规则更好2026-10-09核实"></a>

- **适配后选码：排名反转不等于选择规则更好（2026-10-09核实）** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--适配后选码排名反转不等于选择规则更好2026-10-09核实)

<a id="边界"></a>

- **边界** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--边界)

<a id="384步真实翻码机制可用不等于恢复有效"></a>

- **384步真实翻码：机制可用不等于恢复有效** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--384步真实翻码机制可用不等于恢复有效)

<a id="同预算对照能区分翻码收益与单纯拟合"></a>

- **同预算对照能区分翻码收益与单纯拟合** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--同预算对照能区分翻码收益与单纯拟合)

<a id="工程提速先锁定数值轨迹2026-09-23"></a>

- **工程提速先锁定数值轨迹（2026-09-23）** → [数值与运行性能经验 / 并行、算子、评测与吞吐](runtime/performance.md#liftquant-recovery--工程提速先锁定数值轨迹2026-09-23)

<a id="全驻留评测可以保留原-packed-计算路径2026-09-24"></a>

- **全驻留评测可以保留原 packed 计算路径（2026-09-24）** → [数值与运行性能经验 / 并行、算子、评测与吞吐](runtime/performance.md#liftquant-recovery--全驻留评测可以保留原-packed-计算路径2026-09-24)

<a id="完整逐层恢复的下游收益2026-09-24完成2026-09-30核实"></a>

- **完整逐层恢复的下游收益（2026-09-24完成，2026-09-30核实）** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#liftquant-recovery--完整逐层恢复的下游收益2026-09-24完成2026-09-30核实)

<a id="迁移到-sparse-bit-时同时处理坐标轮次与恢复2026-09-30"></a>

- **迁移到 Sparse Bit 时同时处理坐标、轮次与恢复（2026-09-30）** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--迁移到-sparse-bit-时同时处理坐标轮次与恢复2026-09-30)

<a id="e2e覆盖率对照区分更新坐标和梯度裁剪"></a>

- **E2E覆盖率对照：区分更新坐标和梯度裁剪** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--e2e覆盖率对照区分更新坐标和梯度裁剪)

<a id="已恢复模型追加decoder蒸馏不保证继续提高均分"></a>

- **已恢复模型追加decoder蒸馏不保证继续提高均分** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#liftquant-recovery--已恢复模型追加decoder蒸馏不保证继续提高均分)

<a id="覆盖率完整曲线与真实父源"></a>

- **覆盖率完整曲线与真实父源** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#liftquant-recovery--覆盖率完整曲线与真实父源)

<a id="教师rope数值路径与匹配控制"></a>

- **教师RoPE数值路径与匹配控制** → [数值与运行性能经验 / 数值精度、梯度与舍入路径](runtime/precision.md#liftquant-recovery--教师rope数值路径与匹配控制)

<a id="固定码拟合ste与真实离散邻居"></a>

- **固定码拟合、STE与真实离散邻居** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--固定码拟合ste与真实离散邻居)

<a id="旋转支持集bf16评分与输出接受"></a>

- **旋转支持集、BF16评分与输出接受** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--旋转支持集bf16评分与输出接受)

<a id="选码后的decoder适配与跨层反例"></a>

- **选码后的decoder适配与跨层反例** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--选码后的decoder适配与跨层反例)

<a id="条件mlp目标与完整已恢复背景"></a>

- **条件MLP目标与完整已恢复背景** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--条件mlp目标与完整已恢复背景)

<a id="真实局部接受与模型尾部目标"></a>

- **真实局部接受与模型尾部目标** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--真实局部接受与模型尾部目标)

<a id="提案目标搜索成本与校准分布"></a>

- **提案目标、搜索成本与校准分布** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--提案目标搜索成本与校准分布)

<a id="微小改码的执行路径与匹配更新成本"></a>

- **微小改码的执行路径与匹配更新成本** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--微小改码的执行路径与匹配更新成本)

<a id="当前父源的单层全mlp与跨模型条件空间"></a>

- **当前父源的单层、全MLP与跨模型条件空间** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--当前父源的单层全mlp与跨模型条件空间)

<a id="真实阈值预算与输入方向必须分别测量2026-10-03"></a>

- **真实阈值预算与输入方向必须分别测量（2026-10-03）** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--真实阈值预算与输入方向必须分别测量2026-10-03)

<a id="共享gw的阈值保护几何与联合投影"></a>

- **共享gW的阈值、保护几何与联合投影** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--共享gw的阈值保护几何与联合投影)

<a id="换轮对照须先排除阈值前轨迹分歧2026-10-04"></a>

- **换轮对照须先排除阈值前轨迹分歧（2026-10-04）** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--换轮对照须先排除阈值前轨迹分歧2026-10-04)

<a id="码位联合优化的正向例子要绑定配方与控制2026-10-04"></a>

- **码位联合优化的正向例子要绑定配方与控制（2026-10-04）** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--码位联合优化的正向例子要绑定配方与控制2026-10-04)

<a id="原生长程正例与中途最佳"></a>

- **原生长程正例与中途最佳** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--原生长程正例与中途最佳)

<a id="更高起点的bit-lr零翻码与真实更新调用"></a>

- **更高起点的bit LR、零翻码与真实更新调用** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--更高起点的bit-lr零翻码与真实更新调用)

<a id="阈值前轨迹源码身份与原始记录"></a>

- **阈值前轨迹、源码身份与原始记录** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--阈值前轨迹源码身份与原始记录)

<a id="完整score统计与adamrms步幅对照"></a>

- **完整score统计与Adam、RMS步幅对照** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--完整score统计与adamrms步幅对照)

<a id="校准扩量decoder适配与同最终目标配对"></a>

- **校准扩量、decoder适配与同最终目标配对** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--校准扩量decoder适配与同最终目标配对)

<a id="ldl反馈与跨stage有限邻域"></a>

- **LDL反馈与跨stage有限邻域** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--ldl反馈与跨stage有限邻域)

<a id="枚举空间真实kl接受与固定权重精度参照"></a>

- **枚举空间、真实KL接受与固定权重精度参照** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--枚举空间真实kl接受与固定权重精度参照)

<a id="分散支持混合校准和早期decoder交互"></a>

- **分散支持、混合校准和早期decoder交互** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--分散支持混合校准和早期decoder交互)

<a id="递归encoder重编码与bsq辅助项"></a>

- **递归encoder重编码与BSQ辅助项** → [压缩后恢复微调经验 / 离散候选、输出目标与局部选码](post_recovery/code_search.md#liftquant-recovery--递归encoder重编码与bsq辅助项)

<a id="共享任务梯度的主码投影已有负结果与新对照边界"></a>

- **共享任务梯度的主码投影：已有负结果与新对照边界** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--共享任务梯度的主码投影已有负结果与新对照边界)

<a id="持续adam矩与绝对权重残差"></a>

- **持续Adam矩与绝对权重残差** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--持续adam矩与绝对权重残差)

<a id="同预算完整九项与长预算保留代价"></a>

- **同预算完整九项与长预算保留代价** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#liftquant-recovery--同预算完整九项与长预算保留代价)

<a id="扩层覆盖与保留权重的条件效果"></a>

- **扩层、覆盖与保留权重的条件效果** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#liftquant-recovery--扩层覆盖与保留权重的条件效果)

<a id="u学习率d码端点与有效提交"></a>

- **U学习率、D×码端点与有效提交** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#liftquant-recovery--u学习率d码端点与有效提交)

<a id="定位范围的可学性与参数化预算"></a>

- **定位范围的可学性与参数化预算** → [压缩后恢复微调经验 / 恢复目标、优化器与训练预算](post_recovery/optimization.md#liftquant-recovery--定位范围的可学性与参数化预算)

<a id="知识来源保留教师和覆盖的匹配对照"></a>

- **知识来源、保留教师和覆盖的匹配对照** → [数据与监督经验 / 数据来源、覆盖、混合与实际曝光](data/sources.md#liftquant-recovery--知识来源保留教师和覆盖的匹配对照)

<a id="保护整数协同原生投影与完整九项的边界"></a>

- **保护整数协同：原生投影与完整九项的边界** → [压缩后恢复微调经验 / 恢复初态、参数范围与保持](post_recovery/parameters.md#liftquant-recovery--保护整数协同原生投影与完整九项的边界)

<a id="码位往返与选择性重锚已有证据及待验证边界"></a>

- **码位往返与选择性重锚：已有证据及待验证边界** → [压缩后恢复微调经验 / 离散码优化、阈值与更新坐标](post_recovery/codes.md#liftquant-recovery--码位往返与选择性重锚已有证据及待验证边界)

<!-- /lesson-redirect -->

<!-- lesson-redirect -->

# Checkpoint：先持久化可恢复状态，再导出并独立重载

本专题已按“一级分类 → 二级专题 → 经验条目”拆分；以下保留旧标题锚点和完整去向，原经验正文只在目标专题维护。新总结请写入[经验分类入口](../README.md)对应专题。若并行任务仍向本文件补写，须先保留增量、核对来源后归并，不能覆盖或直接删除。

<a id="按条件选择动作"></a>

- **按条件选择动作** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--按条件选择动作)

<a id="下次怎么做"></a>

- **下次怎么做** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--下次怎么做)

<a id="问题与证据"></a>

- **问题与证据** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--问题与证据)

<a id="训练包装compact格式与子集状态的导出"></a>

- **训练包装、compact格式与子集状态的导出** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--训练包装compact格式与子集状态的导出)

<a id="compact依赖闭合与native前缀转换"></a>

- **compact依赖闭合与native前缀转换** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--compact依赖闭合与native前缀转换)

<a id="参数与优化器状态精度不同的重载"></a>

- **参数与优化器状态精度不同的重载** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--参数与优化器状态精度不同的重载)

<a id="子集解冻元数据及容器契约"></a>

- **子集解冻、元数据及容器契约** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--子集解冻元数据及容器契约)

<a id="边界"></a>

- **边界** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--边界)

<a id="层边界恢复须重建运行缓存2026-09-23"></a>

- **层边界恢复须重建运行缓存（2026-09-23）** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--层边界恢复须重建运行缓存2026-09-23)

<a id="e2e中途评估改变rng时的阶段恢复2026-09-24"></a>

- **E2E中途评估改变RNG时的阶段恢复（2026-09-24）** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--e2e中途评估改变rng时的阶段恢复2026-09-24)

<a id="顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾"></a>

- **顺序搜索保留中途最佳，部署排名须独立重载（2026-10-01收尾）** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾)

<a id="结构转换与解码核切换分开验证2026-09-24"></a>

- **结构转换与解码核切换分开验证（2026-09-24）** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--结构转换与解码核切换分开验证2026-09-24)

<a id="源码身份拒绝恢复时保留原运行版本2026-09-24"></a>

- **源码身份拒绝恢复时保留原运行版本（2026-09-24）** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--源码身份拒绝恢复时保留原运行版本2026-09-24)

<a id="源身份的内容校验与分配空间观测"></a>

- **源身份的内容校验与分配空间观测** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--源身份的内容校验与分配空间观测)

<a id="信号中断与源码恢复分别取证"></a>

- **信号中断与源码恢复分别取证** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--信号中断与源码恢复分别取证)

<a id="全码位多卡断点的独立导出2026-10-02"></a>

- **全码位多卡断点的独立导出（2026-10-02）** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--全码位多卡断点的独立导出2026-10-02)

<a id="cpu构造原权重读取与真实导入上下文"></a>

- **CPU构造、原权重读取与真实导入上下文** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--cpu构造原权重读取与真实导入上下文)

<a id="真实翻码后的独导与回调资源预约"></a>

- **真实翻码后的独导与回调资源预约** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--真实翻码后的独导与回调资源预约)

<a id="headnorm及配置运行字段的导出边界"></a>

- **head、norm及配置运行字段的导出边界** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--headnorm及配置运行字段的导出边界)

<a id="残块迁移扩宽与恢复成品验证"></a>

- **残块迁移、扩宽与恢复成品验证** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--残块迁移扩宽与恢复成品验证)

<a id="独立导出须复现配置规范化顺序2026-10-17"></a>

- **独立导出须复现配置规范化顺序（2026-10-17）** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--独立导出须复现配置规范化顺序2026-10-17)

<a id="持久冻结状态与可重建码缓存2026-10-18"></a>

- **持久冻结状态与可重建码缓存（2026-10-18）** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--持久冻结状态与可重建码缓存2026-10-18)

<a id="cpu账本审计与实际侧车状态"></a>

- **CPU账本审计与实际侧车状态** → [保存与复现经验 / 状态持久化、断点与精确续训](lifecycle/checkpoint.md#checkpoint-lifecycle--cpu账本审计与实际侧车状态)

<a id="生成参数须检查最终有效配置2026-10-04"></a>

- **生成参数须检查最终有效配置（2026-10-04）** → [保存与复现经验 / 结构转换、打包、导出与独立重载](lifecycle/export.md#checkpoint-lifecycle--生成参数须检查最终有效配置2026-10-04)

<!-- /lesson-redirect -->

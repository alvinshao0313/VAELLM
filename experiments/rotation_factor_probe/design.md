# 顺序 CAT 的类别旋转块大小：兼容设计

2026-10-07，状态：仅设计，未实现、未准备训练、未运行 GPU。类别旋转配置是尚未提供的新接口；待完整 LongBench 证据支持投入后，再向用户提交正式支持请求。本设计不改变现有作业或其源码。

## 目的与现有证据边界

目标是从原 HF 开始，在同一条原生 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 轨迹中，让 Q/K 用完整维度双侧旋转（block=0），其余类别用 block32；保留每类后恢复、融合剩余 dense 权重、后续类别读取当前权重和校准状态的既有顺序。

当前 Qwen K38 成品 `4a84b1ba-aede-4a81-a06d-b827f20583ca` 位于 `result/qwen3_8b_mmlu65_20261004/qwen_qk_rate/k38_full/final_model`：72 个 Q/K 来自 fresh HF donor，另180个投影来自已恢复的 f927。其合法逻辑位宽及检索表现证明该组件组合可以运行，不能证明改变 Q/K 后重新压缩其余五类、逐类恢复的效果。完整 LongBench 尚以实际终态汇总为准，本设计不据两检索提前声称九项收益。

参考[压缩选择经验](../../docs/experience/lessons/compression_choices.md)、[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)及[原旋转实现记录](../../docs/experience/records/docs/exp_results/weight_rotation_preprocess.md)。旧 donor 匹配对照可复用数值/几何支持，不能复用为整条新前缀训练的质量证明。

## 源码定位与真正缺口

以下位置为本次只读时的 public 路径；S 中这条 CAT 配置链语义相同，H252 选择器另有明确差别。

| 位置 | 当前实际行为 | 拟修改范围 |
| --- | --- | --- |
| `train_utils/config/cli.py:670,892,1270` | `weight_rotation_block_size` 是全局整数；`resolve_category_config` 对所有类别复制 `core_template` 的同一个值 | 新可选覆盖表，解析和逐类别解析 |
| `train_utils/config/configs.py:557,591` | 每个 `VAECoreConfig` 已能表达非负整数块大小，0代表完整维度 | 不需新增 core 类型或改旧字段含义 |
| `train_utils/cat_category_runtime.py:37,57` | `ResolvedCategoryRuntimeConfig` 未携带旋转；从完整 compression 配置提取时丢失这两个字段 | 让 resolved runtime 明确包含 mode/block |
| `train_utils/cat_train_pipeline.py:1312,1411` | 训练直接读取 `vae_args` 的全局旋转，而非本类 runtime | 两处改为读取同一个 resolved runtime，不按组/类别临时改全局 namespace |
| `train_utils/cat_runtime_state_v6.py:105,124,264` | identity 全量比较；只把旋转全局值写入 `vae_shared`，类别配置尚无旋转 | 新配置的身份必须包含每个类别最终 mode/block，旧配置保持原身份 |
| `train_utils/cat_train_runtime.py:290` | normalized 已自动输出 canonical 和全部 resolved category dataclass | 复用自动输出，不另建旁路配置文件 |
| `train_utils/checkpoint_v6.py:314,1119,1466,1498` | 每个 VAELinear 分别校验、保存和重建自己的 rotation spec/signs | 已支持混合类别，无需改 v6 格式或加载器 |

最小生产修改预计只有 `config/cli.py`、`cat_category_runtime.py`、`cat_train_pipeline.py`、`cat_runtime_state_v6.py` 四处。`cat_runtime_adapter.py:43` 已携带原 `CatCLIConfig`，现有全局 `vae_args` 保留为旧接口/日志字段，不再作为类别训练的旋转决策源。`tools/cat_train.py`、VAE 编码/解码算法、after-category 蒸馏和剩余类融合均无需重写。

## 单个新增参数及明确语义

拟新增（名称尚未实现）：

```text
--weight_rotation two_sided
--weight_rotation_block_size 32
--category_weight_rotation_block_size 'cat:q_proj=0,cat:k_proj=0'
```

- 原 `--weight_rotation` 与 `--weight_rotation_block_size` 的类型、默认值和旧命令完全保留。未提供新参数时，所有类别仍采用旧全局设置。
- 新参数为可选 `OverrideTable[int]`，默认 `None`；复用 `OverrideSpec`、`parse_int_text(min_value=0)`、`parse_override_table` 和 `validate_category_keys`，只接受 `cat:` 选择器。未列类别回退到旧全局整数。
- 不再增加一套 `default=`：默认由现有全局参数唯一决定；不支持 `after:`、逐层/逐 Linear 选择器、运行中变更或自动匹配类别。重复类别、未知类别、负数、空表应立即报错。只解析显式 `compression_categories` 内的名称，Qwen/Llama 不硬编码层数。
- 有覆盖表时必须 `weight_rotation=two_sided`；不能在 none 下静默忽略。与 `rot_llm` 互斥、旋转要求 `recon_loss_type=mse` 的现有约束保持。
- `resolve_category_config(category)` 将解析后的块大小写入该类 `compression.core`；`resolve_category_runtime_configs` 携带相同的 mode/block；训练只消费此 resolved 值。保存 normalized 时两层字段应可直接对照。

这是旋转范围的类别配置扩展，不增加新旋转算法、在线激活旋转、Hadamard 填充或重建后的启发式处理。

## 身份与恢复：从 fresh 起点声明完整计划

现有 `run_cat_train` 在所有 compression categories 上构造 runtime identity（`cat_train_pipeline.py:2369`），再恢复 sidecar 并执行全量严格比较（2386–2390）。因此不能先跑全局 full 的 Q/K，再在 after_k 边界把全局值改成32；即使已完成 Q/K 权重未改，未来类别的约定也已改变。不能删掉未来类别的字段、改 sidecar 或关闭比较来绕过。

拟采用两种明确的 identity 编码，而不“修补”历史 checkpoint：

1. **未提供新参数：** 保持 `vaellm_cat_cross_category_identity_v1` 和原字段内容。在构造旧 identity 的类别字典时，显式排除新加的两个 resolved rotation 字段，仍由原 `vae_shared` 承担全局旋转身份；none 的旧省略语义不变。用保留的实际 old identity 逐字段回归，不能因 dataclass 扩字段意外让全部旧续训失效。
2. **提供新参数：** 使用显式 `vaellm_cat_cross_category_identity_v2`；`resolved_category_runtime` 为全部已配置类别记录解析后的 `weight_rotation` 与 `weight_rotation_block_size`，包括尚未压缩类别。原有 seed、完整类别顺序、转置、保护/校准、几何和优化参数仍全部参与比较。保留全局值用于追溯，不把 CLI 原始字符串的顺序当作语义差异。

新计划从 fresh HF 的第一个类别开始生效。相同 v2 计划的 category-boundary / training-step 可以沿原生恢复路径；改变任何未完成类别的旋转仍必须拒绝。即使新覆盖表给出的实际值都恰好等于旧全局值，也不自动把旧 v1 边界升级为新配置：不提供未经证明的跨版本恢复。

v6 外层 checkpoint 和 CAT sidecar 格式无需修改；后者已保存整个 identity mapping。training-step 沿现有 round-base 引用和 `cat_step_resume_v6.py` 的严格 optimizer/data/runtime 契约，并在 CAT 入口校验 round-base 的跨类别 sidecar。不能只校验 step 的当前类别而漏掉未来计划。

相同 identity 只是配置/状态相容，不能单凭它宣称类别边界具有 uninterrupted 全局 RNG 续接。保留现有 boundary 重新设 seed 的行为及原 training-step exact-resume 边界；本次不修改 RNG 或把两者混称。

## 几何、保存和真实顺序

旋转仍在原坐标保护通道选取/移除后执行，沿 `rotation_spec(mode,block,seed,module_name)` 给出每个 Linear 的规格。符号由原 seed 与完整模块名稳定派生，使用独立 CPU Generator，不消费 VAE 全局 RNG；类别覆盖不引入新的类别重排或种子公式。

Qwen/Llama 的 input64 保护使 Q/K 输入维度为4032，完整旋转需要 H252×16。public 当前 `rotation/hadamard_utils.py` 不含 H252；已验证的 S `source_snapshot_20261005_01` 包含188/220/252。未来实现须使用与活动任务隔离、明确包含已验证 H252 的源码版本；不能仅实现新 CLI 就在 public 宣称完整4032可运行，也不能改正在服务回调的 S。

正块仍须整除保护后的两侧维度，0仍须有精确 Hadamard 分解，不补零、不截断、不静默回退。实际保护个数确定后复用现有构造校验；固定 input64 的候选可在 CPU 按实际模型形状核 Q/K4032，而非用无保护4096冒充。

每个 VAELinear 已有独立 `{type,version,block_size,seed}` 和 INT8 signs，组装/strict load/group8 使用本模块 metadata。full 与block32在相同保护/码率下不改变符号数量或 decoder 大小，因此**仅类别旋转范围不增加理论部署位宽**；K38本身的逻辑/物理 padding 差仍须单列，不能用 block 配置消除它。最终必须按实际训练成品审计。

新运行必须一次 fresh HF 全七类，不用 donor 合成 substitute，也不把压缩 checkpoint 改名当 HF dense 源。原生每类恢复后的 remaining dense 权重自然进入后续类别（`cat_train_pipeline.py:2789` 后的类别循环和1368附近的当前 `ref.module.weight`）；是否更新保护统计沿原 `channel_refresh_after_category` 配方，不能为了“保顺序”暗改为动态重校准。

## 实现后最小必要验证（本轮未执行）

1. **CPU 参数/身份：** 旧无参数、旧全局none/32/full 的实际解析和 v1 identity 保持；新参数解析得 Q/K0、其余32；非法/未知/after/default选择器拒绝。不同 CLI 书写顺序得到同一 resolved v2 身份。保存恢复同计划通过，改尚未压缩 gate 的块大小失败，v1/v2互换失败。
2. **真实最短顺序路径：** 授权实现后，用原 CAT 的一个 full 类别和紧接的 block32 类别覆盖实际训练→转换→既有每类后恢复→下一类压缩，保留所用模型几何/保护/数据形状，只减少更新步数。核下一类取的是本轮恢复后的当前权重，以及两类别的实际 per-Linear rotation spec；不能用两个 fresh donor 的拼接代替。
3. **保存与身份闭环：** 在该同次短测保存 boundary/必要 training-step，正常同计划恢复；故意修改未来类别配置应在真实入口被拒绝。核 model/sidecar 的原始码和signs保存重载、原生冷解码/group8，以及实例中 full/block32 混合共存。复用既有 H252数值、VAE反向、全量v6加载证据，不重做全部Hadamard因子或另加一轮无关GPU冒烟。

不承诺 GPU 最短顺序路径与完整类别训练得到同样质量；正式条件、两臂 matched 配方、资源与用户正式支持确认另行确定。完整八项和21项仍按现有九项协议合并，无八项或MMLU独立门槛。

## 不扩展接口时的已有替代

- 全七类用统一 block32，从 fresh HF 顺序压缩/恢复：现在即可表达，保留顺序，但没有 Q/K full 这一条件。
- 全七类统一full：现在CLI可表达，前提是实际源码/各保护后维度都支持；改变的类别更多，不是本设计的等价对照，也不因Q/K收益推断MLP同样受益。
- 当前 Q/K fresh donor→旧恢复骨干组装→decoder恢复：已有真实路径和结果可继续评估，但会保留其他类别的旧压缩/恢复历史。它不能回答“新Q/K前缀下后续类别重压”的问题。

没有已确认的无代码操作能让旧 after_qk 改未来全局块大小，同时继续声称原方案 exact-resume。等待接口正式支持或使用上述明确不同的现有方法，优于修改 checkpoint 身份。

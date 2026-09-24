> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`docs/exp_results/residual_lora_validation.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# 残差 LoRA：实现验证与经验

验证记录日期、整理日期：2026-09-23。

这三组记录有价值的是计算语义、训练和保存恢复链路的验证，以及接口变更中的踩坑经验。它们没有下游精度对照，不能据此判断残差 LoRA 是否改善 PPL、任务精度，或 additive 是否优于 replace。当前参数和使用方法见 [残差 LoRA 使用说明](../../../../residual_lora.md)。

## 接口演变与当前语义

最初的 `residual_lora_validation` 验证纯替换 skip；随后 `residual_lora_additive_validation` 加入 additive；最后 `residual_lora_mode_validation` 验证统一模式开关。旧总结中的布尔开关和“默认 additive”不能直接作为当前 CLI 用法。

当前唯一开关为 `--residual_lora_mode none|additive|replace`，默认 `none`；旧 `--residual_lora_enabled` 已被 CLI 明确拒绝。定义 `D(x)=(alpha/rank)*B(A(dropout(x)))`，每个 block 的 attention、MLP 两处 skip 各有独立模块：

| 模式 | skip 计算 | 初始化及含义 |
|---|---|---|
| `none` | 不新增残差模块 | 已有残差 checkpoint 仍保留其推理计算，仅冻结残差参数 |
| `additive` | `x + D(x)` | A 使用 Kaiming 初始化、B 为零；新装模块时保持原模型输出 |
| `replace` | `D(x)` | A 行正交、B 为 `A.T/(alpha/rank)`；dropout 关闭时初始 skip 为 rank-r 正交投影，不是恒等映射 |

当前支持 Qwen3/Llama，rank 范围为 1..8。模式差异、初始化和“是否继续训练”需要分别理解：设置 `none` 不会把已训练的残差 checkpoint 还原成无残差模型。

## 历史验证结果

下表来自清理前逐项核对的原始日志，属于历史结果，本次整理没有重新运行测试。不同批次存在重复用例，不能累加成独立覆盖数量。

| 原目录 | 日志 | 结果 |
|---|---|---|
| `residual_lora_validation` | `shared_config_optimizer.log` | 71 passed |
| 同上 | `regression.log` | 100 passed，24 skipped |
| 同上 | `cat_contract.log`、`cat_residual.log` | 分别 12 passed、2 passed |
| 同上 | `forward.log` | 7 passed |
| 同上 | `final_checkpoint.log` | 20 passed，3 skipped |
| 同上 | `cat_modes.log` | 17 passed；也是 additive 阶段总结引用的 CAT 结果 |
| `residual_lora_additive_validation` | `core_integration.log` | 109 passed，24 skipped |
| 同上 | `forward_modes.log` | 17 passed |
| `residual_lora_mode_validation` | `root.log` | 87 passed |
| 同上 | `cat_mode_only.log` | 47 passed |

初始验证 summary 记录环境为 bitvae、Transformers 4.51.0、PyTorch 2.6.0+cu124。初始及 additive 两组都记录使用 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=.` 在 bitvae 环境的 CPU 上验证；其中 24 项跳过项为 CUDA 相关测试。`final_checkpoint.log` 另有 3 项跳过，不计入通过。第三组只有 pytest 汇总日志和退出码，没有完整运行命令或环境清单，不能从这些文件还原其完整运行条件。前两组总结还记载入口脚本 `bash -n` 和 `git diff --check` 通过。

这些记录支持的结论：

- 两处 skip 的前向结果与手写公式一致；覆盖 Qwen3/Llama、非零残差权重、KV cache、attention 输出和 BF16。additive 的零初始化输出与基座一致。
- 普通反向与两种 gradient checkpointing 模式的结果一致；残差参数确实参与更新。
- 真实小模型的 CPU E2E Trainer 单步中，projection LoRA、VAE decoder 和残差 LoRA 均有参数更新；CAT 最后一个类别即使没有剩余 projection LoRA 目标，也可单独训练残差模块。
- FP32 参数存储与 BF16 计算、PEFT 合并导出、v6 full checkpoint 保存重载、跨类别复用均有覆盖；保存重载后的推理输出做零容差一致性检查。
- 残差可变状态和 optimizer 状态恢复后，下一步更新一致；CAT 的续训配置检查拒绝模式等关键设置变化。

这些验证不覆盖整模型长程训练收益、GPU/多卡吞吐或全部设备组合；有测试被跳过，不能写成所有路径均已验证。

## 可复用经验

1. **用初始化语义解释训练起点。** additive 在 B=0 时保留原输出，replace 会改变 skip 的信息通路。比较两种模式时要先记录初始化后的基线，不能把起点差异直接解释成学习效果。
2. **检查真实参数更新，不能只检查模块是否挂上。** PEFT 注入会冻结基座参数，当前实现先构造 PEFT，再启用 decoder/残差参数。独立 optimizer 参数组、跨类别重复安装不重置权重、最后一类无普通 LoRA 目标时仍能训练，都是需要保留的回归检查。
3. **checkpoint 必须保存模式和拓扑。** full checkpoint 保存残差模式、结构与权重，加载时拒绝冲突。exact-step resume 必须保持训练配置；首次向无残差模型添加模块应从基座 checkpoint 新开训练，不能接着旧的无残差 step checkpoint 续训。使用现有 E2E 脚本时可用 `--resume_from_checkpoint ""` 清除其预设续训参数。
4. **区分配置校验异常和 CLI 异常。** 初始 `core_integration.log` 为 2 failed、11 passed：测试把 CLI 非法 rank 的异常写成 `ValueError`，实际 argparse 对外抛出 `SystemExit(2)`。当前测试检查退出码及错误信息，不能据旧失败日志判断 rank 校验失效。
5. **测试输入要遵守实际类型约定。** 初始 `cat_integration.log` 为 2 failed、33 passed：向 `skip_layers` 传入空 tuple，被当成字符串 `()` 解析。正确空值是 `frozenset()`；后续 CAT 用例通过。应修正调用数据，不能用放宽正式解析规则掩盖问题。

## 保留的验证入口

长期维护的用例仍在仓库中：

- [test_residual_lora.py](../../../../../tests/test_residual_lora.py)：模块结构、公式、梯度和状态恢复。
- [test_residual_lora_forward.py](../../../../../tests/test_residual_lora_forward.py)：前向公式、checkpointing、cache、精度和零初始化。
- [test_residual_lora_integration.py](../../../../../tests/test_residual_lora_integration.py)：CLI、optimizer、真实训练、保存重载及下一步更新一致性。
- [test_cat_residual_lora.py](../../../../../tests/test_cat_residual_lora.py)：CAT 最后一类的残差单独训练。
- [test_cat_step_resume_contract.py](../../../../../tests/test_cat_step_resume_contract.py)：CAT 续训配置约束。

需要重新检查当前实现时，在已激活的 bitvae shell 中确认解释器，再运行上述相关用例。下面是当前回归入口，不是历史批次命令的完整复原，测试数量可能随代码变化：

```bash
conda activate bitvae
which python
python -V
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q \
  tests/test_residual_lora.py \
  tests/test_residual_lora_forward.py \
  tests/test_residual_lora_integration.py \
  tests/test_cat_residual_lora.py \
  tests/test_cat_step_resume_contract.py
```

## 原始记录清理

按用户要求，结论合并到本文后删除 `result/residual_lora_validation/`、`result/residual_lora_additive_validation/`、`result/residual_lora_mode_validation/`，共 23 个 summary、日志和退出码文件。这三个目录没有模型权重或训练 checkpoint。旧接口描述、重复测试输出、已纠正的失败堆栈和历史文件哈希不再单独保留；上文文件名只用于说明历史来源，不是可访问的归档链接。

本次仅整理文档和删除上述记录，没有修改实现或测试，也没有重跑训练/测试。其他训练和 A/B 实验目录不在本次清理范围内。

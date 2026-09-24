> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-11-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

### Task 11 — 最终回归

- [ ] 所有新增 tests。
- [ ] 所有相关 existing tests。
- [ ] tiny one-step smoke。
- [ ] 如真实模型/GPU 条件允许，再做真实短跑；否则明确记录未跑原因，不伪造结果。

## 22. 定向测试命令

先在当前 shell 激活并确认 `bitvae`：

```bash
conda activate bitvae
which python
python -V
```

然后按顺序执行：

```bash
python -m pytest -q tests/test_compressed_subspace_lora.py
python -m pytest -q tests/test_cat_compressed_lora_scope.py
python -m pytest -q tests/test_e2e_compressed_lora_scope.py
python -m pytest -q tests/test_e2e_checkpoint_io_legacy.py
python -m pytest -q tests/test_temporary_switch_residency.py
python -m pytest -q tests/test_cat_eval_adapter_match.py
python -m pytest -q tests/test_e2e_teacher_first.py
```

如果某项在修改前已有 unrelated failure，要明确区分；不要为了通过测试修改无关正式逻辑。

---

## 23. tiny one-step smoke

单元测试闭环后，再做一个不下载真实 Qwen 的 tiny smoke：

```text
channel protection
-> VAE compressed module
-> CompressedSubspacePeftProxy + PeftZeroLinearCarrier
-> PEFT LoRA injection
-> finite loss
-> backward
-> one optimizer step
-> extract effective low_rank_a/b
-> export to bare VAELinear
-> checkpoint save/load
-> final forward
```

硬性检查：

```text
loss finite
PEFT lora_A/lora_B gradients finite
optimizer step 后至少一个 PEFT LoRA parameter 变化
carrier sentinel weight.numel() == 1 且始终 frozen/zero
protected coordinates 的 LoRA delta 为 0
non-protected coordinates 至少一处变化
export 后模型中无 subspace proxy/carrier
save/load 后 forward 与 export 前 proxy forward 一致
```

不要用 mock 替代 PEFT carrier 数学；tiny decoder/model 使用真实 PyTorch module + 当前环境真实 PEFT 0.10.0。

---

## 24. 可选真实短跑验证

这属于算法实验，不是单元测试阻塞条件。

在相同 channel-protection checkpoint/config 上只改变：

```text
A: --compressed_lora_scope full
B: --compressed_lora_scope compressed_subspace
```

其它保持完全一致：

```text
seed
protected indices
VAE checkpoint
rank
alpha
dropout
dataset
steps
lr
loss
```

记录：

```text
train/KD loss
PPL
至少一个下游 task
protected weight max_abs_delta
non-protected weight max_abs_delta
trainable LoRA params
```

预期结构性检查：

- full：protected weight delta 可以非零；
- compressed_subspace：LoRA 对 protected coordinates 的额外 delta 必须在数值容忍内为 0。

accuracy 哪个更好由实验决定，不要把“subspace 必然更高”写进代码注释或测试。

---

## 25. 失败场景必须明确处理

### old checkpoint 无 scope

固定解释 `full`。

### requested subspace + existing full low-rank

报错，不转换。

### requested full + existing subspace low-rank

报错，不扩展。

### scope=subspace + A/B full incompatible shape

constructor/load 失败。

### scope=full + A/B compressed-only incompatible shape

constructor/load 失败。

### E2E selected targets mixed scope

E2E 启动前失败。

### protected index 数量与 compressed feature 不一致

`CompressedSubspacePeftProxy` construction 失败，错误信息至少包含：

```text
total features
protected count
expected compressed features
actual non-protected index count
```

### no protection + subspace

允许；shape 与 full 相同、数值等价，但 scope metadata 仍是 subspace。

---

## 26. Code quality 要求

- `VAELinear` 只负责 scope、low-rank tensor shape 与 patch 应用位置。
- `litebsq/low_rank_scope.py` 是 scope constants/normalize 的唯一 truth source，且不依赖训练代码。
- `compressed_subspace_lora.py` 只负责 compressed coordinates、O(1) PEFT carrier/proxy、PEFT A/B restore/extract、proxy 生命周期；不自己实现 LoRA 数学。
- category/E2E 只负责路由、trainable selection、训练和 final export。
- checkpoint IO 只负责 scope metadata，不复制训练逻辑。
- 不新增抽象基类、registry、plugin/factory hierarchy。
- 不重命名 `low_rank_a/b`。
- 不改变已有 checkpoint tensor key。
- 不为了统一 full/subspace 而重构现有 `PeftVAELinearProxy`。
- 不升级或修改 PEFT 0.10.0。
- 不创建 `[Oc,Ic]` dense zero carrier，不调用 PEFT merge/unload。
- 不写 fallback；不满足 contract 或 compatibility gate 失败就显式停止。

---

## 27. 最终验收标准

### 功能

- [ ] `--compressed_lora_scope full` 可用。
- [ ] `--compressed_lora_scope compressed_subspace` 可用。
- [ ] 默认 `full`。
- [ ] input protected channels 在 subspace 下不被 LoRA 修改。
- [ ] output protected channels 在 subspace 下不被 LoRA 修改。
- [ ] subspace A/B shape 真正缩小，不是 full tensor + mask。
- [ ] subspace 训练确实使用 PEFT plain LoRA 管理 A/B/scaling/dropout；没有第二套自定义 LoRA 数学。
- [ ] `PeftZeroLinearCarrier` 每个 target 只有 1 个 frozen 1×1 sentinel weight，不存在 `[Oc,Ic]` dense carrier base。
- [ ] PEFT 0.10.0 的 `inject_adapter_in_model` 与 `get_peft_model` 两个 compatibility gate 都通过。
- [ ] category subspace 可训练、export、save、reload。
- [ ] compressed E2E subspace 训练阶段仍是 root `PeftModel`，Trainer checkpoint/resume 可用。
- [ ] compressed E2E 自动继续训练 subspace checkpoint。
- [ ] E2E `both` 可直接训练 subspace low-rank payload。

### backward compatibility

- [ ] 旧 checkpoint 无新字段直接按 full 加载。
- [ ] 新 full checkpoint 不强制增加 scope metadata key。
- [ ] full category compressed LoRA 仍走现有 PEFT path。
- [ ] full E2E compressed_lora 仍走现有 root PEFT path。
- [ ] full 模式 protected coordinates 仍允许被 LoRA 修改，证明没有全局 mask。
- [ ] `remaining_lora` 不变。
- [ ] block-level PEFT LoRA 不变。
- [ ] 当前两个脚本显式 full 后结果语义不变。

### 数值

- [ ] proxy forward == export 后 `VAELinear` forward。
- [ ] subspace protected coordinate delta == 0。
- [ ] non-protected coordinate LoRA 生效。
- [ ] no-protection subspace == full。
- [ ] save/load 前后数值一致。

### 工程

- [ ] 不修改 PEFT library。
- [ ] 不存在 mask-after-merge。
- [ ] 不存在自动 scope conversion。
- [ ] scope normalize 只有一个 truth source。
- [ ] shape contract 最终只有 `VAELinear` 一个 truth source。
- [ ] 所有新增测试及相关 regression 通过。

---

## 28. Cursor 完成后必须汇报

只汇报以下内容，不写冗长背景：

1. 新增/修改了哪些文件；
2. `full` 与 `compressed_subspace` 分别走哪条代码路径；
3. input/output protection 下 subspace A/B 最终 shape；
4. 旧 checkpoint 如何兼容；
5. 跑了哪些测试及结果；
6. 是否做 tiny one-step smoke；
7. 是否做真实模型短跑；若没有，说明原因；
8. 最终 diff 是否只有本任务相关改动。

不要自动 commit。


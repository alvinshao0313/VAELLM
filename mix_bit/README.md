# mix_bit — Mixed-Bit VAE Allocation

在不进行逐类别蒸馏和端到端微调的前提下，训练完整 VAE 候选压缩池，相对全精度 teacher 计算逐 Linear 配对 KL 代价，在参数量加权平均位宽 `<= target_average_bit`（默认 2.0）约束下求解全局最优分配，并组装、重载、验证最终混合位宽模型。

本包实现的是**混合位宽阶段**。后续类别蒸馏 / 端到端微调属于独立计划，本流程不预留隐式训练入口。

## 支持的执行顺序

```text
1. Resolve run config and build/verify the model inventory.
2. Run production-loader structural preflight.
3. Dry-run C × R candidate jobs.
4. Train/resume and directly export all candidate-only compressed artifacts from the in-memory model.
5. Inventory the pool and require L × R dense module-mode coverage.
6. Write the tensor-free uniform baseline overlay and validate two independent in-memory builds.
7. Prepare the deterministic calibration dataset.
8. Select exactly one KL mode.
9. For teacher_topk, build one K-specific cache.
10. Compute/resume C × (R-1) cost jobs producing L × (R-1) nonbaseline rows.
11. Finalize L × R rows after validated baseline zeros.
12. Solve the exact parameter-weighted MILP.
13. Assemble/reload the optimal mixed model.
14. Validate structure, budget, actual KL and downstream metrics.
```

## 通用计数公式

算法只依赖模型 inventory / 候选空间给出的通用量：

```text
C = 启用逻辑类别数
L = 目标 Linear 数
R = 候选模式数

candidate training jobs              = C × R
cost source jobs                     = C × (R - 1)
non-baseline module evaluations      = L × (R - 1)
complete cost rows                   = L × R
```

平均 bit 只统计目标 Linear 权重参数量 `N_l = in_features × out_features`，使用候选模式显式 `nominal_bit`，不含 decoder / bias / embedding / norm / LM head。

### Profile 回归示例：Qwen3-8B

Qwen3-8B 仅作首个落地 profile 与回归断言，**不得**硬编码进通用调度 / Cost / 求解 / 验证逻辑：

```text
C=7, L=252, R=5
→ candidate jobs=35, cost source jobs=28,
  non-baseline evals=1008, complete rows=1260
```

配置入口：`mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json`。

## Non-goals 与模型边界

当前版本明确不做：

- 逐类别蒸馏或端到端恢复训练
- 超出名义 payload bits 的开销记账（decoder / protect / bias 等不计入平均 bit）
- 成对 / 高阶交互代价项
- 把任务指标（PPL、下游）写入求解目标
- 自动剪枝或候选替换
- `trust_remote_code` / remote-code 模型
- fused-QKV、MoE expert、跨层共享 Linear 等结构——除非新增显式 adapter / 训练扩展

新增普通 decoder-only Transformer 时，优先只加 JSON model profile；只有通用 `generic_decoder` adapter 无法表达的结构才新增专用 adapter。

## 实现阶段 vs 生产阶段

### 实现阶段（Tasks 1–13，Cursor 已完成代码）

实现验收完成条件：

```text
all unit tests and selected repository regressions pass
synthetic tiny-model integration completes end to end
Qwen model inventory command succeeds
Qwen candidate and cost planners produce expected dry-run counts
all CLIs expose --help and reject invalid provenance combinations
no production training subprocess is launched during dry-run acceptance
```

本地合成集成夹具（`mix_bit/tests/test_tiny_integration.py`）覆盖：2 blocks × 2 categories × 3 modes；candidate-only 导出；tensor-free baseline overlay + 双次内存组装；两种 KL 指标；原子 cost finalize；MILP 与 brute-force 对照；最终混装与 strict reload。全程离线、无网络。

实现验收**不得**自动启动 35 个长时 Qwen3-8B 候选训练，也不得为验收创建完整中间模型 checkpoint。

### 生产实验阶段（实现审查通过后，人工故意启动）

生产完成需要：

- 全部 `C × R` candidate-only 压缩 artifact（Qwen 回归：35；每模式 `residual_stages=2`）
- tensor-free uniform baseline overlay（双次内存构建一致；无完整 baseline checkpoint）
- 冻结 calibration 数据；恰选一种 KL（`teacher_topk(K)` 或 `exact_full_vocab`）
- 一张 metric-specific cost table（`L × R` 行；Qwen 回归：1260）
- 经全局最优验证的分配，以及唯一最终 standalone checkpoint
- actual mixed KL、additive prediction，以及 baseline/mixed 的 PPL/下游（下游永不进入求解目标）

每个阶段可 resume；失败时禁止用 `strict=False`、缺行插补、默认候选替换、cost 裁剪、静默重建 cache、自动 metric/baseline 回退等方式掩盖。

## CLI 入口

| 步骤 | 模块 |
|---|---|
| Inventory | `python -m mix_bit.cli.build_model_inventory` |
| Candidate pool | `python -m mix_bit.cli.train_candidate_pool` |
| Pool index | `python -m mix_bit.cli.inventory_candidate_pool` |
| Baseline overlay | `python -m mix_bit.cli.prepare_uniform_baseline` |
| Calibration | `python -m mix_bit.cli.prepare_calibration` |
| Teacher cache | `python -m mix_bit.cli.build_teacher_cache` |
| Cost table | `python -m mix_bit.cli.compute_cost_table` |
| MILP solve | `python -m mix_bit.cli.solve_allocation` |
| Assemble | `python -m mix_bit.cli.assemble_mixed_model` |
| Validate | `python -m mix_bit.cli.validate_mixed_model` |

薄 shell 入口在 `mix_bit/scripts/`（只设路径/环境变量并调用显式传入的 Python 解释器；不含 conda / 业务逻辑）。

**执行环境：** 父 CLI（如 `train_candidate_pool`）必须由 `bitvae` conda 环境中的 Python 启动（例如 `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.train_candidate_pool ...`）。candidate pool 子任务会通过 `sys.executable` 的绝对路径固定到同一解释器，不依赖 worker shell 的 `PATH` 或默认 `python`。

### 实现验收常用命令（Qwen profile）

```bash
python -m mix_bit.cli.build_model_inventory \
  --run_config mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json \
  --output .result/mix_bit/qwen3_8b/model_inventory.json

python -m mix_bit.cli.train_candidate_pool \
  --run_config mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json \
  --inventory .result/mix_bit/qwen3_8b/model_inventory.json \
  --gpus 4,5,6,7 \
  --dry_run
```

Cost planner 的 dry-run 需要已导出的 candidate pool manifest 与 baseline overlay；在 artifact 尚未齐备时，可用 `compute_search_counts(C,L,R)` / 单测回归核对 `28/1008/1260`，不要为了 dry-run 伪造完成态。

### 自定义 candidate pool 根目录

candidate training 阶段可用 `--output_root X` 把候选池写到非 canonical 路径（`X` 不是 `<run_root>/candidate_pool`）。后续所有阶段统一传 `--pool_manifest X/candidate_manifest.json`，该路径即为权威候选池位置：helper 以 manifest 父目录为 pool root，校验 supplied JSON 与按 artifact 重算的 payload 完全相等（artifact 顺序、绝对路径、每个 SHA），不重写、不修复 manifest。`solve_allocation` 的 `--pool_manifest` 可选；未提供时退回 canonical build。cost spawn worker（baseline init 与每个 worker）通过 `pool_manifest_path` 携带同一绝对路径，子进程用 `build_candidate_pool_index_from_manifest` 加载，不再 canonical rebuild。

## 执行门禁（失败即停）

任一 gate 失败必须报告具体错误，禁止降级绕过。完整列表见实现计划 / Task 13 brief。核心硬失败 gate：

- **mode/payload mismatch**：candidate mode metadata 五字段（`codebook_bits`、`codebook_dim`、`residual_stages`、`nominal_bit`、`name`）必须与 candidate space 完全一致；actual candidate structure（stages / dim / logical bits / decoder dims）必须与 metadata 完全一致，否则拒绝。
- **wrong Python executable**：candidate pool 子任务必须使用父进程 absolute `sys.executable`（`bitvae/bin/python`）；任何 worker 使用错误解释器即失败。
- **top-k full-logits CPU transfer regression**：teacher top-k 只允许把 `[N_valid, K]` 搬到 CPU；student top-k 必须直接 gather `[B, T, K]`，禁止生成 / 搬运 `[N_valid, V]`；出现 `shifted.detach().cpu()` 或 `reference_state = ...` / `cpu().clone()` 全量 CPU 拷贝即失败。
- **worker startup/runtime death**：worker startup 必须带 900 秒 timeout，child 死亡立即失败；runtime 阶段任一 worker 非预期死亡立即失败；baseline/ready 路径的所有 `result_queue.get` 必须带 timeout。
- **tokenizer fingerprint mismatch**：calibration tokenizer 必须使用 fingerprint v2（core / chat template / added vocab 均受保护）；resume 缺 version 或 SHA 不一致即失败；final tokenizer 必须 local-only 可重载且 fingerprint 相同。
- **custom manifest root mismatch**：`--pool_manifest.parent` 是唯一真实 candidate pool root；supplied manifest payload 必须与按 artifact 重算的 payload 完全相等（artifact 顺序、绝对路径、每个 SHA），不重写、不修复。
- **final state fingerprint mismatch**：最终 checkpoint state 使用流式 16 MiB SHA，禁止完整 CPU clone；reload 后重算 fingerprint 必须等于保存前，否则失败。

其余 gate（inventory / loader 不一致、candidate artifact 污染或覆盖不全、训练 / baseline / cost 阶段写出完整模型 checkpoint、baseline overlay 含 tensor 或双构建不一致、metric / cache / provenance 不一致、cost 行不完整或非有限、MILP 非 one-hot / 超预算 / 非全局最优、最终 checkpoint 含原 dense weight 或 reload 失败、下游结果用于改写分配）同样硬失败。

## Tokenizer fingerprint v2

最终 checkpoint 必须自包含 tokenizer，且 tokenizer 一致性由 fingerprint v2 守护：

- `mix_bit/calibration.py` 暴露 `TOKENIZER_FINGERPRINT_VERSION = 2`、`build_tokenizer_fingerprint_payload`、`compute_tokenizer_config_sha256`（保留旧函数名，内部升级为 v2）。
- payload 分 provenance 与 content：`reported_name_or_path` 仅作 provenance，不参与 digest；`compute_tokenizer_config_sha256` 只 hash `version` + `content`，因此 source tokenizer 的 `name_or_path="Qwen/Qwen3-8B"` 与 reload tokenizer 的 `name_or_path=<final_model_dir>` 不会造成伪 mismatch。
- content 包含 `class_name`、`vocab_size`、`model_max_length`、`padding_side`、`truncation_side`、`bos/eos/pad/unk_token_id`、`chat_template`、`special_tokens_map`、`added_vocab`、`core_kind`（`backend_tokenizer_json` 优先，回退 `sorted_vocab`，两者皆无则失败）、`core_sha256`、`stable_init_kwargs`（排除路径/鉴权/缓存类键，集合常量 `TOKENIZER_INIT_KWARGS_EXCLUDED`）。
- 递归 JSON 归一化只允许 null/bool/int/finite float/str/list/tuple/dict（键转 str）；`Path` 转 str；Hugging Face `AddedToken`（含 `content` 字段）固定序列化为 `{content, single_word, lstrip, rstrip, normalized, special}`；其他对象记为 `{"unsupported_type": "ClassName"}`，不调用 `repr()` 避免地址泄漏。
- `mix_bit/model_adapter.py` 暴露 `normalize_tokenizer_for_mix_bit(tokenizer, *, source_label)`：先设 `padding_side="right"`，再重置 `mix_bit_pad_token_normalized_from_eos=False`，若 `pad_token_id is None` 则要求 `eos_token_id` 并令 `pad_token_id=eos_token_id`、标记置 True。`GenericDecoderAdapter.load_tokenizer`、assembler 的 local reload、validation 的 local reload 都必须调用该 helper，不得复制三套 normalization 逻辑。
- calibration manifest 新增 `tokenizer_fingerprint_version: 2`；resume 必须要求 version=2 且 SHA 一致；旧 manifest 缺少 version 时必须失败并提示重新生成 calibration，不得静默接受。
- assembler 在 `save_full_checkpoint_from_assignments` 中：通过 profile adapter 加载 source tokenizer（已 normalize）→ 计算 v2 fingerprint → 把 `tokenizer_fingerprint_version`、`tokenizer_fingerprint_sha256`、`source_tokenizer_reported_name_or_path` 写入 `extra_meta.mix_bit` → 调用 `save_model_checkpoint` 显式传入 tokenizer → 从 `output_dir` 以 `local_files_only=True`、`trust_remote_code=False` 重载并 normalize → 重算 content fingerprint，必须等于 source，否则失败并保留 state/meta/tokenizer 文件供排查（assembler 无 completed marker，不虚构不删除）。返回 payload 新增 `tokenizer_fingerprint_sha256` 和 `tokenizer_reported_name_or_path`。
- `assemble_optimal_mixed_checkpoint` 在 Task 6 state fingerprint 検査通过后，还必须 local-only 重载 tokenizer、读取 meta 中的 version 与 expected SHA、重算并比较；tokenizer 文件缺失、旧 version、local reload 失败或 SHA mismatch 时抛 `ValueError` 并要求显式 `--overwrite`；不得从原模型路径补载 tokenizer，不得自动 `save_pretrained` 修复旧目录。只有 state fingerprint 与 tokenizer fingerprint 两项都通过才 `skipped_identical=True`。
- `validate_mixed_model` 从 final dir local-only 加载 tokenizer、校验 meta 中 `tokenizer_fingerprint_version=2`、重算并比较 fingerprint；validation report 新增 `tokenizer` 段（`fingerprint_version`、`fingerprint_sha256`、`reported_name_or_path`、`local_reload_passed`）；下游 evaluator 的 tokenizer 必须来自 final dir，禁止再依赖原模型路径（模型 checkpoint 本身仍按现有 `base_model_path` 机制重建 backbone）。
- `mix_bit/cli/assemble_mixed_model.py` 新增 `--access_token` 并逐层传递。

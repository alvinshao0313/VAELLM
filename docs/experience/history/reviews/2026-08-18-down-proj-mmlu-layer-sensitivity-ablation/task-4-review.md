> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-4-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Review: Deterministic Multi-GPU Phase-1 Scheduling

**Reviewer:** Spec + quality review (Task 4)  
**Artifacts:** `task-4-brief.md`, `task-4-report.md`, `task-4-review-package.diff`  
**Verdict:** **Spec ✅** · **Approved**

---

## Executive Summary

Task 4 交付物与 brief 一致：phase-1 job inventory、baseline-first 约束、确定性 greedy 分配、manifest 命名、`launch_phase_workers` 子进程契约、失败门禁，以及 Step 9 锁定的 `main()` 控制流均已实现。单元测试对 N∈{1,2,4,8} 使用写死的 worker→job_id 期望列表，不是仅测“负载差不多”。

`summarize.py` 的**实现**属于 Task 5/6，但 **`main()` 中 lazy-import 并调用** 属于 Task 4 Step 9 的硬性要求——当前 wiring 正确，不应 defer 到 Task 5 再改 `run.py` 结构。`build_phase2_manifests` 的 `NotImplementedError` stub 不违反 Task 4 brief；完整 phase-2 逻辑由 Task 6 实现，且 Task 6 计划会修改 `run.py` 以处理 `W2` GPU 切片。

---

## Spec Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| 仅修改 `experiments/down_layer_sensitivity/` | ✅ | `run.py` 新建；`test_job_manifest.py` 扩展 |
| CLI 仅 4 个 experiment-control flags | ✅ | `--checkpoint_dir`, `--output_dir`, `--gpus`, `--mode` |
| 科学常量硬编码、不可 CLI 覆盖 | ✅ | 模块级 `SEED`, `RANDOM_CONTROL_SEEDS`, … |
| Run ID `YYYYMMDD_HHMMSS_{mode}` | ✅ | `datetime.now().strftime(...)` |
| 单一 `run_config.json`，含 `phase1_worker_count` | ✅ | `_write_initial_run_config`；失败时原地更新 |
| Formal inventory = `38 + W` | ✅ | W baselines + repeat + all_down + 36 restores |
| Worker 0 baseline → repeat 在前 | ✅ | `build_phase1_manifests` formal 分支 |
| 其它 worker baseline 在 jobs[0] | ✅ | 同上 |
| Scientific 顺序：`all_down_original`, `restore_L00..L35` | ✅ | `scientific_jobs` 构建顺序 |
| Greedy：最少 job 数，tie → 较小 `worker_id`，cost=1 | ✅ | `_least_loaded_worker_id` |
| Smoke W=1，恰好 4 jobs，固定顺序 | ✅ | smoke 分支 + `test_smoke_rejects_multiple_gpus` |
| Smoke 不进入 phase 2 | ✅ | `main()` smoke 分支 `return` |
| 仅 `worker_00` `write_weight_metrics=true` | ✅ | `worker_id == 0` |
| Manifest 路径 `phase1/manifests/worker_XX.json` | ✅ | `launch_phase_workers` |
| 每 GPU 一个 `Popen`；固定 argv；`sys.executable` | ✅ | `test_launch_phase_workers_writes_manifests_and_fixed_command` |
| 环境 = parent copy + `CUDA_VISIBLE_DEVICES=<g>` | ✅ | 同上 |
| 无 torchrun / DDP / multiprocessing worker 入口 | ✅ | 仅 `subprocess.Popen` |
| 失败：写 `failed_workers` + 非零退出；不 summarize/phase2 | ✅ | `launch_phase_workers` + 测试 |
| Step 9 锁定 formal/smoke `main()` 链 | ✅ | 见下文专项分析 |
| Step 10 测试：写死期望分配 + 38+N | ✅ | `EXPECTED_FORMAL_JOB_IDS` + 28 passed |
| 无 git commit | ✅ | report 确认 |

---

## Focus Areas

### 1. `summarize.py` 调用：Task 4 必须 wiring，实现 defer 到 Task 5/6

**结论：当前做法符合 brief，不是 spec 缺口。**

Brief Step 9 明确要求 formal `main()` 链：

```text
… → launch_phase_workers(phase1) → summarize_phase1 → build_phase2_manifests → launch_phase_workers(phase2) → summarize_final → status=completed
```

Smoke 链同理包含 `validate_smoke(...)`。

实现中 phase-1 成功后 lazy-import：

```python
from experiments.down_layer_sensitivity.summarize import summarize_final, summarize_phase1
# smoke: validate_smoke
```

这与 brief **“禁止把 phase 1/2 做成两个需人工串联的独立脚本”** 一致。Task 5 负责创建 `summarize.py` 并实现 `summarize_phase1` / `validate_smoke`；Task 6 负责 `summarize_final` 与 `build_phase2_manifests` 实现（见全局 plan Task 5/6 文件列表）。

因此：

- **Task 4 应做：** 锁定 call chain、保留 import 点（已完成）。
- **Task 4 不应做：** 实现 summarize 逻辑或为了“能跑通”删掉 call chain。
- **已知限制（非 Task 4 缺陷）：** `summarize.py` 尚不存在，formal/smoke 在 phase-1 worker 全部成功后会在 import 处 `ImportError`；report 已如实说明。

Step 8 “失败时不 run summarize/phase2” 也满足：`launch_phase_workers` 在 worker 失败时 `SystemExit(1)`，`main()` 不会到达 summarize。

---

### 2. Phase-2 stub 是否违反 Task 4 brief

**结论：不违反。**

Brief Task 4 标题聚焦 phase-1 scheduling，但 Step 9 仍要求 formal 链包含 `build_phase2_manifests(...)` 与第二次 `launch_phase_workers(phase2)`。当前：

```python
def build_phase2_manifests(...) -> list[dict]:
    raise NotImplementedError("build_phase2_manifests is implemented in Task 6.")
```

- 函数签名与 call site 位置正确，满足“锁定控制流”。
- 完整 phase-2 manifest 构建属于 Task 6（全局 plan 明确 “Modify run.py” + 实现 `build_phase2_manifests`）。
- `NotImplementedError` 优于空实现或假 manifest，避免 Task 4 越界实现 phase-2 科学 job。

**Forward note（非 blocking）：** Task 4 `main()` 向 phase-2 launch 传入完整 `selected_gpus`，而 Task 6 规定 `W2 = min(len(selected_gpus), 9)` 且 `phase2_gpus = selected_gpus[:W2]`。Task 6 修改 `run.py` 时必须同步 slice `selected_gpus`，否则 `launch_phase_workers` 的 `len(manifests) != len(selected_gpus)` 检查会失败。Report concern #2 已记录；归属 Task 6，不构成 Task 4 changes requested。

---

### 3. Scheduling / job assignment 是否与 brief 完全一致

**结论：一致；测试覆盖充分。**

**Formal job inventory**

- W 个 `compressed_baseline_worker{XX:02d}`（每 worker 一个）
- 1 个 `compressed_baseline_worker00_repeat`（仅 worker 0）
- 1 个 `all_down_original`
- 36 个 `restore_L00..L35`

总数 `38 + W`；测试断言 `len(all_ids) == 38 + num_gpus` 且全局无重复 job_id。

**Baseline-first**

- 实现先填充各 worker baseline，再 greedy 分配 scientific jobs → 保证 restore/all_down 不会排在 baseline 之前。
- 测试对 scientific job 检查 `index > 0`（worker 0 则 `index > 1`）。

**Greedy 分配**

- Scientific 入队顺序与 brief 一致：`all_down_original` 先于 `restore_L00..L35`。
- 分配规则 `(len(jobs), worker_id)` 最小化 → 与 brief “最少 job 数，tie 选较小 worker_id” 一致。
- `EXPECTED_FORMAL_JOB_IDS` 对 N=1,2,4,8 写死完整 worker→job_id 列表；实现与期望逐 worker 相等（非仅 balance 启发式）。

**Smoke**

- 固定 4 jobs 顺序：`compressed_baseline_worker00` → `repeat` → `restore_L00` → `all_down_original`。
- `W!=1` 拒绝。

**Spot-check（N=2，与 brief 示例逻辑一致）**

| Step | Job | Loads after |
|------|-----|-------------|
| init | baselines | W0=2, W1=1 |
| +1 | all_down_original → W1 | W0=2, W1=2 |
| +2 | restore_L00 → W0 (tie 2,2 → id 0) | W0=3, W1=2 |
| … | 偶数 L → W0，奇数 L → W1 | 与 `EXPECTED_FORMAL_JOB_IDS[2]` 一致 |

N=4、N=8 由同一算法 + 写死期望表验证；report 28/28 passed。

---

## Quality Notes (Non-blocking)

1. **“byte-equivalent” 测试措辞：** `test_formal_phase1_manifest_allocation_is_deterministic` 用 `json.dumps` 比较两次调用结果，语义等价而非 OS 级字节比较；对同一 Python 结构足够，与 brief 意图一致。
2. **`validate_smoke` 签名：** brief 只命名调用，未冻结参数；当前假设 `validate_smoke(*, run_dir, selected_gpus)`，Task 5 实现时应保持或同步更新 `run.py`。
3. **`run_config.json` schema：** brief 未冻结全部 key 名；当前 snake_case 镜像常量合理。
4. **Formal end-to-end 暂不可跑：** phase-1 scheduling 本身可测；完整 formal run 需 Task 5+6。属计划内分层，不是 Task 4 返工理由。

---

## Test Evidence

Report 声称（本 review 未重跑 pytest，基于 package diff + report）：

```text
pytest -q experiments/down_layer_sensitivity/tests/test_job_manifest.py  → 28 passed
pytest -q experiments/down_layer_sensitivity/tests                    → 50 passed
```

新增测试覆盖 brief Step 10 全部断言项，以及 launch argv、`CUDA_VISIBLE_DEVICES`、失败 worker `run_config` 更新。

---

## Decision

| Dimension | Result |
|-----------|--------|
| **Spec** | ✅ |
| **Review outcome** | **Approved** |

Task 4 可合并进后续 Task 5/6 工作流；无需为 summarize wiring 或 phase-2 stub 返工。Task 5 应直接实现被 `run.py` 已引用的接口；Task 6 实现 `build_phase2_manifests` 并修正 phase-2 GPU slice。

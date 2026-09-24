> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4: Implement Deterministic Multi-GPU Phase-1 Scheduling

**Files:**
- Create: `experiments/down_layer_sensitivity/run.py`
- Extend: `experiments/down_layer_sensitivity/tests/test_job_manifest.py`

**Interfaces:**
- Produces phase-1 worker manifests and launches `worker.py` subprocesses.

## Fixed phase-1 job set

设本次 formal run 选中的 GPU 数为 `W = len(selected_gpus)`。Phase 1 的 job inventory **唯一**定义为：

```text
W x compressed_baseline_workerXX      # 每个 worker/GPU 各跑一次同一 baseline
1 x compressed_baseline_worker00_repeat
1 x all_down_original
36 x restore_L00 ... restore_L35
```

因此 formal phase-1 总 job 数固定为：

```text
38 + W
```

例如 4 GPU 时是 42 jobs，8 GPU 时是 46 jobs。这里额外的跨 GPU baseline 不是可选诊断，而是正式结果可比性的硬门禁。

- [ ] **Step 1: Parse only experiment-control CLI**

`run.py` CLI **只能**暴露：

```text
--checkpoint_dir
--output_dir
--gpus            comma-separated physical GPU IDs
--mode            smoke|formal
```

All scientifically relevant formal settings stay constants in code:

```python
SEED = 31
RANDOM_CONTROL_SEEDS = (31, 32, 33, 34, 35)
PREWARM_GROUP_SIZE = 8
FORMAL_LM_LIMIT = None
SMOKE_LM_LIMIT = 2
EXPECTED_DOWN_LAYERS = 36
HISTORICAL_COMPRESSED_MMLU = 0.4171
HISTORICAL_PRE_DOWN_MMLU = 0.5199
```

Do not expose arbitrary fewshot, task list, MMLU limit, restore count, random-control seed or metric as user-overridable flags in the formal script.

- [ ] **Step 2: Create a unique run directory and config**

Run ID format:

```text
YYYYMMDD_HHMMSS_formal
YYYYMMDD_HHMMSS_smoke
```

Write `run_config.json` before launching workers with all constants, checkpoint path, selected GPUs and mode. `run_config.json` also records `phase1_worker_count=W` and later追加 `phase2_worker_count`、phase completion status；禁止另建第二份配置文件表达同一 run 状态。

- [ ] **Step 3: Build phase-1 manifests exactly**

实现并只使用：

```python
def build_phase1_manifests(*, selected_gpus: list[str], mode: str) -> list[dict]:
```

Formal baseline probes:

```python
for worker_id in range(W):
    job_id = f"compressed_baseline_worker{worker_id:02d}"
    restore_layers = []
```

Worker 0 额外第二个 baseline：

```text
compressed_baseline_worker00_repeat
```

Scientific jobs 固定顺序：

```text
all_down_original
restore_L00
restore_L01
...
restore_L35
```

Smoke 固定只允许 `W=1`，manifest 中按顺序恰好四个 jobs：

```text
compressed_baseline_worker00
compressed_baseline_worker00_repeat
restore_L00
all_down_original
```

Smoke never runs phase 2.

- [ ] **Step 4: Every worker must run baseline before any intervention**

Formal phase 1 中：

```text
worker 0 jobs[0] = compressed_baseline_worker00
worker 0 jobs[1] = compressed_baseline_worker00_repeat
worker i>0 jobs[0] = compressed_baseline_workerXX
```

任何 `restore_*` / `all_down_original` 都不得排在该 worker baseline 之前。这样每张参与 GPU 都先证明“同一 compressed 模型在本卡上得到相同 MMLU”。

- [ ] **Step 5: Balance scientific jobs deterministically**

先放好上述 baseline jobs，再按固定 scientific job 顺序逐个分配：每次选择**当前总 job 数最少**的 worker；并列时选择较小 `worker_id`。所有 job cost 一律视为 1。

禁止随机分配，禁止根据预计运行时间动态 stealing，禁止运行中重新分配。

- [ ] **Step 6: Write manifest files with exact names**

Manifest 路径固定为：

```text
phase1/manifests/worker_00.json
phase1/manifests/worker_01.json
...
```

只有 `worker_00.json` 顶层 `write_weight_metrics=true`；其它 worker 必须为 `false`。

- [ ] **Step 7: Launch one subprocess per selected GPU with one fixed command shape**

实现并只使用：

```python
def launch_phase_workers(
    *,
    checkpoint_dir: str,
    phase_dir: str,
    selected_gpus: list[str],
    manifests: list[dict],
) -> None:
```

每个 physical GPU `g` 的环境是 parent env copy +：

```text
CUDA_VISIBLE_DEVICES=<g>
```

启动命令必须使用当前 bitvae interpreter `sys.executable`，形状固定为：

```text
<sys.executable> experiments/down_layer_sensitivity/worker.py
  --checkpoint_dir <checkpoint_dir>
  --manifest_path <phase_dir>/manifests/worker_XX.json
  --jobs_dir <phase_dir>/jobs
  --worker_meta_path <phase_dir>/worker_logs/worker_XX_meta.json
  --worker_id <XX-as-int>
  --physical_gpu_id <g>
```

Worker 内永远使用 local `cuda:0`。禁止 `torchrun`、DDP、FSDP、torch distributed、shell 后台调度或另一套 multiprocessing worker 入口。

- [ ] **Step 8: Collect process exit codes**

Coordinator waits for all workers. If any worker exit code != 0:

```text
- write failed_workers into run_config.json status
- exit non-zero
- do not run summarize/phase2
```

No partial sensitivity ranking on failed formal phase 1.

- [ ] **Step 9: Lock the top-level call chain**

`run.py::main()` 的 formal 控制流必须严格是：

```text
parse CLI
create run_dir + run_config.json
build_phase1_manifests(...)
launch_phase_workers(phase1)
summarize_phase1(...)
build_phase2_manifests(...)
launch_phase_workers(phase2)
summarize_final(...)
mark run_config status="completed"
```

Smoke 控制流严格是：

```text
parse CLI
create run_dir + run_config.json
build_phase1_manifests(mode="smoke")
launch_phase_workers(phase1)
validate_smoke(...)
mark run_config status="smoke_completed"
return
```

禁止让 Cursor 自行选择把 phase 1/2 做成两个需要人工串联的独立脚本。

- [ ] **Step 10: Test deterministic manifest allocation**

For N GPU in `{1,2,4,8}` unit test:

```text
- total formal phase-1 jobs exactly 38+N
- every worker has its own compressed_baseline_workerXX at jobs[0]
- worker0 has compressed_baseline_worker00_repeat at jobs[1]
- all_down_original appears exactly once
- restore_L00..L35 each appears exactly once
- no scientific job appears before that worker's baseline probe
- repeated calls produce byte-equivalent manifest job ordering
```

再断言 greedy 分配结果等于测试中写死的期望 worker→job_id 列表；不要只测试“负载差不多”。

---


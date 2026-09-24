> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-7-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 7: Make Spawn Worker Startup and Runtime Fail Fast

**Files:**
- Modify: `mix_bit/cost_table.py`
- Modify: `mix_bit/tests/test_cost_table.py`

**Constants:**

```python
BASELINE_STARTUP_TIMEOUT_SECONDS = 900.0
WORKER_STARTUP_TIMEOUT_SECONDS = 900.0
RESULT_QUEUE_POLL_SECONDS = 1.0
WORKER_JOIN_TIMEOUT_SECONDS = 30.0
```

不得新增 CLI 参数；先使用固定生产安全值，避免用户误设成过短时间。

**Interfaces:**

```python
def _wait_for_single_process_message(
    *,
    process: mp.Process,
    result_queue: mp.Queue,
    expected_type: str,
    timeout_seconds: float,
    label: str,
) -> dict[str, Any]:
    """Wait with polling, child liveness checks and a total deadline."""


def _wait_for_workers_ready(
    *,
    processes: Sequence[mp.Process],
    result_queue: mp.Queue,
    timeout_seconds: float,
) -> None:
    """Require one unique ready message from every logical worker."""


def _dead_process_descriptions(processes: Sequence[mp.Process]) -> list[str]:
    """Return stable pid/exitcode strings for processes that are not alive."""
```

### Single baseline process behavior

循环：

1. `queue.get(timeout=min(1.0, remaining))`；
2. 收到 `failure` 立即 raise，错误包含 child traceback；
3. 收到 expected type 返回；
4. timeout 后若 process 已退出，raise，包含 pid/exitcode；
5. 总 deadline 到达，terminate/join child，然后 raise `TimeoutError`；
6. 不允许在收到 message 前无 timeout `join`；
7. 收到 ready 后 `join(timeout=30.0)`；若进程仍 alive，则 terminate/join 并失败；若 exitcode 非 0 也失败。

### Worker ready behavior

- 以 `logical_id` 跟踪 ready set；重复 ready 失败；
- failure message 立即失败；
- 任一 process 在全部 ready 前退出，立即失败；
- 900 秒总 deadline 后 terminate 全部并失败；
- 成功条件是所有 logical_id 恰好 ready 一次。

### Runtime behavior

当前 job loop 的 `queue.Empty` 分支必须改为：

```python
dead = [p for p in processes if not p.is_alive()]
if dead:
    failures += 1
    stopping = True
    failure_detail = "; ".join(_dead_process_descriptions(dead))
    break
```

不能只在“所有 process 都死亡”时失败。因为任意一个 worker 取走 job 后崩溃，其他 worker 仍活着时也会永久等待。

最终 RuntimeError 必须包含 `failure_detail`。

- [ ] **Step 1: Add fake process and queue test utilities**

测试 utility 只放在 `test_cost_table.py`，提供 `pid`、`exitcode`、`is_alive`、`terminate`、`join`。

- [ ] **Step 2: Add baseline startup tests**

完整实现：

- `test_baseline_wait_fails_when_child_exits_without_message`
- `test_baseline_wait_surfaces_failure_traceback`
- `test_baseline_wait_times_out_and_terminates_child`
- `test_baseline_wait_accepts_exact_ready_message`

使用 0.01 秒 timeout，不实际等待 900 秒。

- [ ] **Step 3: Add multi-worker startup tests**

完整实现：

- `test_worker_ready_wait_rejects_duplicate_logical_id`
- `test_worker_ready_wait_fails_if_one_worker_dies`
- `test_worker_ready_wait_times_out_and_terminates_all`
- `test_worker_ready_wait_accepts_all_unique_workers`

- [ ] **Step 4: Add runtime partial-death test**

模拟两个 workers，一个死亡、一个仍 alive、一个 in-flight job；scheduler 必须失败，不得继续 poll。

- [ ] **Step 5: Run tests and confirm old blocking implementation cannot satisfy tests**

- [ ] **Step 6: Implement helper functions and replace baseline `get()`**

删除 `_ensure_baseline_per_sample_spawn` 中的无 timeout `result_queue.get()`。

- [ ] **Step 7: Replace worker-ready `get()` and runtime liveness check**

删除 ready loop 中的无 timeout `result_queue.get()`。

- [ ] **Step 8: Ensure all error paths call existing drain/terminate helper**

任何 startup/runtime failure 后都不得残留 daemon process。

- [ ] **Step 9: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_cost_table.py -q
```

- [ ] **Step 10: Commit Task 7 files**

```bash
git add mix_bit/cost_table.py mix_bit/tests/test_cost_table.py
git commit -m "fix: fail fast on mixed-bit worker crashes"
```

---


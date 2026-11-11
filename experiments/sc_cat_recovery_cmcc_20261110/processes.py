"""Isolated adaptation of the mature finite ownership runner; one shared4-GPU pool."""
from contextlib import contextmanager
import os
from copy import deepcopy
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time

from experiments.sc_cat_live_cmcc_20261106.bindings import require, read, write, binding, verify_sources, check_resources

class Budget:
    """Persist before launch; reserve all dependency windows; charge failed ownership too."""
    def __init__(self, path, authorization, windows, receipt_sources, previous=None, retry_tasks=None, new_batch=False):
        self.path, self.authorization = Path(path), authorization
        self.lock = threading.RLock()
        require(not self.path.exists(), "Existing budget is evidence, not authorization for an automatic retry.")
        self.value = dict(schema="sc_cat_live_actual_ownership_v1", component_limits_seconds={
            key: value * 3600 for key, value in authorization["budget_by_component"].items()},
            total_limit_seconds=authorization["total_GPU_hours"] * 3600,
            technical_receipts=receipt_sources, technical_windows=windows,
            tasks={}, cpu_windows=[], counter_reserved=dict(raws=0, trajectories=0, KD_updates=0, raw_full8=0, node_full8=0))
        technical = sum(row["ended_epoch"] - row["started_epoch"] for row in windows)
        require(technical <= self.value["component_limits_seconds"]["technical"], "Actual technical GPU cost exceeded its approved component.")
        self.value["technical_actual_seconds"] = technical
        if previous is not None:
            limits_ok = (previous['component_limits_seconds']==self.value['component_limits_seconds']
                and previous['total_limit_seconds']==self.value['total_limit_seconds'])
            if new_batch:
                require(authorization.get('batch') == 'remaining_and_replay' and retry_tasks == (),
                    'A new batch must explicitly preserve failed tasks without retrying them.')
                limits_ok = all(self.value['component_limits_seconds'].get(k,0)>=v
                    for k,v in previous['actual_seconds_by_component'].items())
            require(limits_ok
                and windows[:len(previous['technical_windows'])]==previous['technical_windows']
                and not any(t['status']=='running' for t in previous['tasks'].values()),
                'Continuation must preserve the original limits and settled ownership windows.')
            self.value.update(tasks=deepcopy(previous['tasks']), counter_reserved=deepcopy(previous['counter_reserved']),
                cpu_windows=deepcopy(previous['cpu_windows']))
            for name, task in list(self.value['tasks'].items()):
                if task['status']=='failed' and (retry_tasks is None or name in retry_tasks):
                    suffix='/interrupted_20261108' if retry_tasks is None else '/failed_before_'+self.path.parent.name
                    require(name+suffix not in self.value['tasks'], 'Do not overwrite a previous failed ownership window.')
                    self.value['tasks'][name+suffix]=task
                    self.value['tasks'][name]=dict(component=task['component'],seconds=task['seconds'],status='reserved')
            self.value['continued_budget_actual_GPUh']=previous['total_actual_GPU_hours']
            self.value['previous_component_limits_seconds']=previous['component_limits_seconds']
        self._save()
        if previous is not None:
            added=(technical-previous['technical_actual_seconds'])/3600
            require(abs(self.value['total_actual_GPU_hours']-previous['total_actual_GPU_hours']-added)<1e-10,
                'Continuation must carry all actual fees once.')

    def _save(self):
        charged = dict.fromkeys(self.value["component_limits_seconds"], 0.)
        charged["technical"] = self.value["technical_actual_seconds"]
        reserved = dict.fromkeys(charged, 0.)
        for task in self.value["tasks"].values():
            if task["status"] in ("reserved", "running"):
                reserved[task["component"]] += task["seconds"]
            else:
                charged[task["component"]] += task["actual_seconds"]
        self.value.update(actual_seconds_by_component=charged, reserved_seconds_by_component=reserved,
            total_actual_GPU_hours=sum(charged.values()) / 3600,
            total_reserved_GPU_hours=sum(reserved.values()) / 3600)
        write(self.path, self.value, replace=self.path.exists())

    def reserve(self, tasks, *, counts):
        with self.lock:
            require(not set(tasks) & set(self.value["tasks"]), "A task cannot be retried or reserved twice.")
            for key, count in counts.items():
                maximum = self.authorization["maximum_" + key]
                require(type(count) is int and count >= 0 and self.value["counter_reserved"].get(key,0) + count <= maximum,
                    "The finite approved count is exhausted: " + key)
            requested = dict.fromkeys(self.value["component_limits_seconds"], 0.)
            for task in tasks.values():
                require(task["component"] in requested and task["seconds"] > 15, "Use a concrete bounded GPU task.")
                requested[task["component"]] += task["seconds"]
            for key, maximum in self.value["component_limits_seconds"].items():
                require(self.value["actual_seconds_by_component"][key] + self.value["reserved_seconds_by_component"][key]
                    + requested[key] <= maximum, "Insufficient full-dependency component budget: " + key)
            require((self.value["total_actual_GPU_hours"] + self.value["total_reserved_GPU_hours"]) * 3600
                + sum(requested.values()) <= self.value["total_limit_seconds"], "Insufficient approved full-dependency budget.")
            self.value["tasks"].update({key: dict(value, status="reserved") for key, value in tasks.items()})
            for key, count in counts.items():
                self.value["counter_reserved"][key] = self.value["counter_reserved"].get(key,0) + count
            self._save()

    def start(self, task_id, *, gpu, start, deadline, command, log, request):
        with self.lock:
            task = self.value["tasks"][task_id]
            require(task["status"] == "reserved", "A reserved task may start exactly once.")
            require(gpu in self.authorization["gpus"] == [4, 5, 6, 7], "Use only the four approved GPUs.")
            require(not any(row["status"] == "running" and row["gpu"] == gpu for row in self.value["tasks"].values()), "GPU ownership windows cannot overlap.")
            task.update(status="running", gpu=gpu, started_epoch=start, deadline_epoch=deadline,
                command=command, log=str(log), request=binding(request))
            self._save()

    def attach_pid(self, task_id, pid):
        with self.lock:
            self.value["tasks"][task_id]["pid"] = pid
            self._save()

    def finish(self, task_id, *, end, exit_code, error=None):
        with self.lock:
            task = self.value["tasks"][task_id]
            require(task["status"] == "running", "Charge each actual task ownership window exactly once.")
            task.update(status="completed" if exit_code == 0 and error is None else "failed", ended_epoch=end,
                exit_code=exit_code, error=error, actual_seconds=end - task["started_epoch"])
            self._save()
            require(task["actual_seconds"] <= task["seconds"], "An actual ownership window exceeded its reserved maximum.")

    @contextmanager
    def cpu_window(self, name):
        start = time.time()
        try:
            yield
        finally:
            with self.lock:
                self.value["cpu_windows"].append(dict(name=name, started_epoch=start,
                    ended_epoch=time.time(), actual_seconds=time.time() - start, GPU_hours=0))
                self._save()


def command(module, request_path):
    require(module in ('worker', 'evaluate', 'terminal'), 'Use the isolated new-batch workers.')
    return [sys.executable, '-B', '-u', '-m', 'experiments.sc_cat_recovery_cmcc_20261110.'+module,
        '--request', str(Path(request_path).resolve())]

class Processes:
    def __init__(self, root, budget, sources, reused=None):
        self.root, self.budget = Path(root), budget
        self.sources = sources
        self.gpus = queue.Queue()
        for gpu in (4,5,6,7):
            self.gpus.put(gpu)
        self.cancelled = threading.Event()
        self.children, self.lock = {}, threading.RLock()
        self.reused = reused or {}

    def record_failure(self, task_id, error):
        with self.lock:
            path = self.root/'first_failure.json'
            if not path.exists():
                write(path, dict(task_id=task_id, error=type(error).__name__+': '+str(error), epoch=time.time()))

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for child in self.children.values():
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def _stop_evaluation_shards(self, request):
        """The original evaluator creates separate shard groups; own those until they exit."""
        output = Path(request["output"]).resolve()
        record = output / "evaluation.json"
        if not record.is_file():
            return
        value = read(record)
        plan = str(output / "plan.json")
        for stage in value.get("stages", []):
            pid = stage["process_group_id"]
            try:
                arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                require(os.getpgid(pid) == pid and os.fsencode(plan) in arguments,
                    "A recorded evaluator process identity changed before cleanup.")
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass

    def run(self, task_id, module, request, *, output_file):
        if task_id in self.reused:
            from .continuation import reuse_completed
            return reuse_completed(self.reused[task_id], request)
        pool = self.gpus
        while True:
            require(not self.cancelled.is_set(), "A dependency failed; no new GPU task may start.")
            try:
                gpu = pool.get(timeout=.2)
                break
            except queue.Empty:
                continue
        try:
            require(not self.cancelled.is_set(), "A dependency failed; no new GPU task may start.")
            verify_sources(self.sources)
            resources = check_resources([gpu])
            require(shutil.disk_usage(self.root).free >= 20*1024**3,
                f'Less than20GiB free at {self.root}: {shutil.disk_usage(self.root).free} bytes; task={task_id}.')
            task = self.budget.value["tasks"][task_id]
            folder = self.root / "execution" / task_id
            folder.mkdir(parents=True, exist_ok=False)
            start = time.time()
            deadline = start + task["seconds"] - 15
            request = dict(request, gpu=gpu, deadline_epoch=deadline)
            request_path, log = folder / "request.json", folder / "worker.log"
            write(request_path, request)
            argv = command(module, request_path)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if module == "evaluate" else str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
            self.budget.start(task_id, gpu=gpu, start=start, deadline=deadline, command=argv, log=log, request=request_path)
            with self.budget.lock:
                self.budget.value["tasks"][task_id]["resource_check"] = resources
                self.budget._save()
            child, code, error = None, -1, None
            try:
                with log.open("x", encoding="utf-8") as stream:
                    with self.lock:
                        require(not self.cancelled.is_set(), "A dependency failed before launch.")
                        child = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[2], env=env,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        self.children[task_id] = child
                        self.budget.attach_pid(task_id, child.pid)
                    while True:
                        require(not self.cancelled.is_set(), "A dependency failed; terminate this batch-owned worker.")
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(argv, task["seconds"])
                        try:
                            code = child.wait(timeout=min(1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    require(code == 0, "Worker failed with exit_code=" + str(code) + ": " + task_id)
                    require(Path(output_file).is_file(), "Worker exited without its required artifact: " + str(output_file))
            except BaseException as exc:
                self.record_failure(task_id,exc)
                error = type(exc).__name__ + ": " + str(exc)
                if child is not None and child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                code = child.returncode if child is not None else -1
                self.cancel()
                raise
            finally:
                if module == "evaluate":
                    self._stop_evaluation_shards(request)
                self.budget.finish(task_id, end=time.time(), exit_code=code, error=error)
                with self.lock:
                    self.children.pop(task_id, None)
            return binding(output_file)
        except BaseException as exc:
            self.record_failure(task_id,exc)
            raise
        finally:
            pool.put(gpu)

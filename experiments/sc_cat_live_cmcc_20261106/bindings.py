"""Frozen inputs, complete dependency reservations and finite GPU ownership."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


PACKAGE = "experiments.sc_cat_live_cmcc_20261106"
ORDER = ("k_proj", "o_proj", "gate_proj", "q_proj", "v_proj", "down_proj", "up_proj")
COMPONENT_HOURS = dict(technical=4, fit=6, recovery=16, evaluation=9, boundary=1)
APPROVAL = "call_c5f4a5881b534505b5a2c8ce928286c1"
FIT_SECONDS = 4500
RAW_EXPORT_SECONDS = 900
EXPORT_SECONDS = 450
EVALUATION_SECONDS = 900


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), "Do not overwrite an existing artifact: " + str(path))
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def binding(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha)


def bound_path(reference):
    if isinstance(reference, str) or isinstance(reference, Path):
        return Path(reference).resolve()
    path = Path(reference["path"]).resolve()
    require(binding(path)["sha256"] == reference["sha256"], "Frozen input changed: " + str(path))
    return path


def load_batch(authorization_path, root_boundary_path):
    authorization_path, root_boundary_path = Path(authorization_path).resolve(), Path(root_boundary_path).resolve()
    authorization = read(authorization_path)
    require(authorization.get("confirmed") is True and authorization.get("user_approved") is True
        and authorization["authorization_reference"] == APPROVAL
        and authorization["categories"] == list(ORDER[:2])
        and authorization["frozen_order"] == list(ORDER)
        and authorization["gpus"] == list(range(6)) and authorization["protected_gpus"] == [6, 7]
        and authorization["total_GPU_hours"] == 36
        and authorization["budget_by_component"] == COMPONENT_HOURS,
        "Require the exact confirmed finite36GPUh k/o batch.")
    for name, value in dict(maximum_raws=4, maximum_trajectories=8,
            maximum_KD_updates=4000, maximum_raw_full8=4, maximum_node_full8=32).items():
        require(authorization[name] == value, "The approved batch count changed: " + name)
    root = Path(authorization["output_root"]).resolve()
    require(authorization_path.is_relative_to(root) and root_boundary_path.is_relative_to(root), "Bind this isolated authorized root.")
    for name in ("protocol", "execution_protocol"):
        bound_path(authorization[name])
    manifest = read(bound_path(authorization["source_manifest"]))
    selected = read(bound_path(authorization["selected_settings"]))
    require(selected["status"] == "completed" and selected["protocol_identity"] == manifest["protocol_identity"]
        and selected["run_identity"] == manifest["run_identity"], "Use the completed frozen offline selection and its original identity.")
    boundary = read(root_boundary_path)
    require(boundary["completed_prefix"] == [] and boundary["phase"] == "formal"
        and boundary["formal_eligible"] and boundary["uuid"], "Both k branches must start from the same actual formal BF16 root.")
    inventory = read(bound_path(manifest["inventory"]))["inventory"]
    bound_path(manifest["data_plan"])
    bound_path(manifest["baseline"])
    recipes = {}
    for category in ORDER[:2]:
        row = next(item for item in selected["categories"] if item["category_id"] == category)
        references = (row["anchor"], row["best_different"]["result"])
        originals = [read(bound_path(reference)) for reference in references]
        require(originals[0]["candidate_id"] != originals[1]["candidate_id"]
            and originals[0]["recipe"] != originals[1]["recipe"], "This batch has exactly two actually different frozen live recipes.")
        recipes[category] = [dict(is_anchor=index == 0, candidate_id=original["candidate_id"],
            recipe=original["recipe"], source=references[index]) for index, original in enumerate(originals)]
    return dict(authorization=authorization, authorization_reference=binding(authorization_path),
        root=root, root_boundary=boundary, root_boundary_reference=binding(root_boundary_path),
        manifest=manifest, selected=selected, inventory=inventory, recipes=recipes)


def technical_windows(paths):
    """Union actual per-device ownership, so outer validation receipts include children once."""
    rows, sources = [], []
    for path in paths:
        path = Path(path).resolve()
        source = read(path)
        sources.append(binding(path))
        entries = source.get("ownership_windows", [source])
        for row in entries:
            end_value = row.get("ended_epoch", row.get("finished_epoch"))
            require(row["status"] != "running" and end_value is not None,
                "A technical GPU window is still active: " + str(path))
            start, end, gpu = float(row["started_epoch"]), float(end_value), row["gpu"]
            require(type(gpu) is int and gpu in range(6) and start < end <= time.time() + 1,
                "Technical ownership requires real finite actual timestamps and an authorized GPU.")
            rows.append(dict(gpu=gpu, started_epoch=start, ended_epoch=end,
                exit_code=row.get("exit_code"), source=str(path)))
    merged = []
    for gpu in range(6):
        windows = sorted((row for row in rows if row["gpu"] == gpu), key=lambda row: row["started_epoch"])
        for row in windows:
            if merged and merged[-1]["gpu"] == gpu and row["started_epoch"] <= merged[-1]["ended_epoch"]:
                merged[-1]["ended_epoch"] = max(merged[-1]["ended_epoch"], row["ended_epoch"])
                merged[-1]["sources"].append(row["source"])
            else:
                merged.append(dict(gpu=gpu, started_epoch=row["started_epoch"], ended_epoch=row["ended_epoch"], sources=[row["source"]]))
    require(rows and any(row["exit_code"] == 0 for row in rows), "Actual successful technical ownership evidence is required.")
    return merged, sources


def raw_tasks(prefix):
    return {prefix + "/fit": dict(component="fit", seconds=FIT_SECONDS),
        prefix + "/raw_export": dict(component="fit", seconds=RAW_EXPORT_SECONDS),
        prefix + "/raw_full8": dict(component="evaluation", seconds=EVALUATION_SECONDS)}


def trajectory_tasks(prefix, settings):
    tasks = {prefix + "/train": dict(component="recovery", seconds=5400 * settings.schedule_max // 500)}
    for step in settings.node_steps:
        tasks[prefix + f"/node_{step}/export"] = dict(component="recovery", seconds=EXPORT_SECONDS)
        tasks[prefix + f"/node_{step}/full8"] = dict(component="evaluation", seconds=EVALUATION_SECONDS)
    return tasks


class Budget:
    """Persist before launch; reserve all dependency windows; charge failed ownership too."""
    def __init__(self, path, authorization, windows, receipt_sources):
        self.path, self.authorization = Path(path), authorization
        self.lock = threading.RLock()
        require(not self.path.exists(), "Existing budget is evidence, not authorization for an automatic retry.")
        self.value = dict(schema="sc_cat_live_actual_ownership_v1", component_limits_seconds={
            key: value * 3600 for key, value in authorization["budget_by_component"].items()},
            total_limit_seconds=authorization["total_GPU_hours"] * 3600,
            technical_receipts=receipt_sources, technical_windows=windows,
            tasks={}, cpu_windows=[], counter_reserved=dict(raws=0, trajectories=0, KD_updates=0, raw_full8=0, node_full8=0))
        technical = sum(row["ended_epoch"] - row["started_epoch"] for row in windows)
        require(technical <= self.value["component_limits_seconds"]["technical"], "Actual technical GPU cost exceeded its4h component.")
        self.value["technical_actual_seconds"] = technical
        self._save()

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
                require(type(count) is int and count >= 0 and self.value["counter_reserved"][key] + count <= maximum,
                    "The finite approved count is exhausted: " + key)
            requested = dict.fromkeys(self.value["component_limits_seconds"], 0.)
            for task in tasks.values():
                require(task["component"] in requested and task["seconds"] > 15, "Use a concrete bounded GPU task.")
                requested[task["component"]] += task["seconds"]
            for key, maximum in self.value["component_limits_seconds"].items():
                require(self.value["actual_seconds_by_component"][key] + self.value["reserved_seconds_by_component"][key]
                    + requested[key] <= maximum, "Insufficient full-dependency component budget: " + key)
            require((self.value["total_actual_GPU_hours"] + self.value["total_reserved_GPU_hours"]) * 3600
                + sum(requested.values()) <= self.value["total_limit_seconds"], "Insufficient full-dependency36GPUh budget.")
            self.value["tasks"].update({key: dict(value, status="reserved") for key, value in tasks.items()})
            for key, count in counts.items():
                self.value["counter_reserved"][key] += count
            self._save()

    def start(self, task_id, *, gpu, start, deadline, command, log, request):
        with self.lock:
            task = self.value["tasks"][task_id]
            require(task["status"] == "reserved", "A reserved task may start exactly once.")
            require(gpu in ((0, 1) if task_id.endswith(("/fit", "/train")) else (2, 3, 4, 5)), "Respect the fixed fit/train and export/evaluation GPU partition.")
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
    require(module in ("fit_worker", "recovery_worker", "evaluate"), "Run only one of the three concrete batch workers.")
    return [sys.executable, "-B", "-u", "-m", PACKAGE + "." + module, "--request", str(Path(request_path).resolve())]


def frozen_sources(manifest):
    from experiments.cat_auto_protocol_cmcc_20261104.live import source_pins
    sources = dict(source_pins())
    for name in ("code_identity", "consumer_identity"):
        sources.update(manifest["run_identity"][name]["source_sha256"])
    sources.update({str(path.resolve()): binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")})
    verify_sources(sources)
    return sources


def verify_sources(sources):
    for path, sha in sources.items():
        require(binding(path)["sha256"] == sha, "A frozen formal runtime source changed: " + path)


def check_resources(gpus):
    from experiments.cmcc_runtime_20261030.sequence import check_resources as original
    return original([str(gpu) for gpu in gpus])


def readout_devices(gpus):
    gpus = tuple(gpus)
    require(gpus and len(set(gpus)) == len(gpus)
        and all(type(gpu) is int and gpu in (2, 3, 4, 5) for gpu in gpus),
        "Choose a nonempty unique readout GPU subset of2,3,4,5.")
    return gpus


class Processes:
    def __init__(self, root, budget, sources, readout_gpus=(2, 3, 4, 5)):
        self.root, self.budget = Path(root), budget
        self.sources = sources
        self.gpus = {"train": queue.Queue(), "readout": queue.Queue()}
        for gpu in (0, 1):
            self.gpus["train"].put(gpu)
        for gpu in readout_devices(readout_gpus):
            self.gpus["readout"].put(gpu)
        self.cancelled = threading.Event()
        self.children, self.lock = {}, threading.RLock()

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
        train = task_id.endswith(("/fit", "/train"))
        pool = self.gpus["train" if train else "readout"]
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
        finally:
            pool.put(gpu)

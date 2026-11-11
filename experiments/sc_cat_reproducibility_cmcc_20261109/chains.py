"""Two independent, fixed-recipe BF16 replays using the frozen legacy consumers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
import types

from experiments.sc_cat_continue_cmcc_20261107 import bindings as io
from experiments.sc_cat_continue_cmcc_20261107 import run as original
from experiments.sc_cat_finish_cmcc_20261108 import frozen
from experiments.sc_cat_finish_cmcc_20261108.worker import wrapped_command


PACKAGE = "experiments.sc_cat_reproducibility_cmcc_20261109"


class Processes(original.Processes):
    def __init__(self, root, budget, pins, snapshot, train_gpu, readout_gpu):
        super().__init__(root, budget, pins, (readout_gpu,))
        self.gpus["train"] = queue.Queue()
        self.gpus["train"].put(train_gpu)

        def command(module, request_path):
            argv = wrapped_command(original.command(module, request_path), snapshot)
            argv[argv.index("-m") + 1] = PACKAGE + ".worker"
            return argv

        function = original._finite_run
        self.execute = types.FunctionType(function.__code__, dict(function.__globals__, command=command),
            function.__name__, function.__defaults__, function.__closure__)

    def run(self, task_id, module, request, *, output_file):
        io.require(request["stage"] == "replay" and task_id.startswith("replay/"),
            "This batch permits only fixed-recipe replay, never search or recovery tuning.")
        return self.execute(self, task_id, module, request, output_file=output_file)


class PairedResults:
    """Compare actual completed boundaries; UUID differences are intentional."""
    def __init__(self, root):
        self.path = root / "paired_results.json"
        self.lock = threading.RLock()
        self.rows = {}

    def record(self, chain, row, score_reference):
        raw = io.read(io.bound_path(row["raw"]))
        report = io.read(io.bound_path(row["deployment"]))
        score = io.read(io.bound_path(score_reference))
        evidence = dict(raw=row["raw"], deployment=row["deployment"], score=score_reference,
            fit_report=raw["fit_report"], state_digest=report["export_evidence"]["exported_state_digest"],
            task_scores=score["task_scores"], mean_percent=score["mean_percent"],
            selected_step=row["selected_step"], schedule_max=row["schedule_max"])
        with self.lock:
            category = row["category_id"]
            pair = self.rows.setdefault(category, {})
            io.require(chain not in pair, "Record each completed chain/category exactly once.")
            pair[chain] = evidence
            summary = {}
            for key, values in self.rows.items():
                summary[key] = {name: {k: v for k, v in value.items() if k != "fit_report"}
                    for name, value in values.items()}
                if set(values) == {"a", "b"}:
                    a, b = values["a"], values["b"]
                    summary[key]["comparison"] = dict(
                        fit_report_equal=a["fit_report"] == b["fit_report"],
                        exported_tensor_digest_equal=a["state_digest"] == b["state_digest"],
                        eight_task_scores_equal=a["task_scores"] == b["task_scores"],
                        S8_b_minus_a_pp=b["mean_percent"] - a["mean_percent"],
                        source_reports_retained_for_first_divergence_analysis=True)
            io.write(self.path, dict(categories=summary,
                policy="Record differences without selecting a winner or silently retrying; inspect any remaining divergence."),
                replace=self.path.exists())


class Controller(original.Controller):
    def __init__(self, plan, chain, approval, pairs):
        self.name, self.pairs = chain, pairs
        spec = plan["chains"][chain]
        self.root = Path(spec["root"])
        io.require(not self.root.exists(), "Never overwrite or automatically restart a replay.")
        self.root.mkdir(parents=True)
        authorization = deepcopy(spec["authorization_template"])
        authorization.update(confirmed=True, user_approved=True, not_authorization=False, status="approved",
            authorization_reference=approval["authorization_reference"], approval_user_message=approval["user_message"])
        io.write(self.root / "authorization.json", authorization)
        self.authorization_reference = io.binding(self.root / "authorization.json")
        self.authorization = io.read_authorization(self.authorization_reference)
        self.pins = plan["source_pins"]
        manifest = io.read(io.bound_path(authorization["source_manifest"]))
        prefix_recipe = io.read(io.bound_path(authorization["prefix"]["frozen_recipe"]))
        self.prefix = dict(references=authorization["prefix"], frozen=prefix_recipe,
            root=io.read(io.bound_path(prefix_recipe["shared_BF16_root"])))
        self.batch = dict(manifest=manifest,
            inventory=io.read(io.bound_path(manifest["inventory"]))["inventory"])
        self.common = dict(authorization=self.authorization_reference, source_manifest=authorization["source_manifest"],
            inventory_receipt=manifest["inventory"], data_plan_path=manifest["data_plan"], phase="formal", stage="replay",
            execution_policy=plan["execution_policy"])
        self.budget = io.Budget(self.root / "budget.json", authorization, prior_GPU_hours=0)
        self.processes = Processes(self.root, self.budget, self.pins, io.bound_path(plan["snapshot"]),
            spec["train_gpu"], spec["readout_gpu"])
        self.state_lock, self.deployments = threading.RLock(), {}
        self.state = dict(status="initialized", controller_pid=os.getpid(), started_epoch=time.time(),
            chain=chain, source_pins=self.pins, authorization=self.authorization_reference, replay_stages=[],
            execution_change="FLASH_ATTENTION_DETERMINISTIC=1 before importing Transformers",
            train_gpu=spec["train_gpu"], readout_gpu=spec["readout_gpu"])
        self._persist("initialized_independent_BF16_replay")

    def _persist(self, event):
        with self.state_lock:
            self.state.update(last_event=event, updated_epoch=time.time())
            path = self.root / "controller_state.json"
            io.write(path, self.state, replace=path.exists())
        if event == "fixed_recipe_replay_stage_actual_prefix_completed":
            row = self.state["replay_stages"][-1]
            self.pairs.record(self.name, row, io.binding(self.root / "replay" / row["category_id"] / "full8" / "score.json"))
        print(dict(chain=self.name, event=event, status=self.state["status"]), flush=True)

    def run_fixed(self, plan):
        try:
            self.state["status"] = "running"
            self._replay(plan["frozen_recipe"], plan["selected_product"])
            self.state.update(status="completed", exit_code=0, ended_epoch=time.time())
            self._persist("complete_fixed_recipe_and_nine_tasks")
        except BaseException as exc:
            self.processes.cancel()
            self.state.update(status="failed", exit_code=1, ended_epoch=time.time(),
                error=repr(exc), traceback=traceback.format_exc())
            self._persist("failure_no_retry")
            raise
        finally:
            io.write(self.root / "controller.exit.json", dict(status=self.state["status"],
                exit_code=self.state["exit_code"], state=io.binding(self.root / "controller_state.json"),
                actual_GPU_hours=self.budget.value["total_actual_GPU_hours"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Use bitvae with controller CUDA hidden.")
    plan_reference = io.binding(args.plan)
    plan, approval = io.read(args.plan), io.read(args.approval)
    io.require(approval.get("confirmed") is True and approval.get("plan") == plan_reference
        and approval.get("total_GPU_hours") == plan["total_GPU_hours"]
        and approval.get("gpus") == [0, 1, 2, 3] and approval.get("user_message")
        and approval.get("authorization_reference"), "Require actual approval of this exact new batch and budget.")
    frozen.install(io.bound_path(plan["snapshot"]))
    io.base.verify_sources(plan["source_pins"])
    io.base.check_resources((0, 1, 2, 3))
    import shutil
    io.require(shutil.disk_usage(plan["output_root"]).free >= plan["storage_reserve_bytes"],
        "Retain enough space for both independent chains and their diagnostic evidence.")
    root = Path(plan["output_root"])
    io.require(not (root / "controller.exit.json").exists() and not (root / "paired_results.json").exists(),
        "No overwrite or automatic retry.")
    pairs = PairedResults(root)
    controllers = [Controller(plan, chain, approval, pairs) for chain in ("a", "b")]

    def interrupted(signum, _frame):
        for controller in controllers:
            controller.processes.cancel()
        raise InterruptedError("Received signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(controller.run_fixed, plan) for controller in controllers]
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                errors.append(repr(exc))
    result = dict(status="failed" if errors else "completed", exit_code=int(bool(errors)), errors=errors,
        chains={controller.name: io.binding(controller.root / "controller_state.json") for controller in controllers},
        actual_GPU_hours=sum(controller.budget.value["total_actual_GPU_hours"] for controller in controllers))
    if not errors:
        result["nine_tasks"] = {controller.name: controller.state["replay_nine_task"] for controller in controllers}
    io.write(root / "controller.exit.json", result)
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()

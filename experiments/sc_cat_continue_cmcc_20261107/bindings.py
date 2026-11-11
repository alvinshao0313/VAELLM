"""Actual prefix, new authorization, category-specific deadlines and shared accounting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
import math
from pathlib import Path
import threading
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.sc_cat_live_cmcc_20261106 import bindings as base
from experiments.sc_cat_live_cmcc_20261106 import live_io
from experiments.sc_cat_live_cmcc_20261106.boundary import StrictMainChain
from experiments.sc_cat_live_cmcc_20261106.selection import SelectedCategory
from experiments.sc_cat_live_cmcc_20261106.run import restore_quota


ORDER, CATEGORIES = base.ORDER, base.ORDER[2:]
COMPONENTS = ("technical", "core", "terminal_longbench", "replay", "reference")
CORE_COUNTS = dict(raws=10, trajectories=15, KD_updates=8500, raw_full8=10, node_full8=68, terminal_longbench=2)
REPLAY_COUNTS = dict(replay_fits=7, replay_exports=7, replay_full8=7, replay_longbench=1, replay_KD_updates=3875)
LONG_PARTITION_ONE = ("narrativeqa", "qasper", "2wikimqa", "gov_report", "vcsum", "triviaqa", "samsum", "passage_count", "lcc")
read, write, binding, bound_path = base.read, base.write, base.binding, base.bound_path


def validate_plan(value):
    """Structural/cost contract; this does not grant permission to execute."""
    require(value["schema"] == "sc_cat_continuation_authorization_v1"
        and value["categories"] == list(CATEGORIES) and value["frozen_order"] == list(ORDER), "Keep exactly the frozen gate→q→v→down→up continuation.")
    require(value["train_gpus"] == [0, 1] and value["protected_gpus"] == [6, 7]
        and base.readout_devices(value["readout_gpus"]) and value["gpus"] == sorted({0, 1, *value["readout_gpus"]}),
        "Authorize explicit train/readout devices and protect6/7.")
    require(set(value["budget_by_component"]) == set(COMPONENTS)
        and all(type(number) in (int, float) and math.isfinite(number) and number >= 0 for number in value["budget_by_component"].values())
        and sum(value["budget_by_component"].values()) <= value["total_GPU_hours"] and value["total_GPU_hours"] > 0,
        "Declare every new phase budget without transferring prior36GPUh.")
    require(set(value["stages"]) <= set(COMPONENTS) and {"core", "terminal_longbench", "replay"} <= set(value["stages"])
        and value["reference_mode"] == "reuse_only", "Core, terminal selection and fixed-recipe replay are explicit; reference only reuses verified evidence.")
    for key, count in {**CORE_COUNTS, **REPLAY_COUNTS}.items():
        require(value["maximum_" + key] == count, "The approved fixed task/update upper bound changed: " + key)
    limits = value["task_seconds"]
    require(math.isfinite(limits["replay_train_startup_seconds"]) and limits["replay_train_startup_seconds"] >= 0
        and all(math.isfinite(limits["replay_seconds_per_update_by_category"][category])
            and limits["replay_seconds_per_update_by_category"][category] > 0 for category in ORDER),
        "Declare finite replay startup and per-category seconds per real selected update.")
    for category in ORDER:
        for kind in (("fit_by_category", "train500_by_category", "train1000_by_category") if category in CATEGORIES else ("fit_by_category",)):
            require(type(limits[kind][category]) in (int, float) and limits[kind][category] > 15,
                "Declare an actual category-specific finite task window: " + kind + "/" + category)
    for kind in ("raw_export", "node_export", "replay_export", "full8", "longbench", "longbench_shard"):
        require(type(limits[kind]) in (int, float) and limits[kind] > 15, "Declare the finite task window: " + kind)
    require(limits["full8"] == 900, "Keep the original full-eight evaluator's15minute window.")
    require(limits["longbench"] == 2 * limits["longbench_shard"], "Two declared complete-task shards share one actual artifact LongBench budget.")
    for key in ("after_o", "frozen_recipe", "quotas", "controller_state", "controller_exit"):
        require(isinstance(value["prefix"][key], dict) and value["prefix"][key].get("sha256"), "Bind every actual completed-prefix input.")
    return value


def read_authorization(reference):
    path = bound_path(reference)
    value = validate_plan(read(path))
    require(value.get("confirmed") is True and value.get("user_approved") is True
        and value.get("not_authorization") is False and value.get("status") != "plan_draft_not_authorization"
        and value.get("phase") == "formal" and isinstance(value.get("authorization_reference"), str)
        and value["authorization_reference"].strip()
        and value["authorization_reference"] != live_io.APPROVAL,
        "A new actual explicit user approval is required; the old k/o authorization cannot authorize this continuation.")
    for name in ("source_manifest", "selected_settings", "protocol", "execution_protocol"):
        bound_path(value[name])
    return value


def load_prefix(references):
    values = {key: read(bound_path(reference)) for key, reference in references.items()}
    parent, frozen, quota_data, state, terminal = (values[key] for key in ("after_o", "frozen_recipe", "quotas", "controller_state", "controller_exit"))
    require(terminal["status"] == "completed" and terminal["exit_code"] == 0
        and bound_path(terminal["state"]) == bound_path(references["controller_state"])
        and parent["formal_eligible"] and parent["phase"] == "formal" and parent["completed_prefix"] == list(ORDER[:2])
        and frozen["completed_categories"] == quota_data["selected_categories"] == list(ORDER[:2])
        and bound_path(frozen["final_boundary"]) == bound_path(references["after_o"])
        and bound_path(frozen["quotas"]) == bound_path(references["quotas"]), "Use the actual completed and unchanged after_o boundary and its own frozen recipe/quotas.")
    quota = restore_quota(references["quotas"])
    require(quota.categories == ORDER and quota.interaction_categories == list(ORDER[:2]) and quota.long_categories == []
        and len(quota.started) == 8 and quota.counts()["actual_node_evaluations"] == 32,
        "Inherit both consumed interaction slots, both unspent long slots and all eight prior trajectories.")
    root = read(bound_path(frozen["shared_BF16_root"]))
    chain = StrictMainChain(bf16_root_id=root["uuid"], categories=ORDER)
    require(len(state["chain"]) == 2 and state["chain"][-1]["boundary_id"] == parent["uuid"], "The frozen predecessor must be this exact strict main-chain prefix.")
    for index, category in enumerate(ORDER[:2]):
        prior = state["chain"][index]
        selected = SelectedCategory(**dict(prior["selected"], settings=RecoverySettings(**prior["selected"]["settings"])))
        require(asdict(selected) == frozen["categories"][index], "The predecessor recipe and selected sources disagree.")
        certificate = read(bound_path(state["categories"][category]["selection_certificate"]))
        committed = chain.commit(selected, live_parent_id=chain.live_parent_id, boundary_reference=prior["boundary_reference"],
            selection_certificate=certificate, actual_round_base_id=selected.source_round_base_id,
            restored_node_id=selected.source_node_id, runtime_identity=prior["runtime_identity"])
        require(committed == prior, "Reconstructed prefix provenance differs from the actual committed chain.")
    require(chain.live_parent_id == parent["uuid"] and state["chain"][-1]["boundary_reference"] == parent,
        "Only the exact measured after_o artifact may parent gate.")
    return dict(parent=parent, frozen=frozen, quota=quota, chain=chain, state=state, root=root,
        references=deepcopy(references), prior_actual_GPU_hours=terminal["actual_GPU_hours"])


def seconds(authorization, kind, *, category=None, settings=None, selected_step=None):
    limits = authorization["task_seconds"]
    if kind == "fit":
        return limits["fit_by_category"][category]
    if kind == "train":
        return limits[f"train{settings.schedule_max}_by_category"][category]
    if kind == "replay_train":
        require(type(selected_step) is int and selected_step > 0, "Zero-KD replay has no training reservation.")
        return limits["replay_train_startup_seconds"] + limits["replay_seconds_per_update_by_category"][category] * selected_step
    return limits[kind]


def validate_worker_request(request, *, kind):
    authorization = read_authorization(request["authorization"])
    stages = dict(fit=("core", "replay"), train=("core",), raw_export=("core", "replay"),
        node_export=("core", "replay"), full8=("core", "replay"), longbench=("terminal_longbench", "replay"),
        replay_train=("replay",), replay_export=("replay",), replay=("replay",))
    require(kind in stages and request["stage"] in stages[kind], "The actual worker kind cannot move into another budget phase or start a new reference chain.")
    require(request["phase"] == "formal" and request["stage"] in authorization["stages"]
        and type(request["gpu"]) is int and request["gpu"] in authorization["gpus"]
        and time.time() < request["deadline_epoch"], "Use the actual new approved phase/device and original finite deadline.")
    output = Path(request["output"]).resolve()
    require(output.is_relative_to(Path(authorization["output_root"]).resolve()), "Write only this new isolated authorized output root.")
    training_kind = kind in ("fit", "train", "replay_train") or kind == "replay" and request.get("operation") == "train_replay"
    if training_kind:
        require(request["gpu"] in authorization["train_gpus"], "Fit/train only on the explicitly authorized training cards.")
    else:
        require(request["gpu"] in authorization["readout_gpus"], "Readout only on the explicitly authorized readout cards.")
    if request["stage"] == "core" and request.get("category_id"):
        require(request["category_id"] in CATEGORIES, "Core continuation cannot refit the selected k/o prefix.")
    if kind == "fit":
        maximum = seconds(authorization, "fit", category=request["category_id"])
    elif kind == "train":
        maximum = seconds(authorization, "train", category=request["category_id"], settings=RecoverySettings(**request["settings"]))
    elif kind == "replay_train" or kind == "replay" and request.get("operation") == "train_replay":
        maximum = seconds(authorization, "replay_train", category=request["category_id"],
            settings=RecoverySettings(**request["settings"]), selected_step=request["selected_step"])
    else:
        limit_kind = "replay_export" if kind in ("replay", "replay_export") else "longbench_shard" if kind == "longbench" else kind
        maximum = seconds(authorization, limit_kind)
    require(request["deadline_epoch"] - time.time() <= maximum, "The worker's original finite window exceeds the actual declared category/task cap.")
    return authorization


def validate_worker(request, component):
    return validate_worker_request(request, kind=component)


def runtime_pins(manifest):
    pins = base.frozen_sources(manifest)
    repo = Path(__file__).resolve().parents[2]
    paths = ("experiments/sc_cat_pipeline_cmcc_20261106/run.py",
        "experiments/sc_cat_pipeline_cmcc_20261106/node_export.py",
        "experiments/sc_cat_pipeline_cmcc_20261106/handoff.py",
        "experiments/sc_cat_handoff_cmcc_20261106/run.py",
        "experiments/cmcc_runtime_20261030/sequence.py",
        "experiments/compare_longbench_v1.py",
        *("experiments/longbench_v1/" + name + ".py" for name in ("runner", "models", "common", "data", "protocol", "scoring")),
        "experiments/cat_offline_execution_cmcc_20261104/evaluate.py",
        "experiments/cat_offline_execution_cmcc_20261104/protocol.py")
    pins.update({str((repo / name).resolve()): binding(repo / name)["sha256"] for name in paths})
    pins.update({str(path.resolve()): binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")})
    return pins


class Budget(base.Budget):
    def __init__(self, path, authorization, *, prior_GPU_hours):
        self.path, self.authorization, self.lock = Path(path), authorization, threading.RLock()
        self.changed = threading.Condition(self.lock)
        require(not self.path.exists(), "Do not overwrite or automatically retry an existing continuation budget.")
        self.value = dict(schema="sc_cat_continuation_owned_windows_v1",
            component_limits_seconds={key: value * 3600 for key, value in authorization["budget_by_component"].items()},
            total_limit_seconds=authorization["total_GPU_hours"] * 3600, tasks={}, cpu_windows=[],
            counter_reserved={key: 0 for key in {**CORE_COUNTS, **REPLAY_COUNTS}},
            prior_batch_GPU_hours=prior_GPU_hours, prior_cost_transferred=False)
        self._save()

    def _save(self):
        charged, reserved = (dict.fromkeys(COMPONENTS, 0.) for _ in range(2))
        details = {}
        for name, task in self.value["tasks"].items():
            if task["status"] in ("reserved", "running"):
                reserved[task["component"]] += task["seconds"]
            else:
                charged[task["component"]] += task["actual_seconds"]
                details[task.get("kind", name.rsplit("/", 1)[-1])] = details.get(task.get("kind", name.rsplit("/", 1)[-1]), 0.) + task["actual_seconds"]
        self.value.update(actual_seconds_by_component=charged, reserved_seconds_by_component=reserved,
            actual_seconds_by_task_type=details, total_actual_GPU_hours=sum(charged.values()) / 3600,
            total_reserved_GPU_hours=sum(reserved.values()) / 3600)
        write(self.path, self.value, replace=self.path.exists())

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        with self.changed:
            self.changed.notify_all()


def raw_tasks(prefix, authorization, category):
    return {prefix + "/fit": dict(component="core", kind="fit", seconds=seconds(authorization, "fit", category=category)),
        prefix + "/raw_export": dict(component="core", kind="raw_export", seconds=seconds(authorization, "raw_export")),
        prefix + "/raw_full8": dict(component="core", kind="full8", seconds=seconds(authorization, "full8"))}


def trajectory_tasks(prefix, authorization, category, settings):
    tasks = {prefix + "/train": dict(component="core", kind="train", seconds=seconds(authorization, "train", category=category, settings=settings))}
    for step in settings.node_steps:
        tasks[prefix + f"/node_{step}/export"] = dict(component="core", kind="node_export", seconds=seconds(authorization, "node_export"))
        tasks[prefix + f"/node_{step}/full8"] = dict(component="core", kind="full8", seconds=seconds(authorization, "full8"))
    return tasks

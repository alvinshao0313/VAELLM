"""Bindings and native payload receipts for isolated SC-CAT recovery workers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    require(not path.exists(), "Do not replace an existing recovery receipt: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.rename(path)


def binding(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def read_bound_json(reference):
    """A file reference may carry the caller's already frozen SHA256."""
    if isinstance(reference, str):
        path = Path(reference)
    else:
        path = Path(reference["path"])
        require(binding(path)["sha256"] == reference["sha256"], "A bound recovery input changed: " + str(path))
    return read_json(path)


def validate_request(request):
    require(request["operation"] in ("train", "export", "export_zero", "fresh"), "Choose one concrete recovery operation.")
    require(request["phase"] in ("formal", "technical_validation", "cpu_fixture"), "Declare the actual recovery phase.")
    settings = RecoverySettings(**request["settings"])
    if request["phase"] != "cpu_fixture":
        from .live_io import APPROVAL, bound_path

        require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae", "Activate the confirmed bitvae environment in the caller shell.")
        require(type(request["gpu"]) is int and request["gpu"] in range(6), "This batch only authorizes physical GPUs0-5.")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(request["gpu"]), "The caller must expose exactly its assigned physical GPU.")
        require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "Keep the original CUDA allocator policy.")
        authorization = read_json(bound_path(request["authorization"]))
        require(APPROVAL in json.dumps(authorization, ensure_ascii=False)
            and request["deadline_epoch"] > time.time(), "Use the confirmed finite batch authorization and unexpired shared deadline.")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for a CPU mechanism fixture.")
    if request["operation"] == "train":
        require(request.get("technical_stop_after") in (None, 2), "The isolated save-return validation has exactly two updates.")
        require(request["phase"] != "formal" or request.get("technical_stop_after") is None, "A formal trajectory must keep all500/1000 updates.")
        require(not request.get("technical_peak_proof") or request["phase"] == "technical_validation"
            and request.get("technical_stop_after") == 2, "The separate resource-only1024 proof is restricted to the actual two-update technical branch.")
        require(not Path(request["output"]).exists(), "Start recovery in a new isolated output directory.")
    return settings


def node_receipts(run, *, live_raw, trajectory_id, recipe_id, source_candidate_id):
    from train_utils import checkpoint_v6 as v6

    trainer = run.trainer
    steps = (1, 2) if trainer.technical_stop_after is not None else run.trainer.recovery_settings.node_steps
    require(len(trainer.node_save_audits) == len(steps), "Retain and audit every declared node before exporting any trajectory.")
    rows = []
    for step in steps:
        directory = Path(trainer.args.output_dir) / f"checkpoint-{step}"
        meta = v6.load_v6_training_step_meta(str(directory))
        state = read_json(directory / "trainer_state.json")
        raw_directory, raw = v6.resolve_training_step_round_base_ref(str(directory), meta)
        require(state["global_step"] == step and state["max_steps"] == trainer.recovery_settings.schedule_max,
            "Each retained node keeps its actual step and original schedule maximum.")
        require(raw["checkpoint_id"] == live_raw["round_base_uuid"]
            and Path(raw_directory).resolve() == Path(live_raw["round_base_dir"]).resolve(),
            "Every retained node must refer to this trajectory's exact own raw.")
        files = ("checkpoint_meta.json", "training_model_state.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json")
        require(all((directory / name).is_file() for name in files)
            and bool(list(directory.glob("rng_state*.pth"))), "The retained node lacks native mutable/optimizer/schedule/RNG state.")
        rows.append(dict(step=step, schedule_max=trainer.recovery_settings.schedule_max,
            node_directory=str(directory.resolve()), source_node_id=meta["checkpoint_id"],
            source_round_base_id=raw["checkpoint_id"], source_round_base=str(Path(raw_directory).resolve()),
            trajectory_id=trajectory_id, recipe_id=recipe_id, source_candidate_id=source_candidate_id,
            category_id=live_raw["current_category"], settings=asdict(trainer.recovery_settings),
            semantic_seeds=run.provenance["semantic_seeds"],
            checkpoint_meta=binding(directory / "checkpoint_meta.json"),
            native_files={name: dict(path=str((directory / name).resolve()), bytes=(directory / name).stat().st_size) for name in files},
            rng_files=[dict(path=str(path.resolve()), bytes=path.stat().st_size) for path in sorted(directory.glob("rng_state*.pth"))],
            mutable_manifest=meta[v6.MUTABLE_STATE_MANIFEST_KEY],
            exact_resume_verified=False))
    return rows


def selected_from_receipt(receipt, phase):
    from .selection import SelectedCategory

    directory = Path(receipt["node_directory"])
    require(binding(directory / "checkpoint_meta.json") == receipt["checkpoint_meta"], "The retained node metadata changed before export.")
    return SelectedCategory(category_id=receipt["category_id"], recipe_id=receipt["recipe_id"],
        source_candidate_id=receipt["source_candidate_id"], settings=RecoverySettings(**receipt["settings"]),
        selected_step=receipt["step"], source_trajectory_id=receipt["trajectory_id"],
        source_node_id=receipt["source_node_id"], source_round_base_id=receipt["source_round_base_id"],
        seed_identity=receipt["semantic_seeds"], phase=phase)


def package_pins():
    from experiments.cat_auto_protocol_cmcc_20261104.live import source_pins

    paths = tuple(Path(__file__).parent.glob("*.py"))
    return {**source_pins(), **{str(path.resolve()): binding(path)["sha256"] for path in paths}}


def make_train_request(common, *, live_raw, settings, output, trajectory_id,
        recipe_id, source_candidate_id, technical_stop_after=None, technical_peak_proof=False):
    """Use the controller's assigned authorization/device/deadline, never allocate one here."""
    return dict(common, operation="train", live_raw=live_raw, settings=asdict(settings),
        output=str(output), trajectory_id=trajectory_id, recipe_id=recipe_id,
        source_candidate_id=source_candidate_id, technical_stop_after=technical_stop_after,
        technical_peak_proof=technical_peak_proof)


def make_export_request(training_request, *, recovery_result, node_step, output, destination):
    return dict(training_request, operation="export", recovery_result=recovery_result,
        node_step=node_step, output=str(output), destination=str(destination))

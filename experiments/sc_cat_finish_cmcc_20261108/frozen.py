"""Resolve exactly two moved document versions; every other binding stays strict."""
from __future__ import annotations

from pathlib import Path

from experiments.sc_cat_continue_cmcc_20261107 import bindings as io


def own_sources():
    return {str(path.resolve()): io.binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")}


def load_snapshot(path):
    value = io.read(path)
    io.require(value["schema"] == "sc_cat_up250_finish_v1", "Use only this actual interrupted up250 batch.")
    for key in ("authorization", "state", "budget", "quota", "exit", "training_request", "checkpoint250", "node125_export", "node125_score"):
        io.bound_path(value[key])
    io.base.verify_sources(value["original_source_pins"])
    io.base.verify_sources(value["repair_source_pins"])
    io.require(value["repair_source_pins"] == own_sources(), "Freeze every actual repair implementation.")
    auth = io.read(io.bound_path(value["authorization"]))
    io.require(auth["gpus"] == [0, 1, 2, 3] and auth["readout_gpus"] == [2, 3], "Keep the latest user GPU restriction.")
    for name in ("protocol", "execution_protocol"):
        pair = value["document_versions"][name]
        io.require(pair["original"] == auth[name] and pair["snapshot"]["sha256"] == auth[name]["sha256"],
            "A document relocation must contain the exact originally approved bytes.")
        io.bound_path(pair["snapshot"])
    return value


def install(path):
    snapshot = load_snapshot(path)
    original = io.bound_path
    relocated = {(row["original"]["path"], row["original"]["sha256"]): row["snapshot"]
        for row in snapshot["document_versions"].values()}

    def bound_path(reference):
        key = (reference.get("path"), reference.get("sha256")) if isinstance(reference, dict) else None
        return original(relocated[key]) if key in relocated else original(reference)

    io.bound_path = bound_path
    io.read_authorization(snapshot["authorization"])
    return snapshot

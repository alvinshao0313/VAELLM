from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mix_bit.schema import CandidateMode, CandidateSpaceConfig


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_candidate_space(path: str) -> CandidateSpaceConfig:
    raw = _read_json(path)
    modes_raw = raw.get("modes")
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ValueError(f"Candidate space {path} must define a non-empty modes list")

    modes: list[CandidateMode] = []
    names: list[str] = []
    for item in modes_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid mode entry in {path}")
        mode = CandidateMode(
            name=str(item["name"]),
            nominal_bit=float(item["nominal_bit"]),
            codebook_bits=int(item["codebook_bits"]),
            codebook_dim=int(item["codebook_dim"]),
            residual_stages=int(item["residual_stages"]),
        )
        names.append(mode.name)
        modes.append(mode)

    if len(names) != len(set(names)):
        raise ValueError(f"Candidate space {path} has duplicate mode names")

    baseline_mode = str(raw["baseline_mode"])
    if baseline_mode not in set(names):
        raise ValueError(
            f"Candidate space {path} baseline_mode {baseline_mode!r} is not present in modes"
        )

    return CandidateSpaceConfig(
        candidate_space_id=str(raw["candidate_space_id"]),
        baseline_mode=baseline_mode,
        target_average_bit=float(raw["target_average_bit"]),
        modes=tuple(modes),
    )

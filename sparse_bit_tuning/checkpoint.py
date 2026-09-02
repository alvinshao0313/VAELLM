from __future__ import annotations

import json
import os
from typing import Dict, Mapping

import torch

SIDE_CAR_DIR = "sparse_bit_tuning"
PACKED_BITS_FILE = "packed_bits.pt"
COVERAGE_FILE = "coverage.json"
PACKED_FORMAT = "sparse_bit_tuning_packed_bits"
PACKED_VERSION = 1


def sidecar_dir(checkpoint_dir: str) -> str:
    return os.path.join(str(checkpoint_dir), SIDE_CAR_DIR)


def sidecar_exists(checkpoint_dir: str) -> bool:
    root = sidecar_dir(checkpoint_dir)
    return os.path.isfile(os.path.join(root, PACKED_BITS_FILE)) or os.path.isfile(
        os.path.join(root, COVERAGE_FILE)
    )


def sidecar_complete(checkpoint_dir: str) -> bool:
    root = sidecar_dir(checkpoint_dir)
    return os.path.isfile(os.path.join(root, PACKED_BITS_FILE)) and os.path.isfile(
        os.path.join(root, COVERAGE_FILE)
    )


def save_sidecar(
    checkpoint_dir: str,
    *,
    packed_banks: Mapping[str, torch.Tensor],
    coverage: dict,
) -> None:
    root = sidecar_dir(checkpoint_dir)
    os.makedirs(root, exist_ok=True)
    cpu_banks: Dict[str, torch.Tensor] = {}
    for key, tensor in packed_banks.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"packed bank {key!r} is not a Tensor: {type(tensor)}")
        if tensor.dtype != torch.uint8:
            raise TypeError(f"packed bank {key!r} must be uint8, got {tensor.dtype}.")
        cpu_banks[str(key)] = tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    torch.save(
        {
            "format": PACKED_FORMAT,
            "version": PACKED_VERSION,
            "banks": cpu_banks,
        },
        os.path.join(root, PACKED_BITS_FILE),
    )
    with open(os.path.join(root, COVERAGE_FILE), "w", encoding="utf-8") as handle:
        json.dump(coverage, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_sidecar(checkpoint_dir: str) -> tuple[Dict[str, torch.Tensor], dict]:
    root = sidecar_dir(checkpoint_dir)
    packed_path = os.path.join(root, PACKED_BITS_FILE)
    coverage_path = os.path.join(root, COVERAGE_FILE)
    if not os.path.isfile(packed_path) or not os.path.isfile(coverage_path):
        raise FileNotFoundError(
            f"Sparse Bit resume requires complete sidecar under {root}: "
            f"missing packed_bits.pt={not os.path.isfile(packed_path)} "
            f"coverage.json={not os.path.isfile(coverage_path)}"
        )
    payload = torch.load(packed_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid Sparse Bit packed sidecar type: {type(payload)}")
    if str(payload.get("format")) != PACKED_FORMAT or int(payload.get("version", -1)) != PACKED_VERSION:
        raise ValueError(
            f"unsupported Sparse Bit packed sidecar format/version: "
            f"{payload.get('format')!r}/{payload.get('version')!r}."
        )
    raw_banks = payload.get("banks")
    if not isinstance(raw_banks, dict):
        raise TypeError("Sparse Bit packed sidecar banks must be a dict.")
    banks: Dict[str, torch.Tensor] = {}
    for key, tensor in raw_banks.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
            raise TypeError(f"invalid packed bank {key!r}: {type(tensor)}/{getattr(tensor, 'dtype', None)}")
        banks[str(key)] = tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    with open(coverage_path, "r", encoding="utf-8") as handle:
        coverage = json.load(handle)
    if not isinstance(coverage, dict):
        raise TypeError("Sparse Bit coverage sidecar must be a JSON object.")
    return banks, coverage

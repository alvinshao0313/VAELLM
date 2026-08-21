#!/usr/bin/env python3
"""Extract down_proj 4bit VAE compressed state from a donor checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Any

import torch

from train_utils.model_checkpoint_io import (
    META_FILENAME as CHECKPOINT_META_FILENAME,
    _torch_load_state_dict,
    resolve_checkpoint_dir,
)

ARTIFACT_FORMAT = "vaellm_down_proj_transfer_artifact"
ARTIFACT_VERSION = 1
STATE_DICT_FILENAME = "down_proj_compressed_state.pt"
TRANSFER_META_FILENAME = "down_proj_transfer_meta.json"
DOWN_MODULE_SUFFIX = ".mlp.down_proj"
LAYER_NAME_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$")
SHARED_DECODER_ERROR = (
    "vaellm_down_proj_transfer_artifact v1 does not support shared protected-residual decoders"
)


def _load_checkpoint_meta(source_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(source_dir, CHECKPOINT_META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing checkpoint metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Invalid checkpoint metadata type: {type(meta)}")
    return meta


def _select_down_module_specs(converted_modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(converted_modules, list):
        raise ValueError("checkpoint_meta.converted_modules must be a list")

    specs = [
        spec
        for spec in converted_modules
        if isinstance(spec, dict) and str(spec.get("name", "")).endswith(DOWN_MODULE_SUFFIX)
    ]
    specs.sort(key=lambda item: int(LAYER_NAME_RE.match(str(item["name"])).group(1)))
    return specs


def _validate_down_module_specs(module_specs: list[dict[str, Any]]) -> list[str]:
    if not module_specs:
        raise ValueError("No down_proj modules found in checkpoint metadata")

    module_names: list[str] = []
    seen: set[str] = set()
    layer_indices: list[int] = []

    for spec in module_specs:
        name = str(spec.get("name", ""))
        match = LAYER_NAME_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid down module name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate down module name: {name}")
        seen.add(name)
        module_names.append(name)
        layer_indices.append(int(match.group(1)))

    layer_indices.sort()
    expected = list(range(layer_indices[0], layer_indices[0] + len(layer_indices)))
    if layer_indices != expected:
        raise ValueError(
            f"Down layer indices are not continuous: got {layer_indices}, expected {expected}"
        )
    return module_names


def _reject_shared_protected_residual_decoders(module_specs: list[dict[str, Any]]) -> None:
    for spec in module_specs:
        refs = spec.get("protected_residual_shared_decoder_refs")
        if refs:
            raise ValueError(
                f"[{spec['name']}] {SHARED_DECODER_ERROR}"
            )


def _extract_compressed_state(
    source_state: dict[str, torch.Tensor],
    module_names: list[str],
) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"{name}." for name in module_names)
    compressed_state: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if not key.startswith(prefixes):
            continue
        if key.endswith(".original_weight"):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor for state key {key!r}, got {type(value)}")
        compressed_state[key] = value.detach().cpu().contiguous()
    return compressed_state


def _module_state_keys(compressed_state: dict[str, torch.Tensor], module_name: str) -> set[str]:
    prefix = f"{module_name}."
    return {key[len(prefix):] for key in compressed_state if key.startswith(prefix)}


def _validate_compressed_state(
    compressed_state: dict[str, torch.Tensor],
    module_specs: list[dict[str, Any]],
) -> None:
    for key in compressed_state:
        if key.endswith(".original_weight"):
            raise ValueError(f"artifact must not contain original_weight: {key}")

    for spec in module_specs:
        module_name = str(spec["name"])
        local_keys = _module_state_keys(compressed_state, module_name)
        if "vq_weight" not in local_keys:
            raise ValueError(f"[{module_name}] missing required state key: vq_weight")

        residual_stages = int(spec.get("residual_stages", 1))
        if residual_stages > 1 and "vq_weight_s1" not in local_keys:
            raise ValueError(f"[{module_name}] missing required state key: vq_weight_s1")

        if spec.get("protected_input_indices") is not None and "protected_input_indices" not in local_keys:
            raise ValueError(f"[{module_name}] missing required state key: protected_input_indices")
        if spec.get("protected_input_weight") is not None and "protected_input_weight" not in local_keys:
            raise ValueError(f"[{module_name}] missing required state key: protected_input_weight")


def _build_transfer_module_specs(module_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transfer_specs: list[dict[str, Any]] = []
    for spec in module_specs:
        copied = copy.deepcopy(spec)
        copied["has_original_weight"] = False
        transfer_specs.append(copied)
    return transfer_specs


def _build_transfer_meta(
    *,
    source_dir: str,
    source_meta: dict[str, Any],
    module_names: list[str],
    module_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "source_checkpoint": os.path.abspath(source_dir),
        "source_checkpoint_format": str(source_meta.get("format", "")),
        "source_checkpoint_version": int(source_meta.get("version", 0)),
        "source_base_model_path": source_meta.get("base_model_path"),
        "state_dict_file": STATE_DICT_FILENAME,
        "module_count": len(module_names),
        "module_names": list(module_names),
        "module_specs": module_specs,
        "original_weight_policy": "excluded_for_transfer",
    }


def _assert_output_dir_empty(output_dir: str) -> None:
    abs_output_dir = os.path.abspath(output_dir)
    if os.path.exists(abs_output_dir):
        entries = os.listdir(abs_output_dir)
        if entries:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {abs_output_dir}"
            )
    else:
        os.makedirs(abs_output_dir, exist_ok=False)


def _atomic_torch_save(state_dict: dict[str, torch.Tensor], path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


def _atomic_write_json(payload: dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.3f} GiB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.3f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.3f} KiB"
    return f"{num_bytes} B"


def _directory_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def _verify_bit_exact(
    artifact_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> None:
    for key, artifact_tensor in artifact_state.items():
        if key not in source_state:
            raise ValueError(f"artifact key missing in source state: {key}")
        if not torch.equal(artifact_tensor, source_state[key]):
            raise ValueError(f"artifact tensor differs from source for key: {key}")


def extract_down_transfer_artifact(
    *,
    source_checkpoint: str,
    output_dir: str,
) -> dict[str, Any]:
    source_dir = resolve_checkpoint_dir(source_checkpoint)
    source_meta = _load_checkpoint_meta(source_dir)

    converted_modules = source_meta.get("converted_modules", [])
    down_specs = _select_down_module_specs(converted_modules)
    module_names = _validate_down_module_specs(down_specs)
    _reject_shared_protected_residual_decoders(down_specs)

    state_dict_file = str(source_meta.get("state_dict_file", "pytorch_model.bin"))
    source_state_path = os.path.join(source_dir, state_dict_file)
    if not os.path.isfile(source_state_path):
        raise FileNotFoundError(f"Missing source state dict: {source_state_path}")

    source_state = _torch_load_state_dict(source_state_path, map_location="cpu")
    compressed_state = _extract_compressed_state(source_state, module_names)
    _validate_compressed_state(compressed_state, down_specs)
    _verify_bit_exact(compressed_state, source_state)

    transfer_specs = _build_transfer_module_specs(down_specs)
    transfer_meta = _build_transfer_meta(
        source_dir=source_dir,
        source_meta=source_meta,
        module_names=module_names,
        module_specs=transfer_specs,
    )

    _assert_output_dir_empty(output_dir)
    state_path = os.path.join(output_dir, STATE_DICT_FILENAME)
    meta_path = os.path.join(output_dir, TRANSFER_META_FILENAME)
    _atomic_torch_save(compressed_state, state_path)
    _atomic_write_json(transfer_meta, meta_path)

    source_state_size = os.path.getsize(source_state_path)
    artifact_state_size = os.path.getsize(state_path)
    artifact_total_size = _directory_size(output_dir)
    reduction_ratio = 1.0 - (artifact_state_size / source_state_size)

    stats = {
        "source_state_file": source_state_path,
        "source_state_file_size": source_state_size,
        "artifact_state_file": state_path,
        "artifact_state_file_size": artifact_state_size,
        "artifact_total_size": artifact_total_size,
        "size_reduction_ratio": reduction_ratio,
        "module_count": len(module_names),
        "compressed_key_count": len(compressed_state),
    }
    return {
        "output_dir": os.path.abspath(output_dir),
        "state_dict_path": state_path,
        "meta_path": meta_path,
        "transfer_meta": transfer_meta,
        "compressed_state": compressed_state,
        "source_state": source_state,
        "stats": stats,
    }


def _print_stats(stats: dict[str, Any]) -> None:
    print(f"source full state file size: {_format_bytes(int(stats['source_state_file_size']))}")
    print(f"artifact state file size: {_format_bytes(int(stats['artifact_state_file_size']))}")
    print(f"artifact total size: {_format_bytes(int(stats['artifact_total_size']))}")
    print(f"size reduction ratio: {float(stats['size_reduction_ratio']):.4f}")
    print(f"module_count: {int(stats['module_count'])}")
    print(f"compressed_key_count: {int(stats['compressed_key_count'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_checkpoint",
        required=True,
        help="Donor checkpoint run dir, final_model dir, or checkpoint_meta.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the down-only transfer artifact",
    )
    args = parser.parse_args()

    result = extract_down_transfer_artifact(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
    )
    _print_stats(result["stats"])
    print(f"saved artifact to: {result['output_dir']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Sequence, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from litebsq.bitpack import build_bitpack_u8_spec, pack_bool_tensor_to_uint8, validate_bitpack_u8_spec
from train_utils.checkpoint_v6 import META_FILENAME
from train_utils.legacy_checkpoint_io import inspect_legacy_checkpoint


def _normalize_shape(shape: Sequence[int], *, arg_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"{arg_name} must be a list/tuple, got {type(shape)}")
    normalized = tuple(int(v) for v in shape)
    if len(normalized) < 1:
        raise ValueError(f"{arg_name} cannot be empty.")
    if any(v < 0 for v in normalized):
        raise ValueError(f"{arg_name} must contain non-negative integers, got {normalized}")
    return normalized


def _legacy_vq_spec_to_packed(spec: Dict[str, Any], *, arg_name: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError(f"{arg_name} must be a dict, got {type(spec)}")
    if "storage_format" in spec:
        validate_bitpack_u8_spec(spec, arg_name=arg_name)
        raise ValueError("checkpoint is already packed; conversion is not needed.")
    shape = _normalize_shape(spec.get("shape", ()), arg_name=f"{arg_name}.shape")
    dtype_name = str(spec.get("dtype", "")).strip().lower()
    if dtype_name != "bool":
        raise ValueError(f"{arg_name}.dtype must be 'bool' for legacy checkpoint, got {dtype_name!r}")
    return build_bitpack_u8_spec(logical_shape=shape)


def _vq_state_key(module_name: str, *, stage_idx: int, part_idx: int, parallel_parts: int) -> str:
    if stage_idx == 0:
        return f"{module_name}.vq_weight_{part_idx}" if parallel_parts > 1 else f"{module_name}.vq_weight"
    return (
        f"{module_name}.vq_weight_s{stage_idx}_{part_idx}"
        if parallel_parts > 1
        else f"{module_name}.vq_weight_s{stage_idx}"
    )


def _convert_vq_tensor(state_dict: Dict[str, Any], key: str, *, logical_shape: Sequence[int]) -> None:
    if key not in state_dict:
        raise KeyError(f"Missing VQ tensor key in state_dict: {key}")
    tensor = state_dict[key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{key} must be a tensor, got {type(tensor)}")
    if tensor.dtype != torch.bool:
        raise ValueError(f"{key} must be torch.bool in legacy checkpoint, got {tensor.dtype}")
    state_dict[key] = pack_bool_tensor_to_uint8(tensor.contiguous(), logical_shape=logical_shape)


def _convert_converted_modules(
    state_dict: Dict[str, Any],
    converted_modules: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    converted_specs: List[Dict[str, Any]] = []
    for spec in converted_modules:
        module_name = str(spec["name"])
        parallel_parts = int(spec["parallel_parts"])
        residual_stages = int(spec.get("residual_stages", 1))
        spec_copy = dict(spec)
        if residual_stages > 1:
            stage_vq_specs = spec.get("stage_vq_weights")
            if not isinstance(stage_vq_specs, (list, tuple)):
                raise ValueError(f"[{module_name}] legacy stage_vq_weights is invalid.")
            new_stage_specs = []
            for stage_idx, stage_item in enumerate(stage_vq_specs):
                if parallel_parts == 1:
                    if not isinstance(stage_item, dict):
                        raise ValueError(f"[{module_name}] stage_vq_weights[{stage_idx}] must be dict.")
                    packed_spec = _legacy_vq_spec_to_packed(
                        stage_item,
                        arg_name=f"[{module_name}] stage_vq_weights[{stage_idx}]",
                    )
                    _convert_vq_tensor(
                        state_dict,
                        _vq_state_key(module_name, stage_idx=stage_idx, part_idx=0, parallel_parts=parallel_parts),
                        logical_shape=packed_spec["logical_shape"],
                    )
                    new_stage_specs.append(packed_spec)
                else:
                    if not isinstance(stage_item, (list, tuple)) or len(stage_item) != parallel_parts:
                        raise ValueError(f"[{module_name}] stage_vq_weights[{stage_idx}] must match parallel_parts.")
                    packed_stage_specs = []
                    for part_idx, part_spec in enumerate(stage_item):
                        packed_spec = _legacy_vq_spec_to_packed(
                            part_spec,
                            arg_name=f"[{module_name}] stage_vq_weights[{stage_idx}][{part_idx}]",
                        )
                        _convert_vq_tensor(
                            state_dict,
                            _vq_state_key(module_name, stage_idx=stage_idx, part_idx=part_idx, parallel_parts=parallel_parts),
                            logical_shape=packed_spec["logical_shape"],
                        )
                        packed_stage_specs.append(packed_spec)
                    new_stage_specs.append(packed_stage_specs)
            spec_copy["stage_vq_weights"] = new_stage_specs
            spec_copy["vq_weights"] = new_stage_specs[0] if parallel_parts == 1 else list(new_stage_specs[0])
        else:
            vq_specs = spec.get("vq_weights")
            if not isinstance(vq_specs, (list, tuple)) or len(vq_specs) != parallel_parts:
                raise ValueError(f"[{module_name}] legacy vq_weights must match parallel_parts.")
            packed_specs = []
            for part_idx, part_spec in enumerate(vq_specs):
                packed_spec = _legacy_vq_spec_to_packed(
                    part_spec,
                    arg_name=f"[{module_name}] vq_weights[{part_idx}]",
                )
                _convert_vq_tensor(
                    state_dict,
                    _vq_state_key(module_name, stage_idx=0, part_idx=part_idx, parallel_parts=parallel_parts),
                    logical_shape=packed_spec["logical_shape"],
                )
                packed_specs.append(packed_spec)
            spec_copy["vq_weights"] = packed_specs
        converted_specs.append(spec_copy)
    return converted_specs


def _copy_non_checkpoint_files(src_dir: str, dst_dir: str, *, state_dict_file: str) -> None:
    os.makedirs(dst_dir, exist_ok=False)
    for entry in os.listdir(src_dir):
        if entry in {META_FILENAME, state_dict_file}:
            continue
        src_path = os.path.join(src_dir, entry)
        dst_path = os.path.join(dst_dir, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def convert_checkpoint(src_dir: str, dst_dir: str) -> Tuple[str, str]:
    checkpoint_dir = inspect_legacy_checkpoint(src_dir).checkpoint_dir
    if os.path.exists(dst_dir):
        raise FileExistsError(f"Output directory already exists: {dst_dir}")

    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    converted_modules = meta.get("converted_modules", [])
    if not converted_modules:
        raise ValueError("No converted_modules found in checkpoint; nothing to convert.")
    state_dict_file = str(meta.get("state_dict_file", "pytorch_model.bin"))
    state_dict_path = os.path.join(checkpoint_dir, state_dict_file)
    state_dict = torch.load(state_dict_path, map_location="cpu")

    new_meta = dict(meta)
    new_meta["version"] = 5
    new_meta["converted_modules"] = _convert_converted_modules(state_dict, converted_modules)
    new_meta["converted_module_count"] = len(new_meta["converted_modules"])

    _copy_non_checkpoint_files(checkpoint_dir, dst_dir, state_dict_file=state_dict_file)
    torch.save(state_dict, os.path.join(dst_dir, state_dict_file))
    with open(os.path.join(dst_dir, META_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(new_meta, handle, ensure_ascii=False, indent=2)
    return checkpoint_dir, dst_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert legacy cat checkpoint VQ bits to packed uint8 format.")
    parser.add_argument("--input", required=True, help="Legacy checkpoint dir, run dir, or checkpoint_meta.json path.")
    parser.add_argument("--output", required=True, help="Output directory for the converted packed checkpoint.")
    args = parser.parse_args()
    src_dir, dst_dir = convert_checkpoint(args.input, args.output)
    print(f"Converted legacy checkpoint: {src_dir} -> {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

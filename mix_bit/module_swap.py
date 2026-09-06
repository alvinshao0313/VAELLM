from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Protocol

import torch
from torch import nn

from litebsq.bitpack import BITPACK_U8_STORAGE_FORMAT, validate_bitpack_u8_spec
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import _rebuild_converted_modules


class ModuleCandidateLike(Protocol):
    module_name: str
    module_spec: dict[str, Any]
    in_features: int
    out_features: int
    has_bias: bool


class _ShapeOnlyPlaceholder(nn.Module):
    """CPU placeholder exposing device/dtype without allocating dense in×out weight."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        has_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(0, dtype=dtype, device=device))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)


def refresh_vae_runtime(model: nn.Module) -> None:
    """Refresh VAELinear runtime plans and clear decoded caches after state load."""
    for module in model.modules():
        if not isinstance(module, VAELinear):
            continue
        if getattr(module, "_parallel_stage_decoder", None) is not None:
            module._build_parallel_stage_decode_plan()
        if getattr(module, "_protected_residual_parallel_decoder", None) is not None:
            module._build_protected_residual_parallel_decode_plan()
        module.clear_decoded_weight_cache()


def _strip_module_prefix(
    compact_state: Mapping[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{module_name}."
    local_state: dict[str, torch.Tensor] = {}
    for key, value in compact_state.items():
        if not key.startswith(prefix):
            raise ValueError(
                f"State key {key!r} does not use prefix {prefix!r} "
                f"(compact keys not consumed by one declared module)"
            )
        local_state[key[len(prefix) :]] = value
    if not local_state:
        raise ValueError(f"empty compact state for module {module_name!r}")
    return local_state


def _infer_dtype(compact_state: Mapping[str, torch.Tensor]) -> torch.dtype:
    for value in compact_state.values():
        if torch.is_floating_point(value):
            return value.dtype
    return torch.float32


def _validate_packed_vq_against_module(
    module: VAELinear,
    local_state: Mapping[str, torch.Tensor],
) -> None:
    residual_stages = int(getattr(module, "residual_stages", 1))
    parallel_parts = int(getattr(module, "parallel_parts", 1))
    expected_shapes: list[tuple[int, ...]] = []
    for stage_idx in range(residual_stages):
        for part_idx in range(parallel_parts):
            spec = module.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx)
            normalized = validate_bitpack_u8_spec(spec)
            if normalized["storage_format"] != BITPACK_U8_STORAGE_FORMAT:
                raise ValueError(
                    f"expected {BITPACK_U8_STORAGE_FORMAT} VQ storage, got {normalized['storage_format']}"
                )
            expected_shapes.append(tuple(int(x) for x in normalized["shape"]))

    for key, tensor in local_state.items():
        leaf = key.split(".")[-1]
        if not leaf.startswith("vq_weight"):
            continue
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{key}: VQ payload must be torch.uint8, got {tensor.dtype}")
        shape = tuple(int(x) for x in tensor.shape)
        if shape not in expected_shapes:
            raise ValueError(
                f"{key}: uint8 VQ payload shape {shape} does not match packed VQ specs {expected_shapes}"
            )


def _verify_candidate_metadata(module: VAELinear, candidate: ModuleCandidateLike) -> None:
    spec = candidate.module_spec
    if int(module.in_features) != int(candidate.in_features):
        raise ValueError(
            f"{candidate.module_name}: in_features mismatch "
            f"module={module.in_features} candidate={candidate.in_features}"
        )
    if int(module.out_features) != int(candidate.out_features):
        raise ValueError(
            f"{candidate.module_name}: out_features mismatch "
            f"module={module.out_features} candidate={candidate.out_features}"
        )
    has_bias = module.bias is not None
    if has_bias != bool(candidate.has_bias):
        raise ValueError(
            f"{candidate.module_name}: bias presence mismatch "
            f"module={has_bias} candidate={candidate.has_bias}"
        )
    if bool(module.transpose) != bool(spec.get("transpose", False)):
        raise ValueError(
            f"{candidate.module_name}: transpose mismatch "
            f"module={module.transpose} spec={spec.get('transpose')}"
        )
    if int(module.codebook_dim) != int(spec["codebook_dim"]):
        raise ValueError(
            f"{candidate.module_name}: codebook_dim mismatch "
            f"module={module.codebook_dim} spec={spec['codebook_dim']}"
        )
    expected_stages = int(spec.get("residual_stages", 1) or 1)
    if int(module.residual_stages) != expected_stages:
        raise ValueError(
            f"{candidate.module_name}: residual_stages mismatch "
            f"module={module.residual_stages} spec={expected_stages}"
        )
    stage_dims = spec.get("stage_codebook_dims")
    if isinstance(stage_dims, (list, tuple)) and stage_dims:
        module_dims = [int(v) for v in getattr(module, "stage_codebook_dims", [])]
        expected_dims = [int(v) for v in stage_dims]
        if len(expected_dims) == 1 and expected_stages > 1:
            expected_dims = expected_dims * expected_stages
        if module_dims != expected_dims:
            raise ValueError(
                f"{candidate.module_name}: stage_codebook_dims mismatch "
                f"module={module_dims} spec={expected_dims}"
            )


def build_candidate_module(
    candidate: ModuleCandidateLike,
    compact_state: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str,
) -> nn.Module:
    """Build one self-contained VAELinear from ModuleCandidate + prefixed compact state."""
    module_name = str(candidate.module_name)
    module_spec = candidate.module_spec
    if bool(module_spec.get("has_original_weight", False)):
        raise ValueError(
            f"{module_name}: compact candidate rebuild rejects has_original_weight=true"
        )

    device_obj = torch.device(device)
    dtype = _infer_dtype(compact_state)
    holder = nn.Module()
    holder.target = _ShapeOnlyPlaceholder(
        in_features=int(module_spec["in_features"]),
        out_features=int(module_spec["out_features"]),
        has_bias=bool(module_spec.get("has_bias", candidate.has_bias)),
        dtype=dtype,
        device=torch.device("cpu"),
    )
    local_spec = copy.deepcopy(module_spec)
    local_spec["name"] = "target"
    local_spec["has_original_weight"] = False
    _rebuild_converted_modules(holder, [local_spec])
    built = holder.target
    if not isinstance(built, VAELinear):
        raise TypeError(f"Expected VAELinear after rebuild, got {type(built)}")

    local_state = _strip_module_prefix(compact_state, module_name)
    missing_extra = built.load_state_dict(local_state, strict=True)
    if missing_extra.missing_keys or missing_extra.unexpected_keys:
        raise ValueError(
            f"{module_name}: strict load failed missing={missing_extra.missing_keys} "
            f"unexpected={missing_extra.unexpected_keys}"
        )

    refresh_vae_runtime(holder)
    built.clear_decoded_weight_cache()
    built.eval()
    built.to(device_obj)

    _verify_candidate_metadata(built, candidate)
    _validate_packed_vq_against_module(built, local_state)
    return built


@contextmanager
def temporary_module_swap(
    model: nn.Module,
    module_name: str,
    replacement: nn.Module,
) -> Iterator[nn.Module]:
    """Replace one named submodule, then restore the exact original object."""
    parts = str(module_name).split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    original = getattr(parent, parts[-1])
    try:
        set_module_by_name(model, module_name, replacement)
        yield replacement
    finally:
        set_module_by_name(model, module_name, original)

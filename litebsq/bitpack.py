from typing import Any, Dict, Sequence, Tuple

import torch


BITPACK_U8_STORAGE_FORMAT = "bitpack_u8"
BITPACK_U8_PACK_BITS = 8


def normalize_logical_shape(logical_shape: Sequence[int], *, arg_name: str = "logical_shape") -> Tuple[int, ...]:
    if not isinstance(logical_shape, (list, tuple)):
        raise TypeError(f"{arg_name} must be a list/tuple, got {type(logical_shape)}")
    normalized = tuple(int(v) for v in logical_shape)
    if len(normalized) < 1:
        raise ValueError(f"{arg_name} cannot be empty.")
    if any(v < 0 for v in normalized):
        raise ValueError(f"{arg_name} must contain non-negative integers, got {normalized}")
    return normalized


def expected_bitpack_u8_shape(logical_shape: Sequence[int]) -> Tuple[int, ...]:
    normalized = normalize_logical_shape(logical_shape)
    packed_last_dim = (int(normalized[-1]) + BITPACK_U8_PACK_BITS - 1) // BITPACK_U8_PACK_BITS
    return normalized[:-1] + (packed_last_dim,)


def validate_bitpack_u8_spec(spec: Dict[str, Any], *, arg_name: str = "vq_storage_spec") -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError(f"{arg_name} must be a dict, got {type(spec)}")
    storage_format = str(spec.get("storage_format", "")).strip().lower()
    if storage_format != BITPACK_U8_STORAGE_FORMAT:
        raise ValueError(
            f"{arg_name}.storage_format must be {BITPACK_U8_STORAGE_FORMAT!r}, got {storage_format!r}"
        )
    dtype_name = str(spec.get("dtype", "")).strip().lower()
    if dtype_name != "uint8":
        raise ValueError(f"{arg_name}.dtype must be 'uint8', got {dtype_name!r}")
    logical_dtype_name = str(spec.get("logical_dtype", "")).strip().lower()
    if logical_dtype_name != "bool":
        raise ValueError(f"{arg_name}.logical_dtype must be 'bool', got {logical_dtype_name!r}")
    pack_bits = int(spec.get("pack_bits", 0))
    if pack_bits != BITPACK_U8_PACK_BITS:
        raise ValueError(f"{arg_name}.pack_bits must be {BITPACK_U8_PACK_BITS}, got {pack_bits}")
    logical_shape = normalize_logical_shape(spec.get("logical_shape", ()), arg_name=f"{arg_name}.logical_shape")
    storage_shape = normalize_logical_shape(spec.get("shape", ()), arg_name=f"{arg_name}.shape")
    expected_shape = expected_bitpack_u8_shape(logical_shape)
    if storage_shape != expected_shape:
        raise ValueError(
            f"{arg_name}.shape {storage_shape} != expected packed shape {expected_shape} for logical_shape={logical_shape}"
        )
    return {
        "shape": list(storage_shape),
        "dtype": "uint8",
        "storage_format": BITPACK_U8_STORAGE_FORMAT,
        "logical_shape": list(logical_shape),
        "logical_dtype": "bool",
        "pack_bits": BITPACK_U8_PACK_BITS,
    }


def build_bitpack_u8_spec(*, logical_shape: Sequence[int]) -> Dict[str, Any]:
    normalized_logical_shape = normalize_logical_shape(logical_shape)
    return validate_bitpack_u8_spec(
        {
            "shape": list(expected_bitpack_u8_shape(normalized_logical_shape)),
            "dtype": "uint8",
            "storage_format": BITPACK_U8_STORAGE_FORMAT,
            "logical_shape": list(normalized_logical_shape),
            "logical_dtype": "bool",
            "pack_bits": BITPACK_U8_PACK_BITS,
        }
    )


def pack_bool_tensor_to_uint8(
    tensor: torch.Tensor,
    *,
    logical_shape: Sequence[int],
) -> torch.Tensor:
    normalized_logical_shape = normalize_logical_shape(logical_shape)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)}")
    if tensor.dtype != torch.bool:
        raise ValueError(f"pack_bool_tensor_to_uint8 expects torch.bool tensor, got {tensor.dtype}")
    if tuple(int(v) for v in tensor.shape) != normalized_logical_shape:
        raise ValueError(
            f"tensor shape {tuple(int(v) for v in tensor.shape)} != logical_shape {normalized_logical_shape}"
        )
    packed_shape = expected_bitpack_u8_shape(normalized_logical_shape)
    logical_last_dim = int(normalized_logical_shape[-1])
    packed_last_dim = int(packed_shape[-1])
    rows = int(tensor.numel()) // max(1, logical_last_dim) if logical_last_dim > 0 else 0
    flat = tensor.contiguous().view(rows, logical_last_dim) if rows > 0 else tensor.new_zeros((0, logical_last_dim))
    padded_last_dim = packed_last_dim * BITPACK_U8_PACK_BITS
    if logical_last_dim != padded_last_dim:
        padded = torch.zeros((rows, padded_last_dim), dtype=torch.bool, device=tensor.device)
        if logical_last_dim > 0:
            padded[:, :logical_last_dim] = flat
    else:
        padded = flat
    packed = padded.view(rows, packed_last_dim, BITPACK_U8_PACK_BITS).to(dtype=torch.uint8)
    masks = (1 << torch.arange(BITPACK_U8_PACK_BITS, dtype=torch.uint8, device=tensor.device)).view(1, 1, -1)
    packed = (packed * masks).sum(dim=-1).to(dtype=torch.uint8)
    return packed.view(packed_shape).contiguous()


def unpack_uint8_tensor_to_bool(
    tensor: torch.Tensor,
    *,
    logical_shape: Sequence[int],
) -> torch.Tensor:
    normalized_logical_shape = normalize_logical_shape(logical_shape)
    expected_shape = expected_bitpack_u8_shape(normalized_logical_shape)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)}")
    if tensor.dtype != torch.uint8:
        raise ValueError(f"unpack_uint8_tensor_to_bool expects torch.uint8 tensor, got {tensor.dtype}")
    if tuple(int(v) for v in tensor.shape) != expected_shape:
        raise ValueError(
            f"packed tensor shape {tuple(int(v) for v in tensor.shape)} != expected packed shape {expected_shape}"
        )
    logical_last_dim = int(normalized_logical_shape[-1])
    packed_last_dim = int(expected_shape[-1])
    rows = int(tensor.numel()) // max(1, packed_last_dim) if packed_last_dim > 0 else 0
    flat = tensor.contiguous().view(rows, packed_last_dim) if rows > 0 else tensor.new_zeros((0, packed_last_dim))
    masks = (1 << torch.arange(BITPACK_U8_PACK_BITS, dtype=torch.uint8, device=tensor.device)).view(1, 1, -1)
    unpacked = flat.unsqueeze(-1).bitwise_and(masks).ne(0).view(rows, packed_last_dim * BITPACK_U8_PACK_BITS)
    unpacked = unpacked[:, :logical_last_dim]
    return unpacked.view(normalized_logical_shape).contiguous()

import math
from typing import Dict, Optional, Tuple

import torch


SPARSE_RESIDUAL_FORMAT_COO_FP16 = "coo_fp16"
SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED = "blocked_quantized"
SPARSE_RESIDUAL_FORMAT_CHOICES = (
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
)
SPARSE_RESIDUAL_INDEX_BITS_CHOICES = (8, 4)
SPARSE_RESIDUAL_VALUE_BITS_CHOICES = (8, 4)


def validate_sparse_residual_block_shape(
    *,
    block_rows: int,
    block_cols: int,
    index_bits: int,
    arg_name: str = "sparse residual block shape",
) -> Tuple[int, int]:
    block_rows = int(block_rows)
    block_cols = int(block_cols)
    index_bits = int(index_bits)
    if block_rows < 1 or block_cols < 1:
        raise ValueError(f"{arg_name} must be positive, got ({block_rows}, {block_cols}).")
    if index_bits not in SPARSE_RESIDUAL_INDEX_BITS_CHOICES:
        raise ValueError(
            f"Unsupported sparse residual index_bits={index_bits}. "
            f"Expected one of {SPARSE_RESIDUAL_INDEX_BITS_CHOICES}."
        )
    limit = 256 if index_bits == 8 else 16
    if block_rows > limit or block_cols > limit:
        raise ValueError(
            f"{arg_name}=({block_rows}, {block_cols}) exceeds index_bits={index_bits} capacity; "
            f"expected each dimension <= {limit}."
        )
    return block_rows, block_cols


def get_default_block_shape_for_index_bits(index_bits: int) -> Tuple[int, int]:
    index_bits = int(index_bits)
    if index_bits == 8:
        return 256, 256
    if index_bits == 4:
        return 16, 16
    raise ValueError(
        f"Unsupported sparse residual index_bits={index_bits}. "
        f"Expected one of {SPARSE_RESIDUAL_INDEX_BITS_CHOICES}."
    )


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    values_u8 = values.detach().to(device="cpu", dtype=torch.uint8).reshape(-1).contiguous()
    if int(values_u8.numel()) == 0:
        return values_u8
    if int(values_u8.max().item()) > 15:
        raise ValueError(f"uint4 pack expects values <= 15, got max={int(values_u8.max().item())}.")
    packed_len = (int(values_u8.numel()) + 1) // 2
    packed = torch.zeros(packed_len, dtype=torch.uint8)
    packed |= values_u8[0::2]
    if int(values_u8.numel()) > 1:
        packed[: values_u8[1::2].numel()] |= values_u8[1::2] << 4
    return packed.contiguous()


def unpack_uint4(packed: torch.Tensor, count: int) -> torch.Tensor:
    packed_u8 = packed.detach().to(device="cpu", dtype=torch.uint8).reshape(-1).contiguous()
    count = int(count)
    if count < 0:
        raise ValueError(f"uint4 unpack count must be >= 0, got {count}.")
    if count == 0:
        return torch.zeros(0, dtype=torch.uint8)
    expected_len = (count + 1) // 2
    if int(packed_u8.numel()) != expected_len:
        raise ValueError(f"uint4 packed length mismatch: got {int(packed_u8.numel())}, expected {expected_len}.")
    out = torch.empty(count, dtype=torch.uint8)
    out[0::2] = packed_u8 & 0x0F
    if count > 1:
        out[1::2] = packed_u8[: out[1::2].numel()] >> 4
    return out.contiguous()


def encode_blocked_quantized_sparse_residual(
    *,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    values: torch.Tensor,
    out_features: int,
    in_features: int,
    block_rows: int,
    block_cols: int,
    index_bits: int,
    value_bits: int,
) -> Dict[str, object]:
    block_rows, block_cols = validate_sparse_residual_block_shape(
        block_rows=int(block_rows),
        block_cols=int(block_cols),
        index_bits=int(index_bits),
        arg_name="sparse residual block shape",
    )
    value_bits = int(value_bits)
    if value_bits not in SPARSE_RESIDUAL_VALUE_BITS_CHOICES:
        raise ValueError(
            f"Unsupported sparse residual value_bits={value_bits}. "
            f"Expected one of {SPARSE_RESIDUAL_VALUE_BITS_CHOICES}."
        )

    rows = row_idx.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
    cols = col_idx.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
    vals = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    nnz = int(rows.numel())
    if int(cols.numel()) != nnz or int(vals.numel()) != nnz:
        raise ValueError(
            f"Blocked sparse residual payload length mismatch: rows={nnz} cols={int(cols.numel())} values={int(vals.numel())}."
        )
    if nnz == 0:
        raise ValueError("Blocked sparse residual encoding expects nnz > 0.")
    if int(rows.min().item()) < 0 or int(rows.max().item()) >= int(out_features):
        raise ValueError(
            f"Blocked sparse residual row indices must be within [0, {int(out_features)}), "
            f"got [{int(rows.min().item())}, {int(rows.max().item())}]."
        )
    if int(cols.min().item()) < 0 or int(cols.max().item()) >= int(in_features):
        raise ValueError(
            f"Blocked sparse residual col indices must be within [0, {int(in_features)}), "
            f"got [{int(cols.min().item())}, {int(cols.max().item())}]."
        )

    num_block_cols = int(math.ceil(float(in_features) / float(block_cols)))
    num_blocks = int(math.ceil(float(out_features) / float(block_rows))) * num_block_cols
    block_row = torch.div(rows, block_rows, rounding_mode="floor")
    block_col = torch.div(cols, block_cols, rounding_mode="floor")
    block_id = block_row * num_block_cols + block_col
    sort_keys = rows * int(in_features) + cols
    order = torch.argsort(block_id * (int(out_features) * int(in_features)) + sort_keys, stable=True)
    rows = rows.index_select(0, order)
    cols = cols.index_select(0, order)
    vals = vals.index_select(0, order)
    block_id = block_id.index_select(0, order).contiguous()
    local_row = torch.remainder(rows, block_rows).to(dtype=torch.int64)
    local_col = torch.remainder(cols, block_cols).to(dtype=torch.int64)

    active_block_ids, counts = torch.unique_consecutive(block_id, return_counts=True)
    active_block_count = int(active_block_ids.numel())
    if active_block_count == 0:
        raise RuntimeError("Blocked sparse residual encoding produced no active blocks.")
    if int(active_block_ids.min().item()) < 0 or int(active_block_ids.max().item()) >= num_blocks:
        raise RuntimeError(
            f"Blocked sparse residual block_id out of range: "
            f"[{int(active_block_ids.min().item())}, {int(active_block_ids.max().item())}] vs num_blocks={num_blocks}."
        )

    block_ptr_dtype = torch.int32 if nnz <= torch.iinfo(torch.int32).max else torch.int64
    block_ptr = torch.zeros(active_block_count + 1, dtype=block_ptr_dtype)
    block_ptr[1:] = counts.to(dtype=block_ptr_dtype).cumsum(dim=0)
    active_block_dtype = torch.uint16 if (num_blocks - 1) <= torch.iinfo(torch.uint16).max else torch.int32
    active_block_ids = active_block_ids.to(dtype=active_block_dtype).contiguous()

    if index_bits == 8:
        local_indices = torch.empty(nnz * 2, dtype=torch.uint8)
        local_indices[0::2] = local_row.to(dtype=torch.uint8)
        local_indices[1::2] = local_col.to(dtype=torch.uint8)
    else:
        local_indices = ((local_row << 4) | local_col).to(dtype=torch.uint8).contiguous()

    qmax = (1 << value_bits) - 1
    qvalue_chunks = []
    # TODO: Revisit the blocked quantization formula itself. We keep per-block
    # stats in fp32 for now because the current affine zero_point can overflow
    # in fp16 when the residual range is narrow but far from zero.
    scales = torch.empty(active_block_count, dtype=torch.float32)
    zero_points = torch.empty(active_block_count, dtype=torch.float32)
    start = 0
    for block_idx in range(active_block_count):
        count = int(counts[block_idx].item())
        end = start + count
        block_values = vals[start:end]
        vmin = float(block_values.min().item())
        vmax = float(block_values.max().item())
        if vmax == vmin:
            scale = 1.0
        else:
            scale = float(vmax - vmin) / float(qmax)
            if scale <= 0.0:
                raise RuntimeError(f"Invalid blocked sparse residual scale={scale} for block_idx={block_idx}.")
        zero_point = -float(vmin) / float(scale)
        q = torch.round(block_values / float(scale) + float(zero_point)).clamp_(0.0, float(qmax)).to(dtype=torch.uint8)
        qvalue_chunks.append(q)
        scales[block_idx] = torch.tensor(scale, dtype=torch.float32)
        zero_points[block_idx] = torch.tensor(zero_point, dtype=torch.float32)
        start = end
    if start != nnz:
        raise RuntimeError(f"Blocked sparse residual block count mismatch: consumed={start}, nnz={nnz}.")

    qvalues_raw = torch.cat(qvalue_chunks, dim=0).contiguous()
    qvalues = pack_uint4(qvalues_raw) if value_bits == 4 else qvalues_raw
    return {
        "format": SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
        "index_bits": int(index_bits),
        "value_bits": int(value_bits),
        "block_rows": int(block_rows),
        "block_cols": int(block_cols),
        "active_block_ids": active_block_ids,
        "block_ptr": block_ptr.contiguous(),
        "local_indices": local_indices.contiguous(),
        "qvalues": qvalues.contiguous(),
        "scales": scales.contiguous(),
        "zero_points": zero_points.contiguous(),
    }


def decode_blocked_quantized_sparse_residual(
    *,
    active_block_ids: torch.Tensor,
    block_ptr: torch.Tensor,
    local_indices: torch.Tensor,
    qvalues: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    out_features: int,
    in_features: int,
    block_rows: int,
    block_cols: int,
    index_bits: int,
    value_bits: int,
    value_dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_rows, block_cols = validate_sparse_residual_block_shape(
        block_rows=int(block_rows),
        block_cols=int(block_cols),
        index_bits=int(index_bits),
        arg_name="sparse residual block shape",
    )
    value_bits = int(value_bits)
    if value_bits not in SPARSE_RESIDUAL_VALUE_BITS_CHOICES:
        raise ValueError(
            f"Unsupported sparse residual value_bits={value_bits}. "
            f"Expected one of {SPARSE_RESIDUAL_VALUE_BITS_CHOICES}."
        )

    target_device = torch.device("cpu") if device is None else torch.device(device)
    active_blocks = active_block_ids.detach().reshape(-1).contiguous()
    ptr = block_ptr.detach().reshape(-1).contiguous()
    local = local_indices.detach().reshape(-1).contiguous()
    qpacked = qvalues.detach().reshape(-1).contiguous()
    scale_values = scales.detach().reshape(-1).contiguous()
    zero_values = zero_points.detach().reshape(-1).contiguous()

    active_block_count = int(active_blocks.numel())
    if int(ptr.numel()) != active_block_count + 1:
        raise ValueError(f"block_ptr length mismatch: got {int(ptr.numel())}, expected {active_block_count + 1}.")
    if int(scale_values.numel()) != active_block_count or int(zero_values.numel()) != active_block_count:
        raise ValueError(
            f"Per-block quant stats length mismatch: active_blocks={active_block_count} "
            f"scales={int(scale_values.numel())} zero_points={int(zero_values.numel())}."
    )
    if active_block_count == 0:
        empty_idx = torch.zeros(0, dtype=torch.int64, device=target_device)
        empty_values = torch.zeros(0, dtype=value_dtype, device=target_device)
        return empty_idx, empty_idx.clone(), empty_values
    ptr_i64 = ptr.to(dtype=torch.int64)
    if int(ptr_i64[0].item()) != 0:
        raise ValueError(f"block_ptr must start with 0, got {int(ptr_i64[0].item())}.")
    if bool(torch.any(ptr_i64[1:] < ptr_i64[:-1]).item()):
        raise ValueError("block_ptr must be non-decreasing.")

    nnz = int(ptr_i64[-1].item())
    if nnz < 0:
        raise ValueError(f"block_ptr final nnz must be >= 0, got {nnz}.")

    active_blocks = active_blocks.to(device=target_device, dtype=torch.int64, non_blocking=True)
    ptr_i64 = ptr_i64.to(device=target_device, dtype=torch.int64, non_blocking=True)
    local = local.to(device=target_device, dtype=torch.uint8, non_blocking=True)
    qpacked = qpacked.to(device=target_device, dtype=torch.uint8, non_blocking=True)
    scale_values = scale_values.to(device=target_device, dtype=torch.float32, non_blocking=True)
    zero_values = zero_values.to(device=target_device, dtype=torch.float32, non_blocking=True)

    if index_bits == 8:
        expected_local_len = nnz * 2
        if int(local.numel()) != expected_local_len:
            raise ValueError(
                f"local_indices length mismatch for index_bits=8: got {int(local.numel())}, expected {expected_local_len}."
            )
        local_row = local[0::2].to(dtype=torch.int64)
        local_col = local[1::2].to(dtype=torch.int64)
    else:
        expected_local_len = nnz
        if int(local.numel()) != expected_local_len:
            raise ValueError(
                f"local_indices length mismatch for index_bits=4: got {int(local.numel())}, expected {expected_local_len}."
            )
        local_row = (local >> 4).to(dtype=torch.int64)
        local_col = (local & 0x0F).to(dtype=torch.int64)

    if value_bits == 8:
        if int(qpacked.numel()) != nnz:
            raise ValueError(f"qvalues length mismatch for value_bits=8: got {int(qpacked.numel())}, expected {nnz}.")
        q = qpacked
    else:
        expected_qpacked_len = (nnz + 1) // 2
        if int(qpacked.numel()) != expected_qpacked_len:
            raise ValueError(
                f"qvalues length mismatch for value_bits=4: got {int(qpacked.numel())}, expected {expected_qpacked_len}."
            )
        q = torch.empty(nnz, dtype=torch.uint8, device=target_device)
        q[0::2] = qpacked & 0x0F
        if nnz > 1:
            q[1::2] = qpacked[: q[1::2].numel()] >> 4

    counts = ptr_i64[1:] - ptr_i64[:-1]
    block_ids_per_entry = torch.repeat_interleave(active_blocks, counts)
    if int(block_ids_per_entry.numel()) != nnz:
        raise RuntimeError(
            f"Expanded block ids length mismatch: got {int(block_ids_per_entry.numel())}, expected {nnz}."
        )
    num_block_cols = int(math.ceil(float(in_features) / float(block_cols)))
    block_row = torch.div(block_ids_per_entry, num_block_cols, rounding_mode="floor")
    block_col = torch.remainder(block_ids_per_entry, num_block_cols)
    row_idx = block_row * block_rows + local_row
    col_idx = block_col * block_cols + local_col
    if int(row_idx.numel()) != nnz or int(col_idx.numel()) != nnz:
        raise RuntimeError(
            f"Decoded sparse residual index length mismatch: rows={int(row_idx.numel())} cols={int(col_idx.numel())} expected={nnz}."
        )
    if nnz > 0:
        if int(row_idx.min().item()) < 0 or int(row_idx.max().item()) >= int(out_features):
            raise ValueError(
                f"Decoded sparse residual row indices out of range: "
                f"[{int(row_idx.min().item())}, {int(row_idx.max().item())}] vs out_features={int(out_features)}."
            )
        if int(col_idx.min().item()) < 0 or int(col_idx.max().item()) >= int(in_features):
            raise ValueError(
                f"Decoded sparse residual col indices out of range: "
                f"[{int(col_idx.min().item())}, {int(col_idx.max().item())}] vs in_features={int(in_features)}."
            )

    expanded_scales = torch.repeat_interleave(scale_values, counts)
    expanded_zero = torch.repeat_interleave(zero_values, counts)
    values = expanded_scales * (q.to(dtype=torch.float32) - expanded_zero)

    return (
        row_idx.to(dtype=torch.int64),
        col_idx.to(dtype=torch.int64),
        values.to(dtype=value_dtype),
    )


def sparse_residual_coo_storage_bytes(nnz: int) -> int:
    nnz = int(nnz)
    if nnz < 0:
        raise ValueError(f"nnz must be >= 0, got {nnz}.")
    return nnz * (2 + 2 + 2)


def sparse_residual_blocked_storage_bytes(payload: Dict[str, object]) -> int:
    total = 0
    for key in ("active_block_ids", "block_ptr", "local_indices", "qvalues", "scales", "zero_points"):
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Blocked sparse residual payload missing tensor key '{key}'.")
        total += int(tensor.numel()) * int(tensor.element_size())
    return total

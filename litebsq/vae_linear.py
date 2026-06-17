from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from litebsq.autoencoder import pack_decoders
from litebsq.bitpack import (
    build_bitpack_u8_spec,
    pack_bool_tensor_to_uint8,
    unpack_uint8_tensor_to_bool,
    validate_bitpack_u8_spec,
)

from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
    SPARSE_RESIDUAL_FORMAT_CHOICES,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    SPARSE_RESIDUAL_INDEX_BITS_CHOICES,
    SPARSE_RESIDUAL_VALUE_BITS_CHOICES,
    decode_blocked_quantized_sparse_residual,
    validate_sparse_residual_block_shape,
)

class VAELinear(nn.Module):
    """
    Inference-only Linear replacement using packed VQ bits + decoder to reconstruct weights on the fly.

    `vq_weight` 可以是单个 Tensor（单分块）或 Tensor 列表（多分块）。
    不传 `vq_storage_specs` / `stage_vq_storage_specs` 时，默认把逻辑 bool bits 打包成 uint8 存储。
    `decoder` 可以是单个 decoder（单分块）或 decoder 列表（多分块）。
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias,
        original_weight=None,
        vq_weight=None,
        vq_storage_specs: Optional[Sequence[Any]] = None,
        decoder=None,
        stage_vq_weights: Optional[Sequence[Any]] = None,
        stage_vq_storage_specs: Optional[Sequence[Any]] = None,
        stage_decoders: Optional[Sequence[Any]] = None,
        codebook_dim: int,
        stage_codebook_dims: Optional[Sequence[int]] = None,
        transpose: bool,
        parallel_parts: int = 1,
        parallel_rows: Optional[int] = None,
        parallel_cols: Optional[int] = None,
        # 排序代码，已关闭：不再接受 restore_row_indices / restore_col_indices /
        # part_restore_col_indices / stage_* restore 参数。
        compressed_in_features: Optional[int] = None,
        compressed_out_features: Optional[int] = None,
        protected_input_indices: Optional[torch.Tensor] = None,
        protected_input_weight: Optional[torch.Tensor] = None,
        protected_output_indices: Optional[torch.Tensor] = None,
        protected_output_weight: Optional[torch.Tensor] = None,
        sparse_residual_format: str = SPARSE_RESIDUAL_FORMAT_COO_FP16,
        sparse_residual_row_indices: Optional[torch.Tensor] = None,
        sparse_residual_col_indices: Optional[torch.Tensor] = None,
        sparse_residual_values: Optional[torch.Tensor] = None,
        sparse_residual_index_bits: Optional[int] = None,
        sparse_residual_value_bits: Optional[int] = None,
        sparse_residual_block_rows: Optional[int] = None,
        sparse_residual_block_cols: Optional[int] = None,
        sparse_residual_active_block_ids: Optional[torch.Tensor] = None,
        sparse_residual_block_ptr: Optional[torch.Tensor] = None,
        sparse_residual_local_indices: Optional[torch.Tensor] = None,
        sparse_residual_qvalues: Optional[torch.Tensor] = None,
        sparse_residual_scales: Optional[torch.Tensor] = None,
        sparse_residual_zero_points: Optional[torch.Tensor] = None,
        low_rank_a: Optional[torch.Tensor] = None,
        low_rank_b: Optional[torch.Tensor] = None,
        always_use_original: bool = False,
        protect_original_weight: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.transpose = bool(transpose)
        self.codebook_dim = int(codebook_dim)
        self.parallel_parts = int(parallel_parts)
        self.always_use_original = bool(always_use_original)
        self.protect_original_weight = bool(protect_original_weight)
        if compressed_in_features is None:
            compressed_in_features = self.in_features
        if compressed_out_features is None:
            compressed_out_features = self.out_features
        self.compressed_in_features = int(compressed_in_features)
        self.compressed_out_features = int(compressed_out_features)
        if self.compressed_in_features < 1 or self.compressed_in_features > self.in_features:
            raise ValueError(
                f"compressed_in_features must be in [1, {self.in_features}], got {self.compressed_in_features}"
            )
        if self.compressed_out_features < 1 or self.compressed_out_features > self.out_features:
            raise ValueError(
                f"compressed_out_features must be in [1, {self.out_features}], got {self.compressed_out_features}"
            )
        if self.parallel_parts < 1:
            raise ValueError(f"parallel_parts must be >= 1, got {self.parallel_parts}")
        if parallel_rows is None and parallel_cols is None:
            parallel_rows = self.parallel_parts
            parallel_cols = 1
        elif parallel_rows is None:
            parallel_cols = int(parallel_cols)
            if parallel_cols < 1:
                raise ValueError(f"parallel_cols must be >= 1, got {parallel_cols}")
            if self.parallel_parts % parallel_cols != 0:
                raise ValueError(
                    f"parallel_parts={self.parallel_parts} not divisible by parallel_cols={parallel_cols}"
                )
            parallel_rows = self.parallel_parts // parallel_cols
        elif parallel_cols is None:
            parallel_rows = int(parallel_rows)
            if parallel_rows < 1:
                raise ValueError(f"parallel_rows must be >= 1, got {parallel_rows}")
            if self.parallel_parts % parallel_rows != 0:
                raise ValueError(
                    f"parallel_parts={self.parallel_parts} not divisible by parallel_rows={parallel_rows}"
                )
            parallel_cols = self.parallel_parts // parallel_rows

        self.parallel_rows = int(parallel_rows)
        self.parallel_cols = int(parallel_cols)
        if self.parallel_rows < 1 or self.parallel_cols < 1:
            raise ValueError(
                f"parallel_rows/parallel_cols must be >= 1, got ({self.parallel_rows}, {self.parallel_cols})"
            )
        if self.parallel_rows * self.parallel_cols != self.parallel_parts:
            raise ValueError(
                f"parallel_rows*parallel_cols mismatch: {self.parallel_rows}*{self.parallel_cols} != {self.parallel_parts}"
            )

        split_rows = self.compressed_in_features if self.transpose else self.compressed_out_features
        split_cols = self.compressed_out_features if self.transpose else self.compressed_in_features
        if split_rows % self.parallel_rows != 0:
            raise ValueError(
                f"split_rows={split_rows} not divisible by parallel_rows={self.parallel_rows}"
            )
        if split_cols % self.parallel_cols != 0:
            raise ValueError(
                f"split_cols={split_cols} not divisible by parallel_cols={self.parallel_cols}"
            )
        # 排序代码，已关闭。旧全局 restore 参数校验保留如下；参数已从活动签名移除。
        # if restore_row_indices is not None or restore_col_indices is not None:
        #     raise ValueError("排序代码已关闭；VAELinear 不再接受 restore_row_indices / restore_col_indices。")
        # 排序代码，已关闭。旧全局 restore 注册逻辑保留如下：
        # if restore_row_indices is None:
        #     self.register_buffer("restore_row_indices", None, persistent=True)
        # else:
        #     restore_idx = restore_row_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     if restore_idx.ndim != 1:
        #         raise ValueError(
        #             f"restore_row_indices must be 1D, got shape={tuple(restore_idx.shape)}"
        #         )
        #     if int(restore_idx.numel()) != int(split_rows):
        #         raise ValueError(
        #             f"restore_row_indices size {int(restore_idx.numel())} != split_rows {int(split_rows)}"
        #         )
        #     self.register_buffer("restore_row_indices", restore_idx, persistent=True)
        # if restore_col_indices is None:
        #     self.register_buffer("restore_col_indices", None, persistent=True)
        # else:
        #     restore_col_idx = restore_col_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     if restore_col_idx.ndim != 1:
        #         raise ValueError(
        #             f"restore_col_indices must be 1D, got shape={tuple(restore_col_idx.shape)}"
        #         )
        #     if int(restore_col_idx.numel()) != int(split_cols):
        #         raise ValueError(
        #             f"restore_col_indices size {int(restore_col_idx.numel())} != split_cols {int(split_cols)}"
        #         )
        #     self.register_buffer("restore_col_indices", restore_col_idx, persistent=True)
        self.register_buffer("restore_row_indices", None, persistent=True)
        self.register_buffer("restore_col_indices", None, persistent=True)
        cols_per_part = split_cols // self.parallel_cols
        # 排序代码，已关闭。旧 part restore 参数校验保留如下；参数已从活动签名移除。
        # if part_restore_col_indices is not None:
        #     raise ValueError("排序代码已关闭；VAELinear 不再接受 part_restore_col_indices。")
        # 排序代码，已关闭。旧 part restore 注册逻辑保留如下：
        # if part_restore_col_indices is None:
        #     self.register_buffer("part_restore_col_indices", None, persistent=True)
        # else:
        #     part_restore_idx = part_restore_col_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     expected_shape = (int(self.parallel_parts), int(cols_per_part))
        #     if tuple(part_restore_idx.shape) != expected_shape:
        #         raise ValueError(
        #             f"part_restore_col_indices shape {tuple(part_restore_idx.shape)} != {expected_shape}"
        #         )
        #     for part_idx in range(int(part_restore_idx.shape[0])):
        #         local_restore = part_restore_idx[part_idx]
        #         if int(torch.unique(local_restore, sorted=False).numel()) != int(local_restore.numel()):
        #             raise ValueError(f"part_restore_col_indices[{part_idx}] contains duplicates.")
        #         if int(local_restore.min().item()) < 0 or int(local_restore.max().item()) >= int(cols_per_part):
        #             raise ValueError(
        #                 f"part_restore_col_indices[{part_idx}] must be within [0, {int(cols_per_part)}), got "
        #                 f"[{int(local_restore.min().item())}, {int(local_restore.max().item())}]"
        #             )
        #     self.register_buffer("part_restore_col_indices", part_restore_idx, persistent=True)
        self.register_buffer("part_restore_col_indices", None, persistent=True)

        if protected_input_indices is None:
            self.register_buffer("protected_input_indices", None, persistent=True)
        else:
            protected_idx = protected_input_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            if protected_idx.ndim != 1:
                raise ValueError(
                    f"protected_input_indices must be 1D, got shape={tuple(protected_idx.shape)}"
                )
            if protected_idx.numel() == 0:
                protected_idx = None
                self.register_buffer("protected_input_indices", None, persistent=True)
            else:
                if int(torch.unique(protected_idx, sorted=False).numel()) != int(protected_idx.numel()):
                    raise ValueError("protected_input_indices contains duplicates.")
                if int(protected_idx.min().item()) < 0 or int(protected_idx.max().item()) >= self.in_features:
                    raise ValueError(
                        f"protected_input_indices must be within [0, {self.in_features}), got "
                        f"[{int(protected_idx.min().item())}, {int(protected_idx.max().item())}]"
                    )
                self.register_buffer("protected_input_indices", protected_idx, persistent=True)
        protected_count = int(self.protected_input_indices.numel()) if isinstance(self.protected_input_indices, torch.Tensor) else 0
        if protected_count + self.compressed_in_features != self.in_features:
            raise ValueError(
                f"protected_input_indices + compressed_in_features mismatch: "
                f"{protected_count} + {self.compressed_in_features} != in_features {self.in_features}"
            )

        if protected_input_weight is None:
            self.register_parameter("protected_input_weight", None)
        else:
            if isinstance(protected_input_weight, nn.Parameter):
                protected_weight = protected_input_weight
            else:
                protected_weight = nn.Parameter(
                    protected_input_weight.detach().contiguous(),
                    requires_grad=False,
                )
            if protected_weight.ndim != 2:
                raise ValueError(
                    f"protected_input_weight must be 2D, got shape={tuple(protected_weight.shape)}"
                )
            if int(protected_weight.shape[0]) != protected_count or int(protected_weight.shape[1]) != self.out_features:
                raise ValueError(
                    f"protected_input_weight shape {tuple(protected_weight.shape)} != "
                    f"({protected_count}, {self.out_features})"
                )
            protected_weight.requires_grad = False
            self.register_parameter("protected_input_weight", protected_weight)
        if protected_count == 0 and self.compressed_in_features != self.in_features:
            raise ValueError(
                f"compressed_in_features={self.compressed_in_features} requires protected_input_indices to be present."
            )
        if protected_count > 0 and self.protected_input_weight is None:
            raise ValueError("protected_input_weight is required when protected_input_indices is provided.")

        if protected_output_indices is None:
            self.register_buffer("protected_output_indices", None, persistent=True)
        else:
            protected_out_idx = protected_output_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            if protected_out_idx.ndim != 1:
                raise ValueError(
                    f"protected_output_indices must be 1D, got shape={tuple(protected_out_idx.shape)}"
                )
            if protected_out_idx.numel() == 0:
                protected_out_idx = None
                self.register_buffer("protected_output_indices", None, persistent=True)
            else:
                if int(torch.unique(protected_out_idx, sorted=False).numel()) != int(protected_out_idx.numel()):
                    raise ValueError("protected_output_indices contains duplicates.")
                if int(protected_out_idx.min().item()) < 0 or int(protected_out_idx.max().item()) >= self.out_features:
                    raise ValueError(
                        f"protected_output_indices must be within [0, {self.out_features}), got "
                        f"[{int(protected_out_idx.min().item())}, {int(protected_out_idx.max().item())}]"
                    )
                self.register_buffer("protected_output_indices", protected_out_idx, persistent=True)
        protected_out_count = (
            int(self.protected_output_indices.numel())
            if isinstance(self.protected_output_indices, torch.Tensor)
            else 0
        )
        if protected_out_count + self.compressed_out_features != self.out_features:
            raise ValueError(
                f"protected_output_indices + compressed_out_features mismatch: "
                f"{protected_out_count} + {self.compressed_out_features} != out_features {self.out_features}"
            )

        if protected_output_weight is None:
            self.register_parameter("protected_output_weight", None)
        else:
            if isinstance(protected_output_weight, nn.Parameter):
                protected_out_weight = protected_output_weight
            else:
                protected_out_weight = nn.Parameter(
                    protected_output_weight.detach().contiguous(),
                    requires_grad=False,
                )
            if protected_out_weight.ndim != 2:
                raise ValueError(
                    f"protected_output_weight must be 2D, got shape={tuple(protected_out_weight.shape)}"
                )
            if (
                int(protected_out_weight.shape[0]) != protected_out_count
                or int(protected_out_weight.shape[1]) != self.in_features
            ):
                raise ValueError(
                    f"protected_output_weight shape {tuple(protected_out_weight.shape)} != "
                    f"({protected_out_count}, {self.in_features})"
                )
            protected_out_weight.requires_grad = False
            self.register_parameter("protected_output_weight", protected_out_weight)
        if protected_out_count == 0 and self.compressed_out_features != self.out_features:
            raise ValueError(
                f"compressed_out_features={self.compressed_out_features} requires protected_output_indices to be present."
            )
        if protected_out_count > 0 and self.protected_output_weight is None:
            raise ValueError("protected_output_weight is required when protected_output_indices is provided.")

        resolved_sparse_format = str(sparse_residual_format).strip().lower()
        if resolved_sparse_format not in SPARSE_RESIDUAL_FORMAT_CHOICES:
            raise ValueError(
                f"Unsupported sparse_residual_format={sparse_residual_format!r}. "
                f"Expected one of: {', '.join(SPARSE_RESIDUAL_FORMAT_CHOICES)}."
            )
        self.sparse_residual_format = resolved_sparse_format
        self.sparse_residual_index_bits = None if sparse_residual_index_bits is None else int(sparse_residual_index_bits)
        self.sparse_residual_value_bits = None if sparse_residual_value_bits is None else int(sparse_residual_value_bits)
        self.sparse_residual_block_rows = None if sparse_residual_block_rows is None else int(sparse_residual_block_rows)
        self.sparse_residual_block_cols = None if sparse_residual_block_cols is None else int(sparse_residual_block_cols)

        sparse_coo_payload_provided = any(
            item is not None
            for item in (
                sparse_residual_row_indices,
                sparse_residual_col_indices,
                sparse_residual_values,
            )
        )
        sparse_blocked_payload_provided = any(
            item is not None
            for item in (
                sparse_residual_active_block_ids,
                sparse_residual_block_ptr,
                sparse_residual_local_indices,
                sparse_residual_qvalues,
                sparse_residual_scales,
                sparse_residual_zero_points,
            )
        )
        if sparse_coo_payload_provided and sparse_blocked_payload_provided:
            raise ValueError("Sparse residual COO payload and blocked_quantized payload cannot be provided together.")

        if sparse_coo_payload_provided:
            if resolved_sparse_format != SPARSE_RESIDUAL_FORMAT_COO_FP16:
                raise ValueError(
                    f"sparse_residual_format={resolved_sparse_format!r} is incompatible with COO payload."
                )
            if (
                sparse_residual_row_indices is None
                or sparse_residual_col_indices is None
                or sparse_residual_values is None
            ):
                raise ValueError(
                    "sparse_residual_row_indices, sparse_residual_col_indices, and sparse_residual_values "
                    "must be provided together."
                )
            sparse_row_idx = sparse_residual_row_indices.detach().to(device="cpu").contiguous()
            sparse_col_idx = sparse_residual_col_indices.detach().to(device="cpu").contiguous()
            sparse_values = sparse_residual_values.detach().to(device="cpu").contiguous()
            if sparse_row_idx.ndim != 1 or sparse_col_idx.ndim != 1 or sparse_values.ndim != 1:
                raise ValueError("sparse residual COO payload must be 1D tensors.")
            if sparse_row_idx.numel() != sparse_col_idx.numel() or sparse_row_idx.numel() != sparse_values.numel():
                raise ValueError(
                    "sparse residual COO payload length mismatch: "
                    f"rows={int(sparse_row_idx.numel())} cols={int(sparse_col_idx.numel())} "
                    f"values={int(sparse_values.numel())}"
                )
            if sparse_row_idx.numel() == 0:
                self.register_buffer("sparse_residual_row_indices", None, persistent=True)
                self.register_buffer("sparse_residual_col_indices", None, persistent=True)
                self.register_buffer("sparse_residual_values", None, persistent=True)
            else:
                if sparse_row_idx.is_floating_point() or sparse_row_idx.is_complex() or sparse_row_idx.dtype == torch.bool:
                    raise ValueError(f"sparse_residual_row_indices must be integer dtype, got {sparse_row_idx.dtype}")
                if sparse_col_idx.is_floating_point() or sparse_col_idx.is_complex() or sparse_col_idx.dtype == torch.bool:
                    raise ValueError(f"sparse_residual_col_indices must be integer dtype, got {sparse_col_idx.dtype}")
                if not sparse_values.is_floating_point():
                    raise ValueError(f"sparse_residual_values must be floating dtype, got {sparse_values.dtype}")
                sparse_row_check = sparse_row_idx.to(dtype=torch.int64)
                sparse_col_check = sparse_col_idx.to(dtype=torch.int64)
                if int(sparse_row_check.min().item()) < 0 or int(sparse_row_check.max().item()) >= self.out_features:
                    raise ValueError(
                        f"sparse_residual_row_indices must be within [0, {self.out_features}), got "
                        f"[{int(sparse_row_check.min().item())}, {int(sparse_row_check.max().item())}]"
                    )
                if int(sparse_col_check.min().item()) < 0 or int(sparse_col_check.max().item()) >= self.in_features:
                    raise ValueError(
                        f"sparse_residual_col_indices must be within [0, {self.in_features}), got "
                        f"[{int(sparse_col_check.min().item())}, {int(sparse_col_check.max().item())}]"
                    )
                self.register_buffer("sparse_residual_row_indices", sparse_row_idx, persistent=True)
                self.register_buffer("sparse_residual_col_indices", sparse_col_idx, persistent=True)
                self.register_buffer("sparse_residual_values", sparse_values, persistent=True)
            self.register_buffer("sparse_residual_active_block_ids", None, persistent=True)
            self.register_buffer("sparse_residual_block_ptr", None, persistent=True)
            self.register_buffer("sparse_residual_local_indices", None, persistent=True)
            self.register_buffer("sparse_residual_qvalues", None, persistent=True)
            self.register_buffer("sparse_residual_scales", None, persistent=True)
            self.register_buffer("sparse_residual_zero_points", None, persistent=True)
        elif sparse_blocked_payload_provided:
            if resolved_sparse_format != SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED:
                raise ValueError(
                    f"sparse_residual_format={resolved_sparse_format!r} is incompatible with blocked payload."
                )
            if self.sparse_residual_index_bits not in SPARSE_RESIDUAL_INDEX_BITS_CHOICES:
                raise ValueError(
                    f"sparse_residual_index_bits must be one of {SPARSE_RESIDUAL_INDEX_BITS_CHOICES}, "
                    f"got {self.sparse_residual_index_bits}."
                )
            if self.sparse_residual_value_bits not in SPARSE_RESIDUAL_VALUE_BITS_CHOICES:
                raise ValueError(
                    f"sparse_residual_value_bits must be one of {SPARSE_RESIDUAL_VALUE_BITS_CHOICES}, "
                    f"got {self.sparse_residual_value_bits}."
                )
            if self.sparse_residual_block_rows is None or self.sparse_residual_block_cols is None:
                raise ValueError("Blocked sparse residual payload requires sparse_residual_block_rows/block_cols.")
            validate_sparse_residual_block_shape(
                block_rows=self.sparse_residual_block_rows,
                block_cols=self.sparse_residual_block_cols,
                index_bits=self.sparse_residual_index_bits,
                arg_name="sparse residual block shape",
            )
            blocked_items = (
                sparse_residual_active_block_ids,
                sparse_residual_block_ptr,
                sparse_residual_local_indices,
                sparse_residual_qvalues,
                sparse_residual_scales,
                sparse_residual_zero_points,
            )
            if any(item is None for item in blocked_items):
                raise ValueError(
                    "Blocked sparse residual payload requires active_block_ids, block_ptr, local_indices, "
                    "qvalues, scales, and zero_points to be provided together."
                )
            active_block_ids = sparse_residual_active_block_ids.detach().to(device="cpu").contiguous()
            block_ptr = sparse_residual_block_ptr.detach().to(device="cpu").contiguous()
            local_indices = sparse_residual_local_indices.detach().to(device="cpu").contiguous()
            qvalues = sparse_residual_qvalues.detach().to(device="cpu").contiguous()
            scales = sparse_residual_scales.detach().to(device="cpu").contiguous()
            zero_points = sparse_residual_zero_points.detach().to(device="cpu").contiguous()
            if (
                active_block_ids.ndim != 1
                or block_ptr.ndim != 1
                or local_indices.ndim != 1
                or qvalues.ndim != 1
                or scales.ndim != 1
                or zero_points.ndim != 1
            ):
                raise ValueError("Blocked sparse residual payload must be 1D tensors.")
            if (
                active_block_ids.is_floating_point()
                or active_block_ids.is_complex()
                or active_block_ids.dtype == torch.bool
            ):
                raise ValueError(
                    f"sparse_residual_active_block_ids must be integer dtype, got {active_block_ids.dtype}"
                )
            if block_ptr.is_floating_point() or block_ptr.is_complex() or block_ptr.dtype == torch.bool:
                raise ValueError(f"sparse_residual_block_ptr must be integer dtype, got {block_ptr.dtype}")
            if local_indices.is_floating_point() or local_indices.is_complex() or local_indices.dtype == torch.bool:
                raise ValueError(f"sparse_residual_local_indices must be integer dtype, got {local_indices.dtype}")
            if qvalues.is_floating_point() or qvalues.is_complex() or qvalues.dtype == torch.bool:
                raise ValueError(f"sparse_residual_qvalues must be integer dtype, got {qvalues.dtype}")
            if not scales.is_floating_point():
                raise ValueError(f"sparse_residual_scales must be floating dtype, got {scales.dtype}")
            if not zero_points.is_floating_point():
                raise ValueError(f"sparse_residual_zero_points must be floating dtype, got {zero_points.dtype}")
            decoded_row, decoded_col, decoded_values = decode_blocked_quantized_sparse_residual(
                active_block_ids=active_block_ids,
                block_ptr=block_ptr,
                local_indices=local_indices,
                qvalues=qvalues,
                scales=scales,
                zero_points=zero_points,
                out_features=self.out_features,
                in_features=self.in_features,
                block_rows=self.sparse_residual_block_rows,
                block_cols=self.sparse_residual_block_cols,
                index_bits=self.sparse_residual_index_bits,
                value_bits=self.sparse_residual_value_bits,
                value_dtype=torch.float32,
                device=torch.device("cpu"),
            )
            nnz = int(decoded_values.numel())
            if nnz == 0:
                self.register_buffer("sparse_residual_active_block_ids", None, persistent=True)
                self.register_buffer("sparse_residual_block_ptr", None, persistent=True)
                self.register_buffer("sparse_residual_local_indices", None, persistent=True)
                self.register_buffer("sparse_residual_qvalues", None, persistent=True)
                self.register_buffer("sparse_residual_scales", None, persistent=True)
                self.register_buffer("sparse_residual_zero_points", None, persistent=True)
            else:
                self.register_buffer("sparse_residual_active_block_ids", active_block_ids, persistent=True)
                self.register_buffer("sparse_residual_block_ptr", block_ptr, persistent=True)
                self.register_buffer("sparse_residual_local_indices", local_indices, persistent=True)
                self.register_buffer("sparse_residual_qvalues", qvalues, persistent=True)
                self.register_buffer("sparse_residual_scales", scales, persistent=True)
                self.register_buffer("sparse_residual_zero_points", zero_points, persistent=True)
            self.register_buffer("sparse_residual_row_indices", None, persistent=True)
            self.register_buffer("sparse_residual_col_indices", None, persistent=True)
            self.register_buffer("sparse_residual_values", None, persistent=True)
        else:
            self.register_buffer("sparse_residual_row_indices", None, persistent=True)
            self.register_buffer("sparse_residual_col_indices", None, persistent=True)
            self.register_buffer("sparse_residual_values", None, persistent=True)
            self.register_buffer("sparse_residual_active_block_ids", None, persistent=True)
            self.register_buffer("sparse_residual_block_ptr", None, persistent=True)
            self.register_buffer("sparse_residual_local_indices", None, persistent=True)
            self.register_buffer("sparse_residual_qvalues", None, persistent=True)
            self.register_buffer("sparse_residual_scales", None, persistent=True)
            self.register_buffer("sparse_residual_zero_points", None, persistent=True)

        if low_rank_a is None and low_rank_b is None:
            self.register_parameter("low_rank_a", None)
            self.register_parameter("low_rank_b", None)
        elif low_rank_a is None or low_rank_b is None:
            raise ValueError("low_rank_a and low_rank_b must be provided together.")
        else:
            if isinstance(low_rank_a, nn.Parameter):
                low_rank_a_param = low_rank_a
            else:
                low_rank_a_param = nn.Parameter(low_rank_a.detach().contiguous(), requires_grad=False)
            if isinstance(low_rank_b, nn.Parameter):
                low_rank_b_param = low_rank_b
            else:
                low_rank_b_param = nn.Parameter(low_rank_b.detach().contiguous(), requires_grad=False)
            if low_rank_a_param.ndim != 2 or low_rank_b_param.ndim != 2:
                raise ValueError(
                    f"low_rank_a/low_rank_b must be 2D, got {tuple(low_rank_a_param.shape)} and {tuple(low_rank_b_param.shape)}"
                )
            if int(low_rank_a_param.shape[0]) != self.out_features:
                raise ValueError(
                    f"low_rank_a rows {int(low_rank_a_param.shape[0])} != out_features {self.out_features}"
                )
            if int(low_rank_b_param.shape[1]) != self.in_features:
                raise ValueError(
                    f"low_rank_b cols {int(low_rank_b_param.shape[1])} != in_features {self.in_features}"
                )
            if int(low_rank_a_param.shape[1]) != int(low_rank_b_param.shape[0]):
                raise ValueError(
                    f"low rank inner dim mismatch: {int(low_rank_a_param.shape[1])} != {int(low_rank_b_param.shape[0])}"
                )
            if not low_rank_a_param.is_floating_point() or not low_rank_b_param.is_floating_point():
                raise ValueError("low_rank_a and low_rank_b must be floating tensors.")
            low_rank_a_param.requires_grad = False
            low_rank_b_param.requires_grad = False
            self.register_parameter("low_rank_a", low_rank_a_param)
            self.register_parameter("low_rank_b", low_rank_b_param)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias
            self.bias.requires_grad = False

        if original_weight is None:
            self.register_parameter("original_weight", None)
        else:
            if isinstance(original_weight, nn.Parameter):
                self.original_weight = original_weight
            else:
                self.original_weight = nn.Parameter(original_weight)
            self.original_weight.requires_grad = False
            if tuple(self.original_weight.shape) != (self.out_features, self.in_features):
                raise ValueError(
                    f"original_weight shape {tuple(self.original_weight.shape)} != "
                    f"({self.out_features}, {self.in_features})"
                )
        self.temporary = not self.always_use_original
        self.cache_decoded_weight = True
        self.trainable_decode = False
        self.parallel_stage_decode = False
        self._parallel_stage_layout: List[Tuple[int, int]] = []
        self._parallel_stage_layout_is_stage_major = False
        self._parallel_stage_restore_identity = False
        self._parallel_stage_grouped_vq_runtime_key: Optional[Tuple[str, torch.dtype]] = None
        self._parallel_stage_model_indices_runtime_key: Optional[str] = None
        self._parallel_stage_restore_index_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_sparse_residual_row_indices", None, persistent=False)
        self.register_buffer("_cached_sparse_residual_col_indices", None, persistent=False)
        self.register_buffer("_cached_sparse_residual_values", None, persistent=False)
        self.register_buffer("_parallel_stage_grouped_vq_weight", None, persistent=False)
        self.register_buffer("_parallel_stage_grouped_vq_runtime", None, persistent=False)
        self.register_buffer("_parallel_stage_model_indices", None, persistent=False)
        self.register_buffer("_parallel_stage_model_indices_runtime", None, persistent=False)

        use_stage_payload = stage_vq_weights is not None or stage_decoders is not None
        if use_stage_payload:
            if stage_vq_weights is None or stage_decoders is None:
                raise ValueError("stage_vq_weights and stage_decoders must be provided together.")
            if vq_storage_specs is not None:
                raise ValueError("vq_storage_specs cannot be used together with stage_vq_weights/stage_decoders.")
            stage_vq_payload = self._normalize_stage_payload(
                stage_vq_weights,
                parallel_parts=self.parallel_parts,
                payload_name="stage_vq_weights",
            )
            normalized_stage_vq_storage_specs = None
            if stage_vq_storage_specs is not None:
                normalized_stage_vq_storage_specs = self._normalize_stage_payload(
                    stage_vq_storage_specs,
                    parallel_parts=self.parallel_parts,
                    payload_name="stage_vq_storage_specs",
                )
            stage_decoder_payload = self._normalize_stage_payload(
                stage_decoders,
                parallel_parts=self.parallel_parts,
                payload_name="stage_decoders",
            )
        else:
            if vq_weight is None or decoder is None:
                raise ValueError("vq_weight and decoder are required when stage payloads are not provided.")
            if stage_vq_storage_specs is not None:
                raise ValueError("stage_vq_storage_specs cannot be used when stage_vq_weights is not provided.")
            if self.parallel_parts == 1:
                stage_vq_payload = [[vq_weight]]
                normalized_stage_vq_storage_specs = None if vq_storage_specs is None else [[vq_storage_specs]]
                stage_decoder_payload = [[decoder]]
            else:
                if not isinstance(vq_weight, (list, tuple)):
                    raise ValueError("multi-part mode requires vq_weight list/tuple.")
                if vq_storage_specs is not None and not isinstance(vq_storage_specs, (list, tuple)):
                    raise ValueError("multi-part mode requires vq_storage_specs list/tuple.")
                if not isinstance(decoder, (list, tuple)):
                    raise ValueError("multi-part mode requires decoder list/tuple.")
                if len(vq_weight) != self.parallel_parts:
                    raise ValueError(
                        f"vq_weight length {len(vq_weight)} != parallel_parts {self.parallel_parts}"
                    )
                if vq_storage_specs is not None and len(vq_storage_specs) != self.parallel_parts:
                    raise ValueError(
                        f"vq_storage_specs length {len(vq_storage_specs)} != parallel_parts {self.parallel_parts}"
                    )
                if len(decoder) != self.parallel_parts:
                    raise ValueError(
                        f"decoder length {len(decoder)} != parallel_parts {self.parallel_parts}"
                    )
                stage_vq_payload = [list(vq_weight)]
                normalized_stage_vq_storage_specs = None if vq_storage_specs is None else [list(vq_storage_specs)]
                stage_decoder_payload = [list(decoder)]

        if len(stage_vq_payload) != len(stage_decoder_payload):
            raise ValueError(
                f"stage payload length mismatch: "
                f"stage_vq_weights={len(stage_vq_payload)} vs stage_decoders={len(stage_decoder_payload)}"
            )
        if normalized_stage_vq_storage_specs is not None and len(normalized_stage_vq_storage_specs) != len(stage_vq_payload):
            raise ValueError(
                f"stage_vq_storage_specs length {len(normalized_stage_vq_storage_specs)} != "
                f"stage_vq_weights {len(stage_vq_payload)}"
            )
        self.residual_stages = int(len(stage_vq_payload))
        if self.residual_stages < 1:
            raise ValueError("residual_stages must be >= 1")
        self._multi_parts = self.parallel_parts > 1
        if stage_codebook_dims is None:
            self.stage_codebook_dims = [int(self.codebook_dim) for _ in range(self.residual_stages)]
        else:
            dims = [int(v) for v in stage_codebook_dims]
            if len(dims) == 0:
                raise ValueError("stage_codebook_dims cannot be empty.")
            if len(dims) == 1 and self.residual_stages > 1:
                dims = dims * self.residual_stages
            if len(dims) != self.residual_stages:
                raise ValueError(
                    f"stage_codebook_dims length {len(dims)} != residual_stages {self.residual_stages}"
                )
            if any(int(v) < 1 for v in dims):
                raise ValueError(f"stage_codebook_dims must be >=1, got {dims}")
            self.stage_codebook_dims = [int(v) for v in dims]
        # Keep legacy scalar field for compatibility.
        self.codebook_dim = int(self.stage_codebook_dims[0])

        # 排序代码，已关闭。旧多 stage restore 参数校验保留如下；参数已从活动签名移除。
        # if (
        #     stage_restore_row_indices is not None
        #     or stage_restore_col_indices is not None
        #     or stage_part_restore_col_indices is not None
        # ):
        #     raise ValueError("排序代码已关闭；VAELinear 不再接受 stage restore indices。")
        # 排序代码，已关闭。旧多 stage restore 校验和注册逻辑保留如下：
        # normalized_stage_restore_rows = self._normalize_optional_stage_tensor_payload(
        #     stage_restore_row_indices,
        #     residual_stages=self.residual_stages,
        #     payload_name="stage_restore_row_indices",
        # )
        # normalized_stage_restore_cols = self._normalize_optional_stage_tensor_payload(
        #     stage_restore_col_indices,
        #     residual_stages=self.residual_stages,
        #     payload_name="stage_restore_col_indices",
        # )
        # normalized_stage_part_restore_cols = self._normalize_optional_stage_tensor_payload(
        #     stage_part_restore_col_indices,
        #     residual_stages=self.residual_stages,
        #     payload_name="stage_part_restore_col_indices",
        # )
        #
        # legacy_restore_row = getattr(self, "restore_row_indices", None)
        # legacy_restore_col = getattr(self, "restore_col_indices", None)
        # legacy_part_restore_col = getattr(self, "part_restore_col_indices", None)
        #
        # stage_restore_rows: List[Optional[torch.Tensor]] = []
        # for stage_idx in range(self.residual_stages):
        #     item = legacy_restore_row if normalized_stage_restore_rows is None else normalized_stage_restore_rows[stage_idx]
        #     if item is None:
        #         stage_restore_rows.append(None)
        #         continue
        #     restore_idx = item.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     if restore_idx.ndim != 1:
        #         raise ValueError(
        #             f"stage_restore_row_indices[{stage_idx}] must be 1D, got shape={tuple(restore_idx.shape)}"
        #         )
        #     if int(restore_idx.numel()) != int(split_rows):
        #         raise ValueError(
        #             f"stage_restore_row_indices[{stage_idx}] size {int(restore_idx.numel())} != split_rows {int(split_rows)}"
        #         )
        #     stage_restore_rows.append(restore_idx)
        #
        # stage_restore_cols: List[Optional[torch.Tensor]] = []
        # for stage_idx in range(self.residual_stages):
        #     item = legacy_restore_col if normalized_stage_restore_cols is None else normalized_stage_restore_cols[stage_idx]
        #     if item is None:
        #         stage_restore_cols.append(None)
        #         continue
        #     restore_idx = item.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     if restore_idx.ndim != 1:
        #         raise ValueError(
        #             f"stage_restore_col_indices[{stage_idx}] must be 1D, got shape={tuple(restore_idx.shape)}"
        #         )
        #     if int(restore_idx.numel()) != int(split_cols):
        #         raise ValueError(
        #             f"stage_restore_col_indices[{stage_idx}] size {int(restore_idx.numel())} != split_cols {int(split_cols)}"
        #         )
        #     stage_restore_cols.append(restore_idx)
        #
        # stage_part_restore_cols: List[Optional[torch.Tensor]] = []
        # expected_part_restore_shape = (int(self.parallel_parts), int(cols_per_part))
        # for stage_idx in range(self.residual_stages):
        #     item = legacy_part_restore_col if normalized_stage_part_restore_cols is None else normalized_stage_part_restore_cols[stage_idx]
        #     if item is None:
        #         stage_part_restore_cols.append(None)
        #         continue
        #     part_restore_idx = item.detach().to(device="cpu", dtype=torch.long).contiguous()
        #     if tuple(part_restore_idx.shape) != expected_part_restore_shape:
        #         raise ValueError(
        #             f"stage_part_restore_col_indices[{stage_idx}] shape {tuple(part_restore_idx.shape)} != {expected_part_restore_shape}"
        #         )
        #     for part_idx in range(int(part_restore_idx.shape[0])):
        #         local_restore = part_restore_idx[part_idx]
        #         if int(torch.unique(local_restore, sorted=False).numel()) != int(local_restore.numel()):
        #             raise ValueError(
        #                 f"stage_part_restore_col_indices[{stage_idx}][{part_idx}] contains duplicates."
        #             )
        #         if int(local_restore.min().item()) < 0 or int(local_restore.max().item()) >= int(cols_per_part):
        #             raise ValueError(
        #                 f"stage_part_restore_col_indices[{stage_idx}][{part_idx}] must be within [0, {int(cols_per_part)}), got "
        #                 f"[{int(local_restore.min().item())}, {int(local_restore.max().item())}]"
        #             )
        #     stage_part_restore_cols.append(part_restore_idx)
        #
        # self.restore_row_indices = stage_restore_rows[0]
        # self.restore_col_indices = stage_restore_cols[0]
        # self.part_restore_col_indices = stage_part_restore_cols[0]
        # for stage_idx in range(1, self.residual_stages):
        #     self.register_buffer(f"restore_row_indices_s{stage_idx}", stage_restore_rows[stage_idx], persistent=True)
        #     self.register_buffer(f"restore_col_indices_s{stage_idx}", stage_restore_cols[stage_idx], persistent=True)
        #     self.register_buffer(f"part_restore_col_indices_s{stage_idx}", stage_part_restore_cols[stage_idx], persistent=True)
        self.restore_row_indices = None
        self.restore_col_indices = None
        self.part_restore_col_indices = None
        for stage_idx in range(1, self.residual_stages):
            self.register_buffer(f"restore_row_indices_s{stage_idx}", None, persistent=True)
            self.register_buffer(f"restore_col_indices_s{stage_idx}", None, persistent=True)
            self.register_buffer(f"part_restore_col_indices_s{stage_idx}", None, persistent=True)

        self._stage_vq_storage_specs: List[List[Dict[str, Any]]] = []
        for stage_idx, stage_parts in enumerate(stage_vq_payload):
            stage_specs: List[Dict[str, Any]] = []
            spec_parts = None if normalized_stage_vq_storage_specs is None else normalized_stage_vq_storage_specs[stage_idx]
            for part_idx, w in enumerate(stage_parts):
                if not isinstance(w, torch.Tensor):
                    raise TypeError(
                        f"stage_vq_weights[{stage_idx}][{part_idx}] must be Tensor, got {type(w)}"
                    )
                if spec_parts is None:
                    logical_shape = tuple(int(v) for v in w.shape)
                    packed_storage = pack_bool_tensor_to_uint8(
                        w.detach().contiguous(),
                        logical_shape=logical_shape,
                    )
                    storage_spec = build_bitpack_u8_spec(logical_shape=logical_shape)
                else:
                    storage_spec = validate_bitpack_u8_spec(
                        spec_parts[part_idx],
                        arg_name=f"stage_vq_storage_specs[{stage_idx}][{part_idx}]",
                    )
                    if w.dtype != torch.uint8:
                        raise ValueError(
                            f"stage_vq_weights[{stage_idx}][{part_idx}] must be torch.uint8 when "
                            "stage_vq_storage_specs is provided."
                        )
                    if tuple(int(v) for v in w.shape) != tuple(int(v) for v in storage_spec["shape"]):
                        raise ValueError(
                            f"stage_vq_weights[{stage_idx}][{part_idx}] shape {tuple(int(v) for v in w.shape)} != "
                            f"storage spec shape {tuple(int(v) for v in storage_spec['shape'])}"
                        )
                    packed_storage = w.detach().contiguous()
                stage_specs.append(dict(storage_spec))
                if stage_idx == 0:
                    if self._multi_parts:
                        self.register_buffer(f"vq_weight_{part_idx}", packed_storage, persistent=True)
                    else:
                        self.register_buffer("vq_weight", packed_storage, persistent=True)
                else:
                    if self._multi_parts:
                        self.register_buffer(f"vq_weight_s{stage_idx}_{part_idx}", packed_storage, persistent=True)
                    else:
                        self.register_buffer(f"vq_weight_s{stage_idx}", packed_storage, persistent=True)
            self._stage_vq_storage_specs.append(stage_specs)

        for stage_idx, stage_parts in enumerate(stage_decoder_payload):
            for part_idx, dec in enumerate(stage_parts):
                if not isinstance(dec, nn.Module):
                    raise TypeError(
                        f"stage_decoders[{stage_idx}][{part_idx}] must be nn.Module, got {type(dec)}"
                    )
            if stage_idx == 0:
                if self._multi_parts:
                    self.decoders = nn.ModuleList(list(stage_parts))
                else:
                    self.decoder = stage_parts[0]
            else:
                if self._multi_parts:
                    setattr(self, f"decoders_s{stage_idx}", nn.ModuleList(list(stage_parts)))
                else:
                    setattr(self, f"decoder_s{stage_idx}", stage_parts[0])

        expected_numel = self.compressed_in_features * self.compressed_out_features
        for stage_idx in range(self.residual_stages):
            stage_codebook_dim = int(self.stage_codebook_dims[stage_idx])
            total_numel = 0
            for part_idx in range(self.parallel_parts):
                vq_spec = self.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx)
                logical_shape = tuple(int(v) for v in vq_spec["logical_shape"])
                if len(logical_shape) != 3 or int(logical_shape[1]) != 1:
                    raise ValueError(
                        f"stage {stage_idx} part {part_idx} logical_shape must be [N_blocks, 1, latent_dim], "
                        f"got {logical_shape}"
                    )
                total_numel += int(logical_shape[0]) * stage_codebook_dim
            if total_numel != expected_numel:
                raise ValueError(
                    f"stage {stage_idx} vq_weight total mismatch: total={total_numel} "
                    f"expected={expected_numel} (compressed_out*compressed_in), stage_codebook_dim={stage_codebook_dim}"
                )

    @staticmethod
    def _normalize_stage_payload(
        payload: Sequence[Any],
        *,
        parallel_parts: int,
        payload_name: str,
    ) -> List[List[Any]]:
        if not isinstance(payload, (list, tuple)):
            raise ValueError(f"{payload_name} must be a list/tuple, got {type(payload)}")
        if len(payload) < 1:
            raise ValueError(f"{payload_name} cannot be empty.")

        normalized: List[List[Any]] = []
        for stage_idx, stage_item in enumerate(payload):
            if parallel_parts == 1:
                if isinstance(stage_item, (list, tuple)):
                    if len(stage_item) != 1:
                        raise ValueError(
                            f"{payload_name}[{stage_idx}] must contain exactly 1 item for parallel_parts=1."
                        )
                    normalized.append([stage_item[0]])
                else:
                    normalized.append([stage_item])
            else:
                if not isinstance(stage_item, (list, tuple)):
                    raise ValueError(
                        f"{payload_name}[{stage_idx}] must be list/tuple for parallel_parts={parallel_parts}."
                    )
                if len(stage_item) != parallel_parts:
                    raise ValueError(
                        f"{payload_name}[{stage_idx}] length {len(stage_item)} != parallel_parts {parallel_parts}"
                    )
                normalized.append(list(stage_item))
        return normalized

    @staticmethod
    def _normalize_optional_stage_tensor_payload(
        payload: Optional[Sequence[Optional[torch.Tensor]]],
        *,
        residual_stages: int,
        payload_name: str,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if payload is None:
            return None
        if not isinstance(payload, (list, tuple)):
            payload_items = [payload]
        else:
            payload_items = list(payload)
        if len(payload_items) == 0:
            raise ValueError(f"{payload_name} cannot be empty.")
        if len(payload_items) == 1 and int(residual_stages) > 1:
            payload_items = payload_items * int(residual_stages)
        if len(payload_items) != int(residual_stages):
            raise ValueError(
                f"{payload_name} length {len(payload_items)} != residual_stages {int(residual_stages)}"
            )
        return list(payload_items)

    def get_stage_part_vq_weight(self, stage_idx: int, part_idx: int = 0) -> torch.Tensor:
        storage = self.get_stage_part_vq_storage(stage_idx=stage_idx, part_idx=part_idx)
        spec = self.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx)
        return unpack_uint8_tensor_to_bool(
            storage,
            logical_shape=tuple(int(v) for v in spec["logical_shape"]),
        )

    def get_stage_part_vq_storage(self, stage_idx: int, part_idx: int = 0) -> torch.Tensor:
        stage_idx = int(stage_idx)
        part_idx = int(part_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if part_idx < 0 or part_idx >= self.parallel_parts:
            raise IndexError(f"part_idx out of range: {part_idx} vs parallel_parts={self.parallel_parts}")
        if stage_idx == 0:
            if self._multi_parts:
                return getattr(self, f"vq_weight_{part_idx}")
            return self.vq_weight
        if self._multi_parts:
            return getattr(self, f"vq_weight_s{stage_idx}_{part_idx}")
        if part_idx != 0:
            raise IndexError("single-part VAELinear only supports part_idx=0")
        return getattr(self, f"vq_weight_s{stage_idx}")

    def get_stage_part_vq_spec(self, stage_idx: int, part_idx: int = 0) -> Dict[str, Any]:
        stage_idx = int(stage_idx)
        part_idx = int(part_idx)
        if stage_idx < 0 or stage_idx >= len(self._stage_vq_storage_specs):
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={len(self._stage_vq_storage_specs)}")
        if part_idx < 0 or part_idx >= len(self._stage_vq_storage_specs[stage_idx]):
            raise IndexError(
                f"part_idx out of range: {part_idx} vs stage_vq_storage_specs[{stage_idx}]={len(self._stage_vq_storage_specs[stage_idx])}"
            )
        return dict(self._stage_vq_storage_specs[stage_idx][part_idx])

    def get_stage_part_decoder(self, stage_idx: int, part_idx: int = 0) -> nn.Module:
        stage_idx = int(stage_idx)
        part_idx = int(part_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if part_idx < 0 or part_idx >= self.parallel_parts:
            raise IndexError(f"part_idx out of range: {part_idx} vs parallel_parts={self.parallel_parts}")
        packed_decoder = getattr(self, "_parallel_stage_decoder", None)
        if packed_decoder is not None:
            model_idx = self._parallel_stage_model_index(stage_idx=stage_idx, part_idx=part_idx)
            return packed_decoder.get_sub_decoder(model_idx)
        if stage_idx == 0:
            if self._multi_parts:
                return self.decoders[part_idx]
            return self.decoder
        if self._multi_parts:
            return getattr(self, f"decoders_s{stage_idx}")[part_idx]
        if part_idx != 0:
            raise IndexError("single-part VAELinear only supports part_idx=0")
        return getattr(self, f"decoder_s{stage_idx}")

    def _parallel_stage_model_index(self, *, stage_idx: int, part_idx: int) -> int:
        target = (int(stage_idx), int(part_idx))
        for model_idx, item in enumerate(getattr(self, "_parallel_stage_layout", [])):
            if tuple(item) == target:
                return int(model_idx)
        raise RuntimeError(
            f"parallel stage decoder layout does not contain stage={int(stage_idx)} part={int(part_idx)}."
        )

    def _iter_stage_part_decoders_for_pack(self) -> Tuple[List[nn.Module], List[Tuple[int, int]]]:
        if getattr(self, "_parallel_stage_decoder", None) is not None:
            raise RuntimeError("parallel stage decoder is already enabled.")
        decoders: List[nn.Module] = []
        layout: List[Tuple[int, int]] = []
        for stage_idx in range(int(self.residual_stages)):
            for part_idx in range(int(self.parallel_parts)):
                decoders.append(self.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx))
                layout.append((int(stage_idx), int(part_idx)))
        return decoders, layout

    def _clear_parallel_stage_decode_runtime_cache(self) -> None:
        self._parallel_stage_grouped_vq_runtime_key = None
        self._parallel_stage_model_indices_runtime_key = None
        self._parallel_stage_restore_index_cache = {}
        self._parallel_stage_grouped_vq_runtime = None
        self._parallel_stage_model_indices_runtime = None

    def _clear_parallel_stage_decode_plan(self) -> None:
        self._clear_parallel_stage_decode_runtime_cache()
        self._parallel_stage_layout_is_stage_major = False
        self._parallel_stage_restore_identity = False
        self._parallel_stage_grouped_vq_weight = None
        self._parallel_stage_model_indices = None

    def _build_parallel_stage_grouped_vq_weight(self, layout: Sequence[Tuple[int, int]]) -> torch.Tensor:
        if not layout:
            raise RuntimeError("parallel stage layout cannot be empty.")
        vq_tensors: List[torch.Tensor] = []
        first_shape = None
        for stage_idx, part_idx in layout:
            vq_weight = self.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
            if bool(vq_weight.requires_grad):
                raise RuntimeError("parallel_stage_decode runtime VQ packing requires frozen vq_weight tensors.")
            if vq_weight.ndim != 3 or int(vq_weight.shape[1]) != 1:
                raise ValueError(
                    f"parallel_stage_decode expects vq shape [N_blocks, 1, latent_dim], got {tuple(vq_weight.shape)} "
                    f"for stage={int(stage_idx)} part={int(part_idx)}."
                )
            shape = tuple(int(v) for v in vq_weight.shape)
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise ValueError(
                    f"parallel_stage_decode requires identical vq shapes, got {shape} vs {first_shape}."
                )
            vq_tensors.append(vq_weight.detach())
        grouped_vq = torch.cat(vq_tensors, dim=1).contiguous()
        if int(grouped_vq.shape[1]) != len(layout):
            raise RuntimeError(
                f"parallel grouped VQ model axis {int(grouped_vq.shape[1])} != layout length {len(layout)}."
            )
        return grouped_vq

    def _parallel_stage_restore_is_identity(self) -> bool:
        # 排序代码，已关闭。原 restore identity 检查保留如下：
        # for stage_idx in range(int(self.residual_stages)):
        #     if self.get_stage_restore_row_indices(stage_idx) is not None:
        #         return False
        #     if self.get_stage_restore_col_indices(stage_idx) is not None:
        #         return False
        #     if self.get_stage_part_restore_col_indices(stage_idx) is not None:
        #         return False
        return True

    def _build_parallel_stage_decode_plan(self) -> None:
        layout = list(getattr(self, "_parallel_stage_layout", []))
        expected = int(self.residual_stages) * int(self.parallel_parts)
        if len(layout) != expected:
            raise RuntimeError(f"parallel stage layout length {len(layout)} != expected {expected}.")

        stage_major = [
            (int(stage_idx), int(part_idx))
            for stage_idx in range(int(self.residual_stages))
            for part_idx in range(int(self.parallel_parts))
        ]
        model_indices = torch.empty(
            (int(self.residual_stages), int(self.parallel_parts)),
            dtype=torch.long,
            device="cpu",
        )
        seen = set()
        for model_idx, (stage_idx, part_idx) in enumerate(layout):
            key = (int(stage_idx), int(part_idx))
            if key in seen:
                raise RuntimeError(f"parallel stage layout contains duplicate entry: {key}.")
            if int(stage_idx) < 0 or int(stage_idx) >= int(self.residual_stages):
                raise RuntimeError(f"parallel stage layout stage index out of range: {key}.")
            if int(part_idx) < 0 or int(part_idx) >= int(self.parallel_parts):
                raise RuntimeError(f"parallel stage layout part index out of range: {key}.")
            seen.add(key)
            model_indices[int(stage_idx), int(part_idx)] = int(model_idx)
        if len(seen) != expected:
            raise RuntimeError(f"parallel stage layout covers {len(seen)} entries, expected {expected}.")

        grouped_vq = self._build_parallel_stage_grouped_vq_weight(layout)
        self._parallel_stage_grouped_vq_weight = grouped_vq
        self._parallel_stage_model_indices = model_indices
        self._parallel_stage_layout_is_stage_major = list(layout) == stage_major
        self._parallel_stage_restore_identity = self._parallel_stage_restore_is_identity()
        self._clear_parallel_stage_decode_runtime_cache()

    def _get_parallel_stage_grouped_vq(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        grouped_vq = getattr(self, "_parallel_stage_grouped_vq_weight", None)
        if grouped_vq is None:
            raise RuntimeError("parallel_stage_decode packed VQ buffer is missing.")
        target_device = torch.device(device)
        cache_key = (str(target_device), dtype)
        cached = getattr(self, "_parallel_stage_grouped_vq_runtime", None)
        if (
            isinstance(cached, torch.Tensor)
            and self._parallel_stage_grouped_vq_runtime_key == cache_key
            and cached.device == target_device
            and cached.dtype == dtype
        ):
            return cached
        out = grouped_vq.to(device=target_device, dtype=dtype, non_blocking=True)
        self._parallel_stage_grouped_vq_runtime = out
        self._parallel_stage_grouped_vq_runtime_key = cache_key
        return out

    def _get_parallel_stage_model_indices(self, device: torch.device) -> torch.Tensor:
        indices = getattr(self, "_parallel_stage_model_indices", None)
        if indices is None:
            raise RuntimeError("parallel_stage_decode model index plan is missing.")
        target_device = torch.device(device)
        cache_key = str(target_device)
        cached = getattr(self, "_parallel_stage_model_indices_runtime", None)
        if (
            isinstance(cached, torch.Tensor)
            and self._parallel_stage_model_indices_runtime_key == cache_key
            and cached.device == target_device
        ):
            return cached
        out = indices.to(device=target_device, non_blocking=True)
        self._parallel_stage_model_indices_runtime = out
        self._parallel_stage_model_indices_runtime_key = cache_key
        return out

    def _restore_index_to_device(
        self,
        restore_idx: Optional[torch.Tensor],
        *,
        cache_key: Tuple[Any, ...],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if restore_idx is None:
            return None
        target_device = torch.device(device)
        if restore_idx.device == target_device:
            return restore_idx
        full_key = tuple(cache_key) + (str(target_device),)
        cached = self._parallel_stage_restore_index_cache.get(full_key)
        if isinstance(cached, torch.Tensor) and cached.device == target_device:
            return cached
        moved = restore_idx.to(device=target_device, non_blocking=True)
        self._parallel_stage_restore_index_cache[full_key] = moved
        return moved

    def enable_trainable_decode(self, *, parallel_stage_decode: bool = False) -> None:
        self.trainable_decode = True
        self.cache_decoded_weight = False
        self.clear_decoded_weight_cache()
        self.parallel_stage_decode = bool(parallel_stage_decode)
        if self.parallel_stage_decode:
            self.pack_parallel_stage_decoder_(trainable=True)

    def disable_trainable_decode(self) -> None:
        self.trainable_decode = False
        self.parallel_stage_decode = False
        self.cache_decoded_weight = True
        self.clear_decoded_weight_cache()

    def pack_parallel_stage_decoder_(self, *, trainable: bool = False) -> bool:
        packed_decoder = getattr(self, "_parallel_stage_decoder", None)
        if packed_decoder is not None:
            packed_decoder.requires_grad_(bool(trainable))
            packed_decoder.train(self.training)
            self.parallel_stage_decode = True
            if getattr(self, "_parallel_stage_grouped_vq_weight", None) is None:
                self._build_parallel_stage_decode_plan()
            return True
        decoders, layout = self._iter_stage_part_decoders_for_pack()
        if len(decoders) <= 1:
            self.parallel_stage_decode = False
            return False
        stage_codebook_dims = [int(v) for v in getattr(self, "stage_codebook_dims", [])]
        if len(stage_codebook_dims) != int(self.residual_stages):
            raise ValueError(
                f"stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages={int(self.residual_stages)}"
            )
        if len(set(stage_codebook_dims)) != 1:
            raise ValueError(
                "parallel_stage_decode requires identical stage codebook dims, "
                f"got {stage_codebook_dims}."
            )
        packed_decoder = pack_decoders(decoders)
        packed_decoder.requires_grad_(bool(trainable))
        packed_decoder.train(self.training)
        self._parallel_stage_layout = list(layout)
        self._parallel_stage_decoder = packed_decoder
        self.parallel_stage_decode = True
        self._build_parallel_stage_decode_plan()

        if self._multi_parts:
            del self.decoders
            for stage_idx in range(1, int(self.residual_stages)):
                delattr(self, f"decoders_s{stage_idx}")
        else:
            del self.decoder
            for stage_idx in range(1, int(self.residual_stages)):
                delattr(self, f"decoder_s{stage_idx}")
        return True

    def _enable_parallel_stage_decoder(self) -> None:
        self.pack_parallel_stage_decoder_(trainable=True)

    def unpack_parallel_stage_decoder_(self) -> bool:
        packed_decoder = getattr(self, "_parallel_stage_decoder", None)
        if packed_decoder is None:
            return False
        layout = list(getattr(self, "_parallel_stage_layout", []))
        expected = int(self.residual_stages) * int(self.parallel_parts)
        if len(layout) != expected:
            raise RuntimeError(f"parallel stage layout length {len(layout)} != expected {expected}.")

        stage_parts: List[List[nn.Module]] = [
            [None for _part_idx in range(int(self.parallel_parts))]
            for _stage_idx in range(int(self.residual_stages))
        ]
        for model_idx, (stage_idx, part_idx) in enumerate(layout):
            stage_parts[int(stage_idx)][int(part_idx)] = packed_decoder.get_sub_decoder(int(model_idx))

        for stage_idx, parts in enumerate(stage_parts):
            if any(part is None for part in parts):
                raise RuntimeError(f"parallel stage layout is incomplete for stage={stage_idx}.")
            if stage_idx == 0:
                if self._multi_parts:
                    self.decoders = nn.ModuleList(parts)
                else:
                    self.decoder = parts[0]
            else:
                if self._multi_parts:
                    setattr(self, f"decoders_s{stage_idx}", nn.ModuleList(parts))
                else:
                    setattr(self, f"decoder_s{stage_idx}", parts[0])

        del self._parallel_stage_decoder
        self._clear_parallel_stage_decode_plan()
        self._parallel_stage_layout = []
        self.parallel_stage_decode = False
        return True

    def get_stage_restore_row_indices(self, stage_idx: int) -> Optional[torch.Tensor]:
        stage_idx = int(stage_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if stage_idx == 0:
            return getattr(self, "restore_row_indices", None)
        return getattr(self, f"restore_row_indices_s{stage_idx}", None)

    def get_stage_restore_col_indices(self, stage_idx: int) -> Optional[torch.Tensor]:
        stage_idx = int(stage_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if stage_idx == 0:
            return getattr(self, "restore_col_indices", None)
        return getattr(self, f"restore_col_indices_s{stage_idx}", None)

    def get_stage_part_restore_col_indices(self, stage_idx: int) -> Optional[torch.Tensor]:
        stage_idx = int(stage_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if stage_idx == 0:
            return getattr(self, "part_restore_col_indices", None)
        return getattr(self, f"part_restore_col_indices_s{stage_idx}", None)

    def _decode_single_flat(self, decoder: nn.Module, vq_weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # decoder expects [B, num_models=1, latent_dim]; output [B, 1, codebook_dim]
        # 为避免 matmul device/dtype 不一致，先对齐到 decoder 参数设备和 dtype，
        # 再在外层统一转回目标 dtype。
        param = next(decoder.parameters(), None)
        decode_device = param.device if param is not None else vq_weight.device
        decode_dtype = param.dtype if param is not None else dtype
        w_blocks = decoder(vq_weight.to(device=decode_device, dtype=decode_dtype, non_blocking=True))
        return w_blocks.permute(1, 0, 2).contiguous().view(-1)

    def _restore_split_row_order(self, w_split: torch.Tensor, *, stage_idx: int) -> torch.Tensor:
        # 排序代码，已关闭。原 row restore 解码分支保留如下：
        # restore_idx = self.get_stage_restore_row_indices(stage_idx)
        # if restore_idx is None:
        #     return w_split
        # if int(restore_idx.numel()) != int(w_split.shape[0]):
        #     raise ValueError(
        #         f"restore_row_indices size {int(restore_idx.numel())} != decoded split rows {int(w_split.shape[0])}"
        #     )
        # restore_idx = self._restore_index_to_device(
        #     restore_idx,
        #     cache_key=("restore_row", int(stage_idx)),
        #     device=w_split.device,
        # )
        # return w_split.index_select(0, restore_idx)
        return w_split

    def _restore_split_col_order(self, w_split: torch.Tensor, *, stage_idx: int) -> torch.Tensor:
        # 排序代码，已关闭。原 col restore 解码分支保留如下：
        # restore_idx = self.get_stage_restore_col_indices(stage_idx)
        # if restore_idx is None:
        #     return w_split
        # if int(restore_idx.numel()) != int(w_split.shape[1]):
        #     raise ValueError(
        #         f"restore_col_indices size {int(restore_idx.numel())} != decoded split cols {int(w_split.shape[1])}"
        #     )
        # restore_idx = self._restore_index_to_device(
        #     restore_idx,
        #     cache_key=("restore_col", int(stage_idx)),
        #     device=w_split.device,
        # )
        # return w_split.index_select(1, restore_idx)
        return w_split

    def _restore_part_col_order(self, part_matrix: torch.Tensor, part_idx: int, *, stage_idx: int) -> torch.Tensor:
        # 排序代码，已关闭。原 part col restore 解码分支保留如下：
        # restore_all = self.get_stage_part_restore_col_indices(stage_idx)
        # if restore_all is None:
        #     return part_matrix
        # if restore_all.ndim != 2:
        #     raise ValueError(
        #         f"part_restore_col_indices must be 2D, got shape={tuple(restore_all.shape)}"
        #     )
        # if part_idx < 0 or part_idx >= int(restore_all.shape[0]):
        #     raise IndexError(
        #         f"part_idx out of range for part_restore_col_indices: {part_idx} vs {int(restore_all.shape[0])}"
        #     )
        # restore_idx = restore_all[part_idx]
        # if int(restore_idx.numel()) != int(part_matrix.shape[1]):
        #     raise ValueError(
        #         f"part_restore_col_indices[{part_idx}] size {int(restore_idx.numel())} != part cols {int(part_matrix.shape[1])}"
        #     )
        # restore_idx = self._restore_index_to_device(
        #     restore_idx,
        #     cache_key=("part_restore_col", int(stage_idx), int(part_idx)),
        #     device=part_matrix.device,
        # )
        # return part_matrix.index_select(1, restore_idx)
        return part_matrix

    def _decode_part_flat(self, stage_idx: int, part_idx: int, dtype: torch.dtype) -> torch.Tensor:
        decoder = self.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
        vq_weight = self.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
        return self._decode_single_flat(decoder, vq_weight, dtype=dtype)

    def _expected_part_numel(self) -> int:
        total_numel = int(self.compressed_in_features) * int(self.compressed_out_features)
        if total_numel % int(self.parallel_parts) != 0:
            raise ValueError(
                f"compressed weight numel {total_numel} not divisible by parallel_parts={int(self.parallel_parts)}."
            )
        return total_numel // int(self.parallel_parts)

    def _restore_split_weight_from_part_flats(
        self,
        part_flats: torch.Tensor,
        dtype: torch.dtype,
        *,
        stage_idx: int = 0,
    ) -> torch.Tensor:
        split_rows = self.compressed_in_features if self.transpose else self.compressed_out_features
        split_cols = self.compressed_out_features if self.transpose else self.compressed_in_features
        part_flats = part_flats.reshape(int(self.parallel_parts), -1).contiguous()
        expected_part_numel = self._expected_part_numel()
        if int(part_flats.shape[1]) != expected_part_numel:
            raise ValueError(
                f"part flat width mismatch: got {int(part_flats.shape[1])}, expected {expected_part_numel}."
            )

        if not self._multi_parts:
            w_split = part_flats[0].view(split_rows, split_cols)
            w_split = self._restore_part_col_order(w_split, 0, stage_idx=stage_idx)
            w_split = self._restore_split_row_order(w_split, stage_idx=stage_idx)
            w_split = self._restore_split_col_order(w_split, stage_idx=stage_idx)
            return w_split.contiguous().to(dtype=dtype)

        rows_per_part = split_rows // self.parallel_rows
        cols_per_part = split_cols // self.parallel_cols
        expected_per_part = int(rows_per_part) * int(cols_per_part)
        if int(part_flats.shape[1]) != expected_per_part:
            raise ValueError(
                f"per-part flat width mismatch: got {int(part_flats.shape[1])}, expected {expected_per_part}."
            )
        parts = [
            self._restore_part_col_order(
                part_flats[part_idx].view(rows_per_part, cols_per_part),
                part_idx,
                stage_idx=stage_idx,
            )
            for part_idx in range(self.parallel_parts)
        ]

        row_blocks = []
        for row_idx in range(self.parallel_rows):
            start = row_idx * self.parallel_cols
            end = start + self.parallel_cols
            row_blocks.append(torch.cat(parts[start:end], dim=1))
        w_split = torch.cat(row_blocks, dim=0)
        w_split = self._restore_split_row_order(w_split, stage_idx=stage_idx)
        w_split = self._restore_split_col_order(w_split, stage_idx=stage_idx)
        return w_split.contiguous().to(dtype=dtype)

    def _decode_stage_split_weight(self, stage_idx: int, dtype: torch.dtype) -> torch.Tensor:
        decoded_parts = []
        for part_idx in range(self.parallel_parts):
            decoded_parts.append(self._decode_part_flat(stage_idx=stage_idx, part_idx=part_idx, dtype=dtype))
        stacked_parts = torch.stack(decoded_parts, dim=0)
        return self._restore_split_weight_from_part_flats(stacked_parts, dtype=dtype, stage_idx=stage_idx)

    def _restore_stage_part_flats_fast(self, stage_part_flats: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        expected_shape = (int(self.residual_stages), int(self.parallel_parts))
        if stage_part_flats.ndim != 3 or tuple(int(v) for v in stage_part_flats.shape[:2]) != expected_shape:
            raise ValueError(
                f"stage_part_flats must have leading shape {expected_shape}, got {tuple(stage_part_flats.shape)}."
            )

        split_rows = self.compressed_in_features if self.transpose else self.compressed_out_features
        split_cols = self.compressed_out_features if self.transpose else self.compressed_in_features
        if (
            bool(getattr(self, "_parallel_stage_restore_identity", False))
            and int(self.parallel_parts) == 1
        ):
            stage_splits = stage_part_flats[:, 0, :].contiguous().view(
                int(self.residual_stages),
                int(split_rows),
                int(split_cols),
            )
            return stage_splits.sum(dim=0).contiguous().to(dtype=dtype)

        restored_stages = [
            self._restore_split_weight_from_part_flats(
                stage_part_flats[stage_idx],
                dtype=dtype,
                stage_idx=stage_idx,
            )
            for stage_idx in range(int(self.residual_stages))
        ]
        if not restored_stages:
            raise RuntimeError("parallel stage decode produced no reconstruction.")
        if len(restored_stages) == 1:
            return restored_stages[0].contiguous()
        return torch.stack(restored_stages, dim=0).sum(dim=0).contiguous()

    def _decode_split_weight_parallel_stages(self, dtype: torch.dtype) -> torch.Tensor:
        packed_decoder = getattr(self, "_parallel_stage_decoder", None)
        if packed_decoder is None:
            raise RuntimeError("parallel_stage_decode is enabled but packed stage decoder is missing.")
        layout = list(getattr(self, "_parallel_stage_layout", []))
        expected_models = int(self.residual_stages) * int(self.parallel_parts)
        if len(layout) != expected_models:
            raise RuntimeError(f"parallel stage layout length {len(layout)} != expected {expected_models}.")

        param = next(packed_decoder.parameters(), None)
        decode_device = param.device if param is not None else torch.device("cpu")
        decode_dtype = param.dtype if param is not None else dtype
        grouped_vq = self._get_parallel_stage_grouped_vq(dtype=decode_dtype, device=decode_device)
        stage_out = packed_decoder(grouped_vq)
        if tuple(int(v) for v in stage_out.shape[:2]) != (int(grouped_vq.shape[0]), expected_models):
            raise RuntimeError(
                f"parallel stage decoder output shape mismatch: out={tuple(stage_out.shape)} "
                f"expected leading={(int(grouped_vq.shape[0]), expected_models)}."
            )
        model_flats = stage_out.permute(1, 0, 2).contiguous().view(expected_models, -1)
        if bool(getattr(self, "_parallel_stage_layout_is_stage_major", False)):
            stage_part_flats = model_flats.view(int(self.residual_stages), int(self.parallel_parts), -1)
        else:
            indices = self._get_parallel_stage_model_indices(model_flats.device).reshape(-1)
            stage_part_flats = model_flats.index_select(0, indices).view(
                int(self.residual_stages),
                int(self.parallel_parts),
                -1,
            )
        return self._restore_stage_part_flats_fast(stage_part_flats, dtype=dtype)

    def _decode_split_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if bool(getattr(self, "parallel_stage_decode", False)):
            return self._decode_split_weight_parallel_stages(dtype=dtype)
        split_weight = None
        for stage_idx in range(self.residual_stages):
            stage_split = self._decode_stage_split_weight(stage_idx=stage_idx, dtype=dtype)
            split_weight = stage_split if split_weight is None else (split_weight + stage_split)
        if split_weight is None:
            raise RuntimeError("no stage payload found in VAELinear.")
        return split_weight.contiguous()

    def _decode_compressed_weight_from_part_flats(
        self,
        part_flats: torch.Tensor,
        dtype: torch.dtype,
        *,
        stage_idx: int = 0,
    ) -> torch.Tensor:
        w_split = self._restore_split_weight_from_part_flats(part_flats, dtype=dtype, stage_idx=stage_idx)
        if self.transpose:
            return w_split.t().contiguous()
        return w_split.contiguous()

    def _decode_compressed_weight(self, dtype: torch.dtype) -> torch.Tensor:
        w_split = self._decode_split_weight(dtype=dtype)
        if self.transpose:
            return w_split.t().contiguous()
        return w_split.contiguous()

    def _materialize_full_weight(
        self,
        compressed_weight: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tuple(compressed_weight.shape) != (self.compressed_out_features, self.compressed_in_features):
            raise ValueError(
                f"decoded compressed weight shape {tuple(compressed_weight.shape)} != "
                f"({self.compressed_out_features}, {self.compressed_in_features})"
            )

        full_weight = compressed_weight.to(dtype=dtype)
        protected_out_idx = getattr(self, "protected_output_indices", None)
        if protected_out_idx is not None:
            if protected_out_idx.device != full_weight.device:
                protected_out_idx = protected_out_idx.to(device=full_weight.device, non_blocking=True)
            full_out = torch.empty(
                (self.out_features, self.compressed_in_features),
                dtype=dtype,
                device=full_weight.device,
            )
            protected_out_weight = getattr(self, "protected_output_weight", None)
            if protected_out_weight is None:
                raise RuntimeError("protected_output_weight is missing while protected_output_indices is set.")
            # When output protection is enabled, compressed_in_features should still match original in_features.
            if int(protected_out_weight.shape[1]) != int(self.compressed_in_features):
                raise RuntimeError(
                    "protected_output_weight shape is incompatible with compressed_in_features."
                )
            if protected_out_weight.device != full_weight.device or protected_out_weight.dtype != dtype:
                protected_out_weight = protected_out_weight.to(device=full_weight.device, dtype=dtype, non_blocking=True)
            full_out.index_copy_(0, protected_out_idx, protected_out_weight)
            keep_out_mask = torch.ones(self.out_features, dtype=torch.bool, device=full_weight.device)
            keep_out_mask[protected_out_idx] = False
            compressed_out_idx = torch.nonzero(keep_out_mask, as_tuple=False).reshape(-1)
            if int(compressed_out_idx.numel()) != int(full_weight.shape[0]):
                raise ValueError(
                    f"compressed output channel count mismatch: decoded={int(full_weight.shape[0])} "
                    f"vs expected={int(compressed_out_idx.numel())}"
                )
            full_out.index_copy_(0, compressed_out_idx, full_weight)
            full_weight = full_out

        protected_idx = getattr(self, "protected_input_indices", None)
        if protected_idx is None:
            if tuple(full_weight.shape) != (self.out_features, self.in_features):
                raise ValueError(
                    f"decoded full weight shape {tuple(full_weight.shape)} != "
                    f"({self.out_features}, {self.in_features})"
                )
            return full_weight

        if protected_idx.device != full_weight.device:
            protected_idx = protected_idx.to(device=full_weight.device, non_blocking=True)
        full_in = torch.empty(
            (full_weight.shape[0], self.in_features),
            dtype=dtype,
            device=full_weight.device,
        )
        protected_weight = getattr(self, "protected_input_weight", None)
        if protected_weight is None:
            raise RuntimeError("protected_input_weight is missing while protected_input_indices is set.")
        protected_weight = protected_weight.t().contiguous()
        if protected_weight.device != full_weight.device or protected_weight.dtype != dtype:
            protected_weight = protected_weight.to(device=full_weight.device, dtype=dtype, non_blocking=True)
        full_in.index_copy_(1, protected_idx, protected_weight)

        keep_mask = torch.ones(self.in_features, dtype=torch.bool, device=full_weight.device)
        keep_mask[protected_idx] = False
        compressed_idx = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
        if int(compressed_idx.numel()) != int(full_weight.shape[1]):
            raise ValueError(
                f"compressed input channel count mismatch: decoded={int(full_weight.shape[1])} "
                f"vs expected={int(compressed_idx.numel())}"
            )
        full_in.index_copy_(1, compressed_idx, full_weight)
        return full_in

    def _finalize_decoded_weight_from_compressed(
        self,
        compressed_weight: torch.Tensor,
        dtype: torch.dtype,
        *,
        include_low_rank: bool = True,
        include_sparse_residual: bool = True,
    ) -> torch.Tensor:
        full_weight = self._materialize_full_weight(
            compressed_weight,
            dtype=dtype,
        )
        if bool(include_low_rank):
            full_weight = self._apply_low_rank_patch(full_weight, dtype=dtype)
        if bool(include_sparse_residual):
            full_weight = self._apply_sparse_residual_patch(full_weight, dtype=dtype)
        return full_weight.contiguous()

    def _decode_weight(
        self,
        dtype: torch.dtype,
        *,
        include_low_rank: bool = True,
        include_sparse_residual: bool = True,
    ) -> torch.Tensor:
        compressed_weight = self._decode_compressed_weight(dtype=dtype)
        return self._finalize_decoded_weight_from_compressed(
            compressed_weight,
            dtype=dtype,
            include_low_rank=bool(include_low_rank),
            include_sparse_residual=bool(include_sparse_residual),
        )

    def _apply_low_rank_patch(self, full_weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        low_rank_a = getattr(self, "low_rank_a", None)
        low_rank_b = getattr(self, "low_rank_b", None)
        if low_rank_a is None and low_rank_b is None:
            return full_weight
        if low_rank_a is None or low_rank_b is None:
            raise RuntimeError("Low-rank payload is incomplete.")
        low_rank_a = low_rank_a.to(device=full_weight.device, dtype=dtype, non_blocking=True)
        low_rank_b = low_rank_b.to(device=full_weight.device, dtype=dtype, non_blocking=True)
        patch = low_rank_a @ low_rank_b
        if tuple(patch.shape) != (self.out_features, self.in_features):
            raise RuntimeError(
                f"low-rank patch shape {tuple(patch.shape)} != ({self.out_features}, {self.in_features})"
            )
        return full_weight.add(patch)

    def _decode_sparse_residual_patch(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        resolved_format = str(getattr(self, "sparse_residual_format", SPARSE_RESIDUAL_FORMAT_COO_FP16)).strip().lower()
        if resolved_format == SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED:
            active_block_ids = getattr(self, "sparse_residual_active_block_ids", None)
            if active_block_ids is None:
                return None
            return decode_blocked_quantized_sparse_residual(
                active_block_ids=active_block_ids,
                block_ptr=getattr(self, "sparse_residual_block_ptr", None),
                local_indices=getattr(self, "sparse_residual_local_indices", None),
                qvalues=getattr(self, "sparse_residual_qvalues", None),
                scales=getattr(self, "sparse_residual_scales", None),
                zero_points=getattr(self, "sparse_residual_zero_points", None),
                out_features=self.out_features,
                in_features=self.in_features,
                block_rows=int(self.sparse_residual_block_rows),
                block_cols=int(self.sparse_residual_block_cols),
                index_bits=int(self.sparse_residual_index_bits),
                value_bits=int(self.sparse_residual_value_bits),
                value_dtype=dtype,
                device=device,
            )

        row_idx = getattr(self, "sparse_residual_row_indices", None)
        if row_idx is None:
            return None
        col_idx = getattr(self, "sparse_residual_col_indices", None)
        values = getattr(self, "sparse_residual_values", None)
        if col_idx is None or values is None:
            raise RuntimeError("Sparse residual COO payload is incomplete.")
        row_idx = row_idx.to(device=device, dtype=torch.int64, non_blocking=True)
        col_idx = col_idx.to(device=device, dtype=torch.int64, non_blocking=True)
        values = values.to(device=device, dtype=dtype, non_blocking=True)
        return row_idx, col_idx, values

    def _get_sparse_residual_patch(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cached_row = getattr(self, "_cached_sparse_residual_row_indices", None)
        cached_col = getattr(self, "_cached_sparse_residual_col_indices", None)
        cached_values = getattr(self, "_cached_sparse_residual_values", None)
        if (
            isinstance(cached_row, torch.Tensor)
            and isinstance(cached_col, torch.Tensor)
            and isinstance(cached_values, torch.Tensor)
            and cached_row.device == device
            and cached_col.device == device
            and cached_values.device == device
            and cached_values.dtype == dtype
        ):
            return cached_row, cached_col, cached_values

        patch = self._decode_sparse_residual_patch(dtype=dtype, device=device)
        if patch is None:
            return None
        row_idx, col_idx, values = patch
        self._cached_sparse_residual_row_indices = row_idx.detach()
        self._cached_sparse_residual_col_indices = col_idx.detach()
        self._cached_sparse_residual_values = values.detach()
        return row_idx, col_idx, values

    def _apply_sparse_residual_patch(self, full_weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        patch = self._get_sparse_residual_patch(dtype=dtype, device=full_weight.device)
        if patch is None:
            return full_weight
        row_idx, col_idx, values = patch
        if int(row_idx.numel()) == 0:
            return full_weight
        full_weight.index_put_((row_idx, col_idx), values, accumulate=True)
        return full_weight

    def has_protected_outliers(self) -> bool:
        protected_input_weight = getattr(self, "protected_input_weight", None)
        if isinstance(protected_input_weight, torch.Tensor) and int(protected_input_weight.numel()) > 0:
            return True
        protected_output_weight = getattr(self, "protected_output_weight", None)
        return isinstance(protected_output_weight, torch.Tensor) and int(protected_output_weight.numel()) > 0

    def has_sparse_residual(self) -> bool:
        if str(getattr(self, "sparse_residual_format", SPARSE_RESIDUAL_FORMAT_COO_FP16)).strip().lower() == SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED:
            sparse_qvalues = getattr(self, "sparse_residual_qvalues", None)
            return isinstance(sparse_qvalues, torch.Tensor) and int(sparse_qvalues.numel()) > 0
        sparse_values = getattr(self, "sparse_residual_values", None)
        return isinstance(sparse_values, torch.Tensor) and int(sparse_values.numel()) > 0

    def has_low_rank_residual(self) -> bool:
        low_rank_a = getattr(self, "low_rank_a", None)
        low_rank_b = getattr(self, "low_rank_b", None)
        return (
            isinstance(low_rank_a, torch.Tensor)
            and isinstance(low_rank_b, torch.Tensor)
            and int(low_rank_a.numel()) > 0
            and int(low_rank_b.numel()) > 0
        )

    def has_original_linear(self) -> bool:
        return self.original_weight is not None

    def clear_sparse_residual_cache(self) -> None:
        self._cached_sparse_residual_row_indices = None
        self._cached_sparse_residual_col_indices = None
        self._cached_sparse_residual_values = None

    def clear_decoded_weight_cache(self) -> None:
        self._cached_weight = None
        self.clear_sparse_residual_cache()
        self._clear_parallel_stage_decode_runtime_cache()

    @torch.no_grad()
    def prime_sparse_residual_cache(
        self,
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        if not self.has_sparse_residual():
            self.clear_sparse_residual_cache()
            return False

        target_dtype = dtype
        target_device = None
        for param in self.parameters():
            target_device = param.device
            if target_dtype is None and param.is_floating_point():
                target_dtype = param.dtype
            break
        if target_device is None:
            for buffer in self.buffers():
                target_device = buffer.device
                if target_dtype is None and buffer.is_floating_point():
                    target_dtype = buffer.dtype
                break
        if target_device is None:
            target_device = torch.device("cpu")
        if target_dtype is None:
            target_dtype = torch.float32

        patch = self._get_sparse_residual_patch(dtype=target_dtype, device=torch.device(target_device))
        return patch is not None

    @torch.no_grad()
    def prime_decoded_weight_cache(
        self,
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        use_original = bool(getattr(self, "always_use_original", False)) or not bool(getattr(self, "temporary", True))
        if use_original:
            return False

        target_dtype = dtype
        if target_dtype is None:
            param = next(self.parameters(), None)
            target_dtype = param.dtype if (param is not None and param.is_floating_point()) else torch.float32

        w = self._decode_weight(dtype=target_dtype).detach()
        self._cached_weight = w
        return True

    def set_temporary(self, temporary: bool = True) -> None:  # 当 temporary=False 时走原始权重前向。
        if self.always_use_original:
            self.temporary = False
            return
        self.temporary = bool(temporary)

    def unload_original_linear(self) -> bool:
        if self.protect_original_weight:
            return False
        if self.original_weight is None:
            return False
        self.register_parameter("original_weight", None)
        self.temporary = not self.always_use_original
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_original = bool(getattr(self, "always_use_original", False)) or not bool(getattr(self, "temporary", True))
        if use_original:
            if self.original_weight is None:
                raise RuntimeError("VAELinear original_weight has been unloaded, cannot run original linear forward.")
            weight = self.original_weight if self.original_weight.dtype == x.dtype else self.original_weight.to(
                dtype=x.dtype)
            bias = self.bias
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(dtype=x.dtype)
            return F.linear(x, weight, bias)

        can_use_cache = bool(getattr(self, "cache_decoded_weight", True)) and not bool(
            getattr(self, "trainable_decode", False)
        )
        cached = self._cached_weight
        if (
            can_use_cache
            and cached is not None
            and cached.dtype == x.dtype
            and cached.device == x.device
        ):
            w = cached
        else:
            if can_use_cache and torch.is_grad_enabled():
                with torch.no_grad():
                    w = self._decode_weight(dtype=x.dtype)
            else:
                w = self._decode_weight(dtype=x.dtype)
            if can_use_cache:
                self._cached_weight = w.detach()

        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, w, bias)


from litebsq.vae_linear_prewarm import (  # noqa: E402
    NamedVAELinearTarget,
    clear_model_vae_linear_cache,
    prime_model_vae_linear_cache,
    prime_named_vae_linear_cache,
)

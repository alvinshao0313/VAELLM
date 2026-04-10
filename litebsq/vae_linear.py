import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence


class VAELinear(nn.Module):
    """
    Inference-only Linear replacement using (bit_indices + decoder) to reconstruct weights on the fly.

    `vq_weight` 可以是单个 Tensor（单分块）或 Tensor 列表（多分块）。
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
        decoder=None,
        stage_vq_weights: Optional[Sequence[Any]] = None,
        stage_decoders: Optional[Sequence[Any]] = None,
        codebook_dim: int,
        stage_codebook_dims: Optional[Sequence[int]] = None,
        transpose: bool,
        parallel_parts: int = 1,
        parallel_rows: Optional[int] = None,
        parallel_cols: Optional[int] = None,
        restore_row_indices: Optional[torch.Tensor] = None,
        restore_col_indices: Optional[torch.Tensor] = None,
        compressed_in_features: Optional[int] = None,
        compressed_out_features: Optional[int] = None,
        protected_input_indices: Optional[torch.Tensor] = None,
        protected_input_weight: Optional[torch.Tensor] = None,
        protected_output_indices: Optional[torch.Tensor] = None,
        protected_output_weight: Optional[torch.Tensor] = None,
        sparse_residual_row_indices: Optional[torch.Tensor] = None,
        sparse_residual_col_indices: Optional[torch.Tensor] = None,
        sparse_residual_values: Optional[torch.Tensor] = None,
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
        if restore_row_indices is None:
            self.register_buffer("restore_row_indices", None, persistent=True)
        else:
            restore_idx = restore_row_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            if restore_idx.ndim != 1:
                raise ValueError(
                    f"restore_row_indices must be 1D, got shape={tuple(restore_idx.shape)}"
                )
            if int(restore_idx.numel()) != int(split_rows):
                raise ValueError(
                    f"restore_row_indices size {int(restore_idx.numel())} != split_rows {int(split_rows)}"
                )
            self.register_buffer("restore_row_indices", restore_idx, persistent=True)
        if restore_col_indices is None:
            self.register_buffer("restore_col_indices", None, persistent=True)
        else:
            restore_col_idx = restore_col_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            if restore_col_idx.ndim != 1:
                raise ValueError(
                    f"restore_col_indices must be 1D, got shape={tuple(restore_col_idx.shape)}"
                )
            if int(restore_col_idx.numel()) != int(split_cols):
                raise ValueError(
                    f"restore_col_indices size {int(restore_col_idx.numel())} != split_cols {int(split_cols)}"
                )
            self.register_buffer("restore_col_indices", restore_col_idx, persistent=True)

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

        sparse_payload_provided = any(
            item is not None
            for item in (
                sparse_residual_row_indices,
                sparse_residual_col_indices,
                sparse_residual_values,
            )
        )
        if sparse_payload_provided:
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
        else:
            self.register_buffer("sparse_residual_row_indices", None, persistent=True)
            self.register_buffer("sparse_residual_col_indices", None, persistent=True)
            self.register_buffer("sparse_residual_values", None, persistent=True)

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
        self.register_buffer("_cached_weight", None, persistent=False)

        use_stage_payload = stage_vq_weights is not None or stage_decoders is not None
        if use_stage_payload:
            if stage_vq_weights is None or stage_decoders is None:
                raise ValueError("stage_vq_weights and stage_decoders must be provided together.")
            stage_vq_payload = self._normalize_stage_payload(
                stage_vq_weights,
                parallel_parts=self.parallel_parts,
                payload_name="stage_vq_weights",
            )
            stage_decoder_payload = self._normalize_stage_payload(
                stage_decoders,
                parallel_parts=self.parallel_parts,
                payload_name="stage_decoders",
            )
        else:
            if vq_weight is None or decoder is None:
                raise ValueError("vq_weight and decoder are required when stage payloads are not provided.")
            if self.parallel_parts == 1:
                stage_vq_payload = [[vq_weight]]
                stage_decoder_payload = [[decoder]]
            else:
                if not isinstance(vq_weight, (list, tuple)):
                    raise ValueError("multi-part mode requires vq_weight list/tuple.")
                if not isinstance(decoder, (list, tuple)):
                    raise ValueError("multi-part mode requires decoder list/tuple.")
                if len(vq_weight) != self.parallel_parts:
                    raise ValueError(
                        f"vq_weight length {len(vq_weight)} != parallel_parts {self.parallel_parts}"
                    )
                if len(decoder) != self.parallel_parts:
                    raise ValueError(
                        f"decoder length {len(decoder)} != parallel_parts {self.parallel_parts}"
                    )
                stage_vq_payload = [list(vq_weight)]
                stage_decoder_payload = [list(decoder)]

        if len(stage_vq_payload) != len(stage_decoder_payload):
            raise ValueError(
                f"stage payload length mismatch: "
                f"stage_vq_weights={len(stage_vq_payload)} vs stage_decoders={len(stage_decoder_payload)}"
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

        for stage_idx, stage_parts in enumerate(stage_vq_payload):
            for part_idx, w in enumerate(stage_parts):
                if not isinstance(w, torch.Tensor):
                    raise TypeError(
                        f"stage_vq_weights[{stage_idx}][{part_idx}] must be Tensor, got {type(w)}"
                    )
                if stage_idx == 0:
                    if self._multi_parts:
                        self.register_buffer(f"vq_weight_{part_idx}", w, persistent=True)
                    else:
                        self.register_buffer("vq_weight", w, persistent=True)
                else:
                    if self._multi_parts:
                        self.register_buffer(f"vq_weight_s{stage_idx}_{part_idx}", w, persistent=True)
                    else:
                        self.register_buffer(f"vq_weight_s{stage_idx}", w, persistent=True)

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
                w = self.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
                total_numel += int(w.shape[0]) * stage_codebook_dim
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

    def get_stage_part_vq_weight(self, stage_idx: int, part_idx: int = 0) -> torch.Tensor:
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

    def get_stage_part_decoder(self, stage_idx: int, part_idx: int = 0) -> nn.Module:
        stage_idx = int(stage_idx)
        part_idx = int(part_idx)
        if stage_idx < 0 or stage_idx >= self.residual_stages:
            raise IndexError(f"stage_idx out of range: {stage_idx} vs residual_stages={self.residual_stages}")
        if part_idx < 0 or part_idx >= self.parallel_parts:
            raise IndexError(f"part_idx out of range: {part_idx} vs parallel_parts={self.parallel_parts}")
        if stage_idx == 0:
            if self._multi_parts:
                return self.decoders[part_idx]
            return self.decoder
        if self._multi_parts:
            return getattr(self, f"decoders_s{stage_idx}")[part_idx]
        if part_idx != 0:
            raise IndexError("single-part VAELinear only supports part_idx=0")
        return getattr(self, f"decoder_s{stage_idx}")

    def _decode_single_flat(self, decoder: nn.Module, vq_weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # decoder expects [B, num_models=1, latent_dim]; output [B, 1, codebook_dim]
        # 为避免 matmul dtype 不一致，先对齐到 decoder 参数 dtype，再在外层统一转回目标 dtype。
        param = next(decoder.parameters(), None)
        decode_dtype = param.dtype if param is not None else dtype
        w_blocks = decoder(vq_weight.to(dtype=decode_dtype))
        return w_blocks.permute(1, 0, 2).contiguous().view(-1)

    def _restore_split_row_order(self, w_split: torch.Tensor) -> torch.Tensor:
        restore_idx = getattr(self, "restore_row_indices", None)
        if restore_idx is None:
            return w_split
        if int(restore_idx.numel()) != int(w_split.shape[0]):
            raise ValueError(
                f"restore_row_indices size {int(restore_idx.numel())} != decoded split rows {int(w_split.shape[0])}"
            )
        if restore_idx.device != w_split.device:
            restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
        return w_split.index_select(0, restore_idx)

    def _restore_split_col_order(self, w_split: torch.Tensor) -> torch.Tensor:
        restore_idx = getattr(self, "restore_col_indices", None)
        if restore_idx is None:
            return w_split
        if int(restore_idx.numel()) != int(w_split.shape[1]):
            raise ValueError(
                f"restore_col_indices size {int(restore_idx.numel())} != decoded split cols {int(w_split.shape[1])}"
            )
        if restore_idx.device != w_split.device:
            restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
        return w_split.index_select(1, restore_idx)

    def _decode_split_weight(self, dtype: torch.dtype) -> torch.Tensor:
        split_rows = self.compressed_in_features if self.transpose else self.compressed_out_features
        split_cols = self.compressed_out_features if self.transpose else self.compressed_in_features
        if not self._multi_parts:
            w_flat = None
            for stage_idx in range(self.residual_stages):
                decoder = self.get_stage_part_decoder(stage_idx=stage_idx, part_idx=0)
                vq_weight = self.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=0)
                stage_flat = self._decode_single_flat(decoder, vq_weight, dtype=dtype)
                w_flat = stage_flat if w_flat is None else (w_flat + stage_flat)
            if w_flat is None:
                raise RuntimeError("no stage payload found in VAELinear.")
            w_split = w_flat.view(split_rows, split_cols)
            w_split = self._restore_split_row_order(w_split)
            w_split = self._restore_split_col_order(w_split)
            return w_split.contiguous().to(dtype=dtype)

        rows_per_part = split_rows // self.parallel_rows
        cols_per_part = split_cols // self.parallel_cols
        parts = []
        for part_idx in range(self.parallel_parts):
            part_flat = None
            for stage_idx in range(self.residual_stages):
                decoder = self.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                vq_weight = self.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
                stage_flat = self._decode_single_flat(decoder, vq_weight, dtype=dtype)
                part_flat = stage_flat if part_flat is None else (part_flat + stage_flat)
            if part_flat is None:
                raise RuntimeError("no stage payload found in VAELinear.")
            parts.append(part_flat.view(rows_per_part, cols_per_part))

        row_blocks = []
        for row_idx in range(self.parallel_rows):
            start = row_idx * self.parallel_cols
            end = start + self.parallel_cols
            row_blocks.append(torch.cat(parts[start:end], dim=1))
        w_split = torch.cat(row_blocks, dim=0)
        w_split = self._restore_split_row_order(w_split)
        w_split = self._restore_split_col_order(w_split)
        return w_split.contiguous().to(dtype=dtype)

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

    def _decode_weight(self, dtype: torch.dtype) -> torch.Tensor:
        compressed_weight = self._decode_compressed_weight(dtype=dtype)
        full_weight = self._materialize_full_weight(
            compressed_weight,
            dtype=dtype,
        )
        full_weight = self._apply_sparse_residual_patch(full_weight, dtype=dtype)
        return full_weight.contiguous()

    def _apply_sparse_residual_patch(self, full_weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        row_idx = getattr(self, "sparse_residual_row_indices", None)
        if row_idx is None:
            return full_weight
        col_idx = getattr(self, "sparse_residual_col_indices", None)
        values = getattr(self, "sparse_residual_values", None)
        if col_idx is None or values is None:
            raise RuntimeError("Sparse residual COO payload is incomplete.")
        if int(row_idx.numel()) == 0:
            return full_weight
        row_idx = row_idx.to(device=full_weight.device, dtype=torch.int64, non_blocking=True)
        col_idx = col_idx.to(device=full_weight.device, dtype=torch.int64, non_blocking=True)
        values = values.to(device=full_weight.device, dtype=dtype, non_blocking=True)
        full_weight.index_put_((row_idx, col_idx), values, accumulate=True)
        return full_weight

    def has_protected_outliers(self) -> bool:
        protected_input_weight = getattr(self, "protected_input_weight", None)
        if isinstance(protected_input_weight, torch.Tensor) and int(protected_input_weight.numel()) > 0:
            return True
        protected_output_weight = getattr(self, "protected_output_weight", None)
        return isinstance(protected_output_weight, torch.Tensor) and int(protected_output_weight.numel()) > 0

    def has_sparse_residual(self) -> bool:
        sparse_values = getattr(self, "sparse_residual_values", None)
        return isinstance(sparse_values, torch.Tensor) and int(sparse_values.numel()) > 0

    def has_original_linear(self) -> bool:
        return self.original_weight is not None

    def clear_decoded_weight_cache(self) -> None:
        self._cached_weight = None

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

        can_use_cache = bool(getattr(self, "cache_decoded_weight", True))
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


def clear_model_vae_linear_cache(model: nn.Module) -> int:
    cleared = 0
    for module in model.modules():
        if isinstance(module, VAELinear):
            module.clear_decoded_weight_cache()
            cleared += 1
    return cleared


@torch.no_grad()
def prime_model_vae_linear_cache(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    clear_existing: bool = False,
) -> Dict[str, int]:
    total = 0
    warmed = 0
    skipped = 0
    failed = 0

    for module in model.modules():
        if not isinstance(module, VAELinear):
            continue
        if bool(getattr(module, "_skip_global_cache_prewarm", False)):
            skipped += 1
            continue
        total += 1
        if clear_existing:
            module.clear_decoded_weight_cache()
        try:
            if module.prime_decoded_weight_cache(dtype=dtype):
                warmed += 1
            else:
                skipped += 1
        except Exception:
            failed += 1

    return {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(skipped),
        "failed": int(failed),
    }

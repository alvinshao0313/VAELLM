import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional


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
        vq_weight,
        decoder,
        codebook_dim: int,
        transpose: bool,
        parallel_parts: int = 1,
        parallel_rows: Optional[int] = None,
        parallel_cols: Optional[int] = None,
        restore_row_indices: Optional[torch.Tensor] = None,
        restore_col_indices: Optional[torch.Tensor] = None,
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

        split_rows = self.in_features if self.transpose else self.out_features
        split_cols = self.out_features if self.transpose else self.in_features
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

        if isinstance(vq_weight, (list, tuple)):
            if len(vq_weight) != self.parallel_parts:
                raise ValueError(
                    f"vq_weight length {len(vq_weight)} != parallel_parts {self.parallel_parts}"
                )
            self._multi_parts = True
            for idx, w in enumerate(vq_weight):
                self.register_buffer(f"vq_weight_{idx}", w, persistent=True)
        else:
            if self.parallel_parts != 1:
                raise ValueError("single vq_weight requires parallel_parts=1")
            self._multi_parts = False
            self.register_buffer("vq_weight", vq_weight, persistent=True)

        if isinstance(decoder, (list, tuple)):
            if len(decoder) != self.parallel_parts:
                raise ValueError(
                    f"decoder length {len(decoder)} != parallel_parts {self.parallel_parts}"
                )
            self.decoders = nn.ModuleList(decoder)
        else:
            if self.parallel_parts != 1:
                raise ValueError("single decoder requires parallel_parts=1")
            self.decoder = decoder

        expected_numel = self.in_features * self.out_features
        if self._multi_parts:
            total_numel = 0
            for idx in range(self.parallel_parts):
                w = getattr(self, f"vq_weight_{idx}")
                total_numel += int(w.shape[0]) * self.codebook_dim
            if total_numel != expected_numel:
                raise ValueError(
                    f"vq_weight total mismatch: total={total_numel} expected={expected_numel} (in*out)"
                )
        else:
            if (self.vq_weight.shape[0] * self.codebook_dim) != expected_numel:
                raise ValueError(
                    f"vq_weight blocks mismatch: blocks={self.vq_weight.shape[0]} codebook_dim={self.codebook_dim} "
                    f"-> {self.vq_weight.shape[0] * self.codebook_dim} != {expected_numel} (in*out)"
                )

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

    def _decode_weight(self, dtype: torch.dtype) -> torch.Tensor:
        split_rows = self.in_features if self.transpose else self.out_features
        split_cols = self.out_features if self.transpose else self.in_features
        if not self._multi_parts:
            w_flat = self._decode_single_flat(self.decoder, self.vq_weight, dtype=dtype)
            w_split = w_flat.view(split_rows, split_cols)
            w_split = self._restore_split_row_order(w_split)
            w_split = self._restore_split_col_order(w_split)
            if self.transpose:
                return w_split.t().contiguous().to(dtype=dtype)
            return w_split.contiguous().to(dtype=dtype)

        rows_per_part = split_rows // self.parallel_rows
        cols_per_part = split_cols // self.parallel_cols
        parts = []
        for idx, decoder in enumerate(self.decoders):
            vq_weight = getattr(self, f"vq_weight_{idx}")
            part_flat = self._decode_single_flat(decoder, vq_weight, dtype=dtype)
            parts.append(part_flat.view(rows_per_part, cols_per_part))

        row_blocks = []
        for row_idx in range(self.parallel_rows):
            start = row_idx * self.parallel_cols
            end = start + self.parallel_cols
            row_blocks.append(torch.cat(parts[start:end], dim=1))
        w_split = torch.cat(row_blocks, dim=0)
        w_split = self._restore_split_row_order(w_split)
        w_split = self._restore_split_col_order(w_split)
        if self.transpose:
            return w_split.t().contiguous().to(dtype=dtype)
        return w_split.contiguous().to(dtype=dtype)

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

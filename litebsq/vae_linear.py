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

    def _decode_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if not self._multi_parts:
            w_flat = self._decode_single_flat(self.decoder, self.vq_weight, dtype=dtype)
            if self.transpose:
                return w_flat.view(self.in_features, self.out_features).t().contiguous().to(dtype=dtype)
            return w_flat.view(self.out_features, self.in_features).contiguous().to(dtype=dtype)

        if self.transpose:
            if self.in_features % self.parallel_parts != 0:
                raise ValueError(
                    f"in_features={self.in_features} not divisible by parallel_parts={self.parallel_parts}"
                )
            rows_per_part = self.in_features // self.parallel_parts
            parts = []
            for idx, decoder in enumerate(self.decoders):
                vq_weight = getattr(self, f"vq_weight_{idx}")
                part_flat = self._decode_single_flat(decoder, vq_weight, dtype=dtype)
                parts.append(part_flat.view(rows_per_part, self.out_features))
            w_t = torch.cat(parts, dim=0)
            return w_t.t().contiguous().to(dtype=dtype)

        if self.out_features % self.parallel_parts != 0:
            raise ValueError(
                f"out_features={self.out_features} not divisible by parallel_parts={self.parallel_parts}"
            )
        rows_per_part = self.out_features // self.parallel_parts
        parts = []
        for idx, decoder in enumerate(self.decoders):
            vq_weight = getattr(self, f"vq_weight_{idx}")
            part_flat = self._decode_single_flat(decoder, vq_weight, dtype=dtype)
            parts.append(part_flat.view(rows_per_part, self.in_features))
        return torch.cat(parts, dim=0).contiguous().to(dtype=dtype)

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

from typing import Optional, Tuple

import torch


PROTECTED_CHANNEL_QUANT_NONE = "none"
PROTECTED_CHANNEL_QUANT_FP8_E4M3 = "fp8_e4m3"
PROTECTED_CHANNEL_QUANT_FP8_E5M2 = "fp8_e5m2"
PROTECTED_CHANNEL_QUANT_INT8 = "int8"
PROTECTED_CHANNEL_QUANT_CHOICES = (
    PROTECTED_CHANNEL_QUANT_NONE,
    PROTECTED_CHANNEL_QUANT_FP8_E4M3,
    PROTECTED_CHANNEL_QUANT_FP8_E5M2,
    PROTECTED_CHANNEL_QUANT_INT8,
)


def normalize_protected_channel_quant_format(value: object, *, arg_name: str = "protected_channel_quant_format") -> str:
    resolved = str(value).strip().lower()
    if resolved not in PROTECTED_CHANNEL_QUANT_CHOICES:
        raise ValueError(
            f"Unsupported {arg_name}={value!r}. Expected one of: {', '.join(PROTECTED_CHANNEL_QUANT_CHOICES)}."
        )
    return resolved


def _resolve_fp8_dtype(quant_format: str) -> torch.dtype:
    if quant_format == PROTECTED_CHANNEL_QUANT_FP8_E4M3:
        return torch.float8_e4m3fn
    if quant_format == PROTECTED_CHANNEL_QUANT_FP8_E5M2:
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 protected channel quant format: {quant_format!r}.")


def _resolve_quant_max(quant_format: str) -> float:
    if quant_format == PROTECTED_CHANNEL_QUANT_INT8:
        return 127.0
    fp8_dtype = _resolve_fp8_dtype(quant_format)
    return float(torch.finfo(fp8_dtype).max)


def _per_channel_scales(weight_f32: torch.Tensor, quant_format: str) -> torch.Tensor:
    if weight_f32.ndim != 2:
        raise ValueError(f"protected channel weight must be 2D, got shape={tuple(weight_f32.shape)}.")
    max_abs = weight_f32.abs().amax(dim=1)
    quant_max = _resolve_quant_max(quant_format)
    scales = torch.where(max_abs > 0, max_abs / quant_max, torch.ones_like(max_abs))
    return scales.to(dtype=torch.bfloat16).contiguous()


def encode_protected_channel_weight(
    weight: torch.Tensor,
    *,
    quant_format: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    resolved = normalize_protected_channel_quant_format(quant_format)
    if resolved == PROTECTED_CHANNEL_QUANT_NONE:
        raise ValueError("encode_protected_channel_weight does not accept quant_format='none'.")
    weight_f32 = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if int(weight_f32.numel()) == 0:
        raise ValueError("protected channel weight must be non-empty for quantization.")
    scales_bf16 = _per_channel_scales(weight_f32, resolved)
    scales_f32 = scales_bf16.to(dtype=torch.float32)
    normalized = weight_f32 / scales_f32.unsqueeze(1)
    if resolved == PROTECTED_CHANNEL_QUANT_INT8:
        q = torch.round(normalized).clamp_(-128.0, 127.0).to(dtype=torch.int8)
        qvalues = q.view(torch.uint8).contiguous()
    else:
        fp8_dtype = _resolve_fp8_dtype(resolved)
        qvalues = normalized.to(dtype=fp8_dtype).view(torch.uint8).contiguous()
    return qvalues, scales_bf16


def decode_protected_channel_weight(
    qvalues: torch.Tensor,
    scales: torch.Tensor,
    *,
    quant_format: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    resolved = normalize_protected_channel_quant_format(quant_format)
    if resolved == PROTECTED_CHANNEL_QUANT_NONE:
        raise ValueError("decode_protected_channel_weight does not accept quant_format='none'.")
    target_device = torch.device("cpu") if device is None else torch.device(device)
    q_cpu = qvalues.detach().to(device="cpu").contiguous()
    scale_cpu = scales.detach().to(device="cpu", dtype=torch.bfloat16).reshape(-1).contiguous()
    if q_cpu.ndim != 2:
        raise ValueError(f"protected channel qvalues must be 2D, got shape={tuple(q_cpu.shape)}.")
    if scale_cpu.ndim != 1:
        raise ValueError(f"protected channel scales must be 1D, got shape={tuple(scale_cpu.shape)}.")
    if int(scale_cpu.numel()) != int(q_cpu.shape[0]):
        raise ValueError(
            f"protected channel scale count mismatch: scales={int(scale_cpu.numel())} "
            f"vs qvalues_rows={int(q_cpu.shape[0])}."
        )
    scale_f32 = scale_cpu.to(dtype=torch.float32)
    if resolved == PROTECTED_CHANNEL_QUANT_INT8:
        q_signed = q_cpu.view(torch.int8).to(dtype=torch.float32)
        decoded = q_signed * scale_f32.unsqueeze(1)
    else:
        fp8_dtype = _resolve_fp8_dtype(resolved)
        q_fp8 = q_cpu.view(fp8_dtype).to(dtype=torch.float32)
        decoded = q_fp8 * scale_f32.unsqueeze(1)
    return decoded.to(device=target_device, dtype=dtype, non_blocking=True).contiguous()

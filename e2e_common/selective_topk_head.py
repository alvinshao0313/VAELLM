from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from e2e_common.post_norm_head import LMHeadWithPostNormLinear


@dataclass
class TeacherTopKTargets:
    indices_cpu: torch.Tensor
    logits_cpu: torch.Tensor
    k: int
    transfer_chunk_size: Optional[int] = None

    def clear(self) -> None:
        self.indices_cpu = torch.empty(0, dtype=torch.long)
        self.logits_cpu = torch.empty(0, dtype=torch.float32)


def is_selective_student_topk_loss(loss_type: str) -> bool:
    """True only for canonical kl_top (or temporary legacy kl_top_<K> strings)."""
    norm = str(loss_type or "").strip().lower()
    return norm == "kl_top" or norm.startswith("kl_top_")


def parse_selective_student_topk_k(loss_type: str, *, top_k: int) -> int:
    """Resolve selective K from DistillLossConfig.top_k (single truth).

    ``top_k`` is required; wrappers must not guess 1000/100. Legacy ``kl_top_<K>``
    strings are accepted only when callers have not yet split them; the suffix K
    must match the provided ``top_k`` if both are present.
    """
    resolved = int(top_k)
    if resolved < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    norm = str(loss_type or "").strip().lower()
    if norm == "kl_top":
        return resolved
    if not norm.startswith("kl_top_"):
        raise ValueError(f"Selective student top-k only supports kl_top[_K], got {loss_type!r}.")
    suffix = norm[len("kl_top_") :]
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(f"Invalid kl_top suffix in {loss_type!r}.")
    suffix_k = int(suffix)
    if suffix_k != resolved:
        raise ValueError(
            f"Selective student top-k mismatch: loss_type={loss_type!r} encodes K={suffix_k} "
            f"but DistillLossConfig.top_k={resolved}."
        )
    return resolved


@torch.no_grad()
def extract_teacher_topk_targets(
    logits: torch.Tensor,
    *,
    k: int,
    sequence_chunk_size: int,
    pin_memory: bool,
) -> TeacherTopKTargets:
    if logits.ndim != 3:
        raise ValueError(f"teacher logits must have shape [B,L,V], got {tuple(logits.shape)}.")
    resolved_k = min(int(k), int(logits.shape[-1]))
    if resolved_k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    chunk_size = int(sequence_chunk_size)
    if chunk_size < 1:
        raise ValueError(f"sequence_chunk_size must be >= 1, got {sequence_chunk_size}.")

    batch, seq_len, _vocab = (int(v) for v in logits.shape)
    use_pinned = bool(pin_memory and torch.cuda.is_available())
    values_cpu = torch.empty(
        (batch, seq_len, resolved_k),
        dtype=torch.float32,
        device="cpu",
        pin_memory=use_pinned,
    )
    indices_cpu = torch.empty(
        (batch, seq_len, resolved_k),
        dtype=torch.long,
        device="cpu",
        pin_memory=use_pinned,
    )

    async_copy = bool(use_pinned and logits.device.type == "cuda")
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        active = logits[:, start:end, :].detach().to(dtype=torch.float32)
        values, indices = torch.topk(active, k=resolved_k, dim=-1, sorted=False)
        values_cpu[:, start:end, :].copy_(values, non_blocking=async_copy)
        indices_cpu[:, start:end, :].copy_(indices, non_blocking=async_copy)
        del active, values, indices
    if async_copy:
        torch.cuda.synchronize(logits.device)

    return TeacherTopKTargets(
        indices_cpu=indices_cpu,
        logits_cpu=values_cpu,
        k=resolved_k,
    )


def move_teacher_topk_targets_to_device(
    targets: TeacherTopKTargets,
    *,
    device: torch.device,
    sequence_chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = torch.device(device)
    if sequence_chunk_size is None:
        sequence_chunk_size = targets.transfer_chunk_size
    if sequence_chunk_size is None or target_device.type != "cuda":
        indices = targets.indices_cpu.to(
            device=target_device,
            dtype=torch.long,
            non_blocking=bool(targets.indices_cpu.is_pinned() and target_device.type == "cuda"),
        )
        logits = targets.logits_cpu.to(
            device=target_device,
            dtype=torch.float32,
            non_blocking=bool(targets.logits_cpu.is_pinned() and target_device.type == "cuda"),
        )
        return indices, logits

    chunk_size = int(sequence_chunk_size)
    if chunk_size < 1:
        raise ValueError(f"sequence_chunk_size must be >= 1, got {sequence_chunk_size}.")
    if targets.indices_cpu.ndim != 3 or targets.logits_cpu.ndim != 3:
        raise ValueError("teacher top-k targets must have shape [B,L,K].")
    if tuple(targets.indices_cpu.shape) != tuple(targets.logits_cpu.shape):
        raise ValueError("teacher top-k indices/logits shape mismatch.")

    indices = torch.empty(
        targets.indices_cpu.shape,
        dtype=torch.long,
        device=target_device,
    )
    logits = torch.empty(
        targets.logits_cpu.shape,
        dtype=torch.float32,
        device=target_device,
    )
    non_blocking_indices = bool(targets.indices_cpu.is_pinned())
    non_blocking_logits = bool(targets.logits_cpu.is_pinned())
    seq_len = int(targets.indices_cpu.shape[1])
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        indices[:, start:end, :].copy_(
            targets.indices_cpu[:, start:end, :],
            non_blocking=non_blocking_indices,
        )
        logits[:, start:end, :].copy_(
            targets.logits_cpu[:, start:end, :],
            non_blocking=non_blocking_logits,
        )
    return indices, logits


def compute_selected_teacher_topk_kl(
    *,
    student_selected_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    if tuple(student_selected_logits.shape) != tuple(teacher_topk_logits.shape):
        raise ValueError(
            "selected student/teacher top-k shape mismatch: "
            f"{tuple(student_selected_logits.shape)} vs {tuple(teacher_topk_logits.shape)}."
        )
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    student_log_prob = F.log_softmax(student_selected_logits.float() / temp, dim=-1)
    teacher_prob = F.softmax(teacher_topk_logits.detach().float() / temp, dim=-1)
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    if mask is None:
        mask_fp = torch.ones_like(token_kl, dtype=torch.float32)
    else:
        mask_fp = mask.to(device=token_kl.device, dtype=torch.float32)
    denom = mask_fp.sum().clamp_min(1.0)
    return ((token_kl * mask_fp).sum() / denom) * (temp * temp)


class _FrozenSelectiveLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        chunk_rows: int,
    ) -> torch.Tensor:
        if hidden.shape[:-1] != indices.shape[:-1]:
            raise ValueError(
                f"hidden/indices leading shape mismatch: {tuple(hidden.shape)} vs {tuple(indices.shape)}."
            )
        if weight.ndim != 2:
            raise ValueError(f"lm_head weight must be 2D, got {tuple(weight.shape)}.")
        if indices.dtype != torch.long:
            raise ValueError(f"teacher top-k indices must be torch.long, got {indices.dtype}.")
        if indices.device != hidden.device:
            raise ValueError(f"indices device {indices.device} != hidden device {hidden.device}.")
        if weight.device != hidden.device:
            raise ValueError(f"lm_head device {weight.device} != hidden device {hidden.device}.")
        if weight.dtype != hidden.dtype:
            raise ValueError(f"lm_head dtype {weight.dtype} != hidden dtype {hidden.dtype}.")
        if int(hidden.shape[-1]) != int(weight.shape[-1]):
            raise ValueError(
                f"hidden dim {int(hidden.shape[-1])} != lm_head in_features {int(weight.shape[-1])}."
            )
        resolved_rows = int(chunk_rows)
        if resolved_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}.")
        if weight.requires_grad or (bias.numel() > 0 and bias.requires_grad):
            raise ValueError("selective student top-k requires frozen base lm_head weight/bias.")

        hidden_2d = hidden.reshape(-1, int(hidden.shape[-1]))
        indices_2d = indices.reshape(-1, int(indices.shape[-1]))
        rows = int(hidden_2d.shape[0])
        k = int(indices_2d.shape[1])
        out = torch.empty((rows, k), device=hidden.device, dtype=hidden.dtype)

        for start in range(0, rows, resolved_rows):
            end = min(rows, start + resolved_rows)
            idx = indices_2d[start:end]
            selected_weight = F.embedding(idx, weight)
            active_hidden = hidden_2d[start:end].unsqueeze(-1)
            active_out = torch.bmm(selected_weight, active_hidden).squeeze(-1)
            if bias.numel() > 0:
                active_out = active_out + bias.index_select(0, idx.reshape(-1)).view_as(active_out)
            out[start:end].copy_(active_out)
            del selected_weight, active_hidden, active_out

        ctx.save_for_backward(indices, weight)
        ctx.hidden_shape = tuple(int(v) for v in hidden.shape)
        ctx.chunk_rows = resolved_rows
        return out.view(*indices.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, weight = ctx.saved_tensors
        hidden_shape = tuple(int(v) for v in ctx.hidden_shape)
        hidden_dim = int(hidden_shape[-1])
        indices_2d = indices.reshape(-1, int(indices.shape[-1]))
        grad_2d = grad_output.to(dtype=weight.dtype).reshape(-1, int(grad_output.shape[-1]))
        rows = int(indices_2d.shape[0])
        grad_hidden = torch.empty((rows, hidden_dim), device=grad_output.device, dtype=weight.dtype)

        for start in range(0, rows, int(ctx.chunk_rows)):
            end = min(rows, start + int(ctx.chunk_rows))
            idx = indices_2d[start:end]
            selected_weight = F.embedding(idx, weight)
            active_grad = grad_2d[start:end].unsqueeze(-1)
            grad_chunk = torch.bmm(selected_weight.transpose(1, 2), active_grad).squeeze(-1)
            grad_hidden[start:end].copy_(grad_chunk)
            del selected_weight, active_grad, grad_chunk

        return grad_hidden.view(*hidden_shape), None, None, None, None


def _selective_linear(
    hidden: torch.Tensor,
    *,
    indices: torch.Tensor,
    linear: nn.Linear,
    chunk_rows: int,
) -> torch.Tensor:
    bias = linear.bias
    bias_tensor = (
        bias
        if isinstance(bias, torch.Tensor)
        else torch.empty(0, device=linear.weight.device, dtype=linear.weight.dtype)
    )
    return _FrozenSelectiveLinear.apply(
        hidden,
        indices,
        linear.weight,
        bias_tensor,
        int(chunk_rows),
    )


def _resolve_lm_head(model: nn.Module) -> nn.Module:
    current = model
    seen = set()
    for _depth in range(8):
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)

        lm_head = getattr(current, "lm_head", None)
        if isinstance(lm_head, nn.Module):
            return lm_head

        module = getattr(current, "module", None)
        if isinstance(module, nn.Module) and module is not current:
            current = module
            continue

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            base = get_base_model()
            if isinstance(base, nn.Module) and base is not current:
                current = base
                continue
        break
    raise ValueError("Could not resolve model.lm_head for selective student top-k.")


@contextmanager
def selective_student_lm_head(
    model: nn.Module,
    *,
    teacher_topk_indices: torch.Tensor,
    chunk_rows: int,
) -> Iterator[None]:
    head = _resolve_lm_head(model)
    had_instance_forward = "forward" in head.__dict__
    original_instance_forward = head.__dict__.get("forward")
    cached_indices = teacher_topk_indices

    if isinstance(head, LMHeadWithPostNormLinear):
        base_linear = head.lm_head
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                "LMHeadWithPostNormLinear.lm_head must be nn.Linear for selective student top-k."
            )

        def selective_forward(hidden_states: torch.Tensor):
            nonlocal cached_indices
            if cached_indices.device != hidden_states.device:
                cached_indices = cached_indices.to(device=hidden_states.device, non_blocking=True)
            transformed = head.post_norm_linear(hidden_states)
            return _selective_linear(
                transformed,
                indices=cached_indices,
                linear=base_linear,
                chunk_rows=int(chunk_rows),
            )

    elif isinstance(head, nn.Linear):

        def selective_forward(hidden_states: torch.Tensor):
            nonlocal cached_indices
            if cached_indices.device != hidden_states.device:
                cached_indices = cached_indices.to(device=hidden_states.device, non_blocking=True)
            return _selective_linear(
                hidden_states,
                indices=cached_indices,
                linear=head,
                chunk_rows=int(chunk_rows),
            )

    else:
        raise TypeError(
            "selective student top-k currently supports nn.Linear or "
            f"LMHeadWithPostNormLinear, got {type(head)}."
        )

    head.forward = selective_forward
    try:
        yield
    finally:
        if had_instance_forward:
            head.forward = original_instance_forward
        else:
            delattr(head, "forward")

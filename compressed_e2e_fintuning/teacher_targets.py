from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch
from torch import nn

from compressed_e2e_fintuning.offload import OffloadedCheckpointLayer
from rotation.model_utils import get_layers


@dataclass
class TeacherTargetBatch:
    logits_cpu: Optional[torch.Tensor] = None
    eakld_gamma_cpu: Optional[torch.Tensor] = None
    hidden_cpu_by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    hidden_layer_indices: Tuple[int, ...] = ()
    num_hidden_layers: int = 0

    def clear(self) -> None:
        self.logits_cpu = None
        self.eakld_gamma_cpu = None
        self.hidden_cpu_by_layer.clear()
        self.hidden_layer_indices = ()
        self.num_hidden_layers = 0


def resolve_transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    current = model
    seen: set[int] = set()

    for _depth in range(8):
        identity = id(current)
        if identity in seen:
            raise RuntimeError("Detected a cycle while unwrapping the model for hidden hooks.")
        seen.add(identity)

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
    else:
        raise RuntimeError("Exceeded 8 model-wrapper levels while resolving Transformer layers.")

    layers = get_layers(current)
    if len(layers) < 1:
        raise ValueError("Resolved Transformer layer list is empty.")

    hook_layers = []
    for layer in layers:
        if isinstance(layer, OffloadedCheckpointLayer):
            hook_layers.append(layer.layer)
        else:
            hook_layers.append(layer)
    return tuple(hook_layers)


def copy_detached_tensor_to_cpu(
    tensor: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a torch.Tensor.")

    detached = tensor.detach()
    if detached.device.type == "cpu":
        copied = detached.clone()
        if bool(pin_memory) and torch.cuda.is_available() and not copied.is_pinned():
            copied = copied.pin_memory()
        return copied

    cpu_tensor = torch.empty(
        tuple(detached.shape),
        dtype=detached.dtype,
        device="cpu",
        pin_memory=bool(pin_memory),
    )
    cpu_tensor.copy_(detached, non_blocking=False)
    return cpu_tensor


def iter_token_chunk_ranges(
    sequence_length: int,
    chunk_tokens: int,
) -> Iterator[tuple[int, int]]:
    resolved_length = int(sequence_length)
    resolved_chunk = int(chunk_tokens)
    if resolved_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}.")
    if resolved_chunk < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens}.")
    for start in range(0, resolved_length, resolved_chunk):
        yield start, min(resolved_length, start + resolved_chunk)


def copy_teacher_logit_chunk_to_device(
    teacher_logits_cpu: torch.Tensor,
    *,
    start: int,
    end: int,
    target_device: torch.device,
) -> torch.Tensor:
    if teacher_logits_cpu.device.type != "cpu":
        raise ValueError("teacher_logits_cpu must reside on CPU.")
    if teacher_logits_cpu.ndim != 3:
        raise ValueError(
            "teacher_logits_cpu must have shape [B, L, V], "
            f"got {tuple(teacher_logits_cpu.shape)}."
        )
    sequence_length = int(teacher_logits_cpu.shape[1])
    if not (0 <= int(start) < int(end) <= sequence_length):
        raise ValueError(
            f"Invalid token range [{start}:{end}] for sequence length {sequence_length}."
        )
    target = torch.device(target_device)
    non_blocking = bool(teacher_logits_cpu.is_pinned() and target.type == "cuda")
    return teacher_logits_cpu[:, int(start):int(end), :].to(
        device=target,
        non_blocking=non_blocking,
    )


def extract_primary_hidden(value: object, *, context: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"{context} does not contain a primary hidden-state tensor.")

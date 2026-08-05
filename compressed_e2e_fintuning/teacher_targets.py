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


class TeacherHiddenTargetCollector:
    def __init__(
        self,
        *,
        model: nn.Module,
        attention_mask: Optional[torch.Tensor],
        layer_weighting: str,
        pin_memory: bool,
        score_chunk_tokens: int = 64,
    ) -> None:
        from train_utils.lora_training import (
            compute_masked_hidden_transition_cosine,
            is_adaptive_hidden_alignment_layer_weighting,
            parse_adaptive_hidden_alignment_topk,
            parse_distill_hidden_alignment_layer_weighting,
        )

        self._score_fn = compute_masked_hidden_transition_cosine
        self._layers = resolve_transformer_layers(model)
        self._mode = parse_distill_hidden_alignment_layer_weighting(layer_weighting)
        self._adaptive = is_adaptive_hidden_alignment_layer_weighting(self._mode)
        self._topk = (
            parse_adaptive_hidden_alignment_topk(self._mode)
            if self._adaptive
            else len(self._layers)
        )
        self._attention_mask = attention_mask
        self._pin_memory = bool(pin_memory)
        self._score_chunk_tokens = int(score_chunk_tokens)
        if self._score_chunk_tokens < 1:
            raise ValueError("score_chunk_tokens must be >= 1.")
        self._handles = []
        self._seen_layer_ids: set[int] = set()
        self._static_cache: Dict[int, torch.Tensor] = {}
        self._adaptive_candidates: list[tuple[float, int, torch.Tensor]] = []
        self._context_exited = False
        self._max_retained_hidden_count = 0

    def _make_hook(self, layer_id: int):
        fixed_layer_id = int(layer_id)

        def hook(_module, args, output):
            if fixed_layer_id in self._seen_layer_ids:
                raise RuntimeError(
                    f"Teacher Transformer layer {fixed_layer_id} ran more than once "
                    "inside one hidden-target collection context."
                )
            if not args:
                raise RuntimeError(
                    f"Teacher Transformer layer {fixed_layer_id} did not receive "
                    "hidden states as its first positional argument."
                )

            input_hidden = extract_primary_hidden(
                args[0],
                context=f"teacher layer {fixed_layer_id} input",
            )
            output_hidden = extract_primary_hidden(
                output,
                context=f"teacher layer {fixed_layer_id} output",
            )
            score = float(
                self._score_fn(
                    input_hidden=input_hidden,
                    output_hidden=output_hidden,
                    attention_mask=self._attention_mask,
                    sequence_chunk_size=self._score_chunk_tokens,
                ).item()
            )
            self._seen_layer_ids.add(fixed_layer_id)

            if not self._adaptive:
                self._static_cache[fixed_layer_id] = copy_detached_tensor_to_cpu(
                    output_hidden,
                    pin_memory=self._pin_memory,
                )
                self._max_retained_hidden_count = max(
                    self._max_retained_hidden_count,
                    len(self._static_cache),
                )
                return

            candidate_key = (score, fixed_layer_id)
            if len(self._adaptive_candidates) < self._topk:
                cpu_hidden = copy_detached_tensor_to_cpu(
                    output_hidden,
                    pin_memory=self._pin_memory,
                )
                self._adaptive_candidates.append(
                    (score, fixed_layer_id, cpu_hidden)
                )
            else:
                worst_index = max(
                    range(len(self._adaptive_candidates)),
                    key=lambda index: (
                        self._adaptive_candidates[index][0],
                        self._adaptive_candidates[index][1],
                    ),
                )
                worst_score, worst_layer_id, worst_hidden = self._adaptive_candidates[
                    worst_index
                ]
                if candidate_key >= (worst_score, worst_layer_id):
                    return

                self._adaptive_candidates.pop(worst_index)
                del worst_hidden
                cpu_hidden = copy_detached_tensor_to_cpu(
                    output_hidden,
                    pin_memory=self._pin_memory,
                )
                self._adaptive_candidates.append(
                    (score, fixed_layer_id, cpu_hidden)
                )

            self._max_retained_hidden_count = max(
                self._max_retained_hidden_count,
                len(self._adaptive_candidates),
            )
            if len(self._adaptive_candidates) > self._topk:
                raise RuntimeError("Adaptive teacher hidden cache exceeded top-k.")

        return hook

    def __enter__(self):
        if self._handles:
            raise RuntimeError("TeacherHiddenTargetCollector cannot be re-entered.")
        self._context_exited = False
        for layer_id, layer in enumerate(self._layers):
            self._handles.append(
                layer.register_forward_hook(self._make_hook(layer_id))
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context_exited = True
        return False

    def finalize(self):
        if not self._context_exited:
            raise RuntimeError(
                "Teacher hidden targets can be finalized only after context exit."
            )
        expected = set(range(len(self._layers)))
        if self._seen_layer_ids != expected:
            raise RuntimeError(
                "Teacher hidden collector layer coverage mismatch: "
                f"expected={sorted(expected)} seen={sorted(self._seen_layer_ids)}."
            )

        if self._adaptive:
            ordered = sorted(
                self._adaptive_candidates,
                key=lambda item: (item[0], item[1]),
            )
            selected_ids = tuple(layer_id for _score, layer_id, _hidden in ordered)
            hidden_by_layer = {
                layer_id: hidden
                for _score, layer_id, hidden in ordered
            }
        else:
            selected_ids = tuple(range(len(self._layers)))
            hidden_by_layer = {
                layer_id: self._static_cache[layer_id]
                for layer_id in selected_ids
            }
        return selected_ids, hidden_by_layer, len(self._layers)


class StudentHiddenCollector:
    def __init__(
        self,
        *,
        model: nn.Module,
        layer_indices: Tuple[int, ...],
    ) -> None:
        self._layers = resolve_transformer_layers(model)
        normalized = tuple(int(layer_id) for layer_id in layer_indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError("student hidden layer_indices must be unique.")
        for layer_id in normalized:
            if layer_id < 0 or layer_id >= len(self._layers):
                raise ValueError(
                    f"student hidden layer id {layer_id} is outside "
                    f"[0, {len(self._layers)})."
                )
        self._layer_indices = normalized
        self._handles = []
        self._captured: Dict[int, torch.Tensor] = {}
        self._context_exited = False

    def _make_hook(self, layer_id: int):
        fixed_layer_id = int(layer_id)

        def hook(_module, _args, output):
            if fixed_layer_id in self._captured:
                raise RuntimeError(
                    f"Student Transformer layer {fixed_layer_id} ran more than once "
                    "inside one hidden collection context."
                )
            self._captured[fixed_layer_id] = extract_primary_hidden(
                output,
                context=f"student layer {fixed_layer_id} output",
            )

        return hook

    def __enter__(self):
        if self._handles:
            raise RuntimeError("StudentHiddenCollector cannot be re-entered.")
        self._captured.clear()
        self._context_exited = False
        for layer_id in self._layer_indices:
            self._handles.append(
                self._layers[layer_id].register_forward_hook(
                    self._make_hook(layer_id)
                )
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context_exited = True
        return False

    def collected(self) -> Dict[int, torch.Tensor]:
        if not self._context_exited:
            raise RuntimeError(
                "Student hidden tensors can be read only after context exit."
            )
        missing = [
            layer_id
            for layer_id in self._layer_indices
            if layer_id not in self._captured
        ]
        if missing:
            raise RuntimeError(
                f"Student hidden collector missed layers: {missing}."
            )
        return {
            layer_id: self._captured[layer_id]
            for layer_id in self._layer_indices
        }

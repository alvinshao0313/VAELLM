import argparse
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from e2e_common.selective_topk_head import (
    TeacherTopKTargets,
    extract_teacher_topk_targets,
    is_selective_student_topk_loss,
    move_teacher_topk_targets_to_device,
    parse_selective_student_topk_k,
    selective_student_lm_head,
)
from train_utils.distill_loss_core import (
    compute_model_level_loss,
    compute_selected_kl_top_model_level_loss,
    normalize_model_level_loss_type,
)
from train_utils.distill_token_stats import DistillTokenStatsAccumulator
from train_utils.hif4_act import Hif4ActController
from train_utils.distill_teacher import DistillTeacherRuntime, resolve_distill_teacher_required
from train_utils.config.configs import DistillLossConfig


logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

try:
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
except ImportError:
    DataCollatorForCompletionOnlyLM = None
    SFTTrainer = None


_DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES = ("uniform", "linear_depth")
_DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES = (
    "uniform",
    "linear_depth",
    "adaptive",
    "adaptive_top_<K>",
)
_DEFAULT_ADAPTIVE_TOPK = 3


@dataclass
class _TeacherTargets:
    logits: Optional[torch.Tensor]
    selective_topk: Optional[TeacherTopKTargets]
    reference_hidden: Optional[torch.Tensor]
    hidden_by_name: Optional[Dict[str, torch.Tensor]]
    pre_mlp_by_name: Optional[Dict[str, torch.Tensor]]


def parse_distill_hidden_alignment_layer_weighting(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode in _DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES:
        return mode
    if mode == "adaptive":
        return mode
    if mode.startswith("adaptive_top"):
        suffix = mode[len("adaptive_top") :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if not suffix.isdigit() or int(suffix) < 1:
            raise argparse.ArgumentTypeError(
                f"Invalid --distill_hidden_alignment_layer_weighting: {raw!r}. "
                "adaptive_top suffix must be a positive integer, e.g. adaptive_top_3."
            )
        return mode
    raise argparse.ArgumentTypeError(
        f"Invalid --distill_hidden_alignment_layer_weighting: {raw!r}. "
        "Supported: uniform, linear_depth, adaptive, adaptive_top_<K>."
    )


def is_adaptive_hidden_alignment_layer_weighting(layer_weighting: str) -> bool:
    mode = str(layer_weighting).strip().lower()
    return mode == "adaptive" or mode.startswith("adaptive_top")


def parse_adaptive_hidden_alignment_topk(layer_weighting: str, default_k: int = _DEFAULT_ADAPTIVE_TOPK) -> int:
    mode = str(layer_weighting).strip().lower()
    if not is_adaptive_hidden_alignment_layer_weighting(mode):
        raise ValueError(
            f"parse_adaptive_hidden_alignment_topk expects adaptive layer weighting, got {layer_weighting!r}."
        )
    if mode == "adaptive":
        return max(1, int(default_k))
    suffix = mode[len("adaptive_top") :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        raise ValueError(
            f"Invalid adaptive layer weighting: {layer_weighting!r}. "
            "Use adaptive or adaptive_top_<K> with a positive integer K."
        )
    return max(1, int(suffix))


def build_distill_hidden_layer_weights(
    *,
    num_layers: int,
    layer_weighting: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}.")
    mode = str(layer_weighting).strip().lower()
    if is_adaptive_hidden_alignment_layer_weighting(mode):
        raise ValueError(
            "adaptive layer weighting must not use build_distill_hidden_layer_weights; "
            "use _aggregate_hidden_alignment_layer_losses instead."
        )
    if mode == "uniform":
        return torch.ones(num_layers, device=device, dtype=dtype)
    if mode == "linear_depth":
        denom = max(num_layers - 1, 1)
        raw = 1.0 + torch.arange(num_layers, device=device, dtype=dtype) / float(denom)
        return raw / raw.mean()
    raise ValueError(
        f"Unsupported distill hidden layer weighting: {layer_weighting}. "
        f"Supported: {', '.join(_DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES)}."
    )


def _masked_mean_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    a = a.float().detach()
    b = b.float().detach()
    cos = F.cosine_similarity(a, b, dim=-1)
    if attention_mask is None:
        return cos.mean()
    mask = attention_mask.to(device=cos.device, dtype=cos.dtype)
    while mask.ndim < cos.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(cos)
    count = mask.sum().clamp_min(1.0)
    return (cos * mask).sum() / count


def _select_adaptive_hidden_layer_indices(
    teacher_sequence: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    topk: int,
    *,
    reference_hidden: Optional[torch.Tensor] = None,
) -> List[int]:
    num_layers = len(teacher_sequence)
    if num_layers <= 0:
        raise ValueError("teacher_sequence must be non-empty for adaptive layer selection.")
    topk = min(max(1, int(topk)), num_layers)

    scores: List[Tuple[int, float]] = []
    for layer_idx in range(num_layers):
        hidden = teacher_sequence[layer_idx]
        if layer_idx == 0:
            if reference_hidden is None:
                raise ValueError("reference_hidden is required for adaptive selection at layer 0.")
            previous = reference_hidden
        else:
            previous = teacher_sequence[layer_idx - 1]
        cosine = _masked_mean_cosine_similarity(hidden, previous, attention_mask)
        scores.append((layer_idx, float(cosine.item())))

    selected = sorted(scores, key=lambda item: item[1])[:topk]
    return [layer_idx for layer_idx, _ in selected]


def _aggregate_hidden_alignment_layer_losses(
    layer_losses: List[torch.Tensor],
    layer_weighting: str,
    *,
    teacher_sequence_for_selection: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_hidden: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    stacked = torch.stack(layer_losses)
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting):
        topk = parse_adaptive_hidden_alignment_topk(layer_weighting)
        selected = _select_adaptive_hidden_layer_indices(
            teacher_sequence_for_selection,
            attention_mask,
            topk,
            reference_hidden=reference_hidden,
        )
        return stacked[selected].mean()

    weights = build_distill_hidden_layer_weights(
        num_layers=len(layer_losses),
        layer_weighting=layer_weighting,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    return (stacked * weights).mean()


def _masked_mean_square(value: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    value = value.float()
    square = value.pow(2)
    if attention_mask is None:
        return square.mean()
    mask = attention_mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    count = mask.sum().clamp_min(1.0)
    return (square * mask).sum() / count


def compute_masked_hidden_transition_cosine(
    *,
    input_hidden: torch.Tensor,
    output_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sequence_chunk_size: int = 64,
) -> torch.Tensor:
    if input_hidden.ndim != 3 or output_hidden.ndim != 3:
        raise ValueError("hidden tensors must have shape [B, L, H].")
    if tuple(input_hidden.shape) != tuple(output_hidden.shape):
        raise ValueError(
            "input/output hidden shape mismatch: "
            f"{tuple(input_hidden.shape)} vs {tuple(output_hidden.shape)}."
        )
    resolved_chunk = int(sequence_chunk_size)
    if resolved_chunk < 1:
        raise ValueError(
            f"sequence_chunk_size must be >= 1, got {sequence_chunk_size}."
        )
    if attention_mask is not None and tuple(attention_mask.shape) != tuple(input_hidden.shape[:2]):
        raise ValueError(
            "attention_mask shape mismatch: "
            f"expected {tuple(input_hidden.shape[:2])}, got {tuple(attention_mask.shape)}."
        )

    total = torch.zeros((), device=output_hidden.device, dtype=torch.float32)
    count = torch.zeros((), device=output_hidden.device, dtype=torch.float32)
    sequence_length = int(output_hidden.shape[1])
    for start in range(0, sequence_length, resolved_chunk):
        end = min(sequence_length, start + resolved_chunk)
        input_chunk = input_hidden[:, start:end, :].detach().float()
        output_chunk = output_hidden[:, start:end, :].detach().float()
        cosine = F.cosine_similarity(output_chunk, input_chunk, dim=-1)
        if attention_mask is None:
            mask_chunk = torch.ones_like(cosine, dtype=torch.float32)
        else:
            mask_chunk = attention_mask[:, start:end].to(
                device=cosine.device,
                dtype=torch.float32,
            )
        total.add_((cosine * mask_chunk).sum())
        count.add_(mask_chunk.sum())
    return total / count.clamp_min(1.0)


def compute_selected_distill_hidden_alignment_loss(
    *,
    teacher_hidden_by_layer: Dict[int, torch.Tensor],
    student_hidden_by_layer: Dict[int, torch.Tensor],
    hidden_layer_indices: Tuple[int, ...],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    num_layers: int,
    loss_device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not hidden_layer_indices:
        raise ValueError("hidden_layer_indices must be non-empty.")
    normalized_indices = tuple(int(layer_id) for layer_id in hidden_layer_indices)
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("hidden_layer_indices must be unique.")
    resolved_num_layers = int(num_layers)
    if resolved_num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    for layer_id in normalized_indices:
        if layer_id < 0 or layer_id >= resolved_num_layers:
            raise ValueError(
                f"hidden layer id {layer_id} is outside [0, {resolved_num_layers})."
            )

    expected_keys = set(normalized_indices)
    teacher_keys = set(int(key) for key in teacher_hidden_by_layer.keys())
    student_keys = set(int(key) for key in student_hidden_by_layer.keys())
    if teacher_keys != expected_keys:
        raise ValueError(
            "teacher_hidden_by_layer keys must exactly match hidden_layer_indices: "
            f"expected={sorted(expected_keys)} got={sorted(teacher_keys)}."
        )
    if student_keys != expected_keys:
        raise ValueError(
            "student_hidden_by_layer keys must exactly match hidden_layer_indices: "
            f"expected={sorted(expected_keys)} got={sorted(student_keys)}."
        )

    mode = parse_distill_hidden_alignment_layer_weighting(layer_weighting)
    if not is_adaptive_hidden_alignment_layer_weighting(mode):
        full_indices = tuple(range(resolved_num_layers))
        if normalized_indices != full_indices:
            raise ValueError(
                "static hidden alignment requires hidden_layer_indices == "
                f"tuple(range(num_layers)); got {normalized_indices}."
            )

    for layer_id in normalized_indices:
        teacher_hidden_cpu = teacher_hidden_by_layer[int(layer_id)]
        student_hidden = student_hidden_by_layer[int(layer_id)]
        if teacher_hidden_cpu.device.type != "cpu":
            raise ValueError(
                f"teacher hidden for layer {layer_id} must reside on CPU, "
                f"got {teacher_hidden_cpu.device}."
            )
        if tuple(teacher_hidden_cpu.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student hidden shape mismatch at layer {layer_id}: "
                f"{tuple(teacher_hidden_cpu.shape)} vs {tuple(student_hidden.shape)}."
            )

    loss_device = torch.device(loss_device)
    layer_losses = []

    if not is_adaptive_hidden_alignment_layer_weighting(mode):
        full_weights = build_distill_hidden_layer_weights(
            num_layers=int(num_layers),
            layer_weighting=mode,
            device=loss_device,
            dtype=torch.float32,
        )

    for layer_id in hidden_layer_indices:
        student_hidden = student_hidden_by_layer[int(layer_id)]
        teacher_hidden_cpu = teacher_hidden_by_layer[int(layer_id)]
        teacher_hidden = teacher_hidden_cpu.to(
            device=student_hidden.device,
            non_blocking=bool(
                teacher_hidden_cpu.is_pinned()
                and student_hidden.device.type == "cuda"
            ),
        )
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        local_loss = numerator / (denominator + float(eps))
        if is_adaptive_hidden_alignment_layer_weighting(mode):
            layer_losses.append(local_loss.to(device=loss_device))
        else:
            layer_losses.append(
                (local_loss * full_weights[int(layer_id)]).to(device=loss_device)
            )

    return torch.stack(layer_losses).mean()


def compute_distill_hidden_alignment_loss(
    *,
    teacher_hidden_states: Sequence[torch.Tensor],
    student_hidden_states: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if teacher_hidden_states is None or student_hidden_states is None:
        raise ValueError("Hidden states are required when LoRA hidden alignment loss is enabled.")
    if len(teacher_hidden_states) != len(student_hidden_states):
        raise ValueError(
            "Teacher/student hidden state counts differ: "
            f"{len(teacher_hidden_states)} vs {len(student_hidden_states)}."
        )
    if len(teacher_hidden_states) <= 1:
        raise ValueError("Hidden states must include embedding output plus at least one transformer block output.")

    layer_losses: List[torch.Tensor] = []
    for layer_idx, (teacher_hidden, student_hidden) in enumerate(
        zip(teacher_hidden_states[1:], student_hidden_states[1:])
    ):
        if tuple(teacher_hidden.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student hidden shape mismatch at block layer {layer_idx}: "
                f"{tuple(teacher_hidden.shape)} vs {tuple(student_hidden.shape)}."
            )
        teacher_hidden = teacher_hidden.detach()
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        layer_losses.append(numerator / (denominator + float(eps)))

    teacher_block_hiddens = [hidden.detach() for hidden in teacher_hidden_states[1:]]
    return _aggregate_hidden_alignment_layer_losses(
        layer_losses,
        layer_weighting,
        teacher_sequence_for_selection=teacher_block_hiddens,
        attention_mask=attention_mask,
        reference_hidden=teacher_hidden_states[0].detach(),
    )


def compute_distill_pre_mlp_hidden_alignment_loss(
    *,
    teacher_pre_mlp_hiddens: Sequence[torch.Tensor],
    student_pre_mlp_hiddens: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    teacher_reference_hidden: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if teacher_pre_mlp_hiddens is None or student_pre_mlp_hiddens is None:
        raise ValueError("Pre-MLP hidden states are required when pre-MLP hidden alignment loss is enabled.")
    if len(teacher_pre_mlp_hiddens) != len(student_pre_mlp_hiddens):
        raise ValueError(
            "Teacher/student pre-MLP hidden state counts differ: "
            f"{len(teacher_pre_mlp_hiddens)} vs {len(student_pre_mlp_hiddens)}."
        )
    if len(teacher_pre_mlp_hiddens) == 0:
        raise ValueError("Pre-MLP hidden states must include at least one transformer block.")
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting) and teacher_reference_hidden is None:
        raise ValueError(
            "teacher_reference_hidden is required for adaptive pre-MLP hidden alignment layer weighting."
        )

    layer_losses: List[torch.Tensor] = []
    for layer_idx, (teacher_hidden, student_hidden) in enumerate(
        zip(teacher_pre_mlp_hiddens, student_pre_mlp_hiddens)
    ):
        if tuple(teacher_hidden.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student pre-MLP hidden shape mismatch at block layer {layer_idx}: "
                f"{tuple(teacher_hidden.shape)} vs {tuple(student_hidden.shape)}."
            )
        teacher_hidden = teacher_hidden.detach()
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        layer_losses.append(numerator / (denominator + float(eps)))

    teacher_sequence = [hidden.detach() for hidden in teacher_pre_mlp_hiddens]
    reference_hidden = (
        teacher_reference_hidden.detach()
        if teacher_reference_hidden is not None
        else None
    )
    return _aggregate_hidden_alignment_layer_losses(
        layer_losses,
        layer_weighting,
        teacher_sequence_for_selection=teacher_sequence,
        attention_mask=attention_mask,
        reference_hidden=reference_hidden,
    )


def _hidden_states_to_named_blocks(
    hidden_states: Sequence[torch.Tensor],
    *,
    num_layers: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if hidden_states is None:
        raise ValueError("hidden_states is required.")
    expected = int(num_layers) + 1
    if len(hidden_states) != expected:
        raise ValueError(
            f"hidden_states length must be num_layers + 1: expected={expected} got={len(hidden_states)}."
        )
    reference_hidden = hidden_states[0]
    blocks = {
        f"model.layers.{layer_idx}": hidden_states[layer_idx + 1]
        for layer_idx in range(int(num_layers))
    }
    return reference_hidden, blocks


def _materialize_teacher_tensor(
    tensor: torch.Tensor,
    *,
    stage_to_cpu: bool,
    cpu_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    raw = tensor.detach()
    if stage_to_cpu:
        materialized = raw.to(
            device=torch.device("cpu"),
            dtype=cpu_dtype if cpu_dtype is not None else raw.dtype,
            copy=True,
        )
        if bool(getattr(materialized, "is_inference", lambda: False)()):
            materialized = materialized.clone()
    else:
        materialized = raw.clone()
    if bool(getattr(materialized, "is_inference", lambda: False)()):
        raise RuntimeError("materialized teacher target is still an inference tensor.")
    return materialized


def _logical_layer_id(logical_name: str) -> int:
    prefix = "model.layers."
    if not str(logical_name).startswith(prefix):
        raise ValueError(f"Unexpected logical layer name: {logical_name}")
    rest = str(logical_name)[len(prefix):]
    token = rest.split(".", 1)[0]
    return int(token)


def _compute_named_hidden_alignment_loss(
    *,
    teacher_by_name: Dict[str, torch.Tensor],
    student_by_name: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    teacher_reference_hidden: torch.Tensor,
    student_reference_hidden: torch.Tensor,
    teacher_targets_on_cpu: bool,
) -> torch.Tensor:
    teacher_names = tuple(teacher_by_name.keys())
    student_names = tuple(student_by_name.keys())
    if teacher_names != student_names:
        raise ValueError(
            "Teacher/student hidden logical names differ: "
            f"teacher={teacher_names} student={student_names}."
        )
    if not teacher_targets_on_cpu:
        teacher_hidden_states = (teacher_reference_hidden,) + tuple(
            teacher_by_name[name] for name in teacher_names
        )
        student_hidden_states = (student_reference_hidden,) + tuple(
            student_by_name[name] for name in student_names
        )
        return compute_distill_hidden_alignment_loss(
            teacher_hidden_states=teacher_hidden_states,
            student_hidden_states=student_hidden_states,
            attention_mask=attention_mask,
            layer_weighting=layer_weighting,
        )

    num_layers = len(teacher_names)
    all_indices = tuple(range(num_layers))
    selected_indices = all_indices
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting):
        selected_indices = tuple(
            _select_adaptive_hidden_layer_indices(
                [teacher_by_name[name] for name in teacher_names],
                attention_mask,
                parse_adaptive_hidden_alignment_topk(layer_weighting),
                reference_hidden=teacher_reference_hidden,
            )
        )
    teacher_selected = {
        int(idx): teacher_by_name[teacher_names[int(idx)]]
        for idx in selected_indices
    }
    student_selected = {
        int(idx): student_by_name[student_names[int(idx)]]
        for idx in selected_indices
    }
    return compute_selected_distill_hidden_alignment_loss(
        teacher_hidden_by_layer=teacher_selected,
        student_hidden_by_layer=student_selected,
        hidden_layer_indices=selected_indices,
        attention_mask=attention_mask,
        layer_weighting=layer_weighting,
        num_layers=num_layers,
        loss_device=next(iter(student_by_name.values())).device,
    )


def _compute_named_pre_mlp_hidden_alignment_loss(
    *,
    teacher_by_name: Dict[str, torch.Tensor],
    student_by_name: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    teacher_reference_hidden: Optional[torch.Tensor],
    teacher_targets_on_cpu: bool,
) -> torch.Tensor:
    teacher_names = tuple(teacher_by_name.keys())
    student_names = tuple(student_by_name.keys())
    if teacher_names != student_names:
        raise ValueError(
            "Teacher/student pre-MLP logical names differ: "
            f"teacher={teacher_names} student={student_names}."
        )
    if not teacher_targets_on_cpu:
        return compute_distill_pre_mlp_hidden_alignment_loss(
            teacher_pre_mlp_hiddens=tuple(teacher_by_name[name] for name in teacher_names),
            student_pre_mlp_hiddens=tuple(student_by_name[name] for name in student_names),
            attention_mask=attention_mask,
            layer_weighting=layer_weighting,
            teacher_reference_hidden=teacher_reference_hidden,
        )

    num_layers = len(teacher_names)
    all_indices = tuple(range(num_layers))
    selected_indices = all_indices
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting):
        selected_indices = tuple(
            _select_adaptive_hidden_layer_indices(
                [teacher_by_name[name] for name in teacher_names],
                attention_mask,
                parse_adaptive_hidden_alignment_topk(layer_weighting),
                reference_hidden=teacher_reference_hidden,
            )
        )
    teacher_selected = {
        int(idx): teacher_by_name[teacher_names[int(idx)]]
        for idx in selected_indices
    }
    student_selected = {
        int(idx): student_by_name[student_names[int(idx)]]
        for idx in selected_indices
    }
    return compute_selected_distill_hidden_alignment_loss(
        teacher_hidden_by_layer=teacher_selected,
        student_hidden_by_layer=student_selected,
        hidden_layer_indices=selected_indices,
        attention_mask=attention_mask,
        layer_weighting=layer_weighting,
        num_layers=num_layers,
        loss_device=next(iter(student_by_name.values())).device,
    )


@contextmanager
def capture_pre_mlp_hiddens(model: nn.Module):
    modules = _resolve_pre_mlp_capture_modules(model)
    with _capture_pre_mlp_hiddens_from_modules(modules) as captured:
        yield tuple(captured[name] for name, _module in modules)


def _unwrap_accelerator_model(model: nn.Module, accelerator) -> nn.Module:
    if accelerator is None:
        return model
    return accelerator.unwrap_model(model)


def _resolve_peft_and_base_model(unwrapped_model: nn.Module):
    if PeftModel is not None and isinstance(unwrapped_model, PeftModel):
        return unwrapped_model, unwrapped_model.get_base_model()
    return None, unwrapped_model


def _resolve_student_base_model(unwrapped_model: nn.Module) -> nn.Module:
    if PeftModel is not None and isinstance(unwrapped_model, PeftModel):
        return unwrapped_model.get_base_model()
    return unwrapped_model


def _resolve_pre_mlp_capture_modules(base_model: nn.Module) -> Tuple[Tuple[str, nn.Module], ...]:
    backbone = getattr(base_model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("pre-MLP hidden alignment requires model.model.layers.")

    modules: List[Tuple[str, nn.Module]] = []
    for layer_idx, layer in enumerate(layers):
        module = getattr(layer, "post_attention_layernorm", None)
        if not isinstance(module, nn.Module):
            raise ValueError(
                "pre-MLP hidden alignment requires every model.model.layers[*] "
                f"to expose post_attention_layernorm; missing at layer {layer_idx}."
            )
        modules.append((f"model.layers.{layer_idx}.post_attention_layernorm", module))
    if not modules:
        raise ValueError("pre-MLP hidden alignment requires at least one model.model.layers entry.")
    return tuple(modules)


@contextmanager
def _capture_pre_mlp_hiddens_from_modules(modules: Sequence[Tuple[str, nn.Module]]):
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    for layer_idx, (logical_name, module) in enumerate(modules):

        def hook(_module, inputs, _layer_idx=layer_idx, _logical_name=logical_name):
            if not inputs:
                raise RuntimeError(f"post_attention_layernorm pre-hook at layer {_layer_idx} received no inputs.")
            if _logical_name in captured:
                raise RuntimeError(f"post_attention_layernorm pre-hook captured {_logical_name} more than once.")
            captured[_logical_name] = inputs[0]

        handles.append(module.register_forward_pre_hook(hook))

    if not handles:
        raise ValueError("pre-MLP hidden alignment requires at least one model.model.layers entry.")

    try:
        yield captured
        expected = [name for name, _module in modules]
        actual = list(captured.keys())
        if actual != expected:
            raise RuntimeError(
                "pre-MLP hidden capture did not capture every logical layer exactly once: "
                f"expected={expected} actual={actual}."
            )
    finally:
        for handle in handles:
            handle.remove()


def ensure_lora_training_stack_available() -> None:
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise ImportError("未安装 peft。请先安装：pip install peft")
    if SFTTrainer is None or DataCollatorForCompletionOnlyLM is None:
        raise ImportError("未安装 trl。请先安装：pip install trl")


class _DistillOptimizerGroupingMixin:
    def __init__(
        self,
        *args,
        decoder_param_ids: Optional[Sequence[int]] = None,
        decoder_lr: Optional[float] = None,
        **kwargs,
    ):
        self.distill_decoder_param_ids = frozenset(int(v) for v in (decoder_param_ids or ()))
        self.distill_decoder_lr = None if decoder_lr is None else float(decoder_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        selection = getattr(self, "model_level_trainable_selection", None)
        decoder_param_ids = frozenset(int(v) for v in getattr(self, "distill_decoder_param_ids", frozenset()))
        if selection is None and not decoder_param_ids:
            return super().create_optimizer()
        if selection is None:
            raise RuntimeError(
                "CAT distill optimizer requires model_level_trainable_selection "
                "(Task 6 inventories). decoder_param_ids alone is no longer sufficient."
            )
        from train_utils.model_level_optimizer import create_model_level_optimizer

        return create_model_level_optimizer(self)

if SFTTrainer is None:
    class CustomSFTTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装 trl。请先安装：pip install trl")
    GroupedSFTTrainer = CustomSFTTrainer
else:
    class GroupedSFTTrainer(_DistillOptimizerGroupingMixin, SFTTrainer):
        pass


    class CustomSFTTrainer(_DistillOptimizerGroupingMixin, SFTTrainer):
        def __init__(
            self,
            *args,
            loss_type: str = "sft",
            top_k: int = 100,
            temperature: float = 1.0,
            loss_alpha: float = 0.5,
            hidden_loss_weight: float = 0.0,
            pre_mlp_hidden_loss_weight: float = 0.0,
            prompt_kd_weight: float = 0.0,
            hidden_alignment_layer_weighting: str = "uniform",
            teacher_logits_cpu_staging: bool = False,
            selective_student_topk: bool = False,
            selective_student_topk_chunk_rows: int = 32,
            selective_teacher_topk_chunk_tokens: int = 8,
            distill_hif4_act_controller: Optional[Hif4ActController] = None,
            teacher_runtime: Optional[DistillTeacherRuntime] = None,
            loss_config: Optional[DistillLossConfig] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if loss_config is not None:
                loss_config.validate()
                self.loss_config = loss_config
            else:
                self.loss_config = DistillLossConfig(
                    loss_type=normalize_model_level_loss_type(loss_type),
                    top_k=int(top_k),
                    temperature=float(temperature),
                    alpha=float(loss_alpha),
                    prompt_loss_weight=float(prompt_kd_weight),
                    hidden_loss_weight=float(hidden_loss_weight),
                    pre_mlp_hidden_loss_weight=float(pre_mlp_hidden_loss_weight),
                    hidden_layer_weighting=str(hidden_alignment_layer_weighting),
                    selective_student_topk=bool(selective_student_topk),
                    selective_student_topk_chunk_rows=int(selective_student_topk_chunk_rows),
                )
                self.loss_config.validate()
            self.loss_type = str(self.loss_config.loss_type)
            self.top_k = int(self.loss_config.top_k)
            self.temperature = float(self.loss_config.temperature)
            self.loss_alpha = float(self.loss_config.alpha)
            self.hidden_loss_weight = float(self.loss_config.hidden_loss_weight)
            self.pre_mlp_hidden_loss_weight = float(self.loss_config.pre_mlp_hidden_loss_weight)
            self.prompt_kd_weight = float(self.loss_config.prompt_loss_weight)
            self.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                self.loss_config.hidden_layer_weighting
            )
            self.teacher_logits_cpu_staging = bool(teacher_logits_cpu_staging)
            self.selective_student_topk = bool(self.loss_config.selective_student_topk)
            self.selective_student_topk_chunk_rows = int(self.loss_config.selective_student_topk_chunk_rows)
            self.selective_teacher_topk_chunk_tokens = int(selective_teacher_topk_chunk_tokens)
            if self.selective_teacher_topk_chunk_tokens < 1:
                raise ValueError("selective_teacher_topk_chunk_tokens must be >= 1.")
            if self.selective_student_topk and not is_selective_student_topk_loss(self.loss_type):
                raise ValueError("selective_student_topk only supports loss_type=kl_top.")
            self.distill_hif4_act_controller = distill_hif4_act_controller
            self.teacher_runtime = teacher_runtime
            self.teacher_required = resolve_distill_teacher_required(
                loss_type=self.loss_type,
                hidden_loss_weight=self.hidden_loss_weight,
                pre_mlp_hidden_loss_weight=self.pre_mlp_hidden_loss_weight,
            )
            if self.teacher_required and self.teacher_runtime is None:
                raise ValueError("teacher_runtime is required for CAT distillation teacher-required losses.")
            self._runtime_view_cache_key = None
            self._runtime_view_cache = None
            self.distill_token_stats = DistillTokenStatsAccumulator()

        def _resolved_loss_config(self) -> DistillLossConfig:
            """Single-truth DistillLossConfig; rebuild from attrs if tests used ``__new__``."""
            cfg = getattr(self, "loss_config", None)
            if cfg is not None:
                return cfg
            canonical = normalize_model_level_loss_type(getattr(self, "loss_type", "sft"))
            resolved_top_k = int(getattr(self, "top_k", 100))
            cfg = DistillLossConfig(
                loss_type=canonical,
                top_k=resolved_top_k,
                temperature=float(getattr(self, "temperature", 1.0)),
                alpha=float(getattr(self, "loss_alpha", 0.5)),
                prompt_loss_weight=float(getattr(self, "prompt_kd_weight", 0.0)),
                hidden_loss_weight=float(getattr(self, "hidden_loss_weight", 0.0)),
                pre_mlp_hidden_loss_weight=float(getattr(self, "pre_mlp_hidden_loss_weight", 0.0)),
                hidden_layer_weighting=str(
                    getattr(self, "hidden_alignment_layer_weighting", "uniform")
                ),
                selective_student_topk=bool(getattr(self, "selective_student_topk", False)),
                selective_student_topk_chunk_rows=int(
                    getattr(self, "selective_student_topk_chunk_rows", 32)
                ),
            )
            cfg.validate()
            self.loss_config = cfg
            self.loss_type = str(cfg.loss_type)
            self.top_k = int(cfg.top_k)
            return cfg

        def _resolve_runtime_view_cache(self, model, pre_mlp_hidden_loss_enabled: bool):
            unwrapped_model = _unwrap_accelerator_model(
                model,
                getattr(self, "accelerator", None),
            )
            cache_key = id(unwrapped_model)
            cache = (
                getattr(self, "_runtime_view_cache", None)
                if getattr(self, "_runtime_view_cache_key", None) == cache_key
                else None
            )
            needs_pre_mlp = bool(pre_mlp_hidden_loss_enabled)
            if cache is None or (needs_pre_mlp and cache.get("pre_mlp_capture_modules") is None):
                base_model_for_capture = _resolve_student_base_model(unwrapped_model)
                pre_mlp_capture_modules = (
                    _resolve_pre_mlp_capture_modules(base_model_for_capture)
                    if needs_pre_mlp
                    else None
                )
                cache = {
                    "unwrapped_model": unwrapped_model,
                    "base_model_for_capture": base_model_for_capture,
                    "pre_mlp_capture_modules": pre_mlp_capture_modules,
                }
                self._runtime_view_cache_key = cache_key
                self._runtime_view_cache = cache
            return cache

        def _teacher_logits_staging_dtype(self) -> torch.dtype:
            if bool(getattr(self.args, "bf16", False)):
                return torch.bfloat16
            if bool(getattr(self.args, "fp16", False)):
                return torch.float16
            return torch.float32

        def _must_stage_teacher_targets_to_cpu(self) -> bool:
            runtime = getattr(self, "teacher_runtime", None)
            return bool(runtime is not None and getattr(runtime, "model_offload", "none") == "cpu")

        def _stage_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
            return _materialize_teacher_tensor(
                logits,
                stage_to_cpu=(
                    self._must_stage_teacher_targets_to_cpu()
                    or bool(getattr(self, "teacher_logits_cpu_staging", False))
                ),
                cpu_dtype=self._teacher_logits_staging_dtype(),
            )

        def _teacher_logits_for_loss(
            self,
            staged_logits: torch.Tensor,
            student_logits: torch.Tensor,
        ) -> torch.Tensor:
            if staged_logits.device.type == "cpu":
                return staged_logits.to(device=student_logits.device, non_blocking=True)
            return staged_logits

        def _run_teacher_forward(
            self,
            *,
            teacher_inputs,
            need_logits: bool,
            need_output_hidden_states: bool,
            need_pre_mlp_hiddens: bool,
            student_pre_mlp_modules,
        ) -> _TeacherTargets:
            teacher_runtime = getattr(self, "teacher_runtime", None)
            if teacher_runtime is None:
                raise ValueError("teacher_runtime is required for teacher forward.")
            teacher = teacher_runtime.prepare_for_forward()
            targets = None
            try:
                teacher_pre_mlp_modules = (
                    _resolve_pre_mlp_capture_modules(_resolve_student_base_model(teacher))
                    if need_pre_mlp_hiddens
                    else None
                )
                if need_pre_mlp_hiddens:
                    teacher_names = tuple(name for name, _module in teacher_pre_mlp_modules)
                    student_names = tuple(name for name, _module in student_pre_mlp_modules)
                    if teacher_names != student_names:
                        raise ValueError(
                            "Teacher/student pre-MLP logical names differ: "
                            f"teacher={teacher_names} student={student_names}."
                        )
                    for (teacher_name, teacher_module), (student_name, student_module) in zip(
                        teacher_pre_mlp_modules,
                        student_pre_mlp_modules,
                    ):
                        if teacher_name != student_name:
                            raise ValueError(
                                f"Teacher/student pre-MLP logical name mismatch: {teacher_name} vs {student_name}."
                            )
                        if teacher_module is student_module:
                            raise ValueError(
                                f"Teacher/student pre-MLP module is shared for {teacher_name}."
                            )
                pre_mlp_context = (
                    _capture_pre_mlp_hiddens_from_modules(teacher_pre_mlp_modules)
                    if need_pre_mlp_hiddens
                    else nullcontext()
                )
                with pre_mlp_context as captured_pre_mlp:
                    with torch.inference_mode():
                        outputs = teacher(
                            **teacher_inputs,
                            output_hidden_states=need_output_hidden_states,
                        )

                stage_hidden_to_cpu = self._must_stage_teacher_targets_to_cpu()
                selective_topk = (
                    extract_teacher_topk_targets(
                        outputs.logits,
                        k=parse_selective_student_topk_k(self.loss_type, top_k=self.top_k),
                        sequence_chunk_size=self.selective_teacher_topk_chunk_tokens,
                        pin_memory=True,
                    )
                    if need_logits and self.selective_student_topk
                    else None
                )
                logits = (
                    self._stage_teacher_logits(outputs.logits)
                    if need_logits and not self.selective_student_topk
                    else None
                )
                reference_hidden = None
                hidden_by_name = None
                if need_output_hidden_states:
                    teacher_layers = getattr(getattr(_resolve_student_base_model(teacher), "model", None), "layers", None)
                    if teacher_layers is None:
                        raise ValueError("teacher hidden alignment requires teacher.model.layers.")
                    reference_raw, hidden_raw = _hidden_states_to_named_blocks(
                        outputs.hidden_states,
                        num_layers=len(teacher_layers),
                    )
                    reference_hidden = _materialize_teacher_tensor(
                        reference_raw,
                        stage_to_cpu=stage_hidden_to_cpu,
                    )
                    hidden_by_name = {
                        name: _materialize_teacher_tensor(tensor, stage_to_cpu=stage_hidden_to_cpu)
                        for name, tensor in hidden_raw.items()
                    }
                pre_mlp_by_name = None
                if need_pre_mlp_hiddens:
                    pre_mlp_by_name = {
                        name: _materialize_teacher_tensor(tensor, stage_to_cpu=stage_hidden_to_cpu)
                        for name, tensor in captured_pre_mlp.items()
                    }
                targets = _TeacherTargets(
                    logits=logits,
                    selective_topk=selective_topk,
                    reference_hidden=reference_hidden,
                    hidden_by_name=hidden_by_name,
                    pre_mlp_by_name=pre_mlp_by_name,
                )
                del outputs
            finally:
                teacher_runtime.finish_forward()
            if targets is None:
                raise RuntimeError("teacher forward did not produce materialized targets")
            return targets

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            loss_cfg = self._resolved_loss_config()
            if bool(getattr(model, "training", False)):
                original_labels = inputs.get("labels")
                if isinstance(original_labels, torch.Tensor):
                    self.distill_token_stats.update(
                        original_labels, inputs.get("attention_mask")
                    )
            hidden_loss_enabled = float(loss_cfg.hidden_loss_weight) > 0.0
            pre_mlp_hidden_loss_enabled = float(loss_cfg.pre_mlp_hidden_loss_weight) > 0.0
            pre_mlp_reference_hidden_required = bool(
                pre_mlp_hidden_loss_enabled
                and is_adaptive_hidden_alignment_layer_weighting(
                    self.hidden_alignment_layer_weighting
                )
            )
            need_teacher_output_hidden_states = bool(
                hidden_loss_enabled
                or pre_mlp_reference_hidden_required
            )
            need_student_output_hidden_states = bool(hidden_loss_enabled)
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            student_inputs.pop("labels", None)
            full_inputs = dict(inputs)

            runtime_view = self._resolve_runtime_view_cache(
                model,
                pre_mlp_hidden_loss_enabled=pre_mlp_hidden_loss_enabled,
            )
            hif4_act_controller = self.distill_hif4_act_controller
            previous_hif4_enabled = bool(getattr(hif4_act_controller, "enabled", False))
            pre_mlp_capture_modules = runtime_view["pre_mlp_capture_modules"]
            student_pre_mlp_hiddens = None

            def set_hif4_act_enabled(enabled: bool) -> None:
                if hif4_act_controller is not None:
                    hif4_act_controller.enabled = bool(enabled)

            def prepare_student_path() -> None:
                set_hif4_act_enabled(previous_hif4_enabled)

            def student_forward(model_inputs):
                nonlocal student_pre_mlp_hiddens
                pre_mlp_context = (
                    _capture_pre_mlp_hiddens_from_modules(pre_mlp_capture_modules)
                    if pre_mlp_hidden_loss_enabled
                    else nullcontext()
                )
                with pre_mlp_context as captured_pre_mlp:
                    outputs = model(
                        **model_inputs,
                        output_hidden_states=need_student_output_hidden_states,
                    )
                if pre_mlp_hidden_loss_enabled:
                    student_pre_mlp_hiddens = dict(captured_pre_mlp)
                return outputs

            def add_hidden_alignment_loss(loss, teacher_targets, student_outputs):
                if hidden_loss_enabled:
                    if teacher_targets.hidden_by_name is None or teacher_targets.reference_hidden is None:
                        raise RuntimeError("hidden alignment requires teacher hidden targets.")
                    student_layers = getattr(getattr(runtime_view["base_model_for_capture"], "model", None), "layers", None)
                    if student_layers is None:
                        raise ValueError("student hidden alignment requires student.model.layers.")
                    student_reference_hidden, student_hidden_by_name = _hidden_states_to_named_blocks(
                        student_outputs.hidden_states,
                        num_layers=len(student_layers),
                    )
                    hidden_loss = _compute_named_hidden_alignment_loss(
                        teacher_by_name=teacher_targets.hidden_by_name,
                        student_by_name=student_hidden_by_name,
                        attention_mask=full_inputs.get("attention_mask"),
                        layer_weighting=self.hidden_alignment_layer_weighting,
                        teacher_reference_hidden=teacher_targets.reference_hidden,
                        student_reference_hidden=student_reference_hidden,
                        teacher_targets_on_cpu=self._must_stage_teacher_targets_to_cpu(),
                    )
                    loss = loss + float(self.hidden_loss_weight) * hidden_loss
                if pre_mlp_hidden_loss_enabled:
                    if teacher_targets.pre_mlp_by_name is None or student_pre_mlp_hiddens is None:
                        raise RuntimeError("pre-MLP hidden alignment requires teacher and student captured hiddens.")
                    pre_mlp_hidden_loss = _compute_named_pre_mlp_hidden_alignment_loss(
                        teacher_by_name=teacher_targets.pre_mlp_by_name,
                        student_by_name=student_pre_mlp_hiddens,
                        attention_mask=full_inputs.get("attention_mask"),
                        layer_weighting=self.hidden_alignment_layer_weighting,
                        teacher_reference_hidden=(
                            teacher_targets.reference_hidden
                            if pre_mlp_reference_hidden_required
                            else None
                        ),
                        teacher_targets_on_cpu=self._must_stage_teacher_targets_to_cpu(),
                    )
                    loss = loss + float(self.pre_mlp_hidden_loss_weight) * pre_mlp_hidden_loss
                return loss

            try:
                canonical_loss = normalize_model_level_loss_type(loss_cfg.loss_type)
                resolved_top_k = int(loss_cfg.top_k)

                input_ids = full_inputs.get("input_ids")
                labels = full_inputs.get("labels")
                attention_mask = full_inputs.get("attention_mask")
                if not isinstance(input_ids, torch.Tensor):
                    raise ValueError("model-level loss requires input_ids tensor.")
                if not isinstance(labels, torch.Tensor):
                    raise ValueError("model-level loss requires labels tensor.")
                if not isinstance(attention_mask, torch.Tensor):
                    raise ValueError("model-level loss requires attention_mask tensor.")

                need_logits = canonical_loss != "sft"
                teacher_targets = None
                if need_logits or hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
                    teacher_targets = self._run_teacher_forward(
                        teacher_inputs=teacher_inputs,
                        need_logits=need_logits,
                        need_output_hidden_states=need_teacher_output_hidden_states,
                        need_pre_mlp_hiddens=pre_mlp_hidden_loss_enabled,
                        student_pre_mlp_modules=pre_mlp_capture_modules,
                    )

                prepare_student_path()
                if (
                    canonical_loss == "kl_top"
                    and self.selective_student_topk
                ):
                    if teacher_targets is None or teacher_targets.selective_topk is None:
                        raise RuntimeError("selective student top-k requires compact teacher targets.")
                    if int(teacher_targets.selective_topk.k) != int(resolved_top_k):
                        raise RuntimeError("selective teacher top-k K does not match loss K.")
                    input_tensor = next(
                        value for value in student_inputs.values() if torch.is_tensor(value)
                    )
                    selected_indices, selected_teacher_logits = move_teacher_topk_targets_to_device(
                        teacher_targets.selective_topk,
                        device=input_tensor.device,
                    )
                    with selective_student_lm_head(
                        model,
                        teacher_topk_indices=selected_indices,
                        chunk_rows=self.selective_student_topk_chunk_rows,
                    ):
                        outputs = student_forward(student_inputs)
                    loss = compute_selected_kl_top_model_level_loss(
                        student_selected_logits=outputs.logits,
                        teacher_selected_logits=selected_teacher_logits,
                        labels=labels,
                        attention_mask=attention_mask,
                        temperature=float(loss_cfg.temperature),
                        prompt_loss_weight=float(loss_cfg.prompt_loss_weight),
                    )
                else:
                    outputs = student_forward(student_inputs)
                    teacher_logits = None
                    if need_logits:
                        if teacher_targets is None or teacher_targets.logits is None:
                            raise RuntimeError(f"loss_type={canonical_loss} requires teacher logits.")
                        teacher_logits = self._teacher_logits_for_loss(teacher_targets.logits, outputs.logits)
                    loss = compute_model_level_loss(
                        loss_type=canonical_loss,
                        student_logits=outputs.logits,
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        teacher_logits=teacher_logits,
                        temperature=float(loss_cfg.temperature),
                        alpha=float(loss_cfg.alpha),
                        top_k=resolved_top_k,
                        prompt_loss_weight=float(loss_cfg.prompt_loss_weight),
                    )

                if teacher_targets is not None:
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                elif hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
                    raise RuntimeError("hidden alignment requires teacher targets.")
                return (loss, outputs) if return_outputs else loss
            finally:
                set_hif4_act_enabled(previous_hif4_enabled)

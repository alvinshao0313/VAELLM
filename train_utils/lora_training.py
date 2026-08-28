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
    compute_selected_teacher_topk_kl,
    extract_teacher_topk_targets,
    is_selective_student_topk_loss,
    move_teacher_topk_targets_to_device,
    parse_selective_student_topk_k,
    selective_student_lm_head,
)
from train_utils.distill_losses import (
    build_distill_token_regions,
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
    compute_eakld,
    compute_eakld_topk,
    compute_entropy_aware_kl_loss,
    compute_forward_kl_loss,
    compute_kl_topk,
    compute_masked_logit_mse_loss,
    compute_reverse_kl_loss,
    compute_rkl_topk,
    is_eakld_top_loss,
    parse_eakld_top_k,
)
from train_utils.distill_token_stats import DistillTokenStatsAccumulator
from train_utils.hif4_act import Hif4ActController
from train_utils.distill_teacher import DistillTeacherRuntime, resolve_distill_teacher_required


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


def create_lora_adapters(
    model: nn.Module,
    *,
    target_names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    use_dora: bool,
):
    unique_target_names = sorted(set(str(name) for name in target_names if str(name).strip()))
    if not unique_target_names:
        return model, None, unique_target_names

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=unique_target_names,
        inference_mode=False,
        bias="none",
        use_dora=bool(use_dora),
    )
    return get_peft_model(model, lora_config), lora_config, unique_target_names


def merge_all_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return model, 0
    trainable_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            trainable_count += 1
    merged_model = model.merge_and_unload()
    return merged_model, trainable_count


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
        decoder_param_ids = frozenset(int(v) for v in getattr(self, "distill_decoder_param_ids", frozenset()))
        if not decoder_param_ids:
            return super().create_optimizer()

        opt_model = getattr(self, "model_wrapped", None) or self.model
        if self.optimizer is None:
            decay_parameters = set(self.get_decay_parameter_names(opt_model))
            nondecoder_decay = []
            nondecoder_no_decay = []
            decoder = []
            trainable_ids = set()

            for name, param in opt_model.named_parameters():
                if not bool(param.requires_grad):
                    continue
                param_id = id(param)
                trainable_ids.add(param_id)
                if param_id in decoder_param_ids:
                    decoder.append(param)
                elif name in decay_parameters:
                    nondecoder_decay.append(param)
                else:
                    nondecoder_no_decay.append(param)

            grouped_ids = (
                {id(param) for param in nondecoder_decay}
                | {id(param) for param in nondecoder_no_decay}
                | {id(param) for param in decoder}
            )
            group_lengths = len(nondecoder_decay) + len(nondecoder_no_decay) + len(decoder)
            if grouped_ids != trainable_ids or group_lengths != len(grouped_ids):
                raise RuntimeError("Distill optimizer grouping produced duplicate or missing trainable parameters.")
            missing_decoder = decoder_param_ids - trainable_ids
            if missing_decoder:
                raise RuntimeError(
                    "Decoder optimizer group contains ids that are not trainable model parameters: "
                    + ",".join(str(v) for v in sorted(missing_decoder))
                )

            optimizer_grouped_parameters = []
            if nondecoder_decay:
                optimizer_grouped_parameters.append(
                    {
                        "group_name": "nondecoder_decay",
                        "params": nondecoder_decay,
                        "weight_decay": self.args.weight_decay,
                    }
                )
            if nondecoder_no_decay:
                optimizer_grouped_parameters.append(
                    {
                        "group_name": "nondecoder_no_decay",
                        "params": nondecoder_no_decay,
                        "weight_decay": 0.0,
                    }
                )
            if decoder:
                optimizer_grouped_parameters.append(
                    {
                        "group_name": "decoder",
                        "params": decoder,
                        "lr": float(self.distill_decoder_lr),
                        "weight_decay": 0.0,
                    }
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info("skipped %s: %sM params", module, skipped / 2**20)
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug("bitsandbytes: will optimize %s in fp32", module)
                logger.info("skipped: %sM params", skipped / 2**20)
        return self.optimizer


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
            loss_type: str = "r_kl_top_1000",
            temperature: float = 1.0,
            loss_alpha: float = 0.5,
            hidden_loss_weight: float = 0.0,
            pre_mlp_hidden_loss_weight: float = 0.0,
            prompt_kd_weight: float = 0.0,
            hidden_alignment_layer_weighting: str = "uniform",
            eakld_confidence_k: int = 16,
            teacher_logits_cpu_staging: bool = False,
            selective_student_topk: bool = False,
            selective_student_topk_chunk_rows: int = 32,
            selective_teacher_topk_chunk_tokens: int = 8,
            distill_hif4_act_controller: Optional[Hif4ActController] = None,
            teacher_runtime: Optional[DistillTeacherRuntime] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.loss_type = str(loss_type).strip().lower()
            self.temperature = float(temperature)
            self.loss_alpha = float(loss_alpha)
            self.hidden_loss_weight = float(hidden_loss_weight)
            if self.hidden_loss_weight < 0.0:
                raise ValueError(f"hidden_loss_weight must be >= 0, got {self.hidden_loss_weight}.")
            self.pre_mlp_hidden_loss_weight = float(pre_mlp_hidden_loss_weight)
            if self.pre_mlp_hidden_loss_weight < 0.0:
                raise ValueError(
                    f"pre_mlp_hidden_loss_weight must be >= 0, got {self.pre_mlp_hidden_loss_weight}."
                )
            self.prompt_kd_weight = float(prompt_kd_weight)
            if self.prompt_kd_weight < 0.0:
                raise ValueError(f"prompt_kd_weight must be >= 0, got {self.prompt_kd_weight}.")
            self.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                hidden_alignment_layer_weighting
            )
            self.eakld_confidence_k = int(eakld_confidence_k)
            if self.eakld_confidence_k < 2:
                raise ValueError(f"eakld_confidence_k must be >= 2, got {self.eakld_confidence_k}.")
            self.teacher_logits_cpu_staging = bool(teacher_logits_cpu_staging)
            self.selective_student_topk = bool(selective_student_topk)
            self.selective_student_topk_chunk_rows = int(selective_student_topk_chunk_rows)
            self.selective_teacher_topk_chunk_tokens = int(selective_teacher_topk_chunk_tokens)
            if self.selective_student_topk_chunk_rows < 1:
                raise ValueError("selective_student_topk_chunk_rows must be >= 1.")
            if self.selective_teacher_topk_chunk_tokens < 1:
                raise ValueError("selective_teacher_topk_chunk_tokens must be >= 1.")
            if self.selective_student_topk and not is_selective_student_topk_loss(self.loss_type):
                raise ValueError("selective_student_topk only supports loss_type=kl_top[_K].")
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
                        k=parse_selective_student_topk_k(self.loss_type),
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
            args = self.args
            loss_type = self.loss_type
            if bool(getattr(model, "training", False)):
                original_labels = inputs.get("labels")
                if isinstance(original_labels, torch.Tensor):
                    self.distill_token_stats.update(
                        original_labels, inputs.get("attention_mask")
                    )
            hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
            pre_mlp_hidden_loss_enabled = float(self.pre_mlp_hidden_loss_weight) > 0.0
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
            uses_ce_loss = (
                loss_type == "kd"
                or loss_type == "dual_kd"
                or loss_type == "eakld_kd"
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            )
            if not uses_ce_loss:
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

            def parse_k(prefix: str, default_k: int = 1000) -> int:
                if loss_type == prefix:
                    return default_k
                suffix = loss_type[len(prefix):]
                if suffix.startswith("_"):
                    suffix = suffix[1:]
                if not suffix:
                    return default_k
                return max(1, int(suffix))

            def get_teacher_targets():
                return self._run_teacher_forward(
                    teacher_inputs=teacher_inputs,
                    need_logits=loss_type not in {"origin", "sft"},
                    need_output_hidden_states=need_teacher_output_hidden_states,
                    need_pre_mlp_hiddens=pre_mlp_hidden_loss_enabled,
                    student_pre_mlp_modules=pre_mlp_capture_modules,
                )

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

            def build_token_regions(reference_logits):
                return build_distill_token_regions(
                    labels=full_inputs.get("labels"),
                    attention_mask=full_inputs.get("attention_mask"),
                    reference_logits=reference_logits,
                )

            def combine_region_loss(loss_for_mask, regions):
                response_loss = loss_for_mask(regions.response_mask)
                if self.prompt_kd_weight == 0.0:
                    return response_loss
                prompt_loss = loss_for_mask(regions.prompt_mask)
                return response_loss + self.prompt_kd_weight * prompt_loss

            try:
                if loss_type in {"origin", "sft"}:
                    if hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
                        teacher_targets = get_teacher_targets()
                        prepare_student_path()
                        outputs = student_forward(full_inputs)
                        loss = add_hidden_alignment_loss(outputs["loss"], teacher_targets, outputs)
                        return (loss, outputs) if return_outputs else loss
                    try:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch,
                        )
                    except TypeError:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                        )

                prepare_student_path()

                if loss_type == "rkl":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_reverse_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            temperature=float(self.temperature),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_rkl":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_dual_rkl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kl":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_forward_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            temperature=float(self.temperature),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("r_kl_top"):
                    k = parse_k("r_kl_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_rkl_topk(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                            temperature=float(self.temperature),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_r_kl_top"):
                    k = parse_k("dual_r_kl_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_dual_rkl_topk_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kl_top"):
                    k = parse_k("kl_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    if self.selective_student_topk:
                        if teacher_targets.selective_topk is None:
                            raise RuntimeError("selective student top-k requires compact teacher targets.")
                        if int(teacher_targets.selective_topk.k) != int(k):
                            raise RuntimeError("selective teacher top-k K does not match loss K.")
                        input_tensor = next(
                            value for value in student_inputs.values() if torch.is_tensor(value)
                        )
                        selected_indices, selected_teacher_logits = move_teacher_topk_targets_to_device(
                            teacher_targets.selective_topk,
                            device=input_tensor.device,
                        )
                        prepare_student_path()
                        with selective_student_lm_head(
                            model,
                            teacher_topk_indices=selected_indices,
                            chunk_rows=self.selective_student_topk_chunk_rows,
                        ):
                            outputs = student_forward(student_inputs)
                        logits = outputs.logits
                        regions = build_token_regions(logits)
                        loss = combine_region_loss(
                            lambda mask: compute_selected_teacher_topk_kl(
                                student_selected_logits=logits,
                                teacher_topk_logits=selected_teacher_logits,
                                mask=mask,
                                temperature=float(self.temperature),
                            ),
                            regions,
                        )
                        loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                        return (loss, outputs) if return_outputs else loss
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_kl_topk(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                            temperature=float(self.temperature),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kd_top"):
                    k = parse_k("kd_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    regions = build_token_regions(logits)
                    distill_loss = combine_region_loss(
                        lambda mask: compute_kl_topk(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                            temperature=float(T),
                        ),
                        regions,
                    )
                    # T² is already applied inside compute_kl_topk.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "mse":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_masked_logit_mse_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kd":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    regions = build_token_regions(logits)
                    distill_loss = combine_region_loss(
                        lambda mask: compute_forward_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            temperature=float(T),
                        ),
                        regions,
                    )
                    # T² is already applied inside compute_forward_kl_loss.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kl":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_dual_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kl_top"):
                    k = parse_k("dual_kl_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_dual_kl_topk_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kd_top"):
                    k = parse_k("dual_kd_top", default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    ori_loss = outputs["loss"]
                    regions = build_token_regions(logits)
                    distill_loss = combine_region_loss(
                        lambda mask: compute_dual_kl_topk_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                        ),
                        regions,
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kd":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    ori_loss = outputs["loss"]
                    regions = build_token_regions(logits)
                    distill_loss = combine_region_loss(
                        lambda mask: compute_dual_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                        ),
                        regions,
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if is_eakld_top_loss(loss_type):
                    k = parse_eakld_top_k(loss_type, default_k=1000)
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_eakld_topk(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            k=k,
                            temperature=float(self.temperature),
                            confidence_k=int(self.eakld_confidence_k),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "eakld":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    regions = build_token_regions(logits)
                    loss = combine_region_loss(
                        lambda mask: compute_eakld(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            temperature=float(self.temperature),
                            confidence_k=int(self.eakld_confidence_k),
                        ),
                        regions,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "eakld_kd":
                    teacher_targets = get_teacher_targets()
                    ori_logits = teacher_targets.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    regions = build_token_regions(logits)
                    distill_loss = combine_region_loss(
                        lambda mask: compute_entropy_aware_kl_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            mask=mask,
                            temperature=float(T),
                            confidence_k=int(self.eakld_confidence_k),
                        ),
                        regions,
                    )
                    # T² is already applied inside compute_eakld.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_targets, outputs)
                    return (loss, outputs) if return_outputs else loss

                raise ValueError(
                    f"Unsupported lora loss type: {loss_type}. "
                    f"Supported: sft/origin, rkl, dual_rkl, kl, r_kl_top[_K], dual_r_kl_top[_K], "
                    f"kl_top[_K], kd_top[_K], eakld, eakld_kd, eakld_top[_K]/eakld_topk[_K], "
                    f"dual_kl, dual_kd, dual_kl_top[_K], dual_kd_top[_K], mse, kd."
                )
            finally:
                set_hif4_act_enabled(previous_hif4_enabled)

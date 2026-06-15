from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from dense_e2e_fintuning.trainer import _get_output_logits, compute_dense_loss_from_logits
from train_utils.distill_losses import build_distill_token_mask

_HIDDEN_LAYER_WEIGHTING_CHOICES = ("uniform", "linear_depth")


def build_vae_hidden_layer_weights(
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
    if mode == "uniform":
        return torch.ones(num_layers, device=device, dtype=dtype)
    if mode == "linear_depth":
        denom = max(num_layers - 1, 1)
        raw = 1.0 + torch.arange(num_layers, device=device, dtype=dtype) / float(denom)
        return raw / raw.mean()
    raise ValueError(
        f"Unsupported hidden layer weighting: {layer_weighting}. "
        f"Supported: {', '.join(_HIDDEN_LAYER_WEIGHTING_CHOICES)}."
    )


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


def compute_vae_hidden_alignment_loss(
    *,
    teacher_hidden_states,
    student_hidden_states,
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    loss_device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    if teacher_hidden_states is None or student_hidden_states is None:
        raise ValueError("Hidden states are required when VAE hidden alignment loss is enabled.")
    if len(teacher_hidden_states) != len(student_hidden_states):
        raise ValueError(
            "Teacher/student hidden state counts differ: "
            f"{len(teacher_hidden_states)} vs {len(student_hidden_states)}."
        )
    if len(teacher_hidden_states) <= 1:
        raise ValueError("Hidden states must include embedding output plus at least one transformer block output.")

    target_device = torch.device(loss_device)
    layer_losses = []
    for layer_idx, (teacher_hidden, student_hidden) in enumerate(
        zip(teacher_hidden_states[1:], student_hidden_states[1:])
    ):
        if tuple(teacher_hidden.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student hidden shape mismatch at block layer {layer_idx}: "
                f"{tuple(teacher_hidden.shape)} vs {tuple(student_hidden.shape)}."
            )
        teacher_hidden = teacher_hidden.detach().to(device=student_hidden.device)
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        layer_losses.append((numerator / (denominator + float(eps))).to(device=target_device))

    stacked = torch.stack(layer_losses)
    weights = build_vae_hidden_layer_weights(
        num_layers=len(layer_losses),
        layer_weighting=layer_weighting,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    return (stacked * weights).mean()


def _causal_lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(device=shift_logits.device).contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, int(shift_logits.shape[-1])),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class VAEDecoderE2ETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        post_attn: bool = False,
        hidden_loss_weight: float = 0.0,
        hidden_layer_weighting: str = "uniform",
        saved_tensor_offload=None,
        streaming_offload_manager=None,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.post_attn = bool(post_attn)
        self.hidden_loss_weight = float(hidden_loss_weight)
        if self.hidden_loss_weight < 0.0:
            raise ValueError(f"hidden_loss_weight must be >= 0, got {self.hidden_loss_weight}.")
        self.hidden_layer_weighting = str(hidden_layer_weighting).strip().lower()
        if self.hidden_layer_weighting not in _HIDDEN_LAYER_WEIGHTING_CHOICES:
            raise ValueError(
                f"Unsupported hidden_layer_weighting: {hidden_layer_weighting}. "
                f"Supported: {', '.join(_HIDDEN_LAYER_WEIGHTING_CHOICES)}."
            )
        self._teacher_device = None
        self.saved_tensor_offload = saved_tensor_offload
        self.streaming_offload_manager = streaming_offload_manager
        super().__init__(*args, **kwargs)

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def _compute_teacher_outputs(self, teacher_inputs: Dict[str, torch.Tensor], *, output_hidden_states: bool = False):
        if self.teacher_model is None:
            raise RuntimeError("当前 loss_type 需要 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=bool(output_hidden_states))

    def _add_hidden_alignment_loss(self, loss, teacher_outputs, student_outputs, inputs):
        if float(self.hidden_loss_weight) <= 0.0:
            return loss
        hidden_loss = compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher_outputs.hidden_states,
            student_hidden_states=student_outputs.hidden_states,
            attention_mask=inputs.get("attention_mask"),
            layer_weighting=self.hidden_layer_weighting,
            loss_device=loss.device,
        )
        return loss + float(self.hidden_loss_weight) * hidden_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        student_inputs = dict(inputs)
        student_inputs.pop("labels", None)
        hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context:
            outputs = model(**student_inputs, output_hidden_states=hidden_loss_enabled)
        logits = _get_output_logits(outputs)

        ce_loss = None
        if labels is not None:
            ce_loss = _causal_lm_cross_entropy(logits, labels)

        loss_type = self.loss_type
        if loss_type in {"sft", "origin"}:
            if ce_loss is None:
                raise ValueError(f"loss_type={loss_type} requires labels.")
            if hidden_loss_enabled:
                teacher_inputs = dict(inputs)
                teacher_inputs.pop("labels", None)
                teacher_outputs = self._compute_teacher_outputs(teacher_inputs, output_hidden_states=True)
                ce_loss = self._add_hidden_alignment_loss(ce_loss, teacher_outputs, outputs, inputs)
            return (ce_loss, outputs) if return_outputs else ce_loss

        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        teacher_outputs = self._compute_teacher_outputs(teacher_inputs, output_hidden_states=hidden_loss_enabled)
        teacher_logits = _get_output_logits(teacher_outputs).to(device=logits.device)
        token_mask = build_distill_token_mask(
            labels=labels,
            attention_mask=inputs.get("attention_mask"),
            reference_logits=logits,
        )
        loss = compute_dense_loss_from_logits(
            loss_type=loss_type,
            student_logits=logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            mask=token_mask,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            post_attn=self.post_attn,
        )
        loss = self._add_hidden_alignment_loss(loss, teacher_outputs, outputs, inputs)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)
        if self.streaming_offload_manager is not None:
            self.streaming_offload_manager.offload_all(synchronize=True)
        target_device = torch.device(self.args.device)
        if torch.is_tensor(loss) and loss.device != target_device:
            loss = loss.to(device=target_device)
        return loss

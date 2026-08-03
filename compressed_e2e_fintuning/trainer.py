import logging
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainerCallback

try:
    from transformers.trainer_callback import ProgressCallback
except Exception:  # pragma: no cover
    ProgressCallback = None

from e2e_common.dense_loss import compute_dense_loss_from_logits, get_output_logits
from train_utils.distill_losses import build_distill_token_mask
from train_utils.lora_training import (
    build_distill_hidden_layer_weights,
    compute_distill_hidden_alignment_loss,
    parse_distill_hidden_alignment_layer_weighting,
)


class E2ETrainerLogCallback(TrainerCallback):
    """Write Trainer metrics into the run FileHandler, matching LoRA distill logging."""

    def __init__(self, *, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", True)):
            return
        if not logs:
            return
        values = dict(logs)
        values.pop("total_flos", None)
        ordered_keys = (
            "loss",
            "distill_loss",
            "hidden_loss",
            "train_loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "epoch",
        )
        parts = []
        for key in ordered_keys:
            if key in values:
                parts.append(f"{key}={values.pop(key)}")
        for key in sorted(values):
            parts.append(f"{key}={values[key]}")
        if not parts:
            return
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            fn="",
            lno=0,
            msg="E2E train: step=%s %s",
            args=(str(getattr(state, "global_step", "unknown")), " ".join(parts)),
            exc_info=None,
        )
        for handler in list(getattr(self.logger, "handlers", [])):
            if not isinstance(handler, logging.FileHandler):
                continue
            if record.levelno < handler.level:
                continue
            handler.handle(record)


class _QuietProgressCallback(ProgressCallback if ProgressCallback is not None else object):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return


def replace_progress_log_callback(trainer):
    if ProgressCallback is None:
        return trainer
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(callback_handler, "callbacks", None)
    if not isinstance(callbacks, list):
        return trainer
    for idx, callback in enumerate(callbacks):
        if isinstance(callback, ProgressCallback) and not isinstance(callback, _QuietProgressCallback):
            callbacks[idx] = _QuietProgressCallback()
    return trainer


def build_vae_hidden_layer_weights(
    *,
    num_layers: int,
    layer_weighting: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return build_distill_hidden_layer_weights(
        num_layers=int(num_layers),
        layer_weighting=str(layer_weighting),
        device=device,
        dtype=dtype,
    )


def compute_vae_hidden_alignment_loss(
    *,
    teacher_hidden_states,
    student_hidden_states,
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    loss_device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    loss = compute_distill_hidden_alignment_loss(
        teacher_hidden_states=teacher_hidden_states,
        student_hidden_states=student_hidden_states,
        attention_mask=attention_mask,
        layer_weighting=str(layer_weighting),
        eps=float(eps),
    )
    return loss.to(device=torch.device(loss_device))


def _causal_lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(device=shift_logits.device).contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, int(shift_logits.shape[-1])),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_choice_scores_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"choice logits must have shape [B, C, L, V], got {tuple(logits.shape)}.")
    if labels.ndim != 3:
        raise ValueError(f"choice labels must have shape [B, C, L], got {tuple(labels.shape)}.")
    if tuple(logits.shape[:3]) != tuple(labels.shape):
        raise ValueError(
            f"choice logits/labels shape mismatch: {tuple(logits.shape[:3])} vs {tuple(labels.shape)}."
        )

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(device=shift_logits.device).contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * valid_mask.to(dtype=token_log_probs.dtype)).sum(dim=-1)


def compute_choice_kd_loss_from_scores(
    *,
    student_scores: torch.Tensor,
    teacher_scores: Optional[torch.Tensor],
    answer_index: torch.Tensor,
    loss_type: str,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if norm not in {"choice_kd", "choice_kd_ce"}:
        raise ValueError(f"Unsupported choice KD loss_type={loss_type!r}.")
    if teacher_scores is None:
        raise ValueError(f"loss_type={norm} requires teacher_scores.")
    if student_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError("student_scores and teacher_scores must have shape [B, C].")
    if tuple(student_scores.shape) != tuple(teacher_scores.shape):
        raise ValueError(
            f"student/teacher choice score shape mismatch: {tuple(student_scores.shape)} vs {tuple(teacher_scores.shape)}."
        )

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha <= 1, got {alpha}.")

    teacher_scores = teacher_scores.detach().to(device=student_scores.device)
    answer_index = answer_index.to(device=student_scores.device, dtype=torch.long)
    kd_loss = F.kl_div(
        F.log_softmax(student_scores.float() / temperature, dim=-1),
        F.softmax(teacher_scores.float() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    if norm == "choice_kd":
        return kd_loss
    ce_loss = F.cross_entropy(student_scores.float(), answer_index)
    return ce_loss * (1.0 - alpha) + kd_loss * alpha


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
        eakld_confidence_k: int = 16,
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
        self.eakld_confidence_k = int(eakld_confidence_k)
        if self.eakld_confidence_k < 2:
            raise ValueError(f"eakld_confidence_k must be >= 2, got {self.eakld_confidence_k}.")
        try:
            self.hidden_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                str(hidden_layer_weighting)
            )
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(
                    "--distill_hidden_alignment_layer_weighting",
                    "hidden_layer_weighting",
                )
            ) from exc
        self._teacher_device = None
        self.saved_tensor_offload = saved_tensor_offload
        self.streaming_offload_manager = streaming_offload_manager
        self._last_loss_parts: Dict[str, float] = {}
        super().__init__(*args, **kwargs)
        # Custom compute_loss returns token-mean losses. HF treats models with
        # forward(**kwargs) as accepting num_items_in_batch and then skips
        # dividing by gradient_accumulation_steps, which inflates logged loss
        # and gradients. Disable that path explicitly.
        self.model_accepts_loss_kwargs = False
        if self.model is not None:
            self.model.accepts_loss_kwargs = False

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def offload_teacher_to_cpu(self) -> Optional[torch.device]:
        """Move teacher off GPU for eval; returns previous device for restore."""
        if self.teacher_model is None:
            return None
        previous = self._teacher_device
        self.teacher_model.to("cpu")
        self._teacher_device = torch.device("cpu")
        return previous

    def restore_teacher_device(self, device: Optional[torch.device]) -> None:
        if device is None or self.teacher_model is None:
            return
        if device.type == "cpu":
            self.offload_teacher_to_cpu()
            return
        self._ensure_teacher_device(device)

    def _compute_teacher_outputs(self, teacher_inputs: Dict[str, torch.Tensor], *, output_hidden_states: bool = False):
        if self.teacher_model is None:
            raise RuntimeError("当前 loss_type 需要 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=bool(output_hidden_states))

    def _compute_hidden_alignment_loss(self, teacher_outputs, student_outputs, inputs, *, loss_device):
        return compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher_outputs.hidden_states,
            student_hidden_states=student_outputs.hidden_states,
            attention_mask=inputs.get("attention_mask"),
            layer_weighting=self.hidden_layer_weighting,
            loss_device=loss_device,
        )

    def _store_loss_parts(self, *, distill_loss: torch.Tensor, hidden_loss: Optional[torch.Tensor] = None) -> None:
        parts = {"distill_loss": float(distill_loss.detach().float().item())}
        if hidden_loss is not None:
            parts["hidden_loss"] = float(hidden_loss.detach().float().item())
        self._last_loss_parts = parts

    def log(self, logs, start_time=None):
        merged = dict(logs)
        for key, value in getattr(self, "_last_loss_parts", {}).items():
            merged.setdefault(key, value)
        return super().log(merged, start_time=start_time)

    def _compute_choice_kd_loss(self, model, inputs, return_outputs: bool):
        if float(self.hidden_loss_weight) > 0.0:
            raise ValueError("dataset_task=mcqa does not support hidden_loss_weight > 0.")
        if self.teacher_model is None:
            raise ValueError(f"loss_type={self.loss_type} requires teacher_model for MCQA choice KD.")

        choice_input_ids = inputs["choice_input_ids"]
        choice_attention_mask = inputs.get("choice_attention_mask")
        choice_labels = inputs["choice_labels"]
        answer_index = inputs["answer_index"]
        if choice_input_ids.ndim != 3:
            raise ValueError(f"choice_input_ids must have shape [B, C, L], got {tuple(choice_input_ids.shape)}.")
        batch_size, choice_count, seq_len = choice_input_ids.shape
        flat_student_inputs = {
            "input_ids": choice_input_ids.reshape(batch_size * choice_count, seq_len),
        }
        if choice_attention_mask is not None:
            flat_student_inputs["attention_mask"] = choice_attention_mask.reshape(batch_size * choice_count, seq_len)

        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context:
            outputs = model(**flat_student_inputs, output_hidden_states=False)
        student_logits = get_output_logits(outputs).reshape(batch_size, choice_count, seq_len, -1)

        flat_teacher_inputs = dict(flat_student_inputs)
        teacher_outputs = self._compute_teacher_outputs(flat_teacher_inputs, output_hidden_states=False)
        teacher_logits = get_output_logits(teacher_outputs).to(device=student_logits.device)
        teacher_logits = teacher_logits.reshape(batch_size, choice_count, seq_len, -1)

        student_scores = compute_choice_scores_from_logits(student_logits, choice_labels)
        teacher_scores = compute_choice_scores_from_logits(teacher_logits, choice_labels.to(device=teacher_logits.device))
        loss = compute_choice_kd_loss_from_scores(
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            answer_index=answer_index,
            loss_type=self.loss_type,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
        )
        self._store_loss_parts(distill_loss=loss)
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        del num_items_in_batch, kwargs
        if "choice_input_ids" in inputs:
            return self._compute_choice_kd_loss(model, inputs, return_outputs=bool(return_outputs))

        labels = inputs.get("labels")
        student_inputs = dict(inputs)
        student_inputs.pop("labels", None)
        # Custom distill losses ignore HF num_items_in_batch scaling.
        student_inputs.pop("num_items_in_batch", None)
        hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context:
            outputs = model(**student_inputs, output_hidden_states=hidden_loss_enabled)
        logits = get_output_logits(outputs)

        ce_loss = None
        if labels is not None:
            ce_loss = _causal_lm_cross_entropy(logits, labels)

        loss_type = self.loss_type
        if loss_type in {"sft", "origin"}:
            if ce_loss is None:
                raise ValueError(f"loss_type={loss_type} requires labels.")
            hidden_loss = None
            loss = ce_loss
            if hidden_loss_enabled:
                teacher_inputs = dict(inputs)
                teacher_inputs.pop("labels", None)
                teacher_inputs.pop("num_items_in_batch", None)
                teacher_outputs = self._compute_teacher_outputs(teacher_inputs, output_hidden_states=True)
                hidden_loss = self._compute_hidden_alignment_loss(
                    teacher_outputs,
                    outputs,
                    inputs,
                    loss_device=loss.device,
                )
                loss = loss + float(self.hidden_loss_weight) * hidden_loss
            self._store_loss_parts(distill_loss=ce_loss, hidden_loss=hidden_loss)
            return (loss, outputs) if return_outputs else loss

        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        teacher_inputs.pop("num_items_in_batch", None)
        teacher_outputs = self._compute_teacher_outputs(teacher_inputs, output_hidden_states=hidden_loss_enabled)
        teacher_logits = get_output_logits(teacher_outputs).to(device=logits.device)
        token_mask = build_distill_token_mask(
            labels=labels,
            attention_mask=inputs.get("attention_mask"),
            reference_logits=logits,
        )
        distill_loss = compute_dense_loss_from_logits(
            loss_type=loss_type,
            student_logits=logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            mask=token_mask,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            post_attn=self.post_attn,
            eakld_confidence_k=int(self.eakld_confidence_k),
        )
        hidden_loss = None
        loss = distill_loss
        if hidden_loss_enabled:
            hidden_loss = self._compute_hidden_alignment_loss(
                teacher_outputs,
                outputs,
                inputs,
                loss_device=distill_loss.device,
            )
            loss = loss + float(self.hidden_loss_weight) * hidden_loss
        self._store_loss_parts(distill_loss=distill_loss, hidden_loss=hidden_loss)
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

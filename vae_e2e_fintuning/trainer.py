from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from dense_e2e_fintuning.trainer import _get_output_logits, compute_dense_loss_from_logits
from train_utils.distill_losses import build_distill_token_mask


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
        saved_tensor_offload=None,
        streaming_offload_manager=None,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.post_attn = bool(post_attn)
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

    def _compute_teacher_outputs(self, teacher_inputs: Dict[str, torch.Tensor]):
        if self.teacher_model is None:
            raise RuntimeError("当前 loss_type 需要 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        student_inputs = dict(inputs)
        student_inputs.pop("labels", None)
        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context:
            outputs = model(**student_inputs)
        logits = _get_output_logits(outputs)

        ce_loss = None
        if labels is not None:
            ce_loss = _causal_lm_cross_entropy(logits, labels)

        loss_type = self.loss_type
        if loss_type in {"sft", "origin"}:
            if ce_loss is None:
                raise ValueError(f"loss_type={loss_type} requires labels.")
            return (ce_loss, outputs) if return_outputs else ce_loss

        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        teacher_outputs = self._compute_teacher_outputs(teacher_inputs)
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

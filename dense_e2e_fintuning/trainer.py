import os
import warnings
from typing import Dict, Optional

import torch
from torch import nn
from transformers import Trainer, TrainerCallback
from transformers.trainer import SCHEDULER_NAME, reissue_pt_warnings, save_fsdp_optimizer

from e2e_common.dense_loss import compute_dense_loss_from_logits, get_output_logits
from train_utils.distill_losses import build_distill_token_mask
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.hif4_act import build_hif4_act_controller


class DenseAdaLoraCallback(TrainerCallback):
    def __init__(self, trainer: "_DenseLossMixin"):
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = self.trainer._unwrap_student_model()
        updater = getattr(model, "update_and_allocate", None)
        if callable(updater):
            updater(global_step=int(state.global_step) + 1)
        return control


class _DenseLossMixin:
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        post_attn: bool = False,
        lora_hif4_act: bool = False,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.post_attn = bool(post_attn)
        self.lora_hif4_act = bool(lora_hif4_act)
        self.lora_hif4_act_controller = build_hif4_act_controller(self.lora_hif4_act)
        self._teacher_device = None
        super().__init__(*args, **kwargs)

    def _unwrap_student_model(self):
        model = self.model
        if getattr(self, "accelerator", None) is not None:
            model = self.accelerator.unwrap_model(model)
        return model

    def _set_hif4_act_enabled(self, enabled: bool) -> None:
        if self.lora_hif4_act_controller is not None:
            self.lora_hif4_act_controller.enabled = bool(enabled)

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def _compute_teacher_outputs(self, teacher_inputs: Dict[str, torch.Tensor]):
        self._set_hif4_act_enabled(False)
        if self.teacher_model is None:
            raise RuntimeError("当前 loss_type 需要 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        previous_hif4_enabled = bool(getattr(self.lora_hif4_act_controller, "enabled", False))
        loss_type = self.loss_type
        try:
            if loss_type in {"sft", "origin"}:
                try:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                        **kwargs,
                    )
                except TypeError:
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            if not (
                loss_type in {"kd", "dual_kd"}
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            ):
                student_inputs.pop("labels", None)
            full_inputs = dict(inputs)

            teacher_outputs = self._compute_teacher_outputs(teacher_inputs)
            self._set_hif4_act_enabled(previous_hif4_enabled)
            if (
                loss_type in {"kd", "dual_kd"}
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            ):
                outputs = model(**full_inputs)
                ce_loss = outputs["loss"]
            else:
                outputs = model(**student_inputs)
                ce_loss = None
            logits = get_output_logits(outputs)
            token_mask = build_distill_token_mask(
                labels=inputs.get("labels"),
                attention_mask=inputs.get("attention_mask"),
                reference_logits=logits,
            )

            loss = compute_dense_loss_from_logits(
                loss_type=loss_type,
                student_logits=logits,
                teacher_logits=get_output_logits(teacher_outputs),
                ce_loss=ce_loss,
                mask=token_mask,
                temperature=self.distill_temperature,
                alpha=self.distill_alpha,
                post_attn=self.post_attn,
            )
            return (loss, outputs) if return_outputs else loss
        finally:
            self._set_hif4_act_enabled(previous_hif4_enabled)


class DenseFinetuneTrainer(_DenseLossMixin, Trainer):
    pass


class DenseFSDPFinetuneTrainer(_DenseLossMixin, FSDPTrainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        if self.args.should_save:
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                output_dir,
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

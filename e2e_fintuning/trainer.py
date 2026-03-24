import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from e2e_fintuning.lora import LoRAVAELinear, iter_named_vae_module_refs
from train_utils.distill_losses import (
    build_distill_token_mask,
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
)
from train_utils.fsdp_trainer import FSDPTrainer


def _iter_named_temporary_modules(model: nn.Module):
    skip_prefixes = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, LoRAVAELinear):
            skip_prefixes.append(f"{name}.base_layer")
        if callable(getattr(module, "set_temporary", None)):
            yield name, module


def model_requires_external_teacher(model: nn.Module) -> bool:
    for ref in iter_named_vae_module_refs(model):
        if ref.base_layer.original_weight is None:
            return True
    return False


class TemporaryTeacherState:
    def __init__(self, model: nn.Module):
        pairs = list(_iter_named_temporary_modules(model))
        self._modules = [module for _name, module in pairs]
        self._previous = [getattr(module, "temporary", None) for module in self._modules]

    def __enter__(self):
        return self

    def set_teacher_mode(self) -> None:
        for module in self._modules:
            module.set_temporary(False)

    def set_student_mode(self) -> None:
        for module in self._modules:
            module.set_temporary(True)

    def restore(self) -> None:
        for module, previous in zip(self._modules, self._previous):
            module.set_temporary(True if previous is None else bool(previous))

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def set_model_temporary(model: nn.Module, temporary: bool) -> None:
    for _name, module in _iter_named_temporary_modules(model):
        module.set_temporary(bool(temporary))


def _get_output_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    raise AttributeError("Model outputs do not contain `logits`.")


def compute_e2e_loss_from_logits(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    post_attn: bool = False,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if norm in {"sft", "origin"}:
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        return ce_loss
    if teacher_logits is None:
        raise ValueError(f"loss_type={norm} requires teacher_logits.")

    if norm == "rkl":
        return F.kl_div(
            F.log_softmax(teacher_logits.flatten(0, -2), dim=-1),
            F.softmax(student_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_rkl":
        return compute_dual_rkl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm == "kl":
        return F.kl_div(
            F.log_softmax(student_logits.flatten(0, -2), dim=-1),
            F.softmax(teacher_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_kl":
        return compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm.startswith("r_kl_top"):
        k = _parse_topk(norm, prefix="r_kl_top", default_k=1000)
        k = min(int(k), int(student_logits.shape[-1]))
        top_student, indices = student_logits.topk(k, dim=-1, sorted=False)
        top_teacher = teacher_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_teacher.flatten(0, -2), dim=-1),
            F.softmax(top_student.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm.startswith("dual_r_kl_top"):
        k = _parse_topk(norm, prefix="dual_r_kl_top", default_k=1000)
        return compute_dual_rkl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm.startswith("kl_top"):
        k = _parse_topk(norm, prefix="kl_top", default_k=1000)
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        if bool(post_attn):
            ref = F.softmax(teacher_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            return F.kl_div(can, ref, reduction="batchmean")
        top_student = student_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_student.flatten(0, -2), dim=-1),
            F.softmax(top_teacher.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm.startswith("kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = _parse_topk(norm, prefix="kd_top", default_k=1000)
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        temperature = float(temperature)
        if bool(post_attn):
            ref = F.softmax(teacher_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            kd_loss = F.kl_div(can, ref, reduction="batchmean")
        else:
            top_student = student_logits.gather(-1, indices)
            kd_loss = F.kl_div(
                F.log_softmax((top_student / temperature).flatten(0, -2), dim=-1),
                F.softmax((top_teacher / temperature).flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
    if norm.startswith("dual_kl_top"):
        k = _parse_topk(norm, prefix="dual_kl_top", default_k=1000)
        return compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm == "mse":
        return F.mse_loss(student_logits, teacher_logits)
    if norm == "kd":
        if ce_loss is None:
            raise ValueError("loss_type=kd requires ce_loss.")
        temperature = float(temperature)
        kd_loss = F.kl_div(
            F.log_softmax((student_logits / temperature).flatten(0, -2), dim=-1),
            F.softmax((teacher_logits / temperature).flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
    if norm == "dual_kd":
        if ce_loss is None:
            raise ValueError("loss_type=dual_kd requires ce_loss.")
        kd_loss = compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)
    if norm.startswith("dual_kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = _parse_topk(norm, prefix="dual_kd_top", default_k=1000)
        kd_loss = compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)

    raise ValueError(
        f"Unsupported e2e loss type: {loss_type}. "
        "Supported: sft/origin, kl, rkl, dual_rkl, mse, kd, kd_top[_K], dual_kd_top[_K], "
        "dual_kl, dual_kd, kl_top[_K], r_kl_top[_K], dual_r_kl_top[_K], dual_kl_top[_K]."
    )


def _parse_topk(value: str, *, prefix: str, default_k: int) -> int:
    if value == prefix:
        return int(default_k)
    suffix = value[len(prefix):]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return int(default_k)
    return max(1, int(suffix))


class _E2ELossMixin:
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        prewarm_frozen_vae: bool = True,
        prewarm_log_every: int = 32,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.prewarm_frozen_vae = bool(prewarm_frozen_vae)
        self.prewarm_log_every = max(1, int(prewarm_log_every))
        self._teacher_device = None
        self._vae_cache_prepared = False
        self._logger = logging.getLogger("e2e_fintuning")
        super().__init__(*args, **kwargs)

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def _infer_cache_dtype(self, model: nn.Module) -> torch.dtype:
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def prepare_frozen_vae_cache_once(self, model: nn.Module) -> None:
        if self._vae_cache_prepared or not self.prewarm_frozen_vae:
            return

        target_dtype = self._infer_cache_dtype(model)
        total = 0
        warmed = 0
        skipped = 0
        failed = 0
        for index, ref in enumerate(iter_named_vae_module_refs(model), start=1):
            total += 1
            base_layer = ref.base_layer
            if not bool(getattr(base_layer, "cache_decoded_weight", True)):
                skipped += 1
            else:
                try:
                    base_layer.clear_decoded_weight_cache()
                    if base_layer.prime_decoded_weight_cache(dtype=target_dtype):
                        warmed += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    self._logger.warning("Failed to prewarm VAELinear cache for %s: %s", ref.name, exc)
            if index % self.prewarm_log_every == 0:
                self._logger.info(
                    "VAELinear prewarm progress: processed=%d warmed=%d skipped=%d failed=%d",
                    index,
                    warmed,
                    skipped,
                    failed,
                )

        self._vae_cache_prepared = True
        self._logger.info(
            "VAELinear prewarm complete: total=%d warmed=%d skipped=%d failed=%d dtype=%s",
            total,
            warmed,
            skipped,
            failed,
            str(target_dtype),
        )

    def _compute_teacher_outputs(self, model, teacher_inputs: Dict[str, torch.Tensor]):
        if self.teacher_model is not None:
            input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
            self._ensure_teacher_device(device=input_tensor.device)
            with torch.no_grad():
                return self.teacher_model(**teacher_inputs, output_hidden_states=False)

        unwrapped_model = model
        if getattr(self, "accelerator", None) is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        with TemporaryTeacherState(unwrapped_model) as state:
            state.set_teacher_mode()
            with torch.no_grad():
                teacher_outputs = model(**teacher_inputs, output_hidden_states=False)
            return teacher_outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        unwrapped_model = model
        if getattr(self, "accelerator", None) is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        self.prepare_frozen_vae_cache_once(unwrapped_model)

        loss_type = self.loss_type
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

        teacher_outputs = self._compute_teacher_outputs(model, teacher_inputs)
        if self.teacher_model is None:
            set_model_temporary(unwrapped_model, True)
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
        logits = _get_output_logits(outputs)
        token_mask = build_distill_token_mask(
            labels=inputs.get("labels"),
            attention_mask=inputs.get("attention_mask"),
            reference_logits=logits,
        )

        loss = compute_e2e_loss_from_logits(
            loss_type=loss_type,
            student_logits=logits,
            teacher_logits=_get_output_logits(teacher_outputs),
            ce_loss=ce_loss,
            mask=token_mask,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            post_attn=bool(getattr(self.args, "post_attn", False)),
        )
        return (loss, outputs) if return_outputs else loss


class E2EFinetuneTrainer(_E2ELossMixin, Trainer):
    pass


class E2EFSDPFinetuneTrainer(_E2ELossMixin, FSDPTrainer):
    pass

from typing import Optional, Union

import torch
from torch import nn

from train_utils.base_reference import load_frozen_base_reference_model


def distill_loss_requires_teacher(loss_type: str) -> bool:
    normalized = str(loss_type or "").strip().lower()
    return normalized not in {"", "none", "sft", "origin"}


def resolve_distill_teacher_required(
    *,
    loss_type: str,
    hidden_loss_weight: float,
    pre_mlp_hidden_loss_weight: float,
) -> bool:
    return (
        distill_loss_requires_teacher(loss_type)
        or float(hidden_loss_weight) > 0.0
        or float(pre_mlp_hidden_loss_weight) > 0.0
    )


def resolve_distill_teacher_dtype(training_args, student_model: nn.Module) -> torch.dtype:
    if bool(getattr(training_args, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(training_args, "fp16", False)):
        return torch.float16
    for param in student_model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


class DistillTeacherRuntime:
    def __init__(
        self,
        *,
        model_path: str,
        access_token: Optional[str],
        forward_device: Union[str, torch.device],
        dtype: torch.dtype,
        model_offload: str,
        logger,
    ):
        mode = str(model_offload).strip().lower()
        if mode not in {"none", "cpu"}:
            raise ValueError("model_offload must be one of: none, cpu.")
        self.model_path = str(model_path)
        self.access_token = access_token
        self.forward_device = torch.device(forward_device)
        self.dtype = dtype
        self.model_offload = mode
        self.logger = logger
        self._model: Optional[nn.Module] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_or_load(self) -> nn.Module:
        if self._model is None:
            initial_device = "cpu" if self.model_offload == "cpu" else self.forward_device
            if self.logger is not None:
                self.logger.info(
                    "Loading independent distill teacher: model_path=%s offload=%s initial_device=%s dtype=%s",
                    self.model_path,
                    self.model_offload,
                    str(initial_device),
                    str(self.dtype),
                )
            self._model = load_frozen_base_reference_model(
                self.model_path,
                access_token=self.access_token,
                device=initial_device,
                dtype=self.dtype,
            )
        return self._model

    def prepare_for_forward(self) -> nn.Module:
        model = self.get_or_load()
        if self.model_offload == "cpu":
            model.to(self.forward_device)
        return model

    def finish_forward(self) -> None:
        if self.model_offload == "cpu" and self._model is not None:
            self._model.to("cpu")

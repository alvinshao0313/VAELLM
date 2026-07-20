from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporarySwitchLinear(nn.Module):
    """Frozen linear that switches student/teacher weights via set_temporary.

    temporary=True  -> student_weight (decoded / compressed path)
    temporary=False -> teacher_weight (original path)

    teacher_weight may be the same nn.Parameter object as an OriginalWeightBank entry
    so the full model keeps only one copy of each original matrix on GPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        student_weight: torch.Tensor,
        teacher_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.temporary = True

        student_param = student_weight if isinstance(student_weight, nn.Parameter) else nn.Parameter(student_weight)
        student_param.requires_grad = False
        if tuple(student_param.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"student_weight shape {tuple(student_param.shape)} != "
                f"({self.out_features}, {self.in_features})"
            )
        self.register_parameter("student_weight", student_param)

        teacher_param = teacher_weight if isinstance(teacher_weight, nn.Parameter) else nn.Parameter(teacher_weight)
        teacher_param.requires_grad = False
        if tuple(teacher_param.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"teacher_weight shape {tuple(teacher_param.shape)} != "
                f"({self.out_features}, {self.in_features})"
            )
        self.register_parameter("teacher_weight", teacher_param)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            bias_param = bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias.detach().clone())
            bias_param.requires_grad = False
            if tuple(bias_param.shape) != (self.out_features,):
                raise ValueError(f"bias shape {tuple(bias_param.shape)} != ({self.out_features},)")
            self.register_parameter("bias", bias_param)

        self.requires_grad_(False)
        self.eval()

    def set_temporary(self, temporary: bool = True) -> None:
        self.temporary = bool(temporary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.student_weight if bool(self.temporary) else self.teacher_weight
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

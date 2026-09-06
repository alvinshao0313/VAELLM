from __future__ import annotations

import torch
from torch import nn

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from train_utils.hif4_act import Hif4ActController, register_hif4_act_hooks, remove_hif4_act_hooks


class _OneLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(3))

    def forward(self, x):
        return self.proj(x)


def test_distill_hif4_controller_is_student_only_and_scoped_to_student_forward():
    student = _OneLinear().eval()
    teacher = _OneLinear().eval()
    calls = []

    def quantize(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.detach().clone())
        return x + 2.0

    controller = Hif4ActController(quantize)
    handles = register_hif4_act_hooks(student, controller)
    assert handles

    trainer = object.__new__(VAEDecoderE2ETrainer)
    trainer.distill_hif4_act_controller = controller

    x = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.no_grad():
        student_plain = student(x)
        teacher_before = teacher(x)
    assert not calls
    assert controller.enabled is False

    with trainer._student_hif4_act_context():
        with torch.no_grad():
            student_quantized = student(x)
            teacher_during = teacher(x)
        assert controller.enabled is True

    assert controller.enabled is False
    assert len(calls) == 1
    torch.testing.assert_close(student_plain, x)
    torch.testing.assert_close(student_quantized, x + 2.0)
    torch.testing.assert_close(teacher_before, x)
    torch.testing.assert_close(teacher_during, x)

    with torch.no_grad():
        student_after = student(x)
    torch.testing.assert_close(student_after, x)
    assert len(calls) == 1

    remove_hif4_act_hooks(handles)

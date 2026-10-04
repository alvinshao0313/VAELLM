"""Coordinate checks for the native fixed-decoder BlockLDLQ adapter."""

import pytest
import torch

from experiments.qwen3_w25.block_ldlq_model import rotate_hessian, rotate_input_rows
from rotation.weight_preprocess import build_weight_rotation, rotation_spec


@pytest.mark.parametrize("shape,block", [((16, 32), 8), ((24, 48), 12)])
def test_actual_rotation_signs_hessian_axis_and_quadratic_equivalence(shape, block):
    torch.manual_seed(19)
    rotation = build_weight_rotation(*shape, rotation_spec("two_sided", block, 31, "down"))
    # Independent dense V comes from the transformation of each input basis
    # vector. Non-Sylvester coverage detects accidental inverse/transpose swaps.
    identity = torch.eye(shape[1], dtype=torch.float64)
    v_t = rotate_input_rows(identity, rotation)
    a = torch.randn(shape[1], shape[1], dtype=torch.float64)
    h = a @ a.T + 0.01 * identity
    transformed = rotate_hessian(h, rotation)
    torch.testing.assert_close(transformed, v_t.T @ h @ v_t, atol=1e-11, rtol=1e-12)
    e = torch.randn(*shape, dtype=torch.float64)
    er = rotation(e)
    torch.testing.assert_close(((er @ transformed) * er).sum(), ((e @ h) * e).sum(), atol=1e-9, rtol=1e-12)
    torch.testing.assert_close(rotation(er, inverse=True), e, atol=1e-12, rtol=1e-12)


def test_hessian_rotation_rejects_wrong_axis_and_precision():
    rotation = build_weight_rotation(16, 32, rotation_spec("two_sided", 8, 31, "down"))
    for value in (torch.eye(32), torch.eye(16, dtype=torch.float64)):
        with pytest.raises(ValueError, match="FP64"):
            rotate_input_rows(value, rotation)

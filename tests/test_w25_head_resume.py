"""Private resume boundaries and exact state comparison, without a GPU/model load."""
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerControl

from experiments.w25_eval.resume_head_only import ResumeBoundaryCallback
from experiments.w25_eval.validate_head_resume import assert_tensor_equal, fixed_head_probe_indices, tensor_hash


def test_resume_absolute_boundary_preserves_the_original_schedule():
    callback = ResumeBoundaryCallback(400, 800)
    args = SimpleNamespace(max_steps=2000, get_warmup_steps=lambda steps: 20)
    control = TrainerControl()
    callback.on_train_begin(args, SimpleNamespace(global_step=400), control)
    assert not callback.on_step_end(args, SimpleNamespace(global_step=799), control).should_training_stop
    control = callback.on_step_end(args, SimpleNamespace(global_step=800), control)
    assert control.should_training_stop and control.should_save
    with pytest.raises(ValueError, match='exceeded'):
        callback.on_step_end(args, SimpleNamespace(global_step=801), control)
    with pytest.raises(ValueError, match='stop flag'):
        callback.on_train_begin(args, SimpleNamespace(global_step=400), control)
    with pytest.raises(ValueError, match='restored step'):
        callback.on_train_begin(args, SimpleNamespace(global_step=0), TrainerControl())


def test_diagnostic_stop_404_does_not_rebase_global_steps():
    callback = ResumeBoundaryCallback(400, 404)
    for step in (401, 402, 403):
        assert not callback.on_step_end(None, SimpleNamespace(global_step=step), TrainerControl()).should_training_stop
    assert callback.on_step_end(None, SimpleNamespace(global_step=404), TrainerControl()).should_save


def test_exact_restore_comparison_rejects_dtype_shape_and_value_loss():
    value = torch.tensor([.10001, .20003], dtype=torch.float32)
    assert_tensor_equal(value.clone(), value, 'head')
    for changed in (value.bfloat16(), value.reshape(1, 2), value + .001):
        with pytest.raises(ValueError):
            assert_tensor_equal(changed, value, 'head')
    assert tensor_hash(value.clone()) == tensor_hash(value)
    assert tensor_hash(value.reshape(1, 2)) != tensor_hash(value)


def test_real_qwen_head_probe_indices_stay_inside_integer_storage():
    numel = 151936 * 4096
    indices = fixed_head_probe_indices(numel)
    assert indices.dtype == torch.int64 and indices.numel() == indices.unique().numel() == 64
    assert indices[0].item() == 0 and indices[-1].item() == numel - 1
    assert bool(((indices >= 0) & (indices < numel)).all())
    assert bool((indices[1:] > indices[:-1]).all())

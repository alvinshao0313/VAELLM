"""CPU wiring contracts; real H256/packed-bit numerics remain GPU validation."""
import json
from pathlib import Path

import pytest
import torch
from transformers import TrainerControl, TrainerState

from experiments.w25_eval.validate_wide_subset_training import TrainingObservations, configuration


def test_real_recipe_parser_changes_only_source_output_and_save_interval():
    path = Path("result/qwen3_8b_mmlu65_20261004/best1200_down0_h256_validate_s4/launch.json")
    record = json.loads(path.read_text())
    cfg, _, args, names = configuration(record)
    assert args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
    assert args.save_steps == 2 and args.per_device_train_batch_size == 4
    assert cfg.bit_active_ratio == 1 and int(cfg.bit_round_steps) == 512
    assert names == ("model.layers.0.mlp.down_proj",)
    bad = dict(record, training_argv=list(record["training_argv"]))
    bad["training_argv"][bad["training_argv"].index("--decoder_lr") + 1] = "2e-5"
    with pytest.raises((ValueError, RuntimeError), match="scientific argv"):
        configuration(bad)


def test_observation_callback_preserves_four_real_optimizer_updates():
    callback = TrainingObservations(("model.layers.0.mlp.down_proj",))
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    state, control = TrainerState(), TrainerControl()
    for step in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        optimizer.step()
        state.global_step = step
        callback.on_step_end(None, state, control, optimizer=optimizer)
        assert control.should_training_stop == (step == 4)
    assert [x["step"] for x in callback.diagnostic["optimizer_steps"]] == [1, 2, 3, 4]
    assert all(x["main_moment_dtypes"] == ["torch.float32"]
               for x in callback.diagnostic["optimizer_steps"])
    assert not hasattr(callback, "env")
    assert not torch.cuda.is_initialized()

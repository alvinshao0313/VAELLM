"""Check the fresh pipeline's data, precision and schedule boundaries on CPU."""
import pytest

from compressed_e2e_fintuning.args import parse_args
from experiments.qwen3_w25.reproduce_from_bf16 import environment, fast_training_command
from experiments.qwen3_w25.reproduction_recipe import MIXED_DATA, TASK_MIX, training_command
from experiments.qwen3_w25.train_reproduction_stage import validate_configuration


@pytest.mark.parametrize("recipe,loss,alpha,mix,prompt,finish", [
    ("ce10", "kd_top_mass", .9, MIXED_DATA, .3, 2000),
    ("ce50", "kd_top_mass", .5, TASK_MIX, 0., 2000),
    ("decoder", "kl_top_mass", None, MIXED_DATA, .3, 2000),
    ("last_joint", "kd_top_mass", .5, TASK_MIX, 0., 1200),
])
def test_fast_stage_keeps_mathematical_recipe_and_original_schedule(tmp_path, recipe, loss, alpha, mix, prompt, finish):
    original = training_command(recipe, source_dir=tmp_path / "source", hf_model=tmp_path / "hf",
                                output_dir=tmp_path / "run", export_gpu="1")
    command = fast_training_command(recipe, source=tmp_path / "source", model=tmp_path / "hf",
                                    output=tmp_path / "run", gpu="1")
    # Only the execution wrapper and endpoint differ; keep every original native CLI token.
    expected = original[3:]
    index = expected.index("--export_cuda_visible_devices")
    del expected[index:index + 2]
    assert command[3:-2] == expected
    assert command[-2:] == ["--finish_step", str(finish)]
    assert command[command.index("--dataset_mix") + 1] == mix
    cfg, _hf, args = parse_args(command[3:-2] + ["--use_cpu", "true", "--report_to", "none"])
    validate_configuration(cfg, args, finish)
    assert cfg.loss.loss_type == loss and cfg.loss.prompt_loss_weight == prompt
    if alpha is not None:
        assert cfg.loss.alpha == alpha
    assert args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
    assert args.save_steps == 400 and args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 1 and cfg.data.model_max_length == 1024


def test_format_attention_override_is_scoped_to_its_child():
    paired = environment("1", format_recovery=True)
    native = environment("0")
    assert paired["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert paired["MKL_NUM_THREADS"] == "4"
    assert "FLASH_ATTENTION_DETERMINISTIC" not in native
    assert "NVIDIA_TF32_OVERRIDE" not in native
    assert native["CUDA_VISIBLE_DEVICES"] == "0" and paired["CUDA_VISIBLE_DEVICES"] == "1"

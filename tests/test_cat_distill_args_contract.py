import pytest

from train_utils.cat_train_args import process_cat_train_args, resolve_distill_runtime_config


def _parse(args):
    return process_cat_train_args(args)


@pytest.mark.parametrize("mode", ["remaining_lora_decoder", "remaining_lora_all_decoder"])
def test_new_remaining_decoder_modes_are_accepted(mode):
    cat_args, _hf_args, _training_args, _vae_args = _parse(
        [
            "--distill_after_category",
            mode,
            "--distill_dataset",
            "wiki=1.0",
        ]
    )

    assert cat_args.distill_after_category == mode


def test_distill_decoder_lr_default_resolves_to_none():
    cat_args, _hf_args, _training_args, _vae_args = _parse([])
    cfg = resolve_distill_runtime_config(cat_args, after_category="gate_proj")

    assert cfg.decoder_lr is None


def test_distill_decoder_lr_resolves_after_category_override():
    cat_args, _hf_args, _training_args, _vae_args = _parse(
        [
            "--distill_decoder_lr",
            "default=5e-5,after:gate_proj=3e-5",
        ]
    )

    default_cfg = resolve_distill_runtime_config(cat_args, after_category=None)
    gate_cfg = resolve_distill_runtime_config(cat_args, after_category="gate_proj")
    down_cfg = resolve_distill_runtime_config(cat_args, after_category="down_proj")

    assert default_cfg.decoder_lr == pytest.approx(5e-5)
    assert gate_cfg.decoder_lr == pytest.approx(3e-5)
    assert down_cfg.decoder_lr == pytest.approx(5e-5)


@pytest.mark.parametrize("offload", ["none", "cpu"])
def test_distill_teacher_model_offload_accepts_supported_modes(offload):
    _cat_args, _hf_args, training_args, _vae_args = _parse(
        ["--distill_teacher_model_offload", offload]
    )

    assert training_args.distill_teacher_model_offload == offload


def test_distill_teacher_model_offload_rejects_disk():
    with pytest.raises(ValueError, match="distill_teacher_model_offload"):
        _parse(["--distill_teacher_model_offload", "disk"])


@pytest.mark.parametrize(
    "mode",
    ["remaining_lora", "remaining_lora_decoder", "remaining_lora_all_decoder"],
)
def test_final_norm_and_post_norm_are_allowed_for_remaining_family(mode):
    cat_args, _hf_args, _training_args, _vae_args = _parse(
        [
            "--distill_after_category",
            mode,
            "--distill_dataset",
            "wiki=1.0",
            "--distill_tune_final_norm",
            "true",
            "--distill_use_post_norm_head_linear",
            "true",
        ]
    )

    assert cat_args.distill_tune_final_norm is True
    assert cat_args.distill_use_post_norm_head_linear is True


@pytest.mark.parametrize("mode", ["decoder", "both", "compressed_lora"])
def test_final_norm_and_post_norm_are_rejected_for_compressed_modes(mode):
    with pytest.raises(ValueError, match="remaining-family"):
        _parse(
            [
                "--distill_after_category",
                mode,
                "--distill_dataset",
                "wiki=1.0",
                "--distill_tune_final_norm",
                "true",
                "--distill_use_post_norm_head_linear",
                "true",
            ]
        )


def test_cat_parser_no_longer_accepts_unload_vae_original_weights_on_final_save():
    with pytest.raises((SystemExit, ValueError)):
        _parse(["--unload_vae_original_weights_on_final_save"])

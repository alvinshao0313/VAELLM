import argparse
import math

import pytest

import tools.cat_train as cat_train_entry
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.config import (
    DELETED_CLI_FLAGS,
    LoRAConfig,
    parse_cat_cli,
    parse_e2e_cli,
    parse_hidden_layer_weighting,
    parse_target_layers,
    parse_target_modules,
    teacher_required,
    validate_lora_against_checkpoint,
    vae_num_warmup_steps,
)
from train_utils.config.targets import (
    parse_compression_categories,
    parse_skip_layers,
    validate_skip_layers_scope,
)


def _cat(extra):
    return parse_cat_cli(
        [
            "--model_path",
            "dummy-model",
            "--compression_categories",
            "q_proj,k_proj",
            "--target_layers",
            "all",
            *extra,
        ]
    )


def _e2e(extra):
    return parse_e2e_cli(
        [
            "--student_checkpoint_dir",
            "/tmp/student",
            "--dataset_mix",
            "openorca",
            "--train_mode",
            "decoder",
            *extra,
        ]
    )


def test_e2e_parses_new_model_level_cli_names():
    cfg = _e2e(
        [
            "--dataset_task",
            "lm",
            "--model_max_length",
            "1024",
            "--dynamic_padding",
            "true",
            "--seed",
            "42",
            "--data_seed",
            "7",
            "--loss_type",
            "kl_top",
            "--top_k",
            "100",
            "--temperature",
            "1.0",
            "--alpha",
            "0.5",
            "--prompt_loss_weight",
            "0.0",
            "--hidden_loss_weight",
            "0.1",
            "--pre_mlp_hidden_loss_weight",
            "0.001",
            "--hidden_layer_weighting",
            "linear_depth",
            "--selective_student_topk",
            "true",
            "--selective_student_topk_chunk_rows",
            "32",
            "--teacher_output_offload",
            "cpu",
            "--teacher_model_offload",
            "none",
            "--teacher_output_pin_memory",
            "true",
            "--teacher_output_chunk_tokens",
            "8",
            "--steps",
            "5000",
            "--batch_size",
            "8",
            "--learning_rate",
            "1e-5",
            "--decoder_lr",
            "1e-5",
            "--weight_decay",
            "0.001",
            "--gradient_accumulation_steps",
            "1",
            "--max_grad_norm",
            "1.5",
            "--warmup_ratio",
            "0.03",
            "--lr_scheduler_type",
            "cosine",
            "--optim",
            "adamw_torch",
            "--gradient_checkpointing",
            "true",
            "--lora_rank",
            "12",
            "--lora_alpha",
            "24",
            "--lora_dropout",
            "0.03",
            "--norm_train_mode",
            "final",
            "--norm_lr",
            "1e-5",
            "--lm_head_train_mode",
            "linear",
            "--lm_head_lr",
            "1e-5",
            "--target_layers",
            "0-35",
            "--target_modules",
            "all",
        ]
    )

    assert cfg.data.dataset_task == "lm"
    assert cfg.data.model_max_length == 1024
    assert cfg.loss.loss_type == "kl_top"
    assert cfg.loss.top_k == 100
    assert cfg.loss.prompt_loss_weight == 0.0
    assert cfg.opt.steps == 5000
    assert cfg.opt.batch_size == 8
    assert cfg.runtime.teacher_output_offload == "cpu"
    assert cfg.runtime.teacher_model_offload == "none"
    assert cfg.train_mode == "decoder"
    assert cfg.aux.norm_train_mode == "final"
    assert cfg.aux.lm_head_train_mode == "linear"
    assert cfg.lora.rank == 12
    assert cfg.lora.rank_explicit is True


@pytest.mark.parametrize(
    "flag",
    [
        "--distill_temperature",
        "--distill_alpha",
        "--distill_loss_alpha",
        "--distill_loss_type",
        "--distill_hidden_loss_weight",
        "--prompt_kd_weight",
        "--distill_prompt_kd_weight",
        "--eakld_confidence_k",
        "--distill_dataset",
        "--distill_model_max_length",
        "--compressed_lora_scope",
        "--lora_use_dora",
        "--decoder_layers",
        "--sparse_bit_tuning",
        "--vae_tune_bias",
        "--tune_final_norm",
        "--use_post_norm_head_linear",
        "--max_train_samples",
        "--parallel_stage_decode",
        "--packed_vq_decoder_linear",
        "--decode_device",
        "--decode_group_size",
        "--eval_lm_batch_size",
        "--finetune_mode",
    ],
)
def test_e2e_rejects_deleted_cli_names(flag):
    assert flag in DELETED_CLI_FLAGS
    with pytest.raises(SystemExit):
        _e2e([flag, "1"])


@pytest.mark.parametrize(
    "flag",
    [
        "--target_categories",
        "--distill_after_category",
        "--include_all_linears",
        "--steps_per_category",
        "--outlier_protect_mode",
        "--outlier_channel_scope",
        "--outlier_protect_count",
        "--distill_steps",
        "--distill_lr",
        "--lr_warmup_steps",
        "--lora_use_dora",
        "--compressed_lora_scope",
    ],
)
def test_cat_rejects_deleted_cli_names(flag):
    assert flag in DELETED_CLI_FLAGS
    with pytest.raises(SystemExit):
        _cat([flag, "1"])


def test_cat_parses_renamed_public_fields_and_resolves_overrides_to_scalars():
    cfg = _cat(
        [
            "--vae_steps",
            "default=10000,cat:k_proj=2000",
            "--codebook_bits",
            "default=16,cat:k_proj=24",
            "--channel_protect_mode",
            "channel",
            "--channel_scope",
            "layer",
            "--channel_protect_count",
            "default=0,cat:k_proj=8",
            "--after_category_mode",
            "current_lora",
            "--dataset_mix",
            "openorca",
            "--dataset_task",
            "sft",
            "--steps",
            "default=5000,after:k_proj=100",
            "--batch_size",
            "default=4,after:k_proj=2",
            "--learning_rate",
            "default=1e-4,after:k_proj=2e-4",
            "--loss_type",
            "default=kl_top,after:k_proj=kd",
            "--top_k",
            "default=100,after:k_proj=50",
            "--lora_rank",
            "default=12,after:k_proj=8",
        ]
    )

    q_vae, q_opt = cfg.resolve_category_config("q_proj")
    k_vae, k_opt = cfg.resolve_category_config("k_proj")
    assert q_opt.vae_steps == 10000
    assert k_opt.vae_steps == 2000
    assert q_vae.core.codebook_bits == 16
    assert k_vae.core.codebook_bits == 24
    assert q_vae.channel.channel_protect_count == 0
    assert k_vae.channel.channel_protect_count == 8
    assert isinstance(q_opt.vae_steps, int)
    assert isinstance(k_vae.core.codebook_bits, int)

    q_after = cfg.resolve_after_category_config("q_proj")
    k_after = cfg.resolve_after_category_config("k_proj")
    assert q_after.opt.steps == 5000
    assert k_after.opt.steps == 100
    assert q_after.loss.loss_type == "kl_top"
    assert k_after.loss.loss_type == "kd"
    assert q_after.loss.top_k == 100
    assert k_after.loss.top_k == 50
    assert q_after.lora.rank == 12
    assert k_after.lora.rank == 8
    assert not hasattr(q_after.loss, "by_after_category")
    assert not hasattr(k_after.opt, "by_category")


def test_e2e_public_fields_reject_cat_override_strings():
    with pytest.raises(SystemExit):
        _e2e(["--steps", "default=5000"])


def test_dataset_mix_shorthand_equals_explicit_weight():
    shorthand = _e2e(["--dataset_mix", "openorca"])
    explicit = _e2e(["--dataset_mix", "openorca=1.0"])
    assert shorthand.data.dataset_mix == explicit.data.dataset_mix


def test_dataset_mix_normalizes_unequal_weights():
    cfg = _e2e(["--dataset_mix", "openorca=1,wiki=3"])
    assert cfg.data.dataset_mix_sources == ("openorca", "wiki")
    assert cfg.data.dataset_mix_weights[0] == pytest.approx(0.25)
    assert cfg.data.dataset_mix_weights[1] == pytest.approx(0.75)


def test_model_max_length_one_is_rejected():
    with pytest.raises((SystemExit, ValueError)):
        _e2e(["--model_max_length", "1"])


def test_target_layers_parser_strict_rules():
    assert parse_target_layers("all") == "all"
    assert parse_target_layers("0") == (0,)
    assert parse_target_layers("0,2,5") == (0, 2, 5)
    assert parse_target_layers("0-7,12,20-27") == tuple(list(range(8)) + [12] + list(range(20, 28)))

    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_layers("*")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_layers("")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_layers("0,0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_layers("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_layers("5-1")


def test_target_modules_reject_alias_star_empty_and_duplicates():
    assert parse_target_modules("all") == "all"
    assert parse_target_modules("q_proj") == ("q_proj",)
    assert parse_target_modules("q_proj,k_proj,v_proj") == ("q_proj", "k_proj", "v_proj")

    with pytest.raises(argparse.ArgumentTypeError, match="q"):
        parse_target_modules("q")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_modules("*")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_modules("")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_modules("q_proj,q_proj")


def test_skip_layers_must_belong_to_target_layers_and_compression_categories():
    categories = parse_compression_categories("q_proj,k_proj")
    target_layers = parse_target_layers("0-3")
    skip = parse_skip_layers("1.q_proj,3.k_proj")
    validate_skip_layers_scope(skip, target_layers=target_layers, compression_categories=categories)

    with pytest.raises(ValueError, match="target_layers"):
        validate_skip_layers_scope(
            parse_skip_layers("7.q_proj"),
            target_layers=target_layers,
            compression_categories=categories,
        )
    with pytest.raises(ValueError, match="compression_categories"):
        validate_skip_layers_scope(
            parse_skip_layers("1.down_proj"),
            target_layers=target_layers,
            compression_categories=categories,
        )


def test_channel_global_count_is_scalar_ratio_and_rejects_override_string():
    cfg = _cat(
        [
            "--channel_protect_mode",
            "channel",
            "--channel_scope",
            "global",
            "--channel_protect_count",
            "0.001",
            "--channel_axis",
            "input",
        ]
    )
    vae, _opt = cfg.resolve_category_config("q_proj")
    assert vae.channel.channel_scope == "global"
    assert vae.channel.channel_protect_count == pytest.approx(0.001)
    assert isinstance(vae.channel.channel_protect_count, float)

    with pytest.raises((SystemExit, ValueError)):
        _cat(
            [
                "--channel_scope",
                "global",
                "--channel_protect_count",
                "default=0.001,cat:q_proj=0.002",
            ]
        )
    with pytest.raises((SystemExit, ValueError)):
        _cat(
            [
                "--channel_scope",
                "global",
                "--channel_protect_count",
                "1.0",
            ]
        )


def test_channel_layer_scope_keeps_integer_override():
    cfg = _cat(
        [
            "--channel_scope",
            "category",
            "--channel_protect_count",
            "default=4,cat:k_proj=8",
        ]
    )
    q_vae, _ = cfg.resolve_category_config("q_proj")
    k_vae, _ = cfg.resolve_category_config("k_proj")
    assert q_vae.channel.channel_protect_count == 4
    assert k_vae.channel.channel_protect_count == 8


def test_loss_type_rejects_legacy_encoded_topk_and_old_family():
    with pytest.raises((SystemExit, ValueError, argparse.ArgumentTypeError)):
        _e2e(["--loss_type", "kl_top_100"])
    with pytest.raises((SystemExit, ValueError, argparse.ArgumentTypeError)):
        _e2e(["--loss_type", "kd_top_100"])
    with pytest.raises((SystemExit, ValueError, argparse.ArgumentTypeError)):
        _e2e(["--loss_type", "eakld"])
    with pytest.raises((SystemExit, ValueError, argparse.ArgumentTypeError)):
        parse_hidden_layer_weighting("adaptive_top_K")
    assert parse_hidden_layer_weighting("adaptive_top_3") == "adaptive_top_3"


def test_selective_student_topk_only_allowed_for_kl_top():
    _e2e(["--loss_type", "kl_top", "--selective_student_topk", "true"])
    with pytest.raises((SystemExit, ValueError)):
        _e2e(["--loss_type", "kd_top", "--selective_student_topk", "true"])


def test_teacher_required_matches_plan():
    cfg = _e2e(["--loss_type", "sft", "--hidden_loss_weight", "0.0", "--pre_mlp_hidden_loss_weight", "0.0"])
    assert teacher_required(cfg.loss) is False
    cfg = _e2e(["--loss_type", "kl"])
    assert teacher_required(cfg.loss) is True
    cfg = _e2e(["--loss_type", "sft", "--hidden_loss_weight", "0.1"])
    assert teacher_required(cfg.loss) is True


def test_train_mode_none_requires_aux():
    with pytest.raises((SystemExit, ValueError)):
        _e2e(["--train_mode", "none", "--norm_train_mode", "none", "--lm_head_train_mode", "none"])
    cfg = _e2e(["--train_mode", "none", "--norm_train_mode", "final"])
    assert cfg.train_mode == "none"
    assert cfg.aux.norm_train_mode == "final"


def test_lora_explicit_checkpoint_conflict():
    inherited = validate_lora_against_checkpoint(
        LoRAConfig(rank=12, alpha=24.0, dropout=0.03),
        checkpoint=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
    )
    assert inherited == LoRAConfig(rank=8, alpha=16.0, dropout=0.0)

    same = validate_lora_against_checkpoint(
        LoRAConfig(rank=8, alpha=16.0, dropout=0.0, rank_explicit=True, alpha_explicit=True, dropout_explicit=True),
        checkpoint=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
    )
    assert same.rank == 8

    with pytest.raises(ValueError, match="lora_rank"):
        validate_lora_against_checkpoint(
            LoRAConfig(rank=12, rank_explicit=True),
            checkpoint=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        )


def test_vae_warmup_ratio_uses_floor_and_max_grad_norm_none_is_disabled():
    assert vae_num_warmup_steps(0.0, 10000) == 0
    assert vae_num_warmup_steps(0.1, 10000) == 1000
    assert vae_num_warmup_steps(1.0, 10000) == 10000
    assert vae_num_warmup_steps(0.33, 10) == 3
    cfg = _cat(["--vae_warmup_ratio", "0.0", "--vae_steps", "default=10000"])
    _vae, opt = cfg.resolve_category_config("q_proj")
    assert opt.vae_warmup_ratio == 0.0
    assert opt.vae_max_grad_norm is None
    assert vae_num_warmup_steps(opt.vae_warmup_ratio, opt.vae_steps) == 0
    with pytest.raises((SystemExit, ValueError)):
        _cat(["--vae_max_grad_norm", "0"])


def test_teacher_model_cpu_requires_output_cpu():
    with pytest.raises((SystemExit, ValueError)):
        _e2e(["--teacher_model_offload", "cpu", "--teacher_output_offload", "none"])
    cfg = _e2e(["--teacher_model_offload", "cpu", "--teacher_output_offload", "cpu"])
    assert cfg.runtime.teacher_model_offload == "cpu"


def test_after_category_none_is_legal_without_dataset_mix():
    cfg = _cat(["--after_category_mode", "none"])
    assert cfg.after_category_mode == "none"


def test_after_category_active_requires_dataset_mix():
    with pytest.raises((SystemExit, ValueError)):
        _cat(["--after_category_mode", "current_lora"])


def test_cat_production_entry_uses_common_parser_and_internal_adapter(monkeypatch):
    argv = [
        "--model_path", "dummy-model",
        "--compression_categories", "q_proj,k_proj",
        "--target_layers", "0-1",
        "--after_category_mode", "current_lora_decoder",
        "--dataset_mix", "openorca",
        "--vae_steps", "default=100,cat:k_proj=200",
        "--vae_batch_size", "64",
        "--steps", "default=20,after:q_proj=5",
        "--batch_size", "default=3",
        "--learning_rate", "default=2e-4",
        "--loss_type", "default=kl_top",
        "--top_k", "default=100",
        "--bf16", "true",
    ]
    cat_args, _hf_args, training_args, vae_args = parse_cat_runtime_args(argv)
    assert cat_args.after_category_mode == "current_lora_decoder"
    assert cat_args.compression_categories == "q_proj,k_proj"
    assert cat_args.target_layers == "0,1"
    assert cat_args.batch_size == 64
    assert vae_args.model_path == "dummy-model"
    assert training_args.bf16 is True

    captured = {}
    monkeypatch.setattr(cat_train_entry, "run_cat_train", lambda **kwargs: captured.update(kwargs))
    cat_train_entry.main(argv)
    assert captured["cat_args"].after_category_mode == "current_lora_decoder"
    assert captured["cat_args"]._common_cat_config.resolve_after_category_config("q_proj").opt.steps == 5

    with pytest.raises(SystemExit):
        cat_train_entry.main(
            [
                "--model_path", "dummy-model",
                "--target_categories", "q_proj",
                "--compression_categories", "q_proj",
            ]
        )


def test_common_parsers_default_to_process_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cat_train.py",
            "--model_path",
            "dummy-model",
            "--compression_categories",
            "q_proj",
        ],
    )
    assert parse_cat_cli().model_path == "dummy-model"

    monkeypatch.setattr(
        "sys.argv",
        [
            "compressed_e2e_fintuning.main",
            "--student_checkpoint_dir",
            "/tmp/student",
            "--train_mode",
            "decoder",
            "--dataset_mix",
            "openorca",
        ],
    )
    assert parse_e2e_cli().student_checkpoint_dir == "/tmp/student"


def test_deleted_flag_inventory_covers_plan_names():
    required = {
        "--distill_temperature",
        "--distill_alpha",
        "--distill_loss_alpha",
        "--distill_loss_type",
        "--prompt_kd_weight",
        "--distill_prompt_kd_weight",
        "--compressed_lora_scope",
        "--lora_use_dora",
        "--decoder_layers",
        "--sparse_bit_tuning",
        "--vae_tune_bias",
        "--tune_final_norm",
        "--use_post_norm_head_linear",
        "--target_categories",
        "--distill_after_category",
        "--include_all_linears",
        "--max_train_samples",
        "--parallel_stage_decode",
        "--packed_vq_decoder_linear",
        "--decode_device",
        "--decode_group_size",
    }
    missing = required - DELETED_CLI_FLAGS
    assert missing == set()
    assert math.isfinite(1.0)

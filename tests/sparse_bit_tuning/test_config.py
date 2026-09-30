import pytest

from compressed_e2e_fintuning.args import build_parser
from sparse_bit_tuning.config import (
    SparseBitTuningConfig,
    active_count,
    normalize_round_steps,
    resolve_bit_lr,
    resolve_round_steps,
    resolve_stable_steps,
)


def test_parser_defaults_use_canonical_train_mode_and_bit_config():
    parser = build_parser()
    ns, _ = parser.parse_known_args(["--student_checkpoint_dir", "/tmp/fake"])
    assert ns.train_mode == "decoder"
    assert ns.bit_active_ratio == 0.01
    assert ns.bit_optimizer == "rms_sgd"
    assert ns.bit_lr == "auto"
    assert ns.bit_round_steps == "auto"


@pytest.mark.parametrize(
    "mode",
    [
        "decoder",
        "lora",
        "sparse_bit",
        "decoder_lora",
        "decoder_sparse_bit",
        "lora_sparse_bit",
        "decoder_lora_sparse_bit",
    ],
)
def test_parser_accepts_canonical_train_modes(mode):
    parser = build_parser()
    ns, _ = parser.parse_known_args(
        ["--student_checkpoint_dir", "/tmp/fake", "--train_mode", mode]
    )
    assert ns.train_mode == mode


@pytest.mark.parametrize("flag", ["--finetune_mode", "--sparse_bit_tuning"])
def test_parser_rejects_deleted_mode_flags(flag):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--student_checkpoint_dir", "/tmp/fake", flag, "true"])


def test_active_count_and_auto_round_steps():
    assert active_count(100, 0.01) == 1
    assert active_count(101, 0.01) == 2
    assert active_count(5, 1.0) == 5
    assert resolve_round_steps("auto", total_optimizer_steps=5000, active_ratio=0.01) == 50
    assert resolve_stable_steps(20) == 4
    assert resolve_stable_steps(50) == 10


def test_bit_lr_defaults_and_validation():
    assert resolve_bit_lr("auto", optimizer="rms_sgd") == pytest.approx(0.05)
    assert resolve_bit_lr("auto", optimizer="adam") == pytest.approx(0.02)
    assert resolve_bit_lr("0.125", optimizer="adamw") == pytest.approx(0.125)
    with pytest.raises(ValueError):
        resolve_bit_lr("0", optimizer="adam")
    with pytest.raises(ValueError):
        normalize_round_steps("0")


def test_weight_decay_only_adamw():
    with pytest.raises(ValueError):
        SparseBitTuningConfig(enabled=True, optimizer="adam", weight_decay=0.01).normalized()
    cfg = SparseBitTuningConfig(enabled=True, optimizer="adamw", weight_decay=0.01).normalized()
    assert cfg.weight_decay == pytest.approx(0.01)


def test_ratio_validation():
    for value in [0.0, -0.1, 1.1]:
        with pytest.raises(ValueError):
            SparseBitTuningConfig(enabled=True, active_ratio=value).normalized()


@pytest.mark.parametrize("optimizer", ["rms_sgd", "adam", "adamw"])
def test_sensitivity_coordinates_require_explicit_positive_bit_lr(optimizer):
    with pytest.raises(ValueError, match="requires an explicit positive bit_lr"):
        SparseBitTuningConfig(
            enabled=True, optimizer=optimizer, proxy_coordinates="decoder_sensitivity",
        ).normalized()
    for invalid in ("0", "-1", "nan", "inf", "invalid"):
        with pytest.raises(ValueError, match="bit_lr"):
            SparseBitTuningConfig(
                enabled=True, optimizer=optimizer, bit_lr=invalid,
                proxy_coordinates="decoder_sensitivity",
            ).normalized()
    cfg = SparseBitTuningConfig(
        enabled=True, optimizer=optimizer, bit_lr="2e-5", proxy_coordinates="decoder_sensitivity",
    ).normalized()
    assert cfg.proxy_coordinates == "decoder_sensitivity"
    assert cfg.resolved_lr() == pytest.approx(2e-5)
    unit = SparseBitTuningConfig(enabled=True, optimizer=optimizer).normalized()
    assert unit.proxy_coordinates == "unit"
    assert unit.resolved_lr() == resolve_bit_lr("auto", optimizer=optimizer)


def test_proxy_coordinates_validation_and_disabled_config():
    with pytest.raises(ValueError, match="bit_proxy_coordinates"):
        SparseBitTuningConfig(proxy_coordinates="unknown").normalized()
    disabled = SparseBitTuningConfig(enabled=False, proxy_coordinates="decoder_sensitivity").normalized()
    assert disabled.bit_lr == "auto"


def _proxy_cli(*extra):
    from train_utils.config.cli import parse_e2e_cli

    return parse_e2e_cli([
        "--student_checkpoint_dir", "/tmp/unused", "--dataset_mix", "openorca",
        "--train_mode", "decoder_sparse_bit", *extra,
    ])


def test_proxy_coordinates_cli_rejects_auto_before_model_loading():
    with pytest.raises(SystemExit):
        _proxy_cli("--bit_proxy_coordinates", "decoder_sensitivity")
    cfg = _proxy_cli("--bit_proxy_coordinates", "decoder_sensitivity", "--bit_lr", "2e-5")
    assert cfg.bit_proxy_coordinates == "decoder_sensitivity"
    assert cfg.bit_lr == "2e-5"
    assert not cfg.remaining_argv
    with pytest.raises(SystemExit):
        _proxy_cli("--bit_proxy_coordinates", "unknown")


def test_proxy_coordinates_exact_resume_contract_preserves_unit_and_rejects_mode_switch():
    from types import SimpleNamespace

    from compressed_e2e_fintuning.v6_runtime_state import (
        build_e2e_immutable_resume_contract,
        validate_e2e_immutable_resume_contract,
    )

    def contract(cfg):
        return build_e2e_immutable_resume_contract(
            cfg=cfg, training_args=SimpleNamespace(),
            tokenizer=SimpleNamespace(name_or_path="same-tokenizer"), input_checkpoint_id="same-base",
            resolved_target_layers=[0], resolved_target_modules=["q_proj"], teacher_identity=None,
        )

    unit = contract(_proxy_cli("--bit_lr", "2e-5"))
    explicit_unit = contract(_proxy_cli("--bit_lr", "2e-5", "--bit_proxy_coordinates", "unit"))
    assert unit == explicit_unit
    assert unit["sparse_bit"] == {
        "active_ratio": 0.01, "optimizer": "rms_sgd", "bit_lr": "2e-5",
        "weight_decay": 0.0, "round_steps": "auto",
    }
    sensitivity = contract(_proxy_cli("--bit_lr", "2e-5", "--bit_proxy_coordinates", "decoder_sensitivity"))
    assert sensitivity["sparse_bit"] == {**unit["sparse_bit"], "proxy_coordinates": "decoder_sensitivity"}
    assert {k: v for k, v in sensitivity.items() if k != "sparse_bit"} == {
        k: v for k, v in unit.items() if k != "sparse_bit"
    }
    validate_e2e_immutable_resume_contract(unit, explicit_unit)
    validate_e2e_immutable_resume_contract(sensitivity, sensitivity)
    for before, after in ((unit, sensitivity), (sensitivity, unit)):
        with pytest.raises(ValueError, match="immutable contract mismatch"):
            validate_e2e_immutable_resume_contract(before, after)

    disabled = contract(_proxy_cli("--train_mode", "lora"))
    unused_sensitivity = contract(_proxy_cli("--train_mode", "lora", "--bit_proxy_coordinates", "decoder_sensitivity"))
    assert disabled == unused_sensitivity
    assert "proxy_coordinates" not in disabled["sparse_bit"]

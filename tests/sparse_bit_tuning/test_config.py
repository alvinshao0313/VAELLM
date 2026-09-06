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

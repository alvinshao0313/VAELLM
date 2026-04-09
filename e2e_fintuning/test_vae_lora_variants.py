from types import SimpleNamespace

import pytest
import torch
from peft.tuners.adalora.layer import SVDLinear as PeftAdaLoraLinear
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from e2e_fintuning import checkpoint_io
from e2e_fintuning import peft_proxy as peft_proxy_module
from e2e_fintuning import trainer as trainer_module
from e2e_fintuning.args import E2EFinetuneArguments, build_parser, validate_args


def _build_args(**overrides):
    args = E2EFinetuneArguments(
        student_checkpoint_dir="dummy-student",
        dataset_name="dummy-dataset",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _build_fake_proxy_model() -> nn.Module:
    model = nn.Module()
    proxy = object.__new__(peft_proxy_module.PeftVAELinearProxy)
    nn.Module.__init__(proxy)
    proxy.base_layer = nn.Linear(4, 3, bias=False)
    proxy.per_decoded_linear = nn.Linear(4, 3, bias=False)
    proxy.temporary = True
    model.proxy = proxy
    return model


def _seed_adalora_runtime_state(model: nn.Module):
    rankallocator = getattr(model, "_peft_proxy_adalora_rankallocator")
    rank_pattern = {}
    fill_value = 1.0
    for name, param in model.named_parameters():
        if "lora_" not in name or ".default" not in name:
            continue
        rankallocator.ipt[name] = torch.full_like(param, fill_value)
        rankallocator.exp_avg_ipt[name] = torch.full_like(param, fill_value + 1.0)
        rankallocator.exp_avg_unc[name] = torch.full_like(param, fill_value + 2.0)
        fill_value += 3.0
        if "lora_E.default" in name:
            size = int(param.numel())
            rank_pattern[name] = [(idx % 2) == 0 for idx in range(size)]
    model.peft_config["default"].rank_pattern = rank_pattern
    return rankallocator, rank_pattern


def test_validate_args_accepts_dora_gaussian():
    parser = build_parser()
    args = _build_args(vae_lora_variant="dora", vae_lora_init_mode="gaussian")
    validate_args(parser, args, SimpleNamespace(max_steps=10))
    assert args.vae_lora_variant == "dora"
    assert args.vae_lora_init_mode == "gaussian"


def test_validate_args_rejects_adalora_residual_svd():
    parser = build_parser()
    args = _build_args(vae_lora_variant="adalora", vae_lora_init_mode="residual_svd")
    with pytest.raises(SystemExit):
        validate_args(parser, args, SimpleNamespace(max_steps=10))


def test_validate_args_rejects_adalora_without_positive_max_steps():
    parser = build_parser()
    args = _build_args(vae_lora_variant="adalora", vae_lora_init_mode="zero")
    with pytest.raises(SystemExit):
        validate_args(parser, args, SimpleNamespace(max_steps=-1))


@pytest.mark.parametrize(("variant", "expected_type"), [("plain", PeftLoraLinear), ("rslora", PeftLoraLinear), ("dora", PeftLoraLinear)])
def test_ensure_peft_vae_proxy_adapter_injects_lora_family(variant, expected_type):
    model = _build_fake_proxy_model()
    injected = peft_proxy_module.ensure_peft_vae_proxy_adapter(
        model,
        variant=variant,
        rank=4,
        alpha=8.0,
        dropout=0.1,
        init_mode="gaussian",
    )

    assert injected == 1
    assert isinstance(model.proxy.per_decoded_linear, expected_type)
    adapter_name = "default"
    if variant == "rslora":
        assert peft_proxy_module.collect_peft_vae_proxy_adapter_specs(model, train_mode="vae_lora")[0]["use_rslora"] is True
    if variant == "dora":
        assert model.proxy.per_decoded_linear.use_dora[adapter_name] is True


def test_ensure_peft_vae_proxy_adapter_injects_adalora_and_zero_init():
    model = _build_fake_proxy_model()
    injected = peft_proxy_module.ensure_peft_vae_proxy_adapter(
        model,
        variant="adalora",
        rank=6,
        alpha=8.0,
        dropout=0.0,
        init_mode="zero",
        total_step=20,
        adalora_target_r=4,
        adalora_init_r=6,
        adalora_tinit=2,
        adalora_tfinal=4,
        adalora_delta_t=2,
        adalora_beta1=0.8,
        adalora_beta2=0.9,
        adalora_orth_reg_weight=0.3,
    )

    assert injected == 1
    assert isinstance(model.proxy.per_decoded_linear, PeftAdaLoraLinear)
    assert torch.count_nonzero(model.proxy.per_decoded_linear.get_delta_weight("default")).item() == 0
    specs = peft_proxy_module.collect_peft_vae_proxy_adapter_specs(model, train_mode="vae_lora")
    assert specs[0]["adapter_type"] == "peft_proxy_adalora"
    assert specs[0]["target_r"] == 4
    assert specs[0]["init_r"] == 6
    assert getattr(model, "_peft_proxy_adalora_rankallocator", None) is not None


def test_adalora_runtime_state_round_trip_through_checkpoint(tmp_path, monkeypatch):
    model = _build_fake_proxy_model()
    monkeypatch.setattr(checkpoint_io, "iter_named_vae_module_refs", lambda _model: iter(()))
    peft_proxy_module.ensure_peft_vae_proxy_adapter(
        model,
        variant="adalora",
        rank=6,
        alpha=8.0,
        dropout=0.0,
        init_mode="zero",
        total_step=20,
        adalora_target_r=4,
        adalora_init_r=6,
        adalora_tinit=2,
        adalora_tfinal=4,
        adalora_delta_t=2,
        adalora_beta1=0.8,
        adalora_beta2=0.9,
        adalora_orth_reg_weight=0.3,
    )
    rankallocator, rank_pattern = _seed_adalora_runtime_state(model)
    save_dir = tmp_path / "adalora_ckpt"

    checkpoint_io.save_e2e_model_checkpoint(
        model,
        str(save_dir),
        save_config=False,
        extra_meta={"stage": "e2e_fintuning"},
        compact_unload_vae_original_weights=True,
    )

    restored = _build_fake_proxy_model()
    monkeypatch.setattr(checkpoint_io, "iter_named_vae_module_refs", lambda _model: iter(()))
    monkeypatch.setattr(checkpoint_io, "refresh_peft_proxy_decoded_linears", lambda _model: 0)
    checkpoint_io.load_e2e_checkpoint_into_model(
        restored,
        str(save_dir),
        map_location="cpu",
        strict=True,
    )

    restored_rankallocator = getattr(restored, "_peft_proxy_adalora_rankallocator")
    for group_name in ("ipt", "exp_avg_ipt", "exp_avg_unc"):
        original_group = getattr(rankallocator, group_name)
        restored_group = getattr(restored_rankallocator, group_name)
        assert sorted(restored_group.keys()) == sorted(original_group.keys())
        for name in original_group:
            torch.testing.assert_close(restored_group[name], original_group[name], atol=0, rtol=0)
    assert restored.peft_config["default"].rank_pattern == rank_pattern


def test_update_peft_vae_proxy_adalora_rejects_missing_rank_pattern_after_freeze():
    model = _build_fake_proxy_model()
    peft_proxy_module.ensure_peft_vae_proxy_adapter(
        model,
        variant="adalora",
        rank=6,
        alpha=8.0,
        dropout=0.0,
        init_mode="zero",
        total_step=20,
        adalora_target_r=4,
        adalora_init_r=6,
        adalora_tinit=2,
        adalora_tfinal=4,
        adalora_delta_t=2,
        adalora_beta1=0.8,
        adalora_beta2=0.9,
        adalora_orth_reg_weight=0.3,
    )
    model.peft_config["default"].rank_pattern = None

    with pytest.raises(RuntimeError):
        peft_proxy_module.update_peft_vae_proxy_adalora(model, global_step=17)


def test_reject_removed_extra_lora_checkpoint_meta():
    with pytest.raises(ValueError):
        checkpoint_io._reject_removed_extra_lora_checkpoint({"extra_meta": {"lora_embedding": True}})
    with pytest.raises(ValueError):
        checkpoint_io._reject_removed_extra_lora_checkpoint({"adapter_modules": [{"adapter_type": "embedding_lora"}]})


def test_adalora_callback_uses_next_global_step(monkeypatch):
    calls = []

    class DummyTrainer:
        def _unwrap_student_model(self):
            return object()

    def fake_update(_model, *, global_step: int):
        calls.append(global_step)
        return True

    monkeypatch.setattr(trainer_module, "update_peft_vae_proxy_adalora", fake_update)
    callback = trainer_module.E2EAdaLoraCallback(DummyTrainer())
    control = object()
    returned = callback.on_optimizer_step(
        args=None,
        state=SimpleNamespace(global_step=7),
        control=control,
    )

    assert calls == [8]
    assert returned is control

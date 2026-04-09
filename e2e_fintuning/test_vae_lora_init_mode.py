import logging
from types import SimpleNamespace

import torch
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from e2e_fintuning import peft_proxy as peft_proxy_module
from e2e_fintuning import runtime


def _truncated_svd_reconstruction(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    k = min(int(rank), int(s.shape[0]))
    if k == 0:
        return torch.zeros_like(matrix.float())
    return (u[:, :k] * s[:k].unsqueeze(0)) @ vh[:k, :]


def _build_teacher_with_q_proj(*, in_features: int, out_features: int) -> nn.Module:
    teacher = nn.Module()
    teacher.model = nn.Module()
    teacher.model.layers = nn.ModuleList([nn.Module()])
    teacher.model.layers[0].self_attn = nn.Module()
    teacher.model.layers[0].self_attn.q_proj = nn.Linear(in_features, out_features, bias=False)
    return teacher


def _build_teacher_with_q_proj_and_up_proj() -> nn.Module:
    teacher = nn.Module()
    teacher.model = nn.Module()
    teacher.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in teacher.model.layers:
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 3, bias=False)
        layer.mlp.up_proj = nn.Linear(4, 5, bias=False)
    return teacher


def test_initialize_peft_linear_from_residual_svd_matches_truncated_svd():
    base_layer = nn.Linear(5, 4, bias=False, dtype=torch.float32)
    peft_linear = PeftLoraLinear(
        base_layer,
        "default",
        r=2,
        lora_alpha=6,
        lora_dropout=0.0,
    )
    residual = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, -1.0, 2.5, 0.0, 1.0],
            [-3.0, 1.5, 0.0, 2.0, -0.5],
            [4.0, -2.0, 1.0, 3.5, 0.5],
        ],
        dtype=torch.float32,
    )

    peft_proxy_module.initialize_peft_linear_from_residual_svd(
        peft_linear,
        residual,
        module_name="model.layers.0.self_attn.q_proj",
    )

    expected = _truncated_svd_reconstruction(residual, rank=2)
    actual = peft_linear.get_delta_weight("default")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_initialize_peft_linear_from_residual_svd_zero_pads_tail_rank():
    base_layer = nn.Linear(3, 2, bias=False, dtype=torch.float32)
    peft_linear = PeftLoraLinear(
        base_layer,
        "default",
        r=5,
        lora_alpha=10,
        lora_dropout=0.0,
    )
    residual = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [1.5, 0.0, -3.0],
        ],
        dtype=torch.float32,
    )

    peft_proxy_module.initialize_peft_linear_from_residual_svd(
        peft_linear,
        residual,
        module_name="model.layers.0.self_attn.q_proj",
    )

    actual = peft_linear.get_delta_weight("default")
    torch.testing.assert_close(actual, residual, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(peft_linear.lora_A["default"].weight[2:, :]).item() == 0
    assert torch.count_nonzero(peft_linear.lora_B["default"].weight[:, 2:]).item() == 0


def test_initialize_peft_linear_from_residual_svd_recomputes_dora_magnitude():
    base_layer = nn.Linear(3, 2, bias=False, dtype=torch.float32)
    with torch.no_grad():
        base_layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, -2.0, 0.5],
                    [0.5, 1.5, -1.0],
                ],
                dtype=torch.float32,
            )
        )
    peft_linear = PeftLoraLinear(
        base_layer,
        "default",
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        use_dora=True,
    )
    residual = torch.tensor(
        [
            [0.5, 1.0, -0.5],
            [-1.0, 0.0, 1.5],
        ],
        dtype=torch.float32,
    )

    peft_proxy_module.initialize_peft_linear_from_residual_svd(
        peft_linear,
        residual,
        module_name="model.layers.0.self_attn.q_proj",
    )

    actual_delta = peft_linear.get_delta_weight("default")
    expected_magnitude = torch.linalg.norm(base_layer.weight.detach() + actual_delta, dim=1)
    torch.testing.assert_close(
        peft_linear.lora_magnitude_vector["default"].detach(),
        expected_magnitude,
        atol=1e-5,
        rtol=1e-5,
    )


def test_initialize_peft_vae_proxy_lora_from_teacher_residual_uses_matching_module_name(monkeypatch):
    base_layer = nn.Linear(4, 3, bias=False, dtype=torch.float32)
    with torch.no_grad():
        base_layer.weight.copy_(
            torch.tensor(
                [
                    [0.5, -1.0, 2.0, 1.5],
                    [1.0, 0.0, -0.5, 3.0],
                    [-2.0, 1.0, 0.5, -1.5],
                ],
                dtype=torch.float32,
            )
        )
    peft_linear = PeftLoraLinear(
        base_layer,
        "default",
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    module_name = "model.layers.0.self_attn.q_proj"
    teacher = _build_teacher_with_q_proj(in_features=4, out_features=3)
    target_weight = base_layer.weight.detach() + torch.tensor(
        [
            [1.0, 0.0, -2.0, 0.5],
            [0.0, 1.5, 0.5, -1.0],
            [2.0, -0.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        teacher.model.layers[0].self_attn.q_proj.weight.copy_(target_weight)

    monkeypatch.setattr(
        peft_proxy_module,
        "iter_named_peft_vae_proxies",
        lambda _model: iter(
            [
                (
                    module_name,
                    SimpleNamespace(per_decoded_linear=peft_linear),
                )
            ]
        ),
    )

    initialized = peft_proxy_module.initialize_peft_vae_proxy_lora_from_teacher_residual(
        nn.Module(),
        teacher,
        batch_device=torch.device("cpu"),
    )

    expected = _truncated_svd_reconstruction(target_weight - base_layer.weight.detach(), rank=2)
    actual = peft_linear.get_delta_weight("default")
    assert initialized == 1
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_initialize_peft_vae_proxy_lora_from_teacher_residual_rejects_shape_mismatch(monkeypatch):
    base_layer = nn.Linear(4, 3, bias=False, dtype=torch.float32)
    peft_linear = PeftLoraLinear(
        base_layer,
        "default",
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    teacher = _build_teacher_with_q_proj(in_features=4, out_features=2)

    monkeypatch.setattr(
        peft_proxy_module,
        "iter_named_peft_vae_proxies",
        lambda _model: iter(
            [
                (
                    "model.layers.0.self_attn.q_proj",
                    SimpleNamespace(per_decoded_linear=peft_linear),
                )
            ]
        ),
    )

    try:
        peft_proxy_module.initialize_peft_vae_proxy_lora_from_teacher_residual(
            nn.Module(),
            teacher,
            batch_device=torch.device("cpu"),
        )
        raise AssertionError("Expected shape mismatch error.")
    except ValueError as exc:
        assert "shape mismatch" in str(exc)


def test_initialize_peft_vae_proxy_lora_from_teacher_residual_batches_same_category_together(monkeypatch):
    teacher = _build_teacher_with_q_proj_and_up_proj()

    q_proj_0 = PeftLoraLinear(
        nn.Linear(4, 3, bias=False, dtype=torch.float32),
        "default",
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    q_proj_1 = PeftLoraLinear(
        nn.Linear(4, 3, bias=False, dtype=torch.float32),
        "default",
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    up_proj_0 = PeftLoraLinear(
        nn.Linear(4, 5, bias=False, dtype=torch.float32),
        "default",
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
    )

    with torch.no_grad():
        q_proj_0.base_layer.weight.copy_(
            torch.tensor(
                [
                    [0.5, -1.0, 2.0, 1.5],
                    [1.0, 0.0, -0.5, 3.0],
                    [-2.0, 1.0, 0.5, -1.5],
                ],
                dtype=torch.float32,
            )
        )
        q_proj_1.base_layer.weight.copy_(
            torch.tensor(
                [
                    [1.5, -0.5, 0.0, 2.0],
                    [0.5, 1.0, -1.0, 0.0],
                    [2.0, -1.0, 0.5, -0.5],
                ],
                dtype=torch.float32,
            )
        )
        up_proj_0.base_layer.weight.copy_(
            torch.tensor(
                [
                    [0.0, 1.0, -1.0, 0.5],
                    [1.5, -0.5, 2.0, -1.0],
                    [-1.0, 0.5, 0.5, 1.0],
                    [2.0, -1.5, 1.0, 0.0],
                    [0.5, 0.5, -0.5, 1.5],
                ],
                dtype=torch.float32,
            )
        )
        teacher.model.layers[0].self_attn.q_proj.weight.copy_(
            q_proj_0.base_layer.weight.detach()
            + torch.tensor(
                [
                    [1.0, 0.0, -2.0, 0.5],
                    [0.0, 1.5, 0.5, -1.0],
                    [2.0, -0.5, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        teacher.model.layers[1].self_attn.q_proj.weight.copy_(
            q_proj_1.base_layer.weight.detach()
            + torch.tensor(
                [
                    [-1.0, 2.0, 0.5, 0.0],
                    [0.5, -0.5, 1.5, 2.0],
                    [1.0, 1.0, -1.5, 0.5],
                ],
                dtype=torch.float32,
            )
        )
        teacher.model.layers[0].mlp.up_proj.weight.copy_(
            up_proj_0.base_layer.weight.detach()
            + torch.tensor(
                [
                    [0.5, -1.0, 0.0, 1.0],
                    [-0.5, 1.5, -1.0, 0.0],
                    [1.0, 0.0, 0.5, -1.5],
                    [0.0, 2.0, -0.5, 0.5],
                    [-1.0, 0.5, 1.5, 0.0],
                ],
                dtype=torch.float32,
            )
        )

    monkeypatch.setattr(
        peft_proxy_module,
        "iter_named_peft_vae_proxies",
        lambda _model: iter(
            [
                ("model.layers.1.self_attn.q_proj", SimpleNamespace(per_decoded_linear=q_proj_1)),
                ("model.layers.0.mlp.up_proj", SimpleNamespace(per_decoded_linear=up_proj_0)),
                ("model.layers.0.self_attn.q_proj", SimpleNamespace(per_decoded_linear=q_proj_0)),
            ]
        ),
    )

    initialized = peft_proxy_module.initialize_peft_vae_proxy_lora_from_teacher_residual(
        nn.Module(),
        teacher,
        batch_device=torch.device("cpu"),
    )

    expected_q0 = _truncated_svd_reconstruction(
        teacher.model.layers[0].self_attn.q_proj.weight.detach() - q_proj_0.base_layer.weight.detach(),
        rank=2,
    )
    expected_q1 = _truncated_svd_reconstruction(
        teacher.model.layers[1].self_attn.q_proj.weight.detach() - q_proj_1.base_layer.weight.detach(),
        rank=2,
    )
    expected_up0 = _truncated_svd_reconstruction(
        teacher.model.layers[0].mlp.up_proj.weight.detach() - up_proj_0.base_layer.weight.detach(),
        rank=2,
    )

    assert initialized == 3
    torch.testing.assert_close(q_proj_0.get_delta_weight("default"), expected_q0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q_proj_1.get_delta_weight("default"), expected_q1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(up_proj_0.get_delta_weight("default"), expected_up0, atol=1e-5, rtol=1e-5)


def test_should_initialize_vae_lora_residual_svd_skips_resume():
    args = SimpleNamespace(vae_lora_init_mode="residual_svd")
    selection = SimpleNamespace(peft_proxy_modules=["model.layers.0.self_attn.q_proj"])
    assert runtime._should_initialize_vae_lora_residual_svd(
        args=args,
        selection=selection,
        resume_from_checkpoint=None,
    )
    assert not runtime._should_initialize_vae_lora_residual_svd(
        args=args,
        selection=selection,
        resume_from_checkpoint="checkpoint-100",
    )


def test_should_initialize_vae_lora_residual_svd_rejects_adalora():
    args = SimpleNamespace(vae_lora_variant="adalora", vae_lora_init_mode="residual_svd")
    selection = SimpleNamespace(peft_proxy_modules=["model.layers.0.self_attn.q_proj"])
    assert not runtime._should_initialize_vae_lora_residual_svd(
        args=args,
        selection=selection,
        resume_from_checkpoint=None,
    )


def test_resolve_saved_vae_lora_init_mode_preserves_resume_value():
    args = SimpleNamespace(vae_lora_init_mode="zero")
    meta = {"extra_meta": {"vae_lora_init_mode": "residual_svd"}}
    assert runtime._resolve_saved_vae_lora_init_mode(
        args=args,
        meta=meta,
        resume_from_checkpoint="checkpoint-100",
    ) == "residual_svd"


def test_load_teacher_for_e2e_init_only_uses_external_teacher_once(monkeypatch):
    calls = []

    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

    def fake_get_model(model_name, access_token):
        calls.append((model_name, access_token))
        return DummyTeacher()

    monkeypatch.setattr(runtime, "get_model", fake_get_model)
    args = SimpleNamespace(loss_type="sft", teacher_model_path="explicit-teacher")
    hf_args = SimpleNamespace(access_token="token")

    teacher_model, teacher_source, keep_teacher_for_training = runtime._load_teacher_for_e2e(
        args=args,
        hf_args=hf_args,
        meta={"base_model_path": "fallback-teacher"},
        log=logging.getLogger("test"),
        require_for_init=True,
    )

    assert calls == [("explicit-teacher", "token")]
    assert teacher_source == "external_teacher_init_only"
    assert keep_teacher_for_training is False
    assert teacher_model is not None
    assert teacher_model.training is False
    for param in teacher_model.parameters():
        assert param.requires_grad is False


def test_load_teacher_for_e2e_reuses_teacher_when_training_needs_it(monkeypatch):
    calls = []

    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

    def fake_get_model(model_name, access_token):
        calls.append((model_name, access_token))
        return DummyTeacher()

    monkeypatch.setattr(runtime, "get_model", fake_get_model)
    args = SimpleNamespace(loss_type="kd", teacher_model_path="explicit-teacher")
    hf_args = SimpleNamespace(access_token=None)

    teacher_model, teacher_source, keep_teacher_for_training = runtime._load_teacher_for_e2e(
        args=args,
        hf_args=hf_args,
        meta={"base_model_path": "fallback-teacher"},
        log=logging.getLogger("test"),
        require_for_init=True,
    )

    assert calls == [("explicit-teacher", None)]
    assert teacher_source == "external_teacher"
    assert keep_teacher_for_training is True
    assert teacher_model is not None

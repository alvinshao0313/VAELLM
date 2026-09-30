import pytest
import torch

from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.module import BankSpec, SparseBitTuningModule
from sparse_bit_tuning.optimizer import BitOptimizerManager, SparseBitCompositeOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _module(device="cuda:0", proxy_coordinates="unit"):
    specs = [
        BankSpec(
            canonical_key="m0|stage=0|part=0",
            module_path="m0",
            stage_idx=0,
            part_idx=0,
            logical_shape=(2, 1, 16),
            n_bits=32,
            n_active=4,
            device=torch.device(device),
        ),
        BankSpec(
            canonical_key="m1|stage=0|part=0",
            module_path="m1",
            stage_idx=0,
            part_idx=0,
            logical_shape=(2, 1, 16),
            n_bits=32,
            n_active=3,
            device=torch.device(device),
        ),
    ]
    return SparseBitTuningModule(specs, target_chunk_bytes=1024, proxy_coordinates=proxy_coordinates)


def _set_score_and_grad(module, score_values, grad_values):
    score = module.score_chunks[0]
    with torch.no_grad():
        score.copy_(torch.tensor(score_values, dtype=torch.float16, device=score.device))
    score.grad = torch.tensor(grad_values, dtype=torch.float16, device=score.device)
    return score


def test_rms_sgd_matches_reference_and_counts_fp16_sign_flips():
    module = _module()
    score = _set_score_and_grad(
        module,
        [1.0, -1.0, 0.02, -0.02, 1.0, -1.0, 0.01],
        [2.0, -2.0, 1.0, -1.0, 3.0, -3.0, 1.0],
    )
    old = score.detach().clone()
    grad = score.grad.detach().clone().float()
    cfg = SparseBitTuningConfig(enabled=True, optimizer="rms_sgd", bit_lr=0.05)
    manager = BitOptimizerManager(module, cfg)
    counters = manager.step_scores(optimizer_step_in_round=1)
    torch.cuda.synchronize(score.device)

    expected = old.float().clone()
    starts = [(0, 4), (4, 7)]
    for start, end in starts:
        g = grad[start:end]
        rms = torch.sqrt(torch.mean(g * g) + 1e-8)
        expected[start:end] = torch.clamp(expected[start:end] - 0.05 * g / rms, -1.0, 1.0)
    expected_fp16 = expected.to(torch.float16)
    assert torch.equal(score.detach(), expected_fp16)
    expected_flips = int(((old >= 0) != (expected_fp16 >= 0)).sum().item())
    assert sum(int(x.item()) for x in counters) == expected_flips


@pytest.mark.parametrize("optimizer,weight_decay", [("adam", 0.0), ("adamw", 0.1)])
def test_adam_variants_match_reference(optimizer, weight_decay):
    module = _module()
    score = _set_score_and_grad(
        module,
        [1.0, -1.0, 0.02, -0.02, 1.0, -1.0, 0.01],
        [0.5, -0.5, 2.0, -2.0, 0.25, -0.25, 1.0],
    )
    old = score.detach().clone()
    grad = score.grad.detach().clone().float()
    cfg = SparseBitTuningConfig(
        enabled=True,
        optimizer=optimizer,
        bit_lr=0.02,
        weight_decay=weight_decay,
    )
    manager = BitOptimizerManager(module, cfg)
    counters = manager.step_scores(optimizer_step_in_round=1)
    torch.cuda.synchronize(score.device)

    beta1, beta2 = 0.9, 0.999
    m = (1.0 - beta1) * grad
    v = (1.0 - beta2) * grad.square()
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    base = old.float()
    if optimizer == "adamw":
        base = base * (1.0 - 0.02 * weight_decay)
    expected = torch.clamp(base - 0.02 * m_hat / (torch.sqrt(v_hat) + 1e-8), -1.0, 1.0)
    expected_fp16 = expected.to(torch.float16)
    assert torch.equal(score.detach(), expected_fp16)
    expected_flips = int(((old >= 0) != (expected_fp16 >= 0)).sum().item())
    assert sum(int(x.item()) for x in counters) == expected_flips

    state_tensors = list(manager.state_tensors())
    assert len(state_tensors) == 2
    assert all(t.dtype == torch.float32 and t.device == score.device for t in state_tensors)
    manager.reset_round_state()
    assert all(torch.count_nonzero(t).item() == 0 for t in state_tensors)


def test_composite_keeps_bit_state_out_of_torch_optimizer_state():
    module = _module()
    score = _set_score_and_grad(module, [1.0] * 7, [0.1] * 7)
    manager = BitOptimizerManager(
        module,
        SparseBitTuningConfig(enabled=True, optimizer="adam", bit_lr=0.02),
    )
    manager.step_scores(optimizer_step_in_round=1)
    main_param = torch.nn.Parameter(torch.ones(2, device=score.device))
    main = torch.optim.AdamW([main_param], lr=1e-4)
    composite = SparseBitCompositeOptimizer(
        main_optimizer=main,
        bit_manager=manager,
        step_callback=lambda: None,
    )
    payload = composite.state_dict()
    assert all(id(p) not in composite.state for p in module.score_chunks)
    assert "_sparse_bit_main_optimizer" in payload
    assert not any(
        torch.is_tensor(v) and v.numel() == score.numel() and v.dtype == torch.float32
        for state in composite.state.values()
        for v in (state.values() if isinstance(state, dict) else [])
    )


@pytest.mark.parametrize("optimizer,weight_decay", [("rms_sgd", 0.0), ("adam", 0.0), ("adamw", 0.1)])
def test_sensitivity_optimizer_matches_coordinate_reference_over_multiple_steps(optimizer, weight_decay):
    module = _module(proxy_coordinates="decoder_sensitivity")
    module.coordinate_scales.update({spec.canonical_key: scale for spec, scale in zip(module.bank_specs, (0.002, 0.008))})
    score = module.score_chunks[0]
    assert score.dtype == torch.float32
    initial = torch.tensor([0.001, -0.001, 0.00001, -0.00001, 0.004, -0.004, 0.0], device=score.device)
    with torch.no_grad():
        score.copy_(initial)
    cfg = SparseBitTuningConfig(
        enabled=True, optimizer=optimizer, bit_lr=2e-5, weight_decay=weight_decay,
        proxy_coordinates="decoder_sensitivity",
    )
    manager = BitOptimizerManager(module, cfg)
    manager.refresh_coordinate_scales()
    expected = initial.clone()
    m = torch.zeros_like(expected)
    v = torch.zeros_like(expected)
    radii = torch.tensor([0.001] * 4 + [0.004] * 3, device=score.device)
    for step in range(1, 5):
        grad = torch.tensor([500.0, -500.0, 1000.0, -1000.0, -250.0, 250.0, 400.0], device=score.device)
        grad = grad * (1 if step < 3 else -0.25)
        score.grad = grad.clone()
        old = expected.clone()
        if optimizer == "rms_sgd":
            update = torch.empty_like(grad)
            for start, end in ((0, 4), (4, 7)):
                update[start:end] = grad[start:end] / torch.sqrt(grad[start:end].square().mean() + 1e-8)
        else:
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad.square()
            update = (m / (1 - 0.9 ** step)) / (torch.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        base = expected * (1 - 2e-5 * weight_decay) if optimizer == "adamw" else expected
        expected = torch.maximum(-radii, torch.minimum(radii, base - 2e-5 * update))
        counters = manager.step_scores(optimizer_step_in_round=step)
        torch.testing.assert_close(score.detach(), expected, rtol=2e-6, atol=2e-10)
        assert sum(counter.item() for counter in counters) == ((old >= 0) != (expected >= 0)).sum().item()
    if optimizer != "rms_sgd":
        actual_m, actual_v = list(manager.state_tensors())
        torch.testing.assert_close(actual_m, m, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(actual_v, v, rtol=2e-6, atol=1e-5)
    manager.reset_round_state()
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in manager.state_tensors())


def test_fp32_sensitivity_scores_unscale_with_sparse_amp():
    from sparse_bit_tuning.amp import SparseBitGradScaler

    score = torch.nn.Parameter(torch.tensor([0.001, -0.001], device="cuda:0", dtype=torch.float32))
    optimizer = torch.optim.SGD(
        [{"params": [score], "lr": 0.0, "_sparse_bit_score_group": True}], lr=0.0,
    )
    scaler = SparseBitGradScaler("cuda", init_scale=128.0)
    # Accumulated scaled gradients must retain FP32 before the sparse optimizer step.
    scaler.scale(score.sum() * 2).backward()
    scaler.scale(score.sum() * 3).backward()
    scaler.unscale_(optimizer)
    assert score.grad.dtype == torch.float32
    assert torch.equal(score.grad, torch.full_like(score, 5.0))

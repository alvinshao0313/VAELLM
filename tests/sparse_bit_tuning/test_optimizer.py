import pytest
import torch

from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.module import BankSpec, SparseBitTuningModule
from sparse_bit_tuning.optimizer import BitOptimizerManager, SparseBitCompositeOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _module(device="cuda:0"):
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
    return SparseBitTuningModule(specs, target_chunk_bytes=1024)


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

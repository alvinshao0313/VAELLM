import pytest
import torch

from sparse_bit_tuning.amp import SparseBitGradScaler
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.optimizer import SparseBitCompositeOptimizer
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _optimizer(main, bit):
    return _CountingSGD(
        [
            {"params": [main], "lr": 0.1},
            {
                "params": [bit],
                "lr": 0.0,
                "_sparse_bit_score_group": True,
            },
        ],
        lr=0.1,
    )


def test_stock_scaler_rejects_fp16_score_grad_but_sparse_scaler_accepts():
    device = torch.device("cuda:0")
    main = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float32))
    bit = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    opt = _optimizer(main, bit)
    stock = torch.amp.GradScaler("cuda", init_scale=128.0)
    loss = stock.scale(main.float().sum() * 3.0 + bit.float().sum() * 2.0)
    loss.backward()
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients"):
        stock.unscale_(opt)

    main.grad = None
    bit.grad = None
    base = torch.amp.GradScaler("cuda", init_scale=128.0)
    scaler = SparseBitGradScaler.from_existing(base)
    loss = scaler.scale(main.float().sum() * 3.0 + bit.float().sum() * 2.0)
    loss.backward()
    scaler.unscale_(opt)
    assert main.grad is not None and main.grad.item() == pytest.approx(3.0)
    assert bit.grad is not None and bit.grad.dtype == torch.float16
    assert bit.grad.item() == pytest.approx(2.0)


def test_bit_inf_skips_entire_optimizer_step():
    device = torch.device("cuda:0")
    main = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float32))
    bit = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    opt = _optimizer(main, bit)
    scaler = SparseBitGradScaler("cuda", init_scale=128.0)
    loss = scaler.scale(main.float().sum() + bit.float().sum())
    loss.backward()
    assert bit.grad is not None
    bit.grad.fill_(float("inf"))
    before_main = main.detach().clone()
    before_bit = bit.detach().clone()
    scaler.step(opt)
    scaler.update()
    assert opt.step_calls == 0
    assert torch.equal(main.detach(), before_main)
    assert torch.equal(bit.detach(), before_bit)


def test_pure_bit_only_unscale_and_step():
    device = torch.device("cuda:0")
    bit = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    opt = _CountingSGD(
        [{"params": [bit], "lr": 0.0, "_sparse_bit_score_group": True}], lr=0.0
    )
    scaler = SparseBitGradScaler("cuda", init_scale=64.0)
    scaler.scale(bit.float().sum() * 4.0).backward()
    scaler.step(opt)
    scaler.update()
    assert opt.step_calls == 1
    assert bit.grad is not None
    assert bit.grad.item() == pytest.approx(4.0)


def test_unmarked_fp16_main_grad_still_rejected():
    device = torch.device("cuda:0")
    main = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    bit = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    opt = _optimizer(main, bit)
    scaler = SparseBitGradScaler("cuda", init_scale=64.0)
    scaler.scale(main.float().sum() + bit.float().sum()).backward()
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients"):
        scaler.unscale_(opt)


def test_bf16_main_grad_and_fp16_bit_grad_unscale_together():
    device = torch.device("cuda:0")
    main = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.bfloat16))
    bit = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float16))
    opt = _optimizer(main, bit)
    scaler = SparseBitGradScaler("cuda", init_scale=64.0)
    scaler.scale(main.float().sum() * 3.0 + bit.float().sum() * 2.0).backward()
    scaler.unscale_(opt)
    assert main.grad is not None and main.grad.dtype == torch.bfloat16
    assert float(main.grad.item()) == pytest.approx(3.0, rel=1e-2)
    assert bit.grad is not None and bit.grad.dtype == torch.float16
    assert float(bit.grad.item()) == pytest.approx(2.0, rel=1e-3)


def _manager_for_overflow_test(device):
    bits = torch.tensor([[[1, 0, 1, 0, 1, 0, 1, 0]]], dtype=torch.bool)
    decoder = Decoder(
        in_dim=8, out_dim=1, hidden_dim=4, num_res_blocks=0, norm_type="layer",
        decoder_type="linear", use_checkpoint=False, num_models=1,
    )
    layer = VAELinear(
        in_features=1, out_features=1, bias=None, original_weight=None,
        vq_weight=bits, decoder=decoder, codebook_dim=1, transpose=False,
    ).to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = torch.nn.Module()
    root.add_module("layer", layer)
    manager = SparseBitTuningManager(
        root_model=root, targets=[("layer", layer)], target_devices={"layer": device},
        training_seed=11,
        config=SparseBitTuningConfig(enabled=True, active_ratio=0.5, optimizer="rms_sgd", bit_lr=0.5, round_steps=2),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=2)
    manager.initialize_scores()
    return root, layer, manager


def test_bit_overflow_skips_composite_main_and_bit_state_atomically():
    device = torch.device("cuda:0")
    _root, layer, manager = _manager_for_overflow_test(device)
    main = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.bfloat16))
    main_opt = torch.optim.SGD([main], lr=0.1)
    callback_calls = {"count": 0}

    def _bit_step():
        callback_calls["count"] += 1
        manager.optimizer_step()

    composite = SparseBitCompositeOptimizer(
        main_optimizer=main_opt, bit_manager=manager.bit_optimizer, step_callback=_bit_step
    )
    scaler = SparseBitGradScaler("cuda", init_scale=64.0)
    score = manager.score_module.score_chunks[0]
    scaled_loss = scaler.scale(main.float().sum() + score.float().sum())
    scaled_loss.backward()
    assert score.grad is not None
    score.grad.fill_(float("inf"))
    before_main = main.detach().clone()
    before_score = score.detach().clone()
    before_packed = layer.get_stage_part_vq_storage(0, 0).detach().clone()
    before_round = manager.global_bit_round
    before_round_step = manager.bit_round_step
    scaler.step(composite)
    scaler.update()
    assert callback_calls["count"] == 0
    assert torch.equal(main.detach(), before_main)
    assert torch.equal(score.detach(), before_score)
    assert torch.equal(layer.get_stage_part_vq_storage(0, 0), before_packed)
    assert manager.global_bit_round == before_round
    assert manager.bit_round_step == before_round_step

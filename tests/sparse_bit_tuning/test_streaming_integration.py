import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.offload import OffloadedCheckpointLayer, StreamingOffloadManager
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _bits():
    return torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )


def _vae():
    decoder = Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    layer = VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=_bits(),
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    return layer


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = _vae()

    def forward(self, x):
        return self.vae(x)


def test_streaming_prefetch_checkpoint_offload_and_delayed_transition():
    device = torch.device("cuda:0")
    block = _Block().to(dtype=torch.bfloat16)
    vae = block.vae
    stream_manager = StreamingOffloadManager(
        layer_devices={0: device},
        prefetch_distance=0,
        checkpoint_layers=True,
    )
    wrapped = OffloadedCheckpointLayer(layer=block, layer_idx=0, manager=stream_manager)
    assert vae.get_stage_part_vq_storage(0, 0).device.type == "cpu"

    root = nn.Module()
    root.add_module("wrapped", wrapped)
    bit_manager = SparseBitTuningManager(
        root_model=root,
        targets=[("block.vae", vae)],
        target_devices={"block.vae": device},
        training_seed=91,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=0.5,
            round_steps=1,
        ),
        streaming=True,
    )
    bit_manager.configure_schedule(total_optimizer_steps=2)
    bit_manager.initialize_scores()

    x = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    y = wrapped(x)
    loss = y.float().square().mean()
    loss.backward()
    stream_manager.synchronize()
    assert bit_manager.score_module.score_chunks[0].grad is not None
    assert vae.get_stage_part_vq_storage(0, 0).device.type == "cpu"

    telemetry = bit_manager.optimizer_step()
    assert telemetry.round_ended
    assert bit_manager.pending_next_states
    old_hard = bit_manager.checkpoint_packed_snapshot()[bit_manager.bank_specs[0].canonical_key]
    pending_before = dict(bit_manager.pending_next_states)

    # Read-only eval must SET current hard bits onto the fresh residency but must not
    # advance the delayed transition/coverage state.
    with torch.no_grad():
        eval_out = wrapped(torch.randn(2, 4, dtype=torch.bfloat16))
    assert torch.isfinite(eval_out.float()).all()
    assert bit_manager.pending_next_states == pending_before
    stream_manager.offload_all(synchronize=True)
    assert vae.get_stage_part_vq_storage(0, 0).device.type == "cpu"

    # The next real training forward first projects the old-round hard state, commits it,
    # then initializes the next subset. This is the delayed transition boundary.
    for score in bit_manager.score_module.score_chunks:
        score.grad = None
    x2 = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    y2 = wrapped(x2)
    # Reentrant checkpoint runs the first forward under no_grad, so transition must
    # remain pending until the grad-enabled backward recompute.
    assert bit_manager.pending_next_states
    y2.float().sum().backward()
    assert not bit_manager.pending_next_states
    stream_manager.synchronize()
    persistent = vae.get_stage_part_vq_storage(0, 0).detach().cpu().contiguous()
    assert torch.equal(persistent, old_hard)
    assert bit_manager.score_module.score_chunks[0].grad is not None

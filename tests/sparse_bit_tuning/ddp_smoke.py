import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager


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


def _layer():
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
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=_bits(),
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _layer()

    def forward(self, x):
        return self.layer(x)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"DDP smoke expects world_size=2, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(1234)
    model = _Root().to(device=device, dtype=torch.bfloat16)
    model.layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    manager = SparseBitTuningManager(
        root_model=model,
        targets=[("layer", model.layer)],
        target_devices={"layer": device},
        training_seed=77,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=0.5,
            round_steps=3,
        ),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=3)
    manager.initialize_scores()

    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    torch.manual_seed(9000 + rank)
    x = torch.randn(3, 4, device=device, dtype=torch.bfloat16)
    loss = ddp(x).float().square().mean()
    loss.backward()

    score = manager.score_module.score_chunks[0]
    if score.grad is None:
        raise RuntimeError(f"rank{rank}: score.grad is None")
    gathered_grad = [torch.empty_like(score.grad) for _ in range(world)]
    dist.all_gather(gathered_grad, score.grad)
    if not torch.equal(gathered_grad[0], gathered_grad[1]):
        diff = (gathered_grad[0].float() - gathered_grad[1].float()).abs().max().item()
        raise RuntimeError(f"DDP score gradients differ across ranks, max_abs={diff}")

    telemetry = manager.optimizer_step()
    flip = torch.tensor([telemetry.step_flip_count], device=device, dtype=torch.int64)
    gathered_flip = [torch.empty_like(flip) for _ in range(world)]
    dist.all_gather(gathered_flip, flip)
    if int(gathered_flip[0].item()) != int(gathered_flip[1].item()):
        raise RuntimeError(f"DDP flip counts differ: {[int(x.item()) for x in gathered_flip]}")

    packed = model.layer.get_stage_part_vq_storage(0, 0).detach().contiguous()
    gathered_packed = [torch.empty_like(packed) for _ in range(world)]
    dist.all_gather(gathered_packed, packed)
    if not torch.equal(gathered_packed[0], gathered_packed[1]):
        raise RuntimeError("DDP packed hard state differs across ranks after Bit step")

    if rank == 0:
        print(
            f"DDP_SBT_OK grad_numel={score.grad.numel()} flip_count={telemetry.step_flip_count} "
            f"round={telemetry.global_bit_round}"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

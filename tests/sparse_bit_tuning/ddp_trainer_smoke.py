import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager


class _Dataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        ids = torch.tensor([1 + (idx % 3), 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


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


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layer = VAELinear(
            in_features=4,
            out_features=4,
            bias=None,
            original_weight=None,
            vq_weight=_bits(),
            decoder=Decoder(
                in_dim=9,
                out_dim=4,
                hidden_dim=8,
                num_res_blocks=0,
                norm_type="layer",
                decoder_type="linear",
                use_checkpoint=False,
                num_models=1,
            ),
            codebook_dim=4,
            transpose=False,
        )
        self.lm_head = nn.Linear(4, 8, bias=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        del attention_mask, kwargs
        hidden = self.layer(self.embed(input_ids))
        return {"logits": self.lm_head(hidden), "hidden_states": (hidden,) if output_hidden_states else None}


def main():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(2026)

    model = _TinyLM().to(device=device, dtype=torch.bfloat16)
    model.layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    manager = SparseBitTuningManager(
        root_model=model,
        targets=[("layer", model.layer)],
        target_devices={"layer": device},
        training_seed=2026,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=0.5,
            round_steps=3,
        ),
        streaming=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=1,
            max_steps=1,
            save_strategy="no",
            logging_steps=1,
            learning_rate=1e-3,
            bf16=True,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        trainer = VAEDecoderE2ETrainer(
            model=model,
            args=args,
            train_dataset=_Dataset(),
            loss_type="sft",
            sparse_bit_manager=manager,
        )
        result = trainer.train()
        if int(result.global_step) != 1 or manager.bit_round_step != 1:
            raise RuntimeError(
                f"rank{rank}: trainer/bit step mismatch global={result.global_step} bit={manager.bit_round_step}"
            )
        score = manager.score_module.score_chunks[0].detach()
        gathered_score = [torch.empty_like(score) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_score, score)
        if not torch.equal(gathered_score[0], gathered_score[1]):
            raise RuntimeError("HF Trainer DDP produced different Sparse Bit scores across ranks")
        packed = model.layer.get_stage_part_vq_storage(0, 0).detach().contiguous()
        gathered_packed = [torch.empty_like(packed) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_packed, packed)
        if not torch.equal(gathered_packed[0], gathered_packed[1]):
            raise RuntimeError("HF Trainer DDP produced different packed bits across ranks")
        if rank == 0:
            print("DDP_TRAINER_SBT_OK global_step=1 bit_round_step=1")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

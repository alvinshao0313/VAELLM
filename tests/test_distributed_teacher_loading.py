import pytest
import torch
import torch.multiprocessing as mp
from transformers import Qwen3Config, Qwen3ForCausalLM


def _distributed_teacher_worker(rank: int, world_size: int, init_file: str, model_dir: str) -> None:
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import train_utils.base_reference as base_reference

        if rank != 0:
            def fail_if_rank_reads_checkpoint(*_args, **_kwargs):
                raise AssertionError("nonzero rank must not read the teacher checkpoint")

            base_reference.model_utils.get_model = fail_if_rank_reads_checkpoint

        model = base_reference.load_frozen_base_reference_model_distributed(
            model_dir,
            access_token=None,
            device=torch.device(f"cuda:{rank}"),
            dtype=torch.bfloat16,
            logger=None,
        )
        first_parameter = next(model.parameters())
        assert first_parameter.device == torch.device(f"cuda:{rank}")
        assert first_parameter.dtype == torch.bfloat16
        assert model.training is False
        assert all(not parameter.requires_grad for parameter in model.parameters())

        checksum = torch.zeros((), device=first_parameter.device, dtype=torch.float64)
        for parameter in model.parameters():
            checksum += parameter.detach().double().sum()
        checksums = [torch.empty_like(checksum) for _ in range(world_size)]
        torch.distributed.all_gather(checksums, checksum)
        assert all(torch.equal(checksums[0], value) for value in checksums[1:])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_distributed_teacher_loads_checkpoint_only_on_rank0(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    model_dir = tmp_path / "tiny_teacher"
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    torch.manual_seed(123)
    Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(model_dir)

    init_file = tmp_path / "nccl_init"
    mp.spawn(
        _distributed_teacher_worker,
        args=(2, str(init_file), str(model_dir)),
        nprocs=2,
        join=True,
    )

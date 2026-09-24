import pytest
import torch
from torch import nn

from experiments.linear_output.config import parser
from experiments.linear_output.incremental import LinearTrainer
from experiments.linear_output.artifacts import linear_host, load_linear, state_digest


def _args(tmp_path, objective="linear_output_mse"):
    args = parser().parse_args(["--output_dir", str(tmp_path), "--device", "cpu", "--steps", "2", "--objective", objective])
    args.vae_autocast_dtype = "fp32"
    args.base_ch = 8
    args.decoder_base_ch = 8
    args.decoder_num_res_blocks = 0
    args.vae_chunk_vectors = 64
    args.log_every = 1
    return args


def test_incremental_step_and_cpu_resume_match(tmp_path):
    torch.manual_seed(12)
    source = nn.Linear(64, 4, bias=False)
    inputs = torch.randn(17, 64)
    torch.manual_seed(100)
    args_a = _args(tmp_path / "a")
    trainer_a = LinearTrainer(source, "toy", args_a, tmp_path / "a")
    first = trainer_a.step(inputs, 1)
    state = trainer_a.state_dict()
    assert all(not value.is_cuda for value in state["vae"].values() if isinstance(value, torch.Tensor))

    torch.manual_seed(100)
    args_b = _args(tmp_path / "b")
    trainer_b = LinearTrainer(source, "toy", args_b, tmp_path / "b")
    trainer_b.load_state_dict(state)
    second_resume = trainer_b.step(inputs, 2)

    torch.manual_seed(100)
    args_c = _args(tmp_path / "c")
    trainer_c = LinearTrainer(source, "toy", args_c, tmp_path / "c")
    trainer_c.step(inputs, 1)
    second_continuous = trainer_c.step(inputs, 2)
    torch.testing.assert_close(trainer_b.vae.state_dict()["model.encoder.linear_in.linear.weight"], trainer_c.vae.state_dict()["model.encoder.linear_in.linear.weight"])
    assert first["step"] == 1
    assert second_resume["step"] == second_continuous["step"] == 2


def test_incremental_weight_objective_and_export_roundtrip(tmp_path):
    torch.manual_seed(3)
    source = nn.Linear(64, 4, bias=False)
    args = _args(tmp_path, objective="weight_mse")
    trainer = LinearTrainer(source, "toy", args, tmp_path)
    trainer.step(torch.randn(13, 64), 1)
    trainer.step(torch.randn(13, 64), 2)
    state_before = state_digest(trainer.vae)
    assert trainer.vae.training
    record = trainer.export()
    assert record["code_payload_bpw"] == 2.0
    assert record["packed_roundtrip_max_abs"] == 0.0
    assert record["fp32_unfused_to_packed_relative_l2"] < record["fp32_export_tolerance"]
    assert record["export_validation_decoder_dtype"] == "torch.float32"
    assert trainer.vae.training
    assert state_digest(trainer.vae) == state_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 precision separation")
def test_bf16_export_fixes_bits_and_separates_precision_delta(tmp_path):
    torch.manual_seed(4)
    source = nn.Linear(64, 4, bias=True)
    args = _args(tmp_path, objective="weight_mse")
    args.device = "cuda:0"
    args.vae_autocast_dtype = "bf16"
    args.base_ch = 128
    args.decoder_base_ch = 128
    args.decoder_num_res_blocks = 1
    trainer = LinearTrainer(source, "toy", args, tmp_path)
    trainer.step(torch.randn(13, 64), 1)
    trainer.step(torch.randn(13, 64), 2)
    training_weight, bits = trainer._decode_cpu()
    expected_fp32 = trainer._decode_fp32_from_bits(bits)
    state_before = state_digest(trainer.vae)
    record = trainer.export()
    reloaded = load_linear("toy", source, tmp_path / "packed").cuda()
    with torch.no_grad():
        actual = reloaded._decode_weight(dtype=torch.float32).cpu()
        packed_fp32 = reloaded.cpu()._decode_weight(dtype=torch.float32)
    relative = float((packed_fp32 - expected_fp32).norm() / expected_fp32.norm())
    assert relative < 5e-6
    deployment_delta = float((actual - packed_fp32).norm() / packed_fp32.norm())
    assert record["deployment_to_fp32_relative_l2"] == pytest.approx(deployment_delta)
    actual_precision_delta = float((actual - training_weight).norm() / training_weight.norm())
    assert actual_precision_delta > 0
    assert record["training_to_packed_relative_l2"] == pytest.approx(actual_precision_delta)
    assert record["training_to_packed_relative_l2"] > record["fp32_unfused_to_packed_relative_l2"]
    _, bits_after = trainer._decode_cpu()
    assert torch.equal(bits, bits_after)
    assert trainer.vae.training
    assert state_digest(trainer.vae) == state_before


@pytest.mark.parametrize("name", [
    "toy",
    "model.layers.0.self_attn.q_proj",
    "model.layers.9.self_attn.q_proj",
    "model.layers.35.mlp.down_proj",
])
def test_linear_host_preserves_keys_and_native_decoder_inventory(name):
    from e2e_common.residual_lora import get_residual_lora_topology

    source = nn.Linear(64, 4, bias=True)
    host = linear_host(name, source)
    assert isinstance(host.model.layers, nn.ModuleList)
    assert get_residual_lora_topology(host) is None
    assert set(host.state_dict()) == {f"{name}.weight", f"{name}.bias"}
    torch.testing.assert_close(host.get_submodule(name).weight, source.weight)
    torch.testing.assert_close(host.get_submodule(name).bias, source.bias)

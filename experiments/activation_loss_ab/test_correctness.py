"""Mathematical checks and real-weight CAT parity; not downstream evidence."""
import json

import pytest
import torch

from experiments.activation_loss_ab.objectives import block_gram_sum, block_output_mse, gather_grams


def test_block_loss_matches_direct_output_and_gradient():
    torch.manual_seed(7)
    x = torch.randn(71, 96) + 0.3
    grams = block_gram_sum(x) / len(x)
    ids = torch.tensor([8, 0, 4, 2, 7])  # shuffled blocks, several output rows
    target = torch.randn(5, 1, 32)
    recon = torch.randn(5, 1, 32, requires_grad=True)
    actual = block_output_mse(recon, target, grams, ids)
    selected_x = x.reshape(len(x), 3, 32)[:, ids % 3]
    output_error = (selected_x * (recon - target).squeeze(1)).sum(-1)
    expected = output_error.square().mean() / 32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    grad_actual = torch.autograd.grad(actual, recon, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected, recon)[0]
    torch.testing.assert_close(grad_actual, grad_expected, rtol=3e-6, atol=1e-7)


def test_diagonal_matches_existing_amse_formula():
    torch.manual_seed(19)
    diagonal = torch.rand(3, 32) + 0.1
    grams = torch.diag_embed(diagonal)
    ids = torch.tensor([3, 8, 4, 0])
    target = torch.randn(4, 1, 32)
    recon = torch.randn_like(target)
    expected = ((recon - target).square() * diagonal[ids % 3].unsqueeze(1)).mean()
    torch.testing.assert_close(block_output_mse(recon, target, grams, ids), expected)
    identity = torch.eye(32).expand(3, 32, 32)
    torch.testing.assert_close(block_output_mse(recon, target, identity, ids), (recon - target).square().mean())


def test_invalid_layout_fails():
    with pytest.raises(ValueError):
        block_gram_sum(torch.randn(8, 33))
    with pytest.raises(ValueError):
        block_output_mse(torch.randn(3, 2, 32), torch.randn(3, 2, 32), torch.eye(32)[None], torch.arange(3))


@pytest.mark.parametrize("mode", ["mse", "amse"])
def test_restricted_training_matches_production_cat(mode):
    if not torch.cuda.is_available():
        pytest.skip("Real CAT training parity requires CUDA.")
    from experiments.activation_loss_ab.training import configure, train_weight
    from experiments.weight_rotation_ab import _host_for_weight, load_weight_patch
    from train_utils.cat_train_pipeline import train_group_vae_payload, apply_group_vae_payload
    from train_utils.utils import LinearRef

    weight, name, _ = load_weight_patch("Qwen/Qwen3-8B", 0, "up_proj", 128, 64)
    grams = torch.diag_embed(torch.linspace(0.5, 1.5, 64).reshape(2, 32))
    seed, steps, batch = 31, 4, 96  # covers tail minibatch, reshuffle and two stages
    ours, record = train_weight(weight, module_name=name, mode=mode, grams=grams,
                                seed=seed, steps=steps, batch_size=batch, device="cuda")
    torch.manual_seed(seed)
    host, linear = _host_for_weight(weight, name)
    cfg, vae_args, training_args = configure("up_proj", mode, seed, steps, batch)
    refs = [LinearRef(name, linear, "up_proj", False)]
    runtime = {"stats_by_linear": {name: {"sq_mean": grams.diagonal(dim1=-2, dim2=-1).flatten()}}}
    payload = train_group_vae_payload(
        model=host, group_refs=refs, group_tag="parity", runtime_cfg=cfg,
        vae_args=vae_args, training_args=training_args, train_device="cuda", convert_device="cuda",
        do_convert=True, batch_size=batch, gpu_resident_data=True,
        log_every=0, eval_every=0, eval_blocks=32, channel_protect_mode="none",
        channel_rank_metric="channel_weight_abs", channel_axis="input", channel_quant="int8",
        deterministic=True, shuffle_seed=seed, activation_runtime=runtime,
    )
    apply_group_vae_payload(model=host, group_refs=refs, group_tag="parity", payload=payload, convert_device="cuda")
    with torch.no_grad():
        expected = host.get_submodule(name).to("cuda")._decode_weight(dtype=torch.float32).cpu()
    torch.testing.assert_close(ours, expected, rtol=0, atol=0)
    assert record["code_payload_bpw"] == 2.0


def test_new_objective_two_stage_real_weight_training():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")
    from experiments.activation_loss_ab.training import train_weight
    from experiments.weight_rotation_ab import load_weight_patch
    weight, name, _ = load_weight_patch("Qwen/Qwen3-8B", 0, "up_proj", 128, 64)
    torch.manual_seed(17)
    inputs = torch.randn(128, 64)
    grams = block_gram_sum(inputs) / len(inputs)
    result, record = train_weight(weight, module_name=name, mode="block_output", grams=grams,
                                 seed=31, steps=4, batch_size=96, device="cuda")
    assert result.shape == weight.shape and torch.isfinite(result).all()
    assert record["code_payload_bpw"] == 2.0 and len(record["stages"]) == 2


def test_lm_eval_json_serialization(tmp_path):
    from experiments.activation_loss_ab.evaluation import dump_json
    path = tmp_path / "result.json"
    dump_json(path, {"dtype": torch.bfloat16, "device": torch.device("cpu"), "score": torch.tensor(0.5)})
    assert json.loads(path.read_text()) == {"dtype": "torch.bfloat16", "device": "cpu", "score": 0.5}


def test_paired_statistics_and_document_alignment():
    from experiments.activation_loss_ab.evaluation import TASKS, paired_comparison
    def result(values):
        return {"items": {task: [
            {"doc_id": i, "doc_sha256": str(i), "correct": value}
            for i, value in enumerate(values)
        ] for task in TASKS}}
    reference, candidate = result([1, 0, 1, 0]), result([0, 1, 1, 1])
    comparison = paired_comparison(reference, candidate)
    assert comparison["macro_delta_pp"] == 25.0
    assert comparison["candidate_only_correct"] == 6
    assert comparison["reference_only_correct"] == 3
    assert comparison["mcnemar_exact_two_sided_p"] == 0.5078125
    candidate["items"][TASKS[0]][0]["doc_sha256"] = "different"
    with pytest.raises(ValueError, match="document order"):
        paired_comparison(reference, candidate)

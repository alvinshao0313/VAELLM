"""CPU loss wiring and real-archive denominators for the isolated CE50 arms."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from e2e_common.dense_loss import compute_dense_loss_from_offloaded_teacher
from experiments.qwen3_w25.format_recovery_diagnostic import (
    configure_loss_recipe, loss_denominators, summarize_heldout,
)
from train_utils.config.configs import DistillLossConfig


ARCHIVE = Path("/home/star/VAELLM/result/qwen3_8b_w25_20260930/format_recovery_data/batches.pt")
ARCHIVE_SHA = "1e418eca18e91e032b5cd5b3f2ec4fad444e83a300838e0273944a35e982267c"
KL_PROTOCOL = {"loss_type": "kl_top_mass", "top_k": 100, "temperature": 1, "prompt_loss_weight": .3}
CE_PROTOCOL = {**KL_PROTOCOL, "loss_type": "kd_top_mass", "alpha": .5, "prompt_loss_weight": 0}


def test_explicit_existing_loss_recipes_preserve_unrelated_configuration():
    original = DistillLossConfig(loss_type="kl_top_mass", prompt_loss_weight=.3)
    same = deepcopy(original)
    recipe = configure_loss_recipe(same, KL_PROTOCOL)
    assert asdict(same) == asdict(original)
    assert recipe["loss_type"] == "kl_top_mass" and recipe["prompt_loss_weight"] == .3
    ce = deepcopy(original)
    recipe = configure_loss_recipe(ce, CE_PROTOCOL)
    expected = {**asdict(original), "loss_type": "kd_top_mass", "alpha": .5, "prompt_loss_weight": 0.}
    assert asdict(ce) == expected
    assert {key: recipe[key] for key in CE_PROTOCOL} == CE_PROTOCOL
    for bad in ({**CE_PROTOCOL, "alpha": .1}, {**CE_PROTOCOL, "prompt_loss_weight": .3},
                {key: value for key, value in CE_PROTOCOL.items() if key != "alpha"}):
        with pytest.raises(ValueError):
            configure_loss_recipe(deepcopy(original), bad)


def test_ce50_offloaded_loss_value_and_gradient_use_only_causal_response_tokens():
    generator = torch.Generator().manual_seed(17)
    student = torch.randn(2, 5, 128, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 128, generator=generator, requires_grad=True)
    batch = {"input_ids": torch.tensor([[3, 5, 7, 9, 0], [2, 4, 6, 8, 10]]),
             "labels": torch.tensor([[-100, -100, 7, 9, -100], [-100, -100, -100, 8, 10]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])}
    config = DistillLossConfig(loss_type="kl_top_mass", prompt_loss_weight=.3)
    configure_loss_recipe(config, CE_PROTOCOL)
    actual = compute_dense_loss_from_offloaded_teacher(
        loss_type=config.loss_type, student_logits=student, teacher_logits_cpu=teacher,
        **batch, top_k=config.top_k, temperature=config.temperature, alpha=config.alpha,
        prompt_loss_weight=config.prompt_loss_weight, teacher_output_chunk_tokens=8,
    )
    actual_gradient, teacher_gradient = torch.autograd.grad(actual, (student, teacher), allow_unused=True)

    oracle_student = student.detach().clone().requires_grad_(True)
    p = teacher[:, :-1].detach().softmax(-1)
    q_log = oracle_student[:, :-1].log_softmax(-1)
    q = q_log.exp()
    indices = p.topk(100, dim=-1).indices
    selected = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, indices, True)
    top_kl = (p * (p.log() - q_log) * selected).sum(-1)
    p_other = (p * ~selected).sum(-1)
    q_other = (q * ~selected).sum(-1)
    mass_kl = top_kl + p_other * (p_other.log() - q_other.log())
    token_ce = F.cross_entropy(oracle_student[:, :-1].reshape(-1, 128),
                               batch["input_ids"][:, 1:].reshape(-1), reduction="none").reshape(2, 4)
    response = (batch["labels"][:, 1:] != -100) & batch["attention_mask"][:, 1:].bool()
    expected = ((.5 * token_ce + .5 * mass_kl) * response).sum() / response.sum()
    expected_gradient, = torch.autograd.grad(expected, oracle_student)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)
    assert teacher_gradient is None
    assert not bool(actual_gradient[:, :-1][~response].any())
    assert not bool(actual_gradient[:, -1].any())
    assert bool(actual_gradient[:, :-1][response].abs().sum() > 0)
    assert loss_denominators(batch, 0)["loss_denominator_fp32"] == 4.


def test_real_paired_archive_runtime_denominators_leave_original_metadata_unchanged():
    before = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert before == ARCHIVE_SHA
    archive = torch.load(ARCHIVE, map_location="cpu", weights_only=True)
    for variant in ("chatml", "bare_space"):
        data = archive["formats"][variant]
        for batch, manifest in zip(data["batches"], data["manifests"]):
            original = deepcopy(manifest)
            kl = loss_denominators(batch, .3)
            ce = loss_denominators(batch, 0)
            assert kl["loss_denominator_fp32"] == manifest["loss_denominator_fp32"]
            assert ce["loss_denominator_fp32"] == ce["response_denominator_fp32"] == manifest["causal_response_tokens"]
            assert ce["causal_prompt_tokens"] == manifest["causal_prompt_tokens"]
            assert ce["causal_response_tokens"] == manifest["causal_response_tokens"]
            assert ce["prompt_loss_weight"] == 0 and kl["prompt_loss_weight"] == .3
            assert manifest == original
        rows = []
        for index, (batch, manifest) in enumerate(zip(data["batches"][256:], data["manifests"][256:])):
            sequence_stats = []
            for sample in manifest["samples"]:
                tokens = len(sample["answer_positions"])
                sequence_stats.append({"answer_tokens": tokens, "answer_ce_sum": float(tokens),
                                       "answer_ce_mean": 1., "first_answer_token_ce": 1.})
            rows.append({"loss": index + .25, "denominator": loss_denominators(batch, 0)["loss_denominator_fp32"],
                         "student_answer_ce": sequence_stats, "teacher_answer_ce": sequence_stats})
        pooled = summarize_heldout(rows)["loss_denominator_weighted"]
        correct = math.fsum(row["loss"] * row["denominator"] for row in rows) / math.fsum(row["denominator"] for row in rows)
        cached = [m["loss_denominator_fp32"] for m in data["manifests"][256:]]
        wrong = math.fsum(row["loss"] * denom for row, denom in zip(rows, cached)) / math.fsum(cached)
        assert pooled == correct
        assert abs(pooled - wrong) > 1e-6
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == before

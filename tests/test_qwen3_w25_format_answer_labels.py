"""CPU checks of real archive label selection and production CE50 gradients."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from e2e_common.dense_loss import compute_dense_loss_from_offloaded_teacher
from experiments.qwen3_w25.format_answer_labels import answer_labels, derive_training_labels
from experiments.qwen3_w25.format_recovery_diagnostic import loss_denominators
from experiments.qwen3_w25.format_training_data import validate_format_training_archive


ARCHIVE = Path("/home/star/VAELLM/result/qwen3_8b_w25_20260930/format_recovery_data/batches.pt")
ARCHIVE_SHA = "1e418eca18e91e032b5cd5b3f2ec4fad444e83a300838e0273944a35e982267c"
RECIPE = {"loss_type": "kd_top_mass", "alpha": .5, "prompt_loss_weight": 0}


def test_real_archive_exact_answer_labels_preserve_inputs_order_heldout_and_default():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
    archive = torch.load(ARCHIVE, map_location="cpu", weights_only=True)
    validate_format_training_archive(archive)
    data = archive["formats"]["chatml"]
    original_labels = [batch["labels"].clone() for batch in data["batches"]]
    manifests = deepcopy(data["manifests"])
    default, default_audit = derive_training_labels(data, training_label_region="response",
                                                    training_format="chatml", loss_recipe=RECIPE)
    assert default is data
    assert default_audit["original_training_labels_sha256"] == default_audit["derived_training_labels_sha256"]
    selected, audit = derive_training_labels(data, training_label_region="answer",
                                             training_format="chatml", loss_recipe=RECIPE)
    assert audit["original_response_tokens"] == 11911
    assert audit["selected_training_tokens"] == audit["answer_tokens"] == 4743
    assert audit["removed_response_tokens"] == 7168
    assert audit["training_sequences"] == 1024
    assert audit["unchanged_training_inputs_and_attention_sha256"] == default_audit["unchanged_training_inputs_and_attention_sha256"]
    assert audit["derived_training_labels_sha256"] != audit["original_training_labels_sha256"]
    assert selected["manifests"] is data["manifests"] and data["manifests"] == manifests
    tokens = 0
    for index, (batch, derived, manifest) in enumerate(zip(data["batches"], selected["batches"], data["manifests"])):
        assert torch.equal(batch["labels"], original_labels[index])
        for key in ("input_ids", "attention_mask"):
            assert derived[key] is batch[key]
        if index >= 256:
            assert derived is batch
            continue
        for row, sample in enumerate(manifest["samples"]):
            expected = torch.zeros_like(batch["labels"][row], dtype=torch.bool)
            expected[sample["answer_positions"]] = True
            assert torch.equal(derived["labels"][row] != -100, expected)
            assert torch.equal(derived["labels"][row][expected], batch["input_ids"][row][expected])
        tokens += loss_denominators(derived, 0)["loss_denominator_fp32"]
    assert tokens == 4743
    assert sum(loss_denominators(batch, 0)["loss_denominator_fp32"] for batch in selected["batches"][256:]) == 323
    assert sum(len(sample["answer_positions"]) for manifest in selected["manifests"][256:] for sample in manifest["samples"]) == 99
    for variant, recipe in (("bare_space", RECIPE), ("chatml", {**RECIPE, "prompt_loss_weight": .3})):
        with pytest.raises(ValueError):
            derive_training_labels(data, training_label_region="answer", training_format=variant, loss_recipe=recipe)
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA


def test_production_ce50_answer_mask_has_one_shift_exact_denominator_and_zero_wrapper_gradient():
    ids = torch.tensor([[3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12]])
    original = {"input_ids": ids, "labels": torch.cat((torch.full((2, 2), -100), ids[:, 2:]), 1),
                "attention_mask": torch.ones_like(ids)}
    manifest = {"samples": [{"answer_positions": [3]}, {"answer_positions": [3, 4]}]}
    batch = answer_labels(original, manifest)
    assert batch["input_ids"] is ids and batch["attention_mask"] is original["attention_mask"]
    assert loss_denominators(original, 0)["loss_denominator_fp32"] == 8.
    assert loss_denominators(batch, 0)["loss_denominator_fp32"] == 3.
    generator = torch.Generator().manual_seed(71)
    student = torch.randn(2, 6, 128, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 6, 128, generator=generator, requires_grad=True)
    actual = compute_dense_loss_from_offloaded_teacher(
        loss_type="kd_top_mass", student_logits=student, teacher_logits_cpu=teacher, **batch,
        top_k=100, temperature=1., alpha=.5, prompt_loss_weight=0., teacher_output_chunk_tokens=8)
    actual_grad, teacher_grad = torch.autograd.grad(actual, (student, teacher), allow_unused=True)
    reference = student.detach().clone().requires_grad_(True)
    p = teacher[:, :-1].detach().softmax(-1)
    q_log = reference[:, :-1].log_softmax(-1)
    selected = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, p.topk(100, -1).indices, True)
    p_other, q_other = (p * ~selected).sum(-1), (q_log.exp() * ~selected).sum(-1)
    kl = (p * (p.log()-q_log) * selected).sum(-1) + p_other * (p_other.log()-q_other.log())
    ce = F.cross_entropy(reference[:, :-1].reshape(-1, 128), ids[:, 1:].reshape(-1), reduction="none").reshape(2, 5)
    target = torch.zeros((2, 5), dtype=torch.bool)
    target[0, 2], target[1, 2:4] = True, True
    expected = (.5 * ce[target] + .5 * kl[target]).mean()
    expected_grad, = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-7)
    assert teacher_grad is None
    assert not bool(actual_grad[:, :-1][~target].any())
    assert not bool(actual_grad[:, -1].any())
    assert bool((actual_grad[:, :-1][target].abs().sum(-1) > 0).all())

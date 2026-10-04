"""CPU contracts for causal regional reporting, not model quality or GPU claims."""
from copy import deepcopy

import pytest
import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import batch_manifest
from experiments.qwen3_w25.loss_region_diagnostic import (
    ARCHIVE_SHA256, MODEL_NAMES, aggregate_rows, check_inputs, compare_teacher_observations,
    region_rows, summarize_regions,
)
from train_utils.distill_loss_core import (
    compute_kl_top_mass_token_loss, compute_model_level_loss, compute_sft_token_loss,
)


def _batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 10, 11]]),
            "labels": torch.tensor([[-100, -100, -100, 4, 5, -100], [-100, -100, -100, -100, -100, 11]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])}


def test_causal_regions_ignore_padding_and_preserve_exact_response_token_positions():
    batch = _batch()
    kl = torch.tensor([[1., 2., 3., 4., 999.], [5., 6., 7., 8., 9.]])
    rows, check = region_rows(kl, kl * 2, kl * 3, batch, batch_manifest(batch, 256, sample_offset=1024))
    assert rows[0]["causal_response_tokens"] == 2
    assert rows[0]["causal_prompt_tokens"] == 2
    assert rows[0]["kl_response_sum"] == 7
    assert rows[0]["kl_prompt_sum"] == 3
    assert rows[0]["ce_response_sum"] == 14
    assert rows[0]["teacher_ce_response_sum"] == 21
    assert rows[0]["response_tokens"] == [
        {"prediction_position": 2, "target_position": 3, "label_id": 4,
         "student_mass_kl": 3., "student_label_ce": 6., "teacher_label_ce": 9.},
        {"prediction_position": 3, "target_position": 4, "label_id": 5,
         "student_mass_kl": 4., "student_label_ce": 8., "teacher_label_ce": 12.}]
    assert check["reconstructed_fp64_loss"] == pytest.approx((16 + .3 * 29) / (3 + .3 * 6))
    summary = summarize_regions(rows, [check])
    assert summary["all"]["kl_response_token_mean"] == pytest.approx(16 / 3)
    assert summary["all"]["kl_response_sequence_mean"] == 6.25
    assert summary["response_ge100"] == {"sequences": 0}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_regions(rows + rows, [check])


def test_existing_mass100_and_ce_primitives_recompose_the_actual_loss():
    torch.manual_seed(17)
    batch = _batch()
    student, teacher = torch.randn(2, 6, 128), torch.randn(2, 6, 128)
    kl = compute_kl_top_mass_token_loss(student_logits=student[:, :-1], teacher_logits=teacher[:, :-1],
                                       temperature=1., top_k=100)
    ce = compute_sft_token_loss(student_logits=student, input_ids=batch["input_ids"])
    tc = compute_sft_token_loss(student_logits=teacher, input_ids=batch["input_ids"])
    rows, check = region_rows(kl, ce, tc, batch, batch_manifest(batch, 256, sample_offset=1024))
    expected = compute_model_level_loss(loss_type="kl_top_mass", student_logits=student, teacher_logits=teacher,
        input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"],
        temperature=1., top_k=100, prompt_loss_weight=.3)
    assert check["production_fp32_loss"] == float(expected)
    assert aggregate_rows(rows)["kl_original_token_weighted"] == pytest.approx(float(expected), rel=3e-6)
    with pytest.raises(ValueError, match="one causal shift"):
        region_rows(kl[:, 1:], ce, tc, batch, batch_manifest(batch, 256, sample_offset=1024))
    kl[0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        region_rows(kl, ce, tc, batch, batch_manifest(batch, 256, sample_offset=1024))


def test_equal_batch_sequence_and_token_summaries_are_distinct():
    batch = _batch()
    rows, _ = region_rows(torch.ones(2, 5), torch.ones(2, 5), torch.ones(2, 5), batch,
                          batch_manifest(batch, 256, sample_offset=1024))
    third = deepcopy(rows[0])
    third.update(archive_batch_index=257, causal_response_tokens=100, causal_prompt_tokens=2,
                 kl_response_sum=300., kl_response_mean=3., kl_prompt_sum=6., kl_prompt_mean=3.,
                 ce_response_sum=300., ce_response_mean=3., teacher_ce_response_sum=100.,
                 teacher_ce_response_mean=1., kl_weighted_sequence_mean=3., length_group="response_ge100")
    summary = aggregate_rows(rows + [third])
    assert summary["kl_equal_batch"] == 2.
    assert summary["kl_equal_sequence"] == pytest.approx(5 / 3)
    assert summary["kl_original_token_weighted"] == pytest.approx((4.8 + 301.8) / (4.8 + 100.6))
    assert summary["kl_original_token_weighted"] > summary["kl_equal_batch"] > summary["kl_equal_sequence"]


def test_unbound_or_duplicate_model_manifest_cannot_start_loading():
    record = {"archive": {"sha256": ARCHIVE_SHA256},
              "models": [{"name": name, "binding_status": "pending"} for name in MODEL_NAMES]}
    with pytest.raises(ValueError, match="must be bound"):
        check_inputs(record)
    record["models"][-1]["name"] = "decoder"
    with pytest.raises(ValueError, match="unique"):
        check_inputs(record)


def test_empty_response_is_reported_not_dropped_and_batch_denominator_is_clamped():
    batch = {"input_ids": torch.tensor([[1, 0]]), "labels": torch.tensor([[-100, -100]]),
             "attention_mask": torch.tensor([[1, 0]])}
    one = torch.ones(1, 1)
    rows, check = region_rows(one, one, one, batch, batch_manifest(batch, 256, sample_offset=1024))
    assert len(rows) == 1 and rows[0]["response_tokens"] == []
    assert rows[0]["kl_response_mean"] is rows[0]["ce_response_mean"] is None
    assert check["production_fp32_denominator"] == 1.
    summary = summarize_regions(rows, [check])["all"]
    assert summary["sequences"] == summary["zero_response_sequences"] == 1
    assert summary["zero_response_hashes"] == [rows[0]["input_and_label_sha256"]]
    assert summary["nonempty_response_sequences"] == 0
    assert summary["kl_original_token_weighted"] == summary["kl_equal_batch"] == 0.
    assert summary["kl_response_sequence_mean"] is summary["ce_response_sequence_mean"] is None


def test_repeated_teacher_ce_differences_are_observed_not_rejected():
    batch = _batch()
    one = torch.ones(2, 5)
    rows, _ = region_rows(one, one, one, batch, batch_manifest(batch, 256, sample_offset=1024))
    models = {name: {"sequences": deepcopy(rows)} for name in MODEL_NAMES}
    models["joint"]["sequences"][0]["response_tokens"][0]["teacher_label_ce"] += .01
    comparison = compare_teacher_observations(models)
    assert comparison["numerical_acceptance_threshold"] is None
    assert comparison["comparisons"]["width_h128"]["max_absolute_difference"] == 0
    assert comparison["comparisons"]["joint"]["max_absolute_difference"] == pytest.approx(.01)
    models["joint"]["sequences"][0]["response_tokens"][0]["label_id"] = 99
    with pytest.raises(ValueError, match="same response token identities"):
        compare_teacher_observations(models)

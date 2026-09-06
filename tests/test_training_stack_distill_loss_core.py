import math

import pytest
import torch
import torch.nn.functional as F

from train_utils.distill_loss_core import (
    MODEL_LEVEL_LOSS_TYPES,
    build_prediction_token_masks,
    compute_kl_token_loss,
    compute_kl_top_token_loss,
    compute_model_level_loss,
    compute_sft_token_loss,
    reduce_weighted_token_loss,
)


def _manual_forward_kl(student_logits, teacher_logits, temperature: float):
    temp = float(temperature)
    q = F.softmax(teacher_logits.float() / temp, dim=-1)
    log_p = F.log_softmax(student_logits.float() / temp, dim=-1)
    return (q * (torch.log(q.clamp_min(1e-12)) - log_p)).sum(dim=-1) * (temp * temp)


def test_model_level_loss_types_are_exactly_five():
    assert MODEL_LEVEL_LOSS_TYPES == ("sft", "kl", "kl_top", "kd", "kd_top")


def test_prediction_masks_use_target_token_positions():
    labels = torch.tensor([[-100, -100, 5, 6, -100]], dtype=torch.long)
    attention = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    response_mask, prompt_mask = build_prediction_token_masks(
        labels=labels,
        attention_mask=attention,
    )
    assert response_mask.shape == (1, 4)
    assert prompt_mask.shape == (1, 4)
    # target positions 1..4 from labels[:, 1:]
    assert response_mask.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert prompt_mask.tolist() == [[1.0, 0.0, 0.0, 0.0]]


def test_weighted_reduction_is_global_mean_not_region_sum():
    token_loss = torch.tensor([[1.0, 3.0, 5.0, 7.0]], dtype=torch.float32)
    response = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    prompt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    only_response = reduce_weighted_token_loss(
        token_loss,
        response_mask=response,
        prompt_mask=prompt,
        prompt_loss_weight=0.0,
    )
    assert torch.isclose(only_response, torch.tensor(2.0))

    equal = reduce_weighted_token_loss(
        token_loss,
        response_mask=response,
        prompt_mask=prompt,
        prompt_loss_weight=1.0,
    )
    assert torch.isclose(equal, torch.tensor((1 + 3 + 5 + 7) / 4.0))

    boosted = reduce_weighted_token_loss(
        token_loss,
        response_mask=response,
        prompt_mask=prompt,
        prompt_loss_weight=2.0,
    )
    expected = (1 + 3 + 2 * 5 + 2 * 7) / (1 + 1 + 2 + 2)
    assert torch.isclose(boosted, torch.tensor(expected))


def test_sft_ce_uses_input_ids_targets_not_shifted_labels():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7, dtype=torch.float32)
    input_ids = torch.randint(0, 7, (2, 5), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100

    token_loss = compute_sft_token_loss(student_logits=logits, input_ids=input_ids)
    assert token_loss.shape == (2, 4)
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, 7),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(2, 4)
    assert torch.allclose(token_loss, expected)

    # Even when labels are -100 on prompt, CE targets remain input_ids.
    loss = compute_model_level_loss(
        loss_type="sft",
        student_logits=logits,
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(labels),
        prompt_loss_weight=1.0,
    )
    weights = torch.ones(2, 4)
    expected_loss = (expected * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected_loss)


def test_forward_kl_matches_plan_formula():
    torch.manual_seed(1)
    student = torch.randn(2, 4, 11, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 4, 11, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.long).reshape(2, 4)
    attention = torch.ones_like(labels)
    temperature = 2.0

    loss = compute_model_level_loss(
        loss_type="kl",
        student_logits=student,
        teacher_logits=teacher,
        input_ids=labels,
        labels=labels,
        attention_mask=attention,
        temperature=temperature,
        prompt_loss_weight=0.0,
    )
    token_kl = _manual_forward_kl(student[:, :-1], teacher[:, :-1], temperature)
    expected = token_kl.mean()
    assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-5)
    loss.backward()
    assert student.grad is not None


def test_kl_top_renormalizes_teacher_topk_subset():
    torch.manual_seed(2)
    student = torch.randn(1, 3, 8, dtype=torch.float32)
    teacher = torch.randn(1, 3, 8, dtype=torch.float32)
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention = torch.ones_like(labels)
    top_k = 3
    temperature = 1.5

    loss = compute_model_level_loss(
        loss_type="kl_top",
        student_logits=student,
        teacher_logits=teacher,
        input_ids=labels,
        labels=labels,
        attention_mask=attention,
        temperature=temperature,
        top_k=top_k,
    )

    s = student[:, :-1].float() / temperature
    t = teacher[:, :-1].float() / temperature
    _, indices = t.topk(top_k, dim=-1, sorted=False)
    q = F.softmax(t.gather(-1, indices), dim=-1)
    log_p = F.log_softmax(s.gather(-1, indices), dim=-1)
    token = (q * (torch.log(q.clamp_min(1e-12)) - log_p)).sum(dim=-1) * (temperature * temperature)
    assert torch.allclose(loss, token.mean(), rtol=1e-5, atol=1e-5)


def test_kd_blends_separately_reduced_ce_and_kl():
    torch.manual_seed(3)
    student = torch.randn(2, 4, 9, dtype=torch.float32)
    teacher = torch.randn(2, 4, 9, dtype=torch.float32)
    input_ids = torch.randint(0, 9, (2, 4), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, 0] = -100
    attention = torch.ones_like(labels)
    alpha = 0.3
    temperature = 1.25
    prompt_w = 2.0

    loss = compute_model_level_loss(
        loss_type="kd",
        student_logits=student,
        teacher_logits=teacher,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        temperature=temperature,
        alpha=alpha,
        prompt_loss_weight=prompt_w,
    )

    response_mask, prompt_mask = build_prediction_token_masks(
        labels=labels,
        attention_mask=attention,
    )
    ce = compute_sft_token_loss(student_logits=student, input_ids=input_ids)
    kl = compute_kl_token_loss(
        student_logits=student[:, :-1],
        teacher_logits=teacher[:, :-1],
        temperature=temperature,
    )
    ce_r = reduce_weighted_token_loss(
        ce, response_mask=response_mask, prompt_mask=prompt_mask, prompt_loss_weight=prompt_w
    )
    kl_r = reduce_weighted_token_loss(
        kl, response_mask=response_mask, prompt_mask=prompt_mask, prompt_loss_weight=prompt_w
    )
    expected = (1.0 - alpha) * ce_r + alpha * kl_r
    assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-5)


def test_kd_top_uses_same_blend_with_topk_kl():
    torch.manual_seed(4)
    student = torch.randn(1, 5, 10, dtype=torch.float32)
    teacher = torch.randn(1, 5, 10, dtype=torch.float32)
    input_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    attention = torch.ones_like(labels)
    alpha = 0.4
    top_k = 4

    loss = compute_model_level_loss(
        loss_type="kd_top",
        student_logits=student,
        teacher_logits=teacher,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        temperature=1.0,
        alpha=alpha,
        top_k=top_k,
        prompt_loss_weight=0.0,
    )
    response_mask, prompt_mask = build_prediction_token_masks(
        labels=labels,
        attention_mask=attention,
    )
    ce = compute_sft_token_loss(student_logits=student, input_ids=input_ids)
    kl = compute_kl_top_token_loss(
        student_logits=student[:, :-1],
        teacher_logits=teacher[:, :-1],
        temperature=1.0,
        top_k=top_k,
    )
    ce_r = reduce_weighted_token_loss(
        ce, response_mask=response_mask, prompt_mask=prompt_mask, prompt_loss_weight=0.0
    )
    kl_r = reduce_weighted_token_loss(
        kl, response_mask=response_mask, prompt_mask=prompt_mask, prompt_loss_weight=0.0
    )
    expected = (1.0 - alpha) * ce_r + alpha * kl_r
    assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-5)


def test_lm_all_valid_labels_makes_prompt_weight_noop():
    torch.manual_seed(5)
    student = torch.randn(2, 3, 6, dtype=torch.float32)
    input_ids = torch.randint(0, 6, (2, 3), dtype=torch.long)
    labels = input_ids.clone()
    attention = torch.ones_like(labels)
    loss0 = compute_model_level_loss(
        loss_type="sft",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        prompt_loss_weight=0.0,
    )
    loss2 = compute_model_level_loss(
        loss_type="sft",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        prompt_loss_weight=2.0,
    )
    assert torch.allclose(loss0, loss2)


def test_rejects_removed_model_level_loss_types():
    logits = torch.randn(1, 3, 4)
    ids = torch.tensor([[1, 2, 3]])
    for removed in ("origin", "rkl", "dual_kl", "eakld", "mse", "choice"):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_model_level_loss(
                loss_type=removed,
                student_logits=logits,
                input_ids=ids,
                labels=ids,
                attention_mask=torch.ones_like(ids),
            )


@pytest.mark.parametrize("suffix_type", ["kl_top_100", "kd_top_100"])
def test_shared_core_rejects_encoded_topk_suffix(suffix_type):
    from train_utils.distill_loss_core import normalize_model_level_loss_type

    with pytest.raises(ValueError, match="Do not encode top-k"):
        normalize_model_level_loss_type(suffix_type)
    logits = torch.randn(1, 3, 4)
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="Do not encode top-k"):
        compute_model_level_loss(
            loss_type=suffix_type,
            student_logits=logits,
            input_ids=ids,
            labels=ids,
            attention_mask=torch.ones_like(ids),
            teacher_logits=logits,
            top_k=100,
        )


def test_selective_kl_top_matches_dense_gather():
    from train_utils.distill_loss_core import (
        compute_kl_top_token_loss,
        compute_selected_kl_top_model_level_loss,
        compute_selected_kl_top_token_loss,
    )

    torch.manual_seed(11)
    b, l, v, k = 2, 6, 17, 5
    student = torch.randn(b, l, v, dtype=torch.float32)
    teacher = torch.randn(b, l, v, dtype=torch.float32)
    input_ids = torch.randint(0, v, (b, l), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention = torch.ones(b, l, dtype=torch.long)

    dense = compute_model_level_loss(
        loss_type="kl_top",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        temperature=1.25,
        top_k=k,
        prompt_loss_weight=0.5,
    )

    teacher_scaled = teacher.float() / 1.25
    _, indices = teacher_scaled.topk(k, dim=-1, sorted=False)
    student_sel = student.gather(-1, indices)
    teacher_sel = teacher.gather(-1, indices)
    selective = compute_selected_kl_top_model_level_loss(
        student_selected_logits=student_sel,
        teacher_selected_logits=teacher_sel,
        labels=labels,
        attention_mask=attention,
        temperature=1.25,
        prompt_loss_weight=0.5,
    )
    torch.testing.assert_close(selective, dense, rtol=1e-5, atol=1e-6)

    # Token-wise selective path also matches dense gather before reduction.
    dense_tok = compute_kl_top_token_loss(
        student_logits=student[:, :-1],
        teacher_logits=teacher[:, :-1],
        temperature=1.25,
        top_k=k,
    )
    sel_tok = compute_selected_kl_top_token_loss(
        student_selected_logits=student_sel[:, :-1],
        teacher_selected_logits=teacher_sel[:, :-1],
        temperature=1.25,
    )
    torch.testing.assert_close(sel_tok, dense_tok, rtol=1e-5, atol=1e-6)

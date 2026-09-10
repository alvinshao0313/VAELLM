
import pytest
import torch
import torch.nn.functional as F

from train_utils.config import parse_e2e_cli
from train_utils.distill_loss_core import (
    compute_kl_token_loss,
    compute_kl_top_mass_token_loss,
    compute_kl_top_mse_token_loss,
    compute_kl_top_partial_token_loss,
    compute_selected_kl_top_mse_token_loss,
)


def _e2e(extra):
    return parse_e2e_cli(
        [
            "--student_checkpoint_dir",
            "/tmp/student",
            "--dataset_mix",
            "openorca",
            "--train_mode",
            "decoder",
            *extra,
        ]
    )


def test_kl_top_partial_matches_fullprob_topk_contribution():
    torch.manual_seed(11)
    student = torch.randn(2, 3, 13, dtype=torch.float32)
    teacher = torch.randn(2, 3, 13, dtype=torch.float32)
    temperature = 1.6
    top_k = 4

    actual = compute_kl_top_partial_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=temperature,
        top_k=top_k,
    )

    s = student / temperature
    t = teacher / temperature
    q = F.softmax(t, dim=-1)
    log_p = F.log_softmax(s, dim=-1)
    log_q = F.log_softmax(t, dim=-1)
    _, indices = t.topk(top_k, dim=-1, sorted=False)
    manual = (
        q.gather(-1, indices)
        * (log_q.gather(-1, indices) - log_p.gather(-1, indices))
    ).sum(dim=-1) * (temperature * temperature)
    torch.testing.assert_close(actual, manual, rtol=1e-5, atol=1e-6)


def test_kl_top_partial_becomes_full_kl_when_k_covers_vocab():
    torch.manual_seed(15)
    student = torch.randn(2, 4, 9, dtype=torch.float32)
    teacher = torch.randn(2, 4, 9, dtype=torch.float32)
    full = compute_kl_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=1.25,
    )
    partial = compute_kl_top_partial_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=1.25,
        top_k=9,
    )
    torch.testing.assert_close(partial, full, rtol=1e-6, atol=1e-7)


def test_kl_top_mass_matches_manual_k_plus_one_and_full_kl_topk_gradient():
    torch.manual_seed(12)
    student = torch.randn(2, 3, 13, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 3, 13, dtype=torch.float32)
    temperature = 1.4
    top_k = 4

    mass_token = compute_kl_top_mass_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=temperature,
        top_k=top_k,
    )

    s = student / temperature
    t = teacher / temperature
    q_full = F.softmax(t, dim=-1)
    p_full = F.softmax(s, dim=-1)
    _, indices = t.topk(top_k, dim=-1, sorted=False)
    q_top = q_full.gather(-1, indices)
    p_top = p_full.gather(-1, indices)
    q_other = 1.0 - q_top.sum(dim=-1, keepdim=True)
    p_other = 1.0 - p_top.sum(dim=-1, keepdim=True)
    q_coarse = torch.cat([q_top, q_other], dim=-1)
    p_coarse = torch.cat([p_top, p_other], dim=-1)
    manual = (
        q_coarse
        * (
            q_coarse.clamp_min(1e-12).log()
            - p_coarse.clamp_min(1e-12).log()
        )
    ).sum(dim=-1) * (temperature * temperature)
    torch.testing.assert_close(mass_token, manual, rtol=1e-5, atol=1e-6)

    mass_grad = torch.autograd.grad(mass_token.sum(), student, retain_graph=True)[0]
    full_token = compute_kl_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=temperature,
    )
    full_grad = torch.autograd.grad(full_token.sum(), student)[0]
    torch.testing.assert_close(
        mass_grad.gather(-1, indices),
        full_grad.gather(-1, indices),
        rtol=2e-5,
        atol=2e-6,
    )


def test_kl_top_mass_becomes_full_kl_when_k_covers_vocab():
    torch.manual_seed(14)
    student = torch.randn(2, 4, 9, dtype=torch.float32)
    teacher = torch.randn(2, 4, 9, dtype=torch.float32)
    full = compute_kl_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=1.25,
    )
    mass = compute_kl_top_mass_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=1.25,
        top_k=9,
    )
    torch.testing.assert_close(mass, full, rtol=1e-6, atol=1e-7)


def test_kl_top_mse_is_topk_kl_plus_raw_logit_mse_and_selective_matches_dense():
    torch.manual_seed(13)
    student = torch.randn(2, 4, 17, dtype=torch.float32)
    teacher = torch.randn(2, 4, 17, dtype=torch.float32)
    temperature = 1.7
    top_k = 5
    mse_weight = 0.35

    dense = compute_kl_top_mse_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=temperature,
        top_k=top_k,
        top_mse_weight=mse_weight,
    )
    _, indices = teacher.topk(top_k, dim=-1, sorted=False)
    student_sel = student.gather(-1, indices)
    teacher_sel = teacher.gather(-1, indices)

    selective = compute_selected_kl_top_mse_token_loss(
        student_selected_logits=student_sel,
        teacher_selected_logits=teacher_sel,
        temperature=temperature,
        top_mse_weight=mse_weight,
    )
    torch.testing.assert_close(selective, dense, rtol=1e-5, atol=1e-6)

    no_mse = compute_kl_top_mse_token_loss(
        student_logits=student,
        teacher_logits=teacher,
        temperature=temperature,
        top_k=top_k,
        top_mse_weight=0.0,
    )
    expected_delta = mse_weight * (student_sel - teacher_sel).square().mean(dim=-1)
    torch.testing.assert_close(dense - no_mse, expected_delta, rtol=1e-5, atol=1e-6)


def test_cli_accepts_new_losses_and_restricts_selective_mass():
    partial_cfg = _e2e(["--loss_type", "kl_top_partial", "--top_k", "100"])
    assert partial_cfg.loss.loss_type == "kl_top_partial"

    mass_cfg = _e2e(["--loss_type", "kl_top_mass", "--top_k", "100"])
    assert mass_cfg.loss.loss_type == "kl_top_mass"

    kd_mass_cfg = _e2e(["--loss_type", "kd_top_mass", "--top_k", "100", "--alpha", "0.4"])
    assert kd_mass_cfg.loss.loss_type == "kd_top_mass"
    assert kd_mass_cfg.loss.alpha == pytest.approx(0.4)

    mse_cfg = _e2e(
        [
            "--loss_type",
            "kl_top_mse",
            "--top_k",
            "100",
            "--top_mse_weight",
            "0.25",
            "--selective_student_topk",
            "true",
        ]
    )
    assert mse_cfg.loss.loss_type == "kl_top_mse"
    assert mse_cfg.loss.top_mse_weight == pytest.approx(0.25)
    assert mse_cfg.loss.selective_student_topk is True

    with pytest.raises((SystemExit, ValueError)):
        _e2e(
            [
                "--loss_type",
                "kl_top_mass",
                "--selective_student_topk",
                "true",
            ]
        )

    with pytest.raises((SystemExit, ValueError)):
        _e2e(
            [
                "--loss_type",
                "kd_top_mass",
                "--selective_student_topk",
                "true",
            ]
        )

    with pytest.raises((SystemExit, ValueError)):
        _e2e(
            [
                "--loss_type",
                "kl_top_partial",
                "--selective_student_topk",
                "true",
            ]
        )

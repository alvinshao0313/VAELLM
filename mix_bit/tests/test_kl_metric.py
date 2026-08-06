from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mix_bit.kl_metric import (
    compute_metric_audit,
    paired_delta_kl,
    per_sample_exact_forward_kl,
    per_sample_teacher_topk_forward_kl,
    resolve_metric_contract,
    sample_mean_kl,
    validate_kl_mode_arguments,
)
from mix_bit.teacher_cache import (
    build_teacher_topk_cache,
    build_teacher_topk_chunk,
    validate_teacher_cache_against_inputs,
    write_teacher_cache_chunk,
)


def dense_teacher_topk_reference(logits: torch.Tensor, mask: torch.Tensor, k: int):
    """Test-only dense reference: full-row float32 top-k over valid positions."""
    valid = logits.float()[mask]
    values, indices = valid.topk(k, dim=-1, sorted=True)
    return indices.to(torch.int32), values.softmax(dim=-1)


def _manual_exact_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)
    out = []
    for b in range(token_kl.shape[0]):
        vals = token_kl[b][valid_mask[b]]
        out.append(vals.mean())
    return torch.stack(out)


def test_exact_forward_kl_is_zero_for_identical_logits():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)
    kl = per_sample_exact_forward_kl(logits, logits.clone(), mask)
    assert kl.shape == (2,)
    assert kl.dtype == torch.float32
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_exact_forward_kl_matches_manual_distribution():
    torch.manual_seed(1)
    teacher = torch.randn(3, 4, 6)
    student = torch.randn(3, 4, 6)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    got = per_sample_exact_forward_kl(teacher, student, mask)
    expected = _manual_exact_forward_kl(teacher, student, mask)
    assert torch.allclose(got, expected, atol=1e-6)


def test_exact_metric_requires_already_shifted_matching_shapes():
    teacher = torch.randn(2, 4, 5)
    student = torch.randn(2, 3, 5)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        per_sample_exact_forward_kl(teacher, student, mask)

    with pytest.raises(ValueError, match="shape"):
        per_sample_exact_forward_kl(teacher, teacher, torch.ones(2, 3, dtype=torch.bool))


def test_exact_metric_rejects_sample_with_zero_valid_tokens():
    teacher = torch.randn(2, 3, 4)
    student = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    with pytest.raises(ValueError, match="valid"):
        per_sample_exact_forward_kl(teacher, student, mask)


def test_sample_mean_differs_from_flat_token_mean_for_unequal_lengths():
    torch.manual_seed(2)
    teacher = torch.randn(2, 4, 5)
    student = torch.randn(2, 4, 5)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, False, False, False],
        ]
    )
    per_sample = per_sample_exact_forward_kl(teacher, student, mask)
    sample_mean = sample_mean_kl(per_sample)

    teacher_log_prob = F.log_softmax(teacher.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student.float(), dim=-1)
    token_kl = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)
    flat_mean = token_kl[mask].mean()

    assert not torch.allclose(sample_mean, flat_mean, atol=1e-5)
    assert torch.allclose(sample_mean, per_sample.mean(), atol=1e-6)


def test_paired_delta_uses_matching_sample_ids():
    kl_a = torch.tensor([1.0, 2.0, 3.0])
    kl_b = torch.tensor([30.0, 10.0, 20.0])
    ids_a = [10, 20, 30]
    ids_b = [30, 10, 20]
    delta = paired_delta_kl(
        sample_ids_a=ids_a,
        kl_a=kl_a,
        sample_ids_b=ids_b,
        kl_b=kl_b,
    )
    # aligned to ids_a order: b(10)=10, b(20)=20, b(30)=30
    expected = torch.tensor([1.0 - 10.0, 2.0 - 20.0, 3.0 - 30.0])
    assert torch.allclose(delta, expected)

    with pytest.raises(ValueError, match="sample_id"):
        paired_delta_kl(
            sample_ids_a=[1, 2],
            kl_a=torch.tensor([0.0, 0.0]),
            sample_ids_b=[1, 3],
            kl_b=torch.tensor([0.0, 0.0]),
        )


def test_exact_mode_rejects_topk_argument():
    with pytest.raises(ValueError, match="exact_full_vocab"):
        validate_kl_mode_arguments(
            kl_mode="exact_full_vocab",
            teacher_topk=256,
            teacher_cache=None,
        )
    with pytest.raises(ValueError, match="exact_full_vocab"):
        validate_kl_mode_arguments(
            kl_mode="exact_full_vocab",
            teacher_topk=None,
            teacher_cache="/tmp/cache",
        )
    contract = resolve_metric_contract(kl_mode="exact_full_vocab", teacher_topk=None)
    assert contract.kl_mode == "exact_full_vocab"
    assert contract.metric_name == "forward_kl_full_vocab_exact"
    assert contract.teacher_topk is None


def test_teacher_topk_kl_matches_manual_renormalized_subset():
    torch.manual_seed(3)
    b, t, v, k = 2, 3, 8, 3
    teacher = torch.randn(b, t, v)
    student = torch.randn(b, t, v)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="float32",
    )
    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )

    expected = []
    for bi in range(b):
        token_vals = []
        for ti in range(t):
            if not bool(mask[bi, ti]):
                continue
            idx = teacher[bi, ti].topk(k, dim=-1, sorted=True).indices
            t_prob = F.softmax(teacher[bi, ti].gather(-1, idx).float(), dim=-1)
            s_log = F.log_softmax(student[bi, ti].gather(-1, idx).float(), dim=-1)
            token_vals.append((t_prob * (t_prob.log() - s_log)).sum())
        expected.append(torch.stack(token_vals).mean())
    expected_t = torch.stack(expected)
    assert torch.allclose(got, expected_t, atol=1e-5)


def test_teacher_topk_uses_teacher_indices_for_both_models():
    # Student top-1 differs from teacher top-1; metric must still use teacher indices.
    teacher = torch.tensor([[[0.0, 5.0, 1.0, -2.0]]])  # top1 = idx 1
    student = torch.tensor([[[6.0, 0.0, 1.0, -1.0]]])  # top1 = idx 0
    mask = torch.ones(1, 1, dtype=torch.bool)
    chunk = build_teacher_topk_chunk(
        sample_ids=[0],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=2,
        cache_prob_dtype="float32",
    )
    assert chunk["teacher_topk_indices"][0].tolist() == [1, 2]

    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    idx = torch.tensor([1, 2])
    t_prob = F.softmax(teacher[0, 0].gather(-1, idx).float(), dim=-1)
    s_log = F.log_softmax(student[0, 0].gather(-1, idx).float(), dim=-1)
    expected = (t_prob * (t_prob.log() - s_log)).sum()
    assert torch.allclose(got, expected.view(1), atol=1e-6)


def test_teacher_topk_equals_exact_when_k_equals_vocab_size():
    torch.manual_seed(4)
    teacher = torch.randn(2, 3, 5)
    student = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    exact = per_sample_exact_forward_kl(teacher, student, mask)
    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=5,
        cache_prob_dtype="float32",
    )
    topk = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    assert torch.allclose(exact, topk, atol=1e-5)


def test_bf16_cached_probs_renormalized_to_float32_before_kl():
    """Cache may store bf16 probs that do not sum to 1; eval must float32-renorm over K."""
    torch.manual_seed(5)
    teacher = torch.randn(2, 3, 6)
    student = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    chunk_f32 = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=3,
        cache_prob_dtype="float32",
    )
    # Simulate production load: cast stored probs to bf16 (row sums leave 1).
    bf16_probs = chunk_f32["teacher_topk_probs"].to(torch.bfloat16)
    assert not torch.allclose(
        bf16_probs.float().sum(dim=-1),
        torch.ones(bf16_probs.shape[0]),
        atol=1e-6,
    )

    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk_f32["teacher_topk_indices"],
        teacher_topk_probs=bf16_probs,
        token_offsets=chunk_f32["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )

    # Reference: float32 cast then renormalize over K, then same KL.
    probs = bf16_probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    indices = chunk_f32["teacher_topk_indices"].to(dtype=torch.long)
    valid_student = student.float()[mask]
    selected = valid_student.gather(-1, indices)
    student_log = F.log_softmax(selected, dim=-1)
    teacher_log = probs.log()
    token_kl = (probs * (teacher_log - student_log)).sum(dim=-1)
    offsets = chunk_f32["token_offsets"]
    expected = torch.stack(
        [
            token_kl[int(offsets[i]) : int(offsets[i + 1])].mean()
            for i in range(2)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-6)

    # Without eval renorm, bf16 probs yield a different (non-probability) KL.
    bad_probs = bf16_probs.float()
    bad_log = bad_probs.clamp_min(torch.finfo(torch.float32).tiny).log()
    bad_token = (bad_probs * (bad_log - student_log)).sum(dim=-1)
    bad = torch.stack(
        [
            bad_token[int(offsets[i]) : int(offsets[i + 1])].mean()
            for i in range(2)
        ]
    )
    assert not torch.allclose(got, bad, atol=1e-6)

    # K == V: float32 cache matches exact tightly; bf16 cache after float32 renorm
    # stays near exact (residual is bf16 quantization, not missing renorm).
    chunk_full_f32 = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=6,
        cache_prob_dtype="float32",
    )
    topk_full_f32 = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk_full_f32["teacher_topk_indices"],
        teacher_topk_probs=chunk_full_f32["teacher_topk_probs"],
        token_offsets=chunk_full_f32["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    exact = per_sample_exact_forward_kl(teacher, student, mask)
    assert torch.allclose(topk_full_f32, exact, atol=1e-5)

    chunk_full_bf16 = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=6,
        cache_prob_dtype="bfloat16",
    )
    topk_full_bf16 = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk_full_bf16["teacher_topk_indices"],
        teacher_topk_probs=chunk_full_bf16["teacher_topk_probs"],
        token_offsets=chunk_full_bf16["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    bf16_raw = chunk_full_bf16["teacher_topk_probs"].float()
    idx_full = chunk_full_bf16["teacher_topk_indices"].to(dtype=torch.long)
    sel_full = student.float()[mask].gather(-1, idx_full)
    slog_full = F.log_softmax(sel_full, dim=-1)
    bad_full_tok = (
        bf16_raw
        * (bf16_raw.clamp_min(torch.finfo(torch.float32).tiny).log() - slog_full)
    ).sum(dim=-1)
    off_full = chunk_full_bf16["token_offsets"]
    bad_full = torch.stack(
        [
            bad_full_tok[int(off_full[i]) : int(off_full[i + 1])].mean()
            for i in range(2)
        ]
    )
    renorm_err = (topk_full_bf16 - exact).abs().max()
    unreorm_err = (bad_full - exact).abs().max()
    assert float(renorm_err) < float(unreorm_err)
    assert torch.allclose(topk_full_bf16, exact, atol=1e-3)


def test_teacher_topk_rejects_k_above_vocab_size():
    teacher = torch.randn(1, 2, 4)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="teacher_topk"):
        build_teacher_topk_chunk(
            sample_ids=[0],
            shifted_teacher_logits=teacher,
            valid_mask=mask,
            teacher_topk=5,
            cache_prob_dtype="float32",
        )


def test_teacher_topk_rejects_nonpositive_k():
    teacher = torch.randn(1, 2, 4)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="teacher_topk"):
        build_teacher_topk_chunk(
            sample_ids=[0],
            shifted_teacher_logits=teacher,
            valid_mask=mask,
            teacher_topk=0,
            cache_prob_dtype="float32",
        )
    with pytest.raises(ValueError, match="teacher_topk"):
        validate_kl_mode_arguments(
            kl_mode="teacher_topk",
            teacher_topk=-1,
            teacher_cache="/tmp/x",
            vocab_size=100,
        )


def test_teacher_topk_cache_has_no_tail_field():
    teacher = torch.randn(1, 2, 6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    chunk = build_teacher_topk_chunk(
        sample_ids=[7],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=2,
        cache_prob_dtype="bfloat16",
    )
    forbidden = {
        "tail",
        "tail_prob",
        "tail_probs",
        "residual",
        "residual_mass",
        "tail_bucket",
        "omitted_mass",
    }
    assert forbidden.isdisjoint(chunk.keys())
    assert chunk["metric_name"] == "forward_kl_teacher_topk_renorm"
    assert chunk["teacher_topk"] == 2
    assert chunk["vocab_size"] == 6
    assert chunk["teacher_topk_probs"].dtype == torch.bfloat16


def test_cache_flattens_only_valid_causal_positions():
    teacher = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=2,
        cache_prob_dtype="float32",
    )
    assert chunk["teacher_topk_indices"].shape == (4, 2)
    assert chunk["teacher_topk_probs"].shape == (4, 2)
    assert int(chunk["token_offsets"][-1]) == 4
    # No padded [B,T,K]
    assert chunk["teacher_topk_indices"].ndim == 2


def test_cache_token_offsets_reconstruct_per_sample_ranges():
    teacher = torch.randn(3, 4, 5)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )
    chunk = build_teacher_topk_chunk(
        sample_ids=[10, 20, 30],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=2,
        cache_prob_dtype="float32",
    )
    offsets = chunk["token_offsets"].tolist()
    assert offsets[0] == 0
    assert offsets == [0, 2, 3, 6]
    counts = [offsets[i + 1] - offsets[i] for i in range(3)]
    assert counts == [2, 1, 3]
    assert chunk["sample_ids"].tolist() == [10, 20, 30]


def test_cache_rejects_wrong_sample_order(tmp_path: Path):
    teacher = torch.randn(2, 2, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)
    chunk = build_teacher_topk_chunk(
        sample_ids=[1, 2],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=2,
        cache_prob_dtype="float32",
    )
    path = tmp_path / "chunk_0000.pt"
    write_teacher_cache_chunk(path, chunk)

    index = {
        "kind": "mix_bit_teacher_topk_cache_index",
        "kl_mode": "teacher_topk",
        "metric_name": "forward_kl_teacher_topk_renorm",
        "teacher_topk": 2,
        "vocab_size": 4,
        "cache_prob_dtype": "float32",
        "run_config_sha256": "run",
        "model_inventory_fingerprint": "inv",
        "dataset_file_sha256": "ds",
        # Cache stored samples as [1, 2] but dataset manifest order is [2, 1].
        "sample_ids": [1, 2],
        "chunks": [
            {
                "path": str(path.name),
                "sample_start": 0,
                "sample_end": 2,
                "sample_ids": [1, 2],
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    with pytest.raises(ValueError, match="sample"):
        validate_teacher_cache_against_inputs(
            index,
            expected_sample_ids=[2, 1],
            run_config_sha256="run",
            model_inventory_fingerprint="inv",
            dataset_file_sha256="ds",
            teacher_topk=2,
            vocab_size=4,
            cache_prob_dtype="float32",
        )


def test_cache_rejects_dataset_or_inventory_hash_mismatch():
    index = {
        "kind": "mix_bit_teacher_topk_cache_index",
        "kl_mode": "teacher_topk",
        "metric_name": "forward_kl_teacher_topk_renorm",
        "teacher_topk": 2,
        "vocab_size": 4,
        "cache_prob_dtype": "float32",
        "run_config_sha256": "run",
        "model_inventory_fingerprint": "inv",
        "dataset_file_sha256": "ds",
        "sample_ids": [0],
        "chunks": [],
    }
    with pytest.raises(ValueError, match="dataset_file_sha256"):
        validate_teacher_cache_against_inputs(
            index,
            expected_sample_ids=[0],
            run_config_sha256="run",
            model_inventory_fingerprint="inv",
            dataset_file_sha256="other",
            teacher_topk=2,
            vocab_size=4,
            cache_prob_dtype="float32",
        )
    with pytest.raises(ValueError, match="model_inventory_fingerprint"):
        validate_teacher_cache_against_inputs(
            index,
            expected_sample_ids=[0],
            run_config_sha256="run",
            model_inventory_fingerprint="other",
            dataset_file_sha256="ds",
            teacher_topk=2,
            vocab_size=4,
            cache_prob_dtype="float32",
        )


def test_metric_audit_reports_diff_not_selection():
    audit = compute_metric_audit(
        sample_ids=[0, 1, 2],
        exact_kl=torch.tensor([1.0, 2.0, 3.0]),
        topk_kl=torch.tensor([1.1, 1.9, 3.2]),
        teacher_topk=4,
    )
    assert audit["teacher_topk"] == 4
    assert "mean_abs_diff" in audit
    assert "spearman_rank_correlation" in audit
    assert "selected_metric" not in audit
    assert audit["production_metric_unchanged"] is True


def test_teacher_topk_chunk_matches_dense_reference_float32():
    """Device-side top-k must match dense float32 reference (no ties)."""
    torch.manual_seed(100)
    b, t, v, k = 3, 5, 10, 4
    logits = torch.randn(b, t, v, dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1, 2],
        shifted_teacher_logits=logits,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="float32",
    )
    ref_indices, ref_probs = dense_teacher_topk_reference(logits, mask, k)
    torch.testing.assert_close(chunk["teacher_topk_indices"], ref_indices)
    torch.testing.assert_close(chunk["teacher_topk_probs"], ref_probs)
    assert chunk["teacher_topk_indices"].device.type == "cpu"
    assert chunk["teacher_topk_probs"].device.type == "cpu"
    assert chunk["token_offsets"].device.type == "cpu"


def test_teacher_topk_chunk_matches_dense_reference_bfloat16():
    """bfloat16 cache dtype allows quantization error but indices must match."""
    torch.manual_seed(101)
    b, t, v, k = 2, 4, 8, 3
    logits = torch.randn(b, t, v, dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    chunk = build_teacher_topk_chunk(
        sample_ids=[10, 20],
        shifted_teacher_logits=logits,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="bfloat16",
    )
    ref_indices, ref_probs = dense_teacher_topk_reference(logits, mask, k)
    torch.testing.assert_close(chunk["teacher_topk_indices"], ref_indices)
    assert chunk["teacher_topk_probs"].dtype == torch.bfloat16
    torch.testing.assert_close(
        chunk["teacher_topk_probs"].float(),
        ref_probs,
        rtol=1e-2,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_teacher_topk_only_returns_compact_cpu_tensors():
    """CUDA transfer-boundary: only compact [N_valid,K] tensors move to CPU."""
    torch.manual_seed(102)
    b, t, v, k = 2, 4, 12, 5
    logits = torch.randn(b, t, v, dtype=torch.float32, device="cuda")
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ],
        device="cuda",
    )
    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=logits,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="float32",
    )
    assert chunk["teacher_topk_indices"].device.type == "cpu"
    assert chunk["teacher_topk_probs"].device.type == "cpu"
    assert chunk["token_offsets"].device.type == "cpu"
    assert chunk["sample_ids"].device.type == "cpu"
    n_valid = int(mask.sum().item())
    assert chunk["teacher_topk_indices"].shape == (n_valid, k)
    assert chunk["teacher_topk_probs"].shape == (n_valid, k)
    ref_indices, ref_probs = dense_teacher_topk_reference(logits.cpu(), mask.cpu(), k)
    torch.testing.assert_close(chunk["teacher_topk_indices"], ref_indices)
    torch.testing.assert_close(chunk["teacher_topk_probs"], ref_probs)


def test_teacher_cache_source_does_not_transfer_full_logits_to_cpu():
    """Guard against full-logits CPU transfer regression in build_teacher_topk_cache."""
    source = inspect.getsource(build_teacher_topk_cache)
    assert "shifted.detach().cpu()" not in source
    assert "shifted_teacher_logits=shifted.detach().cpu()" not in source


# ---------------------------------------------------------------------------
# Task 5: on-device student top-k gather
# ---------------------------------------------------------------------------


def _dense_teacher_topk_kl_reference(
    *,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    token_offsets: torch.Tensor,
    shifted_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Dense reference: gather student logits via [N_valid, V] then select K.

    This is the forbidden production shape ([N_valid, V]); tests use it only as
    a small-tensor oracle to validate the on-device K-way gather path.
    """
    mask = valid_mask.bool()
    indices = teacher_topk_indices.to(dtype=torch.long)
    probs = teacher_topk_probs.float()
    offsets = token_offsets.to(dtype=torch.long)
    n_valid, _ = indices.shape
    row_mass = probs.sum(dim=-1, keepdim=True)
    probs = probs / row_mass
    valid_student = shifted_student_logits.float()[mask]
    assert valid_student.shape[0] == n_valid
    selected_student = valid_student.gather(-1, indices)
    student_log_prob = F.log_softmax(selected_student, dim=-1)
    teacher_log_prob = probs.log()
    token_kl = (probs * (teacher_log_prob - student_log_prob)).sum(dim=-1)
    batch = int(shifted_student_logits.shape[0])
    out = torch.empty(batch, dtype=torch.float32, device=token_kl.device)
    for i in range(batch):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        out[i] = token_kl[start:end].mean().to(dtype=torch.float32)
    return out


def _build_topk_inputs(teacher, student, mask, k, *, prob_dtype=torch.float32):
    chunk = build_teacher_topk_chunk(
        sample_ids=list(range(int(mask.shape[0]))),
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="float32",
    )
    chunk["teacher_topk_probs"] = chunk["teacher_topk_probs"].to(prob_dtype)
    return chunk


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("k", [1, 3, None])
def test_teacher_topk_kl_matches_dense_reference_various_shapes(batch, k):
    torch.manual_seed(200 + batch * 7 + (k or 0))
    t = 5
    v = 6
    teacher = torch.randn(batch, t, v)
    student = torch.randn(batch, t, v)
    # Non-contiguous valid mask; varying per-sample token counts.
    mask = torch.tensor(
        [[True, False, True, True, False]]
        + [[True, True, False, True, False]] * (batch - 1),
        dtype=torch.bool,
    ) if batch > 1 else torch.tensor([[True, False, True, True, True]], dtype=torch.bool)
    # Ensure each sample has >=1 valid token.
    assert bool(mask.sum(dim=-1).ge(1).all())
    kk = v if k is None else k
    chunk = _build_topk_inputs(teacher, student, mask, kk)
    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    expected = _dense_teacher_topk_kl_reference(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    assert got.shape == (batch,)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_teacher_topk_kl_matches_dense_reference_bf16_probs():
    torch.manual_seed(211)
    batch, t, v, k = 3, 4, 7, 3
    teacher = torch.randn(batch, t, v)
    student = torch.randn(batch, t, v)
    mask = torch.tensor(
        [
            [True, True, False, True],
            [True, False, False, True],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )
    chunk = _build_topk_inputs(teacher, student, mask, k, prob_dtype=torch.bfloat16)
    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    expected = _dense_teacher_topk_kl_reference(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_teacher_topk_kl_matches_dense_reference_mixed_sign_logits():
    torch.manual_seed(222)
    batch, t, v, k = 2, 3, 5, 2
    # Mix of negative and positive logits, including large magnitudes.
    teacher = torch.tensor(
        [
            [[3.0, -1.0, 0.5, -4.0, 2.0]] * t,
            [[-2.0, 5.0, -0.5, 1.0, -3.0]] * t,
        ]
    )
    student = torch.tensor(
        [
            [[-2.0, 1.0, -0.5, 4.0, -1.0]] * t,
            [[1.0, -5.0, 0.5, -1.0, 3.0]] * t,
        ]
    )
    mask = torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool)
    chunk = _build_topk_inputs(teacher, student, mask, k)
    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    expected = _dense_teacher_topk_kl_reference(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_teacher_topk_kl_matches_dense_reference_k_equals_vocab():
    torch.manual_seed(233)
    batch, t, v = 2, 3, 5
    teacher = torch.randn(batch, t, v)
    student = torch.randn(batch, t, v)
    mask = torch.tensor([[True, True, False], [True, False, True]], dtype=torch.bool)
    chunk = _build_topk_inputs(teacher, student, mask, v)
    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    expected = _dense_teacher_topk_kl_reference(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_gather_topk_student_logits_shape_is_n_valid_by_k():
    from mix_bit.kl_metric import _gather_topk_student_logits

    torch.manual_seed(240)
    b, t, v, k = 2, 5, 1000, 4
    student = torch.randn(b, t, v)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    n_valid = int(mask.sum().item())
    indices = torch.randint(0, v, (n_valid, k), dtype=torch.long)
    selected = _gather_topk_student_logits(student, mask, indices)
    assert selected.shape == (n_valid, k)
    assert selected.dtype == torch.float32
    # Cross-check values against a dense gather at the valid positions.
    valid_student = student.float()[mask]
    expected = valid_student.gather(-1, indices)
    torch.testing.assert_close(selected, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gather_topk_student_logits_stays_on_cuda_and_matches_cpu_reference():
    from mix_bit.kl_metric import _gather_topk_student_logits

    torch.manual_seed(250)
    b, t, v, k = 2, 4, 9, 3
    teacher = torch.randn(b, t, v, device="cuda")
    student = torch.randn(b, t, v, device="cuda")
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]],
        device="cuda",
        dtype=torch.bool,
    )
    chunk = build_teacher_topk_chunk(
        sample_ids=[0, 1],
        shifted_teacher_logits=teacher,
        valid_mask=mask,
        teacher_topk=k,
        cache_prob_dtype="float32",
    )
    indices = chunk["teacher_topk_indices"].to(device="cuda", dtype=torch.long)
    selected = _gather_topk_student_logits(student, mask, indices)
    assert selected.device.type == "cuda"
    assert selected.shape[0] == int(mask.sum().item())
    assert selected.shape[1] == k

    got = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student,
        valid_mask=mask,
    )
    assert got.device.type == "cuda"
    expected = _dense_teacher_topk_kl_reference(
        teacher_topk_indices=chunk["teacher_topk_indices"],
        teacher_topk_probs=chunk["teacher_topk_probs"],
        token_offsets=chunk["token_offsets"],
        shifted_student_logits=student.cpu(),
        valid_mask=mask.cpu(),
    )
    torch.testing.assert_close(got.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_kl_source_does_not_gather_full_valid_student_rows():
    """Guard against [N_valid, V] gather regression in the production KL path."""
    source = inspect.getsource(per_sample_teacher_topk_forward_kl)
    assert "shifted_student_logits.float()[mask]" not in source
    assert "shifted_student_logits.float()[valid_mask]" not in source
    assert "shifted_student_logits[mask]" not in source
    assert "shifted_student_logits.cpu()" not in source

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

KL_MODE_TEACHER_TOPK = "teacher_topk"
KL_MODE_EXACT_FULL_VOCAB = "exact_full_vocab"

METRIC_NAME_TEACHER_TOPK = "forward_kl_teacher_topk_renorm"
METRIC_NAME_EXACT_FULL_VOCAB = "forward_kl_full_vocab_exact"

ALLOWED_KL_MODES = frozenset({KL_MODE_TEACHER_TOPK, KL_MODE_EXACT_FULL_VOCAB})


@dataclass(frozen=True)
class MetricContract:
    kl_mode: str
    metric_name: str
    teacher_topk: int | None


def resolve_metric_contract(
    *,
    kl_mode: str,
    teacher_topk: int | None = None,
) -> MetricContract:
    mode = str(kl_mode)
    if mode not in ALLOWED_KL_MODES:
        raise ValueError(
            f"Unsupported kl_mode={mode!r}. Allowed: {sorted(ALLOWED_KL_MODES)}"
        )
    if mode == KL_MODE_EXACT_FULL_VOCAB:
        if teacher_topk is not None:
            raise ValueError(
                "exact_full_vocab rejects teacher_topk; pass teacher_topk=None"
            )
        return MetricContract(
            kl_mode=KL_MODE_EXACT_FULL_VOCAB,
            metric_name=METRIC_NAME_EXACT_FULL_VOCAB,
            teacher_topk=None,
        )
    if teacher_topk is None:
        raise ValueError("teacher_topk mode requires an explicit positive teacher_topk K")
    k = int(teacher_topk)
    if k < 1:
        raise ValueError(f"teacher_topk must be >= 1, got {k}")
    return MetricContract(
        kl_mode=KL_MODE_TEACHER_TOPK,
        metric_name=METRIC_NAME_TEACHER_TOPK,
        teacher_topk=k,
    )


def validate_kl_mode_arguments(
    *,
    kl_mode: str,
    teacher_topk: int | None,
    teacher_cache: str | Path | None,
    vocab_size: int | None = None,
) -> MetricContract:
    mode = str(kl_mode)
    if mode not in ALLOWED_KL_MODES:
        raise ValueError(
            f"Unsupported kl_mode={mode!r}. Allowed: {sorted(ALLOWED_KL_MODES)}"
        )

    if mode == KL_MODE_EXACT_FULL_VOCAB:
        if teacher_topk is not None:
            raise ValueError(
                "exact_full_vocab rejects --teacher_topk; omit K for full-vocabulary KL"
            )
        if teacher_cache is not None:
            raise ValueError(
                "exact_full_vocab rejects --teacher_cache; full-vocab KL is computed online"
            )
        return resolve_metric_contract(kl_mode=mode, teacher_topk=None)

    if teacher_topk is None:
        raise ValueError("teacher_topk mode requires --teacher_topk K")
    if teacher_cache is None:
        raise ValueError("teacher_topk mode requires --teacher_cache <path>")
    k = int(teacher_topk)
    if k < 1:
        raise ValueError(f"teacher_topk must be >= 1, got {k}")
    if vocab_size is not None:
        v = int(vocab_size)
        if k > v:
            raise ValueError(f"teacher_topk={k} exceeds vocab_size={v}")
    return resolve_metric_contract(kl_mode=mode, teacher_topk=k)


def _validate_exact_inputs(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits shape mismatch: "
            f"{tuple(teacher_logits.shape)} vs {tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 3:
        raise ValueError(
            f"Expected shifted logits [B, T-1, V], got shape {tuple(teacher_logits.shape)}"
        )
    if not teacher_logits.is_floating_point() or not student_logits.is_floating_point():
        raise ValueError("teacher_logits and student_logits must be floating-point")
    mask = valid_mask.bool()
    if mask.shape != teacher_logits.shape[:2]:
        raise ValueError(
            "valid_mask shape mismatch: "
            f"{tuple(mask.shape)} vs expected {tuple(teacher_logits.shape[:2])}"
        )
    counts = mask.sum(dim=-1)
    if bool((counts < 1).any()):
        bad = (counts < 1).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Each sample needs at least one valid token; empty samples at indices {bad}"
        )
    return mask


def per_sample_exact_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one exact full-vocabulary forward-KL value per sample."""
    mask = _validate_exact_inputs(teacher_logits, student_logits, valid_mask)
    teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    summed = (token_kl * mask_fp).sum(dim=-1)
    counts = mask_fp.sum(dim=-1).clamp_min(1.0)
    return (summed / counts).to(dtype=torch.float32)


def _gather_topk_student_logits(
    shifted_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Return [N_valid, K] student logits gathered on-device, never [N_valid, V].

    Pads teacher indices to [B, T, K] (zeros at invalid positions), gathers along
    the vocabulary axis of the student logits, then selects the valid rows. The
    intermediate ``selected_all`` tensor is [B, T, K]; only the [N_valid, K] slice
    is materialized via boolean indexing on the first two dims.
    """
    device = shifted_student_logits.device
    mask = valid_mask.to(device=device, dtype=torch.bool)
    indices_flat = teacher_topk_indices.to(device=device, dtype=torch.long)
    B, T, _ = shifted_student_logits.shape
    K = indices_flat.shape[1]
    padded_indices = torch.zeros((B, T, K), dtype=torch.long, device=device)
    padded_indices[mask] = indices_flat
    selected_all = shifted_student_logits.gather(-1, padded_indices)
    return selected_all[mask].float()


def per_sample_teacher_topk_forward_kl(
    *,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    token_offsets: torch.Tensor,
    shifted_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one teacher top-k renormalized forward-KL value per sample.

    Cache storage may keep ``teacher_topk_probs`` in bfloat16 (row sums can
    drift off 1). Evaluation always casts to float32 and renormalizes over K
    before teacher log-prob / KL, so the metric is a strict probability KL.

    The student top-k logits are gathered on the student device via
    ``_gather_topk_student_logits`` (returns [N_valid, K]); the production path
    never materializes the [N_valid, V] dense student rows.
    """
    if shifted_student_logits.ndim != 3:
        raise ValueError(
            "Expected shifted_student_logits [B, T-1, V], got "
            f"{tuple(shifted_student_logits.shape)}"
        )
    if not shifted_student_logits.is_floating_point():
        raise ValueError("shifted_student_logits must be floating-point")
    mask = valid_mask.bool()
    if mask.shape != shifted_student_logits.shape[:2]:
        raise ValueError(
            "valid_mask shape mismatch: "
            f"{tuple(mask.shape)} vs expected {tuple(shifted_student_logits.shape[:2])}"
        )

    indices = teacher_topk_indices.to(dtype=torch.long)
    # Storage dtype (e.g. bf16) is not a probability simplex; restore one in float32.
    probs = teacher_topk_probs.float()
    offsets = token_offsets.to(dtype=torch.long)
    if indices.ndim != 2 or probs.ndim != 2:
        raise ValueError("teacher_topk_indices/probs must be [N_valid, K]")
    if indices.shape != probs.shape:
        raise ValueError(
            "teacher_topk_indices/probs shape mismatch: "
            f"{tuple(indices.shape)} vs {tuple(probs.shape)}"
        )
    n_valid, k = indices.shape
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("token_offsets must be 1-D with length B+1")
    batch = int(shifted_student_logits.shape[0])
    if int(offsets.numel()) != batch + 1:
        raise ValueError(
            f"token_offsets length {int(offsets.numel())} != batch+1={batch + 1}"
        )
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != n_valid:
        raise ValueError(
            "token_offsets must start at 0 and end at N_valid; "
            f"got start={int(offsets[0].item())} end={int(offsets[-1].item())} N_valid={n_valid}"
        )
    mask_count = int(mask.sum().item())
    if mask_count != n_valid:
        raise ValueError(
            f"valid_mask true count {mask_count} != N_valid={n_valid} from cache tensors"
        )
    vocab = int(shifted_student_logits.shape[-1])
    if bool((indices < 0).any()) or bool((indices >= vocab).any()):
        raise ValueError(f"teacher_topk_indices out of vocabulary bounds [0, {vocab})")
    for i in range(batch):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        if end < start:
            raise ValueError(f"Invalid token_offsets range for sample {i}: [{start}, {end})")
        expected = int(mask[i].sum().item())
        if end - start != expected:
            raise ValueError(
                f"token_offsets sample {i} spans {end - start} tokens but mask has {expected}"
            )
        if expected < 1:
            raise ValueError(
                f"Each sample needs at least one valid token; empty sample at index {i}"
            )

    row_mass = probs.sum(dim=-1, keepdim=True)
    if bool((row_mass <= 0).any()):
        raise ValueError("teacher_topk_probs rows must have positive mass before renormalization")
    probs = probs / row_mass

    selected_valid = _gather_topk_student_logits(shifted_student_logits, mask, indices)
    if selected_valid.shape[0] != n_valid:
        raise ValueError("Flattened student rows do not match N_valid")
    student_log_prob = F.log_softmax(selected_valid, dim=-1)
    teacher_log_prob = probs.to(device=selected_valid.device).log()
    token_kl = (probs.to(device=selected_valid.device) * (teacher_log_prob - student_log_prob)).sum(dim=-1)

    out = torch.empty(batch, dtype=torch.float32, device=token_kl.device)
    offsets_dev = offsets.to(device=token_kl.device)
    for i in range(batch):
        start = int(offsets_dev[i].item())
        end = int(offsets_dev[i + 1].item())
        out[i] = token_kl[start:end].mean().to(dtype=torch.float32)
    return out


def sample_mean_kl(per_sample_kl: torch.Tensor) -> torch.Tensor:
    if per_sample_kl.ndim != 1:
        raise ValueError(f"per_sample_kl must be 1-D [B], got {tuple(per_sample_kl.shape)}")
    if per_sample_kl.numel() < 1:
        raise ValueError("per_sample_kl is empty")
    return per_sample_kl.float().mean()


def paired_delta_kl(
    *,
    sample_ids_a: Sequence[int],
    kl_a: torch.Tensor,
    sample_ids_b: Sequence[int],
    kl_b: torch.Tensor,
) -> torch.Tensor:
    """Return kl_a - kl_b aligned to sample_ids_a order via matching sample_ids."""
    if kl_a.ndim != 1 or kl_b.ndim != 1:
        raise ValueError("kl_a and kl_b must be 1-D per-sample tensors")
    ids_a = [int(x) for x in sample_ids_a]
    ids_b = [int(x) for x in sample_ids_b]
    if len(ids_a) != int(kl_a.numel()) or len(ids_b) != int(kl_b.numel()):
        raise ValueError("sample_ids length must match per-sample KL length")
    if len(ids_a) != len(ids_b):
        raise ValueError(
            f"paired delta sample_id count mismatch: {len(ids_a)} vs {len(ids_b)}"
        )
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("sample_ids must be unique within each side")
    if set(ids_a) != set(ids_b):
        missing = sorted(set(ids_a) ^ set(ids_b))
        raise ValueError(f"paired delta sample_id sets differ; mismatched ids={missing}")
    index_b = {sid: i for i, sid in enumerate(ids_b)}
    order = [index_b[sid] for sid in ids_a]
    aligned_b = kl_b[order]
    return (kl_a.float() - aligned_b.float()).to(dtype=torch.float32)


def compute_metric_audit(
    *,
    sample_ids: Sequence[int],
    exact_kl: torch.Tensor,
    topk_kl: torch.Tensor,
    teacher_topk: int,
) -> dict:
    """Compare one top-k K against exact full-vocab KL without selecting a production metric."""
    ids = [int(x) for x in sample_ids]
    if exact_kl.ndim != 1 or topk_kl.ndim != 1:
        raise ValueError("exact_kl and topk_kl must be 1-D")
    if len(ids) != int(exact_kl.numel()) or len(ids) != int(topk_kl.numel()):
        raise ValueError("sample_ids length must match KL tensors")
    if int(teacher_topk) < 1:
        raise ValueError(f"teacher_topk must be >= 1, got {teacher_topk}")

    exact = exact_kl.float().detach().cpu()
    topk = topk_kl.float().detach().cpu()
    diff = topk - exact
    abs_diff = diff.abs()

    n = int(exact.numel())
    if n >= 2:
        exact_rank = exact.argsort().argsort().float()
        topk_rank = topk.argsort().argsort().float()
        exact_c = exact_rank - exact_rank.mean()
        topk_c = topk_rank - topk_rank.mean()
        denom = exact_c.norm() * topk_c.norm()
        if float(denom.item()) == 0.0:
            spearman = float("nan")
        else:
            spearman = float((exact_c * topk_c).sum().item() / float(denom.item()))
    else:
        spearman = float("nan")

    return {
        "kind": "mix_bit_metric_audit",
        "teacher_topk": int(teacher_topk),
        "sample_count": n,
        "sample_ids": ids,
        "mean_exact_kl": float(exact.mean().item()),
        "mean_topk_kl": float(topk.mean().item()),
        "mean_diff": float(diff.mean().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "max_abs_diff": float(abs_diff.max().item()) if n else 0.0,
        "spearman_rank_correlation": spearman,
        "production_metric_unchanged": True,
        "note": (
            "Audit only; does not select or switch the production KL metric. "
            "Caller must keep the explicitly configured kl_mode."
        ),
    }


def write_metric_audit(path: str | Path, audit: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    return out


def metric_contract_to_dict(contract: MetricContract) -> dict:
    return asdict(contract)

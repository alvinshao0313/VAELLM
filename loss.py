"""EdgeRazor 式 QAD 蒸馏损失：EAKLD + LAFD + CE（支持等价分块）。"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "build_token_mask",
    "compute_eakld",
    "compute_eakld_topk",
    "compute_kl_topk",
    "compute_rkl_topk",
    "compute_lafd_mse",
    "compute_lafd_mse_selected",
    "select_adaptive_layer_indices",
    "compute_edgerazor_qad_loss",
    "compute_edgerazor_qad_loss_chunked",
]


def build_token_mask(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if reference.ndim < 2:
        raise ValueError(f"reference must have shape [B, L, ...], got ndim={reference.ndim}")
    expected = tuple(int(x) for x in reference.shape[:2])

    if isinstance(labels, torch.Tensor):
        mask = labels.ne(-100)
    elif isinstance(attention_mask, torch.Tensor):
        mask = attention_mask.ne(0)
    else:
        return torch.ones(expected, dtype=torch.float32, device=reference.device)

    if tuple(int(x) for x in mask.shape) != expected:
        raise ValueError(f"mask shape mismatch: expected {expected}, got {tuple(mask.shape)}")
    return mask.to(device=reference.device, dtype=torch.float32)


def _resolve_temperature(temperature: float) -> float:
    return max(float(temperature), 0.1)


def _masked_token_kl_sum(
    *,
    student_log_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward KL: KL(teacher || student)，student 为 log_prob（可反传）。"""
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    return (token_kl * mask_fp).sum(), mask_fp.sum()


def _masked_token_reverse_kl_sum(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse KL: KL(student || teacher) = Σ p_s (log p_s - log p_t)。

    不能用 kl_div(log_t, softmax(s))：PyTorch 不对 target 反传，多卡下会得到 NaN 梯度。
    """
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    return (token_kl * mask_fp).sum(), mask_fp.sum()


def accumulate_teacher_entropy_stats(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = F.softmax(teacher_logits.detach().float(), dim=-1)
    entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-8))).sum(dim=-1)
    mask_fp = mask.to(device=entropy.device, dtype=torch.float32)
    return (entropy * mask_fp).sum(), mask_fp.sum()


def gamma_from_entropy_sums(
    sum_entropy: torch.Tensor,
    sum_valid: torch.Tensor,
    *,
    confidence_k: int = 16,
) -> torch.Tensor:
    max_entropy = math.log(float(int(confidence_k)))
    avg = sum_entropy / sum_valid.clamp_min(1.0)
    return (1.0 - avg / float(max_entropy)).clamp(0.0, 1.0)


def compute_eakld(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    confidence_k: int = 16,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], device=student_logits.device, dtype=torch.float32)
    temp = _resolve_temperature(temperature)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    sum_e, sum_v = accumulate_teacher_entropy_stats(teacher_logits, mask)
    gamma = gamma_from_entropy_sums(sum_e, sum_v, confidence_k=int(confidence_k))
    rev_sum, n = _masked_token_reverse_kl_sum(
        student_logits=student_scaled,
        teacher_logits=teacher_scaled,
        mask=mask,
    )
    fwd_sum, _ = _masked_token_kl_sum(
        student_log_prob=F.log_softmax(student_scaled, dim=-1),
        teacher_prob=F.softmax(teacher_scaled, dim=-1),
        mask=mask,
    )
    reverse_kl = (rev_sum / n.clamp_min(1.0)) * (temp * temp)
    forward_kl = (fwd_sum / n.clamp_min(1.0)) * (temp * temp)
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl


def _topk_forward_kl_sum(
    *,
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    post_attn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward KL（教师‖学生）top-k 版：索引取教师 top-k。输入须已温度缩放。

    post_attn=False：k 维重归一化（严格 KL）；True：全词表 softmax 后 gather（部分 KL）。
    """
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = teacher_scaled.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        teacher_prob = F.softmax(teacher_scaled, dim=-1).gather(-1, indices)
        student_log_prob = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
    else:
        teacher_prob = F.softmax(teacher_scaled.gather(-1, indices), dim=-1)
        student_log_prob = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
    return _masked_token_kl_sum(
        student_log_prob=student_log_prob,
        teacher_prob=teacher_prob,
        mask=mask,
    )


def _topk_reverse_kl_sum(
    *,
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    post_attn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse KL（学生‖教师）top-k 版：索引取学生 top-k。输入须已温度缩放。

    手写 Σ p_s(log p_s - log p_t)：kl_div 不对 target 反传，学生梯度会断。
    """
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = student_scaled.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        log_s = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
        log_t = F.log_softmax(teacher_scaled, dim=-1).gather(-1, indices)
    else:
        log_s = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
        log_t = F.log_softmax(teacher_scaled.gather(-1, indices), dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    return (token_kl * mask_fp).sum(), mask_fp.sum()


def compute_kl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    post_attn: bool = False,
) -> torch.Tensor:
    """Forward KL（教师‖学生），只在教师 top-k 上计算。"""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], device=student_logits.device, dtype=torch.float32)
    temp = _resolve_temperature(temperature)
    fwd_sum, n = _topk_forward_kl_sum(
        student_scaled=student_logits.float() / temp,
        teacher_scaled=teacher_logits.detach().float() / temp,
        mask=mask,
        k=int(k),
        post_attn=bool(post_attn),
    )
    return (fwd_sum / n.clamp_min(1.0)) * (temp * temp)


def compute_rkl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    post_attn: bool = False,
) -> torch.Tensor:
    """Reverse KL（学生‖教师），只在学生 top-k 上计算。"""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], device=student_logits.device, dtype=torch.float32)
    temp = _resolve_temperature(temperature)
    rev_sum, n = _topk_reverse_kl_sum(
        student_scaled=student_logits.float() / temp,
        teacher_scaled=teacher_logits.detach().float() / temp,
        mask=mask,
        k=int(k),
        post_attn=bool(post_attn),
    )
    return (rev_sum / n.clamp_min(1.0)) * (temp * temp)


def compute_eakld_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    confidence_k: int = 16,
    post_attn: bool = False,
) -> torch.Tensor:
    """EAKLD 的 top-k 版：γ 沿用全词表教师熵，FKL/RKL 各自取教师/学生 top-k。"""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], device=student_logits.device, dtype=torch.float32)
    temp = _resolve_temperature(temperature)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    sum_e, sum_v = accumulate_teacher_entropy_stats(teacher_logits, mask)
    gamma = gamma_from_entropy_sums(sum_e, sum_v, confidence_k=int(confidence_k))
    rev_sum, n = _topk_reverse_kl_sum(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=mask,
        k=int(k),
        post_attn=bool(post_attn),
    )
    fwd_sum, _ = _topk_forward_kl_sum(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=mask,
        k=int(k),
        post_attn=bool(post_attn),
    )
    reverse_kl = (rev_sum / n.clamp_min(1.0)) * (temp * temp)
    forward_kl = (fwd_sum / n.clamp_min(1.0)) * (temp * temp)
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl


def _masked_mean_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    a = a.float().detach()
    b = b.float().detach()
    cos = F.cosine_similarity(a, b, dim=-1)
    if attention_mask is None:
        return cos.mean()
    mask = attention_mask.to(device=cos.device, dtype=cos.dtype)
    while mask.ndim < cos.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(cos)
    return (cos * mask).sum() / mask.sum().clamp_min(1.0)


def select_adaptive_layer_indices(
    teacher_block_hiddens: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    *,
    topk: int = 3,
    reference_hidden: Optional[torch.Tensor] = None,
) -> list[int]:
    num_layers = len(teacher_block_hiddens)
    if num_layers <= 0:
        raise ValueError("teacher_block_hiddens must be non-empty")
    topk = min(max(1, int(topk)), num_layers)
    scores: list[tuple[int, float]] = []
    for layer_idx in range(num_layers):
        hidden = teacher_block_hiddens[layer_idx]
        previous = reference_hidden if layer_idx == 0 else teacher_block_hiddens[layer_idx - 1]
        if previous is None:
            raise ValueError("reference_hidden is required for layer 0 adaptive selection")
        # move to same device for cosine
        if previous.device != hidden.device:
            previous = previous.to(device=hidden.device)
        cosine = _masked_mean_cosine(hidden, previous, attention_mask)
        scores.append((layer_idx, float(cosine.item())))
    selected = sorted(scores, key=lambda item: item[1])[:topk]
    return [idx for idx, _ in selected]


def compute_lafd_mse_selected(
    *,
    teacher_hiddens: Sequence[torch.Tensor],
    student_hiddens: Sequence[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> torch.Tensor:
    if len(teacher_hiddens) == 0:
        raise ValueError("teacher_hiddens must be non-empty")
    if len(teacher_hiddens) != len(student_hiddens):
        raise ValueError("teacher/student selected hidden count mismatch")

    pad_mask = labels.eq(-100) if labels is not None else None
    layer_losses: list[torch.Tensor] = []
    # layer 切分时各层可能在不同 GPU；loss 标量归到第一层 student 设备
    loss_device = student_hiddens[0].device
    for teacher_h, student_h in zip(teacher_hiddens, student_hiddens):
        # teacher 可在 CPU：逐层搬到对应 student 设备
        teacher_h = teacher_h.detach()
        if teacher_h.device != student_h.device:
            teacher_h = teacher_h.to(device=student_h.device, non_blocking=True)
        teacher_h = teacher_h.float()
        student_h = student_h.float()
        if teacher_h.shape != student_h.shape:
            raise ValueError(f"Hidden shape mismatch: {tuple(student_h.shape)} vs {tuple(teacher_h.shape)}")
        mse = F.mse_loss(student_h, teacher_h, reduction="none")
        del teacher_h
        if pad_mask is not None:
            mask = pad_mask.to(device=mse.device)
            mse = mse.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_tokens = (~mask).sum(dim=-1).float().clamp_min(1.0)
            feature_dim = float(student_h.shape[-1])
            per_sample = mse.view(mse.size(0), -1).sum(dim=-1) / (valid_tokens * feature_dim)
            layer_losses.append(per_sample.mean().to(device=loss_device))
        else:
            layer_losses.append(mse.mean().to(device=loss_device))
    return torch.stack(layer_losses).mean()


def compute_lafd_mse(
    *,
    teacher_hidden_states: Sequence[torch.Tensor],
    student_hidden_states: Sequence[torch.Tensor],
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    topk: int = 3,
) -> torch.Tensor:
    if len(teacher_hidden_states) <= 1:
        raise ValueError("Need embedding + at least one block hidden state")
    teacher_blocks = [h.detach() for h in teacher_hidden_states[1:]]
    student_blocks = list(student_hidden_states[1:])
    reference = teacher_hidden_states[0].detach()
    select_mask = attention_mask
    if select_mask is None and labels is not None:
        select_mask = labels.ne(-100).to(dtype=torch.long)
    selected = select_adaptive_layer_indices(
        teacher_blocks, select_mask, topk=int(topk), reference_hidden=reference
    )
    return compute_lafd_mse_selected(
        teacher_hiddens=[teacher_blocks[i] for i in selected],
        student_hiddens=[student_blocks[i] for i in selected],
        labels=labels,
    )


def compute_edgerazor_qad_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden_states: Sequence[torch.Tensor],
    teacher_hidden_states: Sequence[torch.Tensor],
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    task_alpha: float = 0.05,
    eakld_alpha: float = 2.0,
    lafd_alpha: float = 0.5,
    temperature: float = 1.0,
    confidence_k: int = 16,
    lafd_topk: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = build_token_mask(labels=labels, attention_mask=attention_mask, reference=student_logits)
    if ce_loss is None:
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    eakld = compute_eakld(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=float(temperature),
        confidence_k=int(confidence_k),
    )
    lafd = compute_lafd_mse(
        teacher_hidden_states=teacher_hidden_states,
        student_hidden_states=student_hidden_states,
        labels=labels,
        attention_mask=attention_mask,
        topk=int(lafd_topk),
    )
    total = float(task_alpha) * ce_loss + float(eakld_alpha) * eakld + float(lafd_alpha) * lafd
    return total, {
        "ce": ce_loss.detach(),
        "eakld": eakld.detach(),
        "lafd": lafd.detach(),
        "total": total.detach(),
    }


def compute_edgerazor_qad_loss_chunked(
    *,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    teacher_lafd_hiddens: Sequence[torch.Tensor],
    student_lafd_hiddens: Sequence[torch.Tensor],
    chunk_size: int = 512,
    task_alpha: float = 0.05,
    eakld_alpha: float = 2.0,
    lafd_alpha: float = 0.5,
    temperature: float = 1.0,
    confidence_k: int = 16,
    kl_mode: str = "eakld",
    kl_topk: int = 0,
    kl_post_attn: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """分块 lm_head；全局 γ 与 sum/count，与整段公式等价。

    teacher_hidden / teacher_lafd_hiddens 可在 CPU：按 chunk/层搬到 GPU，峰值更低。

    kl_mode: "eakld"=全词表 EAKLD（默认）；"eakld_topk"=top-k 版 EAKLD
    （forward 取教师 top-k，reverse 取学生 top-k）；"kl_topk"=仅 forward
    top-k KL（跳过教师熵预扫描）。kl_topk 为 top-k 的 k；
    kl_post_attn=False 时 k 维重归一化，True 时全词表 softmax 后 gather。
    loss_dict["eakld"] 始终为当前激活的 KL 项。
    """
    if kl_mode not in ("eakld", "eakld_topk", "kl_topk"):
        raise ValueError(f"Unknown kl_mode: {kl_mode!r}")
    if kl_mode != "eakld":
        if int(kl_topk) <= 0:
            raise ValueError(f"kl_topk must be > 0 for kl_mode={kl_mode!r}, got {kl_topk}")
        kl_topk = int(kl_topk)
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"last_hidden shape mismatch: {tuple(student_hidden.shape)} vs {tuple(teacher_hidden.shape)}"
        )
    seq_len = int(student_hidden.shape[1])
    mask = build_token_mask(labels=labels, attention_mask=attention_mask, reference=student_hidden)
    temp = _resolve_temperature(temperature)
    chunk = max(1, int(chunk_size))
    reduce_device = student_hidden.device

    def _teacher_chunk_to_device(start: int, end: int) -> torch.Tensor:
        # 允许整段 teacher 在 CPU；只搬当前 chunk
        piece = teacher_hidden[:, start:end, :].detach()
        if piece.device != reduce_device:
            piece = piece.to(device=reduce_device, non_blocking=True)
        return piece

    # 先算 LAFD（标量进计算图），再跑超长序列的 CE/EAKLD，避免同时峰值叠两份大激活
    lafd = compute_lafd_mse_selected(
        teacher_hiddens=teacher_lafd_hiddens,
        student_hiddens=student_lafd_hiddens,
        labels=labels,
    ).to(device=reduce_device).float()

    # kl_topk 模式不需要 γ，跳过教师熵预扫描（省一遍教师 lm_head）
    gamma: torch.Tensor | None = None
    if kl_mode != "kl_topk":
        sum_entropy = torch.zeros((), device=reduce_device, dtype=torch.float32)
        sum_valid = torch.zeros((), device=reduce_device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, seq_len, chunk):
                end = min(seq_len, start + chunk)
                t_h = _teacher_chunk_to_device(start, end)
                t_logits = lm_head(t_h)
                e_sum, v_sum = accumulate_teacher_entropy_stats(t_logits, mask[:, start:end])
                sum_entropy = sum_entropy + e_sum.to(device=reduce_device, dtype=torch.float32)
                sum_valid = sum_valid + v_sum.to(device=reduce_device, dtype=torch.float32)
                del t_logits, t_h
        gamma = gamma_from_entropy_sums(sum_entropy, sum_valid, confidence_k=int(confidence_k))

    ce_sum = torch.zeros((), device=reduce_device, dtype=torch.float32)
    ce_count = torch.zeros((), device=reduce_device, dtype=torch.float32)
    rev_sum = torch.zeros((), device=reduce_device, dtype=torch.float32)
    fwd_sum = torch.zeros((), device=reduce_device, dtype=torch.float32)
    kl_count = torch.zeros((), device=reduce_device, dtype=torch.float32)

    # 长序列缩小 KL 微批，降低 softmax 峰值（数值仍是全局 sum/count）
    micro = 2 if seq_len >= 8192 else 8

    for start in range(0, seq_len, chunk):
        end = min(seq_len, start + chunk)
        s_logits = lm_head(student_hidden[:, start:end, :])
        with torch.no_grad():
            t_logits = lm_head(_teacher_chunk_to_device(start, end))
        m = mask[:, start:end].to(device=s_logits.device)

        # HF causal: logits[i] predicts labels[i+1]
        i0 = start
        i1 = min(end, seq_len - 1)
        if i1 > i0:
            local_logits = s_logits[:, : i1 - start, :]
            local_labels = labels[:, i0 + 1 : i1 + 1].to(device=local_logits.device)
            flat_logits = local_logits.reshape(-1, local_logits.size(-1))
            flat_labels = local_labels.reshape(-1)
            valid = flat_labels.ne(-100)
            if bool(valid.any()):
                token_ce = F.cross_entropy(
                    flat_logits.float(), flat_labels, reduction="none", ignore_index=-100
                )
                ce_sum = ce_sum + token_ce.sum().to(device=reduce_device)
                ce_count = ce_count + valid.to(dtype=torch.float32).sum().to(device=reduce_device)

        s_scaled = s_logits.float() / temp
        t_scaled = t_logits.detach().float() / temp
        del s_logits, t_logits
        micro_n = min(micro, int(s_scaled.shape[1]))
        for u0 in range(0, int(s_scaled.shape[1]), micro_n):
            u1 = min(int(s_scaled.shape[1]), u0 + micro_n)
            ss = s_scaled[:, u0:u1, :]
            ts = t_scaled[:, u0:u1, :]
            mm = m[:, u0:u1]
            if kl_mode == "eakld":
                r_sum, n = _masked_token_reverse_kl_sum(
                    student_logits=ss,
                    teacher_logits=ts,
                    mask=mm,
                )
                rev_sum = rev_sum + r_sum.to(device=reduce_device)
                kl_count = kl_count + n.to(device=reduce_device, dtype=torch.float32)
                del r_sum, n
                f_sum, _ = _masked_token_kl_sum(
                    student_log_prob=F.log_softmax(ss, dim=-1),
                    teacher_prob=F.softmax(ts, dim=-1),
                    mask=mm,
                )
                fwd_sum = fwd_sum + f_sum.to(device=reduce_device)
                del f_sum
            elif kl_mode == "eakld_topk":
                r_sum, n = _topk_reverse_kl_sum(
                    student_scaled=ss,
                    teacher_scaled=ts,
                    mask=mm,
                    k=kl_topk,
                    post_attn=kl_post_attn,
                )
                rev_sum = rev_sum + r_sum.to(device=reduce_device)
                kl_count = kl_count + n.to(device=reduce_device, dtype=torch.float32)
                del r_sum, n
                f_sum, _ = _topk_forward_kl_sum(
                    student_scaled=ss,
                    teacher_scaled=ts,
                    mask=mm,
                    k=kl_topk,
                    post_attn=kl_post_attn,
                )
                fwd_sum = fwd_sum + f_sum.to(device=reduce_device)
                del f_sum
            else:  # kl_topk：仅 forward top-k KL
                f_sum, n = _topk_forward_kl_sum(
                    student_scaled=ss,
                    teacher_scaled=ts,
                    mask=mm,
                    k=kl_topk,
                    post_attn=kl_post_attn,
                )
                fwd_sum = fwd_sum + f_sum.to(device=reduce_device)
                kl_count = kl_count + n.to(device=reduce_device, dtype=torch.float32)
                del f_sum, n
            del ss, ts, mm
        del s_scaled, t_scaled, m

    ce_loss = ce_sum / ce_count.clamp_min(1.0)
    forward_kl = (fwd_sum / kl_count.clamp_min(1.0)) * (temp * temp)
    if kl_mode == "kl_topk":
        eakld = forward_kl
    else:
        reverse_kl = (rev_sum / kl_count.clamp_min(1.0)) * (temp * temp)
        eakld = gamma * reverse_kl + (1.0 - gamma) * forward_kl
    total = float(task_alpha) * ce_loss + float(eakld_alpha) * eakld + float(lafd_alpha) * lafd
    loss_dict = {
        "ce": ce_loss.detach(),
        "eakld": eakld.detach(),
        "lafd": lafd.detach(),
        "total": total.detach(),
    }
    if gamma is not None:
        loss_dict["gamma"] = gamma.detach()
    return total, loss_dict

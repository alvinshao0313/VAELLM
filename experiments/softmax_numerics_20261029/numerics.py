"""Isolated Eq. 5/6/9 operators; never imported by a training entry point."""
import torch
from torch.nn import functional as F


def fixed_log_softmax(logits):
    """PyTorch translation of the paper's Eq. 5, including its constant max slot."""
    index = logits.argmax(dim=-1, keepdim=True)
    selected = logits.gather(-1, index)
    mask = torch.arange(logits.shape[-1], device=logits.device) == index
    shifted = torch.where(mask, 0., logits - selected)
    others = shifted.masked_fill(mask, -torch.inf)
    return shifted - F.softplus(torch.logsumexp(others, dim=-1, keepdim=True))


def fixed_hard_cross_entropy(logits, target):
    """Eq. 6, with no mask, temperature, or reduction policy added."""
    index = target.unsqueeze(-1)
    selected = logits.gather(-1, index)
    mask = torch.arange(logits.shape[-1], device=logits.device) == index
    others = (logits - selected).masked_fill(mask, -torch.inf)
    return F.softplus(torch.logsumexp(others, dim=-1))


def raw_teacher_gradient(student_logits, teacher_logits):
    """Eq. 9 and the paper's |s|<1 fallback and zero-sum correction."""
    with torch.no_grad():
        log_q = fixed_log_softmax(student_logits)
        q = log_q.exp()
        p = fixed_log_softmax(teacher_logits).exp()
        delta = student_logits - teacher_logits
        centered = delta - (p * delta).sum(-1, keepdim=True)
        ratio = centered - torch.log1p((p * torch.expm1(centered)).sum(-1, keepdim=True))
        difference = torch.where(ratio.abs() < 1, p * torch.expm1(ratio), q - p)
        gradient = difference - q * difference.sum(-1, keepdim=True)
        if not torch.isfinite(gradient).all().item():
            raise ValueError('Eq. 9 is nonfinite for these raw logits; do not repair or replace inputs.')
    return gradient


def fixed_raw_soft_cross_entropy(student_logits, teacher_logits):
    """The paper's soft CE forward with its substituted Eq. 9 student gradient."""
    teacher = teacher_logits.detach()
    with torch.no_grad():
        loss = -(fixed_log_softmax(teacher).exp() * fixed_log_softmax(student_logits)).sum(-1)
    gradient = raw_teacher_gradient(student_logits.detach(), teacher)
    return loss + (gradient * (student_logits - student_logits.detach())).sum(-1)


def cached_kl(logits, cached_logp, weight=1.):
    """Existing R branch semantics: full-vocabulary sum / real tokens, then doc weight."""
    logq = F.log_softmax(logits, dim=-1)
    return F.kl_div(logq, cached_logp, log_target=True, reduction='sum') / len(logq) * weight


def cached_kl_reference(logits, cached_logp, weight=1.):
    """FP64 reference keeps cached target mass instead of silently renormalizing it."""
    logq = fixed_log_softmax(logits)
    return (cached_logp.exp() * (cached_logp - logq)).sum() / len(logq) * weight


def value_gradient(function, logits):
    variable = logits.detach().clone().requires_grad_(True)
    value = function(variable)
    gradient, = torch.autograd.grad(value, variable)
    return value.detach(), gradient.detach()


def vector_comparison(reference, candidate):
    reference, candidate = reference.double(), candidate.double()
    difference = candidate - reference
    refnorm, norm = float(reference.norm()), float(candidate.norm())
    return dict(reference_l2=refnorm, candidate_l2=norm, error_l2=float(difference.norm()),
        error_max=float(difference.abs().max()),
        relative_l2=float(difference.norm()) / refnorm if refnorm else None,
        cosine=float((reference * candidate).sum()) / (refnorm * norm) if refnorm and norm else None,
        candidate_max_abs_row_sum=float(candidate.sum(-1).abs().max()),
        reference_max_abs_row_sum=float(reference.sum(-1).abs().max()))

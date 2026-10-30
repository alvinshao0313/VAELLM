"""Same-target four-option KL audit; no teacher renormalization or training patch."""
import math

import torch
from torch.nn import functional as F

from numerics import fixed_log_softmax, value_gradient

WEIGHT = .5 / 32


def vector_metrics(reference, current):
    reference, current = reference.detach().double(), current.detach().double()
    difference = current - reference
    refnorm, currentnorm = float(reference.norm()), float(current.norm())
    zeroed = (reference != 0) & (current == 0)
    invented = (reference == 0) & (current != 0)
    flipped = (reference * current < 0)
    return dict(reference_l2=refnorm, current_l2=currentnorm, error_l2=float(difference.norm()),
        max_absolute_error=float(difference.abs().max()),
        relative_l2=float(difference.norm())/refnorm if refnorm else None,
        cosine=float((reference*current).sum())/(refnorm*currentnorm) if refnorm and currentnorm else None,
        reference_zero_elements=int((reference == 0).sum()), current_zero_elements=int((current == 0).sum()),
        reference_nonzero_current_zero=int(zeroed.sum()),
        zeroed_reference_max_abs=float(reference[zeroed].abs().max()) if zeroed.any() else 0.,
        reference_zero_current_nonzero=int(invented.sum()),
        invented_current_max_abs=float(current[invented].abs().max()) if invented.any() else 0.,
        sign_flipped_nonzero_elements=int(flipped.sum()),
        sign_flipped_reference_max_abs=float(reference[flipped].abs().max()) if flipped.any() else 0.,
        sign_flipped_error_max_abs=float(difference[flipped].abs().max()) if flipped.any() else 0.)


def audit_logits(selected, cached_logp, weighted_loss):
    if selected.shape != cached_logp.shape or selected.shape != (32, 4):
        raise ValueError('Keep original M32/four-option support.')
    z32, lp32 = selected.detach().float(), cached_logp.detach().to(selected.device)
    current_loss, current = value_gradient(
        lambda z: weighted_loss(F.log_softmax(z, -1), lp32)[0], z32)
    z64, lp64 = z32.double(), lp32.double()
    p = lp64.exp()
    mass = p.sum(-1, keepdim=True)
    logq64 = fixed_log_softmax(z64)
    reference = WEIGHT * (mass*logq64.exp() - p)
    reference_loss, autodiff = value_gradient(
        lambda z: (p*(lp64-fixed_log_softmax(z))).sum(-1).sum()*WEIGHT, z64)
    crosscheck = vector_metrics(autodiff, reference)
    # Bound the absolute arithmetic discrepancy at normalized-probability scale.
    bound = 32*torch.finfo(torch.float64).eps*WEIGHT*float(mass.abs().max())
    if crosscheck['max_absolute_error'] > bound:
        raise ValueError('FP64 analytic and stable-autodiff same-target gradients disagree.')
    current_boundary = current.bfloat16()
    reference_boundary = reference.float().bfloat16()
    argmax = z64.argmax(-1, keepdim=True)
    excluded = z64.scatter(-1, argmax, -torch.inf)
    margins = z64.gather(-1, argmax).squeeze(-1) - torch.logsumexp(excluded, -1)
    rows = []
    for index in range(32):
        top = int(argmax[index])
        rows.append(dict(row=index, student_argmax=top, student_margin=float(margins[index]),
            target_mass=float(mass[index, 0]), formula=vector_metrics(reference[index], current[index]),
            BF16_boundary=vector_metrics(reference_boundary[index], current_boundary[index]),
            top_current_gradient=float(current[index, top]), top_reference_gradient=float(reference[index, top]),
            top_current_zero_reference_nonzero=bool(current[index, top] == 0 and reference[index, top] != 0)))
    data = dict(selected_BF16_logits=selected.detach().cpu(), cached_mixed_FP32_logp=lp32.cpu(),
        current_FP32_gradient=current.cpu(), reference_FP64_gradient=reference.cpu(),
        current_BF16_boundary=current_boundary.cpu(), reference_BF16_boundary=reference_boundary.cpu())
    report = dict(current_weighted_loss=float(current_loss), reference_same_target_FP64_loss=float(reference_loss),
        absolute_loss_error=abs(float(current_loss)-float(reference_loss)),
        formula=vector_metrics(reference, current), BF16_boundary=vector_metrics(reference_boundary, current_boundary),
        FP64_analytic_vs_stable_autodiff=crosscheck, FP64_crosscheck_absolute_bound=bound,
        BF16_boundary_exact=torch.equal(current_boundary, reference_boundary),
        BF16_boundary_changed_elements=int((current_boundary != reference_boundary).sum()),
        student_max_margin=float(margins.max()), student_min_margin=float(margins.min()),
        student_margin_above_16_6=int((margins > 16.6).sum()),
        high_margin_zeroed_top_gradient_rows=[row['row'] for row in rows
            if row['student_margin'] > 16.6 and row['top_current_zero_reference_nonzero']],
        max_abs_cached_mass_minus_one=float((mass-1).abs().max()), rows=rows,
        semantics='(.5/32)*(a*q-p); p=exp(exact stored FP32 logp read in FP64), a=sum(p); no target renormalization.')
    return report, data, current_boundary, reference_boundary


def parameter_comparison(names, current, reference):
    rows = {}
    totals = dict(reference_sq=0., current_sq=0., error_sq=0., dot=0., max_absolute_error=0.,
        elements=0, sign_flipped_nonzero_elements=0, reference_nonzero_current_zero=0,
        reference_zero_current_nonzero=0)
    for name, actual, expected in zip(names, current, reference, strict=True):
        if actual is None or expected is None or actual.dtype != torch.float32 or expected.dtype != torch.float32:
            raise ValueError('Every actual live decoder leaf must have an FP32 VJP: '+name)
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            raise ValueError('Nonfinite actual parameter VJP: '+name)
        values = vector_metrics(expected, actual)
        values.update(elements=actual.numel(), shape=list(actual.shape))
        rows[name] = values
        totals['reference_sq'] += values['reference_l2']**2
        totals['current_sq'] += values['current_l2']**2
        totals['error_sq'] += values['error_l2']**2
        totals['dot'] += float((actual.double()*expected.double()).sum())
        totals['max_absolute_error'] = max(totals['max_absolute_error'], values['max_absolute_error'])
        totals['elements'] += actual.numel()
        for key in ('sign_flipped_nonzero_elements', 'reference_nonzero_current_zero', 'reference_zero_current_nonzero'):
            totals[key] += values[key]
    refnorm, currentnorm, errornorm = (math.sqrt(totals[key]) for key in ('reference_sq','current_sq','error_sq'))
    aggregate = dict(reference_l2=refnorm, current_l2=currentnorm, error_l2=errornorm,
        relative_l2=errornorm/refnorm if refnorm else None,
        cosine=totals['dot']/(refnorm*currentnorm) if refnorm and currentnorm else None,
        max_absolute_error=totals['max_absolute_error'], elements=totals['elements'], tensors=len(rows),
        sign_flipped_nonzero_elements=totals['sign_flipped_nonzero_elements'],
        reference_nonzero_current_zero=totals['reference_nonzero_current_zero'],
        reference_zero_current_nonzero=totals['reference_zero_current_nonzero'],
        error_dominant_entire_leaves=[name for name, row in rows.items()
            if row['error_l2'] > 0 and row['error_l2'] >= row['reference_l2']],
        nonzero_reference_leaves_over_1e_3=[name for name, row in rows.items()
            if row['reference_l2'] > 0 and row['relative_l2'] >= 1e-3])
    return dict(aggregate=aggregate, leaves=rows,
        difference_definition='Two complete VJPs; promote each FP32 parameter gradient to FP64 before subtraction. No standalone delta-logit VJP or Adam comparison.')

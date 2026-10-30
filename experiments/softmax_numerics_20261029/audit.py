"""Audit captured real branch logits without loading or changing a model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import numerics as n


def branch_audit(item, device):
    student = item['batch_logits'].to(device=device, dtype=torch.float32)
    teacher = item['serial_logits'].to(device=device, dtype=torch.float32)
    cached = item['cached_logp'].to(device=device, dtype=torch.float32)
    if student.shape != teacher.shape or student.shape != cached.shape:
        raise ValueError('Capture supports do not agree.')
    weight = item['document_branch_weight']
    old_loss, old_gradient = n.value_gradient(lambda z: n.cached_kl(z, cached, weight), student)
    ref_loss, ref_gradient = n.value_gradient(
        lambda z: n.cached_kl_reference(z, cached.double(), weight), student.double())
    raw_logp64 = n.fixed_log_softmax(teacher.double())
    raw_loss, raw_gradient = n.value_gradient(
        lambda z: n.cached_kl_reference(z, raw_logp64, weight), student.double())
    replayed = item['replayed_logp'].to(device=device)
    mass = cached.double().exp().sum(-1)
    normalized_cached = cached.double().exp() / mass.unsqueeze(-1)
    raw_probability = raw_logp64.exp()
    identity_target = F.log_softmax(student, dim=-1).detach()
    identity_loss, identity_gradient = n.value_gradient(lambda z: n.cached_kl(z, identity_target, weight), student)
    identity_ref_loss, identity_ref_gradient = n.value_gradient(
        lambda z: n.cached_kl_reference(z, identity_target.double(), weight), student.double())
    paper_identity = n.raw_teacher_gradient(student, student) * (weight / len(student))
    paper_gradient = n.raw_teacher_gradient(student, teacher) * (weight / len(student))
    fresh_mass = replayed.double().exp().sum(-1)
    fresh_probability = replayed.double().exp() / fresh_mass.unsqueeze(-1)
    target_index = student.argmax(-1, keepdim=True)
    other_logits = student.double().scatter(-1, target_index, -torch.inf)
    margins = student.double().gather(-1, target_index).squeeze(-1) - torch.logsumexp(other_logits, -1)
    replayed_error = (cached - replayed).abs()
    return dict(case_id=item['case_id'], branch_index=item['branch_index'], shape=list(student.shape),
        document_branch_weight=weight, token_mean_denominator=len(student),
        current_cached_fp32=dict(loss=float(old_loss), reference_same_cached_fp64=float(ref_loss),
            loss_absolute_error=abs(float(old_loss)-float(ref_loss)),
            gradient=n.vector_comparison(ref_gradient, old_gradient),
            gradient_at_BF16_logits_boundary=n.vector_comparison(ref_gradient.bfloat16(), old_gradient.bfloat16()),
            BF16_boundary_changed_gradient_elements=int((ref_gradient.bfloat16()!=old_gradient.bfloat16()).sum())),
        cache_target=dict(max_abs_mass_minus_one=float((mass-1).abs().max()),
            mass_min=float(mass.min()), mass_max=float(mass.max()),
            saved_vs_serial_replay_logp_exact=torch.equal(cached, replayed),
            saved_vs_serial_replay_logp_max_abs=float(replayed_error.max()),
            saved_vs_raw_teacher_normalized_max_tv=float((normalized_cached-raw_probability).abs().sum(-1).max()/2),
            fresh_fp32_logp_vs_raw_teacher_normalized_max_tv=float((fresh_probability-raw_probability).abs().sum(-1).max()/2),
            saved_vs_raw_teacher_gradient=n.vector_comparison(raw_gradient, ref_gradient),
            loss_with_normalized_raw_teacher_fp64=float(raw_loss),
            note='Saved target is preserved. Saved-vs-replay also includes historical replay/kernel differences; BF16 head quantization is not isolated.'),
        identical_raw_logits=dict(current_fp32_loss=float(identity_loss),
            same_cached_fp64_loss=float(identity_ref_loss),
            current_vs_same_cached_fp64=n.vector_comparison(identity_ref_gradient, identity_gradient),
            same_cached_fp64_residual_l2=float(identity_ref_gradient.norm()),
            paper_eq9_raw_identity_l2=float(paper_identity.double().norm()),
            paper_eq9_raw_identity_nonzero_elements=int(torch.count_nonzero(paper_identity)),
            note='Only exact raw-logit identity has a zero true normalized-teacher gradient; rounded cached logp has its own exact objective.'),
        actual_forward_path_difference=dict(logits_exact=torch.equal(student, teacher),
            logits_max_abs=float((student-teacher).abs().max()),
            raw_teacher_gradient_fp64_l2=float(raw_gradient.norm()),
            paper_eq9_gradient=n.vector_comparison(raw_gradient, paper_gradient),
            note='Same G02 BF16 model: serial cache forward versus padded native task_batch. This is a real path difference, not automatically a softmax bug.'),
        confidence=dict(max_margin=float(margins.max()), min_margin=float(margins.min()),
            token_count_margin_gt_fp32_cliff=int((margins>16.6).sum())),
        model_precision_scope='Frozen native BF16 G02 only; FP32 decoder/live and different-model gradients are not sampled.')


def audit_capture(capture, device):
    if capture['schema'] != 'softmax_R_boolq_first4_capture_v1':
        raise ValueError('Unexpected real-logit capture schema.')
    rows = [branch_audit(item, device) for item in capture['branches']]
    return dict(schema='softmax_R_numeric_audit_v1', status='complete_numeric_diagnostic_only',
        operator_device=str(device), torch_version=torch.__version__, input_protocol=capture['protocol'],
        model_source=capture['source'], rows=rows,
        aggregate=dict(original_task_weight=.5/7,
            current_cached_fp32_task_loss=sum(row['current_cached_fp32']['loss'] for row in rows),
            same_cached_fp64_task_loss=sum(row['current_cached_fp32']['reference_same_cached_fp64'] for row in rows),
            weighted_formula_gradient_error_l2=sum(row['current_cached_fp32']['gradient']['error_l2']**2 for row in rows)**.5,
            weighted_same_cached_reference_gradient_l2=sum(row['current_cached_fp32']['gradient']['reference_l2']**2 for row in rows)**.5,
            weighted_actual_path_raw_gradient_l2=sum(row['actual_forward_path_difference']['raw_teacher_gradient_fp64_l2']**2 for row in rows)**.5),
        interpretation='Absolute logit-gradient errors only. No parameter update, no nine-task benefit, no global softmax/attention change.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise ValueError('Do not overwrite an existing numeric audit.')
    capture = torch.load(args.capture, map_location='cpu', weights_only=True)
    report = audit_capture(capture, torch.device(args.device))
    report['capture'] = dict(path=str(args.capture.resolve()), bytes=args.capture.stat().st_size,
        sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(args.output), aggregate=report['aggregate'])), flush=True)


if __name__ == '__main__':
    main()

"""Same M32 native queries, with two predeclared target objectives."""
from . import bindings as b


def distributions(model, cases):
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    inputs, options = b.mature.mmlu.batch_inputs(cases, pad)
    device = model.get_input_embeddings().weight.device
    output = model(**{key: value.to(device) for key, value in inputs.items()}, use_cache=False, logits_to_keep=1)
    b.require(output.past_key_values is None and output.logits.shape == (32, 1, 151936)
        and output.logits.dtype == b.torch.bfloat16, 'Keep the real M32 native last-query BF16 head.')
    logits = output.logits[:, 0].float()
    full = b.torch.nn.functional.log_softmax(logits, dim=-1)
    conditional = b.torch.nn.functional.log_softmax(logits.gather(1, options.to(device)), dim=-1)
    return full, conditional, options


def loss(model, cases, targets, arm, *, diagnostics=False):
    b.require(arm in b.TARGET_ARMS and len(cases) == 32, 'Use one complete declared M32 target arm.')
    full, conditional, options = distributions(model, cases)
    teacher = b.torch.cat([targets[case['case_id']]['target_log_probs'] for case in cases]).to(full.device)
    for case in cases:
        b.require(targets[case['case_id']]['input_signature'] == case['input_signature']
            and case['source']['official_split'] == 'train', 'Bind every full target to its original train input.')
    b.require(teacher.shape == full.shape == (32, 151936) and teacher.dtype == b.torch.float32
        and not teacher.requires_grad and b.torch.isfinite(teacher).all(), 'Keep the full vocabulary teacher target.')
    gold = b.torch.tensor([case['gold_index'] for case in cases], device=full.device)
    ce = -conditional.gather(1, gold[:, None]).squeeze(1)
    full_kl = b.torch.nn.functional.kl_div(full, teacher, log_target=True, reduction='none').sum(-1)
    values = .5 * full_kl + .5 * ce if arm == 'full_teacher' else ce
    result = values.mean() * .5
    b.require(b.torch.isfinite(result).item(), 'Nonfinite M target objective.')
    records = None
    if diagnostics:
        selected = teacher.gather(1, options.to(full.device))
        teacher_conditional = selected - b.torch.logsumexp(selected, dim=-1, keepdim=True)
        cond_kl = b.torch.nn.functional.kl_div(conditional, teacher_conditional,
            log_target=True, reduction='none').sum(-1)
        teacher_mass = selected.logsumexp(-1).exp()
        student_mass = full.gather(1, options.to(full.device)).logsumexp(-1).exp()
        records = dict(case_ids=[case['case_id'] for case in cases],
            teacher_candidate_mass=teacher_mass.detach().cpu().tolist(),
            student_candidate_mass=student_mass.detach().cpu().tolist(),
            full_KL=full_kl.detach().cpu().tolist(), conditional_KL=cond_kl.detach().cpu().tolist(),
            gold_conditional_CE=ce.detach().cpu().tolist(),
            new_full_teacher_conditional_equivalent_objective=float((.5 * cond_kl + .5 * ce).mean().detach() * .5),
            new_weighted_objective=float(result.detach()), options=options.tolist(),
            answer_query_positions=[case['answer_query_positions'].tolist() for case in cases],
            domain_weight=.5, teacher_fraction=.5 if arm == 'full_teacher' else 0.,
            gold_fraction=.5 if arm == 'full_teacher' else 1.,
            old_loss_constant='The cached mixed-target KL differs from teacher-KL/gold-CE only by a target-only entropy constant.')
    return result, records

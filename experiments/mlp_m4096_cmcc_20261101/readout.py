"""Read the complete fixed train pool by domain; no teacher or sample changes."""
from collections import defaultdict
import json
import math
import time

from . import bindings as b


@b.torch.no_grad()
def evaluate(model, dataset, recovery, indices, baseline=None):
    previous = model.training
    began=time.monotonic()
    model.eval()
    sums, counts, cases, groups = defaultdict(float), defaultdict(int), {}, []
    for index in indices:
        group = dataset['groups'][index]
        m_cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
        target = b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        prediction = b.data.mmlu_log_distribution(model, m_cases).cpu()
        per_case = (target.exp()*(target-prediction)).sum(-1)
        gold = b.torch.tensor([case['gold_index'] for case in m_cases])
        nll = -prediction[b.torch.arange(32),gold]
        correct = prediction.argmax(-1).eq(gold)
        sums['mmlu'] += float(per_case.sum())
        sums['M_gold_NLL'] += float(nll.sum())
        counts['mmlu'] += 32
        counts['M_correct'] += int(correct.sum())
        loss = .5*float(per_case.mean())
        for case, ok, value in zip(m_cases, correct.tolist(), nll.tolist(), strict=True):
            cases[case['case_id']] = dict(correct=ok, gold_NLL=value)
        for key in group['case_ids'][32:]:
            case = recovery[key]['case']
            logits = b.data.legacy.base.answer_logits(model, case)
            value = float(b.data.legacy.base.answer_kl(logits, case, recovery[key]['target_log_probs']))
            sums[case['domain']] += value
            counts[case['domain']] += 1
            loss += value*(.5/8)
            del logits
        groups.append(dict(group_index=index, weighted_objective=loss))
        if len(groups)%16==0:
            print(json.dumps(dict(event='M4096_train_pool_readout_progress',completed_groups=len(groups),
                groups=len(indices),M_cases=counts['mmlu'],M_correct=counts['M_correct'],
                elapsed_seconds=time.monotonic()-began)),flush=True)
    model.train(previous)
    result = dict(group_indices=list(indices), domain_KL={name:sums[name]/counts[name]
        for name in ('mmlu','qasper','hotpotqa') if counts[name]}, presentations=dict(counts),
        M_gold_NLL=sums['M_gold_NLL']/counts['mmlu'], M_correct=counts['M_correct'],
        M_accuracy=counts['M_correct']/counts['mmlu'], M_cases=cases, groups=groups,
        weighted_objective=sum(row['weighted_objective'] for row in groups)/len(groups),
        full_original_pool=list(indices)==list(range(128)))
    if baseline is not None:
        b.require(cases.keys() == baseline['M_cases'].keys(), 'Repair/damage requires the identical fixed train M cases.')
        result['M_repairs'] = sum(not baseline['M_cases'][key]['correct'] and row['correct'] for key,row in cases.items())
        result['M_damages'] = sum(baseline['M_cases'][key]['correct'] and not row['correct'] for key,row in cases.items())
    b.require(all(math.isfinite(value) for value in (*result['domain_KL'].values(),result['M_gold_NLL'],result['weighted_objective'])),
        'Nonfinite actual fixed-pool fitting readout.')
    return result

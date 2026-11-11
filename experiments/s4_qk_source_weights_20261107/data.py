"""Reuse exact original inputs/targets; change only each group's loss weights."""
from collections import Counter
import hashlib
import json
from . import bindings as b
from experiments.s4_webmed_recovery_20261105 import data as original


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_name(key):
    return key.split('/')[0] if key.startswith(('webinstruct/','medmcqa/')) else 'QA'


def group_weights(group):
    result=Counter()
    for key in group['case_ids']:result[source_name(key)]+=group['weights'][key]
    return dict(result)


def rebuild(dataset,targets,old_evidence,weights):
    groups=[]
    for group in dataset['groups']:
        counts=Counter(source_name(key) for key in group['case_ids'])
        b.require(counts==dict(webinstruct=16,medmcqa=16,QA=8),'Keep actual Web16/Med16/QA8 identities.')
        current=dict(group,weights={key:weights[source_name(key)]/counts[source_name(key)] for key in group['case_ids']})
        b.require(group_weights(current)==weights,'Actual group source-weight sums differ.')
        b.require(all(current['weights'][key]==group['weights'][key] for key in group['case_ids'][32:]),'QA weights changed.')
        groups.append(current)
    actual=dict(dataset,groups=groups)
    inputs={case['case_id']:dict(input_signature=case['input_signature'],domain=case['domain']) for case in dataset['training_cases']}
    identity=dict(original_input_target_digest=old_evidence['actual_input_target_records_sha256'],groups=groups,source_weights=weights)
    evidence=dict(schema='S4_QK_exact_WebMed_targets_source_loss_weights_v1',teacher_alpha=.5,teacher_gold_mixture=[.5,.5],
        mixture_applications=1,targets_reused_without_remixing=True,source_weights=weights,
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),
        original_input_target_records_sha256=old_evidence['actual_input_target_records_sha256'],
        actual_input_target_records_sha256=digest(identity),
        input_order_without_weights_sha256=digest(dict(cases=inputs,groups=[{k:v for k,v in g.items() if k!='weights'} for g in groups])),
        QA_target_records_sha256=digest({key:original.legacy.core.tensor_record(item['target_log_probs']) for key,item in targets.items() if source_name(key)=='QA'}),
        all4224_targets_reused_exact=True,all_inputs_order_domain_QA_weights_exact=True,
        first64_exposure=exposure(actual,targets,range(64)),objective='source mean KL times source weight; actual per-case weights consumed directly')
    return actual,targets,evidence


def exposure(dataset,targets,indices):
    indices=list(indices);domains=Counter();inputs=Counter();answers=Counter();groups=[];source_weights=[]
    for index in indices:
        group=dataset['groups'][index];cases=[targets[key]['case'] for key in group['case_ids']]
        b.require(group['group_index']==index and len(cases)==40
            and all(case['domain']=='mmlu' for case in cases[:32])
            and all(case['domain']!='mmlu' for case in cases[32:]),'Keep each actual complete M32/QA8 group.')
        counts=Counter(source_name(case['case_id']) for case in cases);weights=group_weights(group)
        b.require(counts==dict(webinstruct=16,medmcqa=16,QA=8) and weights['QA']==.5
            and weights['webinstruct']+weights['medmcqa']==.5,'Actual source counts or total weights changed.')
        for case in cases:
            key=case['case_id'];domain=case['domain'];source=source_name(key)
            b.require(group['weights'][key]==weights[source]/counts[source],'Unequal per-case source loss weight.')
            domains[domain]+=1;inputs[domain]+=len(case['input_ids'])
            answers[domain]+=targets[key]['target_log_probs'].shape[0]
        groups.append(dict(group_index=index,M_batch_size=32,
            M_padded_tokens=32*max(len(case['input_ids']) for case in cases[:32]),
            M_max_length=max(len(case['input_ids']) for case in cases[:32]),
            QA_max_length=max(len(case['input_ids']) for case in cases[32:]),
            QA_domains=sorted({case['domain'] for case in cases[32:]}),
            M_nonpadding_tokens=sum(len(case['input_ids']) for case in cases[:32]),
            M_target_positions=sum(len(case['target_ids']) for case in cases[:32])))
        source_weights.append(dict(group_index=index,**weights))
    return dict(group_indices=indices,presentations=dict(domains),input_tokens=dict(inputs),answer_positions=dict(answers),
        total_input_tokens=sum(inputs.values()),total_answer_positions=sum(answers.values()),groups=groups,
        actual_source_weight_sums=source_weights)


def load(arm):
    dataset,targets,evidence=original.load()
    dataset,targets,actual=rebuild(dataset,targets,evidence,b.ARMS[arm]['source_weights'])
    b.require(actual==b.read(b.RUN/arm/'data_contract.json'),'Actual source-weight/input/target contract differs from CPU proof.')
    return dataset,targets,actual

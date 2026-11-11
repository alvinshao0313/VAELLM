"""Same source hash prefix4096, exact control M1024 and unchanged static QA targets."""
from collections import Counter
import hashlib
import json

from . import bindings as b
from experiments.mlp_learnability_cmcc_20261030 import data as control

legacy, original = control.legacy, control.original
mmlu_log_distribution = control.mmlu_log_distribution


def exposure(dataset,recovery,indices):
    result=control.exposure(dataset,recovery,indices)
    for row in result['groups']:
        group=dataset['groups'][row['group_index']]
        row['M_nonpadding_tokens']=sum(len(recovery[key]['case']['input_ids']) for key in group['case_ids'][:32])
        row['M_target_positions']=sum(len(recovery[key]['case']['target_ids']) for key in group['case_ids'][:32])
    return result


def input_shapes(dataset):
    by_id={case['case_id']:case for case in dataset['training_cases']}
    result=[]
    for group in dataset['groups']:
        m=[by_id[key] for key in group['case_ids'][:32]]
        qa=[by_id[key] for key in group['case_ids'][32:]]
        result.append(dict(group_index=group['group_index'],M_max_length=max(len(case['input_ids']) for case in m),
            M_padded_tokens=32*max(len(case['input_ids']) for case in m),M_nonpadding_tokens=sum(len(case['input_ids']) for case in m),
            M_target_positions=sum(len(case['target_ids']) for case in m),QA_max_length=max(len(case['input_ids']) for case in qa),
            QA_input_tokens=sum(len(case['input_ids']) for case in qa),QA_target_positions=sum(len(case['target_ids']) for case in qa)))
    return result


def checked(record):
    path = b.Path(record['path']).resolve(strict=True)
    b.require(b.binding(path) == record, 'Changed new coverage artifact: '+str(path))
    return path


def groups(cases, old):
    result = []
    for index in range(128):
        suffix = old['groups'][index%32]
        batch = cases[index*32:(index+1)*32]
        ids = [case['case_id'] for case in batch]+suffix['case_ids'][32:]
        signatures = [case['input_signature'] for case in batch]+suffix['input_signatures'][32:]
        result.append(dict(suffix,group_index=index,case_ids=ids,input_signatures=signatures,
            weights={key:.5/(32 if position<32 else 8) for position,key in enumerate(ids)}))
    return result


def validate(dataset, old):
    verifier = legacy.module(original.HOT/'common.py','_coverage_actual_case_verifier')
    cases = dataset['training_cases']
    b.require(len(cases)==4224 and len({case['case_id'] for case in cases})==4224
        and [case['domain'] for case in cases]==['mmlu']*4096+['qasper']*32+['hotpotqa']*96
        and dataset['train_ids']==[case['case_id'] for case in cases]
        and dataset['groups']==groups(cases,old), 'Invalid real M4096/QA128 coverage/order.')
    # Reused prefix and QA retain their complete serialized source and tensor identities.
    verifier.require_same(cases[:1024],old['training_cases'][:1024])
    verifier.require_same(cases[4096:],old['training_cases'][1024:])
    for case in cases:
        verifier.verify_case(case)
        source = case['source']
        b.require(source['official_split']=='train','Only actual train/auxiliary_train sources.')
        if case['domain']=='mmlu':
            b.require(source['repository']=='cais/mmlu' and source['config']=='auxiliary_train'
                and source['revision']=='c30699e8356da336a370243923dbaf21066bb9fe'
                and source['source_split']=='train' and source['subject']==''
                and case['case_id']==f"cais_mmlu/auxiliary_train/train/{source['source_row_0based']}"
                and case['native_continuation_ids'].tolist()==[[362],[425],[356],[422]],'Changed actual native M source/support.')
    for index,row in enumerate(dataset['groups']):
        before = old['groups'][index%32]
        b.require(row['case_ids'][32:]==before['case_ids'][32:]
            and row['input_signatures'][32:]==before['input_signatures'][32:]
            and all(row['weights'][key]==before['weights'][key] for key in row['case_ids'][32:]),
            'QA order/input/weight changed at a real update.')
        if index<32:
            b.require(row==before,'The initial32 complete groups must match the control.')
    counts=Counter(key for row in dataset['groups'] for key in row['case_ids'])
    b.require(sum(counts.values())==5120 and all(counts[case['case_id']]==(1 if case['domain']=='mmlu' else 8)
        for case in cases),'Changed actual128 document exposure.')
    return dict(unique_M=4096,unique_QA=128,M_presentations=4096,QA_presentations=1024,
        first1024_complete_exact=True,QA_all128_groups_exact=True,only_train=True)


def load_dataset():
    record=b.read(b.DATA/'dataset_prepared.json')
    b.executor.check_pins(record['source_sha256'])
    checked(record['source_capture'])
    dataset=b.torch.load(checked(record['dataset']),map_location='cpu',weights_only=True)
    old,_recovery,_evidence=control.load()
    b.require(validate(dataset,old)==record['properties'],'Actual coverage data contract changed.')
    return dataset,record


def load():
    dataset,prepared=load_dataset()
    old,prior,old_evidence=control.load()
    manifest=b.read(b.DATA/'teacher.json')
    b.executor.check_pins(manifest['source_sha256'])
    payload=b.torch.load(checked(manifest['payload']),map_location='cpu',weights_only=True)
    b.require(manifest['status']=='complete_new_M3072_full_BF16_teacher'
        and manifest['teacher_fully_released'] and manifest['actual_new_cases']==3072
        and manifest['dataset']==prepared['dataset']
        and payload['dataset']==prepared['dataset']
        and payload['teacher_identity']==manifest['teacher_identity']
        and set(payload['targets'])==set(dataset['train_ids'][1024:4096]),'Changed actual teacher target coverage/identity.')
    recovery,records={},{}
    for case in dataset['training_cases']:
        key=case['case_id']
        if key in prior:
            target=prior[key]['target_log_probs']
            b.require(prior[key]['case']['input_signature']==case['input_signature'],'Control target input changed.')
        else:
            item=payload['targets'][key]
            b.require(item['input_signature']==case['input_signature'],'New teacher input signature changed.')
            target=legacy.aligned.reference.mixed.helper.target_log_distribution(item['target_log_probs'],case)
        b.require(target.dtype==b.torch.float32 and target.device.type=='cpu' and not target.requires_grad
            and b.torch.isfinite(target).all().item(),'Frozen FP32 target changed.')
        recovery[key]=dict(case=case,target_log_probs=target)
        records[key]=dict(input_signature=case['input_signature'],domain=case['domain'],actual_target=legacy.core.tensor_record(target))
    measured=exposure(dataset,recovery,range(128))
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    evidence=dict(schema='same_C0_M4096_static_TF_data_v1',dataset=prepared['dataset'],
        prepared=b.binding(b.DATA/'dataset_prepared.json'),teacher=b.binding(b.DATA/'teacher.json'),
        actual_input_target_records_sha256=digest,control=old_evidence,
        properties=prepared['properties'],full_exposure=measured,
        M1024_and_QA_targets_reused_exact=True,source_capture=prepared['source_capture'])
    contract_path=b.DATA/'data_contract.json'
    if contract_path.exists():
        b.require(evidence==b.read(contract_path),'Changed actual data/target/exposure binding.')
    return dataset,recovery,evidence

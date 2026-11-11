"""New native4 source cases; exact original static gold-prefix QA remains shared."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from . import bindings as b

original=b.prior.data
legacy=original.legacy
mmlu_log_distribution=original.mmlu_log_distribution
exposure=original.exposure
input_shapes=original.input_shapes
def checked(record):
    path=Path(record['path']).resolve(strict=True)
    b.require(b.binding(path)==record,'Changed WebMed artifact binding: '+str(path))
    return path


def helpers():
    directory=b.ROOT/'result/qwen3_8b_mmlu65_20261004'
    med=legacy.module(directory/'qwen_open_mcqa_coverage64_20261014/coverage_inputs.py','_S4_WebMed_native_Med')
    web=med.module(directory/'qwen_webinstruct_train_inventory_20261015/web_inputs.py','_S4_WebMed_native_Web')
    return med,web


def load_dataset():
    record=b.read(b.DATA/'dataset_prepared.json')
    for source in record['sources'].values():
        b.require(b.sha(source['path'])==source['sha256'],'Official train artifact changed.')
    dataset=b.torch.load(checked(record['dataset']),map_location='cpu',weights_only=True)
    b.require(len(dataset['training_cases'])==4224 and len(dataset['groups'])==128,'Fixed4096M/128QA required.')
    return dataset,record


def load():
    dataset,prepared=load_dataset()
    old,prior,old_evidence=original.load()
    med,web=helpers()
    qa_old={case['case_id']:case for case in old['training_cases'][4096:]}
    qa_new={case['case_id']:case for case in dataset['training_cases'][4096:]}
    med.require_same(qa_old,qa_new)
    manifest=b.read(b.DATA/'teacher.json')
    b.parent.s.check_pins(manifest['source_sha256'])
    payload=b.torch.load(checked(manifest['payload']),map_location='cpu',weights_only=True)
    b.require(manifest['teacher_fully_released'] and manifest['cases']==4096
        and manifest['dataset']==prepared['dataset'] and payload['dataset']==prepared['dataset']
        and payload['teacher_identity']==manifest['teacher_identity'] and manifest['mixture_applications']==0
        and set(payload['targets'])==set(dataset['train_ids'][:4096]),'New raw teacher coverage/identity changed.')
    recovery,records={},{}
    for case in dataset['training_cases']:
        key=case['case_id']
        if case['domain']=='mmlu':
            item=payload['targets'][key]
            b.require(item['input_signature']==case['input_signature'],'New raw teacher input changed.')
            pure=item['target_log_probs']
            target=legacy.aligned.reference.mixed.helper.target_log_distribution(pure,case)
            expected=.5*pure.exp()+.5*b.torch.nn.functional.one_hot(b.torch.tensor([case['gold_index']]),4)
            b.require(b.torch.allclose(target.exp(),expected,atol=1e-7,rtol=0),'Mix exactly once.')
        else:
            med.require_same(case,prior[key]['case']);target=prior[key]['target_log_probs']
        b.require(target.dtype==b.torch.float32 and target.device.type=='cpu' and not target.requires_grad
            and b.torch.isfinite(target).all(),'Frozen FP32 target required.')
        recovery[key]=dict(case=case,target_log_probs=target)
        records[key]=dict(input_signature=case['input_signature'],domain=case['domain'],actual_target=legacy.core.tensor_record(target))
    for index,group in enumerate(dataset['groups']):
        old_group=old['groups'][index]
        b.require(group['case_ids'][32:]==old_group['case_ids'][32:]
            and group['input_signatures'][32:]==old_group['input_signatures'][32:]
            and {k:group['weights'][k] for k in group['case_ids'][32:]}=={k:old_group['weights'][k] for k in old_group['case_ids'][32:]},
            'Original QA128 group order/input/weight changed.')
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    counts=Counter(key for group in dataset['groups'] for key in group['case_ids'])
    b.require(all(counts[case['case_id']]==(1 if case['domain']=='mmlu' else 8) for case in dataset['training_cases']),
        'Unique4096 and exact original QA exposure required.')
    evidence=dict(schema='S4_WebMed4096_static_TF_v1',dataset=prepared['dataset'],prepared=b.binding(b.DATA/'dataset_prepared.json'),
        teacher=b.binding(b.DATA/'teacher.json'),actual_input_target_records_sha256=digest,
        full_exposure=exposure(dataset,recovery,range(128)),QA_full_cases_targets_groups_exact=True,
        M_mixture_applications=1,old_data_reference=old_evidence['actual_input_target_records_sha256'])
    if (b.DATA/'data_contract.json').exists():b.require(evidence==b.read(b.DATA/'data_contract.json'),'Data contract changed.')
    return dataset,recovery,evidence

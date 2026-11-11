"""Rebuild one mixture from pinned raw teacher; preserve every input and QA target."""
import hashlib
import json
import math
from . import bindings as b
from experiments.s4_webmed_recovery_20261105 import data as original

exposure=original.exposure


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def base():
    dataset,targets,evidence=original.load()
    manifest=b.read(b.prior.DATA/'teacher.json')
    raw=b.torch.load(original.checked(manifest['payload']),map_location='cpu',weights_only=True)
    b.require(manifest['mixture_applications']==0 and raw['dataset']==evidence['dataset']
        and raw['teacher_identity']==manifest['teacher_identity'],'Use only original unmixed teacher probabilities.')
    return dataset,targets,evidence,raw


def mixed_target(teacher,case,alpha):
    b.require(alpha in (.25,.5,.75) and teacher.dtype==b.torch.float32 and teacher.shape==(1,4)
        and teacher.device.type=='cpu' and not teacher.requires_grad and b.torch.isfinite(teacher).all(),
        'Finite detached original FP32 four-option raw teacher required.')
    gold=original.legacy.aligned.reference.mixed.helper.check_mmlu_case(case)
    label=b.torch.full_like(teacher,-b.torch.inf);label[0,gold]=0.
    # Preserve original logaddexp math: no clamp, teacher re-softmax, or target renormalization.
    target=b.torch.logaddexp(teacher+math.log(alpha),label+math.log(1-alpha))
    expected=alpha*teacher.exp()+(1-alpha)*b.torch.nn.functional.one_hot(b.torch.tensor([gold]),4)
    b.require(target.dtype==b.torch.float32 and not target.requires_grad and b.torch.isfinite(target).all()
        and b.torch.allclose(target.exp().sum(-1),b.torch.ones(1),rtol=0,atol=2e-6)
        and b.torch.allclose(target.exp(),expected,rtol=0,atol=1e-7),
        'Once-mixed target differs from the original-support probability formula.')
    return target


def rebuild(dataset,old_targets,old_evidence,raw,alpha):
    targets={};records={};inputs={};qa={}
    for case in dataset['training_cases']:
        key=case['case_id'];previous=old_targets[key]
        b.require(previous['case'] is case,'Reuse the actual old case object without editing inputs.')
        if case['domain']=='mmlu':
            item=raw['targets'][key]
            b.require(item['input_signature']==case['input_signature'],'Raw teacher/input identity changed.')
            target=mixed_target(item['target_log_probs'],case,alpha)
        else:
            target=previous['target_log_probs']
            b.require(b.torch.equal(target,old_targets[key]['target_log_probs']),'QA target changed.')
            qa[key]=original.legacy.core.tensor_record(target)
        targets[key]=dict(case=case,target_log_probs=target)
        inputs[key]=dict(input_signature=case['input_signature'],domain=case['domain'])
        records[key]=dict(**inputs[key],actual_target=original.legacy.core.tensor_record(target))
    for group in dataset['groups']:
        b.require(len(group['case_ids'])==40
            and all(group['weights'][key]==(.5/32 if offset<32 else .5/8)
                for offset,key in enumerate(group['case_ids'])),'Keep original knowledge/QA domain and document weights.')
    source=b.binding(b.prior.DATA/'teacher.json');payload=b.binding(b.prior.DATA/'raw_teacher.pt')
    identity=dict(alpha=alpha,teacher_gold_mixture=[alpha,1-alpha],raw_teacher=payload,records=records)
    evidence=dict(schema='S4_QK_alpha_once_mixed_raw_teacher_v1',teacher_alpha=alpha,teacher_gold_mixture=[alpha,1-alpha],
        mixture_applications=1,raw_teacher_mixture_applications=0,raw_teacher=payload,teacher_manifest=source,
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),
        original_half_target_digest=old_evidence['actual_input_target_records_sha256'],
        actual_input_target_records_sha256=digest(identity),input_order_weights_sha256=digest(dict(cases=inputs,groups=dataset['groups'])),
        QA_target_records_sha256=digest(qa),QA_cases=128,QA_targets_inputs_groups_weights_exact=True,
        first64_exposure=exposure(dataset,targets,range(64)),
        target_math='logaddexp(raw_logp+log(alpha),onehot_logp+log(1-alpha)); no clamp or renormalization')
    return targets,evidence


def load(arm):
    dataset,previous,evidence,raw=base();alpha=b.ARMS[arm]['alpha']
    targets,actual=rebuild(dataset,previous,evidence,raw,alpha)
    path=b.RUN/arm/'data_contract.json'
    b.require(actual==b.read(path),'Actual per-alpha data/target contract differs from prepared CPU proof.')
    return dataset,targets,actual

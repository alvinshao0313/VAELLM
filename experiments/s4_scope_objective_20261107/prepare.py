"""CPU mathematical checks, real source inventory and frozen paired requests."""
import ast
from collections import Counter
import io
import math
import os
import time
from . import bindings as b,core,objective
from .protection import effective_scale,integer_coordinate
from litebsq.protected_channel_quant import decode_protected_channel_weight
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,rng_exact


def mathematics():
    t=b.torch;generator=t.Generator().manual_seed(137)
    logits=t.randn(32,4,generator=generator,requires_grad=True)
    raw=t.log_softmax(t.randn(32,4,generator=generator),-1)
    gold=(raw.argmax(-1)+1)%4
    mixed=(.5*raw.exp()+.5*t.nn.functional.one_hot(gold,4)).log()
    prediction=t.log_softmax(logits,-1)
    original,_=core.data.legacy.mmlu.weighted_loss(prediction,mixed)
    value,details=objective.knowledge(prediction,mixed,raw,True,1.)
    candidate=.5*value
    old_grad=t.autograd.grad(original,logits,retain_graph=True)[0]
    new_grad=t.autograd.grad(candidate,logits,retain_graph=True)[0]
    b.require(t.equal(original,candidate) and t.equal(old_grad,new_grad),'lambda1 must preserve original weighted loss and gradient.')
    reverse,_=objective.knowledge(prediction,mixed,raw,True,0.)
    expected=(prediction.exp()*(prediction-mixed)).sum(-1).mean()
    b.require(t.equal(reverse,expected),'Reverse KL direction/reduction incorrect.')
    zero,_=objective.knowledge(mixed,mixed,raw,True)
    b.require(abs(float(zero))<1e-7 and mixed.argmax(-1).equal(gold),'Same-target KL or wrong-teacher/gold mixture failed.')
    q=t.tensor([-129.,-128.,-127.5,-1.5,-.5,0.,.5,1.5,126.5,127.,128.],requires_grad=True)
    hard=integer_coordinate(q);scale=t.tensor([.0078125],dtype=t.bfloat16)
    payload=hard.detach().to(t.int8).view(t.uint8).reshape(1,-1)
    rows=(hard.reshape(1,-1)*scale.float().unsqueeze(1)).bfloat16()
    native=decode_protected_channel_weight(payload,scale,quant_format='int8',dtype=t.bfloat16,device=t.device('cpu'))
    grad=t.autograd.grad(rows.float().sum(),q)[0]
    b.require(t.equal(hard.detach(),t.tensor([-128.,-128.,-128.,-2.,0.,0.,0.,2.,126.,127.,127.]))
        and t.equal(rows,native) and t.equal(grad,t.tensor([0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.])*float(scale[0])),
        'Signed INT8 ties-to-even/clamp/STE/native parity failed.')
    delta=t.zeros(1,requires_grad=True);learned=effective_scale(scale,delta)
    b.require(t.equal(learned,scale) and t.autograd.grad(learned.float().sum(),delta)[0].item()>0,
        'Scale initialization/native BF16 gradient failed.')
    stream=io.BytesIO();t.save(rng(),stream);stream.seek(0)
    b.require(rng_exact(t.load(stream,weights_only=False)),'Actual Python/NumPy/CPU RNG serialization failed.')
    return dict(lambda1_loss_gradient_exact=True,lambda0_reverse_exact=True,same_target_zero=True,
        wrong_teacher_gold_covered=True,round_clamp_signed_native_STE_exact=True,scale_BF16_gradient=True,
        actual_CPU_Python_NumPy_RNG_serialization_exact=True)


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not b.RUN.exists(),'Fresh CPU-only preparation required.')
    started=time.monotonic()
    for p in b.HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    math_checks=mathematics();dataset,targets,raw,evidence=core.load_data()
    counts=Counter(k for group in dataset['groups'] for k in group['case_ids'])
    for source in ('webinstruct','medmcqa'):
        keys=[k for k in counts if k.startswith(source+'/')]
        b.require(len(keys)==2048 and all(counts[k]==1 for k in keys),'Source2048 once required.')
    qa=[case['case_id'] for case in dataset['training_cases'] if case['domain']!='mmlu']
    b.require(len(qa)==128 and all(counts[k]==8 for k in qa),'Original QA128 each8 required.')
    b.require(all(group['weights'][key]==(.5/32 if slot<32 else .5/8)
        for group in dataset['groups'] for slot,key in enumerate(group['case_ids'])),'Original loss weights changed.')
    reference=b.read(b.prior.RUN/'all/request.json');b.parent.s.check_pins(reference['source_sha256'])
    old_run=b.read(b.prior.RUN/'all/run.json');old_result=b.read(b.prior.RUN/'all/result.json')
    b.require(old_run['exit_code']==0 and old_result['paired_M']['current_correct']==10004,
        'Completed original all64 evidence required.')
    source=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    decoder=[name for name in source if '._parallel_stage_decoder.' in name]
    meta=b.read(b.SOURCE/'checkpoint_meta.json');modules=meta['converted_modules']
    b.require(len(decoder)==1512 and len(source)==3927 and len(modules)==252
        and all(m['residual_stages']==4 and m['protected_channel_quant_format']=='int8' for m in modules),'Actual nativeS4 inventory changed.')
    for m in modules:
        name=m['name'];q,s=source[name+'.protected_input_qvalues'],source[name+'.protected_input_scales']
        hard=(q.view(b.torch.int8).float()*s.float().unsqueeze(1)).bfloat16()
        native=decode_protected_channel_weight(q,s,quant_format='int8',dtype=b.torch.bfloat16,device=b.torch.device('cpu'))
        b.require(b.torch.equal(hard,native),'Actual source protection decoding differs: '+name)
    q_elements=sum(source[m['name']+'.protected_input_qvalues'].numel() for m in modules)
    scale_elements=sum(source[m['name']+'.protected_input_scales'].numel() for m in modules)
    b.require(q_elements==117964800 and scale_elements==23040,'Actual protection sizes changed.')
    shape_rows=b.read(core.data.b.DATA/'dataset_prepared.json')['shapes']
    b.require(max(r['M_max_length'] for r in shape_rows)==shape_rows[9]['M_max_length']==969
        and max(r['QA_max_length'] for r in shape_rows)==shape_rows[2]['QA_max_length']==10593,
        'Validation groups must cover actual full128 peaks.')
    coefficients=[]
    for group in dataset['groups']:
        t=b.torch.cat([raw[k]['target_log_probs'] for k in group['case_ids'][:32]])
        coefficients.append(float((-(t.exp()*t).sum(-1)/math.log(4)).mean()))
    parent=b.source();pins=b.pins();requests={}
    for arm,spec in b.ARMS.items():
        names=list(decoder)
        if spec['protection']:
            suffix='protected_log_scale_delta' if spec['protection']=='scale' else 'protected_code_coordinate'
            names.extend(m['name']+'.'+suffix for m in modules)
        b.require(len(names)==spec['leaves'],'Ordered CPU parameter inventory differs.')
        for validation in (False,True) if arm!='A' else (False,):
            tag='validation_'+arm if validation else arm
            request=dict(schema='S4_scope_objective_request_v1',arm=arm,recipe=spec,validation_only=validation,
                groups=([2,9] if arm in ('C','D') else [9]) if validation else list(range(128)),
                contract=b.CONTRACT,parent=parent,data_digest=evidence['actual_input_target_records_sha256'],
                data_contract=evidence,source_sha256=pins,parameter_names=names,
                fixture=reference['fixture'],heldout=reference['heldout'],
                parent_eight={name:b.binding(path) for name,path in b.PARENT_EIGHT.items()},
                reuse_all64=dict(run=b.binding(b.prior.RUN/'all/run.json'),result=b.binding(b.prior.RUN/'all/result.json')))
            b.check_request(request);requests[tag]=request
    b.require(not b.torch.cuda.is_initialized(),'CPU preparation initialized CUDA.')
    b.RUN.mkdir(parents=True)
    for tag,request in requests.items():b.write(b.RUN/tag/'request.json',request)
    b.write(b.RUN/'CPU_validation.json',dict(status='verified_four_recipes_shared_parent_inputs_targets_order',
        mathematics=math_checks,request_bindings={tag:b.binding(b.RUN/tag/'request.json') for tag in requests},
        raw_teacher_lambda=dict(min=min(coefficients),mean=math.fsum(coefficients)/128,max=max(coefficients)),
        source_q_elements=q_elements,source_scale_elements=scale_elements,decoder_leaves=1512,
        full_native_fields=3927,full128_peaks=dict(M_group=9,M_length=969,QA_group=2,QA_length=10593),
        same_inputs_targets_order=True,exposure64=core.data.exposure(dataset,targets,range(64)),
        exposure128=core.data.exposure(dataset,targets,range(128)),CUDA_initialized=False,seconds=time.monotonic()-started))
    print(dict(status='CPU_prepared',run=str(b.RUN)),flush=True)


if __name__=='__main__':main()

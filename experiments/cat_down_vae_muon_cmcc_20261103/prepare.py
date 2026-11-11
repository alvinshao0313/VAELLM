"""CPU checks of actual VAE ownership, BF16 update math and optimizer state."""
import ast
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from . import bindings as b
from .optimizer import VAEMuon,derive_layout,matrices,zeropower
from litebsq.llm_vae import MultiLayerVAE
from transformers import get_scheduler


def mathematical_checks(parsed,binding):
    args=b.stage_args(parsed)
    b.recovery.native.set_seed(31)
    model=MultiLayerVAE(args)
    named,layout,affine=derive_layout(model)
    flat={f'group0.parameter{i}':p.detach() for i,p in enumerate(model.parameters())}
    digest=b.c.digest(flat)
    b.require(digest==binding['original_down_fit']['actual_optimizer_initializations'][0]['initial_parameter_digest'],
        'The actual original CPU-created BF16 VAE initialization must match the control flat-order digest.')
    generator=b.torch.Generator().manual_seed(71431)
    independence=[]
    for name,spec in layout.items():
        g=b.torch.randn(spec['matrix_shape'],generator=generator)
        actual=zeropower(g)
        permutation=b.torch.randperm(36,generator=generator)
        b.require(b.torch.equal(zeropower(g[permutation]),actual[permutation]),'Per-linear NS must be permutation equivariant.')
        changed=g.clone()
        changed[7].add_(.375)
        modified=zeropower(changed)
        keep=b.torch.arange(36)!=7
        b.require(b.torch.equal(modified[keep],actual[keep]),'Perturbing one linear must not change another linear NS update.')
        for index in (0,35):
            b.require(b.torch.equal(zeropower(g[index:index+1]),actual[index:index+1]),
                'Batched NS must match the same independent actual matrix computation.')
        independence.append(dict(parameter=name,shape=spec['matrix_shape'],permutation_exact=True,perturbation_independent=True))
    evidence={}
    optimizer=VAEMuon(model,args,b.recovery.native.create_optimizer,evidence,diagnostic_steps=(1,2))
    scheduler=get_scheduler('linear',optimizer,num_warmup_steps=0,num_training_steps=10000)
    shadow=[p.detach().clone().requires_grad_() for p in model.parameters()]
    native=b.recovery.native.create_optimizer(shadow,args,args.lr)
    auxiliary_indices=[i for i,(name,p) in enumerate(named.items()) if name in affine]
    for step in range(2):
        for p,q in zip(model.parameters(),shadow,strict=True):
            g=b.torch.randn(p.shape,generator=generator).to(b.torch.bfloat16)
            p.grad=g.clone();q.grad=g.clone()
        for group in native.param_groups:group['lr']=.003*(1-step/10000)
        optimizer.step();native.step()
        b.require(all(b.torch.equal(list(model.parameters())[i],shadow[i]) for i in auxiliary_indices),
            'Auxiliary BF16 AdamW parameters must match the original native optimizer exactly.')
        scheduler.step()
    saved=io.BytesIO()
    b.torch.save(optimizer.state_dict(),saved)
    saved.seek(0)
    state=b.torch.load(saved,map_location='cpu',weights_only=True)
    restored_model=MultiLayerVAE(args)
    restored_model.load_state_dict(model.state_dict(),strict=True)
    restored=VAEMuon(restored_model,args,b.recovery.native.create_optimizer,{},diagnostic_steps=(1,2))
    restored.load_state_dict(state)
    original_parameters=list(optimizer.param_groups[0]['params'])+list(optimizer.param_groups[1]['params'])
    loaded_parameters=list(restored.param_groups[0]['params'])+list(restored.param_groups[1]['params'])
    fields=0
    for p,q in zip(original_parameters,loaded_parameters,strict=True):
        for key,value in optimizer.state[p].items():
            actual=restored.state[q][key]
            if isinstance(value,b.torch.Tensor):
                b.require(value.dtype==actual.dtype and b.torch.equal(value,actual),'Exact optimizer state dtype/value roundtrip failed: '+key)
            else:b.require(value==actual,'Scalar optimizer state roundtrip failed.')
            fields+=1
    b.require(optimizer.param_groups[0]['kappa']==restored.param_groups[0]['kappa']
        and optimizer.actual_step==restored.actual_step==2 and restored._adam.state is restored.state,
        'Restore stage kappa, update count and the shared auxiliary state ownership.')
    b.require(all(restored.state[p]['momentum_buffer'].dtype==b.torch.float32 for p in restored.param_groups[0]['params']),
        'The generic loader must not round FP32 Muon momentum through BF16.')
    restored.diagnostic_steps.add(3)
    for group in restored.param_groups:group['lr']=1e-30
    for p,q in zip(model.parameters(),restored_model.parameters(),strict=True):q.grad=p.grad.detach().clone()
    restored.step()
    zero_rounding=restored.evidence['updates'][-1]
    b.require(zero_rounding['step']==3 and all(row['changed_elements']==0 for row in zero_rounding['roles'].values()),
        'A genuine tiny-LR BF16 update may round to zero after the first two updates; record it without failing.')
    return dict(actual_original_flat_order_initial_digest=digest,parameter_elements=767808,matrix_elements=737280,
        auxiliary_elements=30528,actual_owner_matrix_layout=layout,auxiliary_names=affine,
        independence=independence,scheduler='Native Transformers linear10000 accepted this standard Optimizer.',
        CPU_fixture_only=True,CPU_kappa_never_used_for_GPU=True,CPU_calibration=evidence['calibration'],
        actual_CPU_updates=evidence['updates'],auxiliary_native_AdamW_two_steps_exact=True,
        optimizer_state_roundtrip=dict(fields=fields,all_dtypes_and_values_exact=True,Muon_momentum_FP32_exact=True,
            auxiliary_moments_BF16_exact=True,kappa_and_step_exact=True,old_Muon_schema_supported=False),
        zero_BF16_rounding_boundary_after_first_two_updates=zero_rounding)


def prepare(*,after_short_review=False):
    b.cpu_environment()
    b.require(not (b.RESULT / 'prepared.json').exists(),'Do not replace prepared source evidence.')
    started=time.time()
    from . import fit,run,evaluate,sequence
    binding,reused=b.sources()
    parsed,identity=b.recipe(b.OUTPUT / 'not_executed')
    math_checks=mathematical_checks(parsed,binding)
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    b.require(not b.torch.cuda.is_initialized(),'CPU validation must not initialize CUDA.')
    source_pins=b.pins()
    short_source=source_pins
    review=None
    if after_short_review:
        review=b.read(b.RESULT / 'tail_audit_review.json')
        old=b.read(b.RESULT / 'cpu_before_tail_review.json')
        proof=b.read(b.OUTPUT / 'short/report.json')
        b.require(proof['status']=='completed_VAE_Muon_short' and proof['exit_code']==0
            and proof['source_sha256']==old['source_sha256']
            and review['before_CPU_sha256']==b.sha(b.RESULT / 'cpu_before_tail_review.json')
            and review['short_report_sha256']==b.sha(b.OUTPUT / 'short/report.json'),
            'Preserve the real earlier CPU/GPU evidence and its original source binding.')
        changed={name for name in source_pins if source_pins[name]!=old['source_sha256'][name]}
        b.require(changed==set(review['files']) and set(source_pins)==set(old['source_sha256'])
            and all(source_pins[name]==row['after_sha256'] and old['source_sha256'][name]==row['before_sha256']
                for name,row in review['files'].items()),'Only the explicitly reviewed diagnostic/proof-binding source changes may reuse the earlier GPU short.')
        short_source=old['source_sha256']
    short_digest=hashlib.sha256(json.dumps(short_source,sort_keys=True).encode()).hexdigest()
    b.write(b.RESULT / 'prepared.json',dict(status='cpu_verified',exit_code=0,command=[sys.executable,*sys.argv],
        source_binding=binding,reused_source_evidence=reused,contract=b.CONTRACT,runtime_identity=identity,
        CPU_validation=math_checks,CUDA_initialized=False,elapsed_seconds=time.time()-started,source_sha256=source_pins,
        applicable_GPU_short_source_digest=short_digest,short_to_formal_source_review=review))
    print(dict(status='cpu_verified',seconds=time.time()-started,output=str(b.RESULT / 'prepared.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-short-review',action='store_true')
    args=parser.parse_args()
    b.torch.set_num_threads(8)
    prepare(after_short_review=args.after_short_review)

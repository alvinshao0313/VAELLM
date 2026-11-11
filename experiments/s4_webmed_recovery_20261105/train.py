"""Fixed128 original joint objective with Web2048 and Med2048 replacing auxiliary M4096."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np

from . import bindings as b,diagnostics,state
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime,training
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact,live_stamp


def losses(model,dataset,targets,index,include_QA):
    if include_QA:return training.losses(model,dataset,targets,index,True)
    ids=dataset['groups'][index]['case_ids'][:32];cases=[targets[k]['case'] for k in ids]
    target=b.torch.cat([targets[k]['target_log_probs'] for k in ids])
    loss,_=b.data.legacy.mmlu.weighted_loss(b.data.mmlu_log_distribution(model,cases),target)
    b.require(b.torch.isfinite(loss).item(),'Nonfinite unchanged M32/.5 objective.')
    value=float(loss.detach());loss.backward()
    return dict(mmlu=value),ids


def exposure(dataset,targets,indices,include_QA):
    if include_QA:return b.data.exposure(dataset,targets,indices)
    cases=[targets[k]['case'] for index in indices for k in dataset['groups'][index]['case_ids'][:32]]
    return dict(group_indices=list(indices),presentations=dict(mmlu=len(cases),qasper=0,hotpotqa=0),
        input_tokens=dict(mmlu=sum(len(c['input_ids']) for c in cases),qasper=0,hotpotqa=0),
        answer_positions=dict(mmlu=len(cases),qasper=0,hotpotqa=0),
        total_input_tokens=sum(len(c['input_ids']) for c in cases),total_answer_positions=len(cases),
        M_batch_size=32,QA_forward_backward_documents=0)


def call(request_path,checkpoint,output,component,short=False):
    request=b.read(request_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_webmed_recovery_20261105.evaluate',
        '--request',str(request_path),'--checkpoint',str(checkpoint),'--output',str(output),'--component',component]
    if short:argv.append('--short')
    with (output/(component+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(request['gpu']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        b.write(output/(component+'_launch.json'),dict(command=argv,**b.parent.process_event(process.pid),
            paused_parent=b.parent.process_event(os.getpid()),paused_parent_memory=b.memory()))
        code=process.wait()
    b.require(code==0,'Readout failed; stop affected chain without retry: '+component)
    return b.read(output/(component+'.json'))


def execute(path):
    request=b.read(path);b.parent.s.check_pins(request['source_sha256']);b.require(request['parent']==b.source(),'Parent binding changed.')
    resources=b.resources(request['gpu']);root=Path(request['output']);short=request['validation_only'];arm=b.ARMS[request['arm']]
    began=time.monotonic();code=1;report=dict(status='running_same_parent_local_recovery',**b.parent.process_event(os.getpid()),
        source_sha256=request['source_sha256'],parent=request['parent'],arm=request['arm'],validation_only=short,resources=resources,
        contract=b.CONTRACT,round_source=str(b.SOURCE),updates=[])
    b.write(root/'training.json',report)
    try:
        dataset,targets,evidence=b.data.load();b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Static targets changed.')
        heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
        b.require(b.binding(request['heldout']['path'])==request['heldout'],'Diagnostic cases changed.')
        model,meta,_budget=state.load(b.SOURCE)
        random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
        model.to('cuda:0');b.restore_rope_alias(model)
        before=runtime.records(model.state_dict());active={};runtime.install(model,{},7,'terminal',active)
        frozen=b.old.old_life.frozen_proof(model,set(active))
        optimizer=b.torch.optim.AdamW(list(active.values()),lr=arm['base_lr'],betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
        b.require(len(active)==1512,'All1512 original decoder leaves must train.')
        b.require(not optimizer.state,'The same-parent run must start with fresh Adam state.')
        report['optimizer_initial_state_entries']=0
        baseline=None
        if not short:
            baseline=diagnostics.evaluate(model,dataset,targets,heldout)
            b.write(root/'diagnostic_000000.json',baseline)
        indices=request['short_indices'] if short else list(range(128))
        rows=[]
        for step,index in enumerate(indices,1):
            optimizer.zero_grad(set_to_none=True);b.torch.cuda.synchronize();b.torch.cuda.reset_peak_memory_stats(0);tick=time.monotonic()
            previous={name:p.detach().clone() for name,p in active.items()} if step<=2 else None
            values,case_ids=losses(model,dataset,targets,index,arm['include_QA'])
            gradients=training.gradient_contract(model,active)
            norm=float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
            b.require(math.isfinite(norm) and norm>0,'Finite nonzero joint gradient required.')
            lr=arm['base_lr']*(1-(step-1)/500);optimizer.param_groups[0]['lr']=lr;optimizer.step()
            for name,p in active.items():
                item=optimizer.state[p]
                b.require(int(item['step'])==step and item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
                    and b.torch.isfinite(p).all() and b.torch.isfinite(item['exp_avg']).all()
                    and b.torch.isfinite(item['exp_avg_sq']).all(),'Actual FP32 master/Adam value or clock failed: '+name)
            b.torch.cuda.synchronize()
            displacement=None
            if previous is not None:
                displacement=dict(FP32_changed=sum(int((p.detach()!=previous[name]).sum()) for name,p in active.items()),
                    BF16_changed=sum(int((p.detach().bfloat16()!=previous[name].bfloat16()).sum()) for name,p in active.items()))
                b.require(displacement['FP32_changed']>0,'The optimizer did not change actual FP32 decoder masters.')
                del previous
            row=dict(step=step,group_index=index,lr=lr,losses=values,case_ids=case_ids,gradients=gradients,
                gradient_norm_before_clip=norm,weighted_objective=sum(values.values()),seconds=time.monotonic()-tick,memory=b.memory(),
                training_exposure=exposure(dataset,targets,[index],arm['include_QA']),decoder_displacement=displacement)
            rows.append(row);b.write(root/'updates'/f'step_{step:06d}.json',row)
            print(json.dumps(dict(event='actual_recovery_update',arm=request['arm'],**row)),flush=True)
            if not short and step==64:
                optimizer.zero_grad(set_to_none=True)
                b.write(root/'diagnostic_000064.json',diagnostics.evaluate(model,dataset,targets,heldout,baseline))
                snapshot=state.save(model,active,optimizer,request,step,root/f'step_{step:06d}')
                stamp,random_state=live_stamp(active,optimizer),rng()
                observed=call(path,snapshot['checkpoint']['path'],root/f'step_{step:06d}/evaluation','mmlu',short)
                restore_rng(random_state)
                b.require(stamp==live_stamp(active,optimizer) and rng_exact(random_state),'Reader changed paused optimizer/masters/RNG.')
                b.base.cleanup_model(snapshot['checkpoint']['path'],'Consumed midpoint; fixed128 live state continues without quality selection.')
        optimizer.zero_grad(set_to_none=True)
        b.require(b.old.old_life.frozen_proof(model,set(active))==frozen,'A frozen code/protection/rotation/backbone field changed.')
        b.require(any(before[name]!=runtime.records({name:p.detach().to('cpu',dtype=b.torch.bfloat16)})[name] for name,p in active.items()),
            'No decoder change survived BF16 deployment.')
        if short:
            report.update(status='verified_incremental_WebMed_shape_updates',actual_updates=len(indices),group_indices=indices,
                actual_training_exposure=exposure(dataset,targets,indices,arm['include_QA']),full_frozen_complement_exact=True,
                BF16_decoder_change=True,active_decoder_leaves=len(active),reused_validation=request['reused_validation'],
                first_lr=rows[0]['lr'],last_lr=rows[-1]['lr'],peak_allocated=max(x['memory']['peak_allocated'] for x in rows),
                mean_update_seconds=sum(x['seconds'] for x in rows)/len(rows))
            code=0
            return
        b.write(root/'diagnostic_000128.json',diagnostics.evaluate(model,dataset,targets,heldout,baseline))
        final=state.save(model,active,optimizer,request,len(indices),root/'post')
        del optimizer,model;gc.collect();b.torch.cuda.empty_cache()
        fresh=call(path,final['checkpoint']['path'],root/'post/evaluation','fresh')
        b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Actual final fresh must pass.')
        report.update(status='complete_same_parent_WebMed_recovery',actual_updates=len(indices),group_indices=indices,
            actual_training_exposure=exposure(dataset,targets,indices,arm['include_QA']),full_frozen_complement_exact=True,
            final=final,fresh=fresh,first_lr=rows[0]['lr'],last_lr=rows[-1]['lr'],peak_allocated=max(x['memory']['peak_allocated'] for x in rows),
            mean_update_seconds=sum(x['seconds'] for x in rows)/len(rows),midpoint_readout_followed_by_real_update=True)
        code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-began);b.write(root/'training.json',report)
        (root/'training.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    execute(parser.parse_args().request)

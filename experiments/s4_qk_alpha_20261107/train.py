"""Fresh AdamW64 on the common down/pre; only the prepared knowledge mixture changes."""
import argparse
from pathlib import Path
import os
import random
import time
import traceback
import numpy as np
from . import bindings as b,core,state
from experiments.s4_scope_refine_20261106.train import update


def make_optimizer(active):
    optimizer=b.torch.optim.AdamW(list(active.values()),lr=5e-7,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    b.require(not optimizer.state,'Fresh Adam must have no inherited state.')
    return optimizer


def build(request):
    model,meta,budget=state.load(b.SOURCE)
    random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
    model.to('cuda:0');b.restore_rope_alias(model)
    active,warm=core.install(model,request['arm'])
    b.require(set(active)==set(request['exact_scope_parameter_names']),'Actual QK selection differs from original CPU-pinned names.')
    return model,active,make_optimizer(active),warm,budget


def execute(request_path,output):
    request=b.read(request_path);b.initialize();b.parent.s.check_pins(request['source_sha256'])
    arm=request['arm']
    b.require(request['parent']==b.source() and request['contract']==b.contract(arm)
        and request['learning_contract']==b.learning_contract(arm) and request['teacher_alpha']==b.ARMS[arm]['alpha']
        and request['groups']==list(range(128)) and request['stop_step']==64 and not request['validation_only'],
        'Fresh64 alpha request source/contract changed.')
    resource=b.resources(request['gpu']);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();code=1
    report=dict(status='running_fresh_QK_alpha64',**b.parent.process_event(os.getpid()),request=b.binding(request_path),
        resources=resource,teacher_alpha=request['teacher_alpha'],fresh_Adam=True)
    b.write(output/'training.json',report)
    try:
        dataset,targets,evidence=core.data.load(arm)
        b.require(evidence['actual_input_target_records_sha256']==request['data_digest']
            and b.binding(b.RUN/arm/'data_contract.json')==request['data_contract'],'Per-alpha static targets changed.')
        model,active,optimizer,warm,budget=build(request);frozen=b.old.old_life.frozen_proof(model,set(active))
        heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
        b.write(output/'diagnostic_000000.json',core.panel(model,dataset,targets,heldout,arm))
        rows=[]
        for offset in range(64):
            row=update(model,active,optimizer,dataset,targets,request['groups'][offset],offset+1,offset==0)
            row.update(teacher_alpha=request['teacher_alpha'],teacher_gold_mixture=request['contract']['teacher_gold_mixture'],
                data_digest=request['data_digest'],mixture_applications=1)
            b.write(output/'updates'/f'step_{offset+1:06d}.json',row);rows.append(row)
            print({k:row[k] for k in ('step','group_index','teacher_alpha','lr','weighted_objective','seconds','gradients','layer_strata_displacement')},flush=True)
        optimizer.zero_grad(set_to_none=True)
        b.require(b.old.old_life.frozen_proof(model,set(active))==frozen,'Training changed the complete frozen complement.')
        b.write(output/'diagnostic_000064.json',core.panel(model,dataset,targets,heldout,arm))
        saved=state.save(model,active,optimizer,request,64,output/'training_state.pt')
        report.update(status='complete_actual_named_QK_alpha64_training_state',completed_updates=64,executed_updates=64,
            group_indices=list(range(64)),training_state=saved,full_frozen_complement_exact=True,active_leaves=len(active),
            active_projections=72,warmup=warm,bit_budget=budget,mean_update_seconds=sum(row['seconds'] for row in rows)/64,
            data_contract=request['data_contract'],next_group_index=64)
        code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-started)
        b.write(output/'training.json',report);(output/'training.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();execute(args.request,args.output)

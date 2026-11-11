"""Restore the real64 state and execute only original groups64..127."""
import argparse
import os
from pathlib import Path
import time
import traceback
from . import bindings as b,core,state
from experiments.s4_recovery_screen_20261105.train import build,update


def execute(request_path,output):
    request=b.read(request_path);b.initialize();b.check_request(request)
    resource=b.resources('7');output=Path(output);output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();code=1
    report=dict(status='running_attention64_to128',**b.parent.process_event(os.getpid()),
        request=b.binding(request_path),resources=resource,processes={})
    b.write(output/'training.json',report)
    try:
        dataset,targets,evidence=core.load_data()
        b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Data/targets changed.')
        model,active,optimizer,warm,budget=build(request)
        frozen=b.old.old_life.frozen_proof(model,set(active))
        saved=state.restore(b.PREDECESSOR_STATE,model,active,optimizer,request)
        cursor=saved['next_cursor'];b.require(cursor==saved['completed_updates']==64,'Start at the original64 boundary.')
        b.write(output/'resume_verified.json',dict(status='cross_process_exact_masters_Adam_RNG_cursor',
            state=b.binding(b.PREDECESSOR_STATE),original_identity=saved['identity'],continuation=request['continuation'],
            saved_updates=cursor,next_group_index=saved['next_group_index'],parameter_names_exact=True,
            masters_exact=True,complete_Adam_exact=True,RNG_exact=True,frozen_complement_exact=True))
        rows=[]
        for offset in range(cursor,128):
            row=update(model,active,optimizer,dataset,targets,request['groups'][offset],offset+1,offset in (64,65))
            b.write(output/'updates'/f'step_{offset+1:06d}.json',row);rows.append(row)
            print({key:row[key] for key in ('step','group_index','lr','weighted_objective','seconds','gradients','layer_strata_displacement')},flush=True)
        optimizer.zero_grad(set_to_none=True)
        b.require(b.old.old_life.frozen_proof(model,set(active))==frozen,'Frozen complement changed.')
        heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
        b.write(output/'diagnostic_000128.json',core.panel(model,dataset,targets,heldout,128))
        training_state=state.save(model,active,optimizer,request,128,output/'training_state.pt')
        report.update(status='complete_actual_named_attention128_training_state',completed_updates=128,executed_updates=len(rows),
            group_indices=request['groups'][cursor:128],training_state=training_state,full_frozen_complement_exact=True,
            active_leaves=len(active),active_projections=144,warmup=warm,bit_budget=budget,
            resume=b.binding(b.PREDECESSOR_STATE),mean_update_seconds=sum(row['seconds'] for row in rows)/len(rows))
        code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-began)
        b.write(output/'training.json',report);(output/'training.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();execute(args.request,args.output)

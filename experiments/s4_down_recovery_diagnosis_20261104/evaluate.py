"""Strict same-parent native consumers; original full task mathematics."""
import argparse
import gc
from pathlib import Path
import traceback
from . import bindings as b,state,diagnostics
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime

_core=b.prior.bound_implementation('evaluate')
_core.b=b
_core.load=state.load


def fresh(checkpoint,request):
    model,meta,budget=state.load(checkpoint);expected=b.read(Path(checkpoint).parent/'expected_state.json')
    model.to('cuda:0');b.restore_rope_alias(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    fixture=b.torch.load(request['fixture']['path'],map_location='cpu',weights_only=True)
    b.require(b.binding(request['fixture']['path'])==request['fixture'],'Probe fixture changed.')
    logits=b.old.run_short.probe(model,fixture)
    b.require(runtime.records(model.state_dict())==expected,'Single-card fresh consumer changed persistent fields.')
    return dict(checkpoint_id=meta['checkpoint_id'],full_state_exact=True,persistent_fields=len(expected),
        BF16_forward_finite=True,output=b.old.old_life.tensor_record(logits),warmup=warm,bit_budget=budget,placement='single_GPU')


def read_diagnostics(checkpoint,request):
    dataset,targets,evidence=b.data.load()
    b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Diagnostic targets changed.')
    b.require(b.binding(request['heldout']['path'])==request['heldout'],'Held-out diagnostic source changed.')
    heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
    model,meta,_budget=state.load(checkpoint);model.to('cuda:0');b.restore_rope_alias(model)
    b.old.install_precision_runtime(model,b.torch.bfloat16);runtime.warm_frozen(model)
    return dict(checkpoint_id=meta['checkpoint_id'],parent=request['parent'],
        diagnostics=diagnostics.evaluate(model,dataset,targets,heldout),scope='Read-only endpoint diagnostic, not training exposure')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=['fresh','mmlu','seven','predict','diagnostics'],required=True)
    parser.add_argument('--short',action='store_true');parser.add_argument('--config',type=Path);parser.add_argument('--datasets',nargs='+')
    args=parser.parse_args();b.initialize();request=b.read(args.request);b.parent.s.check_pins(request['source_sha256'])
    b.require(b.torch.cuda.device_count()==1,'Consumers use exactly one assigned GPU.')
    args.output.mkdir(parents=True,exist_ok=True);code=1
    try:
        if args.component=='fresh':result=fresh(args.checkpoint,request)
        elif args.component=='diagnostics':result=read_diagnostics(args.checkpoint,request)
        elif args.component=='predict':result=_core.predict(args.config,args.output,args.datasets)
        else:result=_core.downstream(args.checkpoint,args.output,args.component,args.short)
        b.write(args.output/(args.component+'.json'),result);code=0
        print(dict(component=args.component,checkpoint_id=result.get('checkpoint_id'),status='complete'),flush=True)
    except BaseException:
        b.write(args.output/(args.component+'_failure.json'),dict(error=traceback.format_exc()));raise
    finally:(args.output/(args.component+'.exit')).write_text(str(code)+'\n')


if __name__=='__main__':main()

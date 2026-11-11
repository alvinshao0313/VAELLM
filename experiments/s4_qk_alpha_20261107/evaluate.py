"""Fresh strict native consumer and the original fixed64/32 complete eight."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time
from . import bindings as b,core,state


def fresh(checkpoint,request):
    model,meta,budget=state.load(checkpoint);expected=core.runtime.records(model.state_dict())
    model.to('cuda:0');b.restore_rope_alias(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    warm=core.runtime.warm_frozen(model)
    b.require(b.binding(request['fixture']['path'])==request['fixture'],'Pinned fixture changed.')
    fixture=b.torch.load(request['fixture']['path'],map_location='cpu',weights_only=True)
    logits=b.old.run_short.probe(model,fixture)
    b.require(core.runtime.records(model.state_dict())==expected,'Fresh forward changed persistent fields.')
    return dict(checkpoint_id=meta['checkpoint_id'],full_state_exact=True,BF16_forward_finite=True,
        output=b.old.old_life.tensor_record(logits),persistent_fields=len(expected),warmup=warm,bit_budget=budget)


def downstream(checkpoint,output,shard):
    model,meta,budget=state.load(checkpoint)
    model.to('cuda:0');b.restore_rope_alias(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    b.torch.cuda.reset_peak_memory_stats(0)
    warm=core.runtime.warm_frozen(model)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(b.HF),use_fast=True,trust_remote_code=True,local_files_only=True)
    api=b.parent.module(b.parent.SAMPLE/'eval_utils_with_samples.py','_S4_scope64_original_hflm')
    args=SimpleNamespace(tasks='mmlu' if shard=='mmlu' else 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa',
        num_fewshot=0,batch_size='64' if shard=='mmlu' else '32',lm_limit=None,model_path=str(b.HF),
        eval_log_dir=None,eval_run_ts=None,mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output/(shard+'_samples')),mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    model.config.use_cache=False;shapes=Counter()
    def inputs(module,positional,keywords):
        ids=keywords.get('input_ids',positional[0] if positional else None)
        if b.torch.is_tensor(ids):shapes[tuple(ids.shape)]+=1
    handle=model.register_forward_pre_hook(inputs,with_kwargs=True)
    try:result=api.run_lm_eval(model,tokenizer,args)
    finally:handle.remove()
    from experiments.w25_eval.acceptance import leaf_metrics,_group_agrees
    metrics,_,batch=leaf_metrics(result['raw_results'],result['n_samples'],result['n_shot'],result['configs'])
    _group_agrees(metrics,result['raw_results'],result['group_results'])
    reference=b.read(b.parent.BASELINE/(shard+'.json'))
    b.require(result['configs']==reference['configs'] and str(batch)==str(result['batch_size'])
        and all(result['versions'][key]==reference['versions'][key] for key in result['n_samples']),
        'Full task configurations, versions, or batch differ from BF16.')
    return dict(checkpoint_id=meta['checkpoint_id'],checkpoint_dir=str(checkpoint),strict_load=True,
        checkpoint_loader='v6',base_model_path=str(b.HF),adapter_dir=None,eval_hif4_act=False,
        tokenizer_class=type(tokenizer).__name__,tokenizer_is_fast=tokenizer.is_fast,validation_only=False,
        bit_budget=budget,warmup=warm,evals=dict(lm_eval=result),verified_task_metrics=metrics,
        actual_forward_shapes=[dict(shape=list(shape),calls=count) for shape,count in sorted(shapes.items())],
        actual_max_batch=max(shape[0] for shape in shapes),actual_max_sequence=max(shape[1] for shape in shapes),
        reader_peak_allocated=b.torch.cuda.max_memory_allocated(0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=['fresh','mmlu','seven'],required=True)
    args=parser.parse_args();b.initialize();request=b.read(args.request);b.parent.s.check_pins(request['source_sha256'])
    b.require(b.torch.cuda.device_count()==1,'One assigned GPU consumer required.')
    args.output.mkdir(parents=True,exist_ok=True)
    result=fresh(args.checkpoint,request) if args.component=='fresh' else downstream(args.checkpoint,args.output,args.component)
    b.write(args.output/(args.component+'.json'),result)
    (args.output/(args.component+'.exit')).write_text('0\n')
    print(dict(component=args.component,checkpoint_id=result['checkpoint_id'],status='complete'),flush=True)

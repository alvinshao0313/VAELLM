"""Strict new-run native consumers and original task protocols."""
import argparse
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from collections import Counter

from . import bindings as b
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict

SEVEN='boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'


def load(directory):
    directory=Path(directory).resolve(strict=True)
    b.require(directory.is_relative_to(b.OUTPUT),'Only this algorithm new-run checkpoints are consumers.')
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Native strict consumer load failed.')
    provenance=meta['extra_meta']['distill_stage']
    b.require(provenance is not None and provenance['algorithm']=='fresh_BF16_S4_prefix_recovery'
        and provenance['origin']=='original_HF_BF16','Reject old compressed parents and no-recovery provenance.')
    expected_path=directory.parent/'expected_state.json'
    if expected_path.exists():b.require(runtime.records(model.state_dict())==b.read(expected_path),
        'Actual consumer full BF16 state differs from its actual new-run export.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'],'Native storage contract failed.')
    if len(meta['converted_modules'])==252:
        b.require(abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'FullS4 complete projection ledger changed.')
    model.eval().requires_grad_(False)
    return model,meta,budget


def fresh(checkpoint,output,request):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='6,7','Fresh uses the real18/18 map on GPU6/7.')
    model,meta,budget=load(checkpoint)
    expected=b.read(Path(checkpoint).parent/'expected_state.json')
    b.require(runtime.records(model.state_dict())==expected,'Saved native state differs from the actual export.')
    handles,mapping=b.old.old_life.apply_map(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    fixture=b.torch.load(Path(request['output'])/'fixture.pt',map_location='cpu',weights_only=True)
    logits=b.old.run_short.probe(model,fixture)
    b.require(runtime.records(model.state_dict())==expected,'Fresh map/prewarm/forward changed persistent state.')
    for handle in handles:handle.remove()
    return dict(status='verified_fresh_BF16_S4_prefix',checkpoint_id=meta['checkpoint_id'],
        full_state_exact=True,persistent_fields=len(expected),BF16_forward_finite=True,
        output=b.old.old_life.tensor_record(logits),warmup=warm,layer_device_map=mapping,bit_budget=budget)


def downstream(checkpoint,output,shard,short):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('6','7') and b.torch.cuda.device_count()==1,
        'Use only one assigned physical GPU6 or7 for a downstream reader.')
    model,meta,budget=load(checkpoint)
    b.torch.cuda.reset_peak_memory_stats(0)
    model.to('cuda');b.old.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(b.HF),use_fast=True,trust_remote_code=True,local_files_only=True)
    api=b.parent.module(b.parent.SAMPLE/'eval_utils_with_samples.py','_fresh_S4_original_hflm')
    args=SimpleNamespace(tasks='mmlu' if shard=='mmlu' else SEVEN,num_fewshot=0,
        batch_size='64' if shard=='mmlu' else '32',lm_limit=2 if short else None,model_path=str(b.HF),
        eval_log_dir=None,eval_run_ts=None,mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output/(shard+'_samples')),mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    model.config.use_cache=False
    shapes=Counter()
    def inputs(module,positional,keywords):
        ids=keywords.get('input_ids',positional[0] if positional else None)
        if b.torch.is_tensor(ids):shapes[tuple(ids.shape)]+=1
    handle=model.register_forward_pre_hook(inputs,with_kwargs=True)
    try:result=api.run_lm_eval(model,tokenizer,args)
    finally:handle.remove()
    metrics=None
    if not short:
        from experiments.w25_eval.acceptance import leaf_metrics,_group_agrees
        metrics,_,batch=leaf_metrics(result['raw_results'],result['n_samples'],result['n_shot'],result['configs'])
        _group_agrees(metrics,result['raw_results'],result['group_results'])
        reference=b.read(b.parent.BASELINE/(shard+'.json'))
        b.require(result['configs']==reference['configs'] and str(batch)==str(result['batch_size'])
            and all(result['versions'][key]==reference['versions'][key] for key in result['n_samples']),
            'Full task configurations/versions/batch differ from the BF16 baseline.')
    return dict(checkpoint_id=meta['checkpoint_id'],checkpoint_dir=str(checkpoint),strict_load=True,
        checkpoint_loader='v6',base_model_path=str(b.HF),adapter_dir=None,eval_hif4_act=False,
        tokenizer_class=type(tokenizer).__name__,tokenizer_is_fast=tokenizer.is_fast,
        validation_only=short,bit_budget=budget,warmup=warm,evals=dict(lm_eval=result),verified_task_metrics=metrics,
        actual_forward_shapes=[dict(shape=list(shape),calls=count) for shape,count in sorted(shapes.items())],
        actual_max_batch=max(shape[0] for shape in shapes),actual_max_sequence=max(shape[1] for shape in shapes),
        reader_peak_allocated=b.torch.cuda.max_memory_allocated(0),reader_peak_reserved=b.torch.cuda.max_memory_reserved(0),
        reader_memory_scope='Own allocator includes original prewarm and actual HFLM forward batches; parent residency logged separately.')


def predict(config_path,output,datasets):
    from experiments.longbench_v1 import runner
    from experiments.longbench_v1.protocol import model_policy
    def loader(config,logger):
        model,meta,budget=load(config['model']['checkpoint'])
        model.to('cuda');b.old.install_precision_runtime(model,b.torch.bfloat16)
        warm=runtime.warm_frozen(model)
        return model,dict(checkpoint_id=meta['checkpoint_id'],strict_missing=[],strict_unexpected=[],
            model_policy=model_policy(b.HF),bit_budget=budget,warmup=warm)
    runner.load_model=loader
    return runner.run(str(config_path),str(output),datasets=datasets,max_runtime_seconds=86400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=('fresh','mmlu','seven','predict'),required=True)
    parser.add_argument('--short',action='store_true');parser.add_argument('--config',type=Path)
    parser.add_argument('--datasets',nargs='+');args=parser.parse_args();b.initialize()
    request=b.read(args.request);b.parent.s.check_pins(request['source_sha256'])
    args.output.mkdir(parents=True,exist_ok=True);code=1
    try:
        result=(predict(args.config,args.output,args.datasets) if args.component=='predict' else
            fresh(args.checkpoint,args.output,request) if args.component=='fresh' else
            downstream(args.checkpoint,args.output,args.component,args.short))
        b.write(args.output/(args.component+'.json'),result);code=0
        print(dict(component=args.component,status='complete',checkpoint_id=result.get('checkpoint_id')),flush=True)
    except BaseException:
        b.write(args.output/(args.component+'_failure.json'),dict(error=traceback.format_exc()));raise
    finally:(args.output/(args.component+'.exit')).write_text(str(code)+'\n')


if __name__=='__main__':main()

"""Fresh native recovery consumer and unchanged HFLM terminal readouts."""
import argparse
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from . import bindings as b, runtime, training
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict

SEVEN='boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'


def load(directory):
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Recovery native strict load failed.')
    provenance=meta['extra_meta']['recovery_compare']
    b.require(provenance['original_parent_id']==b.SOURCE_ID and provenance['only_decoder_updates']
        and meta['checkpoint_id']!=b.SOURCE_ID and len(meta['converted_modules'])==252,
        'Never label a recovered endpoint as the original no-recovery parent.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,
        'Retain the complete S4 ledger; this is not final2.6 acceptance.')
    model.eval().requires_grad_(False)
    return model,meta,budget


def fresh(root):
    b.resources()
    report=b.read(root/'report.json')
    b.require((root/'run.exit').read_text().strip()=='0', 'Require the actual completed recovery export.')
    b.parent.s.check_pins(report['source_sha256'])
    model,meta,budget=load(root/'final_model')
    expected=b.read(root/'expected_state.json')
    b.require(runtime.records(model.state_dict())==expected, 'Fresh complete state differs from the actual BF16 export.')
    handles,mapping=b.old_life.apply_map(model);b.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    fixture=b.torch.load(root/'fixture.pt',map_location='cpu',weights_only=True)
    logits=b.run_short.probe(model,fixture)
    b.require(runtime.records(model.state_dict())==expected, 'Map/prewarm/forward changed a persistent field.')
    result=dict(status='verified_S4_recovery_fresh',checkpoint_id=meta['checkpoint_id'],
        source_checkpoint_id=b.SOURCE_ID,full_state_exact=True,persistent_fields=len(expected),
        ordinary_group8_warmup=warm,layer_device_map=mapping,real_BF16_forward_finite=True,
        output=b.old_life.tensor_record(logits),bit_budget=budget,memory=b.old_life.memory(),
        numerical_scope='Actual saved BF16 state exact and real consumer forward; no FP32-master logits equivalence claim.')
    if 'live_logits' in fixture:
        result['live_training_to_BF16_deployment']=b.run_short.comparison(fixture['live_logits'],logits)
        result['live_training_path']=fixture['live_path']
        result['live_difference_boundary']='Measured FP32-master packed training -> original BF16 deployment; no exact or quality threshold.'
    result['final_lowbit_acceptance_evaluated']=False
    b.write(root/'fresh.json',result)
    for handle in handles:handle.remove()
    return result


def downstream(root,shard,short):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    b.require(visible in ('6','7') and b.torch.cuda.device_count()==1, 'Use one assigned readout GPU6 or7.')
    report=b.read(root/'report.json');b.parent.s.check_pins(report['source_sha256'])
    b.require(b.read(root/'fresh.json')['status']=='verified_S4_recovery_fresh', 'Fresh validation must precede the endpoint evaluation.')
    model,meta,budget=load(root/'final_model')
    model.to('cuda');b.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(b.HF),use_fast=True,trust_remote_code=True,local_files_only=True)
    api=b.parent.module(b.parent.SAMPLE/'eval_utils_with_samples.py','_S4_recovery_original_hflm')
    output=root/'evaluation';output.mkdir(exist_ok=True)
    args=SimpleNamespace(tasks='mmlu' if shard=='mmlu' else SEVEN,num_fewshot=0,
        batch_size='64' if shard=='mmlu' else '32',lm_limit=1 if short else None,model_path=str(b.HF),
        eval_log_dir=None,eval_run_ts=None,mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output/(shard+'_samples')),mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    model.config.use_cache=False
    result=api.run_lm_eval(model,tokenizer,args)
    metrics=None
    if not short:
        from experiments.w25_eval.acceptance import leaf_metrics,_group_agrees
        metrics,_,batch=leaf_metrics(result['raw_results'],result['n_samples'],result['n_shot'],result['configs'])
        _group_agrees(metrics,result['raw_results'],result['group_results'])
        reference=b.read(b.parent.BASELINE/(shard+'.json'))
        b.require(result['configs']==reference['configs'] and str(batch)==str(result['batch_size'])
            and all(result['versions'][name]==reference['versions'][name] for name in result['n_samples']),
            'Preserve complete original task versions/configs/batches.')
    payload=dict(checkpoint_id=meta['checkpoint_id'],checkpoint_dir=str(root/'final_model'),
        arm=report['arm'],validation_only=short,strict_load=True,bit_budget=budget,warmup=warm,
        evals=dict(lm_eval=result),verified_task_metrics=metrics,actual_updates=report['planned_updates'],
        scope='Eight original HFLM tasks only; no LongBench or complete-nine claim.')
    b.write(output/(shard+'.json'),payload)
    return dict(checkpoint_id=meta['checkpoint_id'],shard=shard,metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--component',choices=('fresh','mmlu','seven'),required=True)
    parser.add_argument('--short',action='store_true')
    args=parser.parse_args();b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
    b.torch.set_float32_matmul_precision('highest')
    code=1
    try:
        result=fresh(args.root) if args.component=='fresh' else downstream(args.root,args.component,args.short)
        code=0;print(result,flush=True)
    except BaseException:
        b.write(args.root/(args.component+'_failure.json'),dict(error=traceback.format_exc()))
        raise
    finally:(args.root/(args.component+'.exit')).write_text(str(code)+'\n')


if __name__=='__main__':main()

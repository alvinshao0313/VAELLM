"""Fresh strict sources, CPU union/sidecar, then unchanged native GPU full-eight inference."""
import argparse
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.qwen_attention_int4_pair.common import ROOT, read, check, SHARDS, verify_shard, summarize, _require
from experiments.qwen_attention_int4_pair.combination import build
from experiments.protected_int4_quality.protection import immutable_inventory, verify_native_cache
from experiments.longbench_v1.common import write_json


def run(path):
    record=read(path); output=Path(record['output_dir']); model=None; code=1
    _require(record['status']=='simulation' and Path.cwd()==ROOT and not (output/'report.json').exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES')==str(record['gpu']), 'Use the native parent stage once.')
    report={'status':'starting','arm':record['arm'],'base_checkpoint_id':record['base_checkpoint_id'],
        'donor_checkpoint_id':record['donor_checkpoint_id'],'started_unix':time.time(),'model_exported':False,
        'goal_acceptance':False,'native_int4_checkpoint':False}
    try:
        check(record)
        _require(torch.cuda.is_available() and torch.cuda.device_count()==1
                 and all(os.environ.get(k)==v for k,v in record['environment'].items()), 'Single-GPU environment differs.')
        for name,version in record['package_versions'].items():
            _require(importlib.metadata.version(name)==version, 'Package changed: '+name)
        model,runtime,frozen_cpu,report['composition']=build(record)
        from tools.cat_eval import _prepare_model_for_eval
        from train_utils.eval_utils import run_lm_eval
        report['warmup']=_prepare_model_for_eval(model,'cuda:0',8,logging.getLogger(__name__),{})
        policy=record['model_policy']
        _require(model.config.model_type==policy['model_type'] and model.config._attn_implementation==policy['attn_implementation']
                 and model.config.rope_scaling==policy['rope_scaling'] and model.config.rope_theta==policy['rope_theta']
                 and model.get_input_embeddings().weight.dtype==torch.bfloat16, 'Native backend/precision changed.')
        rope={k:str(v.dtype) for k,v in model.named_buffers() if 'inv_freq' in k}
        _require(rope and set(rope.values())=={'torch.float32'}, 'FP32 RoPE changed.')
        frozen_gpu=immutable_inventory(model,runtime['selected_targets'])
        _require(frozen_cpu['state']==frozen_gpu['state'] and frozen_cpu['buffers']==frozen_gpu['buffers'], 'Transfer changed frozen state.')
        report['native_cache_validation']=verify_native_cache(model,runtime)
        expected=report['composition']['payload']['accounting']['expanded_simulation_linear_bits']-report['composition']['native_union_budget']['user_linear_bits']+report['composition']['payload']['accounting']['replaced_native_qvalues_scales_bits']
        _require(report['native_cache_validation']['expanded_protected_weight_storage_bits']==expected,'Expanded runtime storage differs.')
        tokenizer=AutoTokenizer.from_pretrained(record['hf_path'],use_fast=True,local_files_only=True)
        ids=tokenizer('A short test sentence.',return_tensors='pt',add_special_tokens=False)['input_ids'][:,:4]
        _require(ids.shape==(1,4),'Require native four-token integration probe.')
        with torch.inference_mode(): logits=model(input_ids=ids.to('cuda:0'),use_cache=False).logits
        _require(logits.dtype==torch.bfloat16 and bool(torch.isfinite(logits).all()),'Native BF16 integration failed.')
        report['integration_probe']={'input_shape':list(ids.shape),'output_shape':list(logits.shape),'dtype':str(logits.dtype),
                                     'finite':True,'model_policy':policy,'rope_buffers':rope}
        del ids,logits,frozen_cpu; write_json(output/'report.json',report)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for shard,tasks,batch in SHARDS:
            args=SimpleNamespace(tasks=tasks,num_fewshot=0,batch_size=batch,lm_limit=None,model_path=record['hf_path'],
                eval_log_dir=str(output),eval_run_ts=shard,mmlu_debug_samples=0)
            metrics=run_lm_eval(model,tokenizer,args)
            verify_shard(metrics,read(record['control_result_paths'][shard])['evals']['lm_eval'],batch)
            report.setdefault('completed_shards',[]).append(shard); write_json(output/'report.json',report)
        _require(immutable_inventory(model,runtime['selected_targets'])==frozen_gpu,'Inference changed frozen state or identity.')
        report['final_native_cache_validation']=verify_native_cache(model,runtime); check(record)
        report.update(status='diagnostic_complete',**summarize(output,record),all_non_protection_state_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            longbench_status='not_evaluated',nine_item_goal_status='not_evaluated'); code=0
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    finally:
        report.update(exit_code=code,finished_unix=time.time(),elapsed_seconds=time.time()-report['started_unix'])
        write_json(output/'report.json',report); del model; gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record',type=Path,required=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(run(parser.parse_args().launch_record))

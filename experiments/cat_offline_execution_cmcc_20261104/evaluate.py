"""Actual PUBLIC v6 two-shard full8 entry; GPU execution needs its own budget."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha, public_consumer_identity
from .protocol import SAMPLE_HELPER, binding, code_identity, collect_native, collect_dense, load_baseline, read, verify_files, write
from .results import worker_result


def make_plan(*, result, baseline, baseline_path, output, authorization=None):
    require(baseline['quality_reference_verified'] and result['endpoint']['model_identity'] == baseline['model_identity'], 'Native endpoint and real BF16 reference must describe the same original model.')
    require(baseline==load_baseline(baseline_path),'Freeze only the actual source-verified BF16 reference receipt.')
    output = Path(output).resolve()
    plan = dict(schema='cat_PUBLIC_native_full8_plan_v0', endpoint=result['endpoint'], baseline_receipt=binding(baseline_path),
        protocol_identity=baseline['protocol_identity'], output_dir=str(output), source_sha256=code_identity(), authorization=authorization,
        phase=result['phase'], formal_eligible=result['formal_eligible'], expected=dict(leaf_count=64,sample_count=24742), expected_stages=[])
    for name, count in (('mmlu',14042),('seven',10700)):
        argv=[sys.executable,'-B','-u','-m',__package__+'.evaluate','--component','shard','--plan',str(output/'plan.json'),'--shard',name]
        plan['expected_stages'].append(dict(name=name,argv=argv,sample_count=count))
    return plan


def verify_plan(plan):
    require(plan['schema']=='cat_PUBLIC_native_full8_plan_v0', 'Versioned actual PUBLIC full8 plan required.')
    verify_files(plan['source_sha256'])
    reference=plan['baseline_receipt']
    require(file_sha(reference['path'])==reference['sha256'], 'Frozen BF16 reference receipt changed.')
    baseline=load_baseline(reference['path'])
    require(baseline['quality_reference_verified'] and digest(baseline['protocol'])==plan['protocol_identity']==baseline['protocol_identity'], 'Common actual full8 execution protocol changed.')
    endpoint=plan['endpoint'];directory=Path(endpoint['path'])
    if endpoint['kind']=='native_v6':
        meta=read(directory/'checkpoint_meta.json')
        require(meta['format']=='vaellm_model_checkpoint_v6' and meta['schema_version']==6 and meta['checkpoint_id']==endpoint['checkpoint_id']
            and file_sha(directory/'checkpoint_meta.json')==endpoint['metadata_sha256']
            and file_sha(directory/meta['state_dict_file'])==endpoint['state_sha256'], 'Actual PUBLIC checkpoint UUID/metadata/weights changed.')
    else:
        require(endpoint['kind']=='BF16' and directory.resolve()==Path(baseline['source_evidence']['model_path']).resolve(), 'Only the original actual BF16 source is a dense evaluation endpoint.')
        verify_files(endpoint['source_files_sha256']);meta=None
    require(endpoint['model_identity']==baseline['model_identity'] and plan['formal_eligible']==endpoint['formal_eligible'], 'Source content or phase eligibility changed.')
    require(public_consumer_identity(), 'The actual PUBLIC consumer must remain loaded.')
    return baseline,meta


def make_bf16_plan(*, inventory_receipt, baseline_path, output, authorization=None):
    baseline=load_baseline(baseline_path);inventory=read(inventory_receipt)['inventory']
    require(baseline['model_identity']==inventory['model_identity']
        and file_sha(inventory_receipt)==baseline['source_evidence']['inventory_receipt']['sha256'], 'Original BF16 source inventory differs from the reference.')
    source=Path(inventory['source_evidence']['path']).resolve()
    endpoint=dict(kind='BF16',path=str(source),reference_id=baseline['reference_id'],model_identity=inventory['model_identity'],
        source_files_sha256={str(source/name):item['sha256'] for name,item in inventory['source_evidence']['files'].items()},formal_eligible=True,phase='formal')
    return make_plan(result=dict(endpoint=endpoint,formal_eligible=True,phase='formal'),baseline=baseline,
        baseline_path=baseline_path,output=output,authorization=authorization)


def load_bf16_model(plan):
    import torch
    from transformers import AutoModelForCausalLM
    baseline,_=verify_plan(plan);policy=baseline['protocol']['execution']
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Dense full8 also requires exactly one assigned visible GPU.')
    model=AutoModelForCausalLM.from_pretrained(plan['endpoint']['path'],torch_dtype=torch.bfloat16,
        attn_implementation=policy['attn_implementation'],local_files_only=True,trust_remote_code=False).to('cuda:0').eval()
    model.config.use_cache=False
    rope={name:dict(dtype=str(value.dtype),shape=list(value.shape)) for name,value in model.named_buffers() if 'inv_freq' in name}
    require(model.config.model_type==policy['model_type'] and model.config.rope_scaling==policy['rope_scaling'] and model.config.rope_theta==policy['rope_theta']
        and model.config._attn_implementation==policy['attn_implementation'] and all(value.dtype==torch.bfloat16 for value in model.parameters())
        and rope and all(row['dtype']=='torch.float32' for row in rope.values()),'Actual original BF16 architecture/backend/dtype/RoPE differs.')
    return model,dict(reference_id=plan['endpoint']['reference_id'],PUBLIC_consumer_verified=True,dtype='torch.bfloat16',rope_FP32_verified=True,
        rope_buffers=rope,attn_implementation=model.config._attn_implementation,persistent_tensors=len(model.state_dict()),cpu_mechanism_only=False,
        parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}))


def load_native_model(plan, *, device='cuda:0', cpu_mechanism=False):
    baseline,expected=verify_plan(plan)
    return load_bound_native(plan['endpoint'],baseline['protocol']['execution'],device=device,cpu_mechanism=cpu_mechanism)


def load_bound_native(endpoint, policy, *, device='cuda:0', cpu_mechanism=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
    from tools.cat_eval import _prepare_model_for_eval
    directory=Path(endpoint['path']);expected=read(directory/'checkpoint_meta.json')
    require(expected['checkpoint_id']==endpoint['checkpoint_id'] and file_sha(directory/'checkpoint_meta.json')==endpoint['metadata_sha256']
        and file_sha(directory/expected['state_dict_file'])==endpoint['state_sha256'], 'Load only the actual bound whole native payload.')
    require(cpu_mechanism or (device=='cuda:0' and torch.cuda.is_available() and torch.cuda.device_count()==1), 'Full8 execution requires exactly one assigned visible CUDA device.')
    require(not cpu_mechanism or device=='cpu', 'The CPU mechanism branch cannot enter a formal CUDA evaluation.')
    config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    require(config.model_type==policy['model_type'] and config.rope_theta==policy['rope_theta'] and config.rope_scaling==policy['rope_scaling'], 'Native architecture/RoPE differs from BF16 source policy.')
    attention='eager' if cpu_mechanism else policy['attn_implementation']
    with no_init_weights():
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16,attn_implementation=attention,trust_remote_code=False)
    model,meta,loaded=load_v6_full_checkpoint_into_model(model,str(directory),strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id']==expected['checkpoint_id'], 'Independent full native load failed.')
    model.eval();model.config.use_cache=False
    warmup=_prepare_model_for_eval(model,device,8,logging.getLogger(__name__),{})
    require(warmup['total']==warmup['warmed']==len(meta['converted_modules']) and not warmup['failed'], 'Every actual PUBLIC native projection must complete group8 prewarm.')
    rope={name:dict(dtype=str(value.dtype),shape=list(value.shape)) for name,value in model.named_buffers() if 'inv_freq' in name}
    require(rope and all(row['dtype']=='torch.float32' for row in rope.values())
        and all(value.dtype==torch.bfloat16 for value in model.parameters()), 'Preserve all BF16 weights and actual FP32 RoPE buffers.')
    return model,dict(checkpoint_id=meta['checkpoint_id'],strict_missing=[],strict_unexpected=[],PUBLIC_consumer_verified=True,
        dtype='torch.bfloat16',rope_FP32_verified=True,rope_buffers=rope,attn_implementation=model.config._attn_implementation,
        warmup=warmup,persistent_tensors=len(model.state_dict()),cpu_mechanism_only=cpu_mechanism,
        parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}))


def authorization(plan):
    auth=plan.get('authorization')
    require(auth and auth['authorization_reference'] and isinstance(auth['physical_GPU'],int) and auth['physical_GPU'] in range(8)
        and auth['authorized_GPU_hours']>0 and auth['authorized_start_epoch']<=time.time()<auth['absolute_deadline_epoch']
        and auth['absolute_deadline_epoch']-auth['authorized_start_epoch']<=auth['authorized_GPU_hours']*3600,
        'A separately confirmed single-card full8 budget and shared live absolute deadline are required.')
    require(plan['formal_eligible'] and plan['phase']=='formal','Technical/CPU workers cannot produce a formal full8 selection score.')
    return auth


def child_environment(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',ACCELERATE_USE_CPU='false')


def sample_api():
    # Load the mature real sample-preserving evaluator after the PUBLIC family
    # has been bound. Its sibling import is temporary and process-local.
    original=list(sys.path)
    try:
        sys.path.insert(0,str(SAMPLE_HELPER.parent))
        spec=importlib.util.spec_from_file_location('_cat_PUBLIC_full8_samples',SAMPLE_HELPER)
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    finally:sys.path[:]=original
    public_consumer_identity()
    return api


def run_shard(plan_path,shard):
    import torch
    from transformers import AutoTokenizer
    plan=read(plan_path);auth=authorization(plan)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(auth['physical_GPU']) and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae', 'The internal shard must inherit its exactly authorized GPU and caller bitvae.')
    baseline,_=verify_plan(plan)
    output=Path(plan['output_dir'])/(shard+'.json')
    require(shard in ('mmlu','seven') and not output.exists(), 'No invalid shard or overwritten evaluation.')
    native=plan['endpoint']['kind']=='native_v6'
    model,audit=load_native_model(plan) if native else load_bf16_model(plan)
    source=baseline['source_evidence']['model_path']
    tokenizer=AutoTokenizer.from_pretrained(source,use_fast=True,local_files_only=True,trust_remote_code=False)
    require(tokenizer.is_fast and type(tokenizer).__name__==baseline['tokenizer_inventory']['tokenizer_class'], 'Actual source tokenizer class/fast policy differs.')
    args=SimpleNamespace(tasks='mmlu' if shard=='mmlu' else 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa',
        num_fewshot=0,batch_size='64' if shard=='mmlu' else '32',lm_limit=None,model_path=source,
        eval_log_dir=None,eval_run_ts=None,mmlu_debug_samples=0,mmlu_debug_log_dir=str(Path(plan['output_dir'])/(shard+'_samples')),
        mmlu_debug_run_ts='full')
    started=time.time();api=sample_api()
    # Reuse the actual HFLM/evaluator and mature serializers, while recording
    # the real HFLM conditions that the legacy convenience wrapper discards.
    lm=api.lm_eval.models.huggingface.HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',trust_remote_code=True,
        batch_size=args.batch_size,add_bos_token=False)
    require(lm.max_length==baseline['protocol']['execution']['max_length'] and lm.add_bos_token is False, 'Actual HFLM likelihood window/BOS differs from BF16.')
    task_names=args.tasks.split(',')
    with torch.no_grad():
        raw=api.evaluator.simple_evaluate(model=lm,tasks=task_names,num_fewshot=0,batch_size=args.batch_size,limit=None,
            random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234,log_samples=True)
    table=api._build_lm_eval_summary_rows(task_names=task_names,results=raw['results'],groups=raw.get('groups',{}))
    result=dict(tasks=task_names,num_fewshot=0,batch_size=args.batch_size,limit=None,raw_results=raw['results'],group_results=raw.get('groups',{}),
        configs=raw['configs'],versions=raw['versions'],higher_is_better=raw.get('higher_is_better',{}),n_shot=raw['n-shot'],n_samples=raw['n-samples'],
        task_metrics={row['task']:row['metric'] for row in table},task_metric_keys={row['task']:row['metric_key'] for row in table},summary_rows=table,
        sample_artifacts=api.write_lm_eval_samples(raw['samples'],output_dir=args.mmlu_debug_log_dir,run_ts='full'))
    audit['HFLM']=dict(max_length=lm.max_length,add_bos_token=lm.add_bos_token,batch_size=args.batch_size,log_samples=True,
        random_seed=0,numpy_seed=1234,torch_seed=1234,fewshot_seed=1234)
    require(model.config.use_cache is False and model.config._attn_implementation==baseline['protocol']['execution']['attn_implementation']
        and all(value.dtype==torch.float32 for name,value in model.named_buffers() if 'inv_freq' in name), 'Evaluation changed actual backend/cache/RoPE policy.')
    verify_files(plan['source_sha256'])
    identity=(dict(checkpoint_dir=plan['endpoint']['path'],checkpoint_id=plan['endpoint']['checkpoint_id'],checkpoint_loader='v6',
        strict_load=True,adapter_dir=None,eval_hif4_act=False) if native else dict(model_kind='BF16',reference_id=plan['endpoint']['reference_id']))
    write(output,dict(**identity,base_model_path=source,tokenizer_class=type(tokenizer).__name__,tokenizer_is_fast=True,
        endpoint=plan['endpoint'],protocol_identity=plan['protocol_identity'],execution_conditions=baseline['protocol']['execution'],load_audit=audit,
        evals=dict(lm_eval=result),elapsed_seconds=time.time()-started,source_plan=binding(plan_path)))


def run(plan_path):
    plan_path=Path(plan_path).resolve();plan=read(plan_path);auth=authorization(plan)
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'The CPU controller requires caller bitvae/hidden CUDA.')
    baseline,_=verify_plan(plan)
    directory=Path(plan['output_dir']);record=dict(status='running',endpoint_identity=plan['endpoint'].get('checkpoint_id',plan['endpoint'].get('reference_id')),source_plan=binding(plan_path),
        started_unix=time.time(),stages=[],exit_code=None)
    if plan['endpoint']['kind']=='native_v6':record['checkpoint_id']=plan['endpoint']['checkpoint_id']
    path=directory/'evaluation.json';require(not path.exists(), 'No automatic retry or overwritten evaluation.')
    try:
        for stage in plan['expected_stages']:
            remaining=int(auth['absolute_deadline_epoch']-time.time())-20
            require(remaining>0, 'Full8 shared budget is exhausted.')
            from experiments.cmcc_runtime_20261030.sequence import check_resources
            resources=check_resources([str(auth['physical_GPU'])])
            argv=['timeout','--signal=TERM','--kill-after=20s',str(remaining)+'s',*stage['argv']]
            with (directory/(stage['name']+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    cwd=Path(__file__).resolve().parents[2],env=child_environment(auth['physical_GPU']))
                row=dict(stage,command=argv,pid=child.pid,process_group_id=child.pid,resource_check=resources,started_unix=time.time())
                record['stages'].append(row)
                path.write_text(json.dumps(record,indent=2)+'\n')
                try:code=child.wait()
                finally:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            row.update(exit_code=code,finished_unix=time.time())
            require(code==0, 'Actual full8 shard failed; no partial score or retry.')
        paths=[directory/'mmlu.json',directory/'seven.json']
        collector=collect_native if plan['endpoint']['kind']=='native_v6' else collect_dense
        summary=collector(paths,endpoint=plan['endpoint'],baseline=baseline)
        write(directory/'full8_summary.json',summary)
        record.update(status='complete_original_full8',exit_code=0,leaf_tasks=summary['leaf_count'],sample_count=summary['sample_count'],full8_mean_percent=summary['mean_percent'])
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        path.write_text(json.dumps(record,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',required=True,choices=('baseline','prepare','prepare-bf16','run','shard'))
    parser.add_argument('--inventory');parser.add_argument('--output',type=Path);parser.add_argument('--completed')
    parser.add_argument('--candidate-id');parser.add_argument('--baseline');parser.add_argument('--authorization')
    parser.add_argument('--request');parser.add_argument('--request-sha256')
    parser.add_argument('--plan');parser.add_argument('--shard',choices=('mmlu','seven'))
    args=parser.parse_args()
    if args.component=='baseline':
        from .protocol import baseline_receipt
        write(args.output,baseline_receipt(args.inventory))
    elif args.component in ('prepare','prepare-bf16'):
        authorization=read(args.authorization) if args.authorization else None
        if args.component=='prepare':
            require(bool(args.request)==bool(args.request_sha256),'Registry override requires both actual path and SHA.')
            result=worker_result(args.completed,args.candidate_id,
                request_reference=dict(path=args.request,sha256=args.request_sha256) if args.request else None)
            baseline=load_baseline(args.baseline)
            plan=make_plan(result=result,baseline=baseline,baseline_path=args.baseline,output=args.output,authorization=authorization)
        else:
            plan=make_bf16_plan(inventory_receipt=args.inventory,baseline_path=args.baseline,output=args.output,authorization=authorization)
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'plan.json',plan)
    elif args.component=='run':run(args.plan)
    else:run_shard(args.plan,args.shard)


if __name__=='__main__':main()

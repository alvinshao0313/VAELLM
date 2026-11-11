"""Same mixed consumer for complete raw full8, train readout and final full9."""
import argparse
import gc
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as s
from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def binding(directory,report_path,*,final=False):
    directory=Path(directory).resolve()
    meta,budget,hashes=b.inspect(directory)
    prepared=s.read(s.RESULT / 'prepared.json')
    s.check_pins(prepared['source_sha256'])
    if final:
        report=s.read(report_path)
        s.require(report['status'] == 'completed_own_down500' and report['exit_code'] == 0
            and report['final_checkpoint_id'] == meta['checkpoint_id'] and report['original_stage_exact']
            and report['source_binding'] == prepared['source_binding'] and report['arm'] == 'd16'
            and report['recovery']['actual_steps'] == 500 and report['recovery']['frozen_persistent_fields_exact']
            and Path(report['final_model']).resolve() == directory,'Require the actual D16 own final down500 endpoint.')
        strict_path=Path(report['final_strict_reload_report'])
    elif directory == s.CONTROL:
        s.require(meta['checkpoint_id'] == prepared['source_binding']['raw_control']['checkpoint_id'],
            'Use the genuine original raw-down control, with no retraining.')
        strict_path=None
    else:
        report=s.read(report_path)
        s.require(report['status'] == 'completed_formal_raw_down' and report['exit_code'] == 0
            and report['raw_checkpoint_id'] == meta['checkpoint_id'] and report['source_binding'] == prepared['source_binding']
            and Path(report['raw_model']).resolve() == directory,'Require this matched fork own completed raw down.')
        strict_path=Path(report['strict_reload_report'])
    if strict_path:
        strict=s.read(strict_path)
        s.require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
            and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
            and all(strict['actual_budget'][key] == budget[key] for key in b.BUDGET_FIELDS),
            'Require the independent actual mixed BF16 deployment fresh check.')
        hashes[str(strict_path)]=s.sha(strict_path)
    hashes[str(report_path)]=s.sha(report_path)
    return dict(checkpoint_dir=str(directory),checkpoint_id=meta['checkpoint_id'],checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory / s.checkpoint.META_FILE)],checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory / s.checkpoint.STATE_FILE),bit_budget=budget,
        pipeline_report=str(report_path),pipeline_provenance=meta['provenance']),meta


def setup():
    cmcc.setup(s.SUPPORT)
    previous=b.source_pins
    b.source=lambda directory,report:binding(directory,report,final=True)
    b.source_pins=lambda value:{**previous(value),**s.pins()}
    b.command=lambda component,*args:[str(s.PYTHON),'-B','-u','-m','experiments.down_geometry_baseline_cmcc_20261103.evaluate',
        '--component',component,*map(str,args)]


def raw_location(arm):
    if arm == 'control':
        return s.CONTROL,s.C0 / 'report.json'
    return s.OUTPUT / arm / 'formal/raw_model',s.OUTPUT / arm / 'formal/report.json'


def raw_full8(arm):
    s.cpu_environment()
    directory,report=raw_location(arm)
    model,_meta=binding(directory,report)
    pins=b.source_pins(model)
    config=b.make_config(model,pins)
    output=s.RESULT / arm / 'raw_full8'
    s.require(not output.exists(),'Do not repeat a raw full8 evaluation.')
    output.mkdir(parents=True)
    s.write(output / 'config.json',config)
    gpu=2 if arm == 'control' else s.ARMS[arm]['gpu']
    record=dict(status='running',checkpoint_id=model['checkpoint_id'],started_unix=time.time(),stages={})
    try:
        for shard in ('mmlu','seven'):
            command=b.command('full8','--config',output / 'config.json','--shard',shard,'--output',output / (shard+'.json'))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            with (output / (shard+'.log')).open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=s.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                record['stages'][shard]=dict(command=command,pid=process.pid,gpu=gpu,started_unix=time.time())
                s.write(output / 'run.json',record)
                code=process.wait()
            record['stages'][shard].update(exit_code=code,finished_unix=time.time())
            s.write(output / 'run.json',record)
            s.require(code == 0,'Raw full8 shard failed; no retry or partial-scoring continuation.')
        from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
        summary=full8_summary(config,[output / 'mmlu.json',output / 'seven.json'],output / 'full8_summary.json')
        record.update(status='completed_raw_full8',exit_code=0,mean_percent=summary['mean_percent'],
            leaf_count=summary['leaf_count'],sample_count=summary['sample_count'])
    except BaseException:
        record.update(status='failed',exit_code=1)
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        s.write(output / 'run.json',record)


def readout(arm):
    from experiments.adaptive_class_20261030.run_pipeline import warm
    directory,_report=raw_location(arm)
    output=s.RESULT / arm / 'raw_readout.json'
    s.require(not output.exists(),'Do not overwrite a deployment readout.')
    tokens=s.torch.load(Path(s.sources()['common']['source_raw2']['path']) / 'cat_runtime_state.pt',map_location='cpu',weights_only=True)
    inputs=tokens['activation_runtime']['cache']['input_ids'][:8]
    s.require(len(inputs) == 8 and all(value.numel() == 256 for value in inputs),'Use fixed first8 original train calibration sequences.')
    parent,pmeta,loaded,_=s.checkpoint.load(s.PARENT)
    s.require(not loaded.missing_keys and not loaded.unexpected_keys,'Strict after-up reference load.')
    warm(parent,180)
    values={f'model.layers.{layer}.mlp.down_proj':[] for layer in range(36)}
    handles=[]
    for name in values:
        def capture(_module,args,name=name):
            values[name].append(args[0].detach().cpu())
        handles.append(parent.get_submodule(name).register_forward_pre_hook(capture))
    with s.torch.inference_mode():
        for row in inputs:
            parent(row.reshape(1,256).to('cuda'),use_cache=False,logits_to_keep=1)
    for handle in handles:
        handle.remove()
    del parent
    gc.collect()
    s.torch.cuda.empty_cache()
    model,meta,loaded,budget=s.checkpoint.load(directory)
    s.require(not loaded.missing_keys and not loaded.unexpected_keys,'Strict raw-down deployed load.')
    warmup=warm(model,216)
    reference=s.torch.load(s.PARENT / s.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    rows=[]
    total=dict(weight_error=0.,weight_reference=0.,weight_elements=0,output_error=0.,output_reference=0.,output_elements=0)
    with s.torch.inference_mode():
        for name,activations in values.items():
            linear=model.get_submodule(name)
            target=reference[name+'.weight'].to('cuda')
            actual=linear._cached_weight
            s.require(actual is not None and actual.dtype == s.torch.bfloat16 and actual.shape == target.shape,
                'Use the actual native group8 BF16 decoded deployment weight.')
            error=(actual.float()-target.float()).square().sum(dtype=s.torch.float64).item()
            scale=target.float().square().sum(dtype=s.torch.float64).item()
            oerror=oref=count=0
            for value in activations:
                value=value.to('cuda')
                decoded=linear(value)
                gold=s.torch.nn.functional.linear(value,target,linear.bias)
                oerror+=(decoded.float()-gold.float()).square().sum(dtype=s.torch.float64).item()
                oref+=gold.float().square().sum(dtype=s.torch.float64).item()
                count+=gold.numel()
            s.require(all(math.isfinite(x) for x in (error,scale,oerror,oref)) and scale>0 and oref>0,'Nonfinite deployed readout.')
            rows.append(dict(name=name,weight_MSE=error/target.numel(),weight_NMSE=error/scale,output_MSE=oerror/count,output_NMSE=oerror/oref))
            for key,value in dict(weight_error=error,weight_reference=scale,weight_elements=target.numel(),
                    output_error=oerror,output_reference=oref,output_elements=count).items():
                total[key]+=value
    import hashlib
    digest=hashlib.sha256(b''.join(value.contiguous().numpy().tobytes() for value in inputs)).hexdigest()
    s.write(output,dict(status='completed_same_train_deployment_readout',arm=arm,checkpoint_id=meta['checkpoint_id'],
        parent_checkpoint_id=pmeta['checkpoint_id'],checkpoint_dir=str(directory),input_ids_sha256=digest,
        data=dict(dataset='alpaca=1',split='train',samples=8,sequence_length=256,indices=list(range(8))),
        deployment=dict(weight_MSE=total['weight_error']/total['weight_elements'],weight_NMSE=total['weight_error']/total['weight_reference'],
            output_MSE=total['output_error']/total['output_elements'],output_NMSE=total['output_error']/total['output_reference'],
            raw_sums=total,per_down=rows),warmup=warmup,bit_budget=budget,
        boundary='Fixed original train input local diagnostic only; full8 and complete final nine determine quality.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',default='evaluate',choices=('evaluate','full8','predict','summarize','merge','raw_full8','readout'))
    parser.add_argument('--arm',choices=('control',*s.ARMS))
    args,remaining=parser.parse_known_args()
    setup()
    b.CLI_ARGUMENTS=list(sys.argv)
    s.torch.set_num_threads(8)
    if args.component == 'raw_full8':
        raw_full8(args.arm)
    elif args.component == 'readout':
        readout(args.arm)
    else:
        sys.argv=[sys.argv[0],*remaining]
        raise SystemExit(importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component).main())

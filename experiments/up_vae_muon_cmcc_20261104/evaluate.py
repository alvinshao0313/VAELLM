"""Complete raw full8 and original same-train up deployment readout."""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as s
from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def location(arm):
    return (s.CONTROL,s.C0 / 'report.json') if arm == 'control' else (s.OUTPUT / arm / 'formal/raw_model',s.OUTPUT / arm / 'formal/report.json')


def source(arm):
    directory, report_path = location(arm)
    prepared = s.read(s.RESULT / 'prepared.json');s.check_pins(prepared['source_sha256'])
    meta,budget,hashes = b.inspect(directory,final=False)
    s.require([len(meta[key]) for key in ('native_modules','compact_modules','pending_dense_modules')] == [180,36,36]
        and all(row['name'].endswith('.down_proj') for row in meta['pending_dense_modules'])
        and not budget['final_budget_applicable'], 'Evaluate the actual partial up raw, never a fabricated final model.')
    if arm == 'control':
        s.require(meta['checkpoint_id'] == prepared['source_binding']['raw_control']['checkpoint_id'], 'Reuse the actual original raw-up control.')
    else:
        report = s.read(report_path)
        s.require(report['status'] == 'completed_up_formal_raw' and report['exit_code'] == 0 and not report['validation_only']
            and report['arm'] == arm and report['source_binding'] == prepared['source_binding']
            and report['raw_checkpoint_id'] == meta['checkpoint_id'] and report['fit']['actual_updates_per_stage'] == 10000
            and report['fit']['vector_batch_per_layer'] == s.ARMS[arm]['batch']
            and report['fit']['all_frozen_prefix_fields_exact'] and report['fit']['protection_and_rotation_fields_exact_to_control'],
            'Require this actual complete fresh own-parent up fit.')
        if arm == 'muon':
            s.require(report['fit']['original_C0_initialization_digest_match'] == [True,True], 'Formal Muon must preserve both original initializations.')
        strict_path = Path(report['strict_reload_report']);strict = s.read(strict_path)
        s.require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
            and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
            and all(strict['actual_budget'][key] == budget[key] for key in b.BUDGET_FIELDS), 'Require the actual independent partial deployment fresh proof.')
        hashes[str(strict_path)] = s.sha(strict_path)
    hashes[str(report_path)] = s.sha(report_path)
    return dict(checkpoint_dir=str(directory),checkpoint_id=meta['checkpoint_id'],checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory / s.checkpoint.META_FILE)],checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory / s.checkpoint.STATE_FILE),bit_budget=budget,
        pipeline_report=str(report_path),pipeline_provenance=meta['provenance']),meta


def setup():
    cmcc.setup(s.SUPPORT)
    previous = b.source_pins
    b.source_pins = lambda value:{**previous(value),**s.pins()}
    b.command = lambda component,*args:[str(s.PYTHON),'-B','-u','-m','experiments.up_vae_muon_cmcc_20261104.evaluate',
        '--component',component,*map(str,args)]


def raw_full8(arm):
    s.cpu_environment()
    binding,_meta = source(arm)
    config = b.make_config(binding,b.source_pins(binding))
    output = s.RESULT / arm / 'raw_full8'
    s.require(not output.exists(), 'Never repeat or overwrite this full8 endpoint.')
    output.mkdir(parents=True);s.write(output / 'config.json',config)
    gpu = s.CONTROL_GPU if arm == 'control' else s.ARMS[arm]['gpu']
    record = dict(status='running',checkpoint_id=binding['checkpoint_id'],started_unix=time.time(),stages={})
    try:
        for shard in ('mmlu','seven'):
            argv = b.command('full8','--config',output / 'config.json','--shard',shard,'--output',output / (shard+'.json'))
            with (output / (shard+'.log')).open('x',encoding='utf-8') as stream:
                process = subprocess.Popen(argv,cwd=s.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row = dict(command=argv,pid=process.pid,gpu=gpu,started_unix=time.time())
                record['stages'][shard] = row;s.write(output / 'run.json',record)
                row['exit_code'] = process.wait()
            row['finished_unix'] = time.time();s.write(output / 'run.json',record)
            s.require(row['exit_code'] == 0, 'Full8 shard failed; no retry or partial quality scoring.')
        from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
        summary = full8_summary(config,[output / 'mmlu.json',output / 'seven.json'],output / 'full8_summary.json')
        s.require(summary['leaf_count'] == 64 and summary['sample_count'] == 24742 and summary['checkpoint_id'] == binding['checkpoint_id'],
            'The real full8 summary must bind all64 leaves/24742 examples to this endpoint.')
        record.update(status='completed_raw_full8',exit_code=0,mean_percent=summary['mean_percent'],leaf_count=summary['leaf_count'],sample_count=summary['sample_count'])
    except BaseException:
        record.update(status='failed',exit_code=1);raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix']);s.write(output / 'run.json',record)


def readout(arm):
    from experiments.adaptive_class_20261030.run_pipeline import warm
    binding,_meta = source(arm)
    directory,_report = location(arm)
    output = s.RESULT / arm / 'raw_readout.json'
    s.require(not output.exists(), 'Never overwrite a deployment readout.')
    prepared = s.read(s.RESULT / 'prepared.json')
    runtime = s.torch.load(Path(prepared['source_binding']['common']['source_raw2']['path']) / 'cat_runtime_state.pt',map_location='cpu',weights_only=True)
    inputs = runtime['activation_runtime']['cache']['input_ids'][:8]
    s.require(len(inputs) == 8 and all(value.numel() == 256 for value in inputs), 'Use the original fixed first8x256 train blocks.')
    parent,pmeta,loaded,_ = s.checkpoint.load(s.PARENT)
    s.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Strict original after-gate reference load.')
    warm(parent,144)
    values = {f'model.layers.{layer}.mlp.up_proj':[] for layer in range(36)}
    handles = []
    for name in values:
        def capture(_module,args,name=name):values[name].append(args[0].detach().cpu())
        handles.append(parent.get_submodule(name).register_forward_pre_hook(capture))
    with s.torch.inference_mode():
        for row in inputs:parent(row.reshape(1,256).to('cuda'),use_cache=False,logits_to_keep=1)
    for handle in handles:handle.remove()
    del parent;gc.collect();s.torch.cuda.empty_cache()
    model,meta,loaded,budget = s.checkpoint.load(directory)
    s.require(meta['checkpoint_id'] == binding['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys, 'Strict own raw-up deployment load.')
    warmup = warm(model,180)
    reference = s.torch.load(s.PARENT / s.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    rows, total = [], dict(weight_error=0.,weight_reference=0.,weight_elements=0,output_error=0.,output_reference=0.,output_elements=0)
    with s.torch.inference_mode():
        for name,activations in values.items():
            layer = model.get_submodule(name);target = reference[name+'.weight'].to('cuda');actual = layer._cached_weight
            s.require(actual is not None and actual.dtype == s.torch.bfloat16 and actual.shape == target.shape, 'Use actual native group8 BF16 decoded weights.')
            error = (actual.float()-target.float()).square().sum(dtype=s.torch.float64).item()
            scale = target.float().square().sum(dtype=s.torch.float64).item()
            oerror = oref = count = 0
            for value in activations:
                value = value.to('cuda');decoded = layer(value);gold = s.torch.nn.functional.linear(value,target,layer.bias)
                oerror += (decoded.float()-gold.float()).square().sum(dtype=s.torch.float64).item()
                oref += gold.float().square().sum(dtype=s.torch.float64).item();count += gold.numel()
            s.require(all(math.isfinite(value) for value in (error,scale,oerror,oref)) and scale > 0 and oref > 0, 'Nonfinite actual deployed readout.')
            rows.append(dict(name=name,weight_MSE=error/target.numel(),weight_NMSE=error/scale,output_MSE=oerror/count,output_NMSE=oerror/oref))
            for key,value in dict(weight_error=error,weight_reference=scale,weight_elements=target.numel(),output_error=oerror,output_reference=oref,output_elements=count).items():total[key] += value
    input_sha = hashlib.sha256(b''.join(value.contiguous().numpy().tobytes() for value in inputs)).hexdigest()
    s.require(input_sha == prepared['fixed_train_input_sha256'], 'All arms must use the actual CPU-frozen original train inputs.')
    s.write(output,dict(status='completed_same_train_deployment_readout',arm=arm,checkpoint_id=meta['checkpoint_id'],parent_checkpoint_id=pmeta['checkpoint_id'],
        checkpoint_dir=str(directory),input_ids_sha256=input_sha,data=dict(dataset='alpaca=1',split='train',samples=8,sequence_length=256,indices=list(range(8))),
        deployment=dict(weight_MSE=total['weight_error']/total['weight_elements'],weight_NMSE=total['weight_error']/total['weight_reference'],
            output_MSE=total['output_error']/total['output_elements'],output_NMSE=total['output_error']/total['output_reference'],raw_sums=total,per_up=rows),
        warmup=warmup,bit_budget=budget,boundary='Actual fixed train local diagnostic; partial raw full8 is not final nine/2.6 quality acceptance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('raw_full8','full8','readout'),required=True)
    parser.add_argument('--arm',choices=('control',*s.ARMS))
    args,remaining = parser.parse_known_args();setup();s.torch.set_num_threads(8)
    if args.component == 'raw_full8':raw_full8(args.arm)
    elif args.component == 'readout':readout(args.arm)
    else:
        from experiments.adaptive_class_evaluation_20261030 import full8
        sys.argv = [sys.argv[0],*remaining];full8.main()

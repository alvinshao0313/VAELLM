"""Existing mixed full8 and fixed train readout with this new source binding."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from . import bindings as s
from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def source():
    directory=s.OUTPUT / 'formal/raw_model'
    path=s.OUTPUT / 'formal/report.json'
    report=s.read(path)
    prepared=s.read(s.RESULT / 'prepared.json')
    s.check_pins(prepared['source_sha256'])
    meta,budget,hashes=b.inspect(directory)
    s.require(report['status']=='completed_VAE_Muon_formal_raw' and report['exit_code']==0 and not report['validation_only']
        and report['raw_checkpoint_id']==meta['checkpoint_id'] and report['source_binding']==prepared['source_binding']
        and report['contract']==s.CONTRACT and report['fit']['canonical_original_initialization_matches']==[True,True]
        and report['fit']['actual_updates_per_stage']==10000 and report['fit']['all_frozen_prefix_fields_exact']
        and report['fit']['protection_and_rotation_fields_exact_to_control'], 'Require this completed original-geometry VAE Muon raw endpoint.')
    strict_path=Path(report['strict_reload_report'])
    strict=s.read(strict_path)
    s.require(strict['status']=='strict_full_state_and_BF16_logits_verified' and strict['exit_code']==0
        and strict['checkpoint_id']==meta['checkpoint_id'] and strict['same_input_logits_exact']
        and all(strict['actual_budget'][key]==budget[key] for key in b.BUDGET_FIELDS), 'The actual independent deployment fresh evidence is missing.')
    for item in (path,strict_path):hashes[str(item)]=s.sha(item)
    return dict(checkpoint_dir=str(directory),checkpoint_id=meta['checkpoint_id'],checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory / s.checkpoint.META_FILE)],checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory / s.checkpoint.STATE_FILE),bit_budget=budget,
        pipeline_report=str(path),pipeline_provenance=meta['provenance']),meta


def setup():
    cmcc.setup(s.SUPPORT)
    previous=b.source_pins
    b.source_pins=lambda value:{**previous(value),**s.pins()}
    b.command=lambda component,*args:[str(s.PYTHON),'-B','-u','-m','experiments.cat_down_vae_muon_cmcc_20261103.evaluate',
        '--component',component,*map(str,args)]


def raw_full8():
    s.cpu_environment()
    binding,_meta=source()
    config=b.make_config(binding,b.source_pins(binding))
    output=s.RESULT / 'raw_full8'
    s.require(not output.exists(),'Never repeat or overwrite a raw full8 evaluation.')
    output.mkdir(parents=True)
    s.write(output / 'config.json',config)
    record=dict(status='running',checkpoint_id=binding['checkpoint_id'],started_unix=time.time(),stages={})
    try:
        for shard in ('mmlu','seven'):
            argv=b.command('full8','--config',output / 'config.json','--shard',shard,'--output',output / (shard+'.json'))
            with (output / (shard+'.log')).open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(argv,cwd=s.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s.GPU)),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row=dict(command=argv,pid=process.pid,gpu=s.GPU,started_unix=time.time())
                record['stages'][shard]=row;s.write(output / 'run.json',record)
                code=process.wait()
            row.update(exit_code=code,finished_unix=time.time())
            s.write(output / 'run.json',record)
            s.require(code==0,'Raw full8 failed; stop dependencies without retry or partial scoring.')
        from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
        summary=full8_summary(config,[output / 'mmlu.json',output / 'seven.json'],output / 'full8_summary.json')
        record.update(status='completed_raw_full8',exit_code=0,mean_percent=summary['mean_percent'],leaf_count=64,sample_count=24742)
    except BaseException:
        record.update(status='failed',exit_code=1)
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        s.write(output / 'run.json',record)


def readout():
    source()
    # Reuse the exact fixed train inputs, after-up activations and native BF16
    # consumer. Rebinding is confined to this fresh process, not active sources.
    from experiments.down_geometry_baseline_cmcc_20261103 import evaluate as existing
    original_binding,original_location=existing.s,existing.raw_location
    proxy=SimpleNamespace(**{key:value for key,value in vars(s).items() if not key.startswith('__')})
    proxy.RESULT=s.RESULT.parent
    proxy.sources=lambda:s.read(s.RESULT / 'prepared.json')['source_binding']
    existing.s=proxy
    existing.raw_location=lambda _arm:(s.OUTPUT / 'formal/raw_model',s.OUTPUT / 'formal/report.json')
    try:existing.readout(s.HERE.name)
    finally:existing.s,existing.raw_location=original_binding,original_location


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('raw_full8','full8','readout'),required=True)
    args,remaining=parser.parse_known_args()
    setup();s.torch.set_num_threads(8)
    if args.component=='raw_full8':raw_full8()
    elif args.component=='readout':readout()
    else:
        from experiments.adaptive_class_evaluation_20261030 import full8
        sys.argv=[sys.argv[0],*remaining]
        full8.main()

"""Unchanged full8/full21 consumers bound only to the actual closed final UUID."""
import argparse
import importlib
import math
from pathlib import Path
import sys
import time

from . import bindings as s
from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def source(directory,report_path):
    directory,report_path=Path(directory).resolve(),Path(report_path).resolve()
    s.require(directory==s.OUTPUT/'recovery/final_model' and report_path==s.OUTPUT/'recovery/report.json',
        'Evaluate only this genuine final500, never V raw, short, intermediate or another arm.')
    prepared=s.prepared()
    raw,_raw_meta=s.raw_source()
    report=s.read(report_path)
    meta,budget,hashes=b.inspect(directory)
    strict_path=Path(report['strict_reload_report'])
    strict=s.read(strict_path)
    s.require(report['schema']=='D16_Muon_own_Adam500_execution_v1'
        and report['status']=='completed_D16_Muon_own_original_Adam500_fresh' and report['exit_code']==0
        and report['phase']=='formal' and report['validation_only'] is False
        and report['source_binding']==prepared['source_binding'] and report['contract']==s.CONTRACT
        and report['prepared_sha256']==s.sha(s.PREPARED) and report['original_stage_exact']
        and report['optimizer_state_fresh'] and report['effective_optimizer']==prepared['original_recovery_optimizer']
        and report['source_raw_checkpoint_id']==raw['raw_checkpoint_id']
        and report['source_raw_report_sha256']==s.sha(s.OUTPUT/'formal/report.json')
        and report['recovery']['actual_steps']==500 and report['recovery']['frozen_persistent_fields_exact'],
        'Require the full unchanged original Adam500 source/data/optimizer contract.')
    s.require(meta['checkpoint_id']==report['final_checkpoint_id']
        and meta['checkpoint_id'] not in (raw['raw_checkpoint_id'],*prepared['controls']['final_ids'].values())
        and Path(report['final_model']).resolve()==directory and strict['status']=='strict_full_state_and_BF16_logits_verified'
        and strict['exit_code']==0 and strict['checkpoint_id']==meta['checkpoint_id'] and strict['same_input_logits_exact']
        and strict['deployment_digest']==report['expected_deployment_digest']
        and all(budget[key]==strict['actual_budget'][key] for key in b.BUDGET_FIELDS),
        'Require the actual same-UUID final deployment and independent strict/BF16 fresh.')
    provenance=meta['provenance']
    s.require(provenance['schema']=='D16_Muon_own_Adam500_provenance_v1' and provenance['phase']=='formal'
        and provenance['validation_only'] is False and provenance['source_binding']==prepared['source_binding']
        and provenance['source_raw_checkpoint_id']==raw['raw_checkpoint_id'] and provenance['actual_recovery_updates']==500
        and provenance['contract']==s.CONTRACT and provenance['effective_optimizer']==prepared['original_recovery_optimizer']
        and meta['pending_dense_modules']==[] and len(meta['native_modules'])==216 and len(meta['compact_modules'])==36,
        'Actual final provenance/topology differs from this single approved closure.')
    s.require(all(budget[key]==raw['raw_budget'][key] for key in
        ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
        'The actual decoder-recovery deployment budget changed.')
    hashes.update({str(path):s.sha(path) for path in (report_path,strict_path,s.PREPARED)})
    return dict(checkpoint_dir=str(directory),checkpoint_id=meta['checkpoint_id'],checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory/s.checkpoint.META_FILE)],checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory/s.checkpoint.STATE_FILE),bit_budget=budget,
        pipeline_report=str(report_path),pipeline_provenance=provenance),meta


def bound_longbench_plan(plan,deadline_unix,now_unix=None):
    now=time.time() if now_unix is None else now_unix
    outer=int(deadline_unix-now)-60
    native=outer-180
    s.require(math.isfinite(deadline_unix) and native>0,
        'Full9 has insufficient remaining time for bounded LongBench and its original termination grace.')
    for unit in plan['execution_units']:
        argv=unit['argv']
        s.require(argv.count('--max-runtime-seconds')==1,'Each actual LB unit must declare exactly one native runtime.')
        argv[argv.index('--max-runtime-seconds')+1]=str(native)
        unit['native_runtime_seconds']=native
        unit['outer_timeout_seconds']=outer
    return plan


def setup(deadline_unix):
    cmcc.setup(s.SUPPORT)
    prior=b.source_pins
    b.source=source
    b.source_pins=lambda binding:{**prior(binding),**s.pins()}
    b.command=lambda component,*args:[str(s.PYTHON),'-B','-u','-m',
        'experiments.down_d16_vae_muon_cmcc_20261103.evaluate','--deadline-unix',str(deadline_unix),
        '--component',component,*map(str,args)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('evaluate','full8','predict','summarize','merge'),default='evaluate')
    parser.add_argument('--deadline-unix',required=True,type=float)
    args,remaining=parser.parse_known_args()
    deadline=args.deadline_unix
    s.require(math.isfinite(deadline) and 0<deadline-time.time()<=28800,
        'Use the fixed actual full9 launch deadline, at most8 hours from this invocation.')
    setup(deadline)
    b.CLI_ARGUMENTS=list(sys.argv)
    module=importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component)
    if args.component=='evaluate':
        original_make_plan=module.make_plan
        def make_plan(*values,**fields):
            return bound_longbench_plan(original_make_plan(*values,**fields),deadline)
        module.make_plan=make_plan
    sys.argv=[sys.argv[0],*remaining]
    return module.main()


if __name__=='__main__':
    raise SystemExit(main())

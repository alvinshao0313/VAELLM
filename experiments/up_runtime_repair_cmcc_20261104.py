"""Isolated real-import repair; keep the running control's original sources intact."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import inspect
import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import transformers

from experiments.up_vae_muon_cmcc_20261104 import bindings as b, fit, run, evaluate
from experiments.cmcc_runtime_20261030 import sequence as mature

ORIGINAL_RESULT, ORIGINAL_OUTPUT, ORIGINAL_PINS = b.RESULT, b.OUTPUT, b.pins
b.RESULT, b.OUTPUT = b.RESULT / 'runtime_repair_v2', b.OUTPUT / 'runtime_repair_v2'
b.pins = lambda:{**ORIGINAL_PINS(), str(Path(__file__).resolve()):b.sha(__file__)}


@contextmanager
def original_schedule(*, short):
    module = importlib.import_module('transformers')
    b.require(module is sys.modules['transformers'], 'Hook the actual current module after N setup, never a stale LazyModule reference.')
    original = module.get_scheduler
    rows = []
    def scheduler(*args,**kwargs):
        name = kwargs.get('name',args[0] if args else None)
        b.require(name == 'linear' and kwargs['num_warmup_steps'] == 0 and kwargs['num_training_steps'] == 2,
            'Intercept only the two actual short VAE linear/no-warmup scheduler constructions.')
        kwargs['num_training_steps'] = 10000
        rows.append(dict(actual_updates=2,schedule_max=10000))
        return original(*args,**kwargs)
    if short:module.get_scheduler = scheduler
    try:yield rows
    finally:module.get_scheduler = original


@contextmanager
def actual_call(native,settings,*,short):
    original = native.train_group_vae_payload
    calls = []
    def actual_fit(**kwargs):
        b.require(importlib.import_module('transformers') is module and module.get_scheduler is getter,
            'The actual N local import must still resolve the captured real scheduler module before fitting.')
        b.require(kwargs['batch_size'] == 8192 and len(kwargs['group_refs']) == 36
            and all(ref.name.endswith('.up_proj') and not ref.transpose for ref in kwargs['group_refs']), 'Only the actual original whole-up call is adapted.')
        kwargs['batch_size'] = settings['batch']
        cfg = kwargs['runtime_cfg']
        b.require(cfg.codebook_dim == settings['dim'] and cfg.codebook_bits == settings['bits'], 'Real D/K/B must agree.')
        calls.append(dict(B=kwargs['batch_size'],D=cfg.codebook_dim,K=cfg.codebook_bits,steps=cfg.steps))
        return original(**kwargs)
    native.train_group_vae_payload = actual_fit
    try:
        with original_schedule(short=short) as schedules:
            module = importlib.import_module('transformers');getter = module.get_scheduler
            yield calls,schedules
        b.require(len(calls) == 1 and (not short or len(schedules) == 2), 'One actual whole-up call/two technical schedules are required.')
    finally:native.train_group_vae_payload = original


fit.actual_call = actual_call


def prepare():
    b.cpu_environment()
    b.require(not (b.RESULT / 'prepared.json').exists(), 'Never overwrite repair evidence.')
    old = b.read(ORIGINAL_RESULT / 'prepared.json');b.check_pins(old['source_sha256'])
    source = inspect.getsource(b.recovery.native.train_group_vae_payload)
    b.require('import transformers' in source and 'transformers.get_scheduler(' in source
        and not hasattr(b.recovery.native,'transformers'), 'Bind the actual N local-import call site; do not fabricate a native attribute.')
    proof_path = ORIGINAL_RESULT / 'actual_N_scheduler_repaired.json'
    proof = b.read(proof_path)
    rates = [row['rates'] for row in proof['optimizer_stages']]
    b.require(proof['status'] == 'actual_N_call_completed' and len(proof['hook_rows']) == 2
        and rates == [[.003,.003*(1-1/10000)]]*2 and proof['current_module_and_getter_restored']
        and not proof['CUDA_initialized'], 'Require real N tinyfit with both genuine stages using the original10000 LR prefix and restored module/getter.')
    value = deepcopy(old)
    value.update(source_sha256=b.pins(),runtime_repair=dict(reason='N imports transformers locally inside train_group_vae_payload; use the actual imported module in a scoped context.',
        repair_source=str(Path(__file__).resolve()),repair_sha256=b.sha(__file__),original_CPU_source=str(ORIGINAL_RESULT / 'prepared.json'),
        original_CPU_sha256=b.sha(ORIGINAL_RESULT / 'prepared.json'),scheduler_actual_rates=rates,scheduler_restored=True,
        actual_N_local_import_verified=True,actual_N_fit_CPU_proof=str(proof_path),actual_N_fit_CPU_sha256=b.sha(proof_path),
        actual_N_negative_proof_sha256=b.sha(ORIGINAL_RESULT / 'actual_N_scheduler_diagnostic.json'),
        actual_CPU_repair_source=str(ORIGINAL_RESULT / 'scheduler_repaired_actual_CPU_source.py'),
        actual_CPU_repair_source_sha256=b.sha(ORIGINAL_RESULT / 'scheduler_repaired_actual_CPU_source.py'),
        previous_repair_source=str(ORIGINAL_RESULT / 'runtime_repair/repair_source_first.py'),
        CUDA_initialized=b.torch.cuda.is_initialized(),new_GPU_budget=False))
    b.write(b.RESULT / 'prepared.json',value)
    print(dict(status='repair_CPU_verified',CUDA_initialized=b.torch.cuda.is_initialized(),rates=rates),flush=True)


def raw_full8(arm):
    b.cpu_environment()
    binding,_meta = evaluate.source(arm)
    api = evaluate.b
    config = api.make_config(binding,api.source_pins(binding))
    output = b.RESULT / arm / 'raw_full8'
    b.require(not output.exists(), 'Never repeat or overwrite a full8 endpoint.')
    output.mkdir(parents=True);b.write(output / 'config.json',config)
    record = dict(status='running',checkpoint_id=binding['checkpoint_id'],started_unix=time.time(),stages={})
    try:
        for shard in ('mmlu','seven'):
            argv = api.command('full8','--config',output / 'config.json','--shard',shard,'--output',output / (shard+'.json'))
            with (output / (shard+'.log')).open('x',encoding='utf-8') as stream:
                process = subprocess.Popen(argv,cwd=b.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(b.ARMS[arm]['gpu'])),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row = dict(command=argv,pid=process.pid,gpu=b.ARMS[arm]['gpu'],started_unix=time.time())
                record['stages'][shard] = row;b.write(output / 'run.json',record)
                row['exit_code'] = process.wait()
            row['finished_unix'] = time.time();b.write(output / 'run.json',record)
            b.require(row['exit_code'] == 0, 'Full8 shard failed; no retry or partial score.')
        from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
        full8_summary(config,[output / 'mmlu.json',output / 'seven.json'],output / 'full8_summary.json')
        summary = b.read(output / 'full8_summary.json')
        b.require(summary['leaf_count'] == 64 and summary['sample_count'] == 24742
            and summary['checkpoint_id'] == binding['checkpoint_id'], 'Validate the real written summary, whose API return omits the endpoint UUID.')
        record.update(status='completed_raw_full8',exit_code=0,mean_percent=summary['mean_percent'],leaf_count=summary['leaf_count'],sample_count=summary['sample_count'])
    except BaseException:
        record.update(status='failed',exit_code=1);raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix']);b.write(output / 'run.json',record)


def control_full8_cpu():
    b.cpu_environment()
    from experiments.adaptive_class_evaluation_20261030.summarize import collect_mixed,collect_results
    api = evaluate.b
    directory = ORIGINAL_RESULT / 'control/raw_full8'
    config,summary,run_record = (b.read(directory / name) for name in ('config.json','full8_summary.json','run.json'))
    b.require(run_record['status'] == 'failed' and all(row['exit_code'] == 0 for row in run_record['stages'].values())
        and set(run_record['stages']) == {'mmlu','seven'}, 'Only close the completed two GPU shards whose wrapper failed after writing the summary.')
    api.check_config(config)
    model = config['model'];meta = b.read(Path(model['checkpoint']) / b.checkpoint.META_FILE)
    b.require(meta['checkpoint_id'] == b.CONTRACT['raw_control'] == summary['checkpoint_id']
        and Path(model['checkpoint']).resolve() == b.CONTROL.resolve(), 'Original control full8 endpoint changed.')
    paths = [directory / 'mmlu.json',directory / 'seven.json']
    computed = collect_mixed(paths,api.model_inventory(api.BASE),b.CONTROL,meta,model['checkpoint_files_sha256'])
    baseline_paths,_ = api.template_inputs();baseline = collect_results(baseline_paths,api.model_inventory(api.BASE))
    b.require(computed['versions'] == baseline['versions'] and all(b.read(path)['evals']['lm_eval']['configs'] == b.read(ref)['configs']
        for path,ref in zip(paths,baseline_paths,strict=True)), 'Actual task versions/configurations must match the original BF16 full8 protocol.')
    b.require(summary == dict(checkpoint_id=meta['checkpoint_id'],checkpoint_format=api.FORMAT,**computed),
        'Original written full8 summary must exactly agree with both real shards and all raw sample SHAs.')
    output = b.RESULT / 'control/full8_cpu_closure.json'
    b.require(not output.exists(), 'Never overwrite the original wrapper-failure closure.')
    b.write(output,dict(status='existing_complete_full8_CPU_verified',CUDA_initialized=b.torch.cuda.is_initialized(),
        checkpoint_id=summary['checkpoint_id'],leaf_count=summary['leaf_count'],sample_count=summary['sample_count'],
        mean_percent=summary['mean_percent'],task_metrics=summary['task_metrics'],source_sha256=b.pins(),
        original_wrapper_failure_preserved=str(directory / 'run.json'),repeated_GPU_evaluation=False,
        evidence_sha256={str(path):b.sha(path) for path in [*paths,directory / 'config.json',directory / 'full8_summary.json',directory / 'run.json']},
        raw_sample_sha256=computed['sample_artifacts']['sha256']))
    print(dict(status='existing_complete_full8_CPU_verified',checkpoint_id=summary['checkpoint_id'],leaf_count=64,sample_count=24742),flush=True)


def execute(arm):
    b.cpu_environment()
    prepared = b.read(b.RESULT / 'prepared.json');b.check_pins(prepared['source_sha256'])
    previous = b.read(ORIGINAL_RESULT / arm / 'run.json')
    b.require(previous['status'] == 'failed' and (arm == 'control' or previous['stages']['short']['exit_code'] == 1),
        'Only these recorded failed wrappers may be continued by the explicit repair.')
    deadline, gpu = previous['absolute_deadline_unix'], b.CONTROL_GPU if arm == 'control' else b.ARMS[arm]['gpu']
    output = b.RESULT / arm;output.mkdir(parents=True,exist_ok=True)
    path = output / 'run.json'
    with (output / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not path.exists(), 'No automatic restart or overwrite of a repair run.')
        record = dict(status='running',arm=arm,pid=os.getpid(),start_ticks=mature.proc_start(os.getpid()),
            started_unix=time.time(),absolute_deadline_unix=deadline,budget_anchor_unix=previous['started_unix'],upper_GPU_hours=1.5 if arm == 'control' else 4,
            gpu=gpu,source_sha256=prepared['source_sha256'],runtime_repair=prepared['runtime_repair'],stages={},automatic_retry=False,automatic_recovery_or_full9=False)
        b.write(path,record)
        def stage(name,component,args,cap,*,cpu=False):
            b.check_pins(prepared['source_sha256'])
            seconds = min(cap,int(deadline-time.time())-20)
            b.require(seconds > 0, 'The original shared4h deadline is exhausted.')
            resources = mature.check_resources([str(gpu)])
            argv = ['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',str(b.PYTHON),'-B','-u','-m',
                'experiments.up_runtime_repair_cmcc_20261104','--component',component,*map(str,args)]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),PYTHONHASHSEED='31',PYTHONDONTWRITEBYTECODE='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline')
            log = output / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process = subprocess.Popen(argv,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row = dict(command=argv,pid=process.pid,start_ticks=mature.proc_start(process.pid),process_group_id=process.pid,gpu=gpu,
                    resource_check=resources,started_unix=time.time(),absolute_deadline_unix=deadline,log=str(log))
                record['stages'][name] = row;record['current_stage'] = name;b.write(path,record)
                try:code = process.wait()
                finally:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
            row.update(exit_code=code,finished_unix=time.time());b.write(path,record)
            b.require(code == 0, 'Actual repaired stage failed; no retry or dependent continuation: ' + name)
        try:
            if arm == 'control':
                b.require((b.RESULT / 'control/full8_cpu_closure.json').exists(), 'First close the actual existing control full8 on CPU.')
            else:
                stage('short','run',['--arm',arm,'--phase','short'],1800)
                proof = b.read(b.OUTPUT / arm / 'short/report.json')
                b.require(proof['status'] == 'completed_up_short' and proof['exit_code'] == 0
                    and proof['fit']['technical_original_scheduler'] == [dict(actual_updates=2,schedule_max=10000)]*2,
                    'Require actual whole-M two-stage updates, original schedule and independent fresh before formal.')
                stage('formal','run',['--arm',arm,'--phase','formal'],5400)
                stage('raw_full8','raw_full8',['--arm',arm],5400,cpu=True)
            stage('raw_readout','readout',['--arm',arm],1800)
            record.update(status='completed_up_raw_full8_and_readout',exit_code=0)
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix']);b.write(path,record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('prepare','sequence','run','raw_full8','readout','control_full8_cpu'),required=True)
    parser.add_argument('--arm',choices=('control',*b.ARMS));parser.add_argument('--phase',choices=('short','formal'))
    args = parser.parse_args();b.torch.set_num_threads(8)
    if args.component == 'prepare':prepare()
    elif args.component == 'sequence':execute(args.arm)
    elif args.component == 'run':
        b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
        run.execute(args.arm,args.phase)
    else:
        evaluate.setup()
        if args.component == 'control_full8_cpu':control_full8_cpu()
        elif args.component == 'raw_full8':raw_full8(args.arm)
        else:evaluate.readout(args.arm)

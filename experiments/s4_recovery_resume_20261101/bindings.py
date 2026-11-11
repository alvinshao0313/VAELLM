"""Bind the actual completed terminal and repair only caller import context."""
from pathlib import Path
import os
import subprocess
import sys
import time

from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import bindings as b

HERE=Path(__file__).resolve().parent
OLD=b.OUTPUT/'pair_20261101_1200'
TERMINAL=OLD/'formal/terminal'
TERMINAL_ID='b85c0689-0d16-40a5-84e4-3b4c6a595c81'
PACKAGE='experiments.progressive_residual_bsq_20261030.recovery_compare_20261101'


def environment(gpu):
    env=b.parent.environment(gpu)
    env['PYTHONPATH']=str(b.parent.SAMPLE)+':'+env['PYTHONPATH']
    return env


def own_pins():
    return {str(path):b.sha(path) for path in sorted(HERE.glob('*.py'))}


def retirement(pid,start_ticks):
    path=Path('/proc')/str(pid)/'stat'
    try:actual=int(path.read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return dict(pid=pid,start_ticks=start_ticks,retired=True)
    b.require(actual!=start_ticks, 'The original pipeline process is still active: '+str(pid))
    return dict(pid=pid,start_ticks=start_ticks,retired=True,current_generation=actual)


def old_binding():
    control=b.read(OLD/'run.json');report=b.read(TERMINAL/'report.json');fresh=b.read(TERMINAL/'fresh.json')
    b.require(control['status']=='failed' and control['exit_code']==1 and (OLD/'run.exit').read_text().strip()=='1',
        'Bind the actual failed first comparison; this is one explicit manual continuation.')
    stages={row['name']:row for row in control['stages']}
    b.require(all(stages[name]['exit_code']==0 for name in ('short_terminal','short_terminal_fresh',
        'short_sequential','short_sequential_fresh','formal_terminal','formal_terminal_fresh'))
        and all(stages[name]['exit_code']==1 for name in ('formal_terminal_mmlu','formal_terminal_seven'))
        and 'formal_sequential' not in stages, 'Do not repeat completed work or overwrite an existing sequential arm.')
    for shard in ('mmlu','seven'):
        failure=b.read(TERMINAL/(shard+'_failure.json'))
        b.require("ModuleNotFoundError: No module named 'sample_artifacts'" in failure['error']
            and (TERMINAL/(shard+'.exit')).read_text().strip()=='1'
            and not (TERMINAL/'evaluation'/(shard+'.json')).exists(), 'Only the recorded pre-score import failure is covered.')
    b.require(report['exit_code']==0 and report['planned_updates']==report['training']['actual_updates']==56
        and report['training']['final_per_category_Adam_steps']==dict(zip(b.CATEGORIES,[56,48,40,32,24,16,8]))
        and report['export']['checkpoint_id']==fresh['checkpoint_id']==TERMINAL_ID
        and fresh['status']=='verified_S4_recovery_fresh' and fresh['full_state_exact']
        and report['source_binding']['checkpoint_id']==b.SOURCE_ID,
        'Require the real completed56 terminal and its already verified fresh export.')
    original=b.read(OLD/'plan.json')['source_sha256']
    b.require(len(original)==1062 and b.pins()==original and report['source_sha256']==original,
        'Original1062 pins must remain exactly unchanged.')
    b.require(b.binding(TERMINAL/'final_model/checkpoint_meta.json')==report['export']['metadata']
        and b.parent.s.stat(TERMINAL/'final_model/pytorch_model.bin')==report['export']['weight_stat'],
        'The completed terminal model changed after its fresh proof.')
    observed=[retirement(row['pid'],row['start_ticks']) for row in [control,*control['stages']]]
    preserved=[OLD/'run.json',OLD/'run.exit',OLD/'continuation.exit',TERMINAL/'report.json',TERMINAL/'fresh.json',
        TERMINAL/'mmlu.exit',TERMINAL/'seven.exit',TERMINAL/'mmlu_failure.json',TERMINAL/'seven_failure.json',
        OLD/'formal_terminal_mmlu.log',OLD/'formal_terminal_seven.log']
    return dict(original_control=b.binding(OLD/'run.json'),terminal_report=b.binding(TERMINAL/'report.json'),
        terminal_fresh=b.binding(TERMINAL/'fresh.json'),metadata=report['export']['metadata'],
        weight_stat=report['export']['weight_stat'],checkpoint_id=TERMINAL_ID,
        CPU_validation=b.binding(OLD/'CPU_validation.json'),retired_processes=observed,
        preserved_evidence={str(path):b.sha(path) for path in preserved},original_source_sha256=original)


def available():
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    selected={parts[0].strip():dict(uuid=parts[1].strip(),used_MiB=int(parts[2]))
        for line in inventory.splitlines() if (parts:=line.split(','))[0].strip() in ('6','7')}
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    occupied=[line for line in apps.splitlines() if line.split(',')[0].strip() in {v['uuid'] for v in selected.values()}]
    b.require(len(selected)==2 and not occupied and all(v['used_MiB']<1024 for v in selected.values()),
        'GPU6/7 must be released; no waiting, interference or retry: '+repr(occupied))
    return dict(checked_unix=time.time(),selected=selected,compute_processes=occupied)


def import_proof():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not b.torch.cuda.is_initialized()
        and str(b.parent.SAMPLE) in os.environ.get('PYTHONPATH','').split(':'),
        'Probe the actual caller-corrected CPU import environment.')
    from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import evaluate
    api=evaluate.b.parent.module(b.parent.SAMPLE/'eval_utils_with_samples.py','_S4_recovery_original_hflm')
    import sample_artifacts
    actual=[Path(api.__file__).resolve(),Path(sample_artifacts.__file__).resolve()]
    expected=[b.parent.SAMPLE/'eval_utils_with_samples.py',b.parent.SAMPLE/'sample_artifacts.py']
    pins=b.read(OLD/'plan.json')['source_sha256']
    b.require(actual==expected and all(b.sha(path)==pins[str(path)] for path in actual)
        and callable(api.run_lm_eval) and callable(sample_artifacts.write_lm_eval_samples)
        and not b.torch.cuda.is_initialized(), 'The real unchanged evaluation API/support import context differs.')
    return dict(status='verified_actual_legacy_HFLM_import_context',modules=[b.binding(path) for path in actual],
        PYTHONPATH=os.environ['PYTHONPATH'],cuda_initialized=False,
        numerical_scope='Imports only; no model construction, forward, scoring or new GPU validation.')

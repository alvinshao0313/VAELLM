"""Prepare and execute one E256 original-C0 suffix then complete-nine on GPU1."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from . import sources as s
from experiments.cmcc_runtime_20261030 import sequence as existing
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, TOPIC

ROOT = ARTIFACTS / 'cat_e256_full_cmcc_20261102'
RESULT = TOPIC / ROOT.name
COMMON = ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'


def prepare():
    existing.cpu_environment()
    s.torch.set_num_threads(8)
    s.c.require(not ROOT.exists() and not RESULT.exists(), 'Never overwrite a plan/model/evaluation.')
    binding = s.bind(COMMON)
    cat, _hf, training, _vae, cfgs, runtime = s.recipe(ROOT / 'formal', binding)
    s.c.require(not s.torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    support = s.c.read(SUPPORT / 'prepared.json')
    pins = {**s.source_pins(), **support['source_sha256'], str(SUPPORT / 'prepared.json'): s.c.sha(SUPPORT / 'prepared.json')}
    existing.check_pins(pins)
    formal = dict(name='E256_formal', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.cat_e256_full_cmcc_20261102.run_pipeline', '--common', str(COMMON), '--output', str(ROOT / 'formal')],
        environment=existing.environment('1'), gpus=['1'], output=str(ROOT / 'formal'),
        report=str(ROOT / 'formal/report.json'), log=str(ROOT / 'formal.log'), exit_file=str(ROOT / 'formal.exit'))
    formal['environment'].update(PYTHONHASHSEED='31', CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='offline')
    quality = dict(name='E256_complete_nine', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.cat_e256_full_cmcc_20261102.evaluate', '--support', str(SUPPORT),
        '--checkpoint', str(ROOT / 'formal/final_model'), '--pipeline-report', formal['report'],
        '--output', str(RESULT), '--gpus', '1'], environment=existing.environment(''), gpus=['1'],
        output=str(RESULT), log=str(ROOT / 'evaluation.log'), exit_file=str(ROOT / 'evaluation.exit'))
    plan = dict(schema='same_parent_E256_original_C0_full_chain_v1', output=str(ROOT), created_unix=time.time(),
        source_binding=binding, source_sha256=pins, formal=formal, evaluation=quality,
        cpu_validation=dict(status='verified_real_E256_identity_runtime_and_C0_suffix_protocol',
            CUDA_initialized=False, full_model_reloaded=False, source_raw2=binding['source_raw2']['checkpoint_id'],
            source_after_o=binding['source_after_o']['checkpoint_id'], seed=cat.seed, training_seed=training.seed,
            gate_encoder=cfgs['gate_proj'].base_ch, gate_decoder=cfgs['gate_proj'].decoder_base_ch,
            runtime_identity=runtime['runtime_identity'],
            reused_evidence=binding['evidence_sha256'],
            new_loading_validation='Actual E256 native-to-fixed2 full-state/BF16 independent fresh and CPU/CUDA RNG proof before first recovery update.'),
        budget=dict(reused_predecessor_gate_fit_updates=20000, additional_gate_fit_updates=0,
            recovery_updates_per_category=500, live_suffix_native_stages=2, live_suffix_native_updates_per_stage=10000,
            full8_evaluations=1, LongBench21_evaluations=1, estimated_suffix_training_hours=[16, 20]),
        automatic_retry=False, automatic_resume=False, gpu='1',
        boundary='One completed E256 raw2 with own live C0 suffix. No old-parent suffix, Rice, extra LoRA, grid, retry or reservation.')
    ROOT.mkdir(parents=True)
    s.c.write(ROOT / 'plan.json', plan)
    print(dict(status='prepared_same_parent_E256_full_chain', plan=str(ROOT / 'plan.json'),
        source_raw2=binding['source_raw2']['checkpoint_id'], control=binding['control_checkpoint_id'], pins=len(pins)), flush=True)


def run():
    existing.cpu_environment()
    plan_path = ROOT / 'plan.json'
    plan = s.c.read(plan_path)
    s.c.require(plan['schema'] == 'same_parent_E256_original_C0_full_chain_v1', 'Use this exact one-candidate plan.')
    existing.check_pins(plan['source_sha256'])
    record_path = ROOT / 'run.json'
    s.c.require(not record_path.exists(), 'The fixed chain has already run; never retry automatically.')
    with (ROOT / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.c.require(not record_path.exists(), 'The fixed chain already started before lock acquisition.')
        record = dict(status='validating_source', pid=os.getpid(), start_ticks=existing.proc_start(os.getpid()),
            started_unix=time.time(), command=[sys.executable, *sys.argv], plan_sha256=s.c.sha(plan_path), stages={})
        s.c.write(record_path, record)
        try:
            s.c.require(s.bind(COMMON) == plan['source_binding'], 'The reviewed source or completed control changed.')

            def changed(stage):
                record['stages'][stage['name']] = stage
                record['status'] = stage['name'] + '_' + stage['status']
                s.c.write(record_path, record)

            record['formal_resource_check'] = existing.check_resources(['1'])
            s.c.write(record_path, record)
            existing.launch_child(plan['formal'], changed)
            from .evaluate import setup
            from experiments.adaptive_class_evaluation_20261030 import bindings as b
            setup(SUPPORT)
            model, _meta = b.source(ROOT / 'formal/final_model', ROOT / 'formal/report.json')
            record['final_binding'] = model
            record['evaluation_resource_check'] = existing.check_resources(['1'])
            s.c.write(record_path, record)
            existing.launch_child(plan['evaluation'], changed)
            record['complete_nine'] = existing.validate_evaluation(RESULT, model['checkpoint_id'])
            existing.check_pins(plan['source_sha256'])
            record.update(status='completed_E256_and_complete_nine', exit_code=0)
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            s.c.write(record_path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()

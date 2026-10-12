"""Recover only failed endpoint observation/reporting; never construct a VAE or optimizer."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed
from .data import ARMS, OUTPUT, ROOT, SOURCE_ID, TARGET, assets, check, read
from .observe import evaluate

RECOVERY = OUTPUT / 'observation_recovery'


def saved_endpoints():
    states, proofs, paths = {}, {}, []
    for arm in ARMS:
        directory = OUTPUT / 'run' / arm
        proof_path = OUTPUT / 'run' / (arm + '.json')
        proof, meta = read(proof_path), read(directory / 'checkpoint_meta.json')
        require(proof['fit']['arm'] == arm and [s['updates'] for s in proof['fit']['stages']] == [256, 256]
                and proof['save']['native13_roundtrip_exact'] and proof['save']['same_kernel_bf16_decode_roundtrip_exact']
                and meta['checkpoint_kind'] == 'final_model' and meta['compressed_targets'] == [TARGET]
                and meta['lora_config'] is None and len(meta['converted_modules']) == 1, 'Incomplete actual saved endpoint.')
        payload = directory / meta['state_dict_file']
        raw = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
        prefix = TARGET + '.'
        require(len(raw) == 13 and all(k.startswith(prefix) for k in raw), 'Standalone endpoint has extra/missing tensors.')
        states[arm] = {k[len(prefix):]: v for k, v in raw.items()}
        require({k: tensor_record(v) for k, v in states[arm].items()} == proof['save']['state'], 'Saved13 tensor hashes differ from actual training/export proof.')
        proofs[arm] = {'checkpoint_id': meta['checkpoint_id'], 'fit_and_save': str(proof_path),
                       'fit_and_save_sha256': sha256(proof_path), 'state': proof['save']['state']}
        paths += [proof_path, directory / 'checkpoint_meta.json', payload]
    a, b = [read(OUTPUT / 'run' / (arm + '.json'))['fit'] for arm in ARMS]
    require(all(x['initial_state_sha256'] == y['initial_state_sha256'] and x['token_order'] == y['token_order']
                for x, y in zip(a['stages'], b['stages'], strict=True)), 'Actual training pair was not matched.')
    return states, proofs, paths


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not RECOVERY.exists(), 'Fresh CPU-only recovery preparation required.')
    torch.set_num_threads(4)
    old = read(OUTPUT / 'launch.prepared.json'); failed = read(OUTPUT / 'launch.json'); process = read(OUTPUT / 'process.json')
    require(failed['status'] == 'failed' and failed['exit_code'] == 1
            and (OUTPUT / 'run.exit').read_text().strip() == (OUTPUT / 'outer.exit').read_text().strip() == '1'
            and not Path('/proc', str(failed['worker_pid'])).exists() and not Path('/proc', str(process['pid'])).exists(),
            'Original execution must have ended before recovery.')
    states, proofs, paths = saved_endpoints()
    record = deepcopy(old)
    observe = Path(__file__).with_name('observe.py').resolve(); repair = read(OUTPUT / 'observation_repair.json')
    require(record['source_sha256'][str(observe)] == repair['original_source_sha256']
            and sha256(observe) == repair['corrected_source_sha256'], 'Unexpected repair source.')
    record['source_sha256'][str(observe)] = sha256(observe)
    record['source_sha256'][str(Path(__file__).resolve())] = sha256(Path(__file__))
    paths += [OUTPUT / p for p in ('launch.prepared.json', 'launch.json', 'process.json', 'run.exit', 'outer.exit',
                                  'run.log', 'observation_repair.patch', 'observation_repair.json')]
    record['input_sha256'].update({str(p): sha256(p) for p in paths})
    record.update(status='prepared_unbound', gpu=None, output_dir=str(RECOVERY), original_failed_launch=str(OUTPUT / 'launch.json'),
        endpoint_proofs=proofs, timeout_seconds=900, termination_grace_seconds=20,
        recovery_scope='No training, endpoint selection, or algorithm changes. Reload two actual saved local13 endpoints, replay all original fixed observations and complete CPU native budget report. Original launch remains failed.',
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.rotated_output_pair.observation_recovery', '--launch-record', str(RECOVERY / 'launch.json')])
    require(not torch.cuda.is_initialized(), 'CPU recovery preparation initialized CUDA.')
    check(record); RECOVERY.mkdir()
    write_json(RECOVERY / 'launch.prepared.json', record)
    print({'prepared': str(RECOVERY / 'launch.prepared.json'), 'sha256': sha256(RECOVERY / 'launch.prepared.json'),
           'sources': len(record['source_sha256']), 'inputs': len(record['input_sha256']), 'cpu': True}, flush=True)


def run(path):
    record = read(path); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and Path.cwd() == ROOT
            and sys.executable == record['python_executable'] and Path(record['output_dir']) == RECOVERY
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']) and not (RECOVERY / 'report.json').exists()
            and not record['queued'] and not record['automatic_retry']
            and all(os.environ.get(k) == v for k, v in record['environment'].items()), 'Explicitly bound one-shot recovery required.')
    check(record)
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record)
    try:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One assigned CUDA device required.')
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
        endpoints, proofs, _ = saved_endpoints(); require(proofs == record['endpoint_proofs'], 'Endpoint evidence changed.')
        data = assets()
        require({d: {k: tensor_record(v) for k, v in b.items()} for d, b in data['batches'].items()}
                == record['token_batches'] and data['spec'] == record['native_spec'], 'Original fixed observation contract differs.')
        observation = evaluate(data, endpoints, record['source_native_policy'], deadline=record['started_unix'] + record['timeout_seconds'])
        require(observation['budget']['user_linear_bits'] == record['source_bit_budget']['user_linear_bits']
                and observation['budget']['components'] == record['source_bit_budget']['components'], 'Native recovered storage ledger differs.')
        check(record)
        report = {'status': 'completed_observation_recovery', 'source_checkpoint_id': SOURCE_ID,
            'original_failed_launch': record['original_failed_launch'], 'endpoint_proofs': proofs,
            'protocol': record['protocol'], 'recovery_scope': record['recovery_scope'], 'observation': observation,
            'elapsed_seconds': time.time() - record['started_unix'], 'training_steps_executed': 0,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        write_json(RECOVERY / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(RECOVERY / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=repr(error)); traceback.print_exc(); raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (RECOVERY / 'run.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    require(args.prepare != (args.launch_record is not None), 'Choose CPU prepare or bound recovery.')
    prepare() if args.prepare else run(args.launch_record)

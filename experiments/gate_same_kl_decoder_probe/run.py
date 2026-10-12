"""Bind three fresh fixed-code gate arms to a two-update final-mass-KL mechanism."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.gate_final_kl_accept_probe.run import BASE, check, read
from experiments.gate_kl_calibration_evaluation.prepare import EXPECTED_XOR, replay_endpoint
from experiments.gate_same_kl_decoder_probe.core import ARMS, HORIZON, STEPS, TOKENS, probe
from experiments.gate_student_context_probe.core import BANKS
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID, TARGET, prepare_data
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def source_states(source, arms):
    meta = read(Path(source['source_checkpoint']) / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh original f927 supplies decoder values.')
    full = torch.load(Path(source['source_checkpoint']) / meta['state_dict_file'],
                      map_location='cpu', weights_only=True, mmap=True)
    original = {key[len(TARGET) + 1:]: value for key, value in full.items() if key.startswith(TARGET + '.')}
    require({key: tensor_record(value) for key, value in original.items()} == source['expected_gate_state']
            and len(original) == 13, 'Fresh source gate differs from the native inventory.')
    states = {'original': original}
    for arm in ARMS[1:]:
        item = arms[arm]
        require(sha256(item['choices_path']) == item['choices_sha256'], 'Prepared compact choices changed.')
        choices = torch.load(item['choices_path'], map_location='cpu', weights_only=True)
        states[arm] = replay_endpoint(full, choices, arm)
        actual = {key: tensor_record(value) for key, value in states[arm].items()}
        require(actual == item['initial_state']
                and all(actual[key] == source['expected_gate_state'][key] for key in original if key not in BANKS),
                'Only the two reviewed banks may differ; all six decoder initial values remain original.')
    return states


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare a fresh CPU-only output in the public root.')
    torch.set_num_threads(4)
    source, batches, _ = prepare_data(BASE)
    batch = batches['train']; data = {'source_data': source}
    tokens = {key: tensor_record(value) for key, value in batch.items()}
    require(tuple(batch['input_ids'].shape) == (32, 256) and int(batch['attention_mask'][:, 1:].sum()) == TOKENS
            and tokens == source['token_batches']['train'], 'Use exactly the old32 training documents, not heldout or FineWeb.')
    calibration, gradient = BASE / 'gate_kl_calibration_probe', BASE / 'gate_final_kl_gradient_probe'
    evidence = []
    proofs = {}
    for label, directory in (('calibration128', calibration), ('leaf_gradient', gradient)):
        report, review, terminal = (read(directory / name) for name in ('run/report.json', 'review.json', 'launch.json'))
        prepared = read(directory / 'launch.prepared.json')
        require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
                and review['status'] == 'verified' and review['source_checkpoint_id'] == report['source_checkpoint_id'] == SOURCE_ID
                and review['report_sha256'] == terminal['report_sha256'] == sha256(directory / 'run/report.json')
                and review['prepared_sha256'] == sha256(directory / 'launch.prepared.json')
                and all((directory / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit')),
                'Require the actual reviewed proposal/backward evidence: ' + label)
        proofs[label] = {'report': report, 'prepared': prepared}
        evidence += [directory / name for name in ('run/report.json', 'review.json', 'launch.json',
                     'launch.prepared.json', 'run.exit', 'outer.exit')]
    proposal = proofs['calibration128']['report']; experiment = proposal['experiment']
    old = proofs['leaf_gradient']['prepared']
    require(source == old['data']['source_data'] and tokens == old['token_batch']
            and experiment['initial_state'] == source['expected_gate_state'] and experiment['all13_final_bytes_exact'],
            'Sources, original decoder values or the original train32 changed.')
    arms = {}
    for arm in ARMS[1:]:
        saved, endpoint = proposal['saved_choices'][arm], experiment['endpoint_records'][arm]
        path = calibration / 'run' / (arm + '_choices.pt')
        require(str(path) == saved['path'] and sha256(path) == saved['sha256'] and saved['roundtrip_exact']
                and endpoint['stage_xor'] == EXPECTED_XOR[arm] and endpoint['complete_bank_replay_exact']
                and endpoint['local11_noncode_bytes_exact']
                and endpoint['endpoint_state'] == experiment['generations'][arm]['applied_state']['endpoint_state'],
                'Reviewed fixed-code proposal identity differs.')
        arms[arm] = {'choices_path': str(path), 'choices_sha256': saved['sha256'],
                     'initial_state': endpoint['endpoint_state'], 'stage_xor': EXPECTED_XOR[arm]}
        evidence.append(path)
    states = source_states(source, arms)
    require(tuple(states) == ARMS, 'Unexpected arm order.')
    sources, inputs = dict(source['source_sha256']), dict(source['input_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                sources[str(path)] = sha256(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for proof in proofs.values():
        for path, digest in sources.items():
            if path in proof['prepared']['source_sha256']:
                require(digest == proof['prepared']['source_sha256'][path], 'Applicable native source changed: ' + path)
    inputs.update({str(path): sha256(path) for path in evidence})
    record = {'schema': 'gate_same_final_mass_kl_decoder_two_updates_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'source_checkpoint_id': SOURCE_ID, 'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': [source['source_weight_stat'], *source['hf_weight_files']],
        'data': data, 'token_batch': tokens, 'arms': arms, 'arm_order': list(ARMS),
        'native_gradient_report': str(gradient / 'run/report.json'), 'calibration_report': str(calibration / 'run/report.json'),
        'source_bit_budget': old['source_bit_budget'], 'environment': old['environment'],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'protocol': {'arms': list(ARMS), 'updates_per_arm': STEPS, 'schedule_steps': HORIZON,
            'initialization': 'Independent fresh f927 six BF16 decoder tensors for each arm; original or reviewed128 compact code choices. No prior decoder500 values.',
            'population': 'Original train32 only,4257 causal tokens, fixed256/micro1. No heldout or FineWeb selection/observation.',
            'objective': 'Same full-model native top100-plus-tail KL atT1 as proposal generation; token sums followed by one4257 normalization, not conditional MLP loss.',
            'updates': 'Six FP32 decoder masters/Adam; BF16 native live packed graph. LR1e-5, betas(.9,.95),eps1e-8,wd0,linear500,warm0; two updates only.',
            'cache': 'Other251 deployed projections remain frozen; selected decoder uses its native trainable packed path. Report training/deployment baselines separately.',
            'teacher': 'One original HF train32 pass, CPU top100 statistics retained; release teacher before loading the student.',
            'save': 'One native BF16 thirteen-tensor state per arm with fixed two banks/seven local tensors, strict local reload and same-path BF16 fullweight comparison.',
            'boundary': 'New full-model decoder-backward mechanism/resource check, not a500-step fit, full8, LongBench or framework integration.'},
        'resource_estimate': {'boundary': 'Actual two-update decoder-backward memory/timing are unmeasured.',
            'prior_cached_weight_gradient32_seconds': 7.3, 'backward_documents': 192,
            'wall_budget_seconds': 900, 'grace_seconds': 20},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_same_kl_decoder_probe.run',
                          '--launch-record', str(output / 'launch.json')], 'cuda_initialized': False}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'launch_sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'stats': len(record['protected_stats']), 'cuda_initialized': False}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Use one explicitly bound fresh execution without retry.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        source, batches, archive = prepare_data(BASE); data = {'source_data': source}; batch = batches['train']
        require(data == record['data'] and {k: tensor_record(v) for k, v in batch.items()} == record['token_batch'], 'Runtime train32 changed.')
        arm_states = source_states(source, record['arms'])
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batch}, archive, device=torch.device('cuda:0'),
            deadline=deadline, observer=lambda *args: None)
        del hf_mlp, archive, batches
        gc.collect()
        require(teacher_report['documents'] == read(record['native_gradient_report'])['teacher_train']['documents'],
                'Actual train32 teacher statistics differ from the native leaf-gradient evidence.')
        teacher_path = work / 'teacher_top100.pt'; torch.save(teacher, teacher_path)
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(arm, row):
                stream.write(json.dumps({'arm': arm, **row}) + '\n'); stream.flush()
                print(json.dumps({'arm': arm, 'step': row['step'], 'kl_mean': row['kl_mean'], 'seconds': row['seconds']}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                result = probe(data, batch, teacher['train'], arm_states, work=work, deadline=deadline, observer=observer)
        require(result['status'] == 'completed' and tuple(result['arms']) == ARMS, 'All three independent mechanism arms must finish.')
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'protocol': record['protocol'], 'teacher_train': teacher_report, 'teacher_cache_sha256': sha256(teacher_path),
            'experiment': result, 'trajectory_sha256': sha256(work / 'trajectory.jsonl'),
            'elapsed_seconds': time.time() - record['started_unix'], 'full8_and_longbench': 'not_evaluated'}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path); mode.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    else:
        sys.exit(run(args.launch_record.resolve()))

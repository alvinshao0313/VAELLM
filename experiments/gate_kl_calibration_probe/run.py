"""CPU bind or execute the two128-document fixed-four-group calibration contrasts."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.gate_final_kl_accept_probe.run import BASE, check, read, validate_teacher_cache
from experiments.gate_final_kl_neighbor_transfer.core import choice_inventory
from experiments.gate_kl_calibration_probe.core import ARMS, OBSERVATIONS, POPULATIONS, probe
from experiments.gate_kl_calibration_probe.data import prepare_data
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def teacher_chunks(batches):
    chunks = {f'a{i}': {key: value[i * 32:(i + 1) * 32] for key, value in batches[ARMS[0]].items()} for i in range(4)}
    chunks.update({f'f{i}': {key: value[64 + i * 32:96 + i * 32] for key, value in batches[ARMS[1]].items()} for i in range(2)})
    chunks.update(a_new=batches['alpaca_confirmation32'], f_new=batches['fineweb_rows500032_500063'])
    require(len(chunks) == 8 and all(tuple(part['input_ids'].shape) == (32, 256) for part in chunks.values())
            and all(torch.equal(batches[ARMS[0]][key][:64], batches[ARMS[1]][key][:64])
                    and torch.equal(batches['train32'][key], chunks['a0'][key]) for key in batches['train32']),
            'Real256 unique teacher documents must preserve the two128 population orders.')
    for arm, names in ((ARMS[0], ('a0', 'a1', 'a2', 'a3')), (ARMS[1], ('a0', 'a1', 'f0', 'f1'))):
        require(all(torch.equal(torch.cat([chunks[name][key] for name in names]), value)
                    for key, value in batches[arm].items()), 'Teacher chunk layout omitted or reordered calibration documents.')
    return chunks


def teacher_populations(chunks, old, batches):
    populations = {ARMS[0]: sum((chunks[k] for k in ('a0', 'a1', 'a2', 'a3')), []),
                   ARMS[1]: sum((chunks[k] for k in ('a0', 'a1', 'f0', 'f1')), []),
                   'train32': chunks['a0'], 'alpaca_held32': old['alpaca_held32'],
                   'fineweb_rows500000_500031': old['fineweb_rows500000_500031'],
                   'alpaca_confirmation32': chunks['a_new'], 'fineweb_rows500032_500063': chunks['f_new']}
    for name, population in populations.items():
        batch = batches[name]
        require(len(population) == batch['input_ids'].shape[0], 'Teacher population length differs.')
        for i, item in enumerate(population):
            valid = batch['attention_mask'][i, 1:].bool()
            require(torch.equal(batch['input_ids'][i, 1:][valid], item['labels']), 'Actual teacher labels differ from native causal tokens.')
    return populations


def completed(directory):
    report, review, terminal = (read(directory / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'verified' and review['report_sha256'] == sha256(directory / 'run/report.json')
            and terminal['report_sha256'] == sha256(directory / 'run/report.json')
            and all((directory / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and report['source_checkpoint_id'] == SOURCE_ID, 'Require the actual completed same-source proof.')
    return report


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use a new CPU-only preparation directory.')
    torch.set_num_threads(4)
    neighbor, transfer = BASE / 'gate_final_kl_neighbor_probe', BASE / 'gate_final_kl_neighbor_transfer'
    reference, historical = completed(neighbor), completed(transfer)
    old = read(transfer / 'launch.prepared.json'); check(old)
    data, batches, _ = prepare_data()
    require(data['old_data'] == old['data'] and tuple(batches) == POPULATIONS
            and [int(batches[arm]['attention_mask'][:, 1:].sum()) for arm in ARMS] == [18582, 25718],
            'Use the predeclared actual text selections and128-document token budgets.')
    chunks = teacher_chunks(batches)
    choices_path = Path(reference['saved_choices']['path'])
    choices = torch.load(choices_path, map_location='cpu', weights_only=True)
    require(sha256(choices_path) == reference['saved_choices']['sha256']
            and tensor_record(choices) == reference['saved_choices']['tensor']
            and choice_inventory(choices) == historical['transfer']['choice_inventory'], 'Historical32 control choices changed.')
    old_teacher = torch.load(old['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
    validate_teacher_cache(old_teacher, batches, read(old['old_teacher_report']))
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for incoming, target in ((data['source_sha256'], sources), (data['input_sha256'], inputs)):
        require(all(key not in target or target[key] == value for key, value in incoming.items()), 'Inherited evidence versions conflict.')
        target.update(incoming)
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result') and path.suffix == '.py':
                digest = sha256(path)
                require(str(path) not in sources or sources[str(path)] == digest, 'A loaded existing source drifted.')
                sources[str(path)] = digest
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for directory in (neighbor, transfer):
        for name in ('launch.prepared.json', 'run/report.json', 'review.json', 'run.exit', 'outer.exit'):
            path = directory / name; inputs[str(path)] = sha256(path)
    # Negative384 quality motivates this bounded contrast; its weight is not an input.
    for directory in (BASE / 'gate_final_kl_neighbor_accept_probe', BASE / 'gate_final_kl_neighbor_accept_full8'):
        for name in ('review.json', 'run/report.json' if directory.name.endswith('_probe') else 'full8_summary.json'):
            path = directory / name; inputs[str(path)] = sha256(path)
    inputs[str(choices_path)] = sha256(choices_path)
    record = {'schema': 'gate_kl_calibration_fixed4_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': old['protected_stats'],
        'data': data, 'token_batches': {name: {key: tensor_record(v) for key, v in batch.items()} for name, batch in batches.items()},
        'teacher_chunks': {name: {key: tensor_record(v) for key, v in batch.items()} for name, batch in chunks.items()},
        'source_bit_budget': old['source_bit_budget'], 'reference_choices_path': str(choices_path),
        'reference_report': str(neighbor / 'run/report.json'), 'historical_transfer_report': str(transfer / 'run/report.json'),
        'old_teacher_path': old['old_teacher_path'], 'old_teacher_report': old['old_teacher_report'],
        'environment': old['environment'], 'timeout_seconds': 900, 'termination_grace_seconds': 20,
        'queued': False, 'automatic_retry': False, 'cuda_initialized': False,
        'protocol': {'arms': list(ARMS), 'groups': [0, 1, 2, 3], 'fresh': 'Both initial derivatives use original f927; no persistent model parameter trains.',
            'gradient': 'Exactly128 documents per arm, native micro1/padded256/BF16 cached leaf; per-token FP32 massKL T1, per-doc FP64 sum, returnedBF16 gradients added sequentially inFP32. No source equalization or document mean.',
            'proposal': 'Reuse immutable16128-row enumeration:64 true BF16 stage-sum singlebit neighbors, FP64 rotated-gradient dot, original0 priority and strict negative best; at most one bit per vector.',
            'endpoint': 'Apply the four preselected groups together; never greedy acceptance, reordering, stopping, decoder fitting or observer-driven proposal selection.',
            'observations': 'Old three populations remain development comparisons; newly fixedAlpaca32/FineWeb32 are measured once per final arm, after both proposals are fixed. No benchmark task is read for calibration.',
            'controls': 'Freshf927 and the immutable historicalAlpaca32 four-group proposal; remeasure on newly added populations, compare old three-domain raw scores without assuming cross-run floating differences are errors.',
            'numerics': 'Rotated-coordinate scores are first-order proposal proxies; report nativeBF16 original-coordinate dot and true finalKL separately. Multiple vector changes need not add after inverse rotation and rounding.',
            'save': 'Only compact choices for two arms, small trajectory/report and teacher statistics; no model/decoder/large gradient saved.',
            'interpretation': 'A128 versusA32 tests coverage; mixed128 versusA128 changes source composition and token count. Does not uniquely diagnose overfitting, gradient noise or numerical curvature; no main-framework integration or downstream acceptance.'},
        'resource_estimate': {'measured32_gradient_seconds': reference['diagnostic']['gradient']['seconds'],
            'linear128_gradient_seconds_per_arm': 4 * reference['diagnostic']['gradient']['seconds'],
            'planned_two_arm_total_seconds': 900, 'note': 'One teacher and one frozen student load;128docs do not retain128 autograd graphs. Original13/cache/full logits are restored between arms and their gradients are released. Longer valid text can increase loss cost; estimate is not a runtime/fit guarantee.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_kl_calibration_probe.run', '--launch-record', str(output / 'launch.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'sources': len(sources), 'inputs': len(inputs),
                      'cuda_initialized': False, 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Require one fresh explicitly bound execution.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = prepare_data()
        require(data == record['data'] and {n: {k: tensor_record(v) for k, v in b.items()} for n, b in batches.items()}
                == record['token_batches'], 'Runtime data identity differs.')
        chunk_teacher, hf_mlp, teacher_report = teacher_pass(data, teacher_chunks(batches), archive,
            device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        old_teacher = torch.load(record['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(old_teacher, batches, read(record['old_teacher_report']))
        teacher = teacher_populations(chunk_teacher, old_teacher, batches)
        reference = read(record['reference_report'])
        teacher_prefix_exact = teacher_report['documents']['a0'] == reference['teacher_train']['documents']['train']
        historical = read(record['historical_transfer_report'])['transfer']
        control_choices = torch.load(record['reference_choices_path'], map_location='cpu', weights_only=True)
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(kind, row):
                stream.write(json.dumps({'kind': kind, **row}) + '\n'); stream.flush()
                print(json.dumps({'kind': kind, **{k: row[k] for k in ('arm', 'index', 'population', 'kl_mean', 'stage_xor') if k in row}}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                choices, result = probe(data, batches, teacher, control_choices, historical, deadline=deadline, observer=observer)
        saved = {}
        for arm, value in choices.items():
            destination = work / (arm + '_choices.pt'); torch.save(value, destination)
            loaded = torch.load(destination, map_location='cpu', weights_only=True)
            require(torch.equal(value, loaded) and tensor_record(loaded) == result['generations'][arm]['choices'], 'Saved compact choices differ.')
            saved[arm] = {'path': str(destination), 'sha256': sha256(destination), 'tensor': tensor_record(loaded),
                          'inventory': choice_inventory(loaded), 'roundtrip_exact': True}
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'protocol': record['protocol'], 'teacher': teacher_report, 'teacher_old_train32_statistics_exact': teacher_prefix_exact,
            'experiment': result, 'saved_choices': saved, 'trajectory_sha256': sha256(work / 'trajectory.jsonl'),
            'elapsed_seconds': time.time() - record['started_unix'], 'saved_model_or_gradient': False,
            'full8_and_longbench': 'not_evaluated'}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path); group.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare.resolve())
    else: sys.exit(run(args.launch_record.resolve()))

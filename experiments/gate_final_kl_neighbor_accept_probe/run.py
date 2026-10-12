"""Prepare/exercise all384 initial-gradient proposals with native final-KL acceptance."""
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

from experiments.gate_final_kl_accept_probe.core import search
from experiments.gate_final_kl_accept_probe.run import BASE, check, data_and_inputs, read, validate_teacher_cache
from experiments.gate_final_kl_neighbor_accept_probe.core import CHUNK, TOTAL, generate
from experiments.gate_group_accept_probe.core import BANKS, GROUPS, VECTORS_PER_GROUP, group_inventory
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def completed(directory):
    report, review, terminal = (read(directory / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'verified' and review['report_sha256'] == sha256(directory / 'run/report.json')
            and terminal['report_sha256'] == sha256(directory / 'run/report.json')
            and all((directory / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and report['source_checkpoint_id'] == SOURCE_ID, 'Require complete same-source evidence.')
    return report


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    neighbor = BASE / 'gate_final_kl_neighbor_probe'; transfer = BASE / 'gate_final_kl_neighbor_transfer'
    historical = BASE / 'gate_final_kl_accept_probe/full384'
    old = read(transfer / 'launch.prepared.json'); check(old)
    reference, observed, old_search = (completed(p) for p in (neighbor, transfer, historical))
    require(observed['transfer']['choices_complete_bank_replay_exact'] and observed['transfer']['all13_final_bytes_exact']
            and old_search['search']['mask_replay_exact'] and old_search['search']['strict_final13_and_group8_weight_reload_exact'],
            'Reuse validated proposal, transfer and full384 native acceptance/storage paths.')
    data, batches, _ = data_and_inputs()
    token_batches = {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
    require(data == old['data'] and token_batches == old['token_batches'], 'Keep all original source/data semantics.')
    choices_path = Path(reference['saved_choices']['path'])
    reference_choices = torch.load(choices_path, map_location='cpu', weights_only=True)
    require(reference_choices.dtype == torch.uint8 and tuple(reference_choices.shape) == (CHUNK,)
            and tensor_record(reference_choices) == reference['saved_choices']['tensor']
            and sha256(choices_path) == reference['saved_choices']['sha256'], 'Use the four-group choices only for a reference observation, never as the384 proposals.')
    old_teacher = torch.load(old['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
    validate_teacher_cache(old_teacher, batches, read(old['old_teacher_report']))
    checkpoint = Path(data['source_data']['source_checkpoint']); meta = read(checkpoint / 'checkpoint_meta.json')
    spec = next(x for x in meta['converted_modules'] if x['name'] == TARGET)
    state = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require(TOTAL == 1548288 and CHUNK == 16128 and TOTAL == 96 * CHUNK
            and spec['weight_rotation']['block_size'] == 32
            and all(state[TARGET + '.' + bank].dtype == torch.uint8 and tuple(state[TARGET + '.' + bank].shape) == (TOTAL, 1, 4) for bank in BANKS),
            'Actual native banks must partition into96 equal chunks matching the verified four-group decoder shape.')
    chunks = [(start, start + CHUNK) for start in range(0, TOTAL, CHUNK)]
    require(len(chunks) == 96 and chunks[0][0] == 0 and chunks[-1][1] == TOTAL
            and all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])), 'Complete vector coverage must have no gaps or overlaps.')
    trajectory = old_search['search']['trajectory']; max_group = max(row['seconds'] for row in trajectory)
    overhead = old_search['elapsed_seconds'] - sum(row['seconds'] for row in trajectory)
    enumeration_extra = 95 * reference['diagnostic']['enumeration']['seconds']
    estimate = GROUPS * max_group + overhead + reference['elapsed_seconds'] + enumeration_extra
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for directory in (neighbor, transfer, historical):
        for name in ('run/report.json', 'review.json', 'run.exit', 'outer.exit'):
            path = directory / name; inputs[str(path)] = sha256(path)
    record = {'schema': 'gate_final_kl_neighbor_accept_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': old['protected_stats'],
        'data': data, 'token_batches': token_batches, 'source_bit_budget': old['source_bit_budget'],
        'reference_report': str(neighbor / 'run/report.json'), 'reference_choices_path': str(choices_path),
        'old_teacher_path': old['old_teacher_path'], 'old_teacher_report': old['old_teacher_report'],
        'historical_search_report': str(historical / 'run/report.json'),
        'proposal_source': 'Fresh original final-model train32 mass-KL gradient computed once in this run; no previous endpoint or HF-weight-MSE proposals.',
        'generated_choices_output': str(output / 'run/choices.pt'), 'groups': GROUPS, 'chunk_vectors': CHUNK,
        'cpu_geometry': {'actual_banks': {bank: {'dtype': str(state[TARGET + '.' + bank].dtype), 'shape': list(state[TARGET + '.' + bank].shape)} for bank in BANKS},
                         'vectors': TOTAL, 'chunks': 96, 'equal_shape_to_verified_four_groups': True, 'coverage_exact': True},
        'time_estimate': {'previous384_elapsed_seconds': old_search['elapsed_seconds'], 'previous_max_group_seconds': max_group,
            'previous_non_group_seconds': overhead, 'neighbor4_elapsed_seconds': reference['elapsed_seconds'],
            'estimated_other95_chunks_seconds': enumeration_extra, 'estimated_total_seconds': estimate,
            'planned_seconds': 2700, 'boundary': 'Conservative same-speed extrapolation, including reject re-prime and an extra fresh student load. Not a guarantee or a reason to truncate, retry or change the protocol.'},
        'environment': old['environment'], 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'queued': False, 'automatic_retry': False, 'cuda_initialized': False,
        'protocol': {'initial': 'Fresh original f927; once-only native BF16 cached L18 leaf gradient over all original train32/4257causal tokens; FP32 accumulation. Persistent parameters stay frozen.',
            'coordinates': 'Exclude protected64 input columns in ascending original order, apply actual FP32 forward two-sided block32 RHT to the original-coordinate gradient. No transpose or repeated normalization/scale fusion.',
            'proposals': 'All1548288 vectors,96chunks of16128 rows,64 real BF16 stage-sum single-bit neighbors, FP64dot, original0 priority/strict negative best/fixed candidate order/at most one bit per vector.',
            'independence': 'All proposals use the same initial gradient before any bit mutation. Then release that model/gradient and load fresh f927 for acceptance; no gradient recomputation or four-group-updated starting state.',
            'acceptance': 'Reuse immutable native search: groups0..383, complete train32 final top100+tail KL T1, FP32token losses/fixed-order FP64 sums; strict decrease accepts, otherwise exact bank restore and nativegroup8 rebuild.',
            'observations': 'Old Alpaca held32 and fixed FineWeb500000..500031 are only endpoint observations, never proposal choice/acceptance/reordering/early stopping.',
            'save': 'Compact choices/accepted mask/effective choices, per-group evidence and one13-tensor gate endpoint for native strict save/reload evidence; no full model, optimizer or large gradient.',
            'comparison': 'Record first-chunk choices/proxy and derivative-statistic differences from the separate four-group run; independent floating-point differences are not assumed exact. Packed replay and frozen state remain exact.',
            'boundary': 'New isolated algorithm prototype, not a core patch or full8/LongBench acceptance. Historical evidence inputs validate mechanisms; their old HF-MSE choices are never supplied to this search.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_final_kl_neighbor_accept_probe.run', '--launch-record', str(output / 'launch.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'source_count': len(sources), 'input_count': len(inputs),
        'estimated_seconds': estimate, 'cuda_initialized': False, 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Require a fresh explicit one-shot binding.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
                  worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = data_and_inputs()
        require(data == record['data'] and {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
                == record['token_batches'], 'Runtime source or fixed tokens changed.')
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batches['train']}, archive, device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        reference = read(record['reference_report'])
        require(teacher_report['documents'] == reference['teacher_train']['documents'], 'Actual original train teacher probabilities or labels changed.')
        old_teacher = torch.load(record['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(old_teacher, batches, read(record['old_teacher_report'])); teacher.update(old_teacher)
        reference_choices = torch.load(record['reference_choices_path'], map_location='cpu', weights_only=True)
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(kind, row):
                stream.write(json.dumps({'kind': kind, **row}) + '\n'); stream.flush()
                print(json.dumps({'kind': kind, **{k: row[k] for k in ('index', 'chunk', 'group', 'accepted', 'delta_kl_sum', 'seconds') if k in row}}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                choices, generation = generate(data, batches['train'], teacher['train'], reference, reference_choices,
                                               deadline=deadline, observer=observer)
                choice_path = Path(record['generated_choices_output']); torch.save(choices, choice_path)
                saved_choices = torch.load(choice_path, map_location='cpu', weights_only=True)
                require(torch.equal(choices, saved_choices) and group_inventory(saved_choices) == generation['enumeration']['group_inventory'], 'Actual generated choices changed on save/reload.')
                write_json(work / 'proposal_generation.json', generation)
                state, mask, result = search(data, batches, teacher, saved_choices, groups=GROUPS, deadline=deadline,
                                             observer=lambda row: observer('acceptance', row))
        state_path, mask_path = work / 'gate_state.pt', work / 'accepted_mask.pt'
        torch.save(state, state_path); torch.save(mask, mask_path)
        saved_state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
        saved_mask = torch.load(mask_path, map_location='cpu', weights_only=True)
        require({key: tensor_record(value) for key, value in saved_state.items()} == result['final_state']
                and all(torch.equal(value, saved_mask[key]) for key, value in mask.items()), 'Actual13 endpoint or mask changed on save/reload.')
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'proposal_source': record['proposal_source'], 'protocol': record['protocol'], 'teacher_train': teacher_report,
            'proposal_generation': generation, 'search': result,
            'fresh_generation_and_search_original_train_scores_equal': generation['before'] == result['before']['train'],
            'saved': {'choices_path': str(choice_path), 'choices_sha256': sha256(choice_path), 'choices_roundtrip_exact': True,
                'state_path': str(state_path), 'state_sha256': sha256(state_path), 'mask_path': str(mask_path),
                'mask_sha256': sha256(mask_path), 'state_and_mask_roundtrip_exact': True},
            'trajectory_sha256': sha256(work / 'trajectory.jsonl'), 'elapsed_seconds': time.time() - record['started_unix'],
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

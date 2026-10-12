"""Prepare or run the fixed four-arm, two-domain native forward diagnostic."""
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

from experiments.gate_tail_objective_probe.core import ARMS, teacher_statistics, probability_metrics, teacher_pass, student_pass
from experiments.gate_tail_objective_probe.data import prepare_data
from experiments.gate_group_accept_evaluation.prepare import patched_state as group_patch
from experiments.gate_student_context_evaluation.prepare import patched_state as original_patch, budget_equal
from experiments.gate_student_context_probe.data import ROOT, SOURCE, SOURCE_ID, TARGET
from experiments.gate_student_context_probe.run import merge
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.distill_loss_core import compute_kl_top_mass_token_loss
from train_utils.utils import configure_deterministic_mode, set_seed


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for row in record['protected_stats']: check_stat(row)


def cpu_mass_kl_check():
    generator = torch.Generator().manual_seed(137)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        teacher = torch.randn((7, 131), generator=generator).to(dtype)
        student = torch.randn((7, 131), generator=generator).to(dtype)
        labels = torch.arange(7)
        stats = teacher_statistics(teacher, labels)
        actual, nll = probability_metrics(student, stats)
        expected = compute_kl_top_mass_token_loss(student_logits=student, teacher_logits=teacher,
                                                   temperature=1.0, top_k=100)
        reference_nll = torch.nn.functional.cross_entropy(student.float(), labels, reduction='none')
        require(torch.equal(actual, expected) and torch.equal(nll, reference_nll),
                'Compressed teacher statistics must preserve native mass-KL and causal CE exactly on CPU.')
        rows.append({'dtype': str(dtype), 'native_kl_exact': True, 'native_nll_exact': True,
                     'teacher_statistics_bytes': sum(t.numel() * t.element_size() for t in stats.values())})
    return rows


def prepare(base, output):
    base, output = base.resolve(), output.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Prepare once in a fresh directory with CUDA explicitly hidden.')
    torch.set_num_threads(4)
    data, batches, _archive = prepare_data(base)
    original_dir = base / 'gate_student_context_full8_r1/original_codes'
    group_dir = base / 'gate_group_accept_full8'
    old, new = read(original_dir / 'launch.json'), read(group_dir / 'launch.json')
    old_audit, new_audit = read(original_dir / 'audit.json'), read(group_dir / 'audit.json')
    old_review, new_review = read(original_dir.parent / 'review.json'), read(group_dir / 'review.json')
    require(old['status'] == new['status'] == 'completed' and old['exit_code'] == new['exit_code'] == 0
            and old_audit['status'] == new_audit['status'] == 'verified'
            and old_review['status'] == new_review['status'] == 'verified',
            'Require the actual completed and reviewed native full8 endpoints.')
    for directory in (original_dir, group_dir):
        for stage in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer'):
            require(int((directory / (stage + '.exit')).read_text()) == 0, 'A prerequisite stage did not exit successfully.')
    require(old_audit['stage_changed_bits'] == [0, 0] and new_audit['stage_changed_bits'] == [194788, 516131]
            and old_audit['checkpoint_id'] == old['checkpoint_id']
            and new_audit['checkpoint_id'] == new['checkpoint_id'] == new_review['checkpoint_id'],
            'Do not substitute a different code-selection arm or model identity.')
    endpoints = {'original_codes500': old['endpoint_path'], 'group_accept500': new['endpoint_path']}
    source_meta = read(SOURCE / 'checkpoint_meta.json')
    require(source_meta['checkpoint_id'] == SOURCE_ID, 'Only original f927 supplies the shared complete model.')
    source = torch.load(SOURCE / source_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    source_budget = audit_state_dict(source_meta, source, max_bpw=2.5, max_lora_rank=0)
    sources, inputs = dict(data['source_sha256']), dict(data['input_sha256'])
    merge(sources, new['source_sha256'])
    states, contracts = {}, {}
    for arm, path in endpoints.items():
        previous = old if arm == 'original_codes500' else new
        require(sha256(path) == previous['endpoint_sha256'], 'A reviewed endpoint payload changed.')
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        expected, contract = (original_patch(source, state, 'original_codes') if arm == 'original_codes500'
                              else group_patch(source, state, 'group_accept_decoder500'))
        budget_equal(audit_state_dict(source_meta, expected, max_bpw=2.5, max_lora_rank=0), source_budget)
        states[arm] = {key: tensor_record(value) for key, value in state.items()}
        contracts[arm] = contract
        inputs[str(Path(path).resolve())] = sha256(path)
    evidence = [original_dir.parent / 'review.json', group_dir / 'review.json']
    for directory in (original_dir, group_dir):
        evidence += [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
                     'deploy.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
        evidence.append(directory / 'final_model/checkpoint_meta.json')
    # Small immutable prerequisite records, never the assembled candidate weight files.
    for path in evidence: inputs[str(path)] = sha256(path)
    failed = base / 'gate_tail_objective_probe'
    release = read(failed / 'teacher_release_debug/report.json')
    require(read(failed / 'launch.json')['status'] == 'failed'
            and (failed / 'run.exit').read_text().strip() == '1'
            and (failed / 'teacher_release_debug.exit').read_text().strip() == '0'
            and release['status'] == 'completed' and release['teacher_weakref_dead']
            and not release['teacher_parameter_refs_alive'] and release['hf_mlp_alive_cpu']
            and not release['cuda_tensor_objects_after_release']
            and release['allocated_after_release'] == 8519680
            and release['allocated_after_clear_cublas'] == release['reserved_after_clear_cublas'] == 0,
            'Require the real teacher-release diagnosis before replacing the failed memory check.')
    release_evidence = [failed / name for name in ('launch.json', 'launch.prepared.json', 'run.log', 'run.exit',
        'outer.exit', 'teacher_release_debug/report.json', 'teacher_release_debug.log', 'teacher_release_debug.exit')]
    release_evidence += list((failed / 'failed_source').glob('*.py'))
    for path in release_evidence: inputs[str(path)] = sha256(path)
    for relative in ('train_utils/distill_loss_core.py', 'rotation/model_utils.py',
                     'litebsq/vae_linear_prewarm.py', 'train_utils/v6_model_loader.py', 'tools/cat_eval.py'):
        path = ROOT / relative; sources[str(path)] = sha256(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    cpu_loss = cpu_mass_kl_check()
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'base_root': str(base), 'output_dir': str(output),
        'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': [data['source_data']['source_weight_stat'], *data['source_data']['hf_weight_files'], *data['input_data_stat']],
        'data': data, 'endpoints': endpoints, 'endpoint_states': states, 'patch_contracts': contracts,
        'source_bit_budget': source_budget, 'cpu_native_loss_parity': cpu_loss,
        'failure_resolution': {'old_run': str(failed), 'teacher_release_proof': str(failed / 'teacher_release_debug/report.json'),
            'diagnosed_workspace_bytes': 8519680,
            'change': 'Verify teacher/model parameter weakrefs are dead, record allocation, explicitly clear proven cuBLAS workspace, then retain zero-allocation requirement; no numerical/objective/data change.'},
        'arms': list(ARMS), 'timeout_seconds': 900, 'termination_grace_seconds': 20,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'protocol': {'documents_per_domain': 32, 'max_length': 256, 'microbatch': 1,
            'numerics': 'Native BF16 teacher/student forwards, Qwen FA2 and untouched FP32 RoPE; same full256 shapes and category group8 decoder-cache geometry.',
            'kl': 'Native teacher top100 plus outside-mass forward KL, T=1, full student logsumexp; all valid shifted positions.',
            'nll': 'All nonpadding causal input_ids labels including EOS, no prompt/response weights; per-document and token-weighted aggregate.',
            'local': 'Actual native L18 MLP output against native original HF MLP at the exact same student post-attention-normalized input; SSE over valid tokens.',
            'oracle': 'Replace only L18 mlp return with HF_MLP(x_student); native block adds the original residual once. Analysis-only, not a budget-compliant deployment candidate.',
            'freeze': 'One persistent f927 model; original or reviewed thirteen-tensor gate patches, restore afterward. No optimizer/search/update/model export.',
            'teacher_storage': 'Only CPU top100 values/indices, full logsumexp, labels and per-token NLL; no full-vocabulary logits archive.',
            'old_local_comparison': 'Previous local fits used valid-token-only GEMMs; these full256 native forwards are measured independently, not required to equal old scalar NMSE.',
            'resource_estimate': 'Prior native student prewarm peaked65.71GiB. Sequential teacher then student, estimated5-10min on one80GB GPU; actual resource/time recorded, no automatic extension.',
            'boundary': 'Frozen endpoint diagnosis on previously used Alpaca held32 and fixed FineWeb rows. Not benchmark tuning, an independent decontaminated test, new training, full8/LongBench, or nine-item acceptance.'},
        'launcher_argv': [sys.executable, '-m', 'experiments.gate_tail_objective_probe.run', '--launch-record', str(output / 'launch.json')],
        'cuda_initialized': False}
    check(record)
    require(not torch.cuda.is_initialized() and all(v.device.type == 'cpu' for b in batches.values() for v in b.values()),
            'CPU preparation must not initialize CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print(json.dumps({'launch': str(output / 'launch.json'), 'sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'valid_tokens': data['valid_tokens'],
        'causal_tokens': data['causal_tokens'], 'cpu_loss_parity': cpu_loss, 'cuda_initialized': False}), flush=True)


def comparisons(arms):
    output = {}
    for reference, candidate in (('f927', 'original_codes500'), ('original_codes500', 'group_accept500'),
                                  ('f927', 'group_accept500'), ('f927', 'conditional_hf_mlp')):
        domains = {}
        for domain, a in arms[reference]['domains'].items():
            b = arms[candidate]['domains'][domain]
            require(a['causal_tokens'] == b['causal_tokens'] and a['local_reference_energy'] == b['local_reference_energy'],
                    'Paired document population or conditional reference differs.')
            metrics = {}
            for field in ('local_nmse', 'kl_top100_mass_mean', 'nll_mean'):
                metrics[field] = {'delta': b[field] - a[field],
                    'improved_documents': sum(y[field] < x[field] for x, y in zip(a['documents'], b['documents'], strict=True)),
                    'worsened_documents': sum(y[field] > x[field] for x, y in zip(a['documents'], b['documents'], strict=True))}
            domains[domain] = metrics
        output[candidate + '_minus_' + reference] = domains
    return output


def run(path):
    path = path.resolve(); record = read(path); output = Path(record['output_dir'])
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and Path.cwd() == Path(record['cwd']) and not record['queued'] and not record['automatic_retry'],
            'Use the explicitly bound GPU, native bitvae interpreter and original source directory.')
    require(all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
            'Keep the prepared native numerical/offline environment.')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not (output / 'run').exists(),
            'One assigned GPU and a fresh run directory are required.')
    check(record)
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
                  worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); work = output / 'run'; work.mkdir()
    deadline = record['started_unix'] + record['timeout_seconds']; code = 1
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0)
        configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = prepare_data(record['base_root'])
        require(data == record['data'], 'Runtime token/source replay changed.')
        endpoints = {arm: torch.load(file, map_location='cpu', weights_only=True, mmap=True) for arm, file in record['endpoints'].items()}
        require({arm: {k: tensor_record(v) for k, v in state.items()} for arm, state in endpoints.items()}
                == record['endpoint_states'], 'A reviewed endpoint tensor changed.')
        with (work / 'observations.jsonl').open('x') as stream:
            def observer(arm, domain, index, row):
                stream.write(json.dumps({'arm': arm, 'domain': domain, **row}) + '\n'); stream.flush()
                if index in (0, 15, 31): print(f'{arm} {domain} {index + 1}/32', flush=True)
            teacher, hf_mlp, teacher_report = teacher_pass(data, batches, archive, device=torch.device('cuda:0'), deadline=deadline, observer=observer)
            torch.save(teacher, work / 'teacher_top100.pt')
            write_json(work / 'teacher.json', teacher_report)
            result = student_pass(data, batches, teacher, hf_mlp, endpoints, device=torch.device('cuda:0'), deadline=deadline, observer=observer)
        report = {'status': 'completed', 'launch_record': str(path), 'protocol': record['protocol'],
                  'teacher': teacher_report, **result, 'comparisons': comparisons(result['arms']),
                  'teacher_cache_sha256': sha256(work / 'teacher_top100.pt'),
                  'observations_sha256': sha256(work / 'observations.jsonl'),
                  'source_sha256': record['source_sha256'], 'input_sha256': record['input_sha256'],
                  'finished_unix': time.time(), 'full8_and_longbench': 'not_evaluated'}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json'))
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time())
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--base-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.base_root is not None and args.output_root is not None and args.launch_record is None, 'Use the preparation arguments only.')
        prepare(args.base_root, args.output_root)
    else:
        require(args.launch_record is not None and args.base_root is None and args.output_root is None, 'Use the bound launch record only.')
        sys.exit(run(args.launch_record))

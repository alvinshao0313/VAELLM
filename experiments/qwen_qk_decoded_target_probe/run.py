"""Explicit CPU binding and one bounded native Q/K target adapter invocation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import torch

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, SHARDS, check, eval_argv, read
from experiments.full_rht_gate_pair.audit import results as full8_results
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen_qk_decoded_target_probe.core import ARMS, parsed, train
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.utils import get_logger

TEMPLATE = RESULT / 'qwen_qk_rotation_pair/block32/launch.json'
GEOMETRY = RESULT / 'qwen_qk_decoded_target_probe/source_geometry.json'


def prepare(arm, steps, output, mechanism_review=None):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Prepare a new CPU-only record in the actual source snapshot.')
    template = read(TEMPLATE); check(template)
    geometry = read(GEOMETRY)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and geometry['status'] == 'verified'
             and geometry['source_checkpoint_ids']['f927'] == template['base_checkpoint_id'],
             'Require the real completed source pair and CPU geometry proof.')
    target_kind, block = ARMS[arm]
    native = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'native'), '--weight_rotation_block_size': str(block),
               '--vae_steps': f'default={steps}', '--log_every': '1' if steps == 2 else '100'}
    for key, value in changes.items():
        _require(native.count(key) == 1, 'Ambiguous inherited flag: ' + key)
        native[native.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(native)
    training.seed = int(cat.seed)
    resolved = resolve_category_runtime_configs(cat, vae, ('q_proj', 'k_proj'))
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    activation_dir = Path(template['donor_checkpoint_dir'])
    source_paths = [*(ROOT / 'experiments/qwen_qk_decoded_target_probe').glob('*.py'),
        ROOT / 'train_utils/cat_runtime_state_v6.py', ROOT / 'train_utils/v6_model_loader.py',
        ROOT / 'litebsq/vae_linear_prewarm.py', ROOT / 'experiments/qwen3_w25/assemble_categories.py',
        ROOT / 'experiments/full_rht_gate_pair/audit.py', ROOT / 'experiments/full_rht_gate_pair/common.py',
        ROOT / 'experiments/rotation_factor_probe/capture.py']
    sources = dict(template['source_sha256'])
    sources.update({str(p): sha256(p) for p in source_paths})
    paths = [TEMPLATE, GEOMETRY, TEMPLATE.parent / 'audit.json', TEMPLATE.parent.parent / 'comparison.json',
             activation_dir / 'checkpoint_meta.json', activation_dir / 'cat_runtime_state.pt']
    inputs = dict(template['input_sha256']); inputs.update({str(p): sha256(p) for p in paths})
    if steps == 10000:
        _require(mechanism_review is not None, 'Formal is separate and needs the actual three-arm mechanism proof.')
        proof_path = Path(mechanism_review).resolve(); proof = read(proof_path)
        _require(proof['status'] == 'verified_three_arm_mechanism' and proof['steps_per_stage'] == 2
                 and proof['fresh_formal_required'] and proof['source_root'] == str(ROOT),
                 'The actual matched short mechanism has not been verified.')
        check_hashes(proof['adapter_source_sha256'])
        _require(all(sources.get(p) == digest for p, digest in proof['adapter_source_sha256'].items()),
                 'The adapter differs from its native mechanism evidence.')
        inputs[str(proof_path)] = sha256(proof_path)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'qwen3', 'arm': arm, 'target_kind': target_kind, 'block_size': block,
        'steps_per_stage': steps, 'full_evaluation': steps == 10000,
        'native_argv': native, 'normalized_expected': normalized,
        'architecture': template['architecture'], 'base_dir': template['base_dir'],
        'base_checkpoint_id': template['base_checkpoint_id'], 'categories': ['q_proj', 'k_proj'],
        'activation_checkpoint_dir': str(activation_dir), 'source_geometry_path': str(GEOMETRY),
        'old_hf_audit_path': str(TEMPLATE.parent / 'audit.json'),
        'expected_user_linear_bits': 17065672704, 'budget_components': template['budget_estimate']['components'],
        'control_result_paths': template['control_result_paths'],
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'environment': template['environment'], 'timeout_seconds': 900 if steps == 2 else 2700,
        'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.qwen_qk_decoded_target_probe.run',
                          '--launch-record', str(output / 'launch.json')],
        'adapter_scope': 'Original native CAT payload algorithm; only source target tensor and explicit fixed protection plan are supplied by the isolated adapter.',
        'target_contract': {'decoder_source': 'fresh VAE, never a short or trained decoder',
            'decoded_source': 'actual f927 BF16 group8 original-coordinate weight',
            'hf_source': 'actual original safetensors72 tensors',
            'backbone_and_other180': 'immutable f927', 'protection': 'old HF common64 input columns; re-encode current target with native INT8',
            'activation': 'retained original HF Alpaca512x256 seed31 statistics, no reranking or new forward',
            'rng': 'After all target/Linear materialization reset seed31 once; q then k, record category entry/exit.',
            'short_boundary': 'Two updates per residual stage change the short scheduler horizon and only establish mechanism/resources. Formal starts fresh at10000.'},
        'primary_comparisons': ['decoded_full_minus_decoded_block32', 'decoded_full_minus_new_hf_full'],
        'historical_hf_pair_is_rng_matched': False,
        'protocol': {**template['protocol'], 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
        'cuda_initialized': False}
    parsed(record); check(record)
    _require(not torch.cuda.is_initialized(), 'Preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'arm': arm, 'steps': steps, 'source_count': len(sources),
           'input_count': len(inputs), 'launch_sha256': sha256(output / 'launch.json')}, flush=True)


def run(path):
    from experiments.qwen_qk_decoded_target_probe.audit import save_and_audit
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1
    try:
        _require(record['status'] == 'bound_ready' and Path.cwd().resolve() == ROOT
                 and str(ROOT) == os.environ.get('PYTHONPATH') and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve(),
                 'Run only an explicitly bound snapshot invocation.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Do not replace the inherited numerical/offline policy.')
        check(record)
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={})
        write_json(path, record)
        os.environ['LOG_FILE'] = str(output / 'train.log'); get_logger('linear_by_category')
        model, base_state, training = train(record, torch.device('cuda:0'))
        (output / 'train.exit').write_text('0\n')
        log = (output / 'train.log').read_text()
        native_steps = {}
        for category in ('q_proj', 'k_proj'):
            for stage in (1, 2):
                steps = record['steps_per_stage']
                tag = f'{category}.L0-35/stage{stage}'
                matches = re.findall(r'\[' + re.escape(tag) + r'\] step=(\d+)/(\d+) loss=(\S+) recon=(\S+) commit=(\S+)', log)
                cadence = 1 if steps == 2 else 100
                _require([int(row[0]) for row in matches] == list(range(cadence, steps + 1, cadence))
                         and all(int(row[1]) == steps and all(math.isfinite(float(v)) for v in row[2:])
                                 for row in matches), 'Native update/loss log is incomplete or nonfinite: ' + tag)
                native_steps[tag] = [{'step': int(row[0]), 'loss': float(row[2]),
                                     'recon': float(row[3]), 'commit': float(row[4])} for row in matches]
        training['native_logged_updates'] = native_steps
        write_json(output / 'training.json', training)
        record['status'] = 'saving'; write_json(path, record)
        audit = save_and_audit(model, base_state, record, output, torch.device('cuda:0'))
        write_json(output / 'audit.json', audit); (output / 'save.exit').write_text('0\n')
        del model, base_state
        gc.collect(); torch.cuda.empty_cache()
        if record['full_evaluation']:
            for name, tasks, batch in SHARDS:
                argv = eval_argv(output / 'final_model', output / (name + '.json'),
                                 tasks=tasks, batch=batch, python=sys.executable)
                record['status'] = name; write_json(path, record)
                with (output / (name + '.log')).open('x') as handle:
                    process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=handle, stderr=subprocess.STDOUT)
                    record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
                    write_json(path, record); result = process.wait()
                (output / (name + '.exit')).write_text(str(result) + '\n')
                record['stages'][name].update(exit_code=result, finished_unix=time.time()); write_json(path, record)
                _require(result == 0, 'Native full8 failed: ' + name)
            summary = full8_results(output, record)
            summary.pop('rotation_pair_comparison')
            summary['adapter_pair_comparison'] = 'pending_other_arms'
            summary['user_linear_bpw'] = audit['bit_budget']['user_linear_bpw']
            write_json(output / 'full8_summary.json', summary)
        check(record)
        record.update(status='completed', checkpoint_id=audit['checkpoint_id'],
            training_sha256=sha256(output / 'training.json'), audit_sha256=sha256(output / 'audit.json'))
        code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')


def compare_mechanism(directories, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(), 'CPU-only unique mechanism comparison.')
    runs = {}; evidence = {}
    for directory in directories:
        directory = Path(directory).resolve(); record = read(directory / 'launch.json'); check(record)
        _require(record['status'] == 'completed' and record['exit_code'] == 0
                 and record['steps_per_stage'] == 2 and not record['full_evaluation'], 'Incomplete actual mechanism arm.')
        for name in ('train', 'save', 'run', 'outer'):
            _require((directory / (name + '.exit')).read_text().strip() == '0', 'Failed stage: ' + name)
        training = read(directory / 'training.json'); audit = read(directory / 'audit.json')
        _require(sha256(directory / 'training.json') == record['training_sha256']
                 and sha256(directory / 'audit.json') == record['audit_sha256']
                 and audit['status'] == 'verified' and audit['native_group8_reload_exact']
                 and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423,
                 'Missing actual native save/fresh/group8 proof.')
        _require(record['arm'] not in runs, 'Duplicate mechanism arm.')
        runs[record['arm']] = (record, training, audit)
        for name in ('launch.json', 'training.json', 'audit.json', 'train.exit', 'save.exit', 'run.exit', 'outer.exit'):
            evidence[str(directory / name)] = sha256(directory / name)
    _require(set(runs) == set(ARMS), 'Require the three preselected target/rotation arms.')
    a = runs['decoded_block32']
    def paired_normalized(record):
        value = deepcopy(record['normalized_expected'])
        for section in ('canonical_config', 'cat_args'):
            _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'native'), 'Wrong native output mirror.')
            value[section]['output_dir'] = '<paired-output>'
        for section in (value['canonical_config']['core_template'], value['vae_args']):
            _require(section['weight_rotation_block_size'] == record['block_size'], 'Wrong native rotation mirror.')
            section['weight_rotation_block_size'] = '<paired-block>'
        return value
    for record, training, audit in runs.values():
        _require(record['base_checkpoint_id'] == a[0]['base_checkpoint_id']
                 and record['source_sha256'] == a[0]['source_sha256']
                 and record['input_sha256'] == a[0]['input_sha256']
                 and paired_normalized(record) == paired_normalized(a[0])
                 and training['post_materialization_seed31_rng'] == a[1]['post_materialization_seed31_rng'],
                 'Source or post-materialization initialization differs across the matched adapter.')
        for category in ('q_proj', 'k_proj'):
            for boundary in ('entry_rng', 'exit_rng'):
                _require(training['categories'][category][boundary] == a[1]['categories'][category][boundary],
                         'Category RNG consumption differs: ' + category + '/' + boundary)
        _require(audit['bit_budget']['user_linear_bits'] == 17065672704, 'Unexpected actual same-rate storage.')
    _require(runs['decoded_full'][1]['materialization']['targets'] == a[1]['materialization']['targets'],
             'Decoded pair did not use identical deployed target tensors.')
    adapters = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    report = {'status': 'verified_three_arm_mechanism', 'source_root': str(ROOT), 'steps_per_stage': 2,
        'fresh_formal_required': True, 'adapter_source_sha256': adapters, 'evidence_sha256': evidence,
        'same_post_materialization_rng_and_category_entry_exit': True, 'decoded_pair_target_exact': True,
        'arms': {arm: {'checkpoint_id': audit['checkpoint_id'], 'budget': audit['bit_budget']['user_linear_bpw'],
            'training_elapsed_seconds': training['elapsed_seconds'], 'peak_allocated_bytes': training['peak_allocated_bytes']}
                 for arm, (_, training, audit) in runs.items()},
        'quality_boundary': 'Actual mechanism only; no full8/LongBench measurement and no short state continues into formal.'}
    write_json(output, report); print({'status': report['status'], 'sha256': sha256(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--launch-record', type=Path)
    mode.add_argument('--compare-mechanism', type=Path, nargs=3)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--steps', type=int, choices=(2, 10000), default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mechanism-review', type=Path)
    args = parser.parse_args()
    if args.prepare:
        parser.error('--prepare needs --arm and --output') if args.arm is None or args.output is None else prepare(args.arm, args.steps, args.output, args.mechanism_review)
    elif args.compare_mechanism:
        parser.error('--compare-mechanism needs --output') if args.output is None else compare_mechanism(args.compare_mechanism, args.output)
    else:
        run(args.launch_record)

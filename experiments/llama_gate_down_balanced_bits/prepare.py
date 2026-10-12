"""CPU-bind independent K40 CAT jobs; bind their composition only after both exit."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_down_pair.compare import arm as down_arm
from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import arm as gate_arm, score
from experiments.llama_gate_down_balanced_bits.audit import TOTAL_BITS, check_donor
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta


CATEGORIES = ('gate_proj', 'down_proj')
CONTROL = RESULT / 'full_rht_gate_down_pair/llama_block32'
CORE_FILES = ('autoencoder.py', 'bitpack.py', 'packed_bit_linear.py', 'fused_multistage_decoder.py',
              'vae_linear.py', 'vae_linear_prewarm.py', 'protected_channel_quant.py')


def recipe(template, output, category):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--codebook_bits': f'default=32,cat:{category}=40'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, [category])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for section in (expected['canonical_config'], expected['vae_args']):
        section['codebook_bits'] = {'default': 32, 'by_category': {category: 40}}
    expected['resolved_category_runtime'][category]['codebook_bits'] = 40
    _require(normalized == expected, 'Parser changed more than output and this category K.')
    _require(cat.compression_categories == category and cat.target_layers == 'all'
             and cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and training.bf16 and not training.fp16 and resolved[category].steps == 10000,
             'Independent category recipe changed.')
    return [sys.executable, 'tools/cat_train.py', *argv], normalized


def evidence_and_template(category):
    directory = RESULT / ('full_rht_gate_pair' if category == 'gate_proj' else 'full_rht_down_pair') / 'llama_block32'
    template, audit, _, _, evidence = (gate_arm if category == 'gate_proj' else down_arm)(directory, 32)
    _require(template['base_checkpoint_id'] == EXPECTED_IDS['llama'], 'Require original CAT3577.')
    matched = read(CONTROL / 'launch.json'); summary = read(CONTROL / 'full8_summary.json')
    _require(matched['status'] == 'completed' and matched['exit_code'] == 0
             and all(int((CONTROL / (name + '.exit')).read_text()) == 0
                     for name in ('assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'The matched historical composition is incomplete.')
    current, _ = score([CONTROL / (name + '.json') for name in ('mmlu', 'seven')],
                       template['architecture'], CONTROL / 'final_model')
    _require(current['task_metrics'] == summary['task_metrics']
             and current['mean_percent'] == 58.93069850464651
             and current['integer_counts']['mmlu']['correct'] == 6707
             and matched['full8_summary_sha256'] == sha256(CONTROL / 'full8_summary.json'),
             'The matched control raw evaluation differs.')
    pair_path = CONTROL.parent / 'llama_comparison.json'; pair = read(pair_path)
    _require(pair['status'] == 'verified_pair' and pair['model_type'] == 'llama',
             'Historical matched pair evidence is not verified.')
    sources = dict(template['source_sha256'])
    for p in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
              ROOT / 'experiments/full_rht_down_pair/compare.py',
              *(ROOT / 'experiments/llama_gate_down_balanced_bits').glob('*.py')]:
        sources[str(p)] = sha256(p)
    evidence.update(template['input_sha256'])
    paths = [CONTROL / name for name in ('launch.json', 'full8_summary.json', 'mmlu.json', 'seven.json',
        'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit',
        'final_model/checkpoint_meta.json', 'final_model/assembly_report.json')]
    paths.append(pair_path)
    proof = ORIGINAL / 'result/qwen3_8b_w25_20260930/validation/mlp25_deployment.json'
    proof_exit = proof.with_suffix('.exit'); proof_record = read(proof)
    _require(proof_record['status'] == 'complete' and int(proof_exit.read_text()) == 0
             and '40' in proof_record['cases']
             and proof_record['configuration']['codebook_dim'] == 32
             and proof_record['configuration']['hidden_dim'] == 128
             and proof_record['configuration']['residual_stages'] == 2,
             'Existing K40 grouped/transposed deployment proof differs.')
    script = ROOT / 'experiments/qwen3_w25/validate_h256_deployment.py'
    _require(sha256(script) == proof_record['code_sha256'], 'Existing K40 proof script changed.')
    sources[str(script)] = sha256(script); paths += [proof, proof_exit]
    native = RESULT / 'best1200_fresh_gate_up_k40_24'
    native_record = read(native / 'launch.json')
    _require(native_record['status'] == 'evaluated' and all(int((native / (key + '.exit')).read_text()) == 0
             for key in ('train', 'assembly', 'run')), 'Existing K40 native chain did not complete.')
    for name in CORE_FILES:
        relative = 'litebsq/' + name; path = ROOT / relative
        _require(sha256(path) == native_record['source_sha256'][relative], 'K40 core proof changed: ' + relative)
        sources[str(path)] = sha256(path)
    paths += [native / name for name in ('launch.json', 'train.exit', 'assembly.exit', 'run.exit',
                                         'final_model/checkpoint_meta.json', 'final_model/assembly_report.json')]
    evidence.update({str(path): sha256(path) for path in paths})
    return template, audit, sources, evidence, str(directory / 'audit.json')


def budget_estimate(meta, budget):
    changes = {}
    for category, old_k, count in (('gate_proj', 32, 64), ('down_proj', 48, 256)):
        specs = [s for s in meta['converted_modules'] if s['name'].endswith('.' + category)]
        _require(len(specs) == 32, 'Expected all 32 source projections.')
        code = decoder = 0
        for spec in specs:
            _require(spec['protected_input_indices']['shape'] == [count]
                     and spec['stage_codebook_dims'] == [32, 32] and spec['residual_stages'] == 2,
                     'Source protected geometry changed.')
            for bank, dec in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
                rows, parts, k = bank['logical_shape']
                _require(k == old_k and parts == 1 and bank['shape'] == [rows, 1, old_k // 8]
                         and dec['in_dim'] == old_k and dec['out_dim'] == 32
                         and dec['hidden_dim'] == 128 and dec['num_res_blocks'] == 0
                         and dec['param_dtype'] == 'bfloat16', 'Source code/decoder geometry changed.')
                code += rows * (40 - old_k); decoder += 128 * (40 - old_k) * 16
        changes[category] = {'code_delta_bits': code, 'decoder_delta_bits': decoder}
    total = budget['user_linear_bits'] + sum(sum(x.values()) for x in changes.values())
    _require(budget['user_linear_bits'] == 17100439552 and total == TOTAL_BITS
             and budget['original_linear_weights'] == 6979321856 and 2 * total < 5 * 6979321856,
             'Source metadata budget estimate changed.')
    return {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'per_category_delta': changes, 'byte_padding_bits': 0,
        'unchanged_components': ['protected_indices', 'protected_weights', 'protected_scales', 'rotation'],
        'boundary': 'Metadata-derived estimate; final native tensor budget audit remains required.'}


def prepare(phase, output, category=None, gate_launch=None, down_launch=None):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Use a fresh CPU-only snapshot preparation.')
    _require(phase in ('donor', 'compose') and (category in CATEGORIES if phase == 'donor' else category is None),
             'Select one explicit independent donor or completed-donor composition.')
    template, historical, sources, evidence, historical_audit = evidence_and_template(category or 'gate_proj')
    meta = load_v6_meta(template['base_dir']); budget = budget_estimate(meta, historical['bit_budget'])
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'phase': phase, 'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'architecture': template['architecture'], 'base_dir': template['base_dir'],
        'base_checkpoint_id': EXPECTED_IDS['llama'], 'control_result_paths': template['control_result_paths'],
        'historical_control_dir': str(CONTROL), 'historical_control_result_paths':
            [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'source_budget_path': historical_audit, 'budget_estimate': budget, 'expected_user_linear_bits': TOTAL_BITS,
        'block_size': 32, 'steps_per_stage': 10000, 'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': list(template['protected_stats']), 'environment': template['environment'],
        'protocol': template['protocol'], 'timeout_seconds': 1800 if phase == 'donor' else 1200,
        'termination_grace_seconds': 20, 'launcher_argv': [sys.executable, '-u', '-m',
            'experiments.llama_gate_down_balanced_bits.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': 'Actual native K40 and grouped/transposed packed proof; no new GPU short.',
        'comparison_boundary': 'Primary: independently fresh gate32/down48 block32 combination. Secondary:3577, also reflecting loss of prior category recovery. K shape changes initialization RNG consumption; equal seeds do not promise identical parameter trajectories.',
        'manual_handoff': 'Donor jobs never assemble/evaluate/start another process after their audit. After both outer exits0, explicitly CPU-prepare compose and separately bind one GPU.',
        'cuda_initialized': False}
    if phase == 'donor':
        argv, normalized = recipe(template, output, category)
        record.update(category=category, category_index=0, shuffle_seeds=[31, 32],
            codebook_bits=40, training_argv=argv, normalized_expected=normalized,
            historical_category_audit=historical_audit, full_evaluation=False,
            resource_estimate={'old_cat_seconds': template['stages']['train']['finished_unix']
                - template['stages']['train']['started_unix'], 'hard_seconds': 1800,
                'boundary': 'K40 runtime is not yet measured; no single-donor task evaluation.'})
    else:
        _require(gate_launch is not None and down_launch is not None, 'Both completed donor launch paths are required.')
        donors = {}
        for cat, path in zip(CATEGORIES, (gate_launch, down_launch), strict=True):
            path = Path(path).resolve(); donor = read(path); check(donor)
            _require(donor['phase'] == 'donor' and donor['category'] == cat
                     and donor['status'] == 'donor_completed' and donor['exit_code'] == 0
                     and all(int((path.parent / (key + '.exit')).read_text()) == 0
                             for key in ('train', 'dispatcher', 'outer')), 'Donor has not naturally completed: ' + cat)
            for key in ('architecture', 'base_dir', 'base_checkpoint_id', 'environment', 'protocol', 'budget_estimate'):
                _require(donor[key] == record[key], 'Donor scientific/source contract differs: ' + key)
            directory = Path(donor['donor_checkpoint_dir']); dm, _ = check_donor(directory, donor)
            proof = read(path.parent / 'audit.json')
            _require(proof['status'] == 'verified' and proof['checkpoint_id'] == dm['checkpoint_id']
                     and proof['historical_protection_sign_exact']
                     and donor['audit_sha256'] == sha256(path.parent / 'audit.json'), 'Donor audit differs.')
            donors[cat] = {'checkpoint_dir': str(directory), 'checkpoint_id': dm['checkpoint_id'],
                           'audit_path': str(path.parent / 'audit.json')}
            for pinset, incoming in ((sources, donor['source_sha256']), (evidence, donor['input_sha256'])):
                for key, value in incoming.items():
                    _require(key not in pinset or pinset[key] == value, 'Conflicting donor pin: ' + key)
                    pinset[key] = value
            for p in [path, path.parent / 'audit.json', directory / 'checkpoint_meta.json', directory / 'config.json',
                      directory.parent / 'normalized_cat_runtime_args.json',
                      *(path.parent / (key + '.exit') for key in ('train', 'dispatcher', 'outer'))]:
                evidence[str(p)] = sha256(p)
            record['protected_stats'].append(file_stat(directory / dm['state_dict_file']))
        record.update(donors=donors, full_evaluation=True, assembly_order=['down_proj', 'gate_proj'],
            resource_estimate={'historical_composition_seconds': 449.851, 'hard_seconds': 1200,
                              'extra_work': 'Two source-union saves instead of one; only final complete eight evaluated.'})
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'phase': phase, 'category': category, 'status': record['status'],
        'sources': len(sources), 'inputs': len(evidence), 'estimated_bpw': budget['estimated_bpw'],
        'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('donor', 'compose'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--category', choices=CATEGORIES)
    parser.add_argument('--gate-launch', type=Path)
    parser.add_argument('--down-launch', type=Path)
    args = parser.parse_args()
    prepare(args.phase, args.output_dir, args.category, args.gate_launch, args.down_launch)

"""Prepare one native transposed Llama gate input64-to128 comparison on CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import score
from experiments.llama_gate_transpose.audit import check_donor as check_control_donor
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

CONTROL = RESULT / 'llama_gate_transpose/transpose'
CONTROL_ID = 'd3abc0f2-baf4-4237-86e6-5918783024ed'
EXPECTED_BITS = 17276747776


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--channel_protect_count': 'default=128'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited native argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['channel_protect_count_table'] = {'default': 128}
    expected['cat_args']['channel_protect_count'] = {'default': 128}
    expected['resolved_category_runtime']['gate_proj']['channel_protect_count'] = 128
    _require(normalized == expected, 'Only output and protected input count64-to128 may change.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,gate_proj,down_proj'
             and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.channel_axis == 'input' and cat.channel_quant == 'int8'
             and cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 32
             and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and cfg.decoder_type == 'symmetric' and cfg.channel_protect_count == 128
             and vae.weight_rotation == 'two_sided' and vae.weight_rotation_block_size == 32
             and vae.lr == .003 and vae.new_quant and training.bf16 and not training.fp16,
             'Keep the completed native gateTrue recipe.')
    return argv, normalized


def estimate(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(meta['checkpoint_id'] == budget['checkpoint_id'] == CONTROL_ID and len(specs) == 32
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17100439552
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
             and budget['budget_passed'] and not budget['contract_violations'],
             'Require the completed transposed gate input64 control budget.')
    components = {key: value['storage_bits'] for key, value in budget['components'].items()}
    changes = dict.fromkeys(components, 0)
    for spec in specs:
        _require(spec['in_features'] == 4096 and spec['out_features'] == 14336
                 and spec['compressed_in_features'] == 4032 and spec['compressed_out_features'] == 14336
                 and spec['transpose'] and spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['weight_rotation']['block_size'] == 32
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, 14336], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and all(p['logical_shape'] == [1806336, 1, 32] and p['shape'] == [1806336, 1, 4]
                         for p in spec['stage_vq_weights']), 'Historical gate geometry differs.')
        additional = 128 - 64
        changes['code'] -= 14336 * additional // 32 * 32 * 2
        changes['protected_weights'] += additional * 14336 * 8
        changes['protected_scales'] += additional * 16
        changes['protected_indices'] += additional * 64
        changes['rotation'] -= additional * 8
    updated = {key: value + changes[key] for key, value in components.items()}
    groups = 14336 * (4096 - 128) // 32; total = sum(updated.values())
    _require(groups == 1777664 and 128 % 32 == (4096 - 128) % 32 == 14336 % 32 == 0
             and changes['decoder'] == 0 and total == EXPECTED_BITS and total * 2 < 6979321856 * 5,
             'Integer storage estimate or D32/block32 alignment differs.')
    return {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'estimated_component_bits': updated,
        'component_delta_bits': changes, 'new_groups_per_stage': groups,
        'protected_input_count': 128, 'gate_count': 32, 'preserved_projection_count': 192,
        'boundary': 'Integer estimate from actual control metadata/storage. Packed codes, BF16 decoders, INT8 protection, BF16 scales, INT64 indices and INT8 signs included. Final saved tensor audit remains required.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and all(int((CONTROL / (name + '.exit')).read_text()) == 0
                     for name in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed fresh gateTrue control.')
    donor, names = check_control_donor(Path(template['donor_checkpoint_dir']), template)
    audit = read(CONTROL / 'audit.json'); meta = load_v6_meta(str(CONTROL / 'final_model'))
    current, _ = score([CONTROL / (name + '.json') for name in ('mmlu', 'seven')],
                        template['architecture'], CONTROL / 'final_model')
    review_path = CONTROL / 'independent_review.json'; review = read(review_path)
    _require(review['status'] == 'independent_cpu_verified' and review['candidate'] == current
             and audit['status'] == 'verified' and audit['checkpoint_id'] == current['checkpoint_id']
             == meta['checkpoint_id'] == CONTROL_ID and audit['donor_checkpoint_id'] == donor['checkpoint_id']
             and current['mean_percent'] == 59.91121590186278 and current['integer_counts']['mmlu']['correct'] == 6638
             and audit['source_union_save_fresh_tensors'] == 2979 and len(names) == 32,
             'Matched control raw results or native evidence differs.')
    check_hashes(review['evidence_sha256'])
    base = Path(template['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and (base / base_meta['state_dict_file']).is_file(), 'Keep the retained original3577 assembly base.')
    budget = estimate(meta, audit['bit_budget']); argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [*(ROOT / 'experiments/llama_gate_protection').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_gate_protection').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = [CONTROL / name for name in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')]
    evidence += [review_path, CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json']
    record = {key: deepcopy(template[key]) for key in ('cwd', 'architecture', 'base_dir', 'base_checkpoint_id',
        'control_result_paths', 'block_size', 'steps_per_stage', 'full_evaluation', 'protected_stats',
        'snapshot_import_paths', 'environment', 'protocol', 'dimension', 'transpose', 'codebook_bits')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type='llama', categories=['gate_proj'],
        hidden_dim=128, protected_input_count=128, expected_user_linear_bits=EXPECTED_BITS,
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        source_sha256=sources, input_sha256={**template['input_sha256'], **{str(p): sha256(p) for p in evidence}},
        historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        historical_control_result_paths=[str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        historical_control_audit_path=str(CONTROL / 'audit.json'), budget_estimate=budget,
        timeout_seconds=2700, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_protection.run',
                       '--launch-record', str(output / 'launch.json')],
        validation_reuse={'new_gpu_short': False, 'source_control': str(CONTROL),
            'existing_paths': 'Reuse native K32/D32/B8192/H128/block32 training, input INT8 protection, source-union assembly and independent full8. The changed existing count preserves D32/block32 alignment; no new algorithm, kernel or serialization.'},
        scope='Fresh HF all32 gateTrue/K32D32/S2/H128R0/BF16/block32/input128 INT8, originalBSQ/MSE10Kx2/B8192/seed31/LR.003; HF calibration512x256 unchanged. Preserve3577 other192/backbone; no KD/LoRA.',
        comparison_boundary='Primary is fresh gateTrue input64, not original3577. Only output and count change in native normalized parameters. New count changes protection selection, remaining matrix, input sign length, VQ vectors and sampling trajectory; protection/sign equality is neither expected nor required. The fixed3577 complement contains earlier recovery. No LongBench or nine-item inference.',
        experience={'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Llama down input256-to512 lost .88722pp; this does not establish the sign for gate. Gate BF16 restoration was the strongest eight-task category diagnostic, while gateTrue decoderH256 did not help.',
            'action': 'Test missing gate protection allocation with the existing count mechanism, preserving matched control settings and complete eight-task protocol.'},
        resource_estimate={'seconds_limit': 2700,
            'historical_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'historical_control_stage_seconds': {name: item['finished_unix'] - item['started_unix']
                                               for name, item in template['stages'].items()},
            'boundary': 'Historical gateTrue/input64 took1205.24s; the exact input128 runtime is unmeasured. One explicitly bound GPU, no queue or automatic retry.'},
        cuda_initialized=False)
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(record['input_sha256']),
           'estimated_bpw': budget['estimated_bpw'], 'prepared_sha256': sha256(output / 'launch.prepared.json'),
           'cuda_initialized': False}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)

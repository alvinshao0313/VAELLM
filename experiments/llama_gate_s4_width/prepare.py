"""Prepare the fixed fresh S4 gate asymmetric H128/H256 pair without CUDA."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_stage4.prepare import construct
from experiments.llama_gate_decoder_width.prepare import prior_evidence
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

REFERENCE = RESULT / 'llama_gate_stage4/s4'
REFERENCE_ID = '13bd608f-c3a4-43d3-b3c0-be269743c544'


def width_spec(source, width):
    result = deepcopy(source)
    _require(width in (128, 256) and source['residual_stages'] == 4 and not source['transpose'],
             'Require the retained S4/False gate geometry.')
    for key in ('decoders', 'stage_decoders'):
        for decoder in result[key]:
            _require(decoder['in_dim'] == 16 and decoder['out_dim'] == 32
                     and decoder['hidden_dim'] == 128 and decoder['num_res_blocks'] == 0
                     and decoder['decoder_type'] == 'symmetric', 'Unexpected historical S4 decoder.')
            decoder.update(hidden_dim=width, decoder_type='asymmetric')
    return result


def recipe(template, output, width):
    argv = list(template['training_argv'][2:])
    changes = {'decoder_type': 'asymmetric', 'decoder_base_ch': width, 'decoder_num_res_blocks': 0}
    for key, value in {'--output_dir': str(output / 'donor'),
                       **{'--' + k: 'default=' + str(v) for k, v in changes.items()}}.items():
        _require(argv.count(key) <= 1, 'Ambiguous inherited argument: ' + key)
        if key in argv:
            argv[argv.index(key) + 1] = value
        else:
            argv.extend([key, value])
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for key, value in changes.items():
        for section in ('canonical_config', 'vae_args'):
            expected[section][key]['default'] = value
        expected['resolved_category_runtime']['gate_proj'][key] = value
    expected['canonical_config']['explicit_cli_flags'] = sorted(set(expected['canonical_config']['explicit_cli_flags'])
        | {'--decoder_base_ch', '--decoder_num_res_blocks'})
    _require(normalized == expected, 'Only the declared decoder settings and output may change from native S4.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj'
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint
             and not cat.channel_refresh_after_category and cat.activation_calib_dataset == 'alpaca=1'
             and cat.activation_calib_nsamples == 512 and cat.activation_calib_seqlen == 256
             and cat.activation_calib_seed == 31 and cfg.steps == 10000
             and cfg.codebook_bits == 16 and cfg.codebook_dim == 32 and cfg.residual_stages == 4
             and cfg.base_ch == 128 and cfg.num_res_blocks == cfg.decoder_num_res_blocks == 0
             and cfg.decoder_type == 'asymmetric' and cfg.decoder_base_ch == width
             and cfg.channel_protect_count == 64 and vae.weight_rotation == 'two_sided'
             and vae.weight_rotation_block_size == 32 and vae.lr == .003
             and training.bf16 and not training.fp16, 'Fixed S4 scientific conditions changed.')
    return [sys.executable, 'tools/cat_train.py', *argv], normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Require fresh CPU-only snapshot preparation.')
    template = read(REFERENCE / 'launch.json'); check(template)
    reference = read(REFERENCE / 'audit.json'); meta = load_v6_meta(str(REFERENCE / 'final_model'))
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and template['checkpoint_id'] == reference['checkpoint_id'] == meta['checkpoint_id'] == REFERENCE_ID
             and template['base_checkpoint_id'] == reference['base_checkpoint_id'] == BASE_ID
             and all((REFERENCE / (name + '.exit')).read_text().strip() == '0'
                     for name in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed original S4 donor/assembly evidence.')
    old_score, _ = score([REFERENCE / (n + '.json') for n in ('mmlu', 'seven')],
                         template['architecture'], REFERENCE / 'final_model')
    _require(old_score['checkpoint_id'] == REFERENCE_ID
             and old_score['integer_counts'] == read(REFERENCE / 'full8_summary.json')['integer_counts'],
             'Historical raw S4 evidence changed.')
    base = Path(template['base_dir']); bm = load_v6_meta(str(base))
    _require(bm['checkpoint_id'] == BASE_ID and (base / bm['state_dict_file']).is_file(),
             'Retain the actual e767 mother checkpoint; do not substitute recovered cd338.')
    gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(len(gates) == 32 and all(s['stage_decoders'] == gates[0]['stage_decoders'] for s in gates),
             'Historical S4 gate geometry is not uniform.')
    before = construct(gates[0]); proof_sources, proof_inputs, reuse = prior_evidence()
    sources = {**template['source_sha256'], **proof_sources}
    for relative in ('experiments/llama_gate_decoder_width/prepare.py', 'experiments/llama_gate_transpose/audit.py'):
        p = ROOT / relative; sources[str(p)] = sha256(p)
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); public = ORIGINAL / p.relative_to(ROOT)
        _require(sha256(p) == sha256(public), 'Public and snapshot additions differ.')
        sources[str(p)] = sources[str(public)] = sha256(p)
    inputs = {**template['input_sha256'], **proof_inputs}
    for p in [*[REFERENCE / n for n in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')],
              REFERENCE / 'final_model/checkpoint_meta.json', REFERENCE / 'final_model/assembly_report.json']:
        inputs[str(p)] = sha256(p)
    records = {}
    for width in (128, 256):
        arm = output / ('h' + str(width)); expected_spec = width_spec(gates[0], width)
        after = construct(expected_spec); extra = (after['physical_bits'] - before['physical_bits']) * 32
        _require(after['state_tensors'] == before['state_tensors'] == 15
                 and extra == (0 if width == 128 else 13369344), 'Native BF16 decoder capacity differs.')
        components = deepcopy(reference['bit_budget']['components'])
        for key in ('storage_bits', 'budget_bits'):
            components['decoder'][key] += extra
        bits = sum(v['budget_bits'] for v in components.values())
        _require(bits == 17099800576 + extra and bits * 2 <= DENOMINATOR * 5
                 and bits == sum(v['storage_bits'] for v in components.values()), 'Invalid exact storage estimate.')
        argv, normalized = recipe(template, arm, width)
        record = {k: deepcopy(template[k]) for k in ('cwd', 'base_dir', 'base_checkpoint_id', 'architecture',
            'control_result_paths', 'protected_stats', 'environment', 'protocol')}
        record.update(schema='llama_gate_s4_width_v1', status='prepared_unbound', gpu=None, queued=False,
            automatic_retry=False, python_executable=sys.executable, output_dir=str(arm), model_type='llama',
            categories=['gate_proj'], decoder_hidden_dim=width, codebook_bits=16, residual_stages=4,
            transpose=False, block_size=32, steps_per_stage=10000, total_stage_updates=40000,
            full_evaluation=True, reference_dir=str(REFERENCE), reference_checkpoint_id=REFERENCE_ID,
            training_argv=argv, normalized_expected=normalized, source_sha256=sources, input_sha256=inputs,
            expected_budget_components=components, expected_user_linear_bits=bits,
            expected_state_tensors=3043, expected_code_banks=512, expected_gate_spec=expected_spec,
            budget_estimate={'user_linear_bits': bits, 'packed_linear_bits': bits, 'bpw': bits / DENOMINATOR,
                'decoder_extra_bits': extra, 'decoder_extra_parameters': extra // 16, 'code_padding_bits': 0,
                'boundary': 'CPU native one-gate storage construction scaled to32; final assembly measures every actual tensor.'},
            native_cpu_construction={'before': before, 'after': after},
            launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_s4_width.run',
                           '--launch-record', str(arm / 'launch.json')],
            validation_reuse={'native_s4': template['validation_reuse'], 'native_asymmetric_h256': reuse,
                'new_gpu_short': False, 'boundary': 'Existing native S4, H256 packed gradient/save, and asymmetric full8 paths are reused. This exact S4 asymmetric width pair has no measured quality or resource result.'},
            comparison_boundary='Fresh HF gate-only sensitivity, not live sequential CAT or cd338 recovery. Both new arms use asymmetric/encoder128R0, differ only decoder_base_ch and output. Wider initialization consumes a different RNG stream; not function-preserving. Retain identical e767 other192 projections/backbone. Historical13bd symmetric is contextual only. No full8 threshold substitutes for full21/nine-item evaluation.',
            experience={'sources': ['docs/experience/lessons/compression_choices.md', 'docs/experience/lessons/checkpoint_lifecycle.md'],
                'prior_llama_gate_s2_decoder256': {'full8_control': 59.91121590186278, 'full8_candidate': 59.463037292532505,
                    'path': str(RESULT / 'llama_gate_decoder_width/full8_summary.json'), 'boundary': 'K32/D32/S2/True and3577 complement; negative, no completeLB.'},
                'new_condition': 'K16/D32/S4/False with e767 complement, unchanged per-stage budget. No completed width pair at this geometry.',
                'scope': 'Codec width comparison; no inference of benefit before complete results; LB root-scheduled separately.'},
            resource_estimate={'estimated_chain_seconds': [1800, 2700],
                'historical_s4_chain_seconds': template['finished_unix'] - template['started_unix'],
                'disk_provision_bytes': template['resource_estimate']['single_arm_disk_provision_bytes'] + 3 * extra // 8,
                'filesystem_free_at_preparation': shutil.disk_usage(output.parent).free,
                'boundary': 'Per arm: native donor after_gate/final plus assembled final, approximately32.33GB decimal; two arms approximately64.66GB. No native saving protocol reduced. Exact H256 runtime/peak unmeasured; full21 excluded.'},
            cuda_initialized=False)
        check(record); records[width] = record
    left, right = (deepcopy(records[w]['normalized_expected']) for w in (128, 256))
    for record in (left, right):
        for section in ('canonical_config', 'cat_args'):
            record[section]['output_dir'] = '<arm>/donor'
        for section in ('canonical_config', 'vae_args'):
            record[section]['decoder_base_ch']['default'] = '<width>'
        record['resolved_category_runtime']['gate_proj']['decoder_base_ch'] = '<width>'
    _require(left == right and not torch.cuda.is_initialized(), 'Pair differs beyond decoder width/output or initialized CUDA.')
    for width, record in records.items():
        arm = Path(record['output_dir']); arm.mkdir(parents=True)
        write_json(arm / 'launch.prepared.json', record)
        script = '#!/usr/bin/env bash\nset -euo pipefail\n'
        for key, value in record['environment'].items():
            script += f'export {key}="{value}"\n'
        script += f'cd "{ROOT}"\npython -u -m experiments.llama_gate_s4_width.run --launch-record "{arm / "launch.json"}"\n'
        (arm / 'run.sh').write_text(script); (arm / 'run.sh').chmod(0o755)
    write_json(output / 'preparation.json', {'status': 'cpu_prepared_unbound', 'cuda_initialized': False,
        'base_checkpoint_id': BASE_ID, 'historical_s4_checkpoint_id': REFERENCE_ID,
        'matched_fields_verified': 'All normalized fields except decoder width and output are identical; both asymmetric.',
        'arms': {str(w): {'prepared': str(Path(r['output_dir']) / 'launch.prepared.json'),
            'sha256': sha256(Path(r['output_dir']) / 'launch.prepared.json'),
            'estimated_bpw': r['budget_estimate']['bpw'], 'disk_provision_bytes': r['resource_estimate']['disk_provision_bytes']}
            for w, r in records.items()}})
    print(read(output / 'preparation.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)

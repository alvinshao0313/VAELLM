"""Prepare fresh gateTrue with encoder H128/R0 and asymmetric decoder H256/R0."""
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
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, load_v6_meta

CONTROL = RESULT / 'llama_gate_transpose/transpose'
CONTROL_ID = 'd3abc0f2-baf4-4237-86e6-5918783024ed'
EXPECTED_BITS = 17109221376
PRIOR = ORIGINAL / 'result/qwen3_8b_w25_20260930'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    changes = {'decoder_type': 'asymmetric', 'decoder_base_ch': 256, 'decoder_num_res_blocks': 0}
    for key, value in {'--output_dir': str(output / 'donor'),
                       **{'--' + key: 'default=' + str(value) for key, value in changes.items()}}.items():
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
    expected['canonical_config']['explicit_cli_flags'] = sorted(set(
        expected['canonical_config']['explicit_cli_flags']) | {'--decoder_base_ch', '--decoder_num_res_blocks'})
    _require(normalized == expected, 'Only output and asymmetric decoder width selection may change.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,gate_proj,down_proj'
             and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 32
             and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and cfg.decoder_type == 'asymmetric' and cfg.decoder_base_ch == 256
             and cfg.decoder_num_res_blocks == 0 and cfg.channel_protect_count == 64
             and vae.weight_rotation == 'two_sided' and vae.weight_rotation_block_size == 32
             and vae.lr == .003 and vae.new_quant and training.bf16 and not training.fp16,
             'Keep the native gateTrue recipe and unchanged H128/R0 encoder.')
    for key, value in {'lfq_weight': 2.5, 'commitment_loss_weight': .25,
                       'inv_temperature': 100., 'entropy_loss_weight': .01}.items():
        _require(normalized['vae_args'][key] == value, 'Keep the primary control BSQ coefficient: ' + key)
    return argv, normalized


def estimate(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(meta['checkpoint_id'] == CONTROL_ID and len(specs) == 32,
             'Require the completed gateTrue metadata.')
    old = specs[0]['stage_decoders'][0]
    _require(all(len(s['stage_decoders']) == 2 and all(d == old for d in s['stage_decoders']) for s in specs)
             and old == {'in_dim': 32, 'out_dim': 32, 'hidden_dim': 128, 'num_res_blocks': 0,
                         'norm_type': 'layer', 'activation_type': 'swish', 'decoder_type': 'symmetric',
                         'use_checkpoint': True, 'param_dtype': 'bfloat16'}, 'Primary decoder specs differ.')
    new = {**old, 'hidden_dim': 256, 'decoder_type': 'asymmetric'}
    old_decoder = _build_decoder_from_spec(old); new_decoder = _build_decoder_from_spec(new)
    old_count = sum(p.numel() for p in old_decoder.parameters())
    new_count = sum(p.numel() for p in new_decoder.parameters())
    added = (new_count - old_count) * 32 * 2 * 16
    total = budget['user_linear_bits'] + added
    _require(old_count == 8608 and new_count == 17184 and added == 8781824
             and len(old_decoder.state_dict()) == len(new_decoder.state_dict()) == 6
             and not packed_symmetric_decoder_supports_fuse(new_decoder)
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17100439552
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
             and budget['budget_passed'] and not budget['contract_violations']
             and total == EXPECTED_BITS and total * 2 < 6979321856 * 5, 'Native decoder budget differs.')
    return new, {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'old_decoder_parameters_per_stage': old_count,
        'new_decoder_parameters_per_stage': new_count, 'decoder_delta_bits': added,
        'estimated_decoder_bits': budget['components']['decoder']['storage_bits'] + added,
        'expected_persistent_tensors': 2979, 'other_storage_components_unchanged': True,
        'boundary': 'Native BF16 decoder construction plus retained measured gateTrue budget; the final assembly still requires actual storage and source-union/reload audit.'}


def prior_evidence():
    evidence = {}; sources = {}; proofs = []
    for name in ('h256_d16', 'r1_d16'):
        path = PRIOR / 'validation' / name / 'launch.json'; proof = read(path)
        _require(proof['status'] == 'passed' and proof['exit_code'] == proof['pytest_exit_code'] == 0
                 and int((path.parent / 'run.exit').read_text()) == 0
                 and len(proof['checks']) == 2 and all(x['outcome'] == 'passed' for x in proof['checks']),
                 'The existing native packed training/persistence proof failed: ' + name)
        for relative, digest in proof['source_sha256'].items():
            if relative == 'rotation/hadamard_utils.py':
                continue  # The snapshot adds factors; this experiment retains the proven block32 rotation.
            source = ROOT / relative
            _require(sha256(source) == digest, 'Existing proof core source changed: ' + relative)
            sources[str(source)] = digest
        evidence[str(path)] = sha256(path)
        evidence[str(path.parent / 'run.exit')] = sha256(path.parent / 'run.exit')
        proofs.append({'path': str(path), 'configuration': proof['configuration']})
    historical = PRIOR / 'reconstruction_decoder256'
    outcome = read(historical / 'acceptance.json')
    normalized = Path(outcome['checkpoint_dir']).parent / 'normalized_cat_runtime_args.json'
    cfg = read(normalized)['resolved_category_runtime']['gate_proj']
    _require(int((historical / 'run.exit').read_text()) == 0
             and cfg['codebook_bits'] == cfg['codebook_dim'] == 32 and cfg['residual_stages'] == 2
             and cfg['base_ch'] == 128 and cfg['num_res_blocks'] == 0 and cfg['decoder_type'] == 'asymmetric'
             and cfg['decoder_base_ch'] == 256 and cfg['decoder_num_res_blocks'] == 1,
             'Historical asymmetric H256/R1 native recipe differs.')
    for item in outcome['input_results']:
        path = Path(item['path']); raw = read(path)
        _require(sha256(path) == item['sha256'] and raw['checkpoint_loader'] == 'v6'
                 and raw['strict_load'] and raw['checkpoint_id'] == outcome['checkpoint_id'],
                 'Historical asymmetric saved checkpoint evaluation evidence differs.')
        evidence[str(path)] = item['sha256']
    for path in (historical / 'acceptance.json', historical / 'run.exit', normalized):
        evidence[str(path)] = sha256(path)
    return sources, evidence, {'new_gpu_short': False, 'current_source_native_proofs': proofs,
        'historical_asymmetric': {'directory': str(historical), 'checkpoint_id': outcome['checkpoint_id'],
            'gate_spec': cfg, 'mean_percent': outcome['mean_percent'],
            'boundary': 'Historical same K32/D32/H256 asymmetric saved v6 strict forward/full8; decoderR1, full rotation/no protection and older source. Retired weights are not loaded and this does not establish a quality benefit or present-source identity.'},
        'coverage': 'H256/R0 proof covers real native packed first-linear, forward/gradients, B8192/36-model two-stage CAT updates and packed v6 persistence. R1 proof covers non-wholefused group8 sequential native prewarm. Symmetric/asymmetric share the same native Decoder and packed forward implementation; R0 uses an empty residual-block loop.',
        'boundary': 'The current-source proofs use K24/D16, whereas this gate uses K32/D32 and 32 models. These existing dimensions and the asymmetric branch have prior model deployment evidence; no new kernel or serialization layout is introduced. Runtime/quality of this exact new recipe remain unmeasured.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and all(int((CONTROL / (name + '.exit')).read_text()) == 0
                     for name in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed gateTrue control.')
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
             'Primary control raw results or native evidence differs.')
    check_hashes(review['evidence_sha256'])
    base = Path(template['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and (base / base_meta['state_dict_file']).is_file(), 'Keep the retained original3577 assembly base.')
    decoder_spec, budget = estimate(meta, audit['bit_budget'])
    argv, normalized = recipe(template, output)
    proof_sources, proof_evidence, reuse = prior_evidence()
    sources = {**template['source_sha256'], **proof_sources}
    for path in [*(ROOT / 'experiments/llama_gate_decoder_width').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_gate_decoder_width').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = [CONTROL / name for name in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')]
    evidence += [review_path, CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json']
    record = {key: deepcopy(template[key]) for key in ('cwd', 'architecture', 'base_dir', 'base_checkpoint_id',
        'control_result_paths', 'block_size', 'steps_per_stage', 'full_evaluation', 'protected_stats',
        'snapshot_import_paths', 'environment', 'protocol', 'dimension', 'transpose', 'codebook_bits')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type='llama', categories=['gate_proj'],
        hidden_dim=256, expected_decoder_spec=decoder_spec, expected_user_linear_bits=EXPECTED_BITS,
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        source_sha256=sources,
        input_sha256={**template['input_sha256'], **proof_evidence, **{str(p): sha256(p) for p in evidence}},
        historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        historical_control_result_paths=[str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        historical_control_audit_path=str(CONTROL / 'audit.json'), budget_estimate=budget,
        timeout_seconds=5400, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_decoder_width.run',
                       '--launch-record', str(output / 'launch.json')], validation_reuse=reuse,
        scope='Fresh HF all32 gateTrue/K32D32/S2, encoderH128R0/decoderH256R0 asymmetric, BF16/block32/input64 INT8, originalBSQ/MSE10Kx2/B8192/seed31/LR.003; retain3577 other192/backbone; no KD/LoRA.',
        comparison_boundary='Primary is fresh gateTrue59.91121590. Only decoder architecture capacity changes; asymmetric also selects existing native packed deployment instead of wholefused. Encoder shapes stay H128R0, but initialization RNG consumption and optimization trajectories change. This does not isolate capacity from its numerical/runtime effects. The3577 complement contains prior recovery.',
        experience={'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Llama D16/down symmetricH256 hurt mean1.2591873pp; Qwen whole-model capacity increases also lacked matched gains.',
            'difference': 'This is Llama single gateTrue and only the decoder is widened; no previous matched result.',
            'action': 'Preserve primary BSQ/protection/signs and compare complete eight tasks; do not assume more capacity helps.'},
        resource_estimate={'seconds_limit': 5400,
            'matched_gate_true_chain_seconds': template['finished_unix'] - template['started_unix'],
            'boundary': 'H256 native packed prewarm is slower than wholefused; exact recipe runtime/peak unmeasured. One explicitly bound GPU, no retry or queue.'},
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

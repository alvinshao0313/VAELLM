"""CPU prepare fresh Llama Q/K block32/full with the measured e767 complement."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, check, read
from experiments.full_rht_gate_pair.compare import normalized_recipe, score
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta

BASE = RESULT / 'llama_fineweb1600_down_d16_combo/final_model'
BASE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
TEMPLATE = RESULT / 'llama_q_up_transpose/q_true/launch.json'
CATEGORIES = ['q_proj', 'k_proj']
BLOCKS = {'block32': 32, 'full': 0}


def recipe(template, output, block):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--compression_categories': 'q_proj,k_proj',
                       '--weight_rotation_block_size': str(block)}.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('cat_args', 'canonical_config'): expected[section]['output_dir'] = str(output / 'donor')
    expected['cat_args']['compression_categories'] = 'q_proj,k_proj'
    expected['canonical_config']['compression_categories'] = CATEGORIES
    expected['resolved_category_runtime']['k_proj'] = {**expected['resolved_category_runtime']['q_proj'], 'category': 'k_proj'}
    expected['vae_args']['weight_rotation_block_size'] = block
    expected['canonical_config']['core_template']['weight_rotation_block_size'] = block
    _require(normalized == expected, 'Native recipe changed beyond categories, output and rotation block.')
    _require(cat.target_layers == 'all' and cat.linear_group_size == 32 and cat.batch_size == 8192
             and cat.seed == 31 and cat.deterministic and not cat.resume_from_checkpoint
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.channel_axis == 'input' and cat.channel_quant == 'int8'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and training.bf16 and not training.fp16,
             'Wrong fresh native Llama Q/K recipe.')
    _require(all(c.steps == 10000 and c.codebook_bits == c.codebook_dim == 32 and c.residual_stages == 2
             and c.base_ch == 128 and c.num_res_blocks == 0 and c.channel_protect_count == 64
             and c.decoder_type == 'symmetric' and c.recon_loss_type == 'mse' for c in resolved.values()), 'Wrong resolved Q/K geometry.')
    return argv, normalized


def geometry(meta, budget, block):
    selected = {s['name']: deepcopy(s) for s in meta['converted_modules'] if s['name'].rsplit('.', 1)[-1] in CATEGORIES}
    _require(len(meta['converted_modules']) == 224 and len(selected) == 64
             and meta['checkpoint_id'] == budget['checkpoint_id'] == BASE_ID and budget['budget_passed']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0,
             'Wrong actual e767 inventory/ledger.')
    native = {}
    for category in CATEGORIES:
        specs = [s for s in selected.values() if s['name'].endswith('.' + category)]; out = 4096 if category == 'q_proj' else 1024
        groups = out * 4032 // 32
        for s in specs:
            _require(s['in_features'] == 4096 and s['out_features'] == out and s['compressed_in_features'] == 4032
                     and s['compressed_out_features'] == out and s['transpose'] is (category == 'q_proj')
                     and s['stage_codebook_dims'] == [32, 32] and s['residual_stages'] == 2
                     and s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                     and s['protected_input_qvalues'] == {'shape': [64, out], 'dtype': 'uint8'}
                     and s['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                     and s['protected_channel_quant_format'] == 'int8' and s['protected_output_indices'] is None
                     and s['weight_rotation']['block_size'] == 32, 'Wrong actual source Q/K geometry.')
            _require(all(p['shape'] == [groups, 1, 4] and p['logical_shape'] == [groups, 1, 32]
                     and p['storage_format'] == 'bitpack_u8' for p in s['stage_vq_weights'])
                     and all(d['in_dim'] == d['out_dim'] == 32 and d['hidden_dim'] == 128
                     and d['num_res_blocks'] == 0 and d['decoder_type'] == 'symmetric'
                     and d['param_dtype'] == 'bfloat16' for d in s['stage_decoders']), 'Wrong packed/decoder spec.')
            s['weight_rotation']['block_size'] = block
        sample = deepcopy(specs[0]); sample['name'] = category
        with torch.random.fork_rng(devices=[]):
            holder = torch.nn.Module(); holder.add_module(category, torch.nn.Linear(4096, out, bias=False))
            _rebuild_converted_modules(holder, [sample]); state = holder.state_dict()
            bits = sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}.values()) * 8
            _require(_collect_vae_linear_specs(holder) == [sample] and len(state) == 13
                     and bits == budget['per_target'][specs[0]['name']]['storage_bits'], 'Native CPU reconstruction/storage differs.')
            native[category] = {'count': len(specs), 'transpose': sample['transpose'], 'groups_per_stage': groups,
                'native_persistent_tensors_per_projection': len(state), 'physical_bits_per_projection': bits,
                'input_full_hadamard_factor': 252, 'input_power2_factor': 16}
    return selected, native


def prepare(arm, output):
    output = Path(output).resolve(); block = BLOCKS[arm]
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized() and not output.exists(),
             'Require a fresh CPU-only snapshot preparation.')
    template = read(TEMPLATE); _require(template['status'] == 'completed' and template['exit_code'] == 0
        and template['model_type'] == 'llama', 'Require the completed native Llama QTrue recipe.')
    meta = load_v6_meta(str(BASE)); assembly = read(BASE / 'assembly_report.json'); budget = assembly['bit_budget']
    _require(assembly['status'] == 'verified' and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == 2979,
             'Require the measured native e767 union/save/fresh proof.')
    control_paths = [BASE.parent / (s + '.json') for s in ('mmlu', 'seven')]
    baseline, _ = score(control_paths, template['architecture'], BASE)
    _require(baseline['checkpoint_id'] == BASE_ID and baseline['mean_percent'] == read(BASE.parent / 'review.json')['mean_percent'], 'Wrong base full8.')
    expected_specs, native = geometry(meta, budget, block); argv, normalized = recipe(template, output, block)
    _, other = recipe(template, output, 32 if block == 0 else 0)
    _require(normalized_recipe({'normalized_expected': normalized, 'output_dir': str(output), 'block_size': block})
             == normalized_recipe({'normalized_expected': other, 'output_dir': str(output), 'block_size': 32 if block == 0 else 0}), 'Pair differs beyond block/output.')
    validation = RESULT / 'full_rht_gate_pair/native_validation_r1/verification.json'; proof = read(validation)
    _require(proof['status'] == 'verified' and proof['source_root'] == str(ROOT)
             and (validation.parent / 'run.exit').read_text().strip() == '0', 'Existing native H252 proof is incomplete.')
    check_hashes(proof['source_sha256'])
    inputs = dict(template['input_sha256']); sources = {**template['source_sha256'], **proof['source_sha256']}
    evidence = [TEMPLATE, validation, validation.parent / 'run.exit', BASE / 'checkpoint_meta.json', BASE / 'assembly_report.json',
        BASE.parent / 'review.json', *control_paths, RESULT / 'qwen_qk_rotation_pair/comparison.json',
        RESULT / 'llama_up_rotation_pair/full/audit.json', RESULT / 'llama_d1200_retrieval_attribution/independent_review.json',
        RESULT / 'llama_best1600_d16_retrieval400/review.json']
    for p in evidence: inputs[str(p)] = sha256(p)
    for p in [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'experiments/llama_gate_up_rate_pair/audit.py',
              ROOT / 'experiments/llama_gate_up_rate_pair/common.py']:
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'New/helper public and S source differ.')
    stats = [s for s in template['protected_stats'] if '/result/' not in s['path']]
    stats.append(file_stat(BASE / meta['state_dict_file']))
    index = Path(meta['base_model_path']) / 'model.safetensors.index.json'; inputs[str(index)] = sha256(index)
    dense_bytes = read(index)['metadata']['total_size']; tensor_bytes = {}; current = dense_bytes
    for category in CATEGORIES:
        specs = [s for s in expected_specs.values() if s['name'].endswith('.' + category)]
        current += sum(budget['per_target'][s['name']]['storage_bits'] // 8 - s['in_features'] * s['out_features'] * 2 for s in specs)
        tensor_bytes['after_' + category] = current
    tensor_bytes.update(final_donor=current, assembled=budget['full_model_tensor_bytes'])
    provision = sum(tensor_bytes.values()) + 2 * 1024**3
    record = {'schema': 'llama_qk_rotation_pair_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'llama', 'arm': arm,
        'categories': CATEGORIES, 'codebook_bits': dict.fromkeys(CATEGORIES, 32), 'block_size': block,
        'architecture': template['architecture'], 'base_dir': str(BASE), 'base_checkpoint_id': BASE_ID,
        'control_result_paths': [str(p) for p in control_paths], 'steps_per_stage': 10000, 'full_evaluation': True,
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'expected_target_specs': expected_specs, 'native_cpu_geometry': native,
        'expected_user_linear_bits': budget['user_linear_bits'], 'expected_budget_components': budget['components'],
        'budget_estimate': {'total_bits': budget['user_linear_bits'], 'bpw': budget['user_linear_bpw'],
            'denominator': budget['original_linear_weights'], 'padding_bits': 0,
            'selected_bits': sum(budget['per_target'][n]['budget_bits'] for n in expected_specs),
            'boundary': 'Actual e767 ledger plus native CPU reconstructed geometry; full Hadamard bases nonpersistent. Actual assembled audit still required.'},
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': stats, 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_qk_rotation_pair.run', '--launch-record', str(output / 'launch.json')],
        'category_rng': {'order': CATEGORIES, 'shuffle_seeds_by_stage': {'q_proj': [31, 32], 'k_proj': [100031, 100032]}},
        'comparison_boundary': 'Primary full minus new fresh block32; e767 secondary because both replace restored Q/K with fresh HF Q/K. Actual pair calibration inputs, protection and signs must match; activation statistics are diagnostic. No LoRA. Conditional BF16 Q/K rescue on16 is motivation, not rotation evidence or predicted gain.',
        'validation_reuse': {'native_full_h252': str(validation), 'no_new_gpu_short': True,
            'boundary': 'Existing native full4032 H252 and transpose4096x4032 plus nontranspose1024x4032 paths. Qwen quality is not transferred to Llama; no historical donor weights loaded.'},
        'protocol': {**template['protocol'], 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
        'future_retrieval': {'required_for_both_arms_regardless_full8': True, 'prepared': False, 'gpu': None, 'automatic_queue': False,
            'tasks': ['passage_retrieval_en', 'passage_retrieval_zh'], 'rows_per_task': 200, 'input_limit': 32768,
            'protocol': 'Original official Llama SDPA/FP32RoPE/EOS/archive/prompt/output settings; independently bind after full8. Only2/21 tasks.',
            'timeout_seconds': 900, 'termination_grace_seconds': 20},
        'resource_estimate': {'estimated_seconds': [1700, 2100], 'seconds_limit': 2700,
            'checkpoint_tensor_bytes': tensor_bytes, 'tensor_bytes_total': sum(tensor_bytes.values()),
            'archive_runtime_tokenizer_headroom_bytes': 2 * 1024**3, 'single_arm_disk_provision_bytes': provision,
            'two_arm_disk_provision_bytes': 2 * provision, 'filesystem_free_at_preparation': shutil.disk_usage(RESULT).free,
            'boundary': 'Geometry estimate, not measured archive sizes. Three partial donors and one assembled model retained through pair review; root rechecks future free space/owner deadline at binding.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.'); check(record)
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'arm': arm, 'prepared_sha256': sha256(output / 'launch.prepared.json'), 'sources': len(sources),
        'inputs': len(inputs), 'stats': len(stats), 'bpw': budget['user_linear_bpw'], 'disk_bytes': provision, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--arm', choices=BLOCKS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True); args = parser.parse_args(); prepare(args.arm, args.output_dir)

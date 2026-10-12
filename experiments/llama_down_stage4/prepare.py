"""Prepare same nominal3bit K12/S4 down against e767's untouched fresh K24/S2."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_down_dimension.compare import arm
from experiments.llama_down_stage4.audit import stage4_spec, TOTAL_BITS, PACKED_BITS, PADDING_BITS, DOWN_ID
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import file_stat, sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta

CONTROL = RESULT / 'llama_down_dimension/block32'


def construct(source):
    with torch.random.fork_rng(devices=[]):
        holder = torch.nn.Module(); holder.add_module('down_proj', torch.nn.Linear(14336, 4096, bias=False))
        expected = deepcopy(source); expected['name'] = 'down_proj'
        _rebuild_converted_modules(holder, [expected]); actual = _collect_vae_linear_specs(holder)[0]
        _require(actual == expected, 'Actual original-size native geometry reconstruction differs.')
        state = holder.state_dict(); storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}
        decoder = holder.down_proj._parallel_stage_decoder
        return {'spec': actual, 'physical_bits': sum(storages.values()) * 8, 'state_tensors': len(state),
            'decoder_elements': sum(p.numel() for p in decoder.parameters()),
            'scope': 'CPU original-size native geometry only; no control learned weights regenerated.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not torch.cuda.is_initialized(), 'Require a fresh CPU-only S preparation.')
    torch.set_num_threads(4)
    template, control_audit, original_score, _, old_evidence = arm(CONTROL, 32)
    _require(original_score['checkpoint_id'] == DOWN_ID, 'Wrong original fresh down source.')
    source = BASE.parent; meta = load_v6_meta(str(BASE)); assembly = read(BASE / 'assembly_report.json')
    review = read(source / 'review.json'); terminal = read(source / 'launch.json')
    base_score, _ = score([source / (n + '.json') for n in ('mmlu', 'seven')], template['architecture'], BASE)
    _require(meta['checkpoint_id'] == assembly['checkpoint_id'] == review['checkpoint_id'] == terminal['checkpoint_id']
             == base_score['checkpoint_id'] == BASE_ID and assembly['status'] == review['status'] == 'verified'
             and terminal['status'] == 'completed' and terminal['exit_code'] == 0
             and assembly['sources']['donor_checkpoint_id'] == DOWN_ID
             and assembly['sources']['donor_checkpoint_dir'] == str(CONTROL / 'final_model')
             and assembly['sources']['categories'] == ['down_proj']
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == 2979
             and review['native_proof']['metadata_specs_exact_source_union']
             and review['native_proof']['reused_union_save_fresh_evidence'], 'e767 must contain the unchanged fresh S2 down tensors.')
    down = [s for s in meta['converted_modules'] if s['name'].endswith('.down_proj')]
    original_specs = {s['name']: s for s in load_v6_meta(str(CONTROL / 'final_model'))['converted_modules']}
    _require(len(down) == 32 and all(s == original_specs[s['name']] for s in down)
             and {s['name'] for s in down} == set(assembly['sources']['replaced_targets']), 'The32 control down specs differ.')
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--codebook_bits': 'default=32,cat:down_proj=12',
                       '--residual_stages': 'default=2,cat:down_proj=4'}.items():
        _require(argv.count(key) == 1, 'Ambiguous option: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv); resolved = resolve_category_runtime_configs(cat, vae, ['down_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'):
        expected[section]['codebook_bits'] = {'default': 32, 'by_category': {'down_proj': 12}}
        expected[section]['residual_stages'] = {'default': 2, 'by_category': {'down_proj': 4}}
    expected['resolved_category_runtime']['down_proj'].update(codebook_bits=12, residual_stages=4)
    _require(normalized == expected, 'Only K/S factorization and output may change from the actual fresh S2 recipe.')
    cfg = resolved['down_proj']
    _require(cat.seed == 31 and argv[argv.index('--data_seed') + 1] == '31' and cat.deterministic
             and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.target_layers == 'all'
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint and not cat.channel_refresh_after_category
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and cfg.steps == 10000
             and cfg.codebook_dim == 16 and cfg.channel_protect_count == 256 and cfg.base_ch == 128
             and cfg.num_res_blocks == 0 and training.bf16 and not training.fp16,
             'Fixed native source/seed/geometry/optimizer conditions differ.')
    before, after = construct(down[0]), construct(stage4_spec(down[0])); budget = assembly['bit_budget']
    extra_decoder = (after['decoder_elements'] - before['decoder_elements']) * 32 * 16
    components = deepcopy(budget['components'])
    for field in ('budget_bits', 'storage_bits'): components['decoder'][field] += extra_decoder
    components['code']['storage_bits'] += PADDING_BITS; components['code']['unique_storages'] += 64
    _require(before['decoder_elements'] == 11040 and after['decoder_elements'] == 15936
             and extra_decoder == 2506752 and before['state_tensors'] == 13 and after['state_tensors'] == 15
             and (after['physical_bits'] - before['physical_bits']) * 32 == extra_decoder + PADDING_BITS
             and before['physical_bits'] == budget['per_target'][down[0]['name']]['storage_bits']
             and sum(v['budget_bits'] for v in components.values()) == TOTAL_BITS
             and sum(v['storage_bits'] for v in components.values()) == PACKED_BITS, 'Actual native CPU geometry/storage differs.')
    parity = RESULT / 'llama_gate_stage4/s4/independent_review.json'; parity_record = read(parity)
    _require(parity_record['status'] == 'independent_cpu_verified', 'Require existing actual S4 native/group8/full8 proof.')
    sources = dict(template['source_sha256']); inputs = {**template['input_sha256'], **old_evidence}
    pending = list(Path(__file__).parent.glob('*.py')); seen = set()
    for p in pending:
        _require(p.read_bytes() == (ORIGINAL / p.relative_to(ROOT)).read_bytes(), 'Public/S new sources differ.')
    while pending:
        p = pending.pop().resolve()
        if p in seen: continue
        seen.add(p); tree = ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                       [i.name for i in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                path = ROOT / (module.replace('.', '/') + '.py')
                if path.is_file(): pending.append(path)
    evidence = [parity, *[source / n for n in ('launch.json', 'review.json', 'full8_summary.json', 'mmlu.json', 'seven.json',
        'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')],
        BASE / 'assembly_report.json', BASE / 'checkpoint_meta.json', BASE / 'config.json', CONTROL.parent / 'comparison.json']
    for p in evidence: inputs[str(p)] = sha256(p)
    stats = list(template['protected_stats']); stats.append(file_stat(BASE / meta['state_dict_file']))
    donor_bytes = template['architecture']['dense_parameter_elements'] * 2 - 32 * 4096 * 14336 * 2 + 32 * after['physical_bits'] // 8
    final_bytes = budget['full_model_tensor_bytes'] + (extra_decoder + PADDING_BITS) // 8
    disk = 2 * donor_bytes + final_bytes + 2 * 1024**3
    record = {'schema': 'llama_down_s4_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'llama',
        'categories': ['down_proj'], 'codebook_bits': 12, 'residual_stages': 4, 'transpose': True, 'block_size': 32,
        'steps_per_stage': 10000, 'total_stage_updates': 40000, 'full_evaluation': True, 'base_dir': str(BASE),
        'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'], 'fresh_down_control_dir': str(CONTROL),
        'fresh_down_control_checkpoint_id': DOWN_ID, 'control_result_paths': [str(source / (n + '.json')) for n in ('mmlu', 'seven')],
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': stats,
        'environment': template['environment'], 'protocol': template['protocol'], 'expected_budget_components': components,
        'expected_user_linear_bits': TOTAL_BITS, 'expected_packed_linear_bits': PACKED_BITS, 'expected_state_tensors': 3043,
        'expected_code_banks': 512, 'native_cpu_construction': {'before': before, 'after': after},
        'budget_estimate': {'user_linear_bits': TOTAL_BITS, 'packed_linear_bits': PACKED_BITS,
            'user_linear_bpw': TOTAL_BITS / DENOMINATOR, 'packed_linear_bpw': PACKED_BITS / DENOMINATOR,
            'code_padding_bits': PADDING_BITS, 'decoder_extra_bits': extra_decoder},
        'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_down_stage4.run', '--launch-record', str(output / 'launch.json')],
        'category_rng': {'category_index': 0, 'group_offset': 0, 'base_seed': 31, 'stage_shuffle_seeds': [31, 32, 33, 34],
            'boundary': 'Same native seed rule. K and stage count change parameter/RNG consumption; no added reset or identical-trajectory claim.'},
        'validation_reuse': 'Native S4/category save/group8/full8 and arbitrary-K bitpack/masked kernels; no changed core algorithm or repeated GPU short. This run must verify actual K12 shape/high4 padding plus full3043 source union. No E2E compatibility claim.',
        'comparison_boundary': 'Primary candidate minus existing e767: retain identical192 projections/backbone and replace its untouched fresh K24/D16/S2 down with fresh same-HF K12/D16/S4. Same3bit main code rate; four10000-stage schedules mean40000 versus20000 updates, extra decoder capacity and padding, changed RNG. Not equal compute, storage or pure stage-count causality. Original S2 calibration cache was retired; compare actual protection/sign hashes and same pinned calibration recipe, not absent old tokens. No LongBench claim.',
        'resource_estimate': {'estimated_full_chain_seconds': [1800, 2500], 'seconds_limit': 3600,
            'historical_s2_full_seconds': template['finished_unix'] - template['started_unix'],
            'historical_s2_train_seconds': template['stages']['train']['finished_unix'] - template['stages']['train']['started_unix'],
            'donor_tensor_bytes_each': donor_bytes, 'assembled_tensor_bytes': final_bytes,
            'single_arm_disk_provision_bytes': disk, 'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'boundary': 'Two CAT saves plus assembled endpoint and2GiB overhead. Training stages double; setup/calibration/full8 do not. Binding must check all concurrent future outputs.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > disk, 'CUDA initialized or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'sources': len(sources), 'inputs': len(inputs),
           'stats': len(stats), 'logical_bits': TOTAL_BITS, 'packed_bits': PACKED_BITS, 'disk_bytes': disk, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)

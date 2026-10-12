"""Prepare fixed K8/D32/S8 gate in a private two-line kernel snapshot."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_stage4.prepare import construct
from experiments.llama_gate_stage8.audit import stage8_spec, TOTAL_BITS, CONTROL_ID
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import sha256, write_json, check_hashes, check_stat
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta

CONTROL = RESULT / 'llama_gate_stage4/s4'
PROBE = RESULT / 'llama_gate_s8_kernel_probe/attempt2'


def prepare(output):
    output = Path(output).resolve(); manifest_path = ROOT.parent / 'snapshot_manifest.json'
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Require fresh CPU-only snapshot preparation.')
    manifest = read(manifest_path); check_hashes(manifest['snapshot_sha256'])
    _require(manifest['snapshot_root'] == str(ROOT) and len(manifest['core_changes']) == 1
             and manifest['core_changes'][0]['replacement_count'] == 2, 'Require only the proven two-line kernel change.')
    template = read(CONTROL / 'launch.json'); check_hashes(template['source_sha256']); check_hashes(template['input_sha256'])
    _require(template['status'] == 'completed' and template['exit_code'] == 0 and template['checkpoint_id'] == CONTROL_ID
             and all((CONTROL / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')), 'Require completed native S4 False control.')
    control, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    audit = read(CONTROL / 'audit.json'); summary = read(CONTROL / 'full8_summary.json')
    _require(control['checkpoint_id'] == audit['checkpoint_id'] == CONTROL_ID and audit['status'] == 'verified'
             and control['integer_counts'] == summary['integer_counts'] and summary['audit_sha256'] == sha256(CONTROL / 'audit.json')
             and control['mean_percent'] == template['mean_percent'] == 61.69747742212227, 'Control raw evidence differs.')
    proof = read(PROBE / 'report.json'); launch = read(PROBE / 'launch.json')
    _require(sha256(PROBE / 'report.json') == 'cedf36f05530ba2ba25919652ff6c10fecffea8d420c89da35d1fa6b0e11c5d1'
             and proof['status'] == 'verified_isolated_k8_kernel' and launch['status'] == 'completed' and launch['exit_code'] == 0
             and all((PROBE / (s + '.exit')).read_text().strip() == '0' for s in ('run', 'outer'))
             and sha256(ROOT / 'litebsq/fused_multistage_decoder.py') == launch['isolated_module_sha256'], 'K8 GPU proof/kernel mismatch.')
    check_hashes(launch['source_sha256']); check_hashes(launch['input_sha256'])
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--codebook_bits': 'default=32,cat:gate_proj=8',
                       '--residual_stages': 'default=2,cat:gate_proj=8', '--vae_steps': 'default=5000'}.items():
        _require(argv.count(key) == 1, 'Ambiguous argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv); resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'):
        expected[section]['codebook_bits'] = {'default': 32, 'by_category': {'gate_proj': 8}}
        expected[section]['residual_stages'] = {'default': 2, 'by_category': {'gate_proj': 8}}
    expected['canonical_config']['vae_steps'] = {'default': 5000}
    expected['resolved_category_runtime']['gate_proj'].update(codebook_bits=8, residual_stages=8, steps=5000)
    _require(normalized == expected, 'Recipe changed beyond K/S/steps/output.')
    cfg = resolved['gate_proj']
    _require(cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and cat.after_category_mode == 'none'
             and not cat.resume_from_checkpoint and cfg.steps == 5000 and cfg.codebook_dim == 32
             and cfg.channel_protect_count == 64 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and training.bf16 and not training.fp16, 'Scientific conditions changed.')
    gates = [s for s in load_v6_meta(str(CONTROL / 'final_model'))['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(len(gates) == 32 and all(s['stage_decoders'] == gates[0]['stage_decoders'] for s in gates), 'Wrong control geometry.')
    before, after = construct(gates[0]), construct(stage8_spec(gates[0])); extra = 32 * (after['physical_bits'] - before['physical_bits'])
    components = deepcopy(audit['bit_budget']['components'])
    for field in ('budget_bits', 'storage_bits'): components['decoder'][field] += extra
    components['code']['unique_storages'] += 128
    count = audit['source_union_save_fresh_tensors'] + 32 * (after['state_tensors'] - before['state_tensors'])
    _require(before['physical_bits'] == audit['bit_budget']['per_target'][gates[0]['name']]['storage_bits']
             and before['state_tensors'] == 15 and after['state_tensors'] == 19 and count == 3171 and extra == 9240576
             and sum(v['budget_bits'] for v in components.values()) == sum(v['storage_bits'] for v in components.values()) == TOTAL_BITS,
             'Actual native S8 geometry/budget differs.')
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_cat_packed_decoder import test_cat_conversion_packed_roundtrip
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='llama_s8_v6_') as temporary:
        test_cat_conversion_packed_roundtrip(Path(temporary), kind='final_model', decoder_type='symmetric', stages=8, parts=1)
        tiny = load_v6_meta(str(Path(temporary) / 'final_model'))
        banks = sum(1 for spec in tiny['converted_modules'] for _ in _code_specs(spec))
        _require(len(tiny['converted_modules']) == 1 and tiny['converted_modules'][0]['residual_stages'] == banks == 8, 'Fixture did not persist S8.')
    cpu_proof = {'status': 'passed', 'function': 'test_cat_conversion_packed_roundtrip', 'kwargs': {'kind': 'final_model', 'decoder_type': 'symmetric', 'stages': 8, 'parts': 1},
        'persisted_stages': 8, 'persisted_code_banks': banks, 'elapsed_seconds': time.time() - started,
        'scope': 'Existing native serial/packed forward, v6 save/fresh strict all-state equality and one decoder SGD update. Tiny Linear4x4/FP32/K64D4/H8R1; not BF16K8/fullsize/80GB evidence.'}
    sources = dict(manifest['snapshot_sha256'])
    old_root = Path(template['cwd'])
    for name, digest in template['source_sha256'].items():
        if not Path(name).is_relative_to(old_root): sources[name] = digest
    inputs = dict(template['input_sha256'])
    native = Path(template['donor_checkpoint_dir'])
    for p in [manifest_path, ROOT.parent / 'kernel.patch', *[CONTROL / n for n in ('launch.json', 'audit.json', 'full8_summary.json', 'independent_review.json', 'mmlu.json', 'seven.json')],
              CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json', native.parent / 'normalized_cat_runtime_args.json',
              *[PROBE / n for n in ('launch.prepared.json', 'launch.json', 'report.json', 'run.exit', 'outer.exit')]]:
        inputs[str(p)] = sha256(p)
    # The old original3577 payload is historical evidence, not an input to fresh HF gate + e767 assembly.
    protected = [s for s in template['protected_stats'] if '/remaining_prefix_block32_s500/' not in s['path']]
    for item in protected: check_stat(item)
    environment = dict(template['environment']); environment['PYTHONPATH'] = str(ROOT)
    provision = 33_000_000_000
    record = {'schema': 'llama_gate_s8_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'llama', 'categories': ['gate_proj'],
        'codebook_bits': 8, 'residual_stages': 8, 'transpose': False, 'block_size': 32, 'steps_per_stage': 5000, 'total_stage_updates': 40000,
        'full_evaluation': True, 'base_dir': template['base_dir'], 'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': CONTROL_ID, 'control_result_paths': template['control_result_paths'],
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': protected, 'environment': environment, 'protocol': template['protocol'],
        'expected_budget_components': components, 'expected_user_linear_bits': TOTAL_BITS, 'expected_state_tensors': count, 'expected_code_banks': 640,
        'budget_estimate': {'user_linear_bits': TOTAL_BITS, 'packed_linear_bits': TOTAL_BITS, 'bpw': TOTAL_BITS / DENOMINATOR,
            'code_padding_bits': 0, 'decoder_extra_bits': extra, 'decoder_extra_parameters': extra // 16},
        'native_cpu_construction': {'before': before, 'after': after}, 's8_cpu_v6_roundtrip': cpu_proof,
        'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_stage8.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'k8_gpu_report': str(PROBE / 'report.json'), 'k8_gpu_scope': proof['scope'], 'native_stage8_fixture': cpu_proof,
            'core_patch': manifest['core_changes'], 'new_gpu_short': False, 'full_model_reloaded_at_preparation': False},
        'comparison_boundary': 'Fresh HF gate K8/D32/S8/5000 each versus K16/D32/S4/10000 each, same40000 updates and2bit main code. All other192 projections from e767. Decoder capacity/init/RNG/linear scheduler horizon per stage and FLOPs differ; equal updates are not equal convergence. No quality conclusion before full8; full LongBench and nine-item score remain unknown. No MMLU or eight-task quality floor.',
        'resource_estimate': {'estimated_seconds': [1800, 2800], 'seconds_limit': 3600, 'single_arm_disk_provision_bytes': provision,
            'measured_s4_full_seconds': template['finished_unix'] - template['started_unix'],
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'group8_elements': {'decoded_codes': 8 * 1806336 * 8 * 8, 'stacked_codes': 8 * 1806336 * 8 * 8, 'outputs': 8 * 1806336 * 32},
            'boundary': 'B*S*K and output element counts equal S4; no save_aux layered inference does not allocate B*S*H. Stage-local CAT B8192/H128 shape unchanged; more stage conversions. K8GPU proof B256 only; no fullsize runtime claim.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision, 'CUDA initialized or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources), 'input': len(inputs), 'stat': len(protected),
           'bits': TOTAL_BITS, 'state': count, 'banks': 640, 'cpu_fixture': cpu_proof, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)

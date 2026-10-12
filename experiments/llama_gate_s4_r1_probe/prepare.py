"""CPU-only preparation of the fixed all-gate S4 R1 validation and paired400 run."""
import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_s4_r1_probe.common import (
    CONTROL, DESIGN, GATE_NAMES, OUTPUT, PREFIXES, PUBLIC, R2, ROOT, SOURCE, SOURCE_ID,
    actual_configuration, arguments, check, configuration, expected_specs, read, require,
    sha256, state_inventory, write_json)
from experiments.longbench_v1.common import file_stat
from experiments.qwen3_w25.assemble_categories import _load_state
from litebsq.autoencoder import Decoder


def source_pins(original):
    sources, inputs = {}, {}
    for raw, digest in original['source_sha256'].items():
        path = Path(raw)
        mapped = path if path.is_relative_to(ROOT) else ROOT / path.relative_to(PUBLIC) if path.is_relative_to(PUBLIC) else path
        chosen = mapped if mapped.is_file() else path
        require(sha256(chosen) == digest, 'Existing native source differs: ' + str(chosen))
        (sources if mapped.is_file() else inputs)[str(chosen)] = digest
    pending, seen = [*Path(__file__).parent.glob('*.py'), ROOT / 'tools/cat_eval.py'], set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path); sources[str(path)] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                     [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                local = ROOT / (name.replace('.', '/') + '.py')
                if local.is_file():
                    pending.append(local)
    for path in Path(__file__).parent.glob('*.py'):
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S source differs.')
    return sources, inputs


def prepare():
    require(ROOT != PUBLIC and Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use actual S with hidden CUDA for CPU preparation.')
    torch.set_num_threads(4)
    for name in ('common', 'migrate', 'train', 'export', 'run'):
        importlib.import_module('experiments.llama_gate_s4_r1_probe.' + name)
    original = read(CONTROL); meta = read(SOURCE / 'checkpoint_meta.json')
    audit_path, review_path = SOURCE.parent / 'audit.json', SOURCE.parent / 'independent_review.json'
    audit, review = read(audit_path), read(review_path)
    require(meta['checkpoint_id'] == audit['checkpoint_id'] == SOURCE_ID and audit['status'] == 'verified'
            and review['status'] == 'verified_completed_step_full8', 'Fixed cd338 source proof is incomplete.')
    specs = expected_specs(meta, 'r0')
    gates = [x for x in specs if x['name'] in GATE_NAMES]
    require(len(gates) == 32 and len(specs) == 224, 'Wrong complete projection/gate count.')
    for spec in gates:
        require(spec['weight_rotation']['block_size'] == 32 and spec['protected_input_indices']['shape'] == [64]
                and all((x['in_dim'], x['out_dim'], x['hidden_dim'], x['num_res_blocks'], x['norm_type'],
                         x['activation_type'], x['decoder_type']) == (16, 32, 128, 0, 'layer', 'swish', 'symmetric')
                        for x in spec['stage_decoders']), 'Source is not the fixed S4 gate geometry.')
    source = _load_state(SOURCE)
    r0 = state_inventory(source)
    small = Decoder(in_dim=16, out_dim=32, hidden_dim=128, num_res_blocks=1, norm_type='layer',
        activation_type='swish', decoder_type='symmetric', use_checkpoint=True, num_models=4).bfloat16()
    added = {prefix + key: value for prefix in PREFIXES for key, value in small.state_dict().items()
             if key.startswith('blocks.0.')}
    r1 = state_inventory({**source, **added})
    require((r0['state_tensors'], r1['state_tensors'], len(r0['selected']), len(r1['selected']))
            == (3043, 3171, 192, 320) and r0['banks'] == r1['banks'] == 512
            and r0['frozen_tensors'] == r1['frozen_tensors'] == 2851
            and r0['selected_elements'] == 839680 and r1['selected_elements'] == 2985984,
            'Actual source/native R1 inventory differs from the approved accounting.')
    for prefix in PREFIXES:
        for key, value in small.state_dict().items():
            if not key.startswith('blocks.0.'):
                require(list(value.shape) == r0['selected'][prefix + key]['shape'], 'Existing native decoder shape differs.')
    extra = sum(v.numel() * v.element_size() * 8 for v in added.values())
    require(extra == 34340864 and audit['bit_budget']['user_linear_bits'] == 17099800576
            and audit['bit_budget']['packed_linear_bits'] == 17099800576, 'Unexpected source or added bit budget.')
    stats_path = Path(original['data']['canonical_statistics']); stats = read(stats_path)
    sampler = stats['models']['llama']['sampler_first8000']
    hold_record_path = R2 / 'llama_column_feedback_fineweb128/launch.prepared.json'
    model_policy = read(hold_record_path)['model_policy']
    sources, inputs = source_pins(original)
    evidence = [CONTROL, DESIGN, SOURCE / 'checkpoint_meta.json', SOURCE / 'config.json', audit_path, review_path,
        SOURCE.parent / 'full8_summary.json', SOURCE.parent / 'mmlu.json', SOURCE.parent / 'seven.json',
        Path(original['data']['manifest']), stats_path, hold_record_path,
        R2 / 'llama_l16_gate_identity_r1_s128/report.json', R2 / 'llama_l16_gate_identity_r1_s128/migration.json',
        R2 / 'llama_s4_lm_validate_s4/verification.json']
    inputs.update({str(p): sha256(p) for p in evidence})
    environment = deepcopy(original['environment']); environment['PYTHONPATH'] = str(ROOT)
    for kind, stop, arms in (('validate_s4', 4, ['r1']), ('paired_s400', 400, ['r0', 'r1'])):
        output = OUTPUT / kind
        require(not output.exists(), 'Never overwrite an existing prepared output: ' + str(output))
        schedule = {'max_steps': 2000, 'warmup_steps': 20, 'actual_stop_step': stop,
            'save_steps': 2 if stop == 4 else 400, 'save_total_limit': 2 if stop == 4 else 1,
            'export_steps': [2, 4] if stop == 4 else [400], 'arms': arms,
            'export_only_after_training_exit': True, 'full8': stop == 400,
            'longbench': 'Root separately schedules complete21/4750, Llama native chat/bare task policy.',
            'initial_full8': False, 'exact_resume': False, 'automatic_continuation': False}
        record = {'schema': 'llama_cd338_all_gate_s4_identity_r1_v1', 'kind': kind, 'status': 'prepared_unbound',
            'gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT),
            'python_executable': sys.executable, 'output_dir': str(output), 'source_checkpoint': str(SOURCE),
            'source_checkpoint_id': SOURCE_ID, 'source_weight_stat': file_stat(SOURCE / 'pytorch_model.bin'),
            'source_bit_budget': audit['bit_budget'], 'architecture': original['architecture'],
            'inventories': {'r0': r0, 'r1': r1}, 'hf_source': original['hf_source'], 'model_policy': model_policy,
            'input_data_stat': original['input_data_stat'], 'source_sha256': dict(sources),
            'input_evidence_sha256': dict(inputs), 'environment': environment, 'schedule': schedule,
            'control_result_paths': [str(SOURCE.parent / (name + '.json')) for name in ('mmlu', 'seven')],
            'control_mean_percent': read(SOURCE.parent / 'full8_summary.json')['mean_percent'],
            'expansion': {'added_parameters': 2146304, 'added_bits': extra, 'added_state_tensors': 128,
                'identity': 'Native per-stage LayerNorm+Swish block; norm gamma1/beta0, linear weight/bias0.',
                'logical_bpw_r0': 2.4500661996694713, 'logical_bpw_r1': 2.454986572265625,
                'external_lora_rank': 0, 'function_error_tolerance_relative_l2': 0.002},
            'data': {**deepcopy(original['data']), 'training_rows': sampler['rows'][:4 * stop],
                'training_batches': sampler['batches'][:stop],
                'boundary': 'Same fixed native sampler prefix; source already saw this training corpus. '
                    'No historical unseen-data claim. Actual400 batches are checked and compared between arms.'},
            'validation': {'required': stop == 400, 'report_path': str(OUTPUT / 'validate_s4/report.json'),
                'report_sha256': None, 'status': 'GPU_not_executed'},
            'estimate': {'disk_bytes': 20_000_000_000 if stop == 4 else 35_000_000_000,
                'gpu_peak_gib': [50, 75], 'elapsed_seconds': [300, 1200] if stop == 4 else [3600, 12000],
                'disk_free_at_prepare': shutil.disk_usage(output.parent.parent).free,
                'boundary': 'New all-gate R1 peak/time are unmeasured; validation precedes formal binding. '
                    'One GPU sequential processes; full8 included only in formal, full21 separate.'},
            'scientific_boundary': 'Internal decoder capacity and residual parametrization plus extra compute/bits. '
                'Matched same-source gate-only recovery; local128-step KL gains do not establish full-model quality. '
                'Judge formal endpoints by complete8 and complete21 jointly, never reject LB based solely on A8.'}
        record['training_argv'] = {arm: arguments(record, arm) for arm in arms}
        record['cpu_configuration'] = {arm: actual_configuration(*configuration(record, arm)) for arm in arms}
        reference = deepcopy(original['training_argv'])
        expected = arguments(record, arms[0])
        for key in ('--student_checkpoint_dir', '--run_root_dir', '--target_modules', '--save_steps', '--save_total_limit'):
            reference[reference.index(key) + 1] = expected[expected.index(key) + 1]
        require(reference == expected, 'Unexpected deviation from the existing FineWeb recovery recipe.')
        record['cpu_checks'] = {'status': 'verified', 'cuda_initialized': torch.cuda.is_initialized(),
            'native_parser': True, 'small_native_s4_r1_shapes': True, 'source_mmap_inventory': True,
            'canonical_sampler_evidence_reused': True, 'gpu_backward_export_fresh_forward': 'not_executed'}
        require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
        check(record)
        record['argv'] = [sys.executable, '-u', '-m', 'experiments.llama_gate_s4_r1_probe.run',
                          '--launch-record', str(output / 'launch.json')]
        output.mkdir(parents=True)
        write_json(output / 'launch.prepared.json', record)
        print({'path': str(output / 'launch.prepared.json'), 'sha256': sha256(output / 'launch.prepared.json'),
               'sources': len(sources), 'inputs': len(inputs), 'cpu': 'verified', 'gpu': 'unbound'}, flush=True)


if __name__ == '__main__':
    prepare()

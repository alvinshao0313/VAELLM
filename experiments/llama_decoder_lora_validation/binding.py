"""CPU preparation for one original-Llama-control joint four-step prefix."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from compressed_e2e_fintuning.args import parse_args
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.llama_decoder_lora_validation.state import geometry
from experiments.long_context_recovery.contract import require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.w25_eval.lora_state import validate_source
from train_utils.runtime_snapshot import to_runtime_jsonable

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
CONTROL = R2 / 'general_lm_llama_d16_combo1600_s2000/launch.json'
OUTPUT = R2 / 'llama_decoder_lora_rank2_validate_s4'


def training_arguments(control, output):
    argv = list(control['training_argv'])
    for key, value in (('--run_root_dir', str(output)), ('--train_mode', 'decoder_lora'),
                       ('--distill_fp32_components', 'decoder,lora'), ('--save_steps', '2')):
        require(argv.count(key) == 1, 'Ambiguous matched option: ' + key)
        argv[argv.index(key) + 1] = value
    require(not any(k in argv for k in ('--lora_rank', '--lora_alpha', '--lora_dropout')), 'Control already has LoRA.')
    return argv + ['--lora_rank', '2', '--lora_alpha', '4', '--lora_dropout', '0']


def configuration(record):
    control = read(record['control_launch'])
    require(record['kind'] == 'validate_s4' and record['training_argv'] == training_arguments(control, record['output_dir']),
            'Only the matched joint four-step validation is implemented.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(control['cpu_configuration']['canonical_config'])
    expected.update(run_root_dir=record['output_dir'], train_mode='decoder_lora')
    expected['opt']['distill_fp32_components'] = ['lora', 'decoder']
    expected['lora'] = {'rank': 2, 'alpha': 4.0, 'dropout': 0.0,
                       'rank_explicit': True, 'alpha_explicit': True, 'dropout_explicit': True}
    expected['explicit_cli_flags'] = sorted(expected['explicit_cli_flags'] + ['--lora_rank', '--lora_alpha', '--lora_dropout'])
    rest = expected['remaining_argv']; rest[rest.index('--save_steps') + 1] = '2'
    require(to_runtime_jsonable(cfg) == expected, 'Parser differs beyond joint mode/rank/output/save2.')
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == record['source_checkpoint_id'] == control['source_checkpoint_id']
            and cfg.student_checkpoint_dir == record['source_checkpoint'] == control['source_checkpoint']
            and validate_source(meta) == record['architecture'] and geometry(meta, record['architecture']) == record['geometry'],
            'Input model identity or native geometry differs.')
    require(args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
            and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16
            and args.learning_rate == cfg.opt.decoder_lr == cfg.opt.learning_rate == 3e-6
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Preserve the original FineWeb schedule and full training shape.')
    return cfg, hf, args


def verify_audit(audit, record):
    g, inv, proposed = record['geometry'], record['expected_inventory'], record['proposed_bit_budget']
    require(audit['status'] == 'verified' and audit['train_mode'] == 'decoder_lora'
            and audit['source_checkpoint_id'] == record['source_checkpoint_id'] and audit['selected_global_step'] == 4
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == inv['deployed_tensors']
            and audit['selected_decoder_tensor_count'] == g['decoder_tensors']
            and audit['frozen_tensors_compared'] == inv['frozen_tensors']
            and audit['frozen_code_bank_count'] == audit['code_bank_count'] == g['code_banks'] and audit['changed_code_bits'] == 0
            and audit['lora_rank'] == 2 and audit['lora_tensor_count'] == g['lora_tensors']
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['selected_lora_matches_native_bf16_cast_exactly']
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_bytes_unchanged']
            and audit['selected_codes_match_frozen_source_exactly'], 'Joint save/fresh/frozen audit failed.')
    require(audit['bit_budget']['budget_passed'] and all(audit['bit_budget'][k] == proposed[k]
            for k in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'code_padding_bits')),
            'Actual joint storage differs from the shape-derived budget.')


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Fresh CPU-only preparation in S required.')
    control = read(CONTROL)
    require(control['status'] == 'completed' and control['completed_steps'] == [400, 800, 1200, 1600, 2000],
            'Matched original decoder-only run must be complete.')
    source = Path(control['source_checkpoint']); meta = read(source / 'checkpoint_meta.json')
    architecture = validate_source(meta)
    require(meta['checkpoint_id'] == control['source_checkpoint_id'] == 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
            and architecture == control['architecture'], 'Wrong source binding for this Llama validation.')
    g = geometry(meta, architecture); inv = deepcopy(control['expected_inventory']); budget = control['source_bit_budget']
    require(g['projections'] == inv['deployed_projections'] and g['decoder_tensors'] == inv['all_decoder_tensors']
            and g['decoder_elements'] == inv['selected_decoder_elements'] and g['code_banks'] == inv['all_code_banks']
            and g['decoder_elements'] * 16 == budget['components']['decoder']['storage_bits'], 'Metadata/native audited inventory differs.')
    outside = architecture['dense_state_tensor_count'] - g['projections']
    require(not budget['storage_aliases'] and outside + sum(v['unique_storages'] for v in budget['components'].values())
            == inv['serialized_tensors'] and inv['frozen_tensors'] == inv['serialized_tensors'] - g['decoder_tensors'],
            'Source serialized/frozen inventory differs from the native component ledger.')
    inv.update(lora_tensors=g['lora_tensors'], lora_elements=g['lora_elements'],
        mutable_tensors=g['decoder_tensors'] + g['lora_tensors'], mutable_elements=g['decoder_elements'] + g['lora_elements'],
        deployed_tensors=inv['serialized_tensors'] + g['lora_tensors'])
    proposed = {k: budget[k] for k in ('original_linear_weights', 'code_padding_bits')}
    for prefix in ('user', 'packed'):
        proposed[prefix + '_linear_bits'] = budget[prefix + '_linear_bits'] + g['lora_bits']
        proposed[prefix + '_linear_bpw'] = proposed[prefix + '_linear_bits'] / proposed['original_linear_weights']
    require(proposed['user_linear_bpw'] <= 2.5, 'The proposed ordinary rank2 exceeds the logical budget.')
    proposed.update(added_lora_bits=g['lora_bits'], measurement_boundary='Metadata shapes plus verified source native budget; deployment not yet measured.')
    sources, evidence = {}, {str(CONTROL): sha256(CONTROL)}
    for raw, digest in control['source_sha256'].items():
        original = Path(raw); mapped = original if original.is_relative_to(ROOT) else ROOT / original.relative_to(PUBLIC) if original.is_relative_to(PUBLIC) else original
        chosen = mapped if mapped.is_file() else original
        require(sha256(chosen) == digest, 'Original native source changed: ' + str(chosen))
        (sources if mapped.is_file() else evidence)[str(chosen)] = digest
    pending, seen = list(Path(__file__).parent.glob('*.py')), set()
    while pending:
        path = pending.pop().resolve()
        if path in seen: continue
        seen.add(path); sources[str(path)] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                     [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                local = ROOT / (name.replace('.', '/') + '.py')
                if local.is_file(): pending.append(local)
    for path in [source / 'checkpoint_meta.json', source / 'config.json', source.parent / 'review.json',
                 CONTROL.parent / 'step_2000/independent_review.json', Path(control['data']['manifest']),
                 Path(control['data']['canonical_statistics']), R2 / 'qwen_decoder_lora_rank2_validate_s4/verification.json']:
        evidence[str(path)] = sha256(path)
    qproof = read(R2 / 'qwen_decoder_lora_rank2_validate_s4/launch.prepared.json')
    for p in (ROOT / 'experiments/qwen_decoder_lora_fallback').glob('*.py'):
        require(sha256(p) == qproof['source_sha256'][str(p)] == sha256(PUBLIC / p.relative_to(ROOT)), 'Old Qwen source changed.')
    for p in Path(__file__).parent.glob('*.py'):
        require(sha256(p) == sha256(PUBLIC / p.relative_to(ROOT)), 'New public/S sources differ.')
    check_hashes(sources); check_hashes(evidence)
    env = deepcopy(control['environment']); env['PYTHONPATH'] = str(ROOT)
    record = {'schema': 'joint_rank2_native_model_validation_v1', 'kind': 'validate_s4', 'status': 'prepared_unbound',
        'gpu': None, 'export_gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT),
        'python_executable': sys.executable, 'output_dir': str(OUTPUT), 'control_launch': str(CONTROL),
        **{k: deepcopy(control[k]) for k in ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat',
            'source_bit_budget', 'hf_source', 'input_data_stat', 'data')},
        'architecture': architecture, 'geometry': g, 'expected_inventory': inv, 'proposed_bit_budget': proposed,
        'source_sha256': sources, 'input_evidence_sha256': evidence, 'environment': env,
        'training_argv': training_arguments(control, OUTPUT),
        'argv': [sys.executable, '-u', '-m', 'experiments.llama_decoder_lora_validation.train', '--launch_record', str(OUTPUT / 'launch.json')],
        'dispatch_argv': [sys.executable, '-u', '-m', 'experiments.llama_decoder_lora_validation.run', '--launch_record', str(OUTPUT / 'launch.json')],
        'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'save_steps': 2, 'save_total_limit': 1,
                     'actual_stop_step': 4, 'independent_export_steps': [4], 'full8_steps': []},
        'resource_budget_seconds': 1200, 'termination_grace_seconds': 20,
        'estimate': {'training_peak_gib': [40, 55], 'elapsed_seconds': [200, 450], 'disk_bytes_reserved': 15000000000,
                     'disk_free_at_prepare': shutil.disk_usage(OUTPUT.parent).free,
                     'boundary': 'Llama decoder-only measured37.09/39.82GiB and Qwen joint short33.25/33.49GiB are references, not Llama joint measurements.'},
        'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled'}, 'formal_prepared': False,
        'validation_boundary': 'Only genuine joint backward/update, native final, independent export, strict fresh reload and frozen/budget audit. No full8/LongBench or quality claim.',
        'scientific_boundary': 'Same source/FineWeb/data/optimizer/2000 horizon; rank2 adds initialization RNG consumption. Original source, never Qwen short weights.'}
    cfg, hf, args = configuration(record)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
                                   'training_args': to_runtime_jsonable(args)}
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record.update(cuda_initialized=False); record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir(); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'prepared': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
           'sources': len(sources), 'evidence': len(evidence), 'geometry': g, 'inventory': inv, 'budget': proposed}, flush=True)


if __name__ == '__main__':
    prepare()

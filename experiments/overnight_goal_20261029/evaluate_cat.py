"""Completed CAT A/B only: unchanged full8, full21 and current transfer acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = TOPIC / 'overnight_goal_20261029'
DATA = Path('/data/star/VAELLM/cat_structure_full_20261029')
N = TOPIC / 'qwen_native_category_rotation/source_snapshot'
sys.path.insert(0, str(ROOT / 'experiments/cat_aligned_evaluation_20261028'))
import evaluate as original

b = original.b
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def projection_spec(spec, arm):
    category = spec['name'].rsplit('.', 1)[1]
    dim = 64 if arm == 'B' and category == 'gate_proj' else 32
    bits = {'q_proj': 38, 'k_proj': 38, 'v_proj': 32, 'o_proj': 48,
            'gate_proj': dim, 'up_proj': 32, 'down_proj': 48}[category]
    decoders = spec['stage_decoders']
    b.require(spec['codebook_dim'] == dim and spec['stage_codebook_dims'] == [dim, dim]
        and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1 and spec['parallel_stage_decode']
        and not spec['has_original_weight'] and not spec['always_use_original']
        and len(decoders) == 2 and all(decoder['in_dim'] == bits and decoder['out_dim'] == dim
            and decoder['hidden_dim'] == 128 and decoder['num_res_blocks'] == 0
            and decoder['norm_type'] == 'layer' and decoder['activation_type'] == 'swish'
            and decoder['decoder_type'] == ('asymmetric' if category == 'gate_proj' else 'symmetric')
            for decoder in decoders), 'Changed actual CAT ' + arm + ' deployment geometry: ' + spec['name'])


def completed_run(arm):
    directory = DATA / arm
    run = b.read(directory / 'run.json')
    b.require(run['status'] == 'completed' and run.get('exit_code') == 0 and run['arm'] == arm
        and run['phase'] == 'formal' and (directory / 'run.exit').is_file()
        and (directory / 'run.exit').read_text().strip() == '0',
        'CAT ' + arm + ' has not completed its actual formal run; no wait or predecessor fallback.')
    event = run['saved_boundaries'][-1]
    checkpoint = Path(event['path']).resolve()
    b.require(event['kind'] == 'final_model' and checkpoint.name == 'final_model'
        and checkpoint.is_relative_to(directory / 'native')
        and event['completed_categories'] == list(CATEGORIES), 'Require this arm own final saved endpoint.')
    return directory, run, event, checkpoint


def source(profile, arm, checkpoint=None, native_audit=None):
    # These two output components select the original evaluator's evaluation/{arm}.
    b.require(profile == 'evaluation' and arm in ('A', 'B') and checkpoint is native_audit is None,
              'Only each completed CAT A/B final is accepted.')
    directory, run, event, checkpoint = completed_run(arm)
    prepared_path = DATA / 'prepared.json'
    prepared = b.read(prepared_path)
    b.require(prepared['status'] == 'cpu_verified' and run['preparation_sha256'] == b.sha256(prepared_path)
        and run['source_checkpoint_id'] == prepared['arms'][arm]['checkpoint_id'], 'Changed actual CAT run preparation/source.')
    b.check_hashes(prepared['implementation_sha256'])
    b.check_hashes(prepared['snapshot_source_sha256'])
    meta = b.read(checkpoint / 'checkpoint_meta.json')
    b.require(meta['checkpoint_kind'] == 'final_model' and meta['checkpoint_id'] == event['checkpoint_id']
        and meta['checkpoint_id'] != run['source_checkpoint_id'] and meta['converted_module_count'] == 252
        and meta['completed_categories'] == meta['compression_categories'] == list(CATEGORIES)
        and not meta['pending_dense_targets'] and not meta['skip_targets'] and meta['lora_config'] is None
        and meta['finalized_status']['inference_ready']
        and b.sha256(checkpoint / 'checkpoint_meta.json') == event['metadata_sha256']
        and b.sha256(checkpoint / 'cat_runtime_state.pt') == event['runtime_sha256'], 'Final metadata/runtime identity changed.')
    saved_stat = event['weight_stat']
    current = b.file_stat(checkpoint / meta['state_dict_file'])
    b.require(current == dict(path=saved_stat['path'], size=saved_stat['bytes'],
        mtime_ns=saved_stat['mtime_ns'], inode=saved_stat['inode']), 'Final weight file changed since actual save.')
    from experiments.w25_eval.model_inventory import model_inventory
    architecture = model_inventory(meta['base_model_path'])
    b.require(architecture['projection_count'] == 252 and architecture['num_hidden_layers'] == 36
        and set(meta['compressed_targets']) == set(architecture['projection_shapes'])
        and len(meta['converted_modules']) == 252, 'Require all actual Qwen3-8B projections.')
    for spec in meta['converted_modules']:
        projection_spec(spec, arm)
        b.require([spec['out_features'], spec['in_features']] == architecture['projection_shapes'][spec['name']],
                  'Projection differs from the original HF shape.')
    ledger = b.module(N / 'train_utils/checkpoint_bpw.py', '_overnight_CAT_N_ledger')
    import torch
    state = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = {name for spec in meta['converted_modules'] for name, _ in ledger._code_specs(spec)}
    b.require(len(state) == 3423 and len(banks) == 504 and banks <= state.keys(), 'Require actual3423 fields/504 code banks.')
    budget = ledger.audit_state_dict(meta, state, max_bpw=2.6, max_lora_rank=0)
    b.require(budget['budget_passed'] and not budget['contract_violations']
        and not budget['unclassified_target_keys'] and budget['target_count'] == budget['compressed_target_count'] == 252
        and all(not budget['components'].get(key, {}).get('storage_bits', 0)
                for key in ('dense_weights', 'lora', 'residual_lora')), 'Actual complete CAT rank0/logical2.6 ledger failed.')
    del state
    pins = {**prepared['implementation_sha256'], **prepared['snapshot_source_sha256']}
    for path in (directory / 'run.json', directory / 'run.exit', prepared_path,
                 checkpoint / 'checkpoint_meta.json', checkpoint / 'cat_runtime_state.pt', Path(ledger.__file__)):
        pins[str(path)] = b.sha256(path)
    audit = dict(schema='CAT_AB_completed_native_source_v1', status='verified', arm=arm,
        checkpoint_dir=str(checkpoint), checkpoint_id=meta['checkpoint_id'],
        source_checkpoint_id=run['source_checkpoint_id'], checkpoint_weight_stat=current,
        bit_budget=budget, serialized_fields=3423, code_bank_count=504,
        validation_scope='Actual completed save/metadata/serialized tensor inventory and N storage ledger. Fresh runtime strict load/prewarm belongs to ensuing full8/full21.',
        input_evidence_sha256=pins)
    audit_path = OUTPUT / 'sources' / (arm + '.json')
    b.require(not audit_path.exists(), 'Never overwrite an existing CAT source audit.')
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    b.write_json(audit_path, audit)
    pins[str(audit_path)] = b.sha256(audit_path)
    return dict(profile=arm, stage='CAT_final', checkpoint_dir=str(checkpoint), checkpoint_id=meta['checkpoint_id'],
        native_audit=str(audit_path), native_audit_sha256=b.sha256(audit_path), checkpoint_weight_stat=current,
        bit_budget=budget, evidence_sha256=pins), meta


def source_pins(binding):
    paths = (Path(__file__), ROOT / 'experiments/w25_eval/summarize_transfer_nine_task.py',
             ROOT / 'experiments/w25_eval/transfer_nine_task_acceptance.py')
    return {**b.source_pins(binding), **{str(path): b.sha256(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--gpus', type=original.gpu_list, required=True)
    args = parser.parse_args()
    b.cpu_environment()
    b.require(not os.path.lexists(OUTPUT / 'evaluation' / args.arm), 'Never overwrite a CAT evaluation.')
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='evaluation', stage=args.arm,
        checkpoint=None, native_audit=None, check_only=False, gpus=args.gpus))
    plan = b.read(record['longbench_plan'])
    output = OUTPUT / 'evaluation' / args.arm / 'transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task', *plan['nine_task_argv'][2:]]
    argv[argv.index('--output') + 1] = str(output)
    b.check_hashes(record['source_sha256'])
    with output.with_suffix('.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0, 1), 'Current transfer report failed validation.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id']
        and report['protocol']['strict_score_drop_bound'] and report['protocol']['maximum_absolute_drop_points'] == .9
        and report['protocol']['max_linear_bpw'] == 2.6, 'Require the same complete endpoint/current profile.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        original_evaluation=record, acceptance_argv=argv, acceptance_exit_code=result.returncode,
        acceptance_report=str(output), score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(output.with_name('current_goal_result.json'), outcome)
    print(json.dumps({key: outcome[key] for key in ('status', 'checkpoint_id', 'score', 'passed')}), flush=True)


if __name__ == '__main__':
    main()

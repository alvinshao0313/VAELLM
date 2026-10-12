"""CPU-only preparation, conditional on the genuine native joint four-step proof."""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_decoder_lora_validation.binding import CONTROL, OUTPUT as SHORT, PUBLIC, R2, ROOT, check_inputs, read, require
from experiments.llama_decoder_lora_recovery.callback import stage_commands, verify_audit
from experiments.llama_decoder_lora_recovery.train import JointObservations, configuration, training_arguments
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from train_utils.runtime_snapshot import to_runtime_jsonable

OUTPUT = R2 / 'general_lm_llama_decoder_lora_rank2_s2000'


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Fresh CPU-only preparation in S required.')
    record = deepcopy(read(SHORT / 'launch.prepared.json')); control = read(CONTROL)
    check_inputs(record)
    proof_path = SHORT / 'verification.json'; proof = read(proof_path)
    require(proof['status'] == 'verified' and proof['source_checkpoint_id'] == record['source_checkpoint_id']
            and proof['expected_inventory'] == record['expected_inventory']
            and proof['native_vs_independent_tensors_compared'] == record['expected_inventory']['deployed_tensors']
            and proof['gradient_update_steps'] == [1, 2, 3, 4] and proof['native_save_steps'] == [2, 4],
            'A completed native joint proof for this exact source/inventory is required.')
    check_hashes(proof['evidence_sha256'])
    exits = [SHORT / p for p in ('training.exit', 'step_0004/export.exit', 'step_0004/audit.exit', 'dispatcher.exit', 'outer.exit')]
    require(all(p.read_text().strip() == '0' for p in exits), 'Short validation did not exit successfully.')
    short_launch = read(SHORT / 'launch.json'); trained = read(SHORT / 'training.json')
    observed = trained['observations']
    require(short_launch['status'] == 'verified' and trained['status'] == 'trained'
            and trained['training_result']['global_step'] == 4 and trained['training_result']['round_base_checkpoint_id'] == record['source_checkpoint_id']
            and [v['step'] for v in observed['native_saves']] == [2, 4]
            and all([v['step'] for v in observed[k]] == [1, 2, 3, 4] for k in ('gradients', 'updates'))
            and [v['step'] for v in observed['batches']] == [1, 2, 3, 4]
            and all(v['shape'] == [4, 1024] for v in observed['batches']), 'Actual native prefix or batch shape differs.')
    for key, field in (('gradients', 'post_clip_norm'), ('updates', 'delta_l2')):
        for row in observed[key]:
            require(set(row[field]) == {'decoder', 'A', 'B'} and all(math.isfinite(v) and v >= 0 for v in row[field].values()),
                    'Invalid actual component observations.')
            require(all(v > 0 or (key == 'updates' and row['step'] == 1) or (k == 'A' and row['step'] <= 2)
                        for k, v in row[field].items()), 'Missing actual update beyond native zero-B/warmup boundaries.')
    require(all(v['moment_dtypes'] == ['torch.float32'] for v in observed['updates']), 'Short Adam precision differs.')
    verify_audit(read(SHORT / 'step_0004/audit.json'), record, 4)
    require(control['status'] == 'completed' and control['completed_steps'] == [400, 800, 1200, 1600, 2000],
            'Matched decoder-only control is incomplete.')
    sources, evidence = record['source_sha256'], record['input_evidence_sha256']
    evidence.update(proof['evidence_sha256'])
    retained = [proof_path, SHORT / 'launch.prepared.json', SHORT / 'launch.json', *exits, *map(Path, control['control_result_paths'])]
    for step in (400, 800, 1200, 1600, 2000):
        retained.extend(CONTROL.parent / f'step_{step:04d}' / name for name in ('full8_summary.json', 'mmlu.json', 'seven.json', 'audit.json'))
    for path in retained: evidence[str(path)] = sha256(path)
    qwen_prepared = R2 / 'qwen_decoder_lora_rank2_validate_s4/launch.prepared.json'
    evidence[str(qwen_prepared)] = sha256(qwen_prepared)
    for package in ('llama_decoder_lora_validation', 'qwen_decoder_lora_fallback'):
        verified_sources = read(qwen_prepared)['source_sha256'] if package == 'qwen_decoder_lora_fallback' else sources
        for path in (ROOT / 'experiments' / package).glob('*.py'):
            require(sha256(path) == verified_sources[str(path)] == sha256(PUBLIC / path.relative_to(ROOT)), 'Verified old source changed.')
            sources[str(path)] = verified_sources[str(path)]
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
    for path in Path(__file__).parent.glob('*.py'):
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S sources differ.')
    record.pop('dispatch_argv')
    record.update(schema='joint_rank2_native_model_formal_v1', kind='formal_s2000', output_dir=str(OUTPUT),
        control_result_paths=control['control_result_paths'], training_argv=training_arguments(control, OUTPUT),
        argv=[sys.executable, '-u', '-m', 'experiments.llama_decoder_lora_recovery.train', '--launch_record', str(OUTPUT / 'launch.json')],
        schedule={'max_steps': 2000, 'warmup_steps': 20, 'save_steps': 400, 'save_total_limit': 1,
                  'actual_stop_step': 2000, 'independent_export_steps': [400, 800, 1200, 1600, 2000],
                  'full8_steps': [400, 800, 1200, 1600, 2000]},
        resource_budget_seconds=64800, termination_grace_seconds=20, formal_prepared=True,
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'short_validation': {'path': str(proof_path), 'sha256': sha256(proof_path)}},
        estimate={'training_peak_gib': [40, 55], 'export_eval_peak_gib': [24, 55], 'elapsed_hours': [11, 13],
                  'disk_bytes_reserved': 35000000000, 'disk_free_at_prepare': shutil.disk_usage(OUTPUT.parent).free,
                  'observed_short_peak_allocated_bytes': trained['peak_allocated_bytes'],
                  'observed_short_peak_reserved_bytes': trained['peak_reserved_bytes'],
                  'matched_decoder_only_elapsed_seconds': 38018.183315992355,
                  'boundary': '11-13h is extrapolated from same-source decoder-only2000; rank2 long-run speed/peak and full8 quality are unmeasured.'},
        validation_boundary='Actual joint4 native save/final/reload and generic joint export/audit reused; every400 full8 uses the unchanged completed Llama protocol. No new GPU startup test.',
        scientific_boundary='Same e767/FineWeb/optimizer/2000 schedule as decoder-only; ordinary all-projection rank2 adds capacity and initialization RNG consumption. Short58ad weights are never inputs.',
        comparison={'primary': str(CONTROL.parent), 'original_source': record['source_checkpoint'],
                    'longbench': 'not_prepared_or_evaluated', 'quality': 'unmeasured; full8 alone is not nine-item acceptance'})
    cfg, hf, args = configuration(record)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
                                   'training_args': to_runtime_jsonable(args)}
    observer = JointObservations(record); require(not observer.short, 'Formal observer must not stop at four steps.')
    record['stage_templates'] = stage_commands('{checkpoint}', record['source_checkpoint'], '{step_output}')
    require([v[0] for v in record['stage_templates']] == ['export', 'audit', 'mmlu', 'seven']
            and not any('--lm_limit' in v[1] for v in record['stage_templates']), 'Formal full8 protocol is incomplete.')
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir(); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'prepared': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
           'sources': len(sources), 'evidence': len(evidence), 'cuda_initialized': False,
           'geometry': record['geometry'], 'budget_bpw': record['proposed_bit_budget']['user_linear_bpw']}, flush=True)


if __name__ == '__main__':
    prepare()

"""Aggregate four existing CAT stages using five new and three reused shards."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.compare import score, delta
from experiments.llama_prefix_stage_full8.prepare import STAGES, check, read, require
from experiments.longbench_v1.common import sha256, write_json


def compare(record_path):
    record = read(record_path); output = Path(record['output_dir'])
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Aggregate on CPU only.')
    require(not (output / 'comparison.json').exists(), 'Do not overwrite the completed comparison.')
    check(record)
    final_record = read(record['final_acceptance_path'])
    final_paths = [Path(row['path']) for row in final_record['compressed']['input_results']]
    final, reference_raw = score(final_paths, record['architecture'], Path(final_record['checkpoint_dir']))
    results, evidence = {}, {}
    for name in STAGES:
        item = record['checkpoints'][name]
        paths = [Path(item[key]) for key in ('mmlu_path', 'seven_path')]
        result, raws = score(paths, record['architecture'], Path(item['path']))
        for payload, reference in zip(raws, reference_raw, strict=True):
            for field in ('configs', 'versions', 'n_shot', 'n_samples'):
                require(payload[field] == reference[field], name + ': protocol differs: ' + field)
        for path in paths:
            raw = read(path); warm = raw['evals']['cache_warmup']
            require(warm['total'] == warm['warmed'] == item['compressed_count'] and warm['failed'] == warm['skipped'] == 0,
                    name + ': actual compressed inventory was not fully warmed.')
            evidence[str(path)] = sha256(path)
        results[name] = {**result,
            'compressed_count': item['compressed_count'], 'pending_dense_count': item['pending_dense_count'],
            'mmlu_reused': item['reused_complete_mmlu'], 'vs_final': delta(result, final)}
    results['final'] = final
    order = [*STAGES, 'final']
    transitions = [{'from': left, 'to': right, **delta(results[right], results[left])}
                   for left, right in zip(order, order[1:])]
    report = {'status': 'full8_stage_diagnostic_verified', 'protocol': record['protocol'],
        'baseline': final_record['baseline'], 'stages': results, 'transitions': transitions,
        'new_gpu_shards': [row['name'] for row in record['stage_commands']],
        'reused_complete_mmlu': [name for name in STAGES if record['checkpoints'][name]['reused_complete_mmlu']],
        'source_preparation_sha256': sha256(output / 'launch.prepared.json'), 'evidence_sha256': evidence,
        'scope': record['scope'], 'goal_acceptance': False, 'longbench_status': 'not_evaluated',
        'down_quantization_scope_reused': 'The existing stage-MMLU comparison proves 2563 non-down tensors unchanged at after_up -> down_round_base; no large weight reload repeated.',
        'cuda_initialized': False}
    check(record)
    write_json(output / 'comparison.json', report)
    print({name: row['mean_percent'] for name, row in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    compare(parser.parse_args().launch_record)

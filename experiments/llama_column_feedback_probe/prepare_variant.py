"""Prepare the fixed FineWeb128 follow-up; reuse the completed v2 algorithm without GPU work."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from transformers import AutoTokenizer

from experiments.llama_column_feedback_probe.data_variant import build_variant_data
from experiments.llama_column_feedback_probe.prepare import ROOT, R2, check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record

PREVIOUS = R2 / 'llama_column_feedback_probe_v2'


def prepare_variant(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not output.exists(), 'CPU variant preparation requires hidden CUDA and a fresh output.')
    torch.set_num_threads(4)
    previous = read(PREVIOUS / 'launch.prepared.json')
    result = read(PREVIOUS / 'report.json')
    review = read(PREVIOUS / 'independent_review.json')
    require(result['status'] == 'completed_native_short_probe'
            and review['status'] == 'completed_independent_cpu_review'
            and (PREVIOUS / 'run.exit').read_text().strip() == (PREVIOUS / 'outer.exit').read_text().strip() == '0',
            'Require the completed and reviewed original Alpaca128 control.')
    check(previous)
    require(sha256(PREVIOUS / 'report.json') == review['evidence_sha256']['report.json']
            and sha256(PREVIOUS / 'launch.prepared.json') == review['evidence_sha256']['launch.prepared.json'],
            'Previous completed evidence changed after review.')
    old_batches = torch.load(PREVIOUS / 'token_batches.pt', weights_only=True, map_location='cpu')
    require({domain: {key: tensor_record(value) for key, value in batch.items()}
             for domain, batch in old_batches.items()} == previous['data']['token_batches'],
            'Old fixed token batches changed.')
    tokenizer = AutoTokenizer.from_pretrained(previous['hf_source']['path'], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    batches, data = build_variant_data(tokenizer, old_batches, previous['data'])
    record = deepcopy(previous)
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
                  output_dir=str(output), python_executable=sys.executable, data=data)
    record.pop('repair', None)
    record['command'][-1] = str(output / 'launch.json')
    for path in (Path(__file__).resolve(), Path(__file__).with_name('data_variant.py').resolve()):
        ast.parse(path.read_text())
        record['source_sha256'][str(path)] = sha256(path)
    for name in ('launch.prepared.json', 'report.json', 'independent_review.json', 'token_batches.pt', 'run.exit', 'outer.exit'):
        record['input_sha256'][str(PREVIOUS / name)] = sha256(PREVIOUS / name)
    record['variant'] = {'previous_completed_control': str(PREVIOUS),
        'condition': 'Replace Alpaca128 Hessian calibration by fixed FineWeb physical rows650000:650128.',
        'reason': 'V2 per-column reduced training local error but all32 FineWeb local errors exceeded off. '
                  'Test calibration-domain/effective-token specialization without expanding the layer or algorithm.',
        'same_original_e767_and_algorithm': True, 'new_cpu_formula_tests_run': False,
        'reused_cpu_checks': previous['cpu_checks'], 'old_observations_are_development': True,
        'new_observations_prespecified': True, 'domain_only_single_variable_claim': False,
        'calibration_token_counts': {'previous_alpaca': data['counts']['previous_alpaca_train_reference_only'],
                                     'new_fineweb': data['counts']['train']},
        'termination_boundary': 'If new observations also have no stable full-model net gain, close this feedback '
                                'configuration without all-layer expansion or formal integration.'}
    record['protocol']['observations'] = ('Native BF16 local SSE/NMSE and complete-model top100+tail-mass KL/NLL; '
        'report calibration, two reused development sets, and two newly fixed observation sets separately. '
        'No observation selects codes, damping, groups, iterations, or stopping.')
    record['protocol']['success_boundary'] = ('Local error reduction is not success; evaluate the full-model paired '
        'KL/NLL on both new observation domains and compare all three feedback arms. No eight-task/LongBench claim.')
    record['resource_estimate'].update(gpu_peak_gib_estimate=75, host_peak_gib_estimate=48,
        wall_seconds_estimate=[150, 360], disk_bytes_upper_estimate=2_000_000_000,
        basis='Completed v2 measured121.43s and68.104GiB allocated/68.617GiB reserved; same one-layer search and '
              'native B1/256 shapes. This variant scores256 rather than192 teacher documents and1024 rather than768 '
              'student documents; FineWeb calibration has more valid tokens. Peak and timing remain estimates.')
    require(shutil.disk_usage(output.parent).free >= 2_000_000_000, 'Reserve2GB for records and three packed endpoints.')
    require(not torch.cuda.is_initialized(), 'CPU variant preparation initialized CUDA.')
    check(record)
    output.mkdir(parents=True)
    torch.save(batches, output / 'token_batches.pt')
    record['input_sha256'][str(output / 'token_batches.pt')] = sha256(output / 'token_batches.pt')
    write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'path': str(output / 'launch.prepared.json'),
        'sha256': sha256(output / 'launch.prepared.json'), 'data_counts': data['counts'],
        'source_pins': len(record['source_sha256']), 'input_pins': len(record['input_sha256']),
        'new_cpu_formula_tests_run': False, 'gpu_bound': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare_variant(parser.parse_args().output_dir)

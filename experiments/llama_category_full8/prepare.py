"""CPU-only preparation of seven independent single-category dense-oracle evaluations."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.llama_category_full8.common import ARMS, ROOT, RESULT, SOURCE_ID, PROTOCOL, read, check
from experiments.longbench_v1.common import file_stat, sha256, source_pins, write_json, check_hashes
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results


def prepare(output_root):
    output_root = Path(output_root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output_root.exists(), 'Use hidden CUDA and a new CPU-only preparation.')
    previous_path = RESULT / 'mmlu_category_attribution/preparation.json'
    review_path = previous_path.parent / 'independent_review.json'
    acceptance_path = RESULT / 'remaining_prefix_block32_s500_full8/acceptance.json'
    old, review, acceptance = map(read, (previous_path, review_path, acceptance_path))
    require(old['source_checkpoint_id'] == review['source_checkpoint_id'] == acceptance['checkpoint_id'] == SOURCE_ID
            and review['status'] == 'independently_verified' and review['all8_dispatch_stages_exit0']
            and old['architecture']['model_type'] == 'llama', 'Reuse the actual completed Llama restoration evidence.')
    reused = ('experiments/mmlu_attribution/run.py', 'experiments/longbench_attribution/restore.py',
              'tools/cat_eval.py', 'train_utils/v6_model_loader.py', 'train_utils/eval_utils.py', 'rotation/model_utils.py')
    check_hashes({str(ROOT / name): old['source_sha256'][str(ROOT / name)] for name in reused})
    source = Path(old['source_checkpoint']); meta = read(source / 'checkpoint_meta.json')
    compressed_paths = [row['path'] for row in acceptance['compressed']['input_results']]
    baseline_paths = [row['path'] for row in acceptance['baseline']['input_results']]
    compressed = collect_results(compressed_paths, old['architecture'], checkpoint=(source, meta))
    baseline = collect_results(baseline_paths, old['architecture'])
    require(compressed['mean_percent'] == acceptance['compressed']['mean_percent']
            and baseline['mean_percent'] == acceptance['baseline']['mean_percent']
            and compressed['versions'] == baseline['versions'], 'Existing complete8 controls changed.')
    sources = source_pins()
    for name in (*reused, 'experiments/mmlu_attribution/protocol.py', 'experiments/w25_eval/acceptance.py',
                 'experiments/qwen3_w25/summarize_results.py'):
        sources[str(ROOT / name)] = sha256(ROOT / name)
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    evidence = {path: digest for path, digest in old['input_evidence_sha256'].items()
                if Path(path) != source / meta['state_dict_file']}
    for path in (previous_path, review_path, acceptance_path, *map(Path, compressed_paths + baseline_paths)):
        evidence[str(path.resolve())] = sha256(path)
    budget = acceptance['bit_budget']
    common = {key: deepcopy(old[key]) for key in ('source_checkpoint', 'source_checkpoint_id', 'hf_path',
        'architecture', 'model_policy', 'expected_serialized_tensors', 'package_versions')}
    common.update(schema='llama_single_category_full8_oracle_v1', status='prepared_unbound', gpu=None,
        queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable, protocol=PROTOCOL,
        source_sha256=sources, input_evidence_sha256=evidence,
        protected_stats=[file_stat(source / meta['state_dict_file']), *old['hf_weight_stats']],
        control_result_paths=dict(zip(('mmlu', 'seven'), compressed_paths, strict=True)),
        control_mean_percent=compressed['mean_percent'], control_task_metrics=compressed['task_metrics'],
        baseline_mean_percent=baseline['mean_percent'], baseline_task_metrics=baseline['task_metrics'],
        source_bit_budget=budget, timeout_seconds=900, termination_grace_seconds=20,
        environment={'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4',
            'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT)},
        validation_reuse={'path': str(review_path), 'no_new_gpu_short': True,
            'reason': 'The unchanged model-specific loader, exact BF16 restoration and frozen-state checks already ran on this source. Native full8 evaluator and64/32 resource conditions also have actual same-source evidence.',
            'boundary': 'Previous M1824 results do not rank eight-task sensitivity; no previous diagnostic is repeated.'},
        cuda_initialized=False, model_exported=False)
    records = {}
    for arm in ARMS:
        path = output_root / arm
        targets = [name for name in old['architecture']['projection_shapes'] if name.rsplit('.', 1)[1] == arm]
        dense_elements = sum(budget['per_target'][name]['original_linear_weights'] for name in targets)
        removed_bits = sum(budget['per_target'][name]['budget_bits'] for name in targets)
        bits = budget['user_linear_bits'] - removed_bits + dense_elements * 16
        require(bits.is_integer() if isinstance(bits, float) else True, 'Expected integer persistent-bit accounting.')
        bpw = bits / budget['original_linear_weights']
        require(bpw > 2.5 and len(targets) == 32, 'Every full-category BF16 oracle exceeds the target budget.')
        record = deepcopy(common)
        record.update(arm=arm, output_dir=str(path), selected_targets=targets,
            diagnostic_budget={'original_linear_weights': budget['original_linear_weights'],
                'source_bits': budget['user_linear_bits'], 'removed_compressed_bits': int(removed_bits),
                'restored_bf16_elements': dense_elements, 'restored_bf16_bits': dense_elements * 16,
                'expected_linear_bits': int(bits), 'expected_linear_bpw': bpw, 'budget_passed': False,
                'basis': 'Measured CAT3577 per-target storage minus the selected category plus original BF16 weights; runtime checks actual restored storage. No mixed checkpoint is exported.'})
        record['argv'] = [sys.executable, '-u', '-m', 'experiments.llama_category_full8.run',
                          '--launch-record', str(path / 'launch.json')]
        check(record); records[arm] = record
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output_root.mkdir()
    for arm, record in records.items():
        path = output_root / arm; path.mkdir()
        write_json(path / 'launch.json', record); write_json(path / 'launch.prepared.json', record)
    summary = {'status': 'cpu_prepared_unbound', 'source_checkpoint_id': SOURCE_ID, 'cuda_initialized': False,
        'arm_prepared_sha256': {arm: sha256(output_root / arm / 'launch.prepared.json') for arm in ARMS},
        'source_pin_count': len(sources), 'input_pin_count': len(evidence), 'all_arms_diagnostic_only': True,
        'control_mean_percent': compressed['mean_percent'], 'baseline_mean_percent': baseline['mean_percent']}
    write_json(output_root / 'preparation.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)

"""CPU-prepare one explicit native gate job in the approved source snapshot."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import (
    EXPECTED_BITS, EXPECTED_IDS, L_TEMPLATE, ORIGINAL, Q_TEMPLATE, ROOT, check, read, recipe,
)
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.packed_decode_evaluation.prepare import evaluation_sources
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_v6 import load_v6_meta


def prepare(model, block_size, steps, output, snapshot_review, validation):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
             'Use a fresh output and CPU-only preparation inside the snapshot.')
    snapshot_review = Path(snapshot_review).resolve()
    snapshot = read(snapshot_review)
    _require(snapshot.get('status') == 'verified', 'The source snapshot CPU review must be verified.')
    q = read(Q_TEMPLATE)
    if model == 'qwen3':
        template_path, template = Q_TEMPLATE, q
        base = Path(q['base']['checkpoint_dir'])
        stats = [q['base']['weight_stat'], *q['model_source']['weight_files']]
        controls = [q['protocol']['source_results'][name] for name in ('mmlu', 'seven')]
        model_path = q['model_source']['path']
        provenance = Path(q['model_source']['prior_verified_source_record']['path'])
    else:
        template_path, template = L_TEMPLATE, read(L_TEMPLATE)
        base = Path(template['source_checkpoint'])
        stats = [template['source_weight_stat'], *template['hf_source']['weight_files']]
        controls = template['control_result_paths']
        model_path = template['architecture']['base_model_path']
        provenance = ORIGINAL / 'result/llama31_8b_w25_20261004/model_source_ranges16.json'
    meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS[model], 'Wrong paired base checkpoint.')
    architecture = model_inventory(model_path)
    _require(architecture['model_type'] == model, 'The actual HF model type differs from the selection.')
    stats.append(q['calibration_source']['arrow_stat'])
    for row in stats:
        check_stat(row)
    argv, normalized = recipe(model_path, output / 'donor',
        layers=architecture['num_hidden_layers'], block_size=block_size, steps=steps)
    import_paths = {}
    for name in ('train_utils.cat_train_pipeline', 'rotation.hadamard_utils', 'rotation.hadamard_extended',
                 'litebsq.vae_linear', 'experiments.w25_eval.assemble_categories', 'tools.cat_train'):
        path = Path(importlib.import_module(name).__file__).resolve()
        _require(path.is_relative_to(ROOT), 'Import escaped the snapshot: ' + name)
        import_paths[name] = str(path)
    sources = {str(ROOT / name): sha256(ROOT / name) for name in q['source_sha256']}
    # The old template pins are checked for every unchanged runtime file.
    for name, value in q['source_sha256'].items():
        if name != 'rotation/hadamard_utils.py':
            _require(sources[str(ROOT / name)] == value, 'Unexpected shared-code difference: ' + name)
    for directory in ('experiments/full_rht_gate_pair', 'rotation'):
        sources.update({str(p): sha256(p) for p in (ROOT / directory).glob('*.py')})
    for name in ('experiments/w25_eval/acceptance.py', 'experiments/longbench_v1/common.py',
                 'experiments/rotation_factor_probe/capture.py',
                 'experiments/packed_decode_evaluation/prepare.py',
                 'experiments/packed_decode_evaluation/run.py', 'train_utils/utils.py',
                 'train_utils/eval_utils.py'):
        sources[str(ROOT / name)] = sha256(ROOT / name)
    sources.update(evaluation_sources(controls))
    evidence = {Path(p) for p in controls}
    evidence.update((Q_TEMPLATE, base / 'checkpoint_meta.json', base / 'config.json',
        snapshot_review, provenance, Path(q['calibration_source']['metadata']['path'])))
    # L_TEMPLATE is live: pin the immutable original inputs, not its changing status record.
    if model == 'llama':
        evidence.update(Path(p) for p in template['input_evidence_sha256']
                        if Path(p).name in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json'))
    evidence.update(Path(model_path) / name for name in
                    ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'))
    if steps == 10000:
        _require(validation is not None, 'Formal full/block pair requires completed native validation.')
        validation = Path(validation).resolve()
        proof = read(validation)
        _require(proof['status'] == 'verified' and proof['source_root'] == str(ROOT),
                 'Native validation does not cover this snapshot.')
        evidence.add(validation)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': model, 'architecture': architecture, 'base_dir': str(base),
        'base_checkpoint_id': meta['checkpoint_id'], 'control_result_paths': controls,
        'block_size': block_size, 'steps_per_stage': steps, 'full_evaluation': steps == 10000,
        'expected_user_linear_bits': EXPECTED_BITS[model],
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': {str(p): sha256(p) for p in evidence},
        'protected_stats': stats, 'snapshot_import_paths': import_paths,
        'environment': {'PYTHONPATH': str(ROOT), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        'timeout_seconds': 2700 if steps == 10000 else 900, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.full_rht_gate_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'scope': 'Fresh original-HF gate category, same protected count and bit rate; no KD or LoRA. Short models never seed formal training.',
        'protocol': {'mmlu_batch': 64, 'seven_batch': 32, 'num_fewshot': 0, 'limit': None,
                     'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'}}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'launch.json', record)
    print({'status': record['status'], 'model_type': model, 'steps_per_stage': steps,
           'block_size': block_size, 'source_pins': len(sources), 'input_pins': len(evidence),
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'llama'), required=True)
    parser.add_argument('--block-size', type=int, choices=(0, 32), required=True)
    parser.add_argument('--steps', type=int, choices=(2, 10000), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot-review', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    prepare(args.model, args.block_size, args.steps, args.output_dir, args.snapshot_review, args.validation)

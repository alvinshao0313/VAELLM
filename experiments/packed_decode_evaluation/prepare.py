"""Bind a CPU-only same-native7200 packed-cache evaluation, without exporting weights."""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys

import torch
import yaml

from experiments.packed_decode_evaluation.run import (
    CONTROL_ID, NATIVE_ID, REPO, SOURCE_FILES, SOURCE_ID, check, control_results, read,
)
from experiments.fp32_decoder_export.export import inspect_scope
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_training_model_state


def evaluation_sources(control_paths):
    """Pin installed evaluator and the actual64 task YAMLs/includes without loading data."""
    from lm_eval.tasks import TaskManager
    modules = ('lm_eval.api.model', 'lm_eval.api.task', 'lm_eval.evaluator',
               'lm_eval.evaluator_utils', 'lm_eval.models.huggingface',
               'lm_eval.models.utils', 'lm_eval.tasks', 'lm_eval.utils')
    paths = {Path(importlib.import_module(name).__file__).resolve() for name in modules}
    manager = TaskManager()
    leaves = {name for path in control_paths for name in read(path)['evals']['lm_eval']['n_samples']}
    _require(len(leaves) == 64, 'Require the original complete64 task leaves.')
    pending = [Path(manager.task_index[name]['yaml_path']).resolve() for name in sorted(leaves | {'mmlu'})]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path); paths.add(path)
        # BaseLoader reads tags as data; no custom task function is executed here.
        config = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        included = config.get('include', [])
        for name in [included] if isinstance(included, str) else included:
            candidate = Path(name)
            pending.append((candidate if candidate.is_file() else path.parent / candidate).resolve())
        paths.update(path.parent.glob('*.py'))
    return {str(path): sha256(path) for path in sorted(paths)}


def prepare(control_dir, output):
    control_dir, output = Path(control_dir).resolve(), Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(),
             'Use a fresh output and explicitly CPU-only preparation.')
    torch.set_num_threads(4)
    previous, review = (read(control_dir / name) for name in ('launch.json', 'review.json'))
    _require(previous['status'] == 'completed' and previous['exit_code'] == 0
             and review['status'] == 'verified' and review['checkpoint_id'] == CONTROL_ID
             and review['native_checkpoint_id'] == NATIVE_ID and review['source_checkpoint_id'] == SOURCE_ID,
             'Require the completed independently reviewed same-native FP32 control.')
    exits = [control_dir / (name + '.exit') for name in
             ('export', 'audit', 'mmlu', 'seven', 'summary', 'dispatcher', 'outer')]
    _require(all(path.read_text().strip() == '0' for path in exits), 'The reviewed control must have exited naturally.')
    check_hashes(previous['source_sha256'])
    native = Path(previous['training_checkpoint_dir']).resolve()
    meta, trainer_state, snapshot, source, base, architecture = inspect_scope(native)
    _require(meta['checkpoint_id'] == NATIVE_ID and base['checkpoint_id'] == SOURCE_ID,
             'Require the actual native7200 and its original552 round base.')
    control = control_dir / 'final_model'
    control_paths = [control_dir / name for name in ('mmlu.json', 'seven.json')]
    record = {'source_checkpoint_dir': str(source), 'source_checkpoint_id': SOURCE_ID,
        'training_checkpoint_dir': str(native), 'native_checkpoint_id': NATIVE_ID,
        'control_checkpoint_dir': str(control), 'control_checkpoint_id': CONTROL_ID,
        'control_result_paths': [str(path) for path in control_paths], 'architecture': architecture}
    control_meta, control_audit, results = control_results(record)
    exported = read(control / 'export_record.json')
    _require(exported['checkpoint_id'] == CONTROL_ID and exported['source_global_step'] == 7200
             and exported['source_checkpoint_id'] == NATIVE_ID
             and Path(exported['source_checkpoint']).resolve() == native,
             'The control must use the identical native decoder values.')
    check_hashes(exported['source_files_sha256'])
    check_stat(exported['round_base_weight_stat'])
    hf = Path(architecture['base_model_path'])
    index = read(hf / 'model.safetensors.index.json')['weight_map']
    weight_paths = {str(source / base['state_dict_file'])}
    weight_paths.update(str(hf / name) for name in set(index.values()))
    stats = [row for row in previous['protected_weight_stats'] if row['path'] in weight_paths]
    _require({row['path'] for row in stats} == weight_paths, 'Original source/HF identity was not reviewed.')
    for row in stats:
        check_stat(row)
    # The retired FP32 control weights are never read or required.
    original = torch.load(source / base['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    mutable, manifest = load_v6_training_model_state(str(native), map_location='cpu')
    keys = {name for name in original if '._parallel_stage_decoder.' in name}
    _require(len(original) == 3423 and len(keys) == len(mutable) == len(manifest) == 1512
             and keys == set(mutable) and all(value.dtype == torch.float32
                 and value.shape == original[name].shape and bool(torch.isfinite(value).all())
                 for name, value in mutable.items()), 'Actual native FP32 mutable inventory differs.')
    expected = {**original, **mutable}
    budget = audit_state_dict(control_meta, expected, max_bpw=2.5, max_lora_rank=0)
    fields = ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components',
              'outside_linear_bits', 'full_model_tensor_bytes')
    _require(budget['budget_passed'] and not budget['contract_violations']
             and all(budget[key] == control_audit['bit_budget'][key] for key in fields),
             'Actual source plus native state must equal the reviewed FP32 storage budget.')
    sources = dict(previous['source_sha256'])
    for relative in SOURCE_FILES:
        path = REPO / relative
        _require(str(path) not in sources or sources[str(path)] == sha256(path), 'A reused source changed.')
        sources[str(path)] = sha256(path)
    sources.update(evaluation_sources(control_paths))
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    evidence_paths = set(map(Path, exported['source_files_sha256']))
    evidence_paths.update((source / 'checkpoint_meta.json', control / 'checkpoint_meta.json',
        control / 'export_record.json', control_dir / 'audit.json', control_dir / 'launch.json',
        control_dir / 'review.json', control_dir / 'full8_summary.json', *control_paths, *exits))
    evidence_paths.update(hf / name for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
        'generation_config.json', 'model.safetensors.index.json'))
    inputs = {str(path): sha256(path) for path in sorted(evidence_paths)}
    _require(inputs[str(native / 'training_model_state.pt')] == control_audit['native_training_model_state_sha256'],
             'Reviewed native tensor bytes changed.')
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        cwd=str(REPO), python_executable=sys.executable, output_dir=str(output),
        source_sha256=sources, input_evidence_sha256=inputs, protected_weight_stats=stats,
        bit_budget=budget, control_mean_percent=results['mean_percent'],
        control_mmlu_percent=100 * results['task_metrics']['mmlu'],
        cpu_checks={'native_global_step': trainer_state['global_step'], 'source_tensors': len(original),
            'native_decoder_tensors': len(mutable), 'native_decoder_elements': sum(v.numel() for v in mutable.values()),
            'frozen_source_tensors': len(original) - len(mutable), 'native_manifest_entries': len(manifest),
            'same_native_file_as_reviewed_control': True, 'control_weights_read': False,
            'runtime_restore_and_cache_parity': 'pending_actual_GPU_run'},
        protocol={'tasks': ['mmlu', 'boolq', 'rte', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa', 'piqa'],
            'mmlu_batch': 64, 'seven_batch': 32, 'limit': None, 'num_fewshot': 0,
            'full8_leaves': 64, 'full8_samples': 24742, 'optimizer_updates': 0,
            'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Same true native7200 FP32 state; only training-path packed BF16 materialization versus reviewed wholefused deployment.'},
        environment={'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        timeout_seconds=1200, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.packed_decode_evaluation.run',
            '--launch_record', str(output / 'launch.json')],
        package_versions={name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'lm_eval', 'datasets', 'flash-attn')})
    _require(not torch.cuda.is_initialized(), 'Preparation must not initialize CUDA.')
    record['cuda_initialized'] = False
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'launch.json', record)
    print({'status': 'prepared_unbound', 'launch_sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'cuda_initialized': False,
        'bpw': budget['user_linear_bpw'], 'control_mean_percent': results['mean_percent']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path,
        default=REPO / 'result/qwen3_8b_mmlu65_20261004/ce7200_fp32_decoder_export')
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.control_dir, args.output_root)

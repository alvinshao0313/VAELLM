"""Build native CPU-only request identities; do not reserve or launch a GPU."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys

from experiments.longbench_v1.common import file_stat, sha256, source_pins, write_json
from experiments.mmlu_attribution.protocol import ARMS, PROTOCOL, case_identity, require


def prepare(output, hf_path, checkpoint):
    import torch
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import TaskManager, get_task_dict
    from experiments.qwen3_w25.summarize_results import MMLU_SAMPLE_COUNTS
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import validate_checkpoint
    from experiments.longbench_v1.protocol import model_policy
    output = Path(output).resolve()
    hf_path, checkpoint = Path(hf_path).resolve(), Path(checkpoint).resolve()
    require(not output.exists(), 'Use a new diagnostic output directory.')
    require(not torch.cuda.is_available(), 'CPU preparation requires CUDA_VISIBLE_DEVICES empty.')
    architecture = model_inventory(hf_path)
    checkpoint, meta, budget = validate_checkpoint(checkpoint, architecture)
    policy = model_policy(hf_path)
    saved = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu', mmap=True, weights_only=True)
    require(isinstance(saved, dict) and saved and all(torch.is_tensor(value) for value in saved.values()),
            'Expected a flat complete native deployment state.')
    expected_saved_tensors = len(saved)
    del saved
    cases, tasks, data_pins = [], {}, {}
    task_dict = get_task_dict(['mmlu'], TaskManager())
    leaves = get_task_list(task_dict)
    require({item.task_name for item in leaves} == set(MMLU_SAMPLE_COUNTS), 'Native task inventory changed.')
    for item in sorted(leaves, key=lambda row: row.task_name):
        task, name = item.task, item.task_name
        task.set_config(key='num_fewshot', value=0)
        task.set_fewshot_seed(seed=1234)
        require(len(task.eval_docs) == MMLU_SAMPLE_COUNTS[name], 'Original subject test count changed.')
        require(task.get_config('dataset_path') == 'hails/mmlu_no_train'
                and task.get_config('dataset_name') == name.removeprefix('mmlu_')
                and task.get_config('test_split') == 'test' and task.OUTPUT_TYPE == 'multiple_choice',
                'Unexpected task dataset/split/output type.')
        task.build_all_requests(limit=32, rank=0, world_size=1, cache_requests=False,
            rewrite_requests_cache=False, system_instruction=None, apply_chat_template=False,
            fewshot_as_multiturn=False, chat_template=None, tokenizer_name='')
        grouped = defaultdict(list)
        for request in task.instances:
            require(request.request_type == 'loglikelihood' and request.repeats == 1,
                    'Native MMLU request type/repeats changed.')
            grouped[request.doc_id].append(request)
        require(set(grouped) == set(range(32)), 'Native diagnostic selection changed.')
        for doc_id, requests in sorted(grouped.items()):
            requests.sort(key=lambda request: request.idx)
            require([request.idx for request in requests] == [0, 1, 2, 3], 'Choice ordering changed.')
            row = case_identity(name, doc_id, requests[0].doc, [request.args for request in requests])
            require(all(request.doc == row['doc'] for request in requests), 'Choice documents disagree.')
            cases.append(row)
        for item in task.eval_docs.cache_files:
            data_pins[item['filename']] = sha256(item['filename'])
        tasks[name] = {'original_samples': len(task.eval_docs), 'effective_samples': 32,
                      'dataset_fingerprint': task.eval_docs._fingerprint, 'metric': 'acc,none'}
    require(len(cases) == 1824 and not torch.cuda.is_initialized(), 'CPU preparation or coverage failed.')
    root = Path(__file__).resolve().parents[2]
    pins = source_pins()
    for directory in (Path(__file__).parent, root / 'experiments/longbench_attribution'):
        pins.update({str(path): sha256(path) for path in directory.glob('*.py')})
    pins[str(root / 'experiments/qwen3_w25/summarize_results.py')] = sha256(root / 'experiments/qwen3_w25/summarize_results.py')
    pins[str(root / 'experiments/w25_eval/acceptance.py')] = sha256(root / 'experiments/w25_eval/acceptance.py')
    import lm_eval.api.model, lm_eval.api.task, lm_eval.evaluator, lm_eval.evaluator_utils
    import lm_eval.models.huggingface, lm_eval.models.utils
    for module in (lm_eval.api.model, lm_eval.api.task, lm_eval.evaluator, lm_eval.evaluator_utils,
                   lm_eval.models.huggingface, lm_eval.models.utils):
        pins[module.__file__] = sha256(module.__file__)
    task_directory = Path(lm_eval.__file__).parent / 'tasks/mmlu'
    require(task_directory.is_dir(), 'Installed MMLU task source directory is missing.')
    pins.update({str(path): sha256(path) for path in task_directory.rglob('*')
                 if path.suffix in ('.yaml', '.py')})
    evidence = {str(path): sha256(path) for path in
        (checkpoint / 'checkpoint_meta.json', checkpoint / meta['state_dict_file'], hf_path / 'config.json',
         hf_path / 'tokenizer.json', hf_path / 'tokenizer_config.json', hf_path / 'generation_config.json',
         hf_path / 'model.safetensors.index.json')}
    evidence.update(data_pins)
    index = json.loads((hf_path / 'model.safetensors.index.json').read_text())['weight_map']
    output.mkdir(parents=True)
    case_path = output / 'cases.jsonl'
    with case_path.open('x', encoding='utf-8') as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    record = {'schema': 'mmlu_category_restore_v1', 'status': 'prepared_not_scheduled', 'gpu': None,
        'source_checkpoint': str(checkpoint), 'source_checkpoint_id': meta['checkpoint_id'], 'hf_path': str(hf_path),
        'protocol': PROTOCOL, 'architecture': architecture, 'model_policy': policy,
        'expected_serialized_tensors': expected_saved_tensors, 'source_bit_budget': budget,
        'selected_targets': {arm: [name for name in architecture['projection_shapes']
            if name.rsplit('.', 1)[1] in modules] for arm, modules in ARMS.items()},
        'cases': {'path': str(case_path), 'sha256': sha256(case_path), 'count': len(cases)},
        'subjects': tasks, 'source_sha256': pins, 'input_evidence_sha256': evidence,
        'hf_weight_stats': [file_stat(hf_path / name) for name in sorted(set(index.values()))],
        'package_versions': {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'lm_eval', 'datasets', 'flash-attn')},
        'input_tokens': 'Not computed on CPU. Runtime observes actual HFLM likelihood-token arguments and model-call input hashes without re-encoding or changing native padding/truncation.',
        'resource_estimate': {'gpus': 1, 'planned_physical_gpu': 0, 'estimated_total_minutes': [20, 40],
            'planning_window_minutes': 50, 'automatic_wallclock_stop': False, 'estimated_peak_gib': [55, 70],
            'basis': 'Rank0 full MMLU reference ~296 seconds, 56168 likelihood requests ~192 seconds. Loading/task construction/state audits do not scale with sample count.'},
        'cuda_initialized': False, 'queued': False, 'automatic_retry': False,
        'acceptance': False, 'notes': 'Mixed BF16 projections are diagnostics only; no training or model export. Equal subject samples differ from full MMLU sample weighting.'}
    record['arm_commands'] = [{'arm': arm, 'argv': [sys.executable, '-u', '-m', 'experiments.mmlu_attribution.run',
        '--preparation', str(output / 'preparation.json'), '--arm', arm, '--output', str(output / 'arms' / arm)]}
        for arm in ARMS]
    record['comparison_argv'] = [sys.executable, '-u', '-m', 'experiments.mmlu_attribution.compare',
        '--preparation', str(output / 'preparation.json'), '--results_root', str(output / 'arms'),
        '--output', str(output / 'comparison.json')]
    write_json(output / 'preparation.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hf_path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.output, args.hf_path, args.checkpoint)
    print(json.dumps({'status': record['status'], 'cases': record['cases'],
        'subject_count': len(record['subjects']), 'source_pins': len(record['source_sha256']),
        'evidence_pins': len(record['input_evidence_sha256']), 'cuda_initialized': False}, indent=2))

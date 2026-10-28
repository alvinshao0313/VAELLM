"""Compare complete native 0/5-shot pairs without changing the original 0-shot result."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from .prepare import BEST, CHECKPOINT, PREVIOUS, SOURCE_ID, read, require, sha

PREPARED_SHA = '3c76ba95c1591b8dcac2ee5d9cf76c45902886ca5b311e070922e923c2f5bca3'
NINE = BEST / 'evaluation_full9/longbench/merged/nine_task_acceptance.json'


def near(a, b):
    return math.isclose(a, b, rel_tol=0., abs_tol=1e-12)


def inputs(directory):
    prepared_path = directory / 'prepared.json'
    require(sha(prepared_path) == PREPARED_SHA, 'The reviewed preparation changed.')
    prepared = read(prepared_path)
    for pins in (prepared['source_sha256'], prepared['reusable_baseline_pins'], prepared['hf_source']['metadata_sha256']):
        for path, expected in pins.items():
            require(sha(path) == expected, 'A bound source/control changed: ' + path)
    for item in (*prepared['hf_source']['weight_files'], *prepared['dataset_cache_stats']):
        value = Path(item['path']).stat()
        require(item == dict(path=item['path'], size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino), 'The HF model or dev/test cache changed.')
    weight = CHECKPOINT / 'pytorch_model.bin'
    value = weight.stat()
    require(prepared['source_weight_stat'] == dict(path=str(weight), bytes=value.st_size,
        mtime_ns=value.st_mtime_ns, inode=value.st_ino), 'The actual bestabs weights changed.')
    evaluator = next(Path(path) for path in prepared['reusable_baseline_pins'] if path.endswith('/lm_eval/evaluator.py'))
    function = next(node for node in ast.parse(evaluator.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'simple_evaluate')
    defaults = dict(zip([arg.arg for arg in function.args.args][-len(function.args.defaults):], function.args.defaults))
    expected = dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    require({key: ast.literal_eval(defaults[key]) for key in expected} == expected, 'Native compressed evaluator default seeds changed.')
    normalized = deepcopy(prepared['baseline_task_configs'])
    for configs in normalized.values():
        for config in configs.values():
            config['num_fewshot'] = 0
    require(normalized['0'] == normalized['5'] and prepared['baseline_task_versions']['0'] == prepared['baseline_task_versions']['5'], 'The two baseline settings differ beyond few-shot count.')
    spec = importlib.util.spec_from_file_location('_bestabs_prior_bf16_extract', PREVIOUS / 'analyze.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    previous = read(PREVIOUS / 'launch.prepared.json')
    for shot, baseline in prepared['reused_BF16'].items():
        actual, configs, versions = prior.extract(Path(baseline['path']), 'bf16', int(shot), 16, previous)
        require(actual == baseline and configs == prepared['baseline_task_configs'][shot]
            and versions == prepared['baseline_task_versions'][shot], 'The complete reused B16 BF16 result/protocol changed.')
    nine = read(NINE)
    require(nine['checkpoint_id'] == SOURCE_ID and nine['checkpoint_dir'] == str(CHECKPOINT)
        and nine['quality_target_evaluated'] and nine['protocol']['longbench_tasks'] == 21
        and nine['protocol']['longbench_samples_per_model'] == 4750
        and nine['protocol']['maximum_absolute_drop_points'] == .8, 'The original complete source nine-term evaluation differs.')
    require(near(math.fsum(nine['baseline_scores'].values()) / 9, nine['baseline_nine_task_score'])
        and near(math.fsum(nine['compressed_scores'].values()) / 9, nine['compressed_nine_task_score']), 'Original nine-term arithmetic differs.')
    for side in ('baseline', 'compressed'):
        for item in nine['downstream_' + side]['input_results']:
            require(sha(item['path']) == item['sha256'], 'Original seven/MMLU evidence changed.')
        entry = nine['longbench_inputs']['dense_scores' if side == 'baseline' else 'compressed_scores']
        require(sha(entry['path']) == entry['sha256'] and sha(entry['config_path']) == entry['config_sha256'], 'Original complete LongBench evidence changed.')
        scores = read(entry['path'])
        require(scores['status'] == 'complete' and scores['sample_count'] == 4750 and len(scores['tasks']) == 21
            and near(scores['task_macro_mean'], nine[side + '_scores']['longbench_v1'])
            and scores['model'] == nine['longbench_model_identities']['dense' if side == 'baseline' else 'compressed'], 'LongBench completeness, identity or macro differs.')
    return prepared, nine, prior, previous


def extract(path, shot, batch, prepared):
    payload = read(path)
    require(payload['checkpoint_id'] == SOURCE_ID and payload['checkpoint_dir'] == str(CHECKPOINT)
        and payload['base_model_path'] == '/home/star/models/Qwen/Qwen3-8B'
        and payload['checkpoint_loader'] == 'v6' and payload['strict_load'] is True
        and payload['adapter_dir'] is None and not payload['eval_hif4_act'] and payload['tokenizer_is_fast'], 'Wrong actual native sparse model/loader/tokenizer.')
    warmup = payload['evals']['cache_warmup']
    require(warmup['total'] == warmup['warmed'] == 252 and warmup['failed'] == warmup['skipped'] == 0
        and warmup['prewarm_dtype'] == 'torch.bfloat16' and warmup['prewarm_group_size'] == 8, 'Original BF16/group8 warmup differs.')
    result = payload['evals']['lm_eval']
    require(result['tasks'] == ['mmlu'] and result['num_fewshot'] == shot
        and result['batch_size'] == batch and result['limit'] is None, 'Wrong MMLU shot, batch, task or sample limit.')
    counts = {task: row['test_rows'] for task, row in prepared['subjects'].items()}
    require(len(counts) == 57 and sum(counts.values()) == 14042
        and set(result['n_samples']) == set(result['n_shot']) == set(result['configs']) == set(counts)
        and result['configs'] == prepared['baseline_task_configs'][str(shot)]
        and result['versions'] == prepared['baseline_task_versions'][str(shot)], 'Student and same-shot BF16 task definitions/versions differ.')
    subjects = {}
    for task, count in counts.items():
        require(result['n_samples'][task] == dict(original=count, effective=count)
            and result['n_shot'][task] == shot, 'A complete subject count/shot differs.')
        accuracy = result['raw_results'][task]['acc,none']
        require(math.isfinite(accuracy) and 0 <= accuracy <= 1 and abs(accuracy * count - round(accuracy * count)) < 1e-8, 'Noninteger native subject correctness.')
        subjects[task] = dict(correct=round(accuracy * count), samples=count)
    correct = sum(item['correct'] for item in subjects.values())
    accuracy = correct / 14042
    aggregates = [group['mmlu']['acc,none'] for group in (result['raw_results'], result.get('group_results', {})) if 'mmlu' in group]
    require(aggregates and all(near(value, accuracy) for value in aggregates), 'MMLU group is not the complete micro average.')
    artifacts = result['sample_artifacts']
    require(artifacts['total_count'] == 14042 and set(artifacts['tasks']) == set(counts), 'Missing complete question-level native artifacts.')
    for task, item in artifacts['tasks'].items():
        sample = Path(item['path']).resolve()
        require(sample.is_relative_to(path.parent.resolve()) and item['count'] == counts[task]
            and sha(sample) == item['sha256'], 'Original native sample binding changed.')
        lines, doc_ids, hits = 0, set(), 0.
        with sample.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                lines += 1
                doc_ids.add(row['doc_id'])
                hits += row['acc']
        require(lines == counts[task] and doc_ids == set(range(lines))
            and near(hits, subjects[task]['correct']), 'Raw complete sample IDs/correctness differ from native counts.')
    return dict(path=str(path), sha256=sha(path), correct=correct, documents=14042,
        subjects=subjects, num_fewshot=shot, batch_size=batch, score_percent=accuracy * 100,
        complete_native_samples_verified=True, sample_artifacts=artifacts)


def compare(nine, baseline_percent, compressed_percent):
    baseline = {**nine['baseline_scores'], 'mmlu': baseline_percent}
    compressed = {**nine['compressed_scores'], 'mmlu': compressed_percent}
    require(len(baseline) == len(compressed) == 9, 'Use exactly seven unchanged downstream terms plus MMLU and LB21.')
    bscore, cscore = math.fsum(baseline.values()) / 9, math.fsum(compressed.values()) / 9
    return dict(baseline_scores=baseline, compressed_scores=compressed,
        baseline_nine_task_score=bscore, compressed_nine_task_score=cscore, mean_drop_points=bscore - cscore)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true', help='Validate extraction and arithmetic on complete real prior results, without current-arm results.')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU analysis requires CUDA hidden.')
    output = args.run_dir / ('analysis_verification.json' if args.verify_existing else 'paired_report.json')
    require(not output.exists(), 'Do not overwrite a completed verification or paired result.')
    prepared, nine, prior, previous = inputs(args.run_dir)
    old_compressed = extract(BEST / 'evaluation_full9/mmlu.json', 0, 64, prepared)
    require(near(old_compressed['score_percent'], nine['compressed_scores']['mmlu']), 'Original B64 compressed score differs.')
    old_baseline_entry = next(item for item in nine['downstream_baseline']['input_results'] if item['tasks'] == ['mmlu'])
    old_baseline, _, _ = prior.extract(Path(old_baseline_entry['path']), 'bf16', 0, 64, previous)
    require(near(old_baseline['score_percent'], nine['baseline_scores']['mmlu']), 'Original B64 BF16 score differs.')
    if args.verify_existing:
        old_report = read(PREVIOUS / 'paired_report.json')
        old_nine = read(prior.NINE)
        verified = {}
        for shot in ('0', '5'):
            values = old_report['paired_comparisons'][shot]
            actual = compare(old_nine, values['bf16_mmlu_percent'], values['a0e5_mmlu_percent'])
            require(near(actual['baseline_nine_task_score'], values['baseline_nine_item_score'])
                and near(actual['compressed_nine_task_score'], values['compressed_nine_item_score'])
                and near(actual['mean_drop_points'], values['nine_item_absolute_drop_points']), 'New arithmetic does not reproduce the real prior paired report.')
            verified[shot] = actual
        report = dict(status='verified_existing_real_data', original_bestabs_B64_correct=old_compressed['correct'],
            original_bestabs_B64_score=old_compressed['score_percent'], prior_pair_arithmetic=verified,
            checks=['All actual source57 subjects/14042 JSONL IDs/hashes/correctness and integer micro agree.',
                'Current student/BF16 same-shot configs and versions agree; 0/5 configs differ only in num_fewshot.',
                'Original source seven+LB21 evidence and identities unchanged.', 'Both real 10/9 paired nine-term results reproduced to1e-12.'])
    else:
        runs, comparisons = {}, {}
        for shot in ('0', '5'):
            directory = args.run_dir / f'bestabs_shot{shot}'
            require((args.run_dir / f'bestabs_shot{shot}.exit').read_text().strip() == '0', 'Incomplete or failed outer arm.')
            run = read(directory / 'run.json')
            plan = prepared['commands'][shot]
            require(run['status'] == 'completed' and run['prepared_sha256'] == PREPARED_SHA
                and run['shot'] == int(shot) and run['gpu'] == plan['gpu']
                and run['stages']['mmlu']['command'] == plan['argv']
                and run['stages']['mmlu']['cwd'] == plan['cwd'] and run['stages']['mmlu']['exit_code'] == 0
                and run['environment'] == {**prepared['environment'], 'CUDA_VISIBLE_DEVICES': str(plan['gpu'])}, 'The real execution differs from the prepared protocol.')
            for path, expected in run['source_sha256'].items():
                require(sha(path) == expected, 'A started driver dependency changed.')
            runs[shot] = extract(directory / 'mmlu.json', int(shot), 16, prepared)
            baseline = prepared['reused_BF16'][shot]
            comparisons[shot] = dict(num_fewshot=int(shot), batch_size=16,
                bf16_correct=baseline['correct'], bestabs_correct=runs[shot]['correct'],
                mmlu_correct_gap=baseline['correct'] - runs[shot]['correct'],
                **compare(nine, baseline['score_percent'], runs[shot]['score_percent']))
            current = comparisons[shot]
            current.update(maximum_absolute_drop_points=.8,
                drop_le_0_8=current['mean_drop_points'] <= .8 + 1e-12,
                logical_budget_passed=prepared['source_audit']['logical_bpw'] <= 2.6,
                rank0=True, protocol='Explicit same-B16 alternative shot pair; original zero-shot acceptance unchanged')
            current['passed'] = current['drop_le_0_8'] and current['logical_budget_passed'] and current['rank0']
        preferred = min(('0', '5'), key=lambda shot: (comparisons[shot]['mmlu_correct_gap'], int(shot)))
        report = dict(status='complete_cpu_verified', source_checkpoint_id=SOURCE_ID, protocol=prepared['protocol'],
            source_audit=prepared['source_audit'], reused_BF16=prepared['reused_BF16'], bestabs_runs=runs,
            paired_comparisons=comparisons, preferred_shot_for_reporting=int(preferred),
            selection=dict(selected_after_observing_results=True, both_protocols_fully_disclosed=True,
                criterion='Smaller observed same-protocol BF16 minus bestabs nine-term drop; exact integer MMLU gap breaks numerical ties; tie goes to0shot.',
                original_zero_shot_acceptance_modified=False),
            original_B64_zero_shot=dict(path=str(NINE), sha256=sha(NINE), baseline=old_baseline,
                bestabs=old_compressed, baseline_nine_task_score=nine['baseline_nine_task_score'],
                compressed_nine_task_score=nine['compressed_nine_task_score'], mean_drop_points=nine['mean_drop_points']),
            batch_calibration=dict(bf16_B16_minus_B64_correct=prepared['reused_BF16']['0']['correct'] - old_baseline['correct'],
                bestabs_B16_minus_B64_correct=runs['0']['correct'] - old_compressed['correct']),
            limits=['Few-shot changes the evaluation protocol; there is no training or model improvement claim.',
                'The same seven other tasks and complete LB21 scores remain tied to their identical original models.',
                'Observed shot selection is disclosed; neither statistical significance nor an independent unseen validation is claimed.',
                'Old BF16 runs lack per-question choices, so this report does not infer paired repairs/damages.'])
    report.update(prepared_sha256=PREPARED_SHA, analysis_source_sha256=sha(__file__), finished_unix=time.time())
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(output), sha256=sha(output),
        paired_drops={key: row['mean_drop_points'] for key, row in report.get('paired_comparisons', {}).items()})), flush=True)


if __name__ == '__main__':
    main()

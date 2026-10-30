"""Select once from the two completed, bound A250/A500 full8 evaluations."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from fractions import Fraction

DATA = Path('/data/star/VAELLM/cat_category_selection_20261029')
PLAN = DATA / 'evaluation_plan.json'
OUTPUT = DATA / 'selection.json'
TASKS = {'boolq': 'acc,none', 'rte': 'acc,none', 'winogrande': 'acc,none',
         'arc_easy': 'acc_norm,none', 'arc_challenge': 'acc_norm,none',
         'openbookqa': 'acc_norm,none', 'piqa': 'acc_norm,none', 'mmlu': 'acc,none'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def close(actual, expected, message):
    # The original acceptance module uses rel_tol0/abs_tol1e-12 for score arithmetic.
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def completed_endpoint(step, row, plan_sha):
    output = Path(row['output']).resolve()
    require(output == DATA / f'evaluation/A{step}/full8', 'Unexpected fixed A endpoint result directory.')
    evaluation_path = output / 'evaluation.json'
    require(evaluation_path.is_file(), f'A{step} has no completed evaluation.json; do not select from partial results.')
    evaluation = read(evaluation_path)
    require(evaluation['status'] == 'complete_original_full8' and evaluation.get('exit_code') == 0
            and evaluation.get('leaf_tasks') == 64 and evaluation.get('sample_count') == 24742,
            f'A{step} complete full8/64-leaf/24742 evidence is not available; no selection.')
    require(evaluation['checkpoint_id'] == row['checkpoint_id']
            and Path(evaluation['checkpoint_dir']).resolve() == Path(row['checkpoint_dir']).resolve()
            and Path(evaluation['source_plan']['path']).resolve() == PLAN
            and evaluation['source_plan']['sha256'] == plan_sha,
            f'A{step} completed evaluation does not belong to the frozen plan endpoint.')
    require(len(evaluation['stages']) == len(row['stages']) == 2, 'Require both original full8 stages.')
    for actual, expected in zip(evaluation['stages'], row['stages'], strict=True):
        require(all(actual[key] == expected[key] for key in ('name', 'argv', 'sample_count'))
                and actual['exit_code'] == 0, f'A{step} completed stage differs from the fixed full8 command.')
    summary_path = output / 'full8_summary.json'
    summary = read(summary_path)
    require(summary['checkpoint_id'] == row['checkpoint_id'] and summary['leaf_count'] == 64
            and summary['sample_count'] == 24742 and set(summary['task_metrics']) == set(TASKS)
            and summary['task_metric_keys'] == TASKS,
            f'A{step} full8 summary identity/coverage/task metric keys differ.')
    require(len(summary['input_results']) == 2, 'Require both bound original result summaries.')
    inputs = []
    for item, expected in zip(summary['input_results'], row['stages'], strict=True):
        path = Path(item['path']).resolve()
        leaves = 57 if expected['name'] == 'mmlu' else 7
        batch = 64 if expected['name'] == 'mmlu' else 32
        require(path == output / (expected['name'] + '.json') and item['sha256'] == sha(path)
                and item['leaf_count'] == leaves and item['sample_count'] == expected['sample_count']
                and item['batch_size'] == batch and item['format'] == 'cat_eval_v6',
                f'A{step} full8 summary is not bound to its completed two result shards.')
        inputs.append(binding(path))
    exact = {}
    for task in TASKS:
        value = summary['task_metrics'][task]
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1, 'Invalid complete task metric: ' + task)
        exact[task] = Fraction(str(value))
    mean = sum(exact.values(), Fraction()) * 100 / len(TASKS)
    original_mean = math.fsum(summary['task_metrics'][task] for task in TASKS) * 100 / len(TASKS)
    close(summary['mean_percent'], original_mean, f'A{step} summary mean differs from the original eight-task arithmetic.')
    close(evaluation['full8_mean_percent'], summary['mean_percent'], f'A{step} evaluation/summary mean binding differs.')
    return dict(checkpoint_id=row['checkpoint_id'], checkpoint_dir=row['checkpoint_dir'],
        evaluation=binding(evaluation_path), full8_summary=binding(summary_path), input_results=inputs,
        task_metrics=summary['task_metrics'], mean_percent=summary['mean_percent'],
        exact_mean_percent=str(mean), leaf_tasks=64, sample_count=24742,
        raw_artifacts_evidence='Reuses each complete evaluator\'s raw artifact validation; predictions are not rescanned.'), exact, mean, summary['versions']


def summarize():
    require(Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(), 'Use caller-activated bitvae.')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Read-only selection runs with CUDA hidden.')
    require(not OUTPUT.exists(), 'One selection only; never overwrite selection.json.')
    plan = read(PLAN)
    plan_sha = sha(PLAN)
    require(plan['schema'] == 'A_gate250_500_original_full8_plan_v1' and set(plan['endpoints']) == {'250', '500'}
            and plan['structure_selected_before_evaluation'] == 'A control; no B selection'
            and plan['leaf_tasks'] == 64 and plan['sample_count'] == 24742 and plan['zero_shot'] is True
            and plan['mmlu_batch'] == 64 and plan['seven_batch'] == 32 and plan['limit'] is None,
            'Use only the frozen A250/A500 original full8 plan.')
    rows, exact, means, versions = {}, {}, {}, {}
    for step in (250, 500):
        rows[str(step)], exact[step], means[step], versions[step] = completed_endpoint(step, plan['endpoints'][str(step)], plan_sha)
    require(versions[250] == versions[500], 'Completed full8 task versions differ.')
    require(sha(PLAN) == plan_sha, 'Frozen evaluation plan changed while reading.')
    selected = 250 if means[250] > means[500] else 500
    changes = {task: dict(A250=rows['250']['task_metrics'][task], A500=rows['500']['task_metrics'][task],
        delta_percentage_points=float((exact[250][task] - exact[500][task]) * 100),
        exact_delta_percentage_points=str((exact[250][task] - exact[500][task]) * 100)) for task in TASKS}
    result = dict(schema='A_gate250_500_completed_full8_selection_v1', status='selected_from_two_complete_full8',
        selected_candidate=f'A{selected}', selected_checkpoint_id=rows[str(selected)]['checkpoint_id'],
        selected_checkpoint_dir=rows[str(selected)]['checkpoint_dir'], fixed_rule=plan['selection_rule'],
        comparison='Exact Fraction(str(recorded task metric)) arithmetic; A250 strictly higher wins, otherwise A500, including exact ties.',
        exact_tie=means[250] == means[500], exact_mean_delta_percentage_points=str(means[250] - means[500]),
        mean_delta_percentage_points=float(means[250] - means[500]), candidates=rows, per_task=changes,
        selection_count=1, development_selection=True, leaf_tasks_per_candidate=64, samples_per_candidate=24742,
        plan=binding(PLAN), summarizer=binding(Path(__file__).resolve()), pid=os.getpid(), created_unix=time.time(),
        GPU_inference_executed_by_summarizer=False, suffix_training_started=False, nine_task_quality_proven=False,
        boundary='One development full8 choice, not a nine-task or suffix result. Existing A/B fixed500 continues; no automatic suffix or changed training protocol.')
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    try:
        report = summarize()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit('Selection unavailable: ' + str(error))
    print(json.dumps(dict(status=report['status'], selected_candidate=report['selected_candidate'])), flush=True)

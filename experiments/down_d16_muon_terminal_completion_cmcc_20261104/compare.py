"""Compare the completed same-parent cell with its three fixed controls on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.cmcc_runtime_20261030 import sequence as original
from .complete import CHECKPOINT_ID, EVALUATION, ORIGINAL, OUTPUT, TOPIC, read, require, write


def main():
    original.cpu_environment()
    result_path = OUTPUT / 'comparison.json'
    require(not result_path.exists(), 'Never overwrite this completed comparison.')
    run = read(OUTPUT / 'run.json')
    require(run['status'] == 'completed_isolated_native_resume_full9' and run['exit_code'] == 0,
        'Do not compare or score partial predictions.')
    prepared = read(ORIGINAL / 'prepared.json')['controls']
    paths = dict(prepared['acceptance_paths'])
    original_d16_path = paths['D16']
    publication_path = TOPIC / 'down_d16_terminal_recovery_cmcc_20261104/publication/validation.json'
    publication = read(publication_path)
    require(publication['status'] == 'strictly_verified_CPU_terminal_publication' and publication['exit_code'] == 0
        and publication['checkpoint_id'] == prepared['final_ids']['D16'], 'Require the validated same-UUID D16 publication.')
    paths['D16'] = publication['complete_nine']['acceptance_report']
    endpoint = OUTPUT / 'longbench/merged/nine_task_acceptance.json'
    current = original.validate_acceptance(endpoint, endpoint.with_name('nine_task.exit'), CHECKPOINT_ID)
    controls = {name: original.validate_acceptance(path, Path(path).with_name('nine_task.exit'), prepared['final_ids'][name])
                for name, path in paths.items()}
    require(controls['D16']['acceptance_sha256'] == publication['complete_nine']['acceptance_sha256'],
        'D16 publication and actual acceptance differ.')
    deltas = {name: dict(nine_task_score=current['nine_task_score'] - control['nine_task_score'],
        task_scores={task: current['task_scores'][task] - control['task_scores'][task] for task in current['task_scores']})
        for name, control in controls.items()}
    best = max(controls, key=lambda name: controls[name]['nine_task_score'])
    reports = [read(path) for path in read(OUTPUT / 'longbench/merge_plan.json')['execution_units'][0]['native_reports']]
    formal = read(ORIGINAL / 'formal_run.json')
    value = dict(status='complete_same_parent_three_control_comparison', checkpoint_id=CHECKPOINT_ID,
        current=current, controls=controls, delta_vs_controls=deltas, best_control=best,
        delta_vs_best_of_three=deltas[best]['nine_task_score'],
        fixed_predeclared_rule='Retain a same-parent single-seed candidate only if complete nine-task score strictly exceeds all three controls.',
        candidate_disposition='single_seed_combination_candidate' if deltas[best]['nine_task_score'] > 0 else 'end_fixed_point',
        D16_control_rebinding=dict(original_prepared_path=original_d16_path, actual_same_UUID_publication=str(publication_path),
            acceptance_report=paths['D16'],
            original_prepared_unchanged=True, reason='Original D16 timed-out path has no final acceptance; independent CPU publication closes the same original predictions.'),
        cost_seconds=dict(original_fit_and_fresh=formal['stages']['fit']['elapsed_seconds'],
            original_Adam500_and_fresh=formal['stages']['Adam500']['elapsed_seconds'],
            original_full9_incomplete=formal['stages']['full9']['elapsed_seconds'],
            original_LB_native=reports[0]['duration_seconds'], additional_LB_native=reports[-1]['duration_seconds'] if len(reports) > 1 else 0,
            additional_completion=run['elapsed_seconds']),
        complete_coverage=dict(full8_leaves=64, full8_samples=24742, longbench_tasks=21, longbench_samples=4750,
            reused_original_predictions=run['present'], additional_predictions=run['remaining']),
        scope='One same-parent recipe interaction; original failed sequence/full9 retained. No original-BF16 whole-chain, extra seed, cross-model or seven-category conclusion.',
        source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        command=[sys.executable, *sys.argv])
    write(result_path, value)
    print(json.dumps({key: value[key] for key in ('status', 'delta_vs_best_of_three', 'best_control', 'candidate_disposition')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

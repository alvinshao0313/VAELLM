"""Publish complete D16 metadata after CPU scoring; keep both earlier failures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from . import recover as r


def main():
    from experiments.cmcc_runtime_20261030 import sequence as original
    original.cpu_environment()
    output = r.OUTPUT / 'publication'
    r.require(not output.exists(), 'Never overwrite a terminal metadata publication.')
    attempted = r.read(r.OUTPUT / 'run.json')
    r.require(attempted['status'] == 'failed_CPU_terminal_recovery'
        and attempted['stages']['bind_0']['exit_code'] == attempted['stages']['merge_nine']['exit_code'] == 0
        and "KeyError: 'full8_leaf_tasks'" in attempted['error'],
        'Publish only the actual completed CPU scoring with its recorded missing metadata field.')
    diagnosis = r.read(r.OUTPUT / 'diagnosis.json')
    original.check_pins(diagnosis['preserved_inputs_sha256'])
    original.check_pins(diagnosis['active_original_dependencies'])
    full8 = r.read(r.OUTPUT / 'full8_summary.json')
    r.require(full8 == r.read(r.EVALUATION / 'full8/full8_summary.json')
        and full8['checkpoint_id'] == r.CHECKPOINT_ID and full8['leaf_count'] == 64
        and full8['sample_count'] == 24742, 'Require the exact original canonical full8 summary.')
    output.mkdir()
    started = time.time()
    report = deepcopy(r.read(r.OUTPUT / 'evaluation.json'))
    report.update(full8_leaf_tasks=full8['leaf_count'], full8_sample_count=full8['sample_count'],
        full8_mean_percent=full8['mean_percent'],
        metadata_publication=dict(previous_CPU_run=str(r.OUTPUT / 'run.json'),
            previous_CPU_run_sha256=r.sha(r.OUTPUT / 'run.json'),
            canonical_full8_summary=str(r.OUTPUT / 'full8_summary.json'),
            canonical_full8_summary_sha256=r.sha(r.OUTPUT / 'full8_summary.json'),
            added_fields=['full8_leaf_tasks', 'full8_sample_count', 'full8_mean_percent'],
            original_and_CPU_failure_records_unchanged=True,
            GPU_inference_rerun=False, scoring_rerun=False))
    r.write(output / 'evaluation.json', report)
    validation = original.validate_evaluation(output, r.CHECKPOINT_ID)
    merged = Path(r.read(report['longbench_plan'])['merged_output'])
    try:
        original.validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', 'foreign-UUID-must-fail')
    except ValueError:
        foreign_rejected = True
    else:
        raise ValueError('The unchanged validator accepted a foreign UUID.')
    c0_path = r.TOPIC / 'fresh_fixed2_cmcc_20261030/single_20261030_v3/longbench/merged/nine_task_acceptance.json'
    c0 = r.read(c0_path)
    r.require(c0['checkpoint_id'] == 'cbb15c9f-1c1f-4394-817c-65c226d76ce7', 'Wrong original C0 control.')
    validation.update(C0_control=dict(path=str(c0_path), sha256=r.sha(c0_path), checkpoint_id=c0['checkpoint_id'],
        nine_task_score=c0['compressed_nine_task_score']),
        nine_delta_vs_C0=validation['nine_task_score']-c0['compressed_nine_task_score'],
        task_deltas_vs_C0={task:score-c0['compressed_scores'][task] for task,score in validation['task_scores'].items()},
        mmlu_correct=round(validation['task_scores']['mmlu']*14042/100), mmlu_samples=14042)
    original.check_pins(diagnosis['preserved_inputs_sha256'])
    r.write(output / 'validation.json', dict(status='strictly_verified_CPU_terminal_publication', exit_code=0,
        checkpoint_id=r.CHECKPOINT_ID, pid=os.getpid(), start_ticks=original.proc_start(os.getpid()),
        argv=[sys.executable,*sys.argv], started_unix=started, finished_unix=time.time(),
        original_validator_used_unchanged=True, foreign_UUID_rejected=foreign_rejected,
        source_sha256={str(Path(__file__).resolve()): r.sha(__file__), str(Path(r.__file__).resolve()):r.sha(r.__file__)},
        original_inputs_unchanged=True, active_dependencies_unchanged=True, complete_nine=validation))
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Republish CPU acceptance with the real return code; preserve inference evidence."""
from pathlib import Path
import os
import subprocess
import time

from experiments.cmcc_runtime_20261030 import sequence as original
from experiments.cmcc_runtime_20261030.paths import REPO

require, read, write, sha = original.require, original.read, original.write, original.sha


def coverage(output, expected_id):
    """The original complete-evaluation checks, before its acceptance-code check."""
    output = Path(output)
    report = read(output/'evaluation.json')
    require(report['status'] in ('complete_full8_full21_current_goal', 'complete_full8_full21_nine')
        and report['checkpoint_id'] == expected_id and report['full8_leaf_tasks'] == 64
        and report['full8_sample_count'] == 24742 and report['longbench_sample_count'] == 4750,
        'Require the actual complete same-UUID full8/full21 terminal report.')
    plan = read(report['longbench_plan'])
    require(plan['checkpoint_id'] == expected_id and plan['expected_samples'] == 4750
        and len(plan['expected_tasks']) == 21 and not plan['pending_output_directories_at_last_terminal']
        and sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750,
        'Require all original LongBench terminal samples.')
    original.check_pins(plan['pins'])
    scores = read(Path(plan['merged_output'])/'scores.json')
    require(scores['status'] == 'complete' and scores['sample_count'] == 4750
        and len(scores['tasks']) == 21 and scores['model']['checkpoint_id'] == expected_id,
        'Require all original same-UUID LongBench task scores.')
    return report, plan


def republish(output, expected_id, destination):
    destination = Path(destination)
    require(not destination.exists(), 'Never overwrite a CPU acceptance publication.')
    report, plan = coverage(output, expected_id)
    merged = Path(plan['merged_output'])
    old_acceptance = merged/'nine_task_acceptance.json'
    old_exit = merged/'nine_task.exit'
    argv = list(plan['nine_task_argv'])
    module_index = argv.index('-m')
    module = argv[module_index+1]
    require(module in ('experiments.fresh_muon_cmcc_20261030.evaluate',
        'experiments.block_reconstruction_formal_cmcc_20261031.recovery_evaluate')
        and argv[argv.index('--component')+1] == 'summarize'
        and argv[argv.index('--output')+1] == str(old_acceptance),
        'Use only the two actual completed summarizer dispatches with the missing exit propagation.')
    wrapper = REPO/(module.replace('.', '/')+'.py')
    wrapper_sha = sha(wrapper)
    require(plan['pins'][str(wrapper)] == wrapper_sha
        and wrapper.read_text().rstrip().endswith("if __name__ == '__main__':\n    main()"),
        'The pinned original caller must actually discard its main return value.')
    destination.mkdir(parents=True)
    acceptance, exit_path, log = destination/'acceptance.json', destination/'acceptance.exit', destination/'summary.log'
    # The original main parses these exact arguments and invokes its existing
    # summarize implementation. Only the outer Python exit propagation changes.
    argv[module_index:module_index+2] = ['-c', 'from '+module+' import main; raise SystemExit(main())']
    argv[argv.index('--output')+1] = str(acceptance)
    started = time.time()
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, cwd=REPO, env={**os.environ, **original.environment('')},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        publication = dict(argv=argv, cwd=str(REPO), environment=original.environment(''),
            pid=process.pid, start_ticks=original.proc_start(process.pid), started_unix=started,
            original_acceptance=dict(path=str(old_acceptance), sha256=sha(old_acceptance)),
            original_exit=dict(path=str(old_exit), value=int(old_exit.read_text().strip())),
            original_wrapper=dict(path=str(wrapper), sha256=wrapper_sha), status='running')
        write(destination/'publication.json', publication)
        code = process.wait()
    exit_path.write_text(str(code)+'\n', encoding='utf-8')
    publication.update(status='exited', exit_code=code, finished_unix=time.time())
    write(destination/'publication.json', publication)
    require(code in (0, 1), 'CPU summarization failed; do not publish an incomplete comparison.')
    require(read(acceptance) == read(old_acceptance),
        'CPU republication must reproduce every original metric/provenance field, without changing inference.')
    result = original.validate_acceptance(acceptance, exit_path, expected_id)
    result.update(evaluation_report=str(Path(output)/'evaluation.json'),
        evaluation_seconds=report['finished_unix']-report['started_unix'],
        original_predictions_plan=report['longbench_plan'],
        acceptance_publication=str(destination/'publication.json'))
    try:
        original.validate_acceptance(acceptance, exit_path, 'foreign-UUID-must-fail')
    except ValueError:
        publication['foreign_UUID_rejected'] = True
    else:
        raise ValueError('The original strict validator accepted a foreign UUID.')
    publication.update(status='strictly_verified_republication', acceptance_sha256=sha(acceptance),
        exact_original_report_reproduced=True, original_validator_used_unchanged=True)
    write(destination/'publication.json', publication)
    return result

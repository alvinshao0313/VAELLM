"""The mature canonical whole-task partition policy, with actual Rice07 CLI targets."""
from pathlib import Path
import subprocess
import time
import sparse07_bindings as b
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

LB_GROUPS = (('gov_report',), ('multi_news',), ('vcsum',),
    ('passage_retrieval_en', 'passage_retrieval_zh', 'repobench-p'),
    ('qasper', 'triviaqa', 'lcc'), ('multifieldqa_zh', 'musique', 'dureader', 'trec'),
    ('narrativeqa', '2wikimqa', 'samsum', 'passage_count'),
    ('multifieldqa_en', 'hotpotqa', 'qmsum', 'lsht'))


def gpu_list(value):
    devices = value.split(',')
    b.require(1 <= len(devices) <= 8 and len(set(devices)) == len(devices)
        and all(device in tuple(map(str, range(8))) for device in devices), 'Choose distinct physical GPUs0..7.')
    return devices


def task_groups(gpus):
    buckets = [set() for _ in gpus]
    for index, group in enumerate(LB_GROUPS):
        buckets[index % len(gpus)].update(group)
    groups = [[task for task in DATASETS if task in bucket] for bucket in buckets]
    assigned = [task for group in groups for task in group]
    b.require(len(assigned) == len(set(assigned)) == 21 and set(assigned) == set(DATASETS)
        and sum(EXPECTED_COUNTS[task] for task in assigned) == 4750, 'Canonical21/4750 must be assigned exactly once.')
    return groups


def check_resources(gpus):
    devices = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    observed = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in devices.stdout.splitlines()}
    b.require(all(gpu in observed for gpu in gpus), 'An assigned GPU is absent.')
    selected = {gpu: observed[gpu] for gpu in gpus}
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    blockers = [line for line in active.stdout.splitlines() if line.split(',')[0].strip() in selected.values()]
    b.require(not blockers, 'Assigned GPUs are occupied; stop without waiting/retry: ' + repr(blockers))
    return dict(checked_unix=time.time(), gpu_uuids=selected, other_compute_processes=blockers)


def make_plan(config, config_path, checkpoint, output, gpus, pins):
    api = b.helpers()
    template = b.read(api.TEMPLATE)
    units = []
    for index, (gpu, tasks) in enumerate(zip(gpus, task_groups(gpus), strict=True)):
        group = output / 'longbench/groups' / f'part_{index+1:02d}'
        runtime = 86400
        argv = [str(b.PYTHON), '-B', '-u', str(b.HERE / 'predict07.py'), '--config', str(config_path),
            '--output', str(group), '--datasets', *tasks, '--max-runtime-seconds', str(runtime)]
        units.append(dict(output_dir=str(group), datasets=tasks, expected_samples=sum(EXPECTED_COUNTS[task] for task in tasks),
            native_reports=[], outer_exit=None, launch_records=[], terminal_attempts=[], gpu=gpu, argv=argv,
            native_runtime_seconds=runtime, outer_timeout_seconds=runtime+180, termination_grace_seconds=20))
    original = template['nine_task_argv']
    b.require(original[1:3] == ['-m', 'experiments.w25_eval.summarize_nine_task'], 'Wrong original BF16 reference template.')
    nine = [str(b.PYTHON), str(b.HERE / 'summary07.py'), *original[3:]]
    merged = output / 'longbench/merged'
    nine[nine.index('--checkpoint_dir')+1] = str(checkpoint)
    index = nine.index('--compressed_results')
    nine[index+1:index+3] = [str(output / 'mmlu.json'), str(output / 'seven.json')]
    nine[nine.index('--compressed_longbench_scores')+1] = str(merged / 'scores.json')
    nine[nine.index('--output')+1] = str(merged / 'nine_task_acceptance.json')
    b.require(nine[nine.index('--base_model_path')+1] == config['model']['hf_path']
        and nine[nine.index('--longbench_manifest')+1] == config['archive']['path'], 'Keep the original HF/tokenizer/BF16/archive reference.')
    return dict(schema='fixed_full21_merge_plan_v1', status='draft_awaiting_explicit_terminal_binding',
        checkpoint_id=config['model']['checkpoint_id'], expected_tasks=list(DATASETS), expected_samples=4750,
        config_path=str(config_path), config_sha256=b.binding(config_path)['sha256'], protocol=config['protocol'],
        execution_units=units, task_file_sources={task: str(Path(unit['output_dir']) / 'predictions' / (task+'.jsonl'))
            for unit in units for task in unit['datasets']}, pins={**pins, str(config_path): b.binding(config_path)['sha256']},
        merged_output=str(merged), scoring_cwd=str(b.ROOT), python_executable=str(b.PYTHON), nine_task_argv=nine,
        scoring_environment=dict(PYTHONPATH=str(b.ROOT), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', VAELLM_ALIGNED_PROFILE=b.PROFILE),
        acceptance={**template['acceptance'], 'max_linear_bpw': 2.6, 'max_lora_rank': 0,
            'source_boundary': 'Actual aligned06-derived abs005/Rice7 complete4287 bundle, full original codec ledger.'},
        nine_task_exit_semantics=template['nine_task_exit_semantics'], automatic_retry=False, automatic_resume=False,
        partial_scoring=False, boundary='Same-UUID complete full8+21/4750 through actual Rice consumers; no partial quality gate/retry.')

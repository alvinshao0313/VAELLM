"""Prepare and, only when explicitly invoked, run the unchanged complete full8."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cat_category_selection_20261029 import sources as s

TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SAMPLE_EVAL = TOPIC / 'lm_eval_sample_artifacts_20261010/cat_eval.py'
SEVEN = 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')


def module(path, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


def prepare(report):
    helper = module(TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_selection_full8_original_helpers')
    pins = {str(path): s.sha(path) for path in [*helper.source_paths(), *Path(__file__).parent.glob('*.py')]}
    endpoints = {}
    for step in (250, 500):
        row = report['endpoints'][str(step)]
        checkpoint = Path(row['checkpoint_dir'])
        meta = s.native.load_v6_meta(str(checkpoint))
        s.require(row['checkpoint_id'] == meta['checkpoint_id'] and row['fresh']['status'] == 'fresh_independent_native_exact',
                  'Only actually independent CPU-verified deployment endpoints may enter the evaluation plan.')
        output = s.DATA / f'evaluation/A{step}/full8'
        stages = []
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', SEVEN, '32')):
            argv = [str(PYTHON), '-B', '-u', str(SAMPLE_EVAL), '--checkpoint_dir', str(checkpoint),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks,
                '--num_fewshot', '0', '--lm_batch_size', batch, '--tokenizer_use_fast', 'true',
                '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')),
                '--output_json', str(output / (name + '.json'))]
            stages.append(dict(name=name, argv=argv, sample_count=14042 if name == 'mmlu' else 10700))
        endpoints[str(step)] = dict(checkpoint_dir=str(checkpoint), checkpoint_id=meta['checkpoint_id'],
            checkpoint_meta_sha256=s.sha(checkpoint / 'checkpoint_meta.json'),
            checkpoint_weight_stat=s.stat(checkpoint / meta['state_dict_file']), bit_budget=row['bit_budget'],
            output=str(output), stages=stages)
    plan = dict(schema='A_gate250_500_original_full8_plan_v1', status='CPU_prepared_GPU_not_started',
        structure_selected_before_evaluation='A control; no B selection', endpoints=endpoints,
        zero_shot=True, leaf_tasks=64, sample_count=24742, mmlu_batch=64, seven_batch=32,
        limit=None, GPU_assigned=None, source_sha256=pins,
        selection_rule='One paired unrounded eight-task mean comparison;250 only if strictly higher, exact ties500. No suffix or nine-task claim.',
        intermediate_budget_boundary='Both contain72 real dense projections; charged in full, no final2.6 eligibility claim.',
        evaluation_forward_path='Both ordinary native v6, FA2/BF16, original group8 prewarm; no proxy or altered batch.')
    s.write(s.DATA / 'evaluation_plan.json', plan)
    return plan


def execute(step, gpu):
    s.cpu_only()
    plan = s.read(s.DATA / 'evaluation_plan.json')
    s.require(gpu in tuple(map(str, range(8))), 'Choose one explicitly assigned physical GPU.')
    active = subprocess.check_output(['nvidia-smi', '--id=' + gpu, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    s.require(not active.strip(), 'Assigned GPU is occupied; no wait, retry or alternate GPU: ' + active.strip())
    for path, expected in plan['source_sha256'].items():
        s.require(s.sha(path) == expected, 'Evaluation source binding changed: ' + path)
    row = plan['endpoints'][str(step)]
    checkpoint, output = Path(row['checkpoint_dir']), Path(row['output'])
    meta = s.native.load_v6_meta(str(checkpoint))
    s.require(s.sha(checkpoint / 'checkpoint_meta.json') == row['checkpoint_meta_sha256']
              and s.stat(checkpoint / meta['state_dict_file']) == row['checkpoint_weight_stat'], 'Bound endpoint changed.')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running_original_full8', pid=os.getpid(), start_unix=time.time(), gpu=gpu,
                  checkpoint_id=meta['checkpoint_id'], checkpoint_dir=str(checkpoint), stages=[], source_plan=s.binding(s.DATA / 'evaluation_plan.json'))
    from experiments.qwen3_w25.reproduce_from_bf16 import environment
    env = environment(gpu)
    try:
        for stage in row['stages']:
            started = time.time()
            with (output / (stage['name'] + '.log')).open('x') as log:
                child = subprocess.Popen(stage['argv'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                entry = dict(**stage, pid=child.pid, start_unix=started)
                record['stages'].append(entry)
                s.write(output / 'evaluation.json', record)
                result = child.wait()
            entry.update(exit_code=result, finish_unix=time.time())
            s.write(output / 'evaluation.json', record)
            s.require(result == 0, 'Full8 stage failed: ' + stage['name'])
        api = module(ROOT / 'experiments/night_sequence_20261026/recovery_eval.py', '_selection_original_full8_summary')
        summary, _, _ = api.full8_summary(checkpoint, output)
        leaf_count = 0
        for stage in row['stages']:
            lm = s.read(output / (stage['name'] + '.json'))['evals']['lm_eval']
            artifacts = lm['sample_artifacts']
            s.require(artifacts['total_count'] == stage['sample_count'], 'Incomplete full8 raw examples.')
            leaf_count += len(artifacts['tasks'])
            for task, item in artifacts['tasks'].items():
                artifact = Path(item['path']).resolve()
                s.require(artifact.is_relative_to(output) and s.sha(artifact) == item['sha256']
                          and item['count'] == lm['n_samples'][task]['effective'], 'Raw full8 artifact mismatch.')
        s.require(leaf_count == 64 and summary['sample_count'] == 24742, 'Require complete64 leaves/24742 examples.')
        record.update(status='complete_original_full8', full8_mean_percent=summary['mean_percent'],
                      leaf_tasks=leaf_count, sample_count=24742, exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['finish_unix'] = time.time()
        s.write(output / 'evaluation.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=(250, 500), type=int, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    print(execute(args.step, args.gpu), flush=True)

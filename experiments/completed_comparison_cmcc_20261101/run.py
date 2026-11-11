"""One CPU-only completion of the actual C0/E/BR/BR->E four-endpoint comparison."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import audit as a
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, TOPIC


def delta(left, right):
    return dict(task_scores={task:left['task_scores'][task]-right['task_scores'][task]
        for task in a.original.TASKS}, nine_task_score=left['nine_task_score']-right['nine_task_score'])


def run(output):
    a.original.cpu_environment()
    output = output.resolve()
    a.require(output.is_relative_to(TOPIC) and not output.exists(), 'Choose a new isolated CPU summary directory.')
    output.mkdir(parents=True)
    started = time.time()
    state = dict(status='running', pid=os.getpid(), start_ticks=a.original.proc_start(os.getpid()),
        argv=[sys.executable, *sys.argv], started_unix=started, GPU_used=False, training_or_inference_repeated=False)
    preserved = {}
    try:
        sequence = TOPIC/'block_reconstruction_sequence_cmcc_20261031/single_C0_v3'
        direct_eval = TOPIC/'fresh_muon_cmcc_20261030/C0_H96_Muon32_gpu5_20261031'
        direct_model = ARTIFACTS/'fresh_muon_cmcc_20261030/C0_H96_Muon32/recovery'
        br_model = ARTIFACTS/'block_reconstruction_sequence_cmcc_20261031'
        c0 = a.read(sequence/'C0_source.json')['source_binding']
        br = a.read(sequence/'BR_source.json')['source_binding']
        direct = a.read(direct_model/'prepared.json')
        restored = a.read(br_model/'BR_E32/prepared.json')
        a.require(direct['source_binding'] == br['source_C0_binding'] == restored['source_C0_binding'] == c0
            and restored['source_binding'] == br, 'All four endpoints must share the actual identical C0 binding.')
        c0_id = c0['checkpoint_id']
        direct_id = a.read(direct_model/'formal32/fresh_audit.json')['checkpoint_id']
        restored_id = a.read(br_model/'BR_E32/BR_E32/fresh_audit.json')['checkpoint_id']
        control = a.read(ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json')
        for path in (sequence/'run.json', sequence/'comparison.log', sequence/'comparison.exit',
            direct_eval.with_suffix('.launch.json'), direct_eval/'pair.run.json', direct_eval/'pair.log'):
            preserved[str(path)] = a.sha(path)
        endpoints = dict(C0=a.original.validate_evaluation(control['evaluation']['output'], c0_id),
            BR=a.original.validate_evaluation(sequence/'evaluation_BR', br['checkpoint_id']))
        for name, original_output, identity in (('E', direct_eval, direct_id),
            ('BR_E', sequence/'evaluation_BR_E', restored_id)):
            try:
                a.original.validate_evaluation(original_output, identity)
            except ValueError as error:
                a.require('complete, valid same-UUID nine-term' in str(error),
                    'The original failure must be the diagnosed acceptance/exit contract, not missing evidence.')
            else:
                raise ValueError('The original acceptance/exit mismatch was not reproduced.')
            endpoints[name] = a.republish(original_output, identity, output/name)
        baseline = a.read(endpoints['C0']['acceptance_report'])
        for endpoint in endpoints.values():
            acceptance = a.read(endpoint['acceptance_report'])
            a.require(acceptance['baseline_scores'] == baseline['baseline_scores']
                and acceptance['protocol'] == baseline['protocol'], 'Keep the same complete-nine BF16 protocol and baseline.')
        comparisons = {left+'_minus_'+right:delta(endpoints[left], endpoints[right])
            for left,right in (('E','C0'), ('BR','C0'), ('BR_E','C0'), ('BR_E','BR'), ('BR_E','E'))}
        result = dict(schema='actual_C0_E_BR_BR_E_complete_comparison_v1', status='complete_four_endpoints',
            source_C0=dict(checkpoint_id=c0_id, checkpoint_dir=c0['checkpoint_dir'], pipeline_report=c0['pipeline_report']),
            endpoints=endpoints, comparisons=comparisons, pending=[],
            complete_protocol=dict(full8_leaf_tasks=64, full8_samples=24742, LongBench_tasks=21, LongBench_samples=4750),
            evidence_boundary='Only the two CPU acceptance publications were repeated with explicit return-code propagation. '
                'Original full predictions, pins, checkpoints and failed controller records remain intact. '
                'Added BR cost is not an equal-cost E control.',
            completed_unix=time.time())
        a.write(output/'four_endpoints.json', result)
        a.write(output/'C0_vs_Muon32.json', dict(schema='actual_C0_vs_Muon32_pair_v1',
            status='complete_same_parent_pair', source_C0=result['source_C0'],
            control=endpoints['C0'], C0_Muon32=endpoints['E'], restored_minus_C0=comparisons['E_minus_C0'],
            actual_four_endpoint_comparison=str(output/'four_endpoints.json')))
        a.require(preserved == {path:a.sha(path) for path in preserved}, 'An original failure record changed.')
        state.update(status='complete_four_endpoints', exit_code=0, preserved_original_records=preserved,
            source_sha256={str(path):a.sha(path) for path in Path(__file__).parent.glob('*.py')},
            strict_validator=dict(path=str(Path(a.original.__file__)), sha256=a.sha(a.original.__file__)),
            old_failure_reproduced=True, actual_republication_validated=True,
            foreign_UUID_rejected=True, full_original_acceptance_reports_exact=True)
        print(json.dumps(dict(status=state['status'], output=str(output),
            nine={name:row['nine_task_score'] for name,row in endpoints.items()},
            delta_BR_E_minus_E=comparisons['BR_E_minus_E']['nine_task_score'])), flush=True)
    except BaseException:
        state.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        state['finished_unix'] = time.time()
        a.write(output/'validation.json', state)
        (output/'run.exit').write_text(str(state['exit_code'])+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()

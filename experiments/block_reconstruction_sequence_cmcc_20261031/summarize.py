"""One snapshot of same-C0 endpoints; never wait for another experiment."""
import argparse
import json
import time
from pathlib import Path

from . import bindings as b


def delta(left, right):
    return dict(task_scores={task:left['task_scores'][task]-right['task_scores'][task]
        for task in b.executor.TASKS}, nine_task_score=left['nine_task_score']-right['nine_task_score'])


def complete(path):
    report = Path(path)/'evaluation.json'
    return report.is_file() and b.read(report).get('status') in (
        'complete_full8_full21_current_goal', 'complete_full8_full21_nine')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    b.executor.cpu_environment()
    output = b.DIRECTORY/'comparison.json'
    b.require(not output.exists(), 'Never overwrite an endpoint comparison snapshot.')
    c0 = b.read(b.DIRECTORY/'C0_source.json')['source_binding']
    br = b.read(b.DIRECTORY/'BR_source.json')['source_binding']
    from experiments.block_reconstruction_formal_cmcc_20261031 import recovery_evaluate
    restored, _meta = recovery_evaluate.source(b.E/'BR_E32/final_model', b.E/'BR_E32/report.json')
    b.require(br['source_C0_binding'] == restored['source_C0_binding'] == c0,
        'BR and BR->E must have this same actual complete C0 ancestor.')
    endpoints = dict(BR=b.executor.validate_evaluation(b.BR_EVAL, br['checkpoint_id']),
        BR_E=b.executor.validate_evaluation(b.E_EVAL, restored['checkpoint_id']))
    pending = []
    control = b.read(b.CONTROL_PLAN)
    control_eval = Path(control['evaluation']['output'])
    if complete(control_eval):
        endpoints['C0'] = b.executor.validate_evaluation(control_eval, c0['checkpoint_id'])
    else:
        pending.append(dict(endpoint='C0', expected_evaluation=str(control_eval/'evaluation.json')))
    direct_root = b.events.TARGET
    direct_eval = direct_root/'evaluation'
    if complete(direct_eval):
        from experiments.fresh_muon_cmcc_20261030 import evaluate as direct
        direct_parent, _meta = direct.source(direct_root/'recovery/formal32/final_model',
            direct_root/'recovery/formal32/report.json')
        provenance = direct_parent['pipeline_provenance']
        b.require(provenance['source_binding'] == c0, 'The direct E result has a different actual C0 parent.')
        endpoints['E'] = b.executor.validate_evaluation(direct_eval, direct_parent['checkpoint_id'])
    else:
        pending.append(dict(endpoint='E', expected_evaluation=str(direct_eval/'evaluation.json')))
    comparisons = {'BR_E_minus_BR':delta(endpoints['BR_E'], endpoints['BR'])}
    for name in ('BR', 'BR_E', 'E'):
        if name in endpoints and 'C0' in endpoints:
            comparisons[name+'_minus_C0'] = delta(endpoints[name], endpoints['C0'])
    if 'E' in endpoints:
        comparisons['BR_E_minus_E'] = delta(endpoints['BR_E'], endpoints['E'])
    result = dict(schema='actual_C0_BR_E_same_parent_comparison_v1', sampled_unix=time.time(),
        status='complete_four_endpoints' if not pending else 'own_two_complete_controls_pending',
        source_C0=c0, endpoints=endpoints, comparisons=comparisons, pending=pending,
        all_available_results_preserved=True, waited_for_other_chains=False,
        boundary='Fixed single recipe; online E trajectories differ. Added BR cost is not an equal-cost comparison.')
    b.write(output, result)
    print(json.dumps(dict(status=result['status'], endpoints=list(endpoints), pending=pending)), flush=True)


if __name__ == '__main__':
    main()

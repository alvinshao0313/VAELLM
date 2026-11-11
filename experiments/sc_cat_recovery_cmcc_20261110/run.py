"""Finite approved first-class search followed by the remaining classes and replay."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from . import io
from .determinism import verify
from .first_batch import FirstBatch
from .identity import source_pins
from .remaining import RemainingBatch


def run(root, authorization=None):
    root = Path(root).resolve()
    auth_file = Path(authorization).resolve() if authorization else root/'authorization_first.json'
    proof = verify(auth_file)
    first = FirstBatch(str(auth_file), str(proof))
    try:
        first.run()
    except BaseException:
        first.processes.cancel()
        first.state['status'] = 'failed_requires_review'
        first.persist()
        raise
    require(first.state['status'] == 'first_class_complete_awaiting_second_budget', 'First class did not close.')
    first_receipt = dict(status='first_class_verified', final_chain_verified=False,
        selected=first.state['categories'][first.category]['selected'], boundary=first.state['boundary'],
        short_screen=first.state['short_screen'], long_categories=first.state['long_categories'],
        new_actual_GPUh=first.budget.value['total_actual_GPU_hours'],
        history=first.manifest['cost_history'], source_pins=source_pins())
    io.write(root/'first_class_acceptance.json', first_receipt)
    print(dict(event='first_class_complete', receipt=str(root/'first_class_acceptance.json'),
               selected=first_receipt['selected'], cost=first_receipt['new_actual_GPUh']), flush=True)
    original = io.read(io.bound_path(first.manifest['prior_second_authorization']))
    previous = first.budget.value
    auth = deepcopy(original)
    remaining_limit = original['new_budget_GPUh'] - first.manifest['cost_history']['previous_second_GPUh']
    require(remaining_limit > 0, 'No second-batch budget remains.')
    # The prior first-class and interrupted second-class charges remain in the
    # historical ledger. This run's first-class fees are carried by Budget once.
    auth.update(run_id=first.manifest['run_id'], status='approved_deterministic_remaining',
        prior_actual_GPUh=previous['total_actual_GPU_hours'], new_budget_GPUh=remaining_limit,
        total_GPU_hours=previous['total_actual_GPU_hours'] + remaining_limit,
        first_boundary=first.state['categories'][first.category]['boundary'],
        inherited_calibration=first.state['short_screen'], inherited_long_categories=first.state['long_categories'],
        authorization_reference=first.auth['authorization_reference'], numerical_boundary=first.manifest['numerical_policy'])
    component_names=set(original['budget_by_component']) | set(previous['actual_seconds_by_component'])
    auth['budget_by_component']={k: auth['total_GPU_hours'] for k in component_names}
    auth['component_limit_meaning']='Individual ceilings; the shared total is the binding aggregate cap, never their sum.'
    counts=previous['counter_reserved']
    fresh_500=142
    auth.update(maximum_raws=counts['raws']+12, maximum_trajectories=counts['trajectories']+fresh_500+2,
        maximum_KD_updates=counts['KD_updates']+fresh_500*500+2000, maximum_raw_full8=counts['raw_full8']+12,
        maximum_node_full8=counts['node_full8']+fresh_500*4+16)
    manifest=deepcopy(first.manifest)
    manifest.update(status='deterministic_remaining_authorized',
        remaining_cpu=first.manifest['remaining_cpu'],
        first_acceptance=io.binding(root/'first_class_acceptance.json'),
        previous_budget=io.binding(first.formal_root/'budget.json'))
    io.write(root/'manifest_remaining.json',manifest)
    auth['manifest']=io.binding(root/'manifest_remaining.json')
    io.write(root/'authorization_remaining.json',auth)
    remaining=RemainingBatch(str(root/'authorization_remaining.json'))
    try:
        remaining.run()
    except BaseException:
        remaining.processes.cancel()
        remaining.state['status']='failed_requires_review'
        remaining.persist()
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--authorization',help='Reviewed authorization revision; the original receipt remains immutable.')
    args=parser.parse_args()
    run(args.root,args.authorization)

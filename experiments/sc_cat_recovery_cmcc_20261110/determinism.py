"""Validate the changed FA2 path; reuse the already verified native state mechanisms."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from . import io, EXECUTION
from .identity import source_pins
from .processes import Budget, Processes, check_resources
from .settings import RecoverySettings


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def verify(authorization):
    auth = io.read(authorization)
    manifest = io.read(io.bound_path(auth['manifest']))
    handoff = io.read(io.bound_path(manifest['selected_settings']))
    root = Path(manifest['output_root']) / auth.get('technical_directory', 'technical')
    check_resources([4, 5, 6, 7])
    previous_windows = []
    previous_receipts = []
    if auth.get('prior_technical_budget'):
        prior = io.read(io.bound_path(auth['prior_technical_budget']))
        require(not any(t['status'] == 'running' for t in prior['tasks'].values()), 'Prior technical work is still active.')
        previous_windows = list(prior['technical_windows']) + [dict(started_epoch=t['started_epoch'],
            ended_epoch=t['ended_epoch'], gpu=t['gpu'], task=name, status=t['status'])
            for name, t in prior['tasks'].items() if t['status'] in ('completed', 'failed')]
        require(abs(sum(w['ended_epoch']-w['started_epoch'] for w in previous_windows)/3600
                    - prior['total_actual_GPU_hours']) < 1e-9, 'Carry every interrupted technical charge once.')
        previous_receipts = [auth['prior_technical_budget']]
    budget = Budget(root/'budget.json', auth, previous_windows, previous_receipts)
    processes = Processes(root, budget, source_pins())
    candidate = handoff['categories'][handoff['frozen_order'][0]]['top2'][0]
    common = dict(manifest=auth['manifest'], authorization=io.binding(authorization),
        source_manifest=handoff['source_manifest'], inventory_receipt=manifest['inventory'],
        data_plan_path=manifest['data_plan'], category_id=handoff['frozen_order'][0],
        phase='technical_validation', source_candidate_id=candidate, execution_policy=auth['execution_policy'])
    budget.reserve({name: dict(component='technical', seconds=seconds) for name, seconds in
                    [('raw', 1800), ('repeat_A', 1200), ('repeat_B', 1200), ('export', 900)]}, counts={})
    try:
        raw_ref = processes.run('raw', 'worker', dict(common, operation='first_raw', output=str(root/'raw')),
                                output_file=root/'raw/live_raw.json')
        raw = io.read(io.bound_path(raw_ref))
        settings = RecoverySettings(groups='DPL', lora_rank=48, lora_alpha=192,
                                    bound_optimizer='muon', lora_optimizer='muon')
        shared = dict(common, live_raw=raw_ref, settings=asdict(settings), recipe_id=digest(raw['recipe']))
        def one(name):
            return processes.run(name, 'worker', dict(shared, operation='train', trajectory_id=name,
                technical_stop_after=2, technical_peak_proof=True, output=str(root/name)),
                output_file=root/name/'training_result.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(one, name) for name in ('repeat_A', 'repeat_B')]
            refs = [future.result() for future in pending]
        reports = [io.read(io.bound_path(ref)) for ref in refs]
        require(all(r['actual']['actual_steps'] == 2 and r['peak_proof']['input_shape'] == [4, 1024]
                    and r['peak_proof']['finite_gradients'] for r in reports), 'Actual repeated peak-shape updates failed.')
        for report in reports:
            groups = {g['group_name']: g for g in report['optimizer']['groups']}
            require({k for k, g in groups.items() if g['kind'] == 'muon'} == {'decoder', 'lora', 'q', 'scale'}
                    and all(groups[k]['muon_kappa'] > 0 for k in ('decoder', 'lora', 'q', 'scale')),
                    'Four independent Muon calibrations missing.')
        nodes = [Path(r['nodes'][-1]['directory']) for r in reports]
        compared = {}
        for name in ('training_model_state.pt', 'optimizer.pt', 'scheduler.pt'):
            payloads = [torch.load(node/name, map_location='cpu', weights_only=True) for node in nodes]
            compared[name] = exact(*payloads)
            require(compared[name], 'Deterministic independent updates differ: '+name)
        exported = processes.run('export', 'worker', dict(shared, operation='export', trajectory_id='repeat_A',
            node_directory=io.binding(nodes[0]/'checkpoint_meta.json'), output=str(root/'export'),
            destination=str(root/'deployment')), output_file=root/'export/deployment_report.json')
        export = io.read(io.bound_path(exported))
        require(export['fresh_forward_evidence']['state_exact'] and export['fresh_forward_evidence']['independent_process'],
                'Changed numerical path failed native export/fresh.')
        prior = io.read(io.bound_path(manifest['reused_GPU_validation']))
        require(prior['status'] == 'actual_GPU_verified' and all(prior['checks'].values()), 'Incomplete reusable native GPU proof.')
        # These mechanisms and their training mathematics were copied unchanged.
        # Only FA2 backward binding is new; its two independent updates are checked above.
        checks = dict(prior['checks'])
        windows = previous_windows + [dict(started_epoch=t['started_epoch'], ended_epoch=t['ended_epoch'], gpu=t['gpu'], task=name)
                   for name, t in budget.value['tasks'].items()]
        proof = dict(status='actual_GPU_verified', exit_code=0, manifest=auth['manifest'], source_pins=source_pins(),
            checks=checks, receipts=[manifest['reused_GPU_validation'], manifest['reused_kappa_validation'], raw_ref, *refs, exported],
            reused_checks=['P_only_frozen_decoder_backward', 'native250_resume_to500_and_data_RNG', 'zero_raw_native_export'],
            new_checks=dict(deterministic_consumer=EXECUTION, independent_two_update_exact=compared,
                            largest_DPL_four_Muon_groups_peak_and_export=True),
            ownership_windows=windows, actual_GPUh=budget.value['total_actual_GPU_hours'],
            formal_trajectory_eligible=False, quality_verified=False)
        io.write(root/'validation.json', proof)
        print(dict(event='deterministic_GPU_validation_passed', comparisons=compared, actual_GPUh=proof['actual_GPUh']), flush=True)
        return root/'validation.json'
    except BaseException:
        processes.cancel()
        raise

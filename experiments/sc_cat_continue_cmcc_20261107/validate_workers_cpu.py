"""Validate real input/export bindings and reject unapproved worker requests on CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
import types

import torch

from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from . import bindings as io
from . import workers as w


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError('The invalid/unapproved real request was accepted.')


def draft_plan(run, output):
    cost = io.read(run / 'audit/continuation_cost_20261107.json')['proposal']
    old = run / 'live_k_o_20261106/formal'
    manifest = io.read(run / 'manifest.json')
    limits = cost['remaining_search']
    value = dict(schema='sc_cat_continuation_authorization_v1', confirmed=False, user_approved=False,
        phase='formal', authorization_reference='', categories=list(io.CATEGORIES), frozen_order=list(io.ORDER),
        train_gpus=[0, 1], readout_gpus=[2, 3, 4, 5], gpus=list(range(6)), protected_gpus=[6, 7],
        stages=['core', 'terminal_longbench', 'replay'], reference_mode='reuse_only', output_root=str(output),
        total_GPU_hours=208.5, budget_by_component=cost['controller_component_view'].copy(),
        source_manifest=io.binding(run / 'manifest.json'), selected_settings=io.binding(run / 'selected_settings.json'),
        protocol=manifest['protocol'], execution_protocol=manifest['execution_protocol'],
        prefix={key: io.binding(old / filename) for key, filename in dict(after_o='after_o.json',
            frozen_recipe='frozen_recipe.json', quotas='quotas.json', controller_state='controller_state.json',
            controller_exit='controller.exit.json').items()},
        task_seconds=dict(fit_by_category=dict.fromkeys(io.ORDER, 4500),
            train500_by_category=limits['per_class500_train_cap_seconds'],
            train1000_by_category=limits['per_class1000_train_cap_seconds'], raw_export=900,
            node_export=450, replay_export=900, full8=900, longbench=43200, longbench_shard=21600, replay_train_startup_seconds=900,
            replay_seconds_per_update_by_category=dict(k_proj=4, o_proj=6,
                **cost['fixed_recipe_BF16_replay']['remaining_per_update_seconds_proposed'])))
    value['budget_by_component'].pop('total')
    value.update({'maximum_' + key: number for key, number in {**io.CORE_COUNTS, **io.REPLAY_COUNTS}.items()})
    return io.validate_plan(value)


def run_checks(run, directory):
    require = io.require
    checks = []
    plan = draft_plan(run, directory / 'unapproved_output')
    plan_path = directory / 'explicitly_unapproved_plan.json'
    io.write(plan_path, plan)
    rejects(io.read_authorization, io.binding(plan_path))
    rejects(io.read_authorization, run / 'live_k_o_20261106/authorization.json')
    checks.append('Actual old authorization and explicitly-unapproved structural draft cannot grant continuation.')
    manifest = io.read(run / 'manifest.json')
    common = dict(stage='core', phase='formal', source_manifest=plan['source_manifest'],
        authorization=io.binding(plan_path), inventory_receipt=manifest['inventory'], data_plan_path=manifest['data_plan'],
        gpu=0, output=str(directory / 'unapproved_output' / 'fit'), deadline_epoch=time.time() + 100)
    w._source_fields(common, plan)
    bad = dict(common, data_plan_path=manifest['inventory'])
    rejects(w._source_fields, bad, plan)
    rejects(w._source_fields, dict(common, destination=str(directory / 'outside_approved_root')), plan)
    w._category(common, 'gate_proj')
    rejects(w._category, common, 'k_proj')
    w._category(dict(common, stage='replay'), 'k_proj')
    checks.append('Frozen actual inventory/DataPlan and core versus replay categories remain distinct.')
    for category in io.CATEGORIES:
        w._deadline(dict(common, deadline_epoch=time.time() + 60), plan, 'train', category=category,
            settings=RecoverySettings())
        rejects(w._deadline, dict(common, deadline_epoch=time.time() + plan['task_seconds']['train500_by_category'][category] + 60),
            plan, 'train', category=category, settings=RecoverySettings())
    checks.append('All five actual category500 windows accept remaining time and reject enlarged deadlines.')
    before = (w.old_fit.io, w.old_recovery.validate_request, w.old_recovery._finish_export,
        w.old_node.validate_request, w.old_node._finish_export)
    proxy = types.SimpleNamespace(**dict(vars(w.live_io), validate_execution=w._validate_fit))
    fitted = w._function(w.old_fit.execute, io=proxy)
    exported = w._function(w.old_recovery.export_zero, validate_request=w._validate_recovery,
        package_pins=w._package_pins, _finish_export=w.finish_export)
    node = w._function(w.old_node.export, validate_request=w._validate_recovery,
        package_pins=w._package_pins, _finish_export=w.finish_export)
    require(fitted.__globals__['io'] is proxy and exported.__globals__['_finish_export'] is w.finish_export
        and node.__globals__['validate_request'] is w._validate_recovery
        and fitted.__kwdefaults__ == w.old_fit.execute.__kwdefaults__, 'Private real-worker adapters lost their actual validators or keyword defaults.')
    require(before == (w.old_fit.io, w.old_recovery.validate_request, w.old_recovery._finish_export,
        w.old_node.validate_request, w.old_node._finish_export), 'Private construction mutated frozen original module bindings.')
    checks.append('Actual fitter, native raw0 and published-node functions privately bind new guards without modifying originals.')
    state = io.read(run / 'live_k_o_20261106/formal/controller_state.json')
    old_raw_ref = next(iter(state['categories']['o_proj']['raws'].values()))['live_raw']
    old_raw = io.read(io.bound_path(old_raw_ref))
    rejects(w._raw_provenance, common, plan, old_raw)
    checks.append('The actual previous-batch raw cannot be relabeled into the new approval/replay stage.')
    budget = io.read(run / 'live_k_o_20261106/formal/budget.json')
    found = {}
    for task, record in budget['tasks'].items():
        if task.endswith('/raw_export') or task.endswith('/node_125/export'):
            previous = io.read(io.bound_path(record['request']))
            path = Path(previous['output']) / 'deployment_export.json'
            if path.is_file():
                found.setdefault('raw_export' if task.endswith('/raw_export') else 'node_export', (previous, io.read(path)))
    require(set(found) == {'raw_export', 'node_export'}, 'Both actual completed export source types are needed for fresh binding validation.')
    exports = []
    for kind, (previous, report) in found.items():
        request = dict(io.read(io.bound_path(previous['training_request']))) if previous.get('training_request') else dict(previous)
        request.update(previous)
        request.update(authorization=io.binding(plan_path), stage='replay')
        fresh = w.make_fresh_request(request, report)
        actual, derived_kind = w._fresh_source(fresh)
        require(derived_kind == kind and actual['uuid'] == report['uuid']
            and fresh['stage'] == 'replay' and fresh['authorization'] == request['authorization']
            and fresh['deadline_epoch'] == previous['deadline_epoch'], 'Fresh construction changed the real source, stage, authorization or deadline.')
        rejects(w._fresh_source, dict(fresh, expected_uuid='explicit_wrong_UUID'))
        rejects(w._fresh_source, dict(fresh, phase='technical_validation'))
        exports.append(dict(kind=kind, actual_source=fresh['source_export_report'], checkpoint_id=actual['uuid']))
    checks.append('Actual raw0/node exports derive900/450 fresh windows and preserve UUID/digest/stage/deadline; wrong artifact/phase rejected.')
    w._deadline(dict(common, stage='replay', deadline_epoch=time.time() + 800), plan, 'replay_export')
    rejects(w._deadline, dict(common, deadline_epoch=time.time() + 800), plan, 'node_export')
    checks.append('Replay export/fresh keeps its declared900second window while core positive-node export stays450.')
    rejects(w.execute_fit, dict(common, category_id='gate_proj'), request_path=plan_path)
    require(not Path(common['output']).exists() and not torch.cuda.is_initialized(), 'An unapproved worker constructed output or initialized CUDA.')
    checks.append('Real worker invocation rejects the unapproved draft before model loading, writes or CUDA.')
    return dict(status='CPU_real_worker_bindings_and_unapproved_execution_guards_verified', exit_code=0,
        cuda_initialized=False, checks=checks, actual_exports=exports,
        frozen_sources=io.base.frozen_sources(manifest), implementation_pins={str(p.resolve()): io.binding(p)['sha256']
            for p in (Path(__file__), Path(w.__file__))},
        limitation='No approved new authorization is fabricated; no model/Trainer/GPU or fresh inference executes. Positive full worker execution awaits actual user approval and the authorized real path.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'This is a CUDA-hidden CPU interface check.')
    began = time.monotonic()
    output = Path(args.output)
    with tempfile.TemporaryDirectory(prefix='workers_cpu_', dir=output.parent) as directory:
        report = run_checks(Path(args.run_root).resolve(), Path(directory))
    report['elapsed_seconds'] = time.monotonic() - began
    io.write(output, report)


if __name__ == '__main__':
    main()

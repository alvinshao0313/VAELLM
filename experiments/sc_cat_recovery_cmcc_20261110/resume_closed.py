"""Verify a completed category search and resume after its record-writing failure."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import difflib
import json
import os
from pathlib import Path
import tempfile

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from . import io
from .boundary import StrictMainChain, complete_boundary_reference
from .hpo_recovery import read_trajectory
from .identity import source_pins
from .processes import Budget
from .remaining import selected_recipe_record
from .selection import SelectedCategory, select_category, long_check
from .settings import RecoverySettings, ROUNDS


def validate_continuation(record, manifest, auth, state):
    previous_auth=io.read(io.bound_path(record['previous_authorization']))
    previous_manifest=io.read(io.bound_path(previous_auth['manifest']))
    require(state==io.read(io.bound_path(record['ready_state'])), 'The reviewed completed prefix changed.')
    for key in ('run_id','total_GPU_hours','budget_by_component','gpus','categories','authorization_reference'):
        require(auth[key]==previous_auth[key], 'Continuation changed approved scope: '+key)
    require({k:v for k,v in auth.items() if k.startswith('maximum_')}==
            {k:v for k,v in previous_auth.items() if k.startswith('maximum_')}, 'Continuation increased candidate counts.')
    for key in ('run_id','parent','data_identity','model_identity','inventory','data_plan','selected_settings','frozen_order','numerical_policy'):
        require(manifest[key]==previous_manifest[key], 'Continuation changed an experiment condition: '+key)
    require(manifest['previous_budget']==record['previous_budget'], 'The settled budget must be inherited exactly.')
    previous=io.read(io.bound_path(record['previous_budget']))
    require(all(t['status']=='completed' for t in previous['tasks'].values()), 'Only a closed search can use this continuation.')
    require([c for c in state['frozen_order'] if state['categories'].get(c,{}).get('status')=='completed']==record['completed_prefix'],
            'The reviewed category prefix is not complete.')
    require(record['completed_prefix']==state['frozen_order'][:len(record['completed_prefix'])], 'Completed categories are not a prefix.')
    require(source_pins()==manifest['recovery_source_sha256'], 'The reviewed implementation changed.')


def prepare(root):
    root=Path(root).resolve();folder=root/'record_fix_20261111'
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Run the record repair on CPU in bitvae.')
    require((root/'exit_code_reviewed').read_text().strip()=='1', 'Require the reviewed stopped controller.')
    state=io.read(folder/'previous_search.json')
    require(state==io.read(root/'search.json') and state['status']=='failed_requires_review', 'Failed state changed.')
    category=state['current_category'];block=state['categories'][category]
    require(category=='o_proj' and [r['parameter'] for r in block['rounds']]==list(ROUNDS)
            and all(r['status']=='complete' for r in block['rounds']), 'All announced O search rounds must be closed.')
    auth=io.read(root/'authorization_remaining.json');manifest=io.read(io.bound_path(auth['manifest']))
    pins=source_pins();changed={p for p in set(pins)|set(manifest['recovery_source_sha256'])
        if pins.get(p)!=manifest['recovery_source_sha256'].get(p)}
    allowed={str(Path(__file__).resolve()),str(Path(__file__).with_name('remaining.py').resolve())}
    require(changed==allowed, 'Only record serialization and the reviewed continuation may change.')
    previous=io.read(root/'remaining/budget.json')
    require(all(t['status']=='completed' for t in previous['tasks'].values()), 'The owned tasks must be settled.')
    for task in previous['tasks'].values():
        pid=task.get('pid');cmd=Path('/proc',str(pid),'cmdline')
        require(not cmd.exists() or b'sc_cat_recovery_cmcc_20261110' not in cmd.read_bytes(), 'A previous owned worker remains active.')
    rows=[read_trajectory(value['result']) for value in block['trajectories'].values()]
    require(all(row.status in ('complete','screened250') for row in rows), 'Pending nodes cannot be finalized.')
    chosen,certificate=select_category(rows)
    winner=next(row for row in rows if row.raw_id==block['default_winner_raw'])
    require(long_check(rows,winner.recipe_id,state['long_categories']) is None, 'A required long check remains incomplete.')
    # Bind every completed measurement to its actual exported node, not just the winner.
    measured=0
    for row in rows:
        for node in row.nodes:
            entry=state['nodes'][node.checkpoint_id];score=read_score(entry['score'])
            report=io.read(io.bound_path(entry['report']))
            require(score==node.score and report['export_evidence']['source_node_id']==node.checkpoint_id
                    and score.evidence['checkpoint_id']==report['uuid'], 'A recorded node or its full-eight identity changed.')
            measured+=1
    source=state['nodes'][certificate['selected_node_id']]
    score=read_score(source['score']);report=io.read(io.bound_path(source['report']))
    deployment=dict(report=source['report'],score=source['score'],
        key=(-score.mean,source['step']==0,-source['step'],source['trajectory_id']))
    boundary=complete_boundary_reference(export_report=report,inventory=io.read(io.bound_path(manifest['inventory']))['inventory'],full8_score=score)
    boundary_path=root/'remaining/boundaries'/category/'boundary.json'
    require(boundary==io.read(boundary_path), 'The reconstructed selection differs from the saved boundary.')
    meta=io.read(Path(boundary['path'])/'checkpoint_meta.json')
    require(meta['checkpoint_id']==boundary['uuid'] and (Path(boundary['path'])/meta['state_dict_file']).is_file(),
            'The selected deployment payload is missing.')
    selected=asdict(SelectedCategory(category,chosen.recipe_id,chosen.candidate_id,chosen.settings,
        certificate['selected_step'],certificate['selected_trajectory_id'],certificate['selected_node_id'],chosen.raw_id,
        recovery_seeds(manifest['model_identity'],category,manifest['search_seed']),'formal'))
    require(certificate['selected_step']>0, 'This reviewed failure concerns a positive recovery node.')
    raw=io.read(io.bound_path(block['raws'][chosen.candidate_id]['raw']))
    training=io.read(io.bound_path(block['trajectories'][chosen.trajectory_id]['training']))
    require(training['settings']==asdict(chosen.settings) and raw['round_base_uuid']==chosen.raw_id,
            'The selected optimizer or raw source changed.')
    try:
        json.dumps(deployment,allow_nan=False)
    except TypeError as exc:
        require('Fraction' in str(exc), 'Unexpected serialization failure.')
    else:
        raise AssertionError('The actual reported failure was not reproduced.')
    fixed=selected_recipe_record(selected,certificate,raw['recipe'],training['optimizer'],deployment)
    require(json.loads(json.dumps(fixed,allow_nan=False))==fixed and fixed['certificate']==certificate,
            'Record serialization changed the exact selection certificate.')
    io.write(folder/'o_selected_recipe.json',fixed)
    block.update(status='completed',selected=selected,certificate=certificate,boundary=io.binding(boundary_path),
                 selected_recipe=io.binding(folder/'o_selected_recipe.json'))
    chain=StrictMainChain(bf16_root_id=manifest['parent']['uuid'],categories=state['frozen_order'])
    prefix=[]
    for cat in state['frozen_order']:
        item=state['categories'].get(cat,{})
        if item.get('status')!='completed':break
        own=io.read(io.bound_path(item['boundary']));value=deepcopy(item['selected'])
        value['settings']=RecoverySettings(**value['settings']);pick=SelectedCategory(**value)
        chain.commit(pick,live_parent_id=chain.live_parent_id,boundary_reference=own,selection_certificate=item['certificate'],
            actual_round_base_id=pick.source_round_base_id,restored_node_id=pick.source_node_id,runtime_identity=own['runtime_identity'])
        prefix.append(cat)
    require(prefix==['k_proj','o_proj'], 'Restore exactly the verified K/O prefix.')
    with tempfile.TemporaryDirectory(prefix='budget_check_',dir=folder) as temporary:
        budget=Budget(Path(temporary)/'budget.json',auth,previous['technical_windows'],[io.binding(root/'remaining/budget.json')],
            previous=previous,retry_tasks=(),new_batch=True)
        require(budget.value['total_actual_GPU_hours']==previous['total_actual_GPU_hours']
                and budget.value['counter_reserved']==previous['counter_reserved'] and budget.value['tasks']==previous['tasks'],
                'Continuation changed settled fees, count reservations or completed tasks.')
    state.update(status='remaining_classes_ready',boundary=boundary,chain=chain.boundaries,
                 approved_categories=state['frozen_order'][len(prefix):])
    io.write(folder/'ready_search.json',state)
    checks=dict(status='passed',source_pins=pins,reused_native_CPU=manifest['remaining_cpu'],
        serialization_failure_reproduced=True,actual_selected_record_JSON_roundtrip=True,
        exact_selection_unchanged=True,verified_O_measured_nodes=measured,closed_prefix=prefix,
        full_cost_and_counters_preserved=True,GPU_executed=False,
        validation_scope='Actual completed records, selection and CPU continuation; unchanged native GPU paths reused.')
    io.write(folder/'cpu_validation.json',checks)
    continuation=dict(previous_authorization=io.binding(root/'authorization_remaining.json'),
        previous_budget=io.binding(root/'remaining/budget.json'),ready_state=io.binding(folder/'ready_search.json'),
        completed_prefix=prefix,validation=io.binding(folder/'cpu_validation.json'),
        reason='Completed O search; internal Fraction comparison cache accidentally entered the JSON artifact record.')
    io.write(folder/'continuation.json',continuation)
    revised=deepcopy(manifest);revised.update(previous_manifest=auth['manifest'],
        previous_budget=continuation['previous_budget'],remaining_cpu=io.binding(folder/'cpu_validation.json'),
        recovery_source_sha256=pins,code_identity=digest(pins),status='reviewed_record_repair_ready')
    io.write(folder/'manifest.json',revised)
    authorization=deepcopy(auth);authorization.update(manifest=io.binding(folder/'manifest.json'),
        closed_continuation=io.binding(folder/'continuation.json'),status='approved_closed_prefix_continuation')
    validate_continuation(continuation,revised,authorization,state)
    io.write(folder/'authorization.json',authorization)
    before=(folder/'remaining_before.py').read_text();after=Path(__file__).with_name('remaining.py').read_text()
    (folder/'implementation.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='remaining.py before record fix',tofile='remaining.py after record fix')))
    io.write(root/'search.json',state,replace=True)
    print(dict(status='record_repair_CPU_passed',completed_prefix=prefix,selected_step=certificate['selected_step'],
        selected_S8=100*float(score.mean),verified_nodes=measured,retained_actual_GPUh=previous['total_actual_GPU_hours'],
        remaining_GPUh=auth['total_GPU_hours']-previous['total_actual_GPU_hours'],
        authorization=str(folder/'authorization.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    prepare(parser.parse_args().root)

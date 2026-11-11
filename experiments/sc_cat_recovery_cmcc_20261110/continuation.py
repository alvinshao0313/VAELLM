"""Reviewed continuation of the interrupted first defaults; no automatic retries."""
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from . import io
from .identity import source_pins


def same_reference(a, b):
    return a['sha256']==b['sha256'] and io.bound_path(a)==io.bound_path(b)


def reuse_completed(entry, request):
    previous = io.read(io.bound_path(entry['request']))
    for key in ('operation','settings','category_id','phase','source_candidate_id','trajectory_id','recipe_id'):
        require(previous.get(key)==request.get(key), 'Reused completed task changed '+key)
    for key in ('live_raw','node_directory','deployment_report','inventory_receipt','data_plan_path'):
        if key in previous:
            require(key in request and same_reference(previous[key],request[key]), 'Reused input changed '+key)
    io.bound_path(entry['result'])
    return entry['result']


def validate_continuation(record, manifest, auth, proof, state):
    old = io.read(io.bound_path(record['manifest']))
    old_auth = io.read(io.bound_path(record['authorization']))
    require(same_reference(proof['manifest'],record['manifest']) and proof['source_pins']==old['recovery_source_sha256'],
        'Retain the original complete GPU validation and source identities.')
    for key in ('run_id','data_identity','code_identity','selected_settings','inventory','baseline','data_plan','frozen_order'):
        require(manifest[key]==old[key], 'Continuation changed the original algorithm/input '+key)
    for key in ('total_GPU_hours','budget_by_component','gpus','categories','authorization_reference'):
        require(auth[key]==old_auth[key], 'Continuation changed approved scope '+key)
    require({k:v for k,v in auth.items() if k.startswith('maximum_')}==
        {k:v for k,v in old_auth.items() if k.startswith('maximum_')}, 'Continuation increased candidate counts.')
    pins = source_pins()
    changed = {p for p in set(pins)|set(proof['source_pins']) if pins.get(p)!=proof['source_pins'].get(p)}
    package = Path(__file__).parent.resolve()
    permitted = {str(package/name) for name in ('first_batch.py','processes.py','trainer.py','continuation.py',
        'validate_continuation_cpu.py','run_continue.sh')}
    require(changed==set(record['changed_sources']) and changed.issubset(permitted)
        and pins==manifest['recovery_source_sha256'], 'Changes exceed reviewed scheduling/storage/resume repair.')
    cpu = io.read(io.bound_path(record['cpu_validation']))
    require(cpu['status']=='CPU_verified' and cpu['native375_resume_to500']
        and cpu['source_pins']==pins, 'Actual375 CPU continuation and ownership checks are required.')
    require(state==io.read(io.bound_path(record['state'])) and not state['categories'][state['current_category']]['rounds'],
        'This reviewed continuation only restores the interrupted defaults before HPO.')
    budget = io.read(io.bound_path(record['budget']))
    require(not any(t['status']=='running' for t in budget['tasks'].values()), 'Previous GPU ownership is not settled.')
    for task, entry in record['completed_tasks'].items():
        require(budget['tasks'][task]['status']=='completed', 'Cannot reuse an incomplete GPU task.')
        io.bound_path(entry['result'])
        io.bound_path(entry['request'])
    require(len(record['trajectories'])==2, 'Resume the two original defaults, not new candidates.')
    for identifier, trajectory in record['trajectories'].items():
        require(budget['tasks'][identifier+'/train']['status']=='failed', 'Only interrupted training is continued.')
        for node in trajectory['nodes']:
            directory = io.bound_path(node['metadata']).parent
            ready = io.read(directory/'node_ready.json')
            require(ready['step']==node['step'] and io.read(directory/'checkpoint_meta.json')['checkpoint_id']==node['node_id'],
                'Saved node provenance changed.')
        require([n['step'] for n in trajectory['nodes']]==[125,250,375], 'Keep every completed prefix node.')
        for filename, ref in trajectory['resume_files'].items():
            require(io.bound_path(ref).name==filename, 'Exact saved375 state file changed.')

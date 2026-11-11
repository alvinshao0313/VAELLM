"""Restore verified completed rows and the frozen interrupted LR round."""
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require,digest
from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore,NodeScore
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from . import io
from .continuation import same_reference
from .identity import source_pins
from .optimizer import calibration_context
from .selection import TrajectoryResult
from .settings import RecoverySettings


def read_trajectory(reference):
    record=io.read(io.bound_path(reference));value=dict(record)
    value['settings']=RecoverySettings(**record['settings'])
    value['raw_score']=FullEightScore(**record['raw_score'])
    value['nodes']=tuple(NodeScore(n['step'],FullEightScore(**n['score']),n['checkpoint_id']) for n in record['nodes'])
    return TrajectoryResult(**value)


def fixed_from_report(report,manifest):
    provenance=report['actual']['provenance']
    require(provenance['root_data_identity']==manifest['data_identity'] and provenance['settings']==report['settings'],
        'The source calibration must belong to the actual unchanged data and settings.')
    context=calibration_context(provenance['round_base_id'],provenance['category_id'],
        provenance['root_data_identity'],provenance['semantic_seeds'],RecoverySettings(**report['settings']))
    result={}
    for group in report['optimizer']['groups']:
        if group['kind']=='muon':
            calibration=dict(group['calibration'])
            require('context' not in calibration or calibration['context']==context,'Recorded calibration context changed.')
            calibration['context']=context
            result[group['group_name']]=calibration
    return result


def restore_search(run):
    block=run.state['categories'][run.category]
    for candidate,value in block['raws'].items():
        raw=io.read(io.bound_path(value['raw']));raw_id=raw['round_base_uuid']
        run.contexts[raw_id]=dict(raw_ref=value['raw'],raw=raw,raw_id=raw_id,candidate=candidate,
            recipe_id=digest(raw['recipe']),raw_score=read_score(value['zero_score']))
    for identifier,value in block['trajectories'].items():
        row=read_trajectory(value['result']);report=io.read(io.bound_path(value['training']))
        require(row.trajectory_id==identifier and row.status=='complete' and report['actual']['actual_steps']==row.settings.schedule_max,
            'Only complete original trajectories may bypass training.')
        run.results[identifier]=row;run.training_reports[identifier]=report
        run.fixed[identifier]=fixed_from_report(report,run.manifest)
    for source,value in run.state['nodes'].items():
        report=io.read(io.bound_path(value['report']));score=read_score(value['score'])
        require(report['export_evidence']['source_node_id']==source and score.evidence['checkpoint_id']==report['uuid'],
            'Restored evaluation must match its exact native artifact.')
        key=(-score.mean,value['step']==0,-value['step'],value['trajectory_id'])
        run.deployments[source]=dict(report=value['report'],score=value['score'],key=key)
        raw_id=report['export_evidence']['round_base_id']
        previous=run.best_exports.get(raw_id)
        if previous is None or key<previous['key']:
            run.best_exports[raw_id]=dict(source=source,key=key)
    for best in run.best_exports.values():
        report=io.read(io.bound_path(run.deployments[best['source']]['report']))
        meta=io.read(Path(report['path'])/'checkpoint_meta.json')
        require((Path(report['path'])/meta['state_dict_file']).is_file(),'The retained best artifact is missing.')


def validate_hpo_continuation(record,manifest,auth,proof,state):
    previous=io.read(io.bound_path(record['manifest']));old_auth=io.read(io.bound_path(record['authorization']))
    require(proof==io.read(io.bound_path(record['original_GPU_validation']))
        and proof['status']=='actual_GPU_verified' and all(proof['checks'].values()),'Original native GPU proof is incomplete.')
    for key in ('run_id','data_identity','code_identity','selected_settings','inventory','baseline','data_plan','frozen_order'):
        require(manifest[key]==previous[key],'HPO repair changed the algorithm/input '+key)
    for key in ('total_GPU_hours','budget_by_component','gpus','categories','authorization_reference'):
        require(auth[key]==old_auth[key],'HPO repair changed approved scope '+key)
    require({k:v for k,v in auth.items() if k.startswith('maximum_')}==
        {k:v for k,v in old_auth.items() if k.startswith('maximum_')},'HPO repair increased counts.')
    require(state==io.read(io.bound_path(record['state'])),'The frozen failed search state changed.')
    block=state['categories'][state['current_category']]
    require(len(block['trajectories'])==17 and len(state['nodes'])==70
        and block['rounds'][-1]['parameter']=='protection_lr' and block['rounds'][-1]['status']=='pending',
        'Restore the reviewed pending protection-LR round without regenerating candidates.')
    budget=io.read(io.bound_path(record['budget']))
    require(not any(t['status']=='running' for t in budget['tasks'].values()),'Prior ownership is not settled.')
    expected={c['trajectory_id']+'/train' for c in block['rounds'][-1]['candidates'] if not c['reuse']}
    require(set(record['retry_tasks'])==expected and all(budget['tasks'][k]['status']=='failed' for k in expected),
        'Retry only the two candidates that failed before a valid checkpoint.')
    pins=source_pins();changed={p for p in set(pins)|set(previous['recovery_source_sha256'])
        if pins.get(p)!=previous['recovery_source_sha256'].get(p)}
    permitted={str(Path(__file__).parent.resolve()/name) for name in ('optimizer.py','builder.py','processes.py',
        'first_batch.py','hpo_recovery.py','validate_kappa_cpu.py','run_hpo_continue.sh')}
    require(changed==set(record['changed_sources']) and changed.issubset(permitted)
        and pins==manifest['recovery_source_sha256'],'Repair changed unreviewed implementation paths.')
    cpu=io.read(io.bound_path(record['cpu_validation']))
    gpu=io.read(io.bound_path(record['GPU_validation']))
    require(cpu['status']=='CPU_verified' and cpu['source_pins']==pins and gpu['source_pins']==pins
        and gpu['status']=='actual_GPU_verified' and gpu['fixed_kappa_exact'],
        'Actual fixed-kappa CPU and GPU checks are required.')
    require(record['ownership_windows'][:len(budget['technical_windows'])]==budget['technical_windows'],
        'Original technical ownership was changed.')
    for row in block['trajectories'].values():
        read_trajectory(row['result']);io.bound_path(row['training'])

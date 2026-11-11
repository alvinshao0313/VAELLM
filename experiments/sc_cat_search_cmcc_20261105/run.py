"""Execute the approved finite offline SC-CAT graph; no live recovery or retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.cat_night_profiles_cmcc_20261104.run import wait_child
from experiments.cat_sequential_search_cmcc_20261105.run import cleanup_technical
from experiments.cmcc_runtime_20261030.sequence import check_resources
from experiments.cat_offline_execution_cmcc_20261104 import evaluate
from experiments.cat_offline_execution_cmcc_20261104.results import worker_result, attach_full8
from . import bindings as b

TERMINAL = ('completed','failed','incomplete','budget_exhausted','storage_blocked')
PENDING = ('declared_pending','running')


def score_key(score,logical,packed,identifier,*,current_id=None):
    key = b.old.selection.quality_key(score,logical,packed,identifier)
    return (*key[:3],identifier != current_id,identifier) if current_id is not None else key


def chosen(manifest,reference,settings):
    b.require(b.sha(reference['path']) == reference['sha256'], 'Selected original canonical changed.')
    result = b.checked_result(reference['path'],manifest)
    return dict(result=reference,settings=deepcopy(settings),candidate_id=result['candidate_id'])


def best_different(manifest,category):
    anchor_id = category['anchor']['candidate_id']
    values = []
    for identifier,reference in category['known'].items():
        if identifier == anchor_id:continue
        row = b.checked_result(reference['path'],manifest)
        values.append((score_key(row['full8_mean'],row['logical_bits'],row['packed_bits'],identifier),reference,identifier))
    return dict(result=min(values)[1],candidate_id=min(values)[2]) if values else None


def select_round(manifest,current,stage):
    b.require(not any(row['status'] in PENDING or row['status'] == 'failed' for row in stage['candidates']),
        'Every announced legal candidate must complete before selection.')
    values = {current['candidate_id']:(current['result'],current['settings'])}
    for item in stage['candidates']:
        if item['status'] in ('completed','reused'):
            values.setdefault(item['candidate_id'],(item['result'],item['settings']))
    ranked = []
    for identifier,(reference,settings) in values.items():
        row = b.checked_result(reference['path'],manifest)
        ranked.append((score_key(row['full8_mean'],row['logical_bits'],row['packed_bits'],identifier,current_id=current['candidate_id']),reference,settings))
    key,reference,settings = min(ranked,key=lambda row:row[0])
    return chosen(manifest,reference,settings),key


def remember(category,stage,cache):
    for row in stage['candidates']:
        if row['status'] in ('completed','reused'):
            category['known'][row['candidate_id']] = row['result']
            cache['results'][b.recipe_key(row['recipe'])] = row['result']


def distinct_count(category):
    identifiers = {category['anchor']['candidate_id']}
    identifiers.update(row['candidate_id'] for stage in category['rounds'] for row in stage['candidates']
        if row.get('candidate_id') and row['status'] not in ('uncovered_illegal','uncovered_cap','mixed_duplicate'))
    return len(identifiers)


def transpose_pair(stage):
    pair = {row['requested_value']:row['result'] for row in stage['candidates']
        if type(row['requested_value']) is bool and row['status'] in ('completed','reused')}
    b.require(set(pair) == {False,True}, 'Mixed transpose requires both complete matching uniform A/B candidates.')
    return pair[False],pair[True]


def advance(manifest,category_name,category,cache):
    if category['status'] in TERMINAL:return
    while True:
        if category['rounds'] and category['rounds'][-1]['status'] == 'running':
            stage = category['rounds'][-1]
            remember(category,stage,cache)
            if any(row['status'] == 'failed' for row in stage['candidates']):
                category['status'] = stage['status'] = 'failed';return
            if any(row['status'] in PENDING for row in stage['candidates']):return
            if stage['parameter'] == 'transpose':
                readout = stage['readout']
                if readout['status'] == 'awaiting_pair':
                    try:
                        a,bref = transpose_pair(stage)
                    except ValueError:
                        category['status'] = 'incomplete';stage['status'] = 'incomplete'
                        stage['reason'] = 'A required full uniform transpose direction was not covered; no C can be fabricated.';return
                    readout.update(status='declared_pending',A=a,B=bref)
                    return
                if readout['status'] in PENDING:return
                if readout['status'] != 'completed':
                    category['status'] = stage['status'] = 'failed';return
                if not stage.get('mixed_generated'):
                    mapping = readout['transpose']
                    settings = dict(stage['input']['settings'],transpose=mapping)
                    row = b.candidate_row(manifest,category_name,'transpose',settings,'per_layer_mixed',cache)
                    row['mixed_evidence'] = readout['evidence'];row['dedicated_mixed_slot'] = True
                    stage['candidates'].append(row);stage['mixed_generated'] = True
                    remember(category,stage,cache)
                    b.require(distinct_count(category) <= 20, 'Mixed candidate exceeds the dedicated twentieth configuration.')
                    if row['status'] in PENDING:return
                    if row['status'] == 'failed':category['status'] = stage['status'] = 'failed';return
            winner,key = select_round(manifest,stage['input'],stage)
            category['current'] = winner
            stage.update(status='completed',selected=deepcopy(winner),comparison_key=list(key),
                current_candidate_id=stage['input']['candidate_id'])
            category['next_parameter'] += 1
        if category['next_parameter'] == len(b.ORDER):
            blocked = [row for stage in category['rounds'] for row in stage['candidates']
                if row['status'] == 'uncovered_illegal' and ('impl_blocked' in row.get('reason','')
                    or row.get('capability',{}).get('status') == 'impl_blocked')]
            category['status'] = 'incomplete' if blocked else 'completed'
            category['best_different'] = best_different(manifest,category)
            category['champion'] = deepcopy(category['current'])
            category['distinct_configurations'] = distinct_count(category)
            b.require(category['distinct_configurations'] <= 20, 'A closed category exceeds twenty distinct configurations.')
            return
        parameter = b.ORDER[category['next_parameter']]
        stage = dict(parameter=parameter,status='running',input=deepcopy(category['current']),
            candidates=b.proposals(manifest,category_name,parameter,category['current']['settings'],cache))
        if parameter == 'transpose':
            stage['readout'] = dict(status='awaiting_pair',entering_transpose=deepcopy(category['current']['settings']['transpose']),
                mapping_rule='Per projection full512 output MSE; exact ties retain round-entry direction; one dedicated mixed slot')
            stage['mixed_generated'] = False
        category['rounds'].append(stage)
        b.require(distinct_count(category) <= 20, 'The declared category candidate set exceeds twenty distinct configurations.')


def candidate_run(manifest,candidate,gpu,root,proof):
    directory = root/candidate['candidate_id'];directory.mkdir()
    started = time.time()
    windows = b.phase_windows(manifest,candidate['recipe'],needs_technical=proof is None)
    deadline = started+sum(windows.values())
    state = dict(task_kind='candidate',candidate_id=candidate['candidate_id'],settings=candidate['settings'],recipe=candidate['recipe'],
        parameter=candidate['parameter'],status='running',gpu=gpu,controller_pid=os.getpid(),started_epoch=started,
        absolute_deadline_epoch=deadline,technical_evidence_reuse=proof,stages=[])
    path = directory/'candidate.json';b.write(path,state)
    try:
        b.require(candidate['parameter'] != 'multiplier' or proof is not None, 'LR candidates reuse the actual same-resource technical evidence; no extra unbudgeted test.')
        technical = None
        for phase in (('formal',) if proof else ('technical_validation','formal')):
            b.integrity(manifest);check_resources([str(gpu)])
            request = b.make_request(manifest,candidate,directory,gpu,phase,deadline,proof)
            registry = directory/(phase+'_request.json');b.write_new(registry,request)
            output = directory/phase;output.mkdir()
            command = [sys.executable,'-B','-u','-m','experiments.cat_night_profiles_cmcc_20261104.search','--request',str(registry),'--worker']
            environment = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ACCELERATE_USE_CPU='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            state['stages'].append(dict(phase=phase,request=b.bound(registry),deadline_epoch=request['absolute_deadline_epoch']));b.write(path,state)
            code = wait_child(command,environment,output,request['absolute_deadline_epoch'])
            b.require(code == 0,'Actual '+phase+' failed; no retry or successor on this category.')
            completed = b.read(output/'completed.json')
            b.require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['request_identity'] == b.digest(request), 'Actual completion differs from the frozen request.')
            if phase == 'technical_validation':
                b.old.search.batched.native._validate_model_path_evidence(completed,b.read(manifest['inventory']['path'])['inventory'],manifest['run_identity'],[candidate])
                technical = completed;proof = b.bound(output/'completed.json')
        result = worker_result(directory/'formal/completed.json',candidate['candidate_id'],request_reference=b.bound(directory/'formal_request.json'))
        b.write_new(directory/'formal/offline_result.json',result)
        if technical:
            state['technical_cleanup'] = cleanup_technical(directory,technical,result);b.write(path,state)
            for row in state['technical_cleanup']:
                target = Path(row['path']);target.unlink();b.require(not target.exists(),'Own superseded technical payload removal failed.')
            state['technical_cleanup_complete'] = True;b.write(path,state)
        b.integrity(manifest);check_resources([str(gpu)])
        now = time.time();end = min(deadline,now+windows['full8']);b.require(end > now,'Complete candidate reservation exhausted before full8.')
        auth = dict(authorization_reference=manifest['rules']['authorization_reference'],physical_GPU=gpu,
            authorized_GPU_hours=windows['full8']/3600,authorized_start_epoch=now,absolute_deadline_epoch=end)
        output = directory/'full8';output.mkdir()
        plan = evaluate.make_plan(result=result,baseline=b.load_baseline(manifest['baseline']['path']),baseline_path=manifest['baseline']['path'],output=output,authorization=auth)
        b.write_new(output/'plan.json',plan)
        state['stages'].append(dict(phase='full8',plan=b.bound(output/'plan.json'),deadline_epoch=end));b.write(path,state)
        evaluate.run(output/'plan.json')
        scored = attach_full8(result,summary_path=output/'full8_summary.json',evaluation_path=output/'evaluation.json')
        target = directory/'formal/offline_full8_result.json';b.write_new(target,scored)
        b.checked_result(target,manifest,expected_recipe=candidate['recipe'])
        state.update(status='completed',exit_code=0,result=b.bound(target),full8_mean=scored['full8_mean'])
    except Exception as error:
        state.update(status='failed',exit_code=1,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    finally:
        state['finished_epoch'] = time.time();state['owned_GPU_hours'] = (state['finished_epoch']-started)/3600;b.write(path,state)
    return state


def readout_run(manifest,category,stage,gpu,root):
    from .layerwise import readout_run as perform
    output = root/category
    started = time.time();deadline = started+manifest['rules']['readout_per_category_seconds']
    record = dict(task_kind='readout',category_id=category,status='running',gpu=gpu,started_epoch=started,absolute_deadline_epoch=deadline)
    output.mkdir(parents=True);path = output/'task.json';b.write_new(path,record)
    try:
        b.integrity(manifest);check_resources([str(gpu)])
        readout = stage['readout']
        result = perform(manifest,readout['A'],readout['B'],readout['entering_transpose'],gpu,output,deadline)
        b.require(result['status'] == 'completed' and result['exit_code'] == 0 and result['evidence'] and 'transpose' in result, 'The full paired readout did not complete.')
        record.update(result)
    except Exception as error:
        record.update(status='failed',exit_code=1,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    finally:
        record['finished_epoch'] = time.time();record['owned_GPU_hours'] = (record['finished_epoch']-started)/3600;b.write(path,record)
    return record


def freeze_order(manifest,state):
    b.require(all(c['status'] == 'completed' for c in state['categories'].values()), 'Every category must close before order is frozen.')
    baseline = b.load_baseline(manifest['baseline']['path']);entries = [];selection_ids = set();sample_ids = set()
    for category,data in state['categories'].items():
        row = b.checked_result(data['current']['result']['path'],manifest)
        b.require(row['run_identity'] == manifest['run_identity'] and row['full8_evidence']['protocol_identity'] == baseline['protocol_identity'], 'Class order requires one original BF16 and common complete protocol.')
        selection_ids.add(b.digest(row['selection_data_identity']))
        sample_ids.add(row['full8_evidence']['ordered_sample_identity_evidence']['ordered_input_identity'])
        b.require(math_isfinite(row['joint_KL']), 'Class-order exact ties require the actual common512 teacher KL.')
        entries.append(dict(category_id=category,champion_candidate_id=row['candidate_id'],full8_mean=row['full8_mean'],
            d_c=baseline['mean_percent']-row['full8_mean'],diagnostic_KL=row['joint_KL'],result=data['current']['result']))
    b.require(len(selection_ids) == len(sample_ids) == 1,'Class order candidates differ in fixed512 teacher or ordered full8 inputs.')
    entries.sort(key=lambda row:(row['d_c'],row['diagnostic_KL'],row['category_id']))
    value = dict(schema='sc_cat_closed_category_order_v1',manifest_identity=manifest['manifest_identity'],
        ordered_categories=[row['category_id'] for row in entries],evidence=entries,baseline=manifest['baseline'],
        protocol_identity=manifest['protocol_identity'],BF16_ordered_pairing_verified=False,
        rule='(signed same-BF16 full8 loss, common512 diagnostic KL only on exact loss ties, stable category ID)')
    return dict(value,order_identity=b.digest(value))


def math_isfinite(value):
    import math
    return type(value) in (float,int) and math.isfinite(value)


def retention(manifest,state):
    protected,pending = set(),set();completed = set()
    for category in state['categories'].values():
        protected.update((category['anchor']['candidate_id'],category['current']['candidate_id']))
        other = best_different(manifest,category)
        category['best_different'] = other
        if other:protected.add(other['candidate_id'])
        for stage in category['rounds']:
            for row in stage['candidates']:
                if row['status'] in PENDING:protected.add(row['candidate_id'])
                if row['status'] == 'completed':completed.add(row['candidate_id'])
            if stage['parameter'] == 'transpose' and stage['status'] == 'running':
                pending.update(row['candidate_id'] for row in stage['candidates'] if row.get('candidate_id'))
    state['payload_retention'] = dict(protected_candidate_ids=sorted(protected),pending_readout_candidate_ids=sorted(pending),
        validation_candidate_ids=[],no_future_raw_readers=sorted(completed-protected-pending),
        rule='Own completed imported raw only; all bank/metadata/evaluation evidence and every historical source retained')


def persist(manifest,state,output,*,cleanup=True):
    from .report import summarize
    retention(manifest,state)
    state['last_updated_epoch'] = time.time();b.write(output/'search.json',state)
    if cleanup:
        from .evidence import cleanup_completed,execute_release_plan
        proposals = cleanup_completed(manifest,state)
        if proposals:
            state.setdefault('raw_releases',[]).extend(execute_release_plan(manifest,proposals,manifest['rules']['authorization_reference']))
            b.write(output/'search.json',state)
    summarize(manifest,state,output/'selected_settings.json',update_markdown=True)


def resume_state(manifest,state,root):
    b.require(state['manifest'] == b.bound(Path(manifest['output_root'])/'manifest.json') and state['status'] not in ('completed','completed_with_failures'), 'Resume this exact unfinished batch, never a finished search.')
    b.require(not Path('/proc',str(state['controller_pid'])).exists(),'The original new controller is still present; no duplicate/takeover.')
    for category in state['categories'].values():
        for stage in category['rounds']:
            rows = [(row,root/row['candidate_id']/'candidate.json','candidate') for row in stage['candidates'] if row['status'] == 'running']
            if stage.get('readout',{}).get('status') == 'running':
                rows.append((stage['readout'],root.parent/'readouts'/stage['category_id']/'task.json','readout'))
            for row,path,kind in rows:
                record = b.read(path)
                b.require(record['status'] in ('completed','failed'),'Existing task has incomplete dependencies; no automatic retry or worker takeover.')
                row.update({key:value for key,value in record.items() if key not in ('settings','recipe')})
                state[kind+'_GPU_hours'] += record['owned_GPU_hours']
                if kind == 'candidate' and record['status'] == 'completed':category['known'][row['candidate_id']] = row['result']


def wait_for_prior(manifest,output):
    from experiments.cmcc_runtime_20261030.sequence import proc_start,wait_identity
    authorization = b.read(manifest['authorization']['path'])
    pid = authorization['prior_controller']['pid'];start = proc_start(pid)
    receipt = dict(schema='sc_cat_one_prior_dependency_wait_v1',prior_pid=pid,prior_start_ticks=start,
        manifest_identity=manifest['manifest_identity'],started_epoch=time.time(),GPU_hours=0.)
    path = output/'prior_wait.json'
    if start is not None:
        argv = Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\0')
        b.require('experiments.cat_sequential_search_cmcc_20261105.run' in argv and '--component' in argv
            and argv[argv.index('--component')+1] == 'run' and '--output' in argv
            and Path(argv[argv.index('--output')+1]).resolve() == Path(manifest['old_output_root']).resolve(),
            'The declared prior PID is not the original old search controller; no blind wait or takeover.')
        receipt['prior_argv'] = argv;b.write(path,receipt)
        receipt['kernel_wait'] = wait_identity(pid,start)
    else:receipt['kernel_wait'] = dict(state='already_absent')
    receipt['finished_epoch'] = time.time();b.write(path,receipt)
    return b.bound(path)


def run(manifest_path,*,resume=False,drain_file=None,resource_receipt=None,wait_prior=False):
    manifest = b.read(manifest_path);b.integrity(manifest)
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'The caller must activate bitvae and hide CUDA from the CPU controller.')
    output = Path(manifest['output_root']);root = output/'candidates';path = output/'search.json'
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if wait_prior:wait_for_prior(manifest,output)
        resource = b.validate_resource_release(manifest,resource_receipt)
        cache = deepcopy(b.freeze_source_cache(manifest))
        # Import the final original batch once; a prepare-time running snapshot is never the cache boundary.
        for reference in cache['results'].values():b.checked_result(reference['path'],manifest)
        if resume:
            state = b.read(path);resume_state(manifest,state,root)
            for category in state['categories'].values():
                for reference in category['known'].values():
                    row = b.checked_result(reference['path'],manifest);cache['results'][b.recipe_key(row['recipe'])] = reference
        else:
            b.require(not path.exists(),'An existing new search needs explicit --resume; no overwrite.')
            root.mkdir();(output/'readouts').mkdir()
            state = dict(schema='sc_cat_offline_search_state_v1',status='running',manifest=b.bound(manifest_path),
                controller_pid=os.getpid(),started_epoch=time.time(),candidate_GPU_hours=0.,readout_GPU_hours=0.,categories={},
                old_batch_terminal_verified=resource['old_search'],resource_release=resource,source_cache=b.bound(output/'source_cache.json'))
            for name,anchor in manifest['anchors'].items():
                state['categories'][name] = dict(status='running',next_parameter=0,anchor=deepcopy(anchor),current=deepcopy(anchor),
                    known={anchor['candidate_id']:anchor['result']},rounds=[],best_different=None)
        state.update(status='running',controller_pid=os.getpid());running = {};free = list(manifest['rules']['gpus'])
        try:
            with ThreadPoolExecutor(max_workers=len(free)) as pool:
                while True:
                    b.integrity(manifest)
                    for name,category in state['categories'].items():
                        try:
                            advance(manifest,name,category,cache)
                            if category['rounds']:category['rounds'][-1]['category_id'] = name
                        except Exception as error:
                            category.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
                    draining = drain_file is not None and Path(drain_file).exists()
                    if not draining:
                        groups = []
                        for name,category in state['categories'].items():
                            if category['status'] != 'running' or not category['rounds']:continue
                            stage = category['rounds'][-1]
                            tasks = [('candidate',row) for row in stage['candidates'] if row['status'] == 'declared_pending']
                            if stage.get('readout',{}).get('status') == 'declared_pending':tasks.append(('readout',stage['readout']))
                            groups.append((name,category,stage,tasks))
                        pending = [(name,category,stage,*tasks[index]) for index in range(max((len(g[3]) for g in groups),default=0))
                            for name,category,stage,tasks in groups if index < len(tasks)]
                        for name,category,stage,kind,row in pending:
                            if not free:break
                            proof = b.proof_for(manifest,row,root) if kind == 'candidate' else None
                            reserve = sum(b.phase_windows(manifest,row['recipe'],needs_technical=proof is None).values())/3600 if kind == 'candidate' else .5
                            used = state[kind+'_GPU_hours']+sum(item['reserve'] for item in running.values() if item['kind'] == kind)
                            if used+reserve > manifest['rules'][kind+'_GPU_hours']:
                                category['status'] = 'budget_exhausted';row.update(status='budget_deferred',reason='Complete dependency reservation exceeds the approved '+kind+' budget');continue
                            b.require(distinct_count(category) <= 20,'The live candidate set exceeds twenty; no transferred mixed quota.')
                            b.require_space(manifest,len(running)+1)
                            gpu = free.pop(0);row.update(status='running',gpu=gpu)
                            future = pool.submit(candidate_run,manifest,row,gpu,root,proof) if kind == 'candidate' else pool.submit(readout_run,manifest,name,stage,gpu,output/'readouts')
                            running[future] = dict(kind=kind,category=name,row=row,gpu=gpu,reserve=reserve)
                    persist(manifest,state,output)
                    if not running:
                        if draining:state['status'] = 'drained';state['exit_code'] = 0
                        else:
                            state['status'] = 'completed' if all(c['status'] == 'completed' for c in state['categories'].values()) else 'completed_with_failures'
                            state['exit_code'] = 0 if state['status'] == 'completed' else 1
                            if state['status'] == 'completed':state['frozen_order'] = freeze_order(manifest,state)
                        break
                    finished,_ = wait(running,return_when=FIRST_COMPLETED)
                    for future in finished:
                        item = running.pop(future);record = future.result();row = item['row'];category = state['categories'][item['category']]
                        row.update({key:record[key] for key in ('status','exit_code','error','result','full8_mean','transpose','evidence') if key in record})
                        task_path = root/row['candidate_id']/'candidate.json' if item['kind'] == 'candidate' else output/'readouts'/item['category']/'task.json'
                        row['execution'] = b.bound(task_path)
                        if item['kind'] == 'candidate' and record['status'] == 'completed':
                            category['known'][row['candidate_id']] = record['result'];cache['results'][b.recipe_key(row['recipe'])] = record['result']
                        state[item['kind']+'_GPU_hours'] += record['owned_GPU_hours'];free.append(item['gpu'])
                    persist(manifest,state,output)
        except BaseException as error:
            state.update(status='stopped',exit_code=1,error=type(error).__name__+': '+str(error));raise
        finally:persist(manifest,state,output)
        return state['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',required=True,choices=('prepare','run','report'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--authorization',type=Path)
    parser.add_argument('--old-manifest',type=Path);parser.add_argument('--result-document',type=Path)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--drain-file',type=Path);parser.add_argument('--resource-receipt',type=Path)
    parser.add_argument('--wait-for-prior',action='store_true')
    args = parser.parse_args()
    if args.component == 'prepare':
        b.require(args.authorization and args.old_manifest and args.result_document,'Prepare needs the real approval, prior manifest and unique result document.')
        manifest = b.make_manifest(args.output,args.authorization,args.old_manifest,args.result_document)
        args.output.mkdir(parents=True,exist_ok=True);b.write_new(args.output/'manifest.json',manifest);return 0
    if args.component == 'run':return run(args.output/'manifest.json',resume=args.resume,drain_file=args.drain_file,resource_receipt=args.resource_receipt,wait_prior=args.wait_for_prior)
    from .report import summarize
    summarize(b.read(args.output/'manifest.json'),b.read(args.output/'search.json'),args.output/'selected_settings.json',update_markdown=True);return 0


if __name__ == '__main__':raise SystemExit(main())

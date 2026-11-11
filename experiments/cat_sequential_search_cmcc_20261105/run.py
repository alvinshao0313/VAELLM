"""Run this finite seven-category comparison batch, retaining actual failure states."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.cat_night_profiles_cmcc_20261104.run import wait_child
from experiments.cmcc_runtime_20261030.sequence import check_resources
from experiments.cat_offline_execution_cmcc_20261104 import evaluate
from experiments.cat_offline_execution_cmcc_20261104.results import worker_result, attach_full8
from . import bindings as b
from .report import summarize


def cleanup_technical(directory, completed, formal):
    """After formal replacement retain technical receipts, remove redundant tensor payloads."""
    b.require(formal['formal_eligible'] and formal['phase'] == 'formal', 'Only a completed formal replacement releases technical payloads.')
    root = (directory/'technical_validation').resolve(); paths = []
    for row in completed['candidates']:
        raw = Path(row['raw_checkpoint']['output_dir']).resolve(); meta = b.read(raw/'checkpoint_meta.json')
        paths.extend([raw/meta['state_dict_file'], Path(row['bank']['state_file'])])
        runtime = meta.get('extra_meta', {}).get('cat_runtime_state_file')
        if runtime: paths.append(raw/runtime)
        paths.extend(raw/name for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json'))
    rows = []
    for path in dict.fromkeys(paths):
        b.require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Never clean a symlink or another task path.')
        if path.is_file():
            stat = path.stat(); rows.append(dict(path=str(path), bytes=stat.st_size, allocated_bytes=stat.st_blocks*512,
                device=stat.st_dev, inode=stat.st_ino, links=stat.st_nlink,
                reason='Own completed technical2 superseded by actual fresh formal10K; immutable technical completed/fit/fresh evidence retained'))
    return rows


def candidate_run(manifest, candidate, gpu, root):
    directory = root/candidate['candidate_id']; directory.mkdir()
    started = time.time(); proof = b.proof_for(manifest, candidate, root)
    windows = b.phase_windows(manifest, candidate['recipe'], needs_technical=proof is None and candidate['parameter'] != 'multiplier')
    deadline = started+sum(windows.values())
    state = dict(candidate_id=candidate['candidate_id'], gpu=gpu, settings=candidate['settings'], recipe=candidate['recipe'],
        parameter=candidate['parameter'], status='running', controller_pid=os.getpid(), started_epoch=started,
        absolute_deadline_epoch=deadline, stages=[])
    path = directory/'candidate.json'; b.write(path, state)
    try:
        b.integrity(manifest); check_resources([str(gpu)])
        b.require(candidate['parameter'] != 'multiplier' or proof is not None,
            'LR stage lacks actual same_resource_path technical evidence; fail explicitly, never add an unbudgeted technical run.')
        state['technical_evidence_reuse'] = proof
        phases = ('formal',) if proof else ('technical_validation', 'formal')
        technical = None
        for phase in phases:
            b.integrity(manifest); check_resources([str(gpu)])
            request = b.make_request(manifest, candidate, directory, gpu, phase, deadline, proof=proof)
            registry = directory/(phase+'_request.json'); b.write_new(registry, request)
            output = directory/phase; output.mkdir()
            command = [sys.executable, '-B', '-u', '-m', 'experiments.cat_night_profiles_cmcc_20261104.search', '--request', str(registry), '--worker']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ACCELERATE_USE_CPU='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            state['stages'].append(dict(phase=phase, request=b.bound(registry), deadline_epoch=request['absolute_deadline_epoch']))
            b.write(path, state)
            code = wait_child(command, environment, output, request['absolute_deadline_epoch'])
            b.require(code == 0, 'Actual '+phase+' failed; this candidate dependencies are blocked without retry.')
            completed = b.read(output/'completed.json')
            b.require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['request_identity'] == b.digest(request), 'Actual completion and frozen request differ.')
            if phase == 'technical_validation':
                inventory = b.read(manifest['inventory']['path'])['inventory']
                b.search.batched.native._validate_model_path_evidence(completed, inventory, manifest['run_identity'], [candidate])
                technical = completed; proof = b.bound(output/'completed.json')
        result = worker_result(directory/'formal/completed.json', candidate['candidate_id'], request_reference=b.bound(directory/'formal_request.json'))
        b.write_new(directory/'formal/offline_result.json', result)
        if technical:
            state['technical_cleanup'] = cleanup_technical(directory, technical, result)
            b.write(path, state)
            for row in state['technical_cleanup']:
                target = Path(row['path']); target.unlink(); b.require(not target.exists(), 'Own redundant technical payload removal failed.')
            state['technical_cleanup_complete'] = True; b.write(path, state)
        b.integrity(manifest); check_resources([str(gpu)])
        now = time.time(); end = min(deadline, now+windows['full8'])
        auth = dict(authorization_reference=manifest['rules']['authorization_reference'], physical_GPU=gpu,
            authorized_GPU_hours=windows['full8']/3600, authorized_start_epoch=now, absolute_deadline_epoch=end)
        output = directory/'full8'; output.mkdir()
        plan = evaluate.make_plan(result=result, baseline=b.load_baseline(manifest['baseline']['path']), baseline_path=manifest['baseline']['path'], output=output, authorization=auth)
        b.write_new(output/'plan.json', plan)
        state['stages'].append(dict(phase='full8', plan=b.bound(output/'plan.json'), deadline_epoch=end)); b.write(path, state)
        evaluate.run(output/'plan.json')
        scored = attach_full8(result, summary_path=output/'full8_summary.json', evaluation_path=output/'evaluation.json')
        b.write_new(directory/'formal/offline_full8_result.json', scored)
        b.checked_result(directory/'formal/offline_full8_result.json', manifest, expected_recipe=candidate['recipe'])
        state.update(status='completed', exit_code=0, result=b.bound(directory/'formal/offline_full8_result.json'), full8_mean=scored['full8_mean'])
    except Exception as error:
        state.update(status='failed', exit_code=1, error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        state['finished_epoch'] = time.time(); state['owned_GPU_hours'] = (state['finished_epoch']-started)/3600
        b.write(path, state)
    return state


def select_round(manifest, current, stage):
    results = [(current['result'], current['settings'])]
    for row in stage['candidates']:
        if row['status'] in ('completed', 'reused'): results.append((row['result'], row['settings']))
    scored = []
    for reference, settings in results:
        b.require(b.sha(reference['path']) == reference['sha256'], 'Actual candidate result changed after completion.')
        result = b.checked_result(reference['path'], manifest)
        scored.append((b.selection.quality_key(result['full8_mean'], result['logical_bits'], result['packed_bits'], result['candidate_id']), reference, settings, result))
    scored.sort(key=lambda row: row[0]); winner = scored[0]
    old = b.checked_result(current['result']['path'], manifest)
    b.require(winner[3]['full8_mean'] >= old['full8_mean'], 'Retaining current evidence must prevent quality regression.')
    return dict(result=winner[1], settings=winner[2], candidate_id=winner[3]['candidate_id'])


def advance(manifest, category, state):
    """No class successor exists until every declared legal current candidate completes."""
    if state['status'] in ('completed', 'failed', 'budget_exhausted'): return
    if state['rounds'] and state['rounds'][-1]['status'] == 'running':
        stage = state['rounds'][-1]
        if any(row['status'] in ('declared_pending', 'running') for row in stage['candidates']): return
        if any(row['status'] == 'failed' for row in stage['candidates']):
            stage['status'] = 'failed'; state['status'] = 'failed'; return
        state['current'] = select_round(manifest, state['current'], stage)
        stage.update(status='completed', selected=deepcopy(state['current']))
        state['next_parameter'] += 1
    if state['next_parameter'] == len(manifest['rules']['order']): state['status'] = 'completed'; return
    parameter = manifest['rules']['order'][state['next_parameter']]
    stage = dict(parameter=parameter, status='running', input=deepcopy(state['current']),
        candidates=b.proposals(manifest, category, parameter, state['current']['settings'], state['known']))
    state['rounds'].append(stage)
    if all(row['status'] in ('reused', 'uncovered_cap', 'uncovered_illegal') for row in stage['candidates']): advance(manifest, category, state)


def run(manifest_path, *, resume=False):
    manifest = b.read(manifest_path); b.integrity(manifest)
    b.require(manifest['manifest_identity'] == b.digest({k:v for k,v in manifest.items() if k != 'manifest_identity'}), 'Frozen manifest changed.')
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU batch controller requires caller bitvae and hidden CUDA.')
    output = Path(manifest['output_root']); root = output/'candidates'; state_path = output/'search.json'
    if resume:
        state = b.read(state_path)
        b.require(state['manifest'] == b.bound(manifest_path) and state['status'] not in ('completed', 'completed_with_failures'), 'Resume this exact unfinished batch only; no duplicate run.')
        for category in state['categories'].values():
            for stage in category['rounds']:
                for row in stage['candidates']:
                    if row['status'] == 'running':
                        record = b.read(root/row['candidate_id']/'candidate.json')
                        b.require(record['status'] in ('completed', 'failed'), 'Existing worker is running/incomplete; never take over or retry it.')
                        row.update(status=record['status'])
                        if record['status'] == 'completed': row['result'] = record['result']; category['known'][row['candidate_id']] = record['result']
                        state['owned_GPU_hours'] += record['owned_GPU_hours']
    else:
        b.require(not state_path.exists(), 'Existing batch requires explicit --resume, never overwrite.')
        root.mkdir()
        state = dict(schema='cat_sequential_search_state_v1', status='running', controller_pid=os.getpid(), started_epoch=time.time(),
            manifest=b.bound(manifest_path), owned_GPU_hours=0., categories={})
        for name, anchor in manifest['anchors'].items():
            state['categories'][name] = dict(status='running', next_parameter=0, current=deepcopy(anchor),
                known={anchor['candidate_id']:anchor['result']}, rounds=[])
    state.update(status='running', controller_pid=os.getpid()); b.write(state_path, state)
    running = {}; free = list(manifest['rules']['gpus'])
    try:
        check_resources([str(gpu) for gpu in free])
        with ThreadPoolExecutor(max_workers=len(free)) as pool:
            while True:
                b.integrity(manifest)
                for name, category in state['categories'].items():
                    try: advance(manifest, name, category)
                    except Exception as error:
                        category.update(status='failed',generation_or_selection_error=type(error).__name__+': '+str(error),
                            failure_parameter=manifest['rules']['order'][category['next_parameter']],error_traceback=traceback.format_exc())
                pending = []
                # Interleave categories before taking a second candidate of one class.
                groups = [(name, category, [row for row in category['rounds'][-1]['candidates'] if row['status'] == 'declared_pending'])
                    for name, category in state['categories'].items() if category['status'] == 'running' and category['rounds']]
                for index in range(max((len(rows) for _,_,rows in groups), default=0)):
                    pending.extend((name, category, rows[index]) for name,category,rows in groups if index < len(rows))
                for name, category, row in pending:
                    if not free: break
                    proof = b.proof_for(manifest,row,root)
                    reserve = sum(b.phase_windows(manifest,row['recipe'],needs_technical=proof is None and row['parameter'] != 'multiplier').values())/3600
                    used = state['owned_GPU_hours']+sum(item['reserve'] for item in running.values())
                    if used+reserve > manifest['rules']['total_GPU_hours']:
                        category['status'] = 'budget_exhausted'; row.update(status='uncovered_budget', reason='Finite complete candidate reservation would exceed confirmed whole-batch GPUh'); continue
                    b.require(len(category['known'])+sum(r['status'] in ('running','declared_pending') for s in category['rounds'] for r in s['candidates']) <= 19,
                        'Distinct full-fitting candidate count exceeds19; no borrowed empty slot.')
                    gpu = free.pop(0); row['status'] = 'running'; row['gpu'] = gpu
                    future = pool.submit(candidate_run, manifest, row, gpu, root)
                    running[future] = dict(category=name, row=row, gpu=gpu, reserve=reserve)
                b.write(state_path, state)
                if not running: break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = running.pop(future); record = future.result(); row = item['row']; category = state['categories'][item['category']]
                    row.update(status=record['status'], execution=b.bound(root/row['candidate_id']/'candidate.json'))
                    if record['status'] == 'completed': row['result'] = record['result']; category['known'][row['candidate_id']] = record['result']
                    else: row['error'] = record['error']
                    state['owned_GPU_hours'] += record['owned_GPU_hours']; free.append(item['gpu'])
                b.write(state_path, state)
        state['status'] = 'completed' if all(row['status'] == 'completed' for row in state['categories'].values()) else 'completed_with_failures'
        state['exit_code'] = 0 if state['status'] == 'completed' else 1
    except BaseException as error:
        state.update(status='stopped', exit_code=1, error=type(error).__name__+': '+str(error)); raise
    finally:
        state['last_updated_epoch'] = time.time(); b.write(state_path, state)
        summarize(manifest, state, output/'selected_settings.json', update_markdown=state['status'] in ('completed', 'completed_with_failures'))
    return state['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--component', choices=('prepare', 'run', 'report'), required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--rules', type=Path); parser.add_argument('--cap', type=Path); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.component == 'prepare':
        b.require(args.rules and args.cap, 'Prepare requires actual confirmed rules and complete cap.')
        manifest = b.make_manifest(args.output, args.rules, args.cap); args.output.mkdir(parents=True, exist_ok=False)
        b.write_new(args.output/'manifest.json', manifest); return 0
    if args.component == 'run': return run(args.output/'manifest.json', resume=args.resume)
    manifest = b.read(args.output/'manifest.json'); state = b.read(args.output/'search.json')
    summarize(manifest, state, args.output/'selected_settings.json', update_markdown=True); return 0


if __name__ == '__main__': raise SystemExit(main())

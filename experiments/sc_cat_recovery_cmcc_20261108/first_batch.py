"""Finite first-class execution after explicit budget approval and GPU validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import threading
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from train_utils import checkpoint_v6 as v6
from . import io, search
from .boundary import complete_boundary_reference, StrictMainChain
from .identity import source_pins
from .processes import Budget, Processes, check_resources
from .selection import TrajectoryResult, SelectedCategory, compare, long_check, select_category
from .settings import RecoverySettings, ROUNDS, trajectory_id


def record(row):
    def score(s):
        return dict(task_scores=dict(s.task_scores), evaluation_id=s.evaluation_id, evidence=dict(s.evidence))
    return dict(category_id=row.category_id,parent_id=row.parent_id,raw_id=row.raw_id,recipe_id=row.recipe_id,
        candidate_id=row.candidate_id,trajectory_id=row.trajectory_id,logical_bits=row.logical_bits,
        packed_bits=row.packed_bits,settings=asdict(row.settings),status=row.status,raw_score=score(row.raw_score),
        nodes=[dict(step=n.step, checkpoint_id=n.checkpoint_id, score=score(n.score)) for n in row.nodes])


class FirstBatch:
    def __init__(self, authorization, technical_receipt):
        self.auth_ref = io.binding(authorization)
        self.auth = io.read(authorization)
        self.manifest_ref = self.auth['manifest']
        self.manifest = io.read(io.bound_path(self.manifest_ref))
        self.root = Path(self.manifest['output_root'])
        self.handoff = io.read(io.bound_path(self.manifest['selected_settings']))
        self.category = self.handoff['frozen_order'][0]
        self.state = io.read(self.root/'search.json')
        continuation_ref = self.auth.get('continuation')
        continuation = io.read(io.bound_path(continuation_ref)) if continuation_ref else None
        self.hpo_resume = bool(continuation and continuation.get('schema')=='reviewed_hpo_continuation_v1')
        self.formal_root = self.root/('formal_kappa_fix' if self.hpo_resume else 'formal_storage_fix' if continuation else 'formal')
        require(self.auth.get('confirmed') is True and self.auth.get('user_approved') is True
            and self.auth['categories']==[self.category] and self.auth['gpus']==[4,5,6,7]
            and self.state['status']==('failed_requires_review' if continuation else 'awaiting_first_class_budget'),
            'Only the explicitly approved first batch or its reviewed continuation may start.')
        proof = io.read(io.bound_path(io.binding(technical_receipt)))
        if continuation:
            if self.hpo_resume:
                from .hpo_recovery import validate_hpo_continuation
                validate_hpo_continuation(continuation,self.manifest,self.auth,proof,self.state)
            else:
                from .continuation import validate_continuation
                validate_continuation(continuation,self.manifest,self.auth,proof,self.state)
        else:
            require(proof['manifest']==self.manifest_ref and proof['source_pins']==source_pins(), 'GPU implementation proof changed.')
        require(proof['status']=='actual_GPU_verified' and proof['exit_code']==0,
            'Require the actual new implementation GPU proof, not an old experiment or CPU proof.')
        required = {'formal_B4_length1024_backward', 'joint_q_scale_and_four_Muon_groups',
            'rank48_native_export_and_independent_fresh', 'P_only_frozen_decoder_backward',
            'native250_resume_to500_and_data_RNG', 'zero_raw_native_export'}
        require(set(proof['checks'])==required and all(proof['checks'].values()) and proof['receipts'], 'GPU checks are incomplete.')
        for ref in proof['receipts']:
            io.bound_path(ref)
        self.inventory = io.read(io.bound_path(self.manifest['inventory']))['inventory']
        check_resources([4,5,6,7])
        previous = io.read(io.bound_path(continuation['budget'])) if continuation else None
        windows = continuation['ownership_windows'] if self.hpo_resume else proof['ownership_windows']
        self.budget = Budget(self.formal_root/'budget.json', self.auth, windows,
            [io.binding(technical_receipt)]+([continuation['budget']] if continuation else []), previous=previous,
            retry_tasks=continuation['retry_tasks'] if self.hpo_resume else None)
        self.processes = Processes(self.formal_root, self.budget, source_pins(), continuation['completed_tasks'] if continuation else None)
        self.resumed = continuation['trajectories'] if continuation else {}
        self.lock = threading.RLock()
        self.export_slots = threading.BoundedSemaphore(3)
        self.results, self.contexts, self.deployments, self.best_exports = {}, {}, {}, {}
        self.training_reports, self.fixed = {}, {}
        if self.hpo_resume:
            from .hpo_recovery import restore_search
            restore_search(self)
        self.common = dict(manifest=self.manifest_ref, authorization=self.auth_ref,
            source_manifest=self.handoff['source_manifest'], inventory_receipt=self.manifest['inventory'],
            data_plan_path=self.manifest['data_plan'], phase='formal', category_id=self.category,
            execution_policy=dict(train_device='cuda:0',attention_implementation='flash_attention_2',
                teacher_model_offload='none',teacher_output_offload='cpu',teacher_output_pin_memory=True,
                teacher_output_chunk_tokens=8,deterministic=False))
        self.state.update(status='first_class_running', approved_categories=[self.category])
        if not continuation:
            self.state['categories'][self.category] = dict(status='running', raws={}, trajectories={}, rounds=[])
        self.state['active_authorization'] = self.auth_ref
        self.state['active_manifest'] = self.manifest_ref
        self.persist()

    def persist(self):
        with self.lock:
            self.state['cumulative_GPUh'] = self.budget.value['total_actual_GPU_hours']
            search.save(self.root/'search.json',self.state)
            from .report import update
            update(self.manifest,self.state,self.budget.value)

    def parallel(self, functions):
        with ThreadPoolExecutor(max_workers=max(1,len(functions))) as executor:
            pending = [executor.submit(fn) for fn in functions]
            try:
                return [f.result() for f in as_completed(pending)]
            except BaseException:
                self.processes.cancel()
                for f in pending:
                    f.cancel()
                raise

    def reserve_trajectory(self, identifier, settings):
        count = len(settings.node_steps)
        tasks = {identifier+'/train':dict(component='long_training' if settings.schedule_max==1000 else 'training',
            seconds=5400 if settings.schedule_max==1000 else 2700)}
        for step in settings.node_steps:
            tasks[identifier+f'/node_{step}/export'] = dict(component='export',seconds=450)
            tasks[identifier+f'/node_{step}/full8'] = dict(component='evaluation',seconds=900)
        if identifier in self.resumed:
            require(set(tasks).issubset(self.budget.value['tasks']), 'Inherited trajectory reservation is incomplete.')
        else:
            self.budget.reserve(tasks,counts=dict(trajectories=1,KD_updates=settings.schedule_max,node_full8=count))

    def prune_weight(self, report, reason):
        """Only completed evaluated dense copies dominated within this same raw."""
        folder = Path(report['path']).resolve()
        require(folder.is_relative_to(self.root.resolve()), 'Cleanup cannot escape this new batch.')
        meta = v6.load_v6_meta(str(folder))
        path = folder/meta['state_dict_file']
        require(meta['checkpoint_id']==report['uuid'] and path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(folder), 'Cleanup identity/path changed.')
        entry = dict(path=str(path),bytes=path.stat().st_size,reason=reason,source=report['export_evidence']['source_node_id'])
        self.state.setdefault('cleanup',[]).append(dict(entry,status='planned'))
        self.persist()
        path.unlink()
        self.state['cleanup'][-1].update(status='completed')
        self.persist()

    def node(self, context, settings, identifier, step, node_directory=None):
        prefix = identifier+f'/node_{step}' if step else context['candidate']+'/raw'
        output = self.formal_root/'candidates'/prefix
        request = dict(self.common,operation='export' if step else 'export_zero',live_raw=context['raw_ref'],
            settings=asdict(settings),output=str(output/'export'),destination=str(output/'deployment'),
            trajectory_id=identifier,recipe_id=context['recipe_id'],source_candidate_id=context['candidate'])
        if node_directory is not None:
            request['node_directory'] = io.binding(Path(node_directory)/'checkpoint_meta.json')
            # Worker takes a bound metadata file to identify its exact directory.
        with self.export_slots:
            report_ref = self.processes.run(prefix+'/export','worker',request,output_file=output/'export'/'deployment_report.json')
            report = io.read(io.bound_path(report_ref))
            score_ref = self.processes.run(prefix+'/full8','evaluate',dict(self.common,deployment_report=report_ref,
                output=str(output/'full8')),output_file=output/'full8'/'score.json')
            score = read_score(score_ref)
            source = report['export_evidence']['source_node_id']
            require(score.evidence['checkpoint_id']==report['uuid'] and score.evidence['source_checkpoint_id']==source,
                'Full8 must bind the actual evaluated deployment and native source.')
            with self.lock:
                key = (-score.mean,step==0,-step,identifier)
                retention_key = (context['raw_id'],'raw' if step==0 else 'recovery') if getattr(self,'terminal_pair',False) else context['raw_id']
                previous = self.best_exports.get(retention_key)
                self.deployments[source] = dict(report=report_ref,score=score_ref,key=key)
                self.state.setdefault('nodes',{})[source]=dict(report=report_ref,score=score_ref,
                    candidate_id=context['candidate'],trajectory_id=identifier,step=step,settings=asdict(settings),task_prefix=prefix)
                if previous is None or key < previous['key']:
                    self.best_exports[retention_key] = dict(source=source,key=key)
                    if previous is not None:
                        old = io.read(io.bound_path(self.deployments[previous['source']]['report']))
                        self.prune_weight(old,'Lower completed full8 node of the same raw; exact masters and metrics retained.')
                else:
                    self.prune_weight(report,'Lower completed full8 node of the same raw; exact masters and metrics retained.')
                self.persist()
            return NodeScore(step,score,source)

    def trajectory(self, context, settings, fixed=None):
        identifier = trajectory_id(context['raw_id'],settings,self.manifest['data_identity'],self.manifest['code_identity'])
        output = self.formal_root/'candidates'/identifier/'train'
        request = dict(self.common,operation='train',live_raw=context['raw_ref'],settings=asdict(settings),
            output=str(output),trajectory_id=identifier,recipe_id=context['recipe_id'],
            source_candidate_id=context['candidate'],fixed_calibrations=fixed)
        inherited = self.resumed.get(identifier)
        if inherited:
            require(inherited['settings']==asdict(settings) and inherited['live_raw']==context['raw_ref'], 'Changed resumed trajectory inputs.')
            request['resume_from'] = str(io.bound_path(inherited['resume_metadata']).parent)
        previous_nodes = {n['step']:n for n in inherited['nodes']} if inherited else {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            training = executor.submit(self.processes.run,identifier+'/train','worker',request,
                output_file=output/'training_result.json')
            def measured(step):
                directory = Path(previous_nodes[step]['directory']) if step in previous_nodes else output/'trainer_state'/f'checkpoint-{step}'
                while not (directory/'node_ready.json').is_file():
                    require(not self.processes.cancelled.is_set(), 'A batch dependency failed.')
                    if training.done():
                        training.result()
                        require((directory/'node_ready.json').is_file(), 'Completed training omitted a scheduled node.')
                    time.sleep(.25)
                return self.node(context,settings,identifier,step,directory)
            nodes = self.parallel([lambda s=s:measured(s) for s in settings.node_steps])
            training_ref = training.result()
        report = io.read(io.bound_path(training_ref))
        if inherited:
            require(report['actual']['saved_state_restore_verified'] and report['actual']['starting_step']==375,
                'The actual saved375 state must be restored before continuation can be accepted.')
            report = dict(report, nodes=[*inherited['nodes'],*report['nodes']], inherited_nodes=inherited['nodes'],
                resumed_segment=training_ref, prior_execution=inherited['prior_request'])
            aggregate = output/'combined_training_result.json'
            io.write(aggregate,report)
            training_ref = io.binding(aggregate)
        require(report['settings']==asdict(settings) and report['actual']['actual_steps']==settings.schedule_max,
            'Training did not finish its unchanged declared schedule.')
        fee = context['raw']['actual_ledger']['per_category'][self.category]
        if 'raw_score' not in context:
            context['raw_score'] = context['raw_score_future'].result().score
        row = TrajectoryResult(self.category,self.handoff['parent']['uuid'],context['raw_id'],context['recipe_id'],
            context['candidate'],identifier,fee['logical_bits'],fee['packed_bits'],settings,
            context['raw_score'],tuple(sorted(nodes,key=lambda n:n.step)))
        self.training_reports[identifier] = report
        self.fixed[identifier] = {g['group_name']:g['calibration'] for g in report['optimizer']['groups'] if g['kind']=='muon'}
        io.write(output.parent/'trajectory.json',record(row))
        with self.lock:
            self.results[identifier] = row
            self.state['categories'][self.category]['trajectories'][identifier] = dict(
                result=io.binding(output.parent/'trajectory.json'), training=training_ref)
            # Full500/1000 trajectories never continue from their terminal optimizer.
            # Keep each trajectory's best exact master for re-export. Other nodes
            # cannot beat that node under the unchanged within-trajectory rule.
            for node in report['nodes']:
                directory = Path(node['directory'])
                names=['optimizer.pt','scheduler.pt','rng_state.pth']
                if node['step']!=row.best_node.step:
                    names.append('training_model_state.pt')
                for name in names:
                    path = directory/name
                    if path.is_file():
                        entry = dict(path=str(path),bytes=path.stat().st_size,
                            reason='Complete trajectory and full8; no continuation; best node master, all source metadata/audits and metrics retained.')
                        self.state.setdefault('cleanup',[]).append(dict(entry,status='planned'))
                        self.persist()
                        path.unlink()
                        self.state['cleanup'][-1]['status']='completed'
            self.persist()
        return row

    def default(self, candidate):
        raw_out = self.formal_root/'candidates'/candidate/'raw_source'
        ref = self.processes.run(candidate+'/raw_source','worker',dict(self.common,operation='first_raw',
            source_candidate_id=candidate,output=str(raw_out)),output_file=raw_out/'live_raw.json')
        raw = io.read(io.bound_path(ref))
        require(raw['formal_eligible'] and raw['source_parent_uuid']==self.handoff['parent']['uuid'], 'Wrong BF16 root.')
        context = dict(raw_ref=ref,raw=raw,raw_id=raw['round_base_uuid'],candidate=candidate,recipe_id=digest(raw['recipe']))
        settings = RecoverySettings()
        identifier = trajectory_id(context['raw_id'],settings,self.manifest['data_identity'],self.manifest['code_identity'])
        self.reserve_trajectory(identifier,settings)
        # Raw full8 and training use different free GPUs. Training can start once
        # raw exists; the score is needed only when constructing its final record.
        with ThreadPoolExecutor(max_workers=1) as executor:
            raw_eval = executor.submit(self.node,context,settings,'raw0:'+context['raw_id'],0)
            # A future score blocks only the final record, not actual KD.
            context['raw_score_future'] = raw_eval
            row = self.trajectory(context,settings)
            raw_eval.result()
        self.contexts[context['raw_id']] = context
        with self.lock:
            self.state['categories'][self.category]['raws'][candidate] = dict(raw=ref, recipe=raw['recipe'],
                zero_score=self.deployments[context['raw_id']]['score'])
            self.persist()
        return row

    def run(self):
        entries = self.handoff['categories'][self.category]['top2']
        tasks = {}
        for candidate in entries:
            tasks[candidate+'/raw_source'] = dict(component='raw',seconds=1800)
            tasks[candidate+'/raw/export'] = dict(component='raw',seconds=900)
            tasks[candidate+'/raw/full8'] = dict(component='evaluation',seconds=900)
        if self.hpo_resume:
            winner_raw=self.state['categories'][self.category]['default_winner_raw']
            winner=next(r for r in self.results.values() if r.raw_id==winner_raw and r.settings==RecoverySettings())
        elif self.resumed:
            require(set(tasks).issubset(self.budget.value['tasks']), 'Inherited raw reservation is incomplete.')
        else:
            self.budget.reserve(tasks,counts=dict(raws=len(entries),raw_full8=len(entries)))
        if not self.hpo_resume:
            defaults = self.parallel([lambda c=c:self.default(c) for c in entries])
            winner,_ = compare(defaults)[0]
        block = self.state['categories'][self.category]
        block['default_winner_raw'] = winner.raw_id
        context, base = self.contexts[winner.raw_id], winner.settings
        current = [p['actual_name'] for p in self.inventory['projections'] if p['category_id']==self.category]
        owners = dict(prefix=current,current=current,
            protected=[n for n,c in context['raw']['recipe']['protection']['counts'].items() if c],
            future=[p['actual_name'] for p in self.inventory['projections'] if p['category_id']!=self.category])
        for parameter in ROUNDS:
            existing=next((r for r in block['rounds'] if r['parameter']==parameter),None)
            if self.hpo_resume and existing is not None and existing['status']=='complete':
                if existing['winner'] is not None:
                    base=self.results[existing['winner']].settings
                continue
            if self.hpo_resume and existing is not None:
                row=existing
                require(row['base']==asdict(base) and row['status']=='pending','Resume the frozen pending round unchanged.')
            else:
                row = search.begin_round(self.state,category=self.category,parameter=parameter,base=base,raw_id=winner.raw_id,
                    inventory=owners,data_identity=self.manifest['data_identity'],code_identity=self.manifest['code_identity'],completed=self.results)
            require(not row['short_screen'], 'First class has no inherited calibration; never silently use250 screening.')
            new = [c for c in row['candidates'] if not c['reuse']]
            for c in new:
                if not (self.hpo_resume and existing is not None):
                    self.reserve_trajectory(c['trajectory_id'],RecoverySettings(**c['settings']))
            reference = trajectory_id(context['raw_id'],base,self.manifest['data_identity'],self.manifest['code_identity'])
            fixed = self.fixed.get(reference) if parameter in ('decoder_lr','protection_lr','lora_lr') else None
            self.persist()
            self.parallel([lambda c=c:self.trajectory(context,RecoverySettings(**c['settings']),fixed) for c in new])
            chosen = search.finish_round(self.state,self.category,{c['trajectory_id']:self.results[c['trajectory_id']] for c in row['candidates']})
            if chosen is not None:
                base = chosen
            self.persist()
        check = long_check(list(self.results.values()),context['recipe_id'],self.state['long_categories'])
        if check is not None:
            settings = replace(check.settings,schedule_max=1000)
            identifier = trajectory_id(check.raw_id,settings,self.manifest['data_identity'],self.manifest['code_identity'])
            self.reserve_trajectory(identifier,settings)
            self.state['long_categories'].append(self.category)
            self.persist()
            self.trajectory(context,settings,self.fixed[check.trajectory_id])
        chosen, certificate = select_category(list(self.results.values()))
        deployment = self.deployments[certificate['selected_node_id']]
        report = io.read(io.bound_path(deployment['report']))
        boundary = complete_boundary_reference(export_report=report,inventory=self.inventory,full8_score=read_score(deployment['score']))
        seeds = recovery_seeds(self.inventory['model_identity'],self.category,self.handoff['run_identity']['search_seed'])
        selected = SelectedCategory(self.category,chosen.recipe_id,chosen.candidate_id,
            RecoverySettings() if certificate['selected_step']==0 else chosen.settings,
            certificate['selected_step'],certificate['selected_trajectory_id'],certificate['selected_node_id'],chosen.raw_id,seeds,'formal')
        chain = StrictMainChain(bf16_root_id=self.handoff['parent']['uuid'],categories=self.handoff['frozen_order'])
        chain.commit(selected,live_parent_id=self.handoff['parent']['uuid'],boundary_reference=boundary,
            selection_certificate=certificate,actual_round_base_id=chosen.raw_id,restored_node_id=selected.source_node_id,
            runtime_identity=boundary['runtime_identity'])
        io.write(self.formal_root/'first_boundary.json',boundary)
        io.write(self.formal_root/'first_selected_recipe.json',dict(selected=asdict(selected),certificate=certificate,
            optimizer=None if selected.selected_step==0 else self.training_reports[chosen.trajectory_id]['optimizer'],
            replay=selected.training_plan()))
        block.update(status='completed',selected=asdict(selected),certificate=certificate,
            boundary=io.binding(self.formal_root/'first_boundary.json'))
        search.commit_first_batch(self.state,boundary=boundary,selected=certificate,
            cost=self.budget.value['total_actual_GPU_hours'],long_categories=self.state['long_categories'])
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--technical-receipt',required=True)
    args = parser.parse_args()
    run = FirstBatch(args.authorization,args.technical_receipt)
    try:
        run.run()
    except BaseException:
        run.processes.cancel()
        run.state['status']='failed_requires_review'
        run.persist()
        raise


if __name__=='__main__':
    main()

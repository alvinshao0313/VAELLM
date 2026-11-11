"""Continue the verified first boundary through the remaining fixed-order classes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import threading
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from train_utils import checkpoint_v6 as v6
from . import io, search
from .boundary import StrictMainChain, complete_boundary_reference
from .first_batch import FirstBatch, record
from .identity import source_pins
from .processes import Budget, Processes, check_resources
from .selection import TrajectoryResult, SelectedCategory, compare, long_check, select_category, terminal_endpoints
from .settings import RecoverySettings, ROUNDS, normalize, trajectory_id


def owners(raw, inventory):
    meta=v6.load_v6_meta(raw['round_base_dir'])
    current=[p['actual_name'] for p in inventory['projections'] if p['category_id']==raw['current_category']]
    modules={entry['name']:entry for entry in meta['converted_modules']}
    protected=[name for name in meta['compressed_targets'] if modules[name].get('protected_input_qvalues')
        and modules[name].get('protected_input_scales')]
    return dict(prefix=meta['compressed_targets'],current=current,protected=protected,future=meta['pending_dense_targets'])


def selected_recipe_record(selected, certificate, recipe, optimizer, deployment):
    """Persist artifact references; the Fraction sorting key is only an in-memory cache."""
    return dict(selected=selected, certificate=certificate, recipe=recipe, optimizer=optimizer,
                deployment={name: deployment[name] for name in ('report', 'score')})


class RemainingBatch(FirstBatch):
    def __init__(self, authorization):
        self.auth_ref=io.binding(authorization);self.auth=io.read(authorization)
        self.manifest_ref=self.auth['manifest'];self.manifest=io.read(io.bound_path(self.manifest_ref))
        self.root=Path(self.manifest['output_root']);self.formal_root=self.root/'remaining'
        self.handoff=io.read(io.bound_path(self.manifest['selected_settings']))
        self.state=io.read(self.root/'search.json')
        continuation=io.read(io.bound_path(self.auth['closed_continuation'])) if self.auth.get('closed_continuation') else None
        expected_status='remaining_classes_ready' if continuation else 'first_class_complete_awaiting_second_budget'
        require(self.auth.get('confirmed') is True and self.auth.get('user_approved') is True
            and self.auth['batch']=='remaining_and_replay' and self.auth['gpus']==[4,5,6,7]
            and self.auth['categories']==self.handoff['frozen_order']
            and self.state['status']==expected_status, 'Require the approved second batch and a verified closed prefix.')
        if continuation:
            from .resume_closed import validate_continuation
            validate_continuation(continuation,self.manifest,self.auth,self.state)
        proof=io.read(io.bound_path(self.manifest['remaining_cpu']))
        require(proof['status']=='passed' and proof['source_pins']==source_pins(), 'Second-batch CPU implementation changed.')
        first=io.read(io.bound_path(self.manifest['first_acceptance']))
        require(first['status']=='first_class_verified' and not first['final_chain_verified'], 'First class must have its own completed acceptance.')
        self.inventory=io.read(io.bound_path(self.manifest['inventory']))['inventory']
        check_resources([4,5,6,7])
        previous=io.read(io.bound_path(self.manifest['previous_budget']))
        budget_path=io.bound_path(self.auth['closed_continuation']).parent/'budget.json' if continuation else self.formal_root/'budget.json'
        self.budget=Budget(budget_path,self.auth,previous['technical_windows'],
            [self.manifest['previous_budget']],previous=previous,retry_tasks=(),new_batch=True)
        self.processes=Processes(self.formal_root,self.budget,source_pins())
        self.lock=threading.RLock();self.export_slots=threading.BoundedSemaphore(3)
        self.chain=StrictMainChain(bf16_root_id=self.handoff['parent']['uuid'],categories=self.handoff['frozen_order'])
        for category in self.handoff['frozen_order']:
            block=self.state['categories'].get(category,{})
            if block.get('status')!='completed':break
            self.parent_ref=block['boundary'];self.parent=io.read(io.bound_path(self.parent_ref))
            self.commit_chain(block['selected'],block['certificate'],self.parent)
        require(self.chain.boundaries, 'A verified selected prefix is required.')
        self.state.update(status='remaining_classes_running',approved_categories=self.handoff['frozen_order'][len(self.chain.boundaries):],
            active_authorization=self.auth_ref,active_manifest=self.manifest_ref)
        self.persist()

    def commit_chain(self, selected, certificate, boundary):
        value=dict(selected);value['settings']=RecoverySettings(**value['settings']);selected=SelectedCategory(**value)
        return self.chain.commit(selected,live_parent_id=self.chain.live_parent_id,boundary_reference=boundary,
            selection_certificate=certificate,actual_round_base_id=selected.source_round_base_id,
            restored_node_id=selected.source_node_id,runtime_identity=boundary['runtime_identity'])

    def setup_category(self, category):
        self.category=category
        self.terminal_pair=category==self.handoff['frozen_order'][-1]
        self.technical_gate=threading.Event()
        self.needs_technical=category in (self.handoff['frozen_order'][1],self.handoff['frozen_order'][-1])
        if not self.needs_technical:self.technical_gate.set()
        self.results,self.contexts,self.deployments,self.best_exports={},{},{},{}
        self.training_reports,self.fixed={},{}
        self.state['current_category']=category
        self.state['categories'][category]=dict(status='running',parent=self.parent_ref,raws={},trajectories={},rounds=[])
        self.common=dict(manifest=self.manifest_ref,authorization=self.auth_ref,source_manifest=self.handoff['source_manifest'],
            inventory_receipt=self.manifest['inventory'],data_plan_path=self.manifest['data_plan'],phase='formal',category_id=category,
            parent=self.parent_ref,execution_policy=self.auth['execution_policy'])
        self.persist()

    def reserve_segment(self, identifier, settings, stop, *, resumed=False):
        steps=tuple(s for s in settings.node_steps if (s>250 if resumed else s<=stop))
        full=self.auth['task_seconds']['train500_by_category'][self.category]
        seconds=2*full if settings.schedule_max==1000 else (full/2+180 if stop==250 or resumed else full)
        name=identifier+('/resume500' if resumed else '/train')
        tasks={name:dict(component='long_training' if settings.schedule_max==1000 else 'training',seconds=seconds)}
        for step in steps:
            tasks[identifier+f'/node_{step}/export']=dict(component='export',seconds=450)
            tasks[identifier+f'/node_{step}/full8']=dict(component='evaluation',seconds=900)
        counts=dict(node_full8=len(steps))
        if not resumed:counts.update(trajectories=1,KD_updates=settings.schedule_max)
        self.budget.reserve(tasks,counts=counts)

    def trajectory(self, context, settings, fixed=None, *, stop=None, resumed=False):
        stop=stop or settings.schedule_max
        identifier=trajectory_id(context['raw_id'],settings,self.manifest['data_identity'],self.manifest['code_identity'])
        self.reserve_segment(identifier,settings,stop,resumed=resumed)
        base=self.formal_root/'candidates'/identifier
        output=base/('resume500' if resumed else 'train')
        request=dict(self.common,operation='train',live_raw=context['raw_ref'],settings=asdict(settings),output=str(output),
            trajectory_id=identifier,recipe_id=context['recipe_id'],source_candidate_id=context['candidate'],fixed_calibrations=fixed)
        old=self.training_reports.get(identifier) if resumed else None
        if resumed:
            require(self.results[identifier].status=='pending250' and old['actual']['actual_steps']==250,'Resume only this real paused250 trajectory.')
            request['resume_from']=old['nodes'][-1]['directory']
        elif stop==250:
            request['stop_at']=250
        steps=[s for s in settings.node_steps if (s>250 if resumed else s<=stop)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            training=pool.submit(self.processes.run,identifier+('/resume500' if resumed else '/train'),'worker',request,
                output_file=output/'training_result.json')
            def measured(step):
                directory=output/'trainer_state'/f'checkpoint-{step}'
                while not (directory/'node_ready.json').is_file():
                    require(not self.processes.cancelled.is_set(),'A dependency failed.')
                    if training.done():
                        training.result();require((directory/'node_ready.json').is_file(),'Training omitted a scheduled node.')
                    time.sleep(.25)
                return self.node(context,settings,identifier,step,directory)
            nodes=self.parallel([lambda s=s:measured(s) for s in steps])
            training_ref=training.result()
        report=io.read(io.bound_path(training_ref))
        require(report['actual']['actual_steps']==stop and report['settings']==asdict(settings),'Unchanged schedule segment did not complete.')
        if resumed:
            require(report['actual']['saved_state_restore_verified'] and report['actual']['starting_step']==250,'250 state was not restored exactly.')
            nodes=[*self.results[identifier].nodes,*nodes]
            report=dict(report,nodes=[*old['nodes'],*report['nodes']],prior_segment=self.state['categories'][self.category]['trajectories'][identifier]['training'],resumed_segment=training_ref)
            io.write(output/'combined_training_result.json',report);training_ref=io.binding(output/'combined_training_result.json')
        if 'raw_score' not in context:context['raw_score']=context['raw_score_future'].result().score
        fees=context['raw']['actual_ledger']['per_category'][self.category]
        row=TrajectoryResult(self.category,self.parent['uuid'],context['raw_id'],context['recipe_id'],context['candidate'],identifier,
            fees['logical_bits'],fees['packed_bits'],settings,context['raw_score'],tuple(sorted(nodes,key=lambda n:n.step)),
            'pending250' if stop==250 else 'complete')
        self.training_reports[identifier]=report
        self.fixed[identifier]={g['group_name']:g['calibration'] for g in report['optimizer']['groups'] if g['kind']=='muon'}
        self.save_row(row,training_ref)
        if row.status=='complete':self.prune_training(identifier)
        return row

    def save_row(self, row, training_ref=None):
        with self.lock:
            path=self.formal_root/'candidates'/row.trajectory_id/'trajectory.json'
            io.write(path,record(row),replace=path.exists());self.results[row.trajectory_id]=row
            block=self.state['categories'][self.category]['trajectories']
            block[row.trajectory_id]=dict(result=io.binding(path),training=training_ref or block[row.trajectory_id]['training'])
            self.persist()

    def prune_training(self, identifier):
        row=self.results[identifier]
        require(row.status in ('complete','screened250'),'A pending pause must keep exact optimizer/RNG state.')
        with self.lock:
            for node in self.training_reports[identifier]['nodes']:
                for name in ('optimizer.pt','scheduler.pt','rng_state.pth','training_model_state.pt'):
                    if name=='training_model_state.pt' and node['step']==row.best_node.step:continue
                    path=Path(node['directory'])/name
                    if path.is_file():
                        require(path.resolve().is_relative_to(self.formal_root.resolve()) and not path.is_symlink(),'Cleanup escaped this batch.')
                        entry=dict(path=str(path),bytes=path.stat().st_size,reason='Completed or screened trajectory, no continuation; retain best master and metrics.',status='planned')
                        self.state.setdefault('cleanup',[]).append(entry);self.persist();path.unlink();entry['status']='completed'
            self.persist()

    def default(self, candidate):
        output=self.formal_root/'candidates'/candidate/'raw_source'
        ref=self.processes.run(candidate+'/raw_source','worker',dict(self.common,operation='refit',source_candidate_id=candidate,output=str(output)),output_file=output/'live_raw.json')
        raw=io.read(io.bound_path(ref))
        require(raw['formal_eligible'] and raw['source_parent_uuid']==self.parent['uuid'],'Later VAE fit used a different true parent.')
        context=dict(raw_ref=ref,raw=raw,raw_id=raw['round_base_uuid'],candidate=candidate,recipe_id=digest(raw['recipe']))
        if self.needs_technical and candidate==self.handoff['categories'][self.category]['top2'][0]:
            self.verify_changed_GPU_paths(context)
            self.technical_gate.set()
        while not self.technical_gate.wait(.2):
            require(not self.processes.cancelled.is_set(),'A required new GPU path failed.')
        settings=normalize(RecoverySettings(),**owners(raw,self.inventory))
        with ThreadPoolExecutor(max_workers=1) as pool:
            context['raw_score_future']=pool.submit(self.node,context,settings,'raw0:'+context['raw_id'],0)
            row=self.trajectory(context,settings);context['raw_score_future'].result()
        self.contexts[context['raw_id']]=context
        with self.lock:
            self.state['categories'][self.category]['raws'][candidate]=dict(raw=ref,recipe=raw['recipe'],zero_score=self.deployments[context['raw_id']]['score'])
            self.persist()
        return row

    def verify_changed_GPU_paths(self, context):
        """Two real updates of the new scopes/final no-LoRA path, never a formal seed."""
        folder=self.formal_root/'technical'/self.category
        technical_raw=dict(context['raw'],phase='technical_validation',formal_eligible=False,
            validation_only_origin=context['raw_ref'])
        io.write(folder/'raw.json',technical_raw);raw_ref=io.binding(folder/'raw.json')
        inventory=owners(context['raw'],self.inventory)
        configurations=[normalize(RecoverySettings(groups='DPL',decoder_scope='current' if inventory['future'] else 'prefix',protection_scope='prefix',
            lora_rank=48,lora_alpha=192,bound_optimizer='muon',lora_optimizer='muon'),**inventory)]
        if inventory['future']:
            configurations.append(replace(configurations[0],decoder_scope='prefix',protection_scope='current'))
        tasks={}
        for index in range(len(configurations)):
            prefix=f'technical/{self.category}/{index}'
            tasks[prefix+'/train']=dict(component='technical',seconds=1800)
            tasks[prefix+'/export']=dict(component='technical',seconds=900)
        self.budget.reserve(tasks,counts={})
        def one(index, settings):
            prefix=f'technical/{self.category}/{index}';output=folder/str(index)
            common=dict(self.common,phase='technical_validation',source_candidate_id=context['candidate'],live_raw=raw_ref,
                recipe_id=context['recipe_id'],trajectory_id=prefix,settings=asdict(settings))
            training_ref=self.processes.run(prefix+'/train','worker',dict(common,operation='train',technical_stop_after=2,
                technical_peak_proof=True,output=str(output/'train')),output_file=output/'train/training_result.json')
            training=io.read(io.bound_path(training_ref));node=Path(training['nodes'][-1]['directory'])
            require(training['actual']['actual_steps']==2 and training['peak_proof']['input_shape']==[4,1024]
                and training['peak_proof']['finite_gradients'],'Changed path lacks actual formal-shape updates.')
            report_ref=self.processes.run(prefix+'/export','worker',dict(common,operation='export',node_directory=io.binding(node/'checkpoint_meta.json'),
                output=str(output/'export'),destination=str(output/'deployment')),output_file=output/'export/deployment_report.json')
            report=io.read(io.bound_path(report_ref))
            require(report['fresh_forward_evidence']['independent_process'] and report['fresh_forward_evidence']['state_exact'],
                'New scoped/final native export failed actual independent reload.')
            return dict(settings=asdict(settings),training=training_ref,export=report_ref)
        receipts=self.parallel([lambda i=i,s=s:one(i,s) for i,s in enumerate(configurations)])
        io.write(folder/'validation.json',dict(status='actual_GPU_verified',formal_seed=False,receipts=receipts))
        self.state.setdefault('remaining_technical',{})[self.category]=io.binding(folder/'validation.json');self.persist()

    def search_category(self):
        entries=self.handoff['categories'][self.category]['top2'];tasks={}
        for candidate in entries:
            tasks[candidate+'/raw_source']=dict(component='fit',seconds=self.auth['task_seconds']['fit_by_category'][self.category])
            tasks[candidate+'/raw/export']=dict(component='export',seconds=900)
            tasks[candidate+'/raw/full8']=dict(component='evaluation',seconds=900)
        self.budget.reserve(tasks,counts=dict(raws=len(entries),raw_full8=len(entries)))
        defaults=self.parallel([lambda c=c:self.default(c) for c in entries]);winner,_=compare(defaults)[0]
        block=self.state['categories'][self.category];block['default_winner_raw']=winner.raw_id
        context=self.contexts[winner.raw_id];base=winner.settings;inventory=owners(context['raw'],self.inventory)
        for parameter in ROUNDS:
            row=search.begin_round(self.state,category=self.category,parameter=parameter,base=base,raw_id=winner.raw_id,
                inventory=inventory,data_identity=self.manifest['data_identity'],code_identity=self.manifest['code_identity'],completed=self.results)
            reference=trajectory_id(context['raw_id'],base,self.manifest['data_identity'],self.manifest['code_identity'])
            fixed=self.fixed.get(reference) if parameter in ('decoder_lr','protection_lr','lora_lr') else None
            self.persist()
            new=[c for c in row['candidates'] if not c['reuse']]
            self.parallel([lambda c=c:self.trajectory(context,RecoverySettings(**c['settings']),fixed,stop=c['first_stop']) for c in new])
            current=lambda:{c['trajectory_id']:self.results[c['trajectory_id']] for c in row['candidates']}
            chosen=search.finish_round(self.state,self.category,current())
            if isinstance(chosen,dict):
                survivors=chosen['continue_to500'];self.persist()
                self.parallel([lambda i=i:self.trajectory(context,self.results[i].settings,self.fixed[i],resumed=True) for i in survivors])
            if row['short_screen']:
                for candidate in row['candidates']:
                    i=candidate['trajectory_id']
                    if self.results[i].status=='pending250':
                        self.save_row(replace(self.results[i],status='screened250'));self.prune_training(i)
                if row['status']=='pending':chosen=search.finish_round(self.state,self.category,current())
            if chosen is not None:base=chosen
            self.persist()
        check=long_check(list(self.results.values()),context['recipe_id'],self.state['long_categories'])
        if check is not None:
            self.state['long_categories'].append(self.category);self.persist()
            self.trajectory(context,replace(check.settings,schedule_max=1000),self.fixed[check.trajectory_id])
        return select_category(list(self.results.values()))

    def selected_record(self, chosen, certificate):
        return asdict(SelectedCategory(self.category,chosen.recipe_id,chosen.candidate_id,
            normalize(RecoverySettings(),**owners(self.contexts[chosen.raw_id]['raw'],self.inventory)) if certificate['selected_step']==0 else chosen.settings,
            certificate['selected_step'],certificate['selected_trajectory_id'],certificate['selected_node_id'],chosen.raw_id,
            recovery_seeds(self.inventory['model_identity'],self.category,self.handoff['run_identity']['search_seed']),'formal'))

    def finalize_category(self, chosen, certificate):
        deployment=self.deployments[certificate['selected_node_id']]
        boundary=complete_boundary_reference(export_report=io.read(io.bound_path(deployment['report'])),inventory=self.inventory,full8_score=read_score(deployment['score']))
        selected=self.selected_record(chosen,certificate);self.commit_chain(selected,certificate,boundary)
        folder=self.formal_root/'boundaries'/self.category
        io.write(folder/'boundary.json',boundary)
        io.write(folder/'selected_recipe.json',selected_recipe_record(selected,certificate,self.contexts[chosen.raw_id]['raw']['recipe'],
            None if certificate['selected_step']==0 else self.training_reports[chosen.trajectory_id]['optimizer'],deployment))
        self.state['categories'][self.category].update(status='completed',selected=selected,certificate=certificate,
            boundary=io.binding(folder/'boundary.json'),selected_recipe=io.binding(folder/'selected_recipe.json'))
        self.parent_ref=io.binding(folder/'boundary.json');self.parent=boundary
        self.state.update(boundary=boundary,chain=self.chain.boundaries);self.persist()

    def run(self):
        from .terminal import choose_terminal, replay
        for category in self.handoff['frozen_order'][len(self.chain.boundaries):]:
            self.setup_category(category);chosen,certificate=self.search_category()
            if category==self.handoff['frozen_order'][-1]:
                chosen,certificate=choose_terminal(self,terminal_endpoints(list(self.results.values())))
            self.finalize_category(chosen,certificate)
        self.state.update(status='all_categories_complete_replay_pending',actual_bits_accepted=self.parent['actual_ledger']['final_budget_passed'])
        self.persist();replay(self)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--authorization',required=True);args=parser.parse_args()
    run=RemainingBatch(args.authorization)
    try:run.run()
    except BaseException:
        run.processes.cancel();run.state['status']='failed_requires_review';run.persist();raise


if __name__=='__main__':main()

"""Same-artifact full-nine selection and frozen-recipe original-BF16 replay."""
import argparse
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import signal
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.live import materialize_bf16_root
from experiments.sc_cat_continue_cmcc_20261107 import longbench as native_lb
from experiments.sc_cat_continue_cmcc_20261107.final_score import summarize
from experiments.longbench_v1 import runner
from experiments.longbench_v1.common import sha256
from experiments.longbench_v1.protocol import EXPECTED_COUNTS
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from . import io
from .boundary import complete_boundary_reference
from .settings import RecoverySettings


def predict(request):
    io.validate_request(request)
    config=io.read(io.bound_path(request['config']))
    require(config['mode']=='full' and config['model']['sc_cat']['complete'],'Require a complete same-artifact LongBench configuration.')
    report=io.read(io.bound_path(config['model']['sc_cat']['deployment_report']))
    manifest=io.read(io.bound_path(request['manifest']))
    require(Path(report['path']).resolve().is_relative_to(Path(manifest['output_root']).resolve())
        and config['model']['checkpoint_id']==report['uuid'],'LongBench cannot substitute an old experiment artifact.')
    tasks=request['datasets'];output=Path(request['output'])
    require(not output.exists() and len(tasks)==len(set(tasks)),'Use a new complete-task partition.')
    runner.load_model=native_lb.load_model
    native=runner.run(io.bound_path(request['config']),output,datasets=tasks,max_runtime_seconds=request['deadline_epoch']-time.time()-20)
    require(native['status']=='partition_complete' and native['exit_code']==0 and native['already_present']==0
        and native['new_predictions']==sum(EXPECTED_COUNTS[t] for t in tasks),'Partial LongBench is not a complete result.')
    reports=list((output/'runs').glob('*.json'))
    require(len(reports)==1 and io.read(reports[0])==native,'Bind the actual unique partition execution.')
    result=dict(status='completed_same_artifact_LongBench_partition',exit_code=0,config=request['config'],datasets=tasks,
        output=str(output),native_report=io.binding(reports[0]),checkpoint_id=report['uuid'],
        prediction_sha256={str(output/'predictions'/(task+'.jsonl')):sha256(output/'predictions'/(task+'.jsonl')) for task in tasks})
    io.write(output/'partition_result.json',result)


def full_nine(run, deployment, label, *, replaying=False):
    output=run.formal_root/label
    config=native_lb.make_config(deployment['report'],run.manifest['inventory'],run.auth['longbench_archive'])
    config['source_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    io.write(output/'config.json',config);config_ref=io.binding(output/'config.json')
    partitions=run.auth['longbench_partitions']
    tasks={label+f'/partition{i}':dict(component='replay' if replaying else 'terminal_longbench',seconds=21600) for i in range(len(partitions))}
    run.budget.reserve(tasks,counts={'replay_longbench' if replaying else 'terminal_longbench':1})
    def one(i, tasks):
        folder=output/f'partition{i}'
        return run.processes.run(label+f'/partition{i}','terminal',dict(run.common,operation='longbench',config=config_ref,
            datasets=tasks,output=str(folder)),output_file=folder/'partition_result.json')
    refs=run.parallel([lambda i=i,tasks=tasks:one(i,tasks) for i,tasks in enumerate(partitions)])
    with run.budget.cpu_window(label+'/merge'):
        merged=native_lb.merge(dict(config=config_ref,partitions=refs,output=str(output/'merged')))
        score=summarize(dict(deployment_report=deployment['report'],full8_score=deployment['score'],
            inventory_receipt=run.manifest['inventory'],baseline=run.manifest['baseline'],
            baseline_longbench_scores=run.auth['baseline_longbench_scores'],compressed_longbench_scores=merged['scores'],
            archive=run.auth['longbench_archive']))
        io.write(output/'nine_task.json',score)
    return io.binding(output/'nine_task.json')


def choose_terminal(run, endpoints):
    def one(pair):
        row,certificate=pair
        deployment=run.deployments[certificate['selected_node_id']]
        reference=full_nine(run,deployment,'terminal/'+row.candidate_id)
        score=io.read(io.bound_path(reference))
        exact=score['compressed_nine_exact'];value=Fraction(exact['numerator'],exact['denominator'])
        require(score['budget_passed'] and score['exit_code']==0,'Final endpoint lacks complete quality/bit evidence.')
        return dict(candidate_id=row.candidate_id,trajectory_id=certificate['selected_trajectory_id'],certificate=certificate,
            nine_task=reference,nine_score=str(value),logical_bits=row.logical_bits,packed_bits=row.packed_bits,
            checkpoint_id=score['checkpoint_id']),row
    evaluated=run.parallel([lambda p=p:one(p) for p in endpoints])
    chosen,row=min(evaluated,key=lambda pair:(-Fraction(pair[0]['nine_score']),pair[0]['logical_bits'],pair[0]['packed_bits'],pair[0]['candidate_id'],pair[0]['trajectory_id']))
    selection=dict(selected=chosen,compared=[v for v,_ in evaluated],rule='unrounded_S9_then_logical_packed_candidate_trajectory',deliver_existing_artifact=True)
    io.write(run.formal_root/'terminal_selection.json',selection)
    run.state.update(final_nine_complete=True,terminal_selection=io.binding(run.formal_root/'terminal_selection.json'))
    run.persist()
    return row,chosen['certificate']


def replay(run):
    categories=[]
    for category in run.handoff['frozen_order']:
        block=run.state['categories'][category];selected=block['selected']
        categories.append(dict(category_id=category,source_candidate_id=selected['source_candidate_id'],selected=selected,
            resolved_recipe=run.handoff['candidates'][selected['source_candidate_id']]['recipe'],search_boundary=block['boundary']))
    frozen=dict(schema='sc_cat_new_fixed_recipe_replay_v1',run_id=run.manifest['run_id'],original_BF16_root=run.handoff['parent'],
        frozen_order=run.handoff['frozen_order'],categories=categories,model_identity=run.manifest['model_identity'],
        data_identity=run.manifest['data_identity'],schedule_policy='original_schedule_max_stop_at_selected_step',
        repeat_search=False,repeat_terminal_pair=False,terminal_selection=run.state['terminal_selection'])
    root=run.formal_root/'replay';io.write(root/'frozen_recipe.json',frozen);frozen_ref=io.binding(root/'frozen_recipe.json')
    run.state.update(status='fixed_recipe_BF16_replay_running',replay_manifest=frozen_ref,replay_categories={});run.persist()
    with run.budget.cpu_window('replay/BF16_root'):
        inventory,_,_,tokenizer=io.load_inputs(run.common,with_sources=False)
        parent=materialize_bf16_root(inventory=inventory,frozen_order=frozen['frozen_order'],destination=root/'BF16_root',
            attention_implementation='sdpa',tokenizer=tokenizer,phase='formal')
        io.write(root/'BF16_root.json',parent);parent_ref=io.binding(root/'BF16_root.json')
    last_deployment=None
    for item in categories:
        category=item['category_id'];selected=item['selected'];settings=RecoverySettings(**selected['settings']);step=selected['selected_step']
        folder=root/category;prefix='replay/'+category
        seconds=run.auth['task_seconds'];train_seconds=0 if step==0 else 900+seconds['replay_seconds_per_update_by_category'][category]*step
        tasks={prefix+'/fit':dict(component='replay',seconds=seconds['fit_by_category'][category]),
            prefix+'/export':dict(component='replay',seconds=900),prefix+'/full8':dict(component='replay',seconds=900)}
        if step:tasks[prefix+'/train']=dict(component='replay',seconds=train_seconds)
        run.budget.reserve(tasks,counts=dict(replay_fits=1,replay_exports=1,replay_full8=1,replay_KD_updates=step))
        common=dict(run.common,category_id=category,source_candidate_id=item['source_candidate_id'],parent=parent_ref,replay_manifest=frozen_ref)
        raw_ref=run.processes.run(prefix+'/fit','worker',dict(common,operation='refit',output=str(folder/'raw_source')),output_file=folder/'raw_source/live_raw.json')
        raw=io.read(io.bound_path(raw_ref))
        require(raw['source_parent_uuid']==parent['uuid'] and raw['recipe']==item['resolved_recipe'],'Replay must fit its own parent and frozen recipe.')
        identifier=digest(dict(replay=frozen_ref,category=category,raw_id=raw['round_base_uuid']))
        base=dict(common,live_raw=raw_ref,settings=asdict(settings),recipe_id=selected['recipe_id'],trajectory_id=identifier)
        training_ref=None
        if step:
            training_ref=run.processes.run(prefix+'/train','worker',dict(base,operation='train',replay_selected_step=step,
                output=str(folder/'train')),output_file=folder/'train/training_result.json')
            training=io.read(io.bound_path(training_ref))
            require(training['actual']['actual_steps']==step and training['actual']['schedule_max']==settings.schedule_max
                and training['actual']['provenance']['semantic_seeds']==selected['seed_identity'],'Fixed replay changed steps/schedule/seeds.')
            base['node_directory']=io.binding(Path(training['nodes'][-1]['directory'])/'checkpoint_meta.json')
        report_ref=run.processes.run(prefix+'/export','worker',dict(base,operation='export' if step else 'export_zero',
            output=str(folder/'export'),destination=str(folder/'deployment')),output_file=folder/'export/deployment_report.json')
        score_ref=run.processes.run(prefix+'/full8','evaluate',dict(common,deployment_report=report_ref,output=str(folder/'full8')),output_file=folder/'full8/score.json')
        last_deployment=dict(report=report_ref,score=score_ref)
        parent=complete_boundary_reference(export_report=io.read(io.bound_path(report_ref)),inventory=run.inventory,full8_score=read_score(score_ref))
        require(parent['completed_prefix']==frozen['frozen_order'][:frozen['frozen_order'].index(category)+1],'Replay skipped or reordered a category.')
        io.write(folder/'boundary.json',parent);parent_ref=io.binding(folder/'boundary.json')
        run.state['replay_categories'][category]=dict(boundary=parent_ref,raw=raw_ref,training=training_ref,deployment=last_deployment,
            schedule_max=settings.schedule_max,selected_step=step,search_selected=selected)
        run.persist()
    replay_nine=full_nine(run,last_deployment,'replay/final_longbench',replaying=True)
    product=io.read(io.bound_path(io.read(io.bound_path(run.state['terminal_selection']))['selected']['nine_task']))
    repeated=io.read(io.bound_path(replay_nine))
    comparison=dict(status='completed_requires_numerical_review',search_nine=product['compressed_nine_task_score'],
        replay_nine=repeated['compressed_nine_task_score'],difference_pp=repeated['compressed_nine_task_score']-product['compressed_nine_task_score'],
        logical_bits_equal=product['actual_ledger']['logical_bits']==repeated['actual_ledger']['logical_bits'],
        packed_bits_equal=product['actual_ledger']['packed_bits']==repeated['actual_ledger']['packed_bits'],
        replay_score=replay_nine,fixed_recipe=frozen_ref,numerical_reproduction_accepted=False)
    io.write(root/'comparison.json',comparison)
    run.state.update(status='all_categories_and_replay_complete_requires_acceptance',BF16_replay_complete=True,
        BF16_replay_accepted=False,replay_comparison=io.binding(root/'comparison.json'))
    run.persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);args=parser.parse_args()
    request=io.read(args.request)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Allocated LongBench window expired.')))
    signal.setitimer(signal.ITIMER_REAL,max(.1,request['deadline_epoch']-time.time()))
    torch.set_num_threads(4);predict(request)


if __name__=='__main__':main()

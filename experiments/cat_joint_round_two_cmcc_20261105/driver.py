"""One-shot R1 barrier -> confirmed R2 neighbors -> actual all-proposal cap."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_joint_export_cmcc_20261104 import producer as p
from experiments.cat_offline_execution_cmcc_20261104.protocol import read_native_eight
from experiments.cat_offline_execution_cmcc_20261104.selection import freeze_joint,select_joint
from .rules import enumerate_pool,nominations,rank_actual,allocate_Nc

HERE=Path(__file__).resolve().parent


def own_sources():
    return {str(path.resolve()):p.file_sha(path) for path in HERE.glob('*.py')}


def bound(reference):
    require(p.binding(reference['path'])==reference,'Bound actual source changed: '+reference['path'])
    return p.read(reference['path'])


def context(r1):
    batch=p.read(r1/'prepared_batch_UNAPPROVED.json');first=bound(batch['arms'][0]['proposal'])
    draft=bound(batch['source_draft']);coverage=bound(batch['source_coverage'])
    require({row['proposal_id'] for row in batch['arms']}=={row['proposal_id'] for row in draft['proposals']}
        and len(batch['arms'])==len(draft['proposals']),'The announced R1 set differs from its original machine nominations.')
    require(coverage['status_counts']=={'complete':coverage['expected_profile_count']},'All original category/rate profile sources must be complete.')
    inventory=bound(first['inventory_receipt'])['inventory'];bound(first['baseline_receipt'])
    baseline=p.load_baseline(first['baseline_receipt']['path'])
    require(coverage['common_run_identity']==first['run_identity'] and coverage['common_protocol_identity']==first['protocol_identity'], 'Original profile run/protocol differs.')
    pool,common=enumerate_pool(inventory,coverage['profiles'],baseline['mean_percent'])
    require(common['working_limit_bits']==draft['working_limit_bits'] and common['hard_limit_bits']==draft['hard_limit_bits']
        and first['limits']['max_logical_bits']==common['common_limit_bits'] and common['common_set_count']==draft['common_working_set_count']
        and common['hard_feasible_count']==draft['hard_feasible_count'],'The two rounds must use exactly the original common cost set.')
    return batch,first,coverage,inventory,pool,common


def measured(arms):
    """Real producer/run/exit and existing full8 reader; never accept supplied S8."""
    pending=[];rows=[]
    # Terminal barrier first: no half-written result or early finisher is read.
    for arm in arms:
        proposal=bound(arm['proposal']);root=Path(proposal['output_root']);run_path=root/'run.json'
        shell_exit=root.parent/'controller.exit'
        if not shell_exit.is_file() or shell_exit.read_text().strip()!='0' or not run_path.is_file():
            pending.append(dict(label=arm['label'],reason='controller_not_successfully_terminal'));continue
        run=p.read(run_path)
        if run.get('status')!='completed_explicit_joint_full8' or run.get('exit_code')!=0:
            pending.append(dict(label=arm['label'],reason=run.get('status','unknown')))
    if pending:return dict(status='pending',pending=pending,results=[],winner=None)
    contexts=set();announced={}
    for arm in arms:
        proposal=bound(arm['proposal']);root=Path(proposal['output_root']);run_path=root/'run.json';run=p.read(run_path)
        contexts.add(digest({key:proposal[key] for key in ('run_identity','inventory_receipt','baseline_receipt','protocol_identity')}))
        result_path=root/'joint_result.json';result=p.read(result_path);exit_record=p.read(root/'exit.json')
        auth_path=root.parent/'actual_authorization.json';auth=p.read(auth_path);request=p.read(root/'execution_request.json')
        p.authorization_contract(proposal,arm['proposal'],auth,now=request['started_epoch'])
        require(auth['authorized_start_epoch']<=request['started_epoch']<=exit_record['started_epoch']<=exit_record['finished_epoch']<=request['export_deadline_epoch']
            and request['export_deadline_epoch']<=min(request['absolute_deadline_epoch'],request['started_epoch']+proposal['limits']['export_seconds'])
            and request['absolute_deadline_epoch']<=auth['absolute_deadline_epoch'] and run['finished_epoch']<=auth['absolute_deadline_epoch'], 'Actual request/export/complete windows exceed the original authorization.')
        batch_reference=auth['batch_source'];announced.setdefault(digest(batch_reference),dict(reference=batch_reference,arms=[]))['arms'].append(arm)
        require(run['proposal']==arm['proposal']==result['source_proposal'] and run['authorization']==p.binding(auth_path)==result['source_authorization']
            and auth['user_approved'] is True and auth['execution_scope']==request['execution_scope']=='export_and_full8'
            and auth['proposal_sha256']==arm['proposal']['sha256'] and auth['proposal_id']==proposal['proposal_id']
            and auth['output_root']==proposal['output_root'] and auth['limits']==proposal['limits'], 'Actual authorization/producer/proposal binding differs.')
        require(result['schema']==p.SCHEMA and result['status']=='exported_actual_joint_awaiting_full8' and result['formal_eligible']
            and result['phase']=='formal' and result['proposal_id']==proposal['proposal_id'] and result['banks']==proposal['banks']
            and result['run_identity']==proposal['run_identity'] and result['source_execution']==p.binding(root/'execution_request.json')
            and request['proposal']==arm['proposal'] and request['authorization']==p.binding(auth_path)
            and exit_record['exit_code']==0 and exit_record['request_sha256']==p.file_sha(root/'execution_request.json')
            and exit_record['command']==[sys.executable,'-B','-u','-m','experiments.cat_joint_export_cmcc_20261104.producer','--component','export-worker','--request',str(root/'execution_request.json')], 'Actual formal worker/exit/source scope differs.')
        p.verify_files(result['source_sha256']);fresh=result['fresh_evidence'];union=result['union_evidence']
        require(union['actual_complete_bank_union_verified'] and union['non_target_before_digest']==union['non_target_after_digest']
            and fresh['state_exact'] and fresh['gpu_validation'] and fresh['forward_executed'] and not fresh['CPU_mechanism_only']
            and fresh['same_input_same_BF16_path']['passed'],'Actual joint export/fresh/complete complement evidence is missing.')
        endpoint=result['endpoint'];ledger=endpoint['ledger']
        require(endpoint['formal_eligible'] and endpoint['phase']=='formal' and ledger['source']['kind']=='actual_native_v6_tensor_storage'
            and ledger['source']['checkpoint_id']==endpoint['checkpoint_id']==run['checkpoint_id']==fresh['checkpoint_id']
            and ledger['source']['metadata_sha256']==endpoint['metadata_sha256']
            and ledger['source']['state_file_sha256']==endpoint['state_sha256']==fresh['weight_digest'], 'Actual whole-model UUID/ledger/fresh bytes differ.')
        p.check_ledger(ledger,proposal['source_class_fees'],proposal['limits'],formal=True)
        full8=root/'full8';score,receipt=read_native_eight(summary_path=full8/'full8_summary.json',evaluation_path=full8/'evaluation.json')
        plan=p.read(full8/'plan.json')
        require(plan['joint_source_proposal_id']==proposal['proposal_id'] and plan['authorization']['physical_GPU']==auth['physical_GPU']
            and plan['authorization']['absolute_deadline_epoch']<=request['absolute_deadline_epoch'], 'Actual full8 resource/proposal/deadline differs from its own chain.')
        require(receipt['endpoint']==endpoint and receipt['protocol_identity']==proposal['protocol_identity'] and receipt['complete']
            and receipt['formal_eligible'] and receipt['phase']=='formal' and math.isclose(float(score.mean)*100,receipt['mean_percent'],rel_tol=0,abs_tol=1e-12), 'Actual joint full8 differs from its own measured deployment.')
        final=p.read(root/'joint_full8_result.json')
        selection_proposal=dict(proposal_id=proposal['proposal_id'],rate_vector=proposal['rate_vector'],banks=proposal['banks'],
            full8=dict(summary=str(full8/'full8_summary.json'),evaluation=str(full8/'evaluation.json')))
        require(final['status']=='completed_explicit_joint_full8' and final['endpoint']==endpoint and final['complete_full8_evaluated']
            and final['full8_receipt']==p.binding(full8/'full8_summary.json') and math.isclose(final['mean_percent'],receipt['mean_percent'],rel_tol=0,abs_tol=1e-12)
            and final['selection_proposal']==selection_proposal, 'Producer whole-eight completion receipt differs.')
        rows.append(dict(label=arm['label'],proposal_id=proposal['proposal_id'],rate_vector=proposal['rate_vector'],full8_mean=receipt['mean_percent'],
            logical_bits=ledger['logical_bits'],packed_bits=ledger['packed_bits'],ledger=ledger,proposal=arm['proposal'],
            selection_proposal=selection_proposal,ordered_inputs=receipt['ordered_sample_identity_evidence']['ordered_input_identity'],
            sources=dict(run=p.binding(run_path),producer=p.binding(result_path),full8=p.binding(full8/'full8_summary.json'),evaluation=p.binding(full8/'evaluation.json'))))
    require(len(contexts)==1 and len({row['ordered_inputs'] for row in rows})==1,'Joint proposals used different actual source contexts or ordered full8 inputs.')
    for group in announced.values():
        original=bound(group['reference'])
        require({digest(arm['proposal']) for arm in group['arms']}=={digest(arm['proposal']) for arm in original['arms']},
            'The actual authorization names additional or missing announced proposals; cannot silently drop a failed endpoint.')
    ranked=rank_actual(rows)
    return dict(status='complete',results=ranked,winner=ranked[0])


def prepare(r1,output):
    batch,first,coverage,inventory,pool,common=context(r1)
    actual=measured(batch['arms'])
    if actual['status']!='complete':return actual
    require(not output.exists(),'Never replace an already frozen R2 set.')
    chosen=nominations(pool,actual['winner'],actual['results']);output.mkdir(parents=True)
    p.write(output/'R1_actual_selection.json',dict(actual,common=common,source_R1_batch=p.binding(r1/'prepared_batch_UNAPPROVED.json')))
    arms=[]
    for index,row in enumerate(chosen):
        label='R2_'+str(index+1);directory=output/label;directory.mkdir()
        profiles={category:dict(canonical=source['sources']['canonical'],checkpoint_id=source['endpoint']['checkpoint_id'],
            full8=dict(summary=source['sources']['summary'],evaluation=source['sources']['evaluation'])) for category,source in row['profiles'].items()}
        spec=dict(inventory_receipt=first['inventory_receipt'],baseline_receipt=first['baseline_receipt'],run_identity=first['run_identity'],profiles=profiles,
            rate_vector=row['rate_vector'],output_root=str(directory/'execution'),allowed_physical_GPUs=[0,1,2,3,4,5],
            limits=dict(export_seconds=900,full8_seconds=900,total_seconds=1800,max_logical_bits=common['common_limit_bits']))
        proposal=p.prepare(spec);require(proposal['proposal_id']==row['proposal_id'],'Actual prepared R2 bank identities differ from its frozen common vector.')
        p.write(directory/'explicit_spec.json',spec);p.write(directory/'proposal_UNAPPROVED.json',proposal)
        arms.append(dict(label=label,suggested_physical_GPU=index,proposal=p.binding(directory/'proposal_UNAPPROVED.json'),proposal_id=row['proposal_id'],
            rate_vector=row['rate_vector'],logical_bits=row['logical_bits'],packed_bits=row['packed_bits'],signed_profile_loss_exact=str(row['signed_profile_loss'])))
    result=dict(schema='confirmed_joint_R2_frozen_neighbors_v1',status='prepared_R2_requires_actual_runtime_authorization',source_R1_batch=p.binding(r1/'prepared_batch_UNAPPROVED.json'),
        source_R1_selection=p.binding(output/'R1_actual_selection.json'),source_profile_coverage=batch['source_coverage'],common=common,
        parent_proposal_id=actual['winner']['proposal_id'],arms=arms,maximum_new_GPU_hours=len(arms)*.5,source_sha256=own_sources(),
        generation_rule='Two distinct categories,one+0.5 andone-0.5; original common set; exclude allR1; signed profile loss thenactualL/P/stablevector; atmost4; noR3',
        CPU_or_technical_scores_used=False,formal_eligible=False)
    p.write(output/'R2_prepared.json',result)
    return result


def finalize(r1,output):
    r2=p.read(output/'R2_prepared.json');p.verify_files(r2['source_sha256']);require(own_sources()==r2['source_sha256'],'R2 source set changed after nominations.')
    batch,first,coverage,inventory,pool,common=context(r1);bound(r2['source_R1_selection'])
    require(r2['common']==common and r2['source_R1_batch']==p.binding(r1/'prepared_batch_UNAPPROVED.json'),'R2 changed its original announced source/common set.')
    all_arms=batch['arms']+r2['arms'];actual=measured(all_arms)
    if actual['status']!='complete':return actual
    r1_ids={row['proposal_id'] for row in batch['arms']};r1_actual=rank_actual([row for row in actual['results'] if row['proposal_id'] in r1_ids])
    expected=nominations(pool,r1_actual[0],r1_actual)
    require(r2['parent_proposal_id']==r1_actual[0]['proposal_id'] and [row['proposal_id'] for row in r2['arms']]==[row['proposal_id'] for row in expected],
        'Actual R2 differs from the original full R1 winner/common-set one-up-one-down nominations.')
    directory=output/'final_cap';require(not directory.exists(),'Do not overwrite a cap certificate.');directory.mkdir()
    proposals=[row['selection_proposal'] for row in actual['results']]
    frozen=freeze_joint(inventory_receipt=first['inventory_receipt']['path'],baseline_path=first['baseline_receipt']['path'],run_identity=first['run_identity'],proposals=proposals)
    p.write(directory/'frozen_all_joint_proposals.json',frozen)
    selection=select_joint(frozen);require(selection['status']=='budget_allocation_pending','All actual joint proposals must be complete before Nc allocation.')
    require(selection['winner_proposal_id']==actual['winner']['proposal_id'],'Small producer/readout bridge and actual state/ledger consumer selected different winners.')
    p.write(directory/'selection_without_allocation.json',selection)
    caps,evidence=allocate_Nc(inventory,selection['winner']['ledger'])
    allocation=dict(authorization_reference='2026-11-05 user instruction to complete remaining comparisons; confirmed main protocol section4.6 Nc allocation',
        categories=caps,method=evidence,source_actual_selection=p.binding(directory/'selection_without_allocation.json'),
        common_working_set=common,source_R1_selection=r2['source_R1_selection'],source_R2_manifest=p.binding(output/'R2_prepared.json'))
    p.write(directory/'Nc_allocation.json',allocation)
    with_caps=freeze_joint(inventory_receipt=first['inventory_receipt']['path'],baseline_path=first['baseline_receipt']['path'],run_identity=first['run_identity'],
        proposals=proposals,cap_allocation=allocation)
    p.write(directory/'frozen_all_with_allocation.json',with_caps)
    certificate=select_joint(with_caps)
    require(certificate['status']=='complete' and certificate['formal_eligible'] and certificate['winner_proposal_id']==selection['winner_proposal_id']
        and certificate['sum_cap_bits']==certificate['hard_cap_bits'],'The actual all-source cap certificate failed final closure.')
    p.write(directory/'cap_certificate.json',certificate)
    return dict(status='complete',cap_certificate=p.binding(directory/'cap_certificate.json'),winner_proposal_id=certificate['winner_proposal_id'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--component',choices=('prepare','finalize'),required=True)
    parser.add_argument('--r1-batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run one-shot joint selection/preparation in caller bitvae with CUDA hidden.')
    began=time.monotonic();result=(prepare if args.component=='prepare' else finalize)(args.r1_batch.resolve(),args.output.resolve())
    print(dict(status=result['status'],elapsed_seconds=time.monotonic()-began,**{key:result[key] for key in ('pending','parent_proposal_id','maximum_new_GPU_hours','cap_certificate') if key in result}),flush=True)


if __name__=='__main__':main()

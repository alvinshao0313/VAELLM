"""Targeted real-source CPU validation; neither generates HPO workers nor uses CUDA."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

from . import bindings as b
from .run import advance, select_round
from .report import stage_amplitudes
from experiments.cat_auto_protocol_cmcc_20261104.search import resource_path_signature


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    b.require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use caller bitvae with CUDA hidden.')
    import torch
    torch.set_num_threads(2); started=time.time()
    original,inventory,identity=b.prior.prior.common()
    rules=dict(confirmed=True,authorization_reference='CPU mathematical validation only; no GPU authorization',order=list(b.PARAMETERS),
        maximum_distinct_per_category=19,extra_layerwise_or_BSQ_or_revisit=False,gpus=list(range(6)),total_GPU_hours=292.25,
        formal_base_seconds={'8':7200,'16':5400,'32':3600,'64':3600,'128':3600},full8_seconds=900,technical_base_seconds=900,technical_large_seconds=1800)
    b.validate_rules(rules); mathematical=dict(rules=rules,caps={category:{} for category in inventory['category_ids']})
    for field,value in [('gpus',[True]),('gpus',[6]),('total_GPU_hours',float('inf')),('total_GPU_hours',float('nan')),('total_GPU_hours',True)]:
        invalid=dict(rules,**{field:value})
        try:b.validate_rules(invalid)
        except ValueError:pass
        else:raise AssertionError('Non-finite/bool/forbidden resource authorization accepted.')
    b.require(b.worst_case_GPU_hours(mathematical)==292.25,'Per-slot D/width budget must close to292.25GPUh, without blanket worst-case multiplication.')
    probes=[]
    for dim in (8,16,32,64,128):
        settings=dict(b.DEFAULT_SETTINGS,dimension=dim)
        recipe=b.recipe(inventory,'q_proj','5/2',settings)
        b.require(recipe['K']=={8:10,16:20,32:40,64:80,128:160}[dim] and recipe['B']*dim==262144 and recipe['S']==2,'Actual integer geometry or scalar exposure changed.')
        quote=b.component_plan(inventory,recipe); b.require(quote['capability']['constructible'],'Real geometry/factory rejected CPU probe.')
        modified=b.recipe(inventory,'q_proj','5/2',dict(settings,multiplier=4))
        b.require(resource_path_signature(recipe)==resource_path_signature(modified),'LR must reuse a real same-resource-path technical receipt.')
        probes.append(dict(D=dim,K=recipe['K'],B=recipe['B'],counts=sorted(set(recipe['protection']['counts'].values())),
            windows=b.phase_windows(mathematical,recipe),logical_bits=quote['logical_bits'],packed_bits=quote['packed_bits']))
    for level in ('128','256','full'):
        recipe=b.recipe(inventory,'q_proj','2',dict(b.DEFAULT_SETTINGS,rotation=level))
        b.require(set(recipe['protection']['counts'].values())=={0},'Nearest legal protection must be remapped to actual zero for these real Q geometries.')
    # Actual complete profile results supply selection evidence; no synthetic scores or artifacts.
    baseline=b.load_baseline(b.prior.BASELINE); profile=b.profile_paths()[0]; result=b.read(profile)
    context=dict(run_identity=identity,protocol_identity=baseline['protocol_identity'],caps={result['category_id']:dict(cap_bits=result['logical_bits'])})
    b.checked_result(profile,context,expected_recipe=result['recipe'])
    snapshot=b.coverage.Snapshot()
    expected,profile_context=b.coverage.expected_sources(snapshot,b.coverage.DEFAULT_UP,b.coverage.DEFAULT_EIGHT,b.coverage.DEFAULT_NIGHT)
    expected=next(row for row in expected if row['candidate_id']==result['candidate_id'])
    verified=b.coverage.verify_complete(snapshot,expected,profile_context)
    b.require(verified['full8']['mean_percent']==result['full8_mean'] and verified['actual_class_fee']['logical_bits']==result['logical_bits']
        and verified['endpoint']['checkpoint_id']==result['endpoint']['checkpoint_id'],'Actual small-source profile re-read failed.')
    amplitudes=stage_amplitudes(result)
    b.require(len(amplitudes)==2 and all(row['actual_steps']==10000 and row['first_actual_optimizer_group_LRs']==[.003]
        and row['initial_matrix_effective_LR']==.003 and row['kappa_calibration'] is None for row in amplitudes),
        'Actual complete profile fit_report must produce both recorded stage LR rows before final reporting.')
    reference=b.bound(profile); current=dict(result=reference,settings=deepcopy(b.DEFAULT_SETTINGS),candidate_id=result['candidate_id'])
    stage=dict(candidates=[dict(status='reused',result=reference,settings=deepcopy(b.DEFAULT_SETTINGS))])
    b.require(select_round(context,current,stage)==current,'Retained current evidence must survive exact same-recipe selection.')
    state=dict(status='running',next_parameter=0,current=deepcopy(current),known={result['candidate_id']:reference},
        rounds=[dict(status='running',parameter='transpose',candidates=[dict(status='failed',error='explicit mathematical dependency state')])])
    advance(context,result['category_id'],state)
    b.require(state['status']=='failed' and state['next_parameter']==0 and len(state['rounds'])==1,'An announced failure must block this class successors.')
    proof=b.proof_for(dict(context,inventory=b.bound(original['inventory_receipt'])),dict(candidate_id=result['candidate_id'],recipe=result['recipe']),args.output.parent/'no_candidates')
    b.require(proof is not None,'Existing real source9 model-path evidence must remain reusable after technical payload cleanup.')
    b.require(len(identity['code_identity']['source_sha256'])==9 and not torch.cuda.is_initialized(),'Preserve source9 identity and hidden CUDA.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    b.write_new(args.output,dict(status='CPU_actual_sources_geometry_LR_reuse_selection_failure_and_budget_verified',exit_code=0,
        elapsed_seconds=time.time()-started,source_sha256=b.pins(),fit_code_identity=identity['code_identity'],
        actual_geometry_probes=probes,budget_worst_case_GPU_hours=292.25,existing_technical_receipt=proof,
        actual_profile_stage_amplitudes=amplitudes,
        actual_profile_small_source_verification=verified,nonfinite_bool_and_protected_GPU_authorizations_rejected=True,
        CUDA_initialized=False,actual_HPO_candidate_sets_generated=False,GPU_launched=False,
        boundary='CPU real source/profile/factory paths only. New geometry/optimizer/resource paths still require actual technical2 before formal.'))
    print(dict(status='CPU_verified',elapsed_seconds=time.time()-started,budget_worst_case_GPU_hours=292.25),flush=True)


if __name__=='__main__':main()

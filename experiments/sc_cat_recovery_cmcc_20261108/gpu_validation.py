"""Approved model-scale checks; technical states never seed formal recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from . import io
from .identity import source_pins
from .processes import Budget, Processes, check_resources
from .settings import RecoverySettings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization',required=True)
    args=parser.parse_args()
    authorization=io.read(args.authorization)
    manifest=io.read(io.bound_path(authorization['manifest']))
    root=Path(authorization['technical_output'])
    require(root.resolve().is_relative_to(Path(manifest['output_root']).resolve()), 'Technical output must stay in this run.')
    require(not root.exists() and authorization.get('confirmed') is True and authorization.get('user_approved') is True
        and authorization['gpus']==[4,5,6,7] and authorization['categories']==manifest['frozen_order'][:1], 'Require explicit fresh first-class approval.')
    check_resources([4,5,6,7])
    handoff=io.read(io.bound_path(manifest['selected_settings']))
    candidate=handoff['categories'][manifest['frozen_order'][0]]['top2'][0]
    previous=[]
    prior_ref=authorization.get('prior_technical_budget')
    if prior_ref:
        prior=io.read(io.bound_path(prior_ref))
        require(all(t['status']!='running' for t in prior['tasks'].values()),'Do not edit or restart an active attempt.')
        previous=list(prior['technical_windows'])
        previous += [dict(started_epoch=t['started_epoch'],ended_epoch=t['ended_epoch'],gpu=t['gpu'],task=name,
            status=t['status'],source_budget=prior_ref) for name,t in prior['tasks'].items() if t['status'] in ('completed','failed')]
        require(abs(sum(w['ended_epoch']-w['started_epoch'] for w in previous)/3600-prior['total_actual_GPU_hours'])<1e-9,
            'Carry every successful/failed prior ownership window exactly once.')
    budget=Budget(root/'budget.json',authorization,previous,[prior_ref] if prior_ref else [])
    processes=Processes(root,budget,source_pins())
    tasks={'raw':1800,'largest2':1200,'P2':1200,'pause250':2700,'resume500':2700,
        'zero_export':900,'largest_export':900,'P_export':900,'resumed_export':900}
    if authorization.get('reuse_technical_raw'):
        tasks.pop('raw')
    budget.reserve({k:dict(component='technical',seconds=v) for k,v in tasks.items()},counts={})
    common=dict(manifest=authorization['manifest'],authorization=io.binding(args.authorization),
        source_manifest=handoff['source_manifest'],inventory_receipt=manifest['inventory'],data_plan_path=manifest['data_plan'],
        category_id=manifest['frozen_order'][0],phase='technical_validation',source_candidate_id=candidate,
        execution_policy=dict(train_device='cuda:0',attention_implementation='flash_attention_2',teacher_model_offload='none',
            teacher_output_offload='cpu',teacher_output_pin_memory=True,teacher_output_chunk_tokens=8,deterministic=False))
    references=[]
    def task(name,operation,extra,result):
        output=root/'artifacts'/name
        ref=processes.run(name,'worker',dict(common,operation=operation,output=str(output),**extra),output_file=output/result)
        references.append(ref)
        return io.read(io.bound_path(ref)),ref
    try:
        if authorization.get('reuse_technical_raw'):
            raw_ref=authorization['reuse_technical_raw']
            raw=io.read(io.bound_path(raw_ref))
            require(raw['phase']=='technical_validation' and raw['source_candidate_id']==candidate
                and raw['source_parent_uuid']==handoff['parent']['uuid'] and raw['fresh_evidence']['same_input_logits_exact'],
                'Reuse only the completed same-bank/parent technical raw; no recovery state is reused.')
            references.append(raw_ref)
        else:
            raw,raw_ref=task('raw','first_raw',{},'live_raw.json')
        settings=RecoverySettings(groups='DPL',lora_rank=48,lora_alpha=192,lora_dropout=.1,
            bound_optimizer='muon',lora_optimizer='muon')
        def train(name,config,**kwargs):
            return task(name,'train',dict(live_raw=raw_ref,settings=asdict(config),recipe_id=digest(raw['recipe']),
                trajectory_id='technical_'+name,**kwargs),'training_result.json')
        def export(name,config,training=None,zero=False):
            extra=dict(live_raw=raw_ref,settings=asdict(config),recipe_id=digest(raw['recipe']),trajectory_id='technical_'+name,
                destination=str(root/'artifacts'/name/'deployment'))
            if not zero:
                extra['node_directory']=io.binding(Path(training['nodes'][-1]['directory'])/'checkpoint_meta.json')
            return task(name,'export_zero' if zero else 'export',extra,'deployment_report.json')
        with ThreadPoolExecutor(max_workers=4) as pool:
            largest=pool.submit(train,'largest2',settings,technical_stop_after=2,technical_peak_proof=True)
            protection=pool.submit(train,'P2',replace(settings,groups='P'),technical_stop_after=2)
            pause=pool.submit(train,'pause250',settings,stop_at=250)
            zero=pool.submit(export,'zero_export',RecoverySettings(),zero=True)
            largest_report,_=largest.result()
            p_report,_=protection.result()
            largest_export=pool.submit(export,'largest_export',settings,largest_report)
            p_export=pool.submit(export,'P_export',replace(settings,groups='P'),p_report)
            paused,_=pause.result()
            resumed,_=train('resume500',settings,stop_at=500,resume_from=paused['nodes'][-1]['directory'])
            resumed_export=pool.submit(export,'resumed_export',settings,resumed)
            exports=[f.result()[0] for f in (zero,largest_export,p_export,resumed_export)]
        groups={g['group_name']:g for g in paused['optimizer']['groups']}
        require({k for k,g in groups.items() if g['kind']=='muon'}=={'decoder','lora','q','scale'}
            and all(groups[k]['muon_kappa']>0 for k in ('decoder','lora','q','scale')), 'Actual four-group calibration is missing.')
        before={g['group_name']:g['calibration'] for g in paused['optimizer']['groups']}
        require(before=={g['group_name']:g['calibration'] for g in resumed['optimizer']['groups']}, 'Resume changed kappa.')
        masters=torch.load(Path(paused['nodes'][-1]['directory'])/'training_model_state.pt',map_location='cpu',weights_only=True)
        q=[v for n,v in masters.items() if n.endswith('protected_code_coordinate')]
        scale=[v for n,v in masters.items() if n.endswith('protected_log_scale_delta')]
        require(len(q)==len(scale)==36 and all(torch.isfinite(v).all() for v in [*q,*scale])
            and any((v!=v.round()).any() for v in q) and any(v.ne(0).any() for v in scale), 'Both joint protection masters must actually update.')
        require({g['group_name'] for g in p_report['optimizer']['groups']}=={'q','scale'}, 'P-only unexpectedly trains decoder/LoRA.')
        require(largest_report['peak_proof']['input_shape']==[4,1024] and largest_report['peak_proof']['finite_gradients']
            and resumed['actual']['saved_state_restore_verified'] and resumed['actual']['actual_steps']==500,
            'Actual peak backward or complete resume proof is missing.')
        require(all(r['fresh_forward_evidence']['gpu_validation'] and r['fresh_forward_evidence']['state_exact']
            and r['fresh_forward_evidence']['independent_process'] for r in exports),'Native independent export/fresh failed.')
        checks=dict(formal_B4_length1024_backward=True,joint_q_scale_and_four_Muon_groups=True,
            rank48_native_export_and_independent_fresh=True,P_only_frozen_decoder_backward=True,
            native250_resume_to500_and_data_RNG=True,zero_raw_native_export=True)
        proof=dict(status='actual_GPU_verified',exit_code=0,manifest=authorization['manifest'],source_pins=source_pins(),
            checks=checks,receipts=references,ownership_windows=previous+[dict(started_epoch=t['started_epoch'],ended_epoch=t['ended_epoch'],
                gpu=t['gpu'],task=name) for name,t in budget.value['tasks'].items()],
            actual_GPUh=budget.value['total_actual_GPU_hours'],formal_trajectory_eligible=False,
            quality_verified=False,cleanup=[])
        io.write(root/'validation.json',proof)
        # All workers exited; no technical tensor is a formal input. Preserve
        # configs, native metadata, scalar audits and hashes, remove own payloads.
        targets=set()
        for metadata in (root/'artifacts').rglob('checkpoint_meta.json'):
            value=io.read(metadata)
            if value.get('state_dict_file'):
                targets.add(metadata.parent/value['state_dict_file'])
        for name in ('training_model_state.pt','optimizer.pt','scheduler.pt','rng_state.pth'):
            targets.update((root/'artifacts').rglob(name))
        for path in sorted(targets):
            require(path.resolve().is_relative_to((root/'artifacts').resolve()) and not path.is_symlink(), 'Technical cleanup escaped its own artifacts.')
            if path.is_file():
                entry=dict(path=str(path),bytes=path.stat().st_size,reason='Technical validation complete; forbidden as formal seed.',status='planned')
                proof['cleanup'].append(entry)
                io.write(root/'validation.json',proof,replace=True)
                path.unlink()
                entry['status']='completed'
        io.write(root/'validation.json',proof,replace=True)
        print('Actual GPU checks complete. Technical states removed; formal batch must start fresh.',flush=True)
    except BaseException:
        processes.cancel()
        raise


if __name__=='__main__':
    main()

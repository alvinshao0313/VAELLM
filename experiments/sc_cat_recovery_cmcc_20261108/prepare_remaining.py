"""Build a reviewable second-batch cost/identity proposal without authorizing GPU work."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from . import io
from .identity import source_pins


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--cpu-proof',required=True)
    args=parser.parse_args();root=Path(args.root).resolve()
    state=io.read(root/'search.json');first=io.read(root/'first_class_acceptance.json')
    require(state['status']=='first_class_complete_awaiting_second_budget' and first['status']=='first_class_verified','Continue the completed first class only.')
    previous=io.read(root/'formal_kappa_fix/budget.json');old=io.read(root/'manifest_kappa_final.json')
    cpu=io.read(args.cpu_proof)
    require(cpu['status']=='passed' and cpu['source_pins']==source_pins(),'Bind the finished new CPU proof.')
    for name in ('builder.py','trainer.py','mutable.py','protection.py','optimizer.py','muon_math.py'):
        path=str((Path(__file__).parent/name).resolve())
        require(old['recovery_source_sha256'][path]==source_pins()[path],'Changed native training mathematics requires new affected GPU proof.')
    legacy_path=Path('artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/live_gate_up_20261107/authorization.json')
    legacy=io.read(legacy_path)
    for name in ('baseline_longbench_scores','longbench_archive'):
        io.bound_path(legacy[name])
    order=state['frozen_order']
    train=dict(k_proj=2700,o_proj=3600,gate_proj=7200,q_proj=9000,v_proj=10800,down_proj=14400,up_proj=18000)
    # The three already-passed >=4-way items save at least12 node evaluations
    # per nonterminal category at the fully-active26-trajectory upper bound.
    new_500=5*26+12
    node500=4*new_500-5*(8+2+2)
    training_ceiling=sum(23*train[c] for c in order[1:-1])/3600+12*train[order[-1]]/3600+new_500*.1
    components=dict(technical=4.,fit=15.,training=365.,long_training=18.,export=69.,evaluation=134.,terminal_longbench=24.,replay=60.,boundary=0.)
    require(new_500==142 and node500==508 and training_ceiling<=components['training'],'Wrong full dependency count.')
    require((node500+16+12)*900/3600<=components['evaluation'],'Missing complete eight-task costs.')
    require(((node500+16)*450+12*900)/3600<=components['export'],'Missing export/fresh costs.')
    prior={k:v/3600 for k,v in previous['actual_seconds_by_component'].items()}
    cumulative={k:components.get(k,0)+prior.get(k,0) for k in sorted(set(components)|set(prior))}
    counters=previous['counter_reserved']
    proposal=dict(schema='sc_cat_remaining_budget_v1',batch='remaining_and_replay',run_id=old['run_id'],
        status='reviewable_proposal_not_authorization',confirmed=False,user_approved=False,
        scope_user_message='按规范做完所有类别的啊',scope_confirmed=True,GPU_budget_confirmed=False,
        categories=order,gpus=[4,5,6,7],max_simultaneous_GPU_tasks=4,max_live_export_tasks=3,
        new_categories=order[1:],first_boundary=state['categories'][order[0]]['boundary'],
        inherited_calibration=state['short_screen'],inherited_long_categories=state['long_categories'],
        prior_actual_GPUh=previous['total_actual_GPU_hours'],new_budget_GPUh=sum(components.values()),
        new_budget_by_component=components,total_GPU_hours=sum(cumulative.values()),budget_by_component=cumulative,
        old_unused_budget_transferred=False,expected_new_GPUh=[220,360],
        expectation_boundary='Planning range from measured original pipeline and first new class; later joint-P/Muon settings remain unmeasured.',
        maximum_raws=counters['raws']+12,maximum_trajectories=counters['trajectories']+new_500+2,
        maximum_KD_updates=counters['KD_updates']+new_500*500+2000,
        maximum_raw_full8=counters['raw_full8']+12,maximum_node_full8=counters['node_full8']+node500+16,
        maximum_terminal_longbench=2,maximum_replay_fits=7,maximum_replay_exports=7,maximum_replay_full8=7,
        maximum_replay_longbench=1,maximum_replay_KD_updates=4125,
        new_count_bounds=dict(VAE_fits=12,trajectories500=new_500,trajectories1000=2,node_full8=node500+16,raw_full8=12,
            fixed_replay_fits=7,fixed_replay_full8=7,complete_LongBench_artifacts=3),
        task_seconds=dict(fit_by_category={c:4500 for c in order},train500_by_category=train,
            replay_seconds_per_update_by_category=dict(k_proj=4,o_proj=6,gate_proj=14.4,q_proj=18,v_proj=21.6,down_proj=28.8,up_proj=36)),
        longbench_archive=legacy['longbench_archive'],baseline_longbench_scores=legacy['baseline_longbench_scores'],
        longbench_partitions=[part['tasks'] for part in legacy['LongBench_partition_policy']['partitions']],
        execution_policy=dict(train_device='cuda:0',attention_implementation='flash_attention_2',teacher_model_offload='none',
            teacher_output_offload='cpu',teacher_output_pin_memory=True,teacher_output_chunk_tokens=8,deterministic=False),
        resource_snapshot=dict(time=datetime.now().astimezone().isoformat(),
            GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
            output_free_bytes=shutil.disk_usage(root).free),
        technical_scope='Reuse unchanged native250 resume/LoRA/QAT proofs. Before second-class KD run two scoped2-update export/fresh checks; before final-class KD run one no-LoRA DP2-update export/fresh check.',
        numerical_boundary='Search and fixed replay preserve the first-class numerical policy. Replay output and numeric acceptance are separate; no invented tolerance or promise of bitwise matching.',
        no_automatic_retry=True,reference_mode='reuse_verified_BF16_full8_and_LongBench',reference_GPUh=0,
        cost_sources=[io.binding(root/'formal_kappa_fix/budget.json'),io.binding(legacy_path),
            io.binding(legacy_path.parent/'gpu0_3/formal/budget.json'),
            io.binding(legacy_path.parent.parent/'live_k_o_20261106/formal/budget.json')])
    manifest=deepcopy(old)
    manifest.update(status='second_batch_prepared_not_authorized',previous_manifest=io.binding(root/'manifest_kappa_final.json'),
        implementation_revision='remaining_search_terminal_and_replay_20261109',recovery_source_sha256=source_pins(),
        code_identity=digest(source_pins()),remaining_cpu=io.binding(args.cpu_proof),first_acceptance=io.binding(root/'first_class_acceptance.json'),
        previous_budget=io.binding(root/'formal_kappa_fix/budget.json'),new_actual_GPUh=0.,GPU_authorized=False)
    io.write(root/'manifest_remaining.json',manifest)
    proposal['manifest']=io.binding(root/'manifest_remaining.json')
    io.write(root/'budget_proposal_second.json',proposal)
    print(dict(new_GPUh=proposal['new_budget_GPUh'],cumulative_ceiling=proposal['total_GPU_hours'],expected=proposal['expected_new_GPUh'],counts=proposal['new_count_bounds']),flush=True)


if __name__=='__main__':main()

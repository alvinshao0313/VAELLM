"""Native BF16 copies of live masters with explicit same-parent provenance."""
from pathlib import Path
import gc
from . import bindings as b
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact,live_stamp
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


def load(directory):
    directory=Path(directory).resolve(strict=True)
    source=directory==b.SOURCE;control=directory==b.CONTROL
    b.require(source or control or directory.is_relative_to(b.OUTPUT.resolve()),'Unknown checkpoint/source root.')
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Actual native strict load failed.')
    if source or control:
        b.require(meta['checkpoint_id']==(b.SOURCE_ID if source else b.CONTROL_ID),'Original endpoint UUID changed.')
    else:
        p=meta['extra_meta']['distill_stage']
        b.require(p['algorithm']==b.ALGORITHM and p['origin']=='same_parent_local_recovery'
            and p['parent']==b.source() and p['contract']==b.CONTRACT,'Local provenance/parent/contract mismatch.')
    expected=b.read(directory.parent/'expected_state.json')
    b.require(runtime.records(model.state_dict())==expected,'Complete actual native BF16 tensor state differs.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['compressed_target_count']==252
        and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'Actual S4 storage contract changed.')
    model.eval().requires_grad_(False)
    return model,meta,budget


@b.torch.no_grad()
def save(model,active,optimizer,request,step,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    random_state,stamp,mode=rng(),live_stamp(active,optimizer),model.training
    deployment,meta,loaded=load_v6_model_checkpoint(str(b.SOURCE),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Snapshot source must be the exact native parent.')
    frozen=runtime.records({k:v for k,v in deployment.state_dict().items() if k not in active})
    parameters=dict(deployment.named_parameters())
    for name,p in active.items():parameters[name].copy_(p.detach().to(device='cpu',dtype=b.torch.bfloat16))
    for module in runtime.projections(deployment).values():module.disable_trainable_decode()
    deployment.gradient_checkpointing_disable();deployment.eval().requires_grad_(False)
    b.old.native.refresh_vae_linear_runtime_after_state_load(deployment)
    b.old.install_precision_runtime(deployment,b.torch.bfloat16)
    b.require(runtime.records({k:v for k,v in deployment.state_dict().items() if k not in active})==frozen,
        'Snapshot changed a nondecoder parent field.')
    provenance=dict(algorithm=b.ALGORITHM,origin='same_parent_local_recovery',scope=b.CONTRACT['scope'],
        parent=request['parent'],round_source=str(b.SOURCE),arm=request['arm'],actual_updates=step,
        validation_only=request['validation_only'],did_train=True,category='down_proj',active_projection_count=252,
        contract=b.CONTRACT,source_sha256=request['source_sha256'],data_digest=request['data_digest'])
    extra=dict(meta['extra_meta']);extra.update(stage='after_category',category='down_proj',
        distill_stage=provenance,distill_stage_history=[*meta['extra_meta']['distill_stage_history'],provenance])
    saved=b.old.native.save_v6_full_checkpoint(deployment,str(root/'model'),checkpoint_kind='category_boundary',
        compressed_targets=list(runtime.projections(deployment)),pending_dense_targets=[],skip_targets=[],train_mode='decoder_only',
        after_category_mode='none',completed_categories=list(b.CATEGORIES),compression_categories=list(b.CATEGORIES),
        target_modules=list(b.CATEGORIES),base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),extra_meta=extra)
    b.write(root/'expected_state.json',runtime.records(deployment.state_dict()))
    budget=audit_state_dict(saved['meta_payload'],deployment.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'Native export ledger changed.')
    del deployment;gc.collect();b.torch.cuda.empty_cache();restore_rng(random_state)
    b.require(rng_exact(random_state) and live_stamp(active,optimizer)==stamp and model.training==mode,
        'Snapshot changed live FP32 masters, Adam, mode, or RNG.')
    report=dict(checkpoint=b.event(root/'model'),actual_updates=step,live_masters_Adam_RNG_preserved=True,bit_budget=budget)
    b.write(root/'snapshot.json',report)
    return report

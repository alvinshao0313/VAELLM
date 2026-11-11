"""Independent BF16 native export; never cast or serialize the live Adam masters."""
import gc

from . import bindings as b, runtime
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


@b.torch.no_grad()
def save(model,active,optimizer,identity,training,fixture,root):
    from .training import adam_stamp
    stamp=adam_stamp(optimizer,active)
    rng=b.torch.get_rng_state().clone()
    cuda=[state.clone() for state in b.torch.cuda.get_rng_state_all()]
    deployment,source_meta,loaded=load_v6_model_checkpoint(str(b.SOURCE),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Build the independent deployment from the strict original bank.')
    before=runtime.records({name:value for name,value in deployment.state_dict().items() if name not in active})
    parameters=dict(deployment.named_parameters())
    for name,p in active.items():
        target=parameters[name]
        b.require(target.dtype==b.torch.bfloat16 and target.shape==p.shape, 'Actual source decoder topology changed.')
        target.copy_(p.detach().to(device='cpu',dtype=b.torch.bfloat16))
    deployment.eval().requires_grad_(False)
    for module in runtime.projections(deployment).values():module.disable_trainable_decode()
    b.native.refresh_vae_linear_runtime_after_state_load(deployment)
    b.install_precision_runtime(deployment,b.torch.bfloat16)
    after=runtime.records({name:value for name,value in deployment.state_dict().items() if name not in active})
    b.require(before==after, 'Export must preserve all original nondecoder packed/protection/rotation/backbone fields.')
    expected=runtime.records(deployment.state_dict())
    directory=root/'final_model'
    extra=dict(source_meta['extra_meta'])
    extra.update(category=None,stage='final',distill_stage=dict(kind='static_TF_recovery_compare',arm=identity['arm'],
        actual_updates=training['actual_updates']),distill_stage_history=[],
        recovery_compare=dict(execution=identity,training=training,original_parent_id=b.SOURCE_ID,
            no_category_refit=True,only_decoder_updates=True,original_nondecoder_exact=True))
    saved=b.native.save_v6_full_checkpoint(deployment,str(directory),checkpoint_kind='final_model',
        compressed_targets=list(runtime.projections(deployment)),pending_dense_targets=[],skip_targets=[],
        train_mode='decoder_only',after_category_mode='none',completed_categories=list(b.CATEGORIES),
        compression_categories=list(b.CATEGORIES),target_modules=list(b.CATEGORIES),base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),
        extra_meta=extra)
    meta=b.read(directory/'checkpoint_meta.json')
    budget=audit_state_dict(meta,deployment.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,
        'Recovery must preserve the full actual4.094bpw research-parent ledger.')
    b.write(root/'expected_state.json',expected)
    b.torch.save(fixture,root/'fixture.pt')
    b.require(adam_stamp(optimizer,active)==stamp, 'Export changed a live master/Adam identity, storage or version.')
    b.torch.set_rng_state(rng);b.torch.cuda.set_rng_state_all(cuda)
    del deployment
    gc.collect()
    return dict(directory=str(directory),checkpoint_id=meta['checkpoint_id'],metadata=b.binding(directory/'checkpoint_meta.json'),
        weight_stat=b.parent.s.stat(directory/'pytorch_model.bin'),bit_budget=budget,
        expected_state=b.binding(root/'expected_state.json'),fixture=b.binding(root/'fixture.pt'),
        live_masters_Adam_unchanged=True,all_original_nondecoder_fields_exact=True,
        final_lowbit_acceptance_evaluated=False,
        numerical_scope='BF16 deployment cast of live FP32 decoder masters; not an exact live-vs-deployed logits claim.')

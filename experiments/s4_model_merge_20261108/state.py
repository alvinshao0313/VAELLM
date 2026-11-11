"""Native-only merge checkpoint with truthful two-donor provenance."""
from pathlib import Path
from . import bindings as b,merge
from experiments.s4_scope_objective_20261107 import core
from experiments.s4_scope_objective_20261107 import state as prior_state
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


def load(directory):
    directory=Path(directory).resolve(strict=True)
    b.require(directory.is_relative_to(b.OUTPUT.resolve()),'Unknown merge output root.')
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Native strict merge load failed.')
    provenance=meta['extra_meta']['model_merge']
    b.require(provenance['algorithm']==b.ALGORITHM and provenance['parent']==b.source()
        and provenance['contract']==b.CONTRACT and provenance['did_train'] is False
        and provenance['arm'] in b.ARMS
        and provenance['ratio_late']==b.ARMS[provenance['arm']]['ratio_late'],'Merge provenance differs.')
    b.require(core.runtime.records(model.state_dict())==b.read(directory.parent/'expected_state.json'),
        'Merged persistent state differs.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['compressed_target_count']==252
        and abs(budget['user_linear_bpw']-b.BPW)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'Merged native ledger changed.')
    return model.eval().requires_grad_(False),meta,budget


@b.torch.no_grad()
def build(request,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    saved={}
    for step in (64,128):
        record=request['donors'][str(step)]['state']
        b.require(b.binding(record['path'])==record,'Donor FP32 state binding changed.')
        saved[step]=b.torch.load(record['path'],map_location='cpu',weights_only=False,mmap=True)['masters']
    b.require(list(saved[64])==list(saved[128])==request['parameter_names'],'Actual merge names/order changed.')
    model,meta,_=prior_state.load(b.SOURCE);parameters=dict(model.named_parameters());modules=core.runtime.projections(model)
    mutable=merge.active_keys(request['parameter_names'])
    b.require(b.old.old_life.frozen_proof(model,mutable)==request['frozen_complement'],'Parent frozen proof differs.')
    for name in request['parameter_names']:
        value=merge.blend(saved[64][name],saved[128][name],request['recipe']['ratio_late']);key,hard=merge.native(name,value)
        if '._parallel_stage_decoder.' in key:parameters[key].copy_(hard)
        else:modules[key.rsplit('.',1)[0]].protected_input_qvalues.copy_(hard)
    for module in modules.values():module.disable_trainable_decode()
    model.gradient_checkpointing_disable();model.eval().requires_grad_(False)
    b.old.native.refresh_vae_linear_runtime_after_state_load(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    b.require(b.old.old_life.frozen_proof(model,mutable)==request['frozen_complement'],
        'Merge altered maincodes/scale/index/rotation/backbone.')
    b.require(core.runtime.records(model.state_dict())==request['expected_native_state'],'Merge differs from CPU-bound formula.')
    provenance=dict(algorithm=b.ALGORITHM,parent=request['parent'],contract=b.CONTRACT,donors=request['donors'],
        ratio_late=request['recipe']['ratio_late'],arm=request['arm'],source_sha256=request['source_sha256'],
        did_train=False,optimizer_merged=False,not_module_splicing=True,
        generalization_evidence='Unverified; same-seed same-trajectory exploratory model merge only')
    extra=dict(meta['extra_meta']);extra['model_merge']=provenance
    result=b.old.native.save_v6_full_checkpoint(model,str(output/'model'),checkpoint_kind='category_boundary',
        compressed_targets=list(modules),pending_dense_targets=[],skip_targets=[],train_mode='decoder_only',
        after_category_mode='none',completed_categories=list(b.CATEGORIES),compression_categories=list(b.CATEGORIES),
        target_modules=list(b.CATEGORIES),base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),extra_meta=extra)
    b.write(output/'expected_state.json',core.runtime.records(model.state_dict()))
    budget=audit_state_dict(result['meta_payload'],model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-b.BPW)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'Merge storage ledger changed.')
    report=dict(checkpoint=b.event(output/'model'),ratio_late=request['recipe']['ratio_late'],bit_budget=budget,
        expected_native_state_exact=True,full_frozen_complement_exact=True,native_fields=3927,
        did_train=False,optimizer_merged=False,motion=request['motion'])
    b.write(output/'merge.json',report);return report

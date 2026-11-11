"""Construct only the local up-skip-down interface fixture; never a formal parent."""
from pathlib import Path
import gc
import time

from . import bindings as b
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
from train_utils.cat_train_runtime import load_cat_resume_distill_progress
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state,validate_cat_runtime_identity
from train_utils.cat_checkpoint_v6 import save_cat_v6_full_checkpoint
from rotation.model_utils import temporarily_disable_torch_weight_init
from transformers import Qwen3ForCausalLM


def cpu_HF():
    # No forward is performed. Hidden CUDA cannot pass the native FA2 constructor check.
    with temporarily_disable_torch_weight_init():
        model=Qwen3ForCausalLM.from_pretrained(str(b.HF),torch_dtype='auto',low_cpu_mem_usage=True,
            attn_implementation='sdpa',local_files_only=True)
    model.seqlen=2048
    model.config._attn_implementation='flash_attention_2'
    return model


def prepare(root):
    root=Path(root);began=time.monotonic()
    b.require(not root.exists() and root.resolve().is_relative_to(b.OUTPUT),'Use a new local validation fixture.')
    cat,hf,training,vae,_cfgs,identity,_normalized,_argv=b.native_recipe(root/'unused_factory')
    model,meta,loaded=load_v6_full_checkpoint_into_model(cpu_HF(),str(b.FIXTURE_SOURCE),map_location='cpu',strict=True)
    source=b.event(b.FIXTURE_SOURCE)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys
        and source==b.read(b.CONTROL/'formal/fit.json')['nodes'][-1]['pre']['checkpoint'],'Exact protected fixture source changed.')
    future=tuple(name+'.' for name in runtime.projections(model) if name.rsplit('.',1)[-1] in ('up_proj','down_proj'))
    frozen=runtime.records({name:value for name,value in model.state_dict().items() if not name.startswith(future)})
    factory=cpu_HF().eval().requires_grad_(False)
    dense={name:module for name,module in runtime.projections(factory).items() if name.rsplit('.',1)[-1] in ('up_proj','down_proj')}
    dense_records=runtime.records({name+'.weight':module.weight for name,module in dense.items()})
    b.require(len(dense)==72 and all(type(module) is b.torch.nn.Linear for module in dense.values()),'Restore72 actual original HF dense future projections.')
    for name,module in dense.items():b.old.native.set_module_by_name(model,name,module)
    del factory,dense;gc.collect()
    state=model.state_dict()
    b.require(runtime.records({name:value for name,value in state.items() if not name.startswith(future)})==frozen
        and all(b.old.old_life.tensor_record(state[name])==value for name,value in dense_records.items()),
        'Fixture must preserve its source prefix and replace both future categories with exact original HF values.')
    model.eval().requires_grad_(False)
    history=[dict(category=category,did_train=False,actual_updates=0,validation_fixture_inherited=True,
        source_checkpoint_id=source['checkpoint_id']) for category in b.CATEGORIES[:5]]
    root.mkdir(parents=True)
    saved=save_cat_v6_full_checkpoint(model,str(root/'model'),checkpoint_kind='category_boundary',category='gate_proj',
        completed_categories=b.CATEGORIES[:5],compression_categories=b.CATEGORIES,cat_args=cat,vae_args=vae,
        training_args=training,base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.FIXTURE_SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),
        distill_stage_meta=dict(algorithm='fresh_BF16_S4_prefix_recovery',origin='incremental_validation_fixture',
            validation_only=True,candidate=b.CONTRACT['candidate'],contract=b.CONTRACT,did_train=False,
            source_checkpoint=source,future_HF_categories=['up_proj','down_proj']),distill_stage_history=history)
    expected=runtime.records(model.state_dict());b.write(root/'expected_state.json',expected)
    del state,model;gc.collect()
    reloaded,reloaded_meta,strict=load_v6_full_checkpoint_into_model(cpu_HF(),saved['output_dir'],map_location='cpu',strict=True)
    b.require(not strict.missing_keys and not strict.unexpected_keys
        and runtime.records(reloaded.state_dict())==expected,'Actual local fixture strict readback is not exact.')
    progress=load_cat_resume_distill_progress(saved['output_dir'])
    b.require(progress.completed_categories==b.CATEGORIES[:5]
        and b.CATEGORIES[len(progress.completed_categories)]=='up_proj','Actual native resume must next execute up.')
    activation,global_plan,saved_identity=restore_cat_runtime_state(
        b.torch.load(Path(saved['output_dir'])/'cat_runtime_state.pt',map_location='cpu',weights_only=True),
        access_token=getattr(hf,'access_token',None))
    validate_cat_runtime_identity(saved_identity,identity)
    from train_utils import cat_train_pipeline as native
    b.require(Path(native.__file__).resolve().is_relative_to(b.parent.s.SNAPSHOT)
        and reloaded.config._attn_implementation=='flash_attention_2','CPU construction difference must not enter saved/GPU policy.')
    result=dict(validation_only=True,model=b.event(saved['output_dir']),prefix_source=source,
        source_HF_identity=b.parent.s.hf_identity(b.HF),frozen_prefix_exact=True,original_HF_future_exact=True,
        installed_S4_projections=180,future_dense_projections=72,actual_new_fit_categories=['up_proj','down_proj'],
        CPU_constructor='Explicit SDPA only to instantiate real HF tensors; no CPU forward; FA2 policy restored before save',
        GPU_policy='Original frozen native FA2 loader/consumer',full_fixture_strict_state_exact=True,
        actual_resume_completed=list(progress.completed_categories),actual_next_category='up_proj',
        actual_resume_runtime_identity_exact=True,native_source=str(Path(native.__file__).resolve()),
        formal_initialization_allowed=False,expected_state=b.binding(root/'expected_state.json'),seconds=time.monotonic()-began)
    b.write(root/'fixture.json',result)
    return result

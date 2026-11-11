"""Keep the original objective and allow input backward through frozen MLPs."""
from collections import Counter
from . import bindings as b
from experiments.s4_webmed_recovery_20261105 import data,diagnostics
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime,training


def load_data():
    dataset,targets,evidence=data.load()
    groups=dataset['groups'][:64]
    counts=Counter(k for group in groups for k in group['case_ids'])
    for name in ('webinstruct','medmcqa'):
        found=[k for k in counts if k.startswith(name+'/')]
        b.require(len(found)==1024 and all(counts[k]==1 for k in found),'Fixed1024/source once required.')
    qa=[c for c in dataset['training_cases'] if c['domain']!='mmlu']
    b.require(len(qa)==128 and all(counts[c['case_id']]==4 for c in qa),'OriginalQA128 must appear four times.')
    return dataset,targets,evidence


def selected_projections(model,arm):
    spec=b.ARMS[arm]
    found=runtime.projections(model)
    b.require(len(found)==252 and all(isinstance(m,b.old.VAELinear) and m.residual_stages==4 for m in found.values()),
        'All252 projections must remain actual nativeS4, including frozen180.')
    expected={f'model.layers.{layer}.self_attn.{category}'
        for layer in range(36) for category in spec['categories']}
    selected={name:module for name,module in found.items() if name.rsplit('.',1)[-1] in spec['categories']}
    b.require(set(selected)==expected and len(selected)==spec['projections'],'Exact QK/VO projection set changed.')
    return selected


def install(model,arm):
    from train_utils.decoder_execution import enable_vae_linear_by_execution_plan
    selected=selected_projections(model,arm);active={}
    b.require(not any(p.requires_grad for p in model.parameters()),'Start from the entirely frozen common parent.')
    for name,module in selected.items():
        module.packed_vq_decoder_linear=True
        plan=enable_vae_linear_by_execution_plan(module,mode='trainable_decoder')
        b.require(plan.use_packed,'Use unchanged native packed decoder training path.')
        module._parallel_stage_decoder.use_checkpoint=True
        for relative,parameter in module._parallel_stage_decoder.named_parameters():
            parameter.data=parameter.data.float();parameter.requires_grad_(True)
            active[name+'._parallel_stage_decoder.'+relative]=parameter
    expected={name+'._parallel_stage_decoder.'+relative
        for name,module in selected.items() for relative,_ in module._parallel_stage_decoder.named_parameters()}
    b.old.install_precision_runtime(model,b.torch.bfloat16)
    b.require(set(active)==expected=={name for name,p in model.named_parameters() if p.requires_grad}
        and len(active)==b.ARMS[arm]['leaves'] and all(p.dtype==b.torch.float32 for p in active.values()),
        'Only the exact selected432 FP32 decoder leaves may train.')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();model.config.use_cache=False
    warm=runtime.warm_frozen(model)
    b.require(sum(row['total'] for row in warm['devices'].values())==180,'All180 frozen projections must warm.')
    return active,warm


def panel(model,dataset,targets,heldout,baseline=None):
    result=diagnostics.evaluate(model,dataset,targets,heldout,baseline)
    result['early']['training_by64']=True
    result['late']['training_by64']=False
    result['late']['scope']='Official train cases beyond group63; untrained in this fixed64 batch, mixed target read-only'
    result['heldout']['training_by64']=False
    return result

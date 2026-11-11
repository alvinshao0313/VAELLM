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


def install(model,arm):
    active={};spec=b.ARMS[arm]
    runtime.install(model,{},spec['stage'],'terminal',active)
    found=runtime.projections(model)
    b.require(len(found)==252 and all(isinstance(m,b.old.VAELinear) and m.residual_stages==4 for m in found.values()),
        'All252 projections must remain actual nativeS4, including frozenMLP.')
    b.require(len(active)==spec['leaves'] and len({n.split('._parallel_stage_decoder.')[0] for n in active})==spec['projections'],
        'Actual scope leaf/projection inventory changed.')
    warm=runtime.warm_frozen(model)
    return active,warm


def panel(model,dataset,targets,heldout,baseline=None):
    result=diagnostics.evaluate(model,dataset,targets,heldout,baseline)
    result['early']['training_by64']=True
    result['late']['training_by64']=False
    result['late']['scope']='Official train cases beyond group63; untrained in this fixed64 batch, mixed target read-only'
    result['heldout']['training_by64']=False
    return result

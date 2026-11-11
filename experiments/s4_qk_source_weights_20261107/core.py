"""Verified QK install with source-weight-aware updates and diagnostic aggregation."""
from . import bindings as b,data,objective
from experiments.s4_scope_refine_20261106 import core as previous

runtime=previous.runtime


def install(model,arm):
    b.require(arm in b.ARMS,'Unknown fixed source-weight arm.')
    return previous.install(model,'qk')


def panel(model,dataset,targets,heldout,arm):
    result=previous.panel(model,dataset,targets,heldout)
    weights=b.ARMS[arm]['source_weights']
    for name in ('early','late'):
        item=result[name]
        item['actual_M_weighted_objective']=sum(weights[source]*item[source]['actual_mixed_KL'] for source in ('webinstruct','medmcqa'))
        item['actual_source_weights']={source:weights[source] for source in ('webinstruct','medmcqa')}
    result.update(teacher_alpha=.5,teacher_gold_mixture=[.5,.5],source_weights=weights,
        objective_comparison='Weighted objective uses each recipe; its absolute cross-arm value is not a quality ranking',
        common_comparison='Per-source unweighted KL, gold NLL/hits and complete eight use shared conditions')
    return result

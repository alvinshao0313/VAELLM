"""Reuse the verified exact QK forward/update path and label each diagnostic target."""
from . import bindings as b,data
from experiments.s4_scope_refine_20261106 import core as previous

runtime,training=previous.runtime,previous.training


def install(model,arm):
    b.require(arm in b.ARMS,'Unknown fixed alpha arm.')
    return previous.install(model,'qk')


def panel(model,dataset,targets,heldout,arm):
    result=previous.panel(model,dataset,targets,heldout)
    result.update(teacher_alpha=b.ARMS[arm]['alpha'],teacher_gold_mixture=b.contract(arm)['teacher_gold_mixture'],
        mixed_KL_comparison='Compare own0-to64 only; absolute mixed KL across different alpha targets is not a ranking',
        common_comparison='Gold NLL/hits and complete eight use shared conditions')
    return result

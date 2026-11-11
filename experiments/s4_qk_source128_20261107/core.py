"""Dispatch to each unchanged training operator; label panels by the actual cursor."""
from . import bindings as b
from experiments.s4_scope_refine_20261106 import core as equal,train as equal_train,state as equal_state
from experiments.s4_qk_source_weights_20261107 import core as weighted,train as weighted_train,state as weighted_state
from experiments.s4_attention128_20261106.core import exposure_labels

runtime=equal.runtime


def operators(arm):
    b.require(arm in b.ARMS,'Unknown paired continuation arm.')
    return (equal_train,equal_state) if arm=='qk' else (weighted_train,weighted_state)


def load_data(arm):
    return equal.load_data() if arm=='qk' else weighted.data.load(arm)


def panel(model,dataset,targets,heldout,arm,cursor):
    result=equal.panel(model,dataset,targets,heldout) if arm=='qk' else weighted.panel(model,dataset,targets,heldout,arm)
    for name,item in exposure_labels(dataset,cursor).items():
        result[name].pop('training_by64',None)
        result[name].update(item)
    return result


def exposure(dataset,targets,indices):
    return weighted.data.exposure(dataset,targets,indices)

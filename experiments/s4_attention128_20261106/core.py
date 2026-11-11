"""Reuse the original learning path and label panels by actual consumed identities."""
from . import bindings as b
from experiments.s4_recovery_screen_20261105 import core as prior

load_data,install=prior.load_data,prior.install
runtime,training,data=prior.runtime,prior.training,prior.data


def exposure_labels(dataset,cursor):
    consumed={key for group in dataset['groups'][:cursor] for key in group['case_ids']}
    result={}
    for name in ('early','late'):
        indices=b.prior.CONTRACT[name+'_groups']
        keys={key for index in indices for key in dataset['groups'][index]['case_ids'][:32]}
        result[name]=dict(completed_updates=cursor,training_consumed_count=len(keys&consumed),
            training_case_count=len(keys),trained_by_cursor=keys<=consumed,
            scope='Official train panel; consumption checked against actual completed groups')
    result['heldout']=dict(completed_updates=cursor,training_consumed_count=0,trained_by_cursor=False,
        scope='Pinned disjoint official-train heldout; never used for training')
    return result


def panel(model,dataset,targets,heldout,cursor):
    result=prior.diagnostics.evaluate(model,dataset,targets,heldout)
    for name,item in exposure_labels(dataset,cursor).items():result[name].update(item)
    return result

"""Reuse all252 packed decoder training, original data and fixed diagnostic panels."""
from . import bindings as b
from experiments.s4_recovery_screen_20261105 import core as original
from experiments.s4_webmed_recovery_20261105 import data,diagnostics
from experiments.s4_attention128_20261106.core import exposure_labels
from .protection import Protection

runtime=original.runtime


def load_data():
    dataset,targets,evidence=data.load()
    manifest=b.read(data.b.DATA/'teacher.json')
    raw=b.torch.load(data.checked(manifest['payload']),map_location='cpu',weights_only=True)['targets']
    b.require(len(dataset['groups'])==128 and len(raw)==4096,'Original128 groups/raw teacher required.')
    return dataset,targets,raw,evidence


def install(model,arm):
    active,warm=original.install(model,'all')
    modules=runtime.projections(model);mode=b.ARMS[arm]['protection']
    controller=Protection(model,modules,mode) if mode else None
    if controller:active.update(controller.parameters)
    b.require(len(active)==b.ARMS[arm]['leaves'],'Actual active leaf inventory changed.')
    return active,controller,warm


def mutable_names(active,arm):
    names={n for n in active if '._parallel_stage_decoder.' in n}
    mode=b.ARMS[arm]['protection']
    if mode:
        suffix='protected_input_scales' if mode=='scale' else 'protected_input_qvalues'
        names.update(n.rsplit('.',1)[0]+'.'+suffix for n in active if '._parallel_stage_decoder.' not in n)
    return names


def panel(model,dataset,targets,heldout,cursor):
    result=diagnostics.evaluate(model,dataset,targets,heldout)
    for name,item in exposure_labels(dataset,cursor).items():result[name].update(item)
    result['common_objective']='Original once-mixed forward KL/gold NLL; do not compare recipe loss values'
    return result

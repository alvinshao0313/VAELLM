"""One isolated skip-up contract; immutable previous implementations are read-only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

from experiments.s4_from_bf16_20261102 import bindings as base

old,parent,data=base.old,base.parent,base.data
torch,require,read,write,sha=base.torch,base.require,base.read,base.write,base.sha
ROOT,HF,PYTHON,CATEGORIES=base.ROOT,base.HF,base.PYTHON,base.CATEGORIES
HERE=Path(__file__).resolve().parent
# The unchanged generic S4 consumer already accepts this isolated sub-root.
OUTPUT=base.OUTPUT/'skip_up_20261103'
CONTROL=base.OUTPUT/'trial_20261102_1100'
FIXTURE_SOURCE=CONTROL/'formal/nodes/down_proj/pre/model'
CONTRACT=deepcopy(base.CONTRACT)
CONTRACT['candidate']='only_skip_up_category_recovery_20261103'
CONTRACT['recovery'].update(skip_categories=['up_proj'],skip_updates=0,
    skip_semantics='No optimizer, training groups or RNG consumption; complete live persistent state unchanged')
CONTRACT['readouts']['identical_up']='Reuse this new up/pre M only after full tensor/config/protocol identity; no new post inference exit claimed'
binding,event,initialize,resources,environment=base.binding,base.event,base.initialize,base.resources,base.environment
restore_rope_alias,native_recipe=base.restore_rope_alias,base.native_recipe


def bound_implementation(name):
    path=base.HERE/(name+'.py')
    spec=importlib.util.spec_from_file_location('experiments.s4_from_bf16_20261102._skip_up_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.b=__import__(__name__,fromlist=['bindings'])
    return module


def pins():
    prior=read(CONTROL/'plan.json')['source_sha256']
    parent.s.check_pins(prior)
    return {**prior,**{str(path):sha(path) for path in sorted(HERE.glob('*.py'))}}


def cleanup_model(directory,reason):
    require(Path(directory).resolve().is_relative_to(OUTPUT.resolve()),'Clean only this new candidate own completed payloads.')
    return base.cleanup_model(directory,reason)

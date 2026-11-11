"""Bind real parent/control identities, frozen recipe and CPU source compatibility."""
import ast
from dataclasses import asdict
import json
import sys
import time

from . import bindings as b


def prepare():
    b.cpu_environment()
    b.require(not (b.RESULT / 'prepared.json').exists(), 'Do not replace prepared experiment evidence.')
    from . import fit, run, evaluate
    binding = b.sources()
    recipes = {}
    for arm in ('control', *b.ARMS):
        parsed, identity, baseline = b.recipe(arm, b.OUTPUT / arm)
        activation, _plan, restored = b.restore_cat_runtime_state(parsed[-1], access_token=parsed[1].access_token)
        b.require(restored == baseline and activation['cache'].nsamples == 512 and activation['cache'].seqlen == 256,
            'Restore the genuine shared512x256 train calibration without changing source identity.')
        if arm == 'control':
            b.require(identity == baseline, 'Control recipe must reproduce original C0 runtime identity.')
        stage=asdict(b.recovery.canonical.resolve_cat_after_category_stage(parsed[0],parsed[2],category='down_proj',round_idx=6))
        current=json.loads(json.dumps(stage))
        reference=b.read(b.C0 / 'report.json')['stages'][2]['original_stage']
        current.pop('output_dir')
        reference.pop('output_dir')
        b.require(current == reference,'Original down500 stage/data/seed/config differs beyond independent output_dir.')
        recipes[arm] = dict(category_configs={name:asdict(cfg) for name,cfg in parsed[-2].items()},
            runtime_identity=identity, original_runtime_identity=baseline,
            recovery=stage,original_recovery_contract_equal=True)
    # Read the real dense parent and control without allocating a CUDA model.
    pmeta = b.read(b.PARENT / b.checkpoint.META_FILE)
    pstate = b.torch.load(b.PARENT / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    target = {name.removesuffix('.weight'):value for name,value in pstate.items() if '.mlp.down_proj.' in name}
    b.require(len(target) == 36
        and b.c.digest(target) == binding['original_down_fit']['source_target_digest'],
        'Use the exact live dense down source actually consumed by the original C0 fit.')
    control = b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    prefix = lambda values:{name:value for name,value in values.items() if '.mlp.down_proj.' not in name}
    b.require(b.c.digest(prefix(pstate)) == b.c.digest(prefix(control)), 'Raw control changed a frozen prefix field.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    paths = {name:str(module.__file__) for name,module in sys.modules.items()
        if name.split('.')[0] in ('train_utils','litebsq','rotation','e2e_common') and getattr(module,'__file__',None)}
    b.require(all(b.Path(path).resolve().is_relative_to(b.SNAPSHOT) for path in paths.values()),
        'Formal import escaped the bound N runtime.')
    b.write(b.RESULT / 'prepared.json',dict(status='cpu_verified',command=[sys.executable,*sys.argv],
        created_unix=time.time(),source_binding=binding,recipes=recipes,source_sha256=b.pins(),
        live_down_source_digest=b.c.digest(target),frozen_prefix_digest=b.c.digest(prefix(pstate)),
        control_prefix_exact=True,CUDA_initialized=b.torch.cuda.is_initialized(),
        boundary='CPU verifies source/recipe/frozen prefix. Real new-shape fit/deployment/fresh remains required.'))
    print(dict(status='cpu_verified',output=str(b.RESULT / 'prepared.json')),flush=True)


if __name__ == '__main__':
    prepare()

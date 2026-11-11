"""Shared CPU source/config verification for two mature scalar-LR candidates."""
import ast
from copy import deepcopy
import sys
import time

from . import bindings as b


def main():
    b.cpu_environment()
    b.require(not b.OUTPUT.exists() and not (b.RESULT / 'prepared.json').exists(), 'Use fresh LR result/output roots.')
    started = time.monotonic()
    binding = b.sources()
    recipes = {}
    for arm in b.ARMS:
        parsed, identity, baseline = b.recipe(arm, b.OUTPUT / arm / 'formal')
        activation, _plan, restored = b.restore_cat_runtime_state(parsed[-1], access_token=parsed[1].access_token)
        b.require(restored == baseline and activation['cache'].nsamples == 512
            and activation['cache'].seqlen == 256, 'Restore original source calibration identity; do not relabel historical LR.')
        recipes[arm] = identity
    original = b.read(b.ORIGINAL_PREPARED)
    parent = b.torch.load(b.PARENT / b.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    control = b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    prefix = lambda state: {key: value for key, value in state.items() if '.mlp.down_proj.' not in key}
    target = {key.removesuffix('.weight'): value for key, value in parent.items() if '.mlp.down_proj.' in key}
    b.require(len(target) == 36 and b.c.digest(target) == original['live_down_source_digest']
        and b.c.digest(prefix(parent)) == b.c.digest(prefix(control)) == original['frozen_prefix_digest'],
        'Both LR candidates must use the real common parent target and unchanged D16 frozen prefix.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.write(b.RESULT / 'prepared.json', dict(status='cpu_verified', exit_code=0, source_binding=binding,
        effective_runtime_identities=recipes, live_down_source_digest=b.c.digest(target),
        frozen_prefix_digest=b.c.digest(prefix(parent)), actual_D16_initialization_reference=b.initialization_reference(),
        source_sha256=b.pins(), CUDA_initialized=False, command=[sys.executable, *sys.argv],
        elapsed_seconds=time.monotonic() - started,
        evidence_reuse='Same M36/D16/K24/B16384/S2/native/block32 resource and consumer; only normal Adam scalar LR changes. No new short, data or recovery path.'))
    print(dict(status='cpu_verified', elapsed_seconds=time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()

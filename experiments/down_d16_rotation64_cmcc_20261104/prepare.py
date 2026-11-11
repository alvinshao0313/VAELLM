"""CPU source, resolved-rotation and frozen-field proof; no repeated GPU short."""
import ast
from copy import deepcopy
import sys
import time

from . import bindings as b


def main():
    b.cpu_environment()
    b.require(not b.OUTPUT.exists() and not (b.RESULT / 'prepared.json').exists(), 'Use independent result/output roots.')
    began = time.monotonic()
    binding = b.sources()
    parsed, identity, baseline = b.recipe(b.ARM, b.OUTPUT / b.ARM / 'formal')
    activation, _plan, restored = b.restore_cat_runtime_state(parsed[-1], access_token=parsed[1].access_token)
    b.require(restored == baseline and activation['cache'].nsamples == 512
        and activation['cache'].seqlen == 256, 'Reuse original train512×256 calibration and its actual parent binding.')
    original = b.read(b.ORIGINAL_PREPARED)
    parent = b.torch.load(b.PARENT / b.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    control = b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    prefix = lambda state: {key: value for key, value in state.items() if '.mlp.down_proj.' not in key}
    target = {key.removesuffix('.weight'): value for key, value in parent.items() if '.mlp.down_proj.' in key}
    b.require(len(target) == 36 and b.c.digest(target) == original['live_down_source_digest']
        and b.c.digest(prefix(parent)) == b.c.digest(prefix(control)) == original['frozen_prefix_digest'],
        'The actual parent target and all frozen fields must match the original D16 control.')
    from rotation.weight_preprocess import build_weight_rotation
    metadata = b.read(b.CONTROL / b.checkpoint.META_FILE)
    rotations = []
    for item in metadata['native_modules']:
        if not item['name'].endswith('.down_proj'):
            continue
        spec = deepcopy(item['weight_rotation'])
        b.require(spec['block_size'] == 32, 'The actual existing control must use block32.')
        spec['block_size'] = 64
        transform = build_weight_rotation(4096, 12032, spec)
        transform.validate_state()
        rotations.append(dict(name=item['name'], expected_rotation=spec,
            actual_unprotected_shape=[4096, 12032], no_padding=True))
    b.require(len(rotations) == 36 and not b.torch.cuda.is_initialized(), 'Resolve all36 exact block64 consumers on CPU.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    b.OUTPUT.mkdir(parents=True)
    b.write(b.RESULT / 'prepared.json', dict(status='cpu_verified', exit_code=0,
        source_binding=binding, runtime_identity=identity, baseline_runtime_identity=baseline,
        live_down_source_digest=b.c.digest(target), frozen_prefix_digest=b.c.digest(prefix(parent)),
        actual_D16_initialization_reference=b.initialization_reference(), expected_rotations=rotations,
        source_sha256=b.pins(), command=[sys.executable, *sys.argv], CUDA_initialized=False,
        elapsed_seconds=time.monotonic() - began,
        validation_reuse='Actual M36/D16/B16384 fitter, block64 consumer and mixed full8 already have applicable evidence; verify only new binding/resolved64/initialization/saved consumer here. No recovery path enabled.'))
    print(dict(status='cpu_verified', elapsed_seconds=time.monotonic() - began), flush=True)


if __name__ == '__main__':
    main()

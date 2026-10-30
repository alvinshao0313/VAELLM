"""Build one profile-specific shared R896 cache after the actual02 native audit exists."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import bindings as b
import data
import retention_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    b.source()  # Refuse absent actual02 before loading inputs or touching CUDA.
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and b.torch.cuda.device_count() == 1,
        'Use caller-activated bitvae, snapshot cwd and exactly one assigned visible GPU.')
    b.torch.set_num_threads(4); b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    _, archive, materials = data.load_materials()
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    record = retention_cache.build(archive, materials, observe)
    observe('actual02_R896_verified', dict(path=str(retention_cache.CACHE / 'cache.json'),
        readback=record['readback_validation'], restored_projection_count=0))


if __name__ == '__main__':
    main()

"""Generate one actual04-initial native R896 shared by this profile's J32 and B32."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import selected_bindings as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    directory, meta, _, audit_path = b.source04()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and b.torch.cuda.device_count() == 1,
        'Use bitvae/snapshot cwd/exactly one assigned GPU.')
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    _, archive, materials = b.inputs.load_materials()
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    record = b.retention.build_native(archive, materials, observe, directory=directory, meta=meta,
        source_audit=b.binding(audit_path), cache_dir=b.R_CACHE, profile=b.PROFILE, schema=b.R_SCHEMA)
    observe('actual04_R896_verified', dict(path=str(b.R_CACHE / 'cache.json'), readback=record['readback_validation'],
        restored_projection_count=record['restored_projection_count']))


if __name__ == '__main__':
    main()

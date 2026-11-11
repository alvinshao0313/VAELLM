"""Use the mature full8 and fixed train deployment consumers in this process."""
import argparse
import importlib
import os
import sys
import time

from . import bindings as s
from experiments.down_geometry_baseline_cmcc_20261103 import evaluate as mature
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def setup():
    # Rebind only this independent process; the original source files stay frozen.
    mature.s = s
    mature.setup()
    b.command = lambda component,*args: [str(s.PYTHON),'-B','-u','-m',
        'experiments.down_d16_batch8192_cmcc_20261103.evaluate','--component',component,*map(str,args)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True, choices=('raw_full8','full8','readout'))
    parser.add_argument('--arm', choices=(s.ARM,))
    args, remaining = parser.parse_known_args()
    setup()
    b.CLI_ARGUMENTS = list(sys.argv)
    s.torch.set_num_threads(8)
    if args.component == 'raw_full8':
        mature.raw_full8(args.arm)
    elif args.component == 'readout':
        s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '4', 'Use only the assigned GPU4.')
        start = time.time()
        mature.readout(args.arm)
        path = s.RESULT / s.ARM / 'raw_readout.json'
        result = s.read(path)
        s.require(result['input_ids_sha256'] == s.INPUT_SHA and len(result['deployment']['per_down']) == 36,
            'Require the exact same eight original train inputs and all36 deployed down projections.')
        result.update(exit_code=0, started_unix=start, elapsed_seconds=time.time()-start,
            peak_allocated=s.torch.cuda.max_memory_allocated(), peak_reserved=s.torch.cuda.max_memory_reserved(),
            boundary='Fixed train local diagnostic and complete raw eight tasks only; no final-nine claim.')
        s.write(path,result)
    else:
        sys.argv = [sys.argv[0],*remaining]
        return importlib.import_module('experiments.adaptive_class_evaluation_20261030.full8').main()


if __name__ == '__main__':
    raise SystemExit(main())

"""Fixed groups of real selected-B gate layers, no DDP or shared entropy."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.adaptive_vector_rvq import common as c
from . import sources as s
from .fit import fit_group, make_value, route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--h', type=Path, required=True)
    parser.add_argument('--start-layer', type=int, required=True)
    parser.add_argument('--stop-layer', type=int, required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
        and not args.output.exists() and 0 <= args.start_layer < args.stop_layer <= 36
        and args.start_layer % 6 == args.stop_layer % 6 == 0, 'Use assigned single GPU, fresh output and fixed complete six-layer groups.')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    sources = s.code_identity()
    report = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time(),
        gpu=os.environ['CUDA_VISIBLE_DEVICES'], short_validation=args.short, source_sha256=sources, groups=[])
    c.write(args.output / 'report.json', report)
    try:
        gate, specs, manifest, live = s.raw_inputs()
        h_report = c.read(args.h / 'report.json')
        c.require(h_report['status'] == 'completed_live36gate_train64_h' and h_report['source_after_o'] == s.BASE_ID
            and c.sha(args.h / 'input_h.pt') == h_report['h_file_sha256'], 'Use actual matching completed live-parent train64 h.')
        energy = torch.load(args.h / 'input_h.pt', map_location='cpu', weights_only=True)
        c.require(c.digest(energy) == h_report['h_digest'], 'Actual fixed h changed.')
        by_layer = {int(spec['name'].split('.')[2]): spec for spec in specs}
        report.update(raw_B=manifest['source_raw']['checkpoint_id'], live_after_o=s.BASE_ID,
            h_report_sha256=c.sha(args.h / 'report.json'), h_digest=h_report['h_digest'])
        for first in range(args.start_layer, args.stop_layer, 6):
            group_dir = args.output / f'group_{first:02d}_{first + 5:02d}'
            values = [make_value(gate, by_layer[layer], live, energy, group_dir / f'layer_{layer:02d}', 'cuda:0')
                      for layer in range(first, first + 6)]
            stages, training = fit_group(values, 10000, 'cuda:0', stop_after=2 if args.short else None)
            endpoints = [route(value, stage, live, 'cuda:0') for value, stage in zip(values, stages, strict=True)]
            report['groups'].append(dict(first_layer=first, last_layer=first + 5, training=training, endpoints=endpoints))
            c.write(args.output / 'report.json', report)
            del values, stages, endpoints
            gc.collect()
            torch.cuda.empty_cache()
        c.require(all(c.sha(path) == digest for path, digest in sources.items()), 'A consumed source changed while fitting the fixed group.')
        report.update(status='completed_short_actual_group_paths' if args.short else 'completed_formal_stage3_and_once_routes', exit_code=0)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        c.write(args.output / 'report.json', report)
    print(dict(status=report['status'], seconds=report['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()

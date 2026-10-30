"""Existing official partition runner with the existing native rank0/2.6 loader."""
import argparse
import json
import bindings as b
from experiments.longbench_v1 import runner

load_model = b.module(b.ROOT / 'experiments/cat_recompression/longbench.py', '_aligned_native_longbench_loader').load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--max-runtime-seconds', required=True, type=float)
    args = parser.parse_args()
    b.require(b.read(args.config)['mode'] == 'full', 'Only full official partitions; no sample limits or resume.')
    runner.load_model = load_model
    report = runner.run(args.config, args.output, datasets=args.datasets,
                        max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({key: report[key] for key in ('status', 'new_predictions', 'already_present', 'duration_seconds')}), flush=True)


if __name__ == '__main__':
    main()

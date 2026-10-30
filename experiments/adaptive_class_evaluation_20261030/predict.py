"""Canonical LongBench runner using the isolated mixed v2 model loader."""
import argparse
import json

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.adaptive_class_evaluation_20261030.loading import load_model
from experiments.longbench_v1 import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--max-runtime-seconds', type=float)
    args = parser.parse_args()
    runner.check_config, runner.load_model = b.check_config, load_model
    report = runner.run(args.config, args.output, datasets=args.datasets, max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({key: report[key] for key in ('status', 'new_predictions', 'already_present', 'duration_seconds')}), flush=True)


if __name__ == '__main__':
    main()

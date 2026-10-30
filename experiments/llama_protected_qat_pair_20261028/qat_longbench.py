"""Canonical complete21 generation through the validated Llama native sparse loader."""
import argparse
import os
from pathlib import Path
import native_core as n
from experiments.longbench_v1 import runner
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.protocol import DATASETS

runtime = n.c.native.module(n.RUNTIME, '_llama_QAT64_proven_sparse_native_runtime')
n.require(runtime.n is n, 'Bind the proven native runtime to this actual matched64 endpoint adapter.')


def load_model(config, logger):
    n.require(config == n.config(n._arm), 'Keep exact bound full21 protocol and native2.6 adapter.')
    check_hashes(config['model']['evaluation_adapter_sha256'])
    model, audit = runtime.load_model(config, logger)
    audit['Llama_QAT_authorized_budget'] = config['model']['Llama_QAT_authorized_budget']
    return model, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('D', 'DQ'), required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-runtime-seconds', type=int, required=True)
    args = parser.parse_args(); n.bind(args.arm); n.c.check_runtime(args.gpu)
    n.require(Path.cwd() == n.S and os.environ['CUDA_VISIBLE_DEVICES'] == str(args.gpu)
        and not (args.output / 'config.json').exists(), 'Use assigned physical GPU and a fresh full21 output.')
    config = n.read(args.config); check_config(config)
    runner.load_model = load_model
    report = runner.run(args.config, args.output, datasets=list(DATASETS), resume=False,
        max_runtime_seconds=args.max_runtime_seconds)
    n.require(report['status'] == 'partition_complete' and report['exit_code'] == 0
        and report['new_predictions'] == 4750 and report['already_present'] == 0,
        'Complete all21/4750 before scoring; no partial quality verdict or automatic retry.')
    n.check_export(); check_hashes(config['model']['evaluation_adapter_sha256'])

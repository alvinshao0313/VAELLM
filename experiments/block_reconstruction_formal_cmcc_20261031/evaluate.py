"""Use the unchanged official complete-nine evaluator on a real BR36 endpoint."""
import argparse
import importlib
from pathlib import Path
import sys

from . import bindings as r
from .source import source
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def setup(support):
    r.configure_readers(support)
    original = b.source_pins

    def source_pins(binding):
        return {**original(binding), **r.code_bindings()}

    def command(component, *arguments):
        return [str(r.PYTHON), '-B', '-u', '-m', 'experiments.block_reconstruction_formal_cmcc_20261031.evaluate',
                '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--component', choices=('evaluate', 'full8', 'predict', 'summarize', 'merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    setup(args.support.resolve(strict=True))
    b.CLI_ARGUMENTS = list(sys.argv)
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component)
    sys.argv = [sys.argv[0], *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

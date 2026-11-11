"""Original complete-nine consumer with a distinct SciQ source/contract profile."""
import argparse
import importlib
from pathlib import Path
import sys

from . import bindings as r
from experiments.rice_muon_cmcc_20261031 import evaluate as original
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def setup(support):
    r.c1.configure_readers(support)
    r.configure()
    original.r = r
    prior_pins = b.source_pins

    def source_pins(binding):
        return {**prior_pins(binding),**{path:item['sha256'] for path,item in r.code_bindings().items()}}

    def command(component,*arguments):
        return [str(b.PYTHON),'-B','-u','-m','experiments.c1_sciq_recovery_cmcc_20261103.evaluate',
            '--support',str(support),'--component',component,*arguments]

    b.source,b.source_pins,b.command = original.source,source_pins,command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support',required=True,type=Path)
    parser.add_argument('--component',choices=('evaluate','full8','predict','summarize','merge'),default='evaluate')
    args,remaining = parser.parse_known_args()
    setup(args.support.resolve())
    b.CLI_ARGUMENTS = list(sys.argv)
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component)
    sys.argv = [sys.argv[0],*remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

"""Call the original C1 run/fit under this process-only data binding."""
import argparse
from pathlib import Path

from . import bindings as b
from experiments.rice_muon_cmcc_20261031 import run as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--gpu',required=True,type=int,choices=(4,))
    parser.add_argument('--timeout-seconds',type=int,default=14400)
    args = parser.parse_args()
    original.b = b
    original.run(args.prepared,args.gpu,'formal32',args.timeout_seconds)


if __name__ == '__main__':
    main()

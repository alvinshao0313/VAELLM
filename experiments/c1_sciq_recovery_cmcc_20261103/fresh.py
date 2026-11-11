"""Original independent fresh export audit with the isolated SciQ contract."""
import argparse
from pathlib import Path

from . import bindings as b
from experiments.rice_muon_cmcc_20261031 import fresh as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--gpu',required=True,type=int,choices=(4,))
    args = parser.parse_args()
    original.b = b
    original.fresh(args.prepared,args.gpu,'formal32')


if __name__ == '__main__':
    main()

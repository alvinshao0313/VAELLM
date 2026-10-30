"""Reuse the complete Muon strict native exporter with this explicit input binding."""
import argparse
from pathlib import Path

import bindings as b

original = b.module(b.MUON / 'export.py', '_hotpot192_mature_native_exporter')
export = original.export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    b.torch.set_num_threads(4)
    print(export(args.report, args.output_dir), flush=True)

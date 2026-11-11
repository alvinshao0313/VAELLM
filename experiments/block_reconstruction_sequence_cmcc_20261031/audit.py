"""CPU source checks using the actual formal readers; never construct a parent."""
import argparse
import json
from pathlib import Path

from . import bindings as b


def audit(kind, checkpoint, report, output):
    b.executor.cpu_environment()
    b.require(not output.exists(), 'Never overwrite a source audit.')
    from experiments.block_reconstruction_formal_cmcc_20261031 import bindings as formal
    from experiments.block_reconstruction_formal_cmcc_20261031 import recovery_bindings as recovery
    formal.configure_readers()
    parent, _meta = (formal.bind_C0 if kind == 'C0' else recovery.source)(checkpoint, report)
    b.require(not formal.torch.cuda.is_initialized(), 'CPU source audit initialized CUDA.')
    value = dict(schema='BR_sequence_actual_source_audit_v1', status='verified', kind=kind,
        source_binding=parent, CUDA_initialized=False)
    b.write(output, value)
    print(json.dumps(dict(status='verified', kind=kind, checkpoint_id=parent['checkpoint_id'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('C0', 'BR'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.kind, args.checkpoint, args.report, args.output)


if __name__ == '__main__':
    main()

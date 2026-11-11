"""CPU bridge fixture/source/freeze/ledger proof; explicitly ineligible for full8."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as b
from . import bridge, evaluate
from experiments.down_tail23_short_cmcc_20261104 import artifact


def fresh(path, expected, output):
    model, meta, loaded, budget = b.checkpoint.load(path)
    b.c.require(meta['provenance']['phase'] == 'cpu_fixture' and not loaded.missing_keys and not loaded.unexpected_keys
        and b.c.digest(model.state_dict()) == expected['state_digest'] and budget == expected['budget'],
        'Independent CPU bridge fixture did not preserve actual fields/topology/ledger.')
    bridge.topology(model, mode='fixture')
    bridge.verify_frozen(model, expected['freeze'])
    b.c.write(output, dict(status='independent_CPU_fixture_strict_verified', exit_code=0, pid=os.getpid(),
        checkpoint_id=meta['checkpoint_id'], state_digest=expected['state_digest'], actual_budget=budget,
        formal_eligible=False, cuda_initialized=b.torch.cuda.is_initialized()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    b.cpu_environment()
    b.torch.set_num_threads(8)
    if args.fresh:
        fresh(args.fresh, b.c.read(args.expected), args.output)
        return
    b.c.require(not b.PREPARED.exists(), 'Never overwrite a prior bridge preparation.')
    # Import every new actual path before fixing a static source set; output files never enter it.
    from . import capture, fit, run, sequence
    import ast
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    started = time.time()
    raw, live, specs, source = b.source(initial=True)
    short_data = b.short_data_pins()
    b.c.require(b.c.digest(bridge.complement(raw)) == b.c.digest(bridge.complement(live)),
        'The actual raw and live parent must share the exact unchanged non-down complement.')
    old = b.c.read(b.SHORT_RUN / 'report.json')
    b.c.require(old['status'] == 'completed_short_pair' and old['exit_code'] == 0, 'Require the verified short source for diagnostic fixture identity only.')
    model, bundles, frozen = bridge.build(raw, specs, source)
    root = b.OUTPUT / 'cpu_bridge_fixture'
    b.c.require(not root.exists(), 'Use independent fresh CPU bridge fixture output.')
    root.mkdir(parents=True)
    cases = []
    for tag in ('control', 'partial3_k24'):
        if tag != 'control':
            for layer in (0, 18, 35):
                name = f'model.layers.{layer}.mlp.down_proj'
                path = b.SHORT_RUN / (name + '.K24.pt')
                reference = next(row for row in old['arms']['24']['layers'] if row['name'] == name)
                b.c.require(b.c.sha(path) == reference['artifact_sha256'], 'The protected real K24 short fixture changed.')
                _bundle, _lengths, module = artifact.load(path)
                b.checkpoint.native.set_module_by_name(model, name, module)
        bridge.topology(model, mode='control' if tag == 'control' else 'fixture')
        bridge.verify_frozen(model, frozen)
        directory = root / tag
        directory.mkdir()
        meta, budget, state = b.checkpoint.save(model, directory / 'model',
            provenance=bridge.provenance(tag, phase='cpu_fixture', source=source))
        evidence = dict(state_digest=b.c.digest(state), budget=budget, freeze=frozen)
        b.c.write(directory / 'expected.json', evidence)
        b.c.write(directory / 'report.json', dict(schema=b.SCHEMA, phase='cpu_fixture', arm=tag,
            status='cpu_fixture_only', exit_code=0, checkpoint_id=meta['checkpoint_id'], formal_eligible=False))
        try:
            evaluate.binding(directory / 'model', directory / 'report.json')
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError('The full8 source binding accepted a CPU fixture.')
        command = [sys.executable, '-B', '-u', '-m', 'experiments.down_tail24_full_cmcc_20261104.prepare',
            '--fresh', str(directory / 'model'), '--expected', str(directory / 'expected.json'),
            '--output', str(directory / 'fresh.json')]
        with (directory / 'fresh.log').open('x') as log:
            subprocess.run(command, cwd=b.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        verified = b.c.read(directory / 'fresh.json')
        cases.append(dict(tag=tag, checkpoint_id=meta['checkpoint_id'], budget=budget,
            state_digest=evidence['state_digest'], independent_fresh=verified,
            full8_fixture_rejected=True, rejection=rejection, formal_eligible=False))
        del state
        gc.collect()
    pins = b.pins()
    b.check_pins(short_data)
    b.c.write(b.PREPARED, dict(status='CPU_full_bridge_verified', exit_code=0, created_unix=time.time(),
        pid=os.getpid(), command=[sys.executable, *sys.argv], source_binding=source, source_sha256=pins,
        short_identity_files_sha256=short_data,
        native_full8_reference=b.native_reference(),
        source_raw_native_full8_mean_percent=67.01259272688102,
        frozen_identity=frozen, expected_groups=b.GROUPS, actual_topology=[180, 72, 0], CPU_fixtures=cases,
        calibration_token_digest=b.c.digest({'input_ids': b.calibration()}),
        elapsed_seconds=time.time() - started, cuda_initialized=b.torch.cuda.is_initialized(),
        boundary='Only complete bridge source/freeze/topology/alias/full-ledger/save-load and fixture rejection. No new fit/GPU/bank audit or formal quality claim.'))
    print(dict(status='CPU_full_bridge_verified', output=str(b.PREPARED), seconds=time.time() - started), flush=True)


if __name__ == '__main__':
    main()

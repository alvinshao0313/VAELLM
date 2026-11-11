"""CPU-only actual binding/static checks for this fixed parallel batch."""
import argparse
import ast
from pathlib import Path
import time

from . import bindings as b
from experiments.cat_auto_protocol_cmcc_20261104.search import validate_request
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--batch-root', required=True); parser.add_argument('--report', required=True)
    args = parser.parse_args(); started = time.monotonic()
    original, pilot, inventory, identity = b.common()
    validate = validate_request
    checks = []
    now = time.time()
    for label, card in b.CARDS.items():
        phase = 'formal' if label == 'k32' else 'technical_validation'
        request = b.make_request(label, phase, output=Path(args.batch_root) / label / phase,
            chain_deadline=now + card['technical_seconds'] + card['formal_seconds'], now=now)
        validate(request)
        require(request['run_identity'] == original['run_identity'] and request['candidates'][0]['recipe']['K'] == card['K'], 'The pilot fit identity must remain exactly reusable.')
        if label != 'k32':
            bad = b.make_request(label, 'formal', output='/not_executed', chain_deadline=now + 3600, technical_receipt=b.PILOT / 'completed.json', now=now)
            try:
                validate(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('A different K reused the K32 receipt without actual GPU shape evidence.')
        checks.append(dict(label=label, gpu=card['gpu'], rate=card['rate'], K=card['K'], initial_phase=phase,
            actual_validation_passed=True, K32_receipt_reused=label == 'k32', own_new_K_receipt_required=label != 'k32'))
    paths = sorted(b.HERE.glob('*.py'))
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))
    import torch
    require(not torch.cuda.is_initialized(), 'CPU binding/static work initialized CUDA.')
    batch = Path(args.batch_root).resolve(); require(not batch.exists(), 'Use a new batch root.')
    batch.mkdir(parents=True)
    source = b.pins()
    manifest = dict(schema='fixed_parallel_up_profiles_v0', source_sha256=source, cards=b.CARDS,
        batch_GPU_hours=2.75, phase_formal_seconds=2700, new_K_technical_seconds=900,
        model_identity=inventory['model_identity'], data_identity=identity['data_identity'], original_pilot=str(b.PILOT),
        checks=checks, no_full8=True, no_KD=True, no_retry=True, technical_tests_parallel_on_distinct_GPU3_and4=True)
    b.write(batch / 'batch.json', manifest)
    report = dict(status='actual_CPU_binding_and_static_verified', exit_code=0, cuda_initialized=False,
        elapsed_seconds=time.monotonic() - started, source_sha256=source, checks=checks,
        old_core_sources_unchanged=True, fit_identity_matches_actual_K32_receipt=True,
        scope='Binding/registry/host-policy/static checks only; no new K GPU result is claimed')
    b.write(args.report, report)
    print({key: report[key] for key in ('status', 'exit_code', 'elapsed_seconds', 'cuda_initialized')}, flush=True)


if __name__ == '__main__':
    main()

"""One real protected-q two-update proof or one matched fresh64 arm."""
import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
import qat_common as c
from qat_common import retention
from qat_fit import fit
from qat_export import export_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('short2', 'formal64'))
    parser.add_argument('--arm', choices=('D', 'DQ'), required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args(); c.check_runtime(args.gpu); record = c.prepared()
    c.require(args.phase != 'short2' or args.arm == 'DQ', 'New-path short2 uses DQ; mature D-only evidence is reused.')
    if args.phase == 'formal64':
        short = c.read(c.HERE / 'DQ/short2/report.json')
        c.require(short['status'] == 'complete' and short['exit_code'] == 0 and short['training']['steps'] == 2
            and short['source_checkpoint_id'] == c.SOURCE_ID and short['prepared_sha256'] == c.sha(c.HERE / 'prepared.json')
            and short['training']['group_indices'] == c.SHORT_GROUPS and short['training']['source_restored_exact']
            and short['export']['status'] == 'verified' and short['export']['reloaded_tensors_compared'] == 3619,
            'Require completed real S4/sparse/protected-q two-update proof.')
    output = c.HERE / args.arm / args.phase; output.mkdir(parents=True, exist_ok=True)
    path = output / 'report.json'; c.require(not path.exists(), 'Fresh explicit run only; no retry/overwrite.')
    started = time.monotonic(); budget = 7200 if args.phase == 'short2' else 43200
    report = {'status': 'running', 'source_checkpoint_id': c.SOURCE_ID, 'phase': args.phase, 'arm': args.arm,
        'prepared_sha256': c.sha(c.HERE / 'prepared.json'), 'actual_argv': sys.argv, 'pid': os.getpid(),
        'gpu': args.gpu, 'seed': c.SEED, 'python': sys.executable, 'started_unix': time.time(),
        'budget_seconds': budget, 'automatic_retry': False, 'quality_evaluated': False}
    c.write(path, report)
    observe = lambda event, values: print(json.dumps({'event': event, **values}), flush=True)
    try:
        data, original = c.inputs(); targets = c.training.load_targets(data, original)
        archive, _ = retention.load_retention(); r_targets = retention.load_targets()
        model, audit = c.load_student(); report['load_audit'] = audit
        training = fit(model, data, targets, archive, r_targets, record, args.arm, args.phase,
            c.DATA / args.arm / args.phase, observe, started + budget)
        report['training'] = training
        del model, targets, r_targets
        gc.collect(); c.torch.cuda.empty_cache(); c.torch._C._cuda_clearCublasWorkspaces(); c.torch.cuda.empty_cache()
        report['allocated_after_model_release'] = c.torch.cuda.memory_allocated()
        c.require(report['allocated_after_model_release'] == 0, 'Release GPU tensors before CPU-only native export.')
        c.write(path, report)
        report['export'] = export_endpoint(record, training, c.DATA / args.arm / args.phase / 'native_model')
        report.update(status='complete', exit_code=0)
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, finished_unix=time.time()); c.write(path, report)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s'); main()

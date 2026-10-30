"""CPU-only real A250 export and A500 control audit, each with independent fresh."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cat_category_selection_20261029 import sources as s
from experiments.cat_category_selection_20261029 import export as e


def prepare():
    s.cpu_only()
    s.require(not s.DATA.exists(), 'Use the new independent category-selection directory; never overwrite.')
    s.DATA.mkdir()
    report = dict(schema='A_gate250_500_readonly_native_export_v1', status='checking_real_CPU_sources',
        pid=os.getpid(), start_unix=time.time(), command=[sys.executable, *sys.argv],
        source_sha256=s.own_pins(), endpoints={}, selection_status='not_evaluated_not_selected', GPU_used=False,
        structure_fixed_before_evaluation='A D32/K32/S2 asymmetric gate; original A/B fixed500 untouched')
    path = s.DATA / 'report.json'
    s.write(path, report)
    try:
        raw_meta, raw, nodes = s.source_metadata()
        stage, training = s.original_stage()
        report.update(raw_parent_checkpoint_id=s.RAW_ID, common_after_o_checkpoint_id=s.BASE_ID,
                      source_raw_archive=s.binding(s.RAW / 'manifest.json'), source_nodes={str(step): s.binding(s.NODES / f'gate{step}/manifest.json') for step in nodes})
        for step in (250, 500):
            report['phase'] = f'restore_finalize_{step}'
            s.write(path, report)
            model, info, dense = e.restore_finalize(step, raw_meta, raw, nodes[step][0], stage, training)
            endpoint = e.save250(model, raw_meta, nodes[step][0], info) if step == 250 else e.control500(model, raw_meta, nodes[step][0], info, dense)
            report['endpoints'][str(step)] = endpoint
            s.write(path, report)
            del model
            gc.collect()
            report['phase'] = f'independent_fresh_{step}'
            s.write(path, report)
            child = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__)), '--fresh', str(step)],
                                   cwd=ROOT, check=True, stdin=subprocess.DEVNULL)
            report['endpoints'][str(step)]['fresh'] = s.read(s.DATA / f'fresh_{step}.json')
            s.write(path, report)
            (s.DATA / f'fresh_{step}.json').unlink()
        from experiments.cat_category_selection_20261029.evaluate import prepare as evaluation_plan
        plan = evaluation_plan(report)
        report.update(status='CPU_exports_fresh_budget_and_full8_inputs_prepared', exit_code=0,
            evaluation_plan=str(s.DATA / 'evaluation_plan.json'), full8_tasks=plan['leaf_tasks'], full8_samples=plan['sample_count'],
            CUDA_initialized=s.torch.cuda.is_initialized(), GPU_inference_verified=False, full8_started=False,
            seconds=time.time() - report['start_unix'])
        s.write(path, report)
        print(dict(status=report['status'], seconds=report['seconds']), flush=True)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc(), seconds=time.time() - report['start_unix'])
        s.write(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', choices=(250, 500), type=int)
    args = parser.parse_args()
    s.torch.set_num_threads(4)
    if args.fresh is not None:
        row = s.read(s.DATA / 'report.json')['endpoints'][str(args.fresh)]
        result = e.verify(row['checkpoint_dir'], row['state_digest'])
        result['pid'] = os.getpid()
        s.write(s.DATA / f'fresh_{args.fresh}.json', result)
        print(dict(status=result['status'], step=args.fresh, pid=os.getpid()), flush=True)
    else:
        prepare()


if __name__ == '__main__':
    main()

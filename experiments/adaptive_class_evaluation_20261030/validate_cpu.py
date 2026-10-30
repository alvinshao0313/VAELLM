"""Check real v2 tensors, canonical archive/baselines and all partition layouts."""
import argparse
import json
from pathlib import Path
import time
import traceback

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.adaptive_class_evaluation_20261030.evaluate import canonical
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS
from experiments.w25_eval.acceptance import collect_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    b.cpu_environment()
    b.require(not args.output.exists(), 'Never overwrite CPU validation evidence.')
    args.output.mkdir(parents=True)
    report = dict(status='running', checkpoint_dir=str(args.checkpoint.resolve()), started_unix=time.time())
    try:
        meta, budget, hashes = b.inspect(args.checkpoint, final=False)
        partial_rejected = False
        if meta['pending_dense_modules'] or len(meta['compact_modules']) != 36:
            try:
                b.inspect(args.checkpoint)
            except ValueError as error:
                partial_rejected = True
                report['final_rejection_reason'] = str(error)
            b.require(partial_rejected, 'An incomplete boundary must never be accepted as a final model.')
        binding = dict(checkpoint_dir=str(args.checkpoint.resolve()), checkpoint_id=meta['checkpoint_id'],
            metadata_sha256=hashes[str(args.checkpoint.resolve() / 'adaptive_model_meta.json')],
            checkpoint_files_sha256=hashes, checkpoint_weight_stat=b.file_stat(args.checkpoint / 'pytorch_model.bin'), bit_budget=budget)
        pins = b.source_pins(binding)
        config = b.make_config(binding, pins, mode='validation')
        reference = b.read(b.REFERENCE)
        b.require(all(config[key] == reference[key] for key in reference if key not in ('model', 'mode')),
                  'Generation/archive/package policy must exactly match the original reference.')
        manifest, rows = load_archive(config['archive']['path'])
        b.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS, 'Incomplete canonical archive.')
        layouts = {}
        api = canonical()
        for count in range(1, 9):
            groups = api.task_groups(list(map(str, range(count))))
            assigned = [task for group in groups for task in group]
            b.require(len(assigned) == len(set(assigned)) == 21 and set(assigned) == set(DATASETS), 'Duplicate/missing official task.')
            layouts[str(count)] = groups
        inventory = b.model_inventory(b.BASE)
        baseline_paths, baseline_lb = b.template_inputs()
        baseline = collect_results(baseline_paths, inventory)
        b.require(baseline['leaf_count'] == 64 and baseline['sample_count'] == 24742, 'Original full8 baseline changed.')
        lb = b.read(baseline_lb)
        b.require(lb['status'] == 'complete' and lb['sample_count'] == 4750 and set(lb['tasks']) == set(DATASETS)
            and lb['protocol'] == config['protocol'], 'Original full21 BF16 reference changed.')
        samples = b.module(b.SAMPLE_DIR / 'eval_utils_with_samples.py', '_adaptive_cpu_existing_hflm_samples')
        b.require(callable(samples.run_lm_eval) and callable(samples.write_lm_eval_samples), 'Original HFLM/raw artifact implementation unavailable.')
        b.check_config(config); b.check_hashes(pins)
        report.update(status='verified_cpu_format_and_protocol', checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
            checkpoint_files_sha256=hashes, source_sha256=pins, bit_budget=budget,
            final_endpoint_eligible=not meta['pending_dense_modules'] and len(meta['compact_modules']) == 36,
            incomplete_boundary_rejected_as_final=partial_rejected, full8_leaves=64, full8_samples=24742,
            full21_tasks=21, full21_samples=4750, partition_layouts=layouts,
            boundary='Actual CPU serialized-format/protocol checks only. No model forward, strict model load, full quality or final2.6 claim for partial inputs.')
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        b.write_json(args.output / 'report.json', report)
    print(json.dumps(dict(status=report['status'], checkpoint_id=report['checkpoint_id'])), flush=True)


if __name__ == '__main__':
    main()

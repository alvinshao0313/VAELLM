"""One new full LongBench21 supplement for completed G CE10/CE50 full8 donors."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01'
sys.path.insert(0, str(SNAPSHOT))
spec = importlib.util.spec_from_file_location('night_donor_native_longbench', ROOT / 'experiments/cat_recompression/longbench.py')
longbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longbench)
from experiments.longbench_v1.models import make_config
from experiments.longbench_v1.common import check_hashes, sha256, write_json

sys.path.insert(0, str(ROOT / 'experiments/night_sequence_20261026'))
import recovery_eval
from recovery_core import TOPIC, check_gpu, module_from, own_sources, read, require, stat

F = TOPIC / 'qwen_recompression_followup_20261026'
DATA = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026')
JOBS = {
    'ce10': dict(gpu='0', uuid='bde0db08-b61b-436c-999d-83e67d96abe1',
        relative='repaired_recovery/G/ce10', stamp='20261027_001201', mean=66.6956574663),
    'ce50': dict(gpu='7', uuid='3b82e64b-28b7-45a3-942e-296134daabb6',
        relative='recovery/G/ce50', stamp='20261027_000607', mean=67.2976083420),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor', required=True, choices=list(JOBS))
    args = parser.parse_args()
    job = JOBS[args.donor]
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == job['gpu'], 'Use exactly the assigned donor GPU.')
    check_gpu(job['gpu'])
    output = F / job['relative'] / 'evaluation_full8'
    checkpoint = DATA / job['relative'] / 'training' / ('home_star_models_Qwen_Qwen3-8B_' + job['stamp']) / 'final_model'
    supplement = output / 'longbench_supplement'
    report_path = supplement / 'report.json'
    require(output.is_dir() and not report_path.exists() and not (output / 'longbench').exists()
        and not (output / 'longbench_config.json').exists() and not (output / 'nine_task_acceptance.json').exists(),
        'Use the original full8 directory once; no overwrite, retry, resume or duplicate LongBench.')
    original = read(output / 'evaluation.json')
    meta = read(checkpoint / 'checkpoint_meta.json')
    require(original['status'] == 'complete_full8' and original['checkpoint_id'] == meta['checkpoint_id'] == job['uuid']
        and Path(original['checkpoint_dir']).resolve() == checkpoint.resolve(), 'Actual completed donor identity changed.')
    protected = [output / name for name in ('evaluation.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'mmlu.exit', 'seven.exit')]
    preserved = {str(path): sha256(path) for path in protected}
    supplement.mkdir(parents=True, exist_ok=True)
    record = dict(status='running', donor=args.donor, gpu=job['gpu'], checkpoint_id=job['uuid'],
        checkpoint_dir=str(checkpoint), output=str(output), pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(),
        max_runtime_seconds=86400, full8_rerun=False, resume=False, automatic_retry=False,
        original_full8_report_modified=False, preserved_sha256=preserved,
        runtime_note='Original runner checks24h between samples; an official in-flight generation is not truncated.')
    write_json(report_path, record)
    try:
        # Reuse the exact existing verifier while preserving every existing summary byte.
        original_write = recovery_eval.write
        def preserve_or_write(path, value):
            path = Path(path)
            if path.exists():
                require(read(path) == value, 'Existing summary differs: ' + str(path))
            else:
                original_write(path, value)
        recovery_eval.write = preserve_or_write
        full8, baseline_paths, template = recovery_eval.full8_summary(checkpoint, output)
        require(abs(full8['mean_percent'] - job['mean']) < 1e-9, 'The confirmed full8 donor score changed.')
        api = module_from(TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_night_donor_helpers')
        reference = read(api.REFERENCE_CONFIG)
        config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
            checkpoint=checkpoint, mode='full')
        require(all(config[key] == reference[key] for key in reference if key != 'model'),
            'Keep the original full21 archive,32K policy and complete official generation protocol.')
        paths = [Path(__file__), ROOT / 'experiments/cat_recompression/longbench.py',
            checkpoint / 'checkpoint_meta.json', *own_sources(), *api.source_paths()]
        pins = {str(path.resolve()): sha256(path) for path in paths}
        check_hashes(reference['source_sha256'])
        pins.update(config['source_sha256'])
        record['reference_source_sha256'] = reference['source_sha256']
        config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0,
            bit_budget=original['bit_budget'], source_sha256=pins)
        config_path = output / 'longbench_config.json'
        write_json(config_path, config)
        record.update(source_sha256=pins, config_path=str(config_path), config_sha256=sha256(config_path),
            full8_mean_percent=full8['mean_percent'], full8_sample_count=full8['sample_count'],
            checkpoint_weight_stat=stat(checkpoint / meta['state_dict_file']))
        write_json(report_path, record)
        longbench.runner.load_model = longbench.load_model
        result = longbench.runner.run(config_path, output / 'longbench', resume=False, max_runtime_seconds=86400)
        record['longbench_run'] = result
        require(result['status'] == 'partition_complete' and result['exit_code'] == 0
            and result['already_present'] == 0 and result['new_predictions'] == 4750,
            'The one-shot complete donor LongBench did not finish; do not score partial predictions.')
        check_hashes(preserved)
        check_hashes(pins)
        nine = recovery_eval.summarize_nine(checkpoint, output, config_path, baseline_paths, template)
        check_hashes(preserved)
        check_hashes(pins)
        require(stat(record['checkpoint_weight_stat']['path']) == record['checkpoint_weight_stat'], 'Native donor weights changed.')
        record.update(status='complete_existing_full8_plus_full21_nine', exit_code=0,
            nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'],
            nine=str(output / 'nine_task_acceptance.json'), scores=str(output / 'longbench/scores.json'))
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        write_json(report_path, record)
    print(dict(status=record['status'], nine_task_score=record['nine_task_score'], passed=record['passed']), flush=True)


if __name__ == '__main__':
    main()

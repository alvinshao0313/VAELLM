"""Complete LongBench21 for the fixed D-CE50 donor, preserving its completed full8."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01'
sys.path.insert(0, str(SNAPSHOT))
spec = importlib.util.spec_from_file_location('d_donor_native_longbench', ROOT / 'experiments/cat_recompression/longbench.py')
longbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longbench)
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.common import check_hashes, sha256, write_json

sys.path.insert(0, str(ROOT / 'experiments/night_sequence_20261026'))
import recovery_eval
from recovery_core import TOPIC, check_gpu, module_from, own_sources, read, require, stat

DONOR = TOPIC / 'qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027/ce50'
OUTPUT = DONOR / 'evaluation_full8'
SUPPLEMENT = OUTPUT / 'longbench_supplement_20261028'
UUID = '68fad835-d938-4280-8213-3e9a1ecd2616'
GPU = '4'


def prepare():
    require(not SUPPLEMENT.exists() and not list(OUTPUT.glob('longbench*'))
        and not (OUTPUT / 'nine_task_acceptance.json').exists(),
        'Use this donor once; no overwrite, duplicate LongBench, retry or resume.')
    pipeline = read(DONOR / 'pipeline.json')
    checkpoint = Path(pipeline['final_model']).resolve()
    original = read(OUTPUT / 'evaluation.json')
    audit = read(DONOR / 'native_audit.json')
    meta = read(checkpoint / 'checkpoint_meta.json')
    require(pipeline['status'] == 'complete_donor' and pipeline['profile'] == 'D'
        and pipeline['recipe'] == 'ce50' and original['status'] == 'complete_full8'
        and audit['status'] == 'verified'
        and pipeline['checkpoint_id'] == original['checkpoint_id'] == audit['checkpoint_id'] == meta['checkpoint_id'] == UUID
        and Path(original['checkpoint_dir']).resolve() == Path(audit['checkpoint_dir']).resolve() == checkpoint,
        'Use the actual audited complete D-CE50 native donor and its completed full8.')
    protected = [OUTPUT / name for name in ('evaluation.json', 'full8_summary.json', 'mmlu.json',
        'seven.json', 'mmlu.exit', 'seven.exit')]
    protected += sorted(OUTPUT.rglob('*.jsonl')) + sorted(OUTPUT.rglob('manifest.json'))
    preserved = {str(path): sha256(path) for path in protected}
    original_write = recovery_eval.write
    def preserve_or_write(path, value):
        require(Path(path).exists() and read(path) == value, 'Existing full8 summary differs: ' + str(path))
    recovery_eval.write = preserve_or_write
    try:
        full8, baseline_paths, template = recovery_eval.full8_summary(checkpoint, OUTPUT)
    finally:
        recovery_eval.write = original_write
    require(full8['sample_count'] == 24742 and full8['leaf_count'] == 64
        and abs(full8['mean_percent'] - 66.02881802097906) < 1e-12
        and abs(full8['task_metrics']['mmlu'] * 100 - 56.37373593505198) < 1e-12,
        'The confirmed full8 donor score or complete sample coverage changed.')
    api = module_from(TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_d_donor_helpers')
    reference = read(api.REFERENCE_CONFIG)
    config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
        checkpoint=checkpoint, mode='full')
    require(all(config[key] == reference[key] for key in reference if key != 'model'),
        'Keep the reference archive,32K policy and complete official21 generation protocol, including source hashes.')
    for name in ('rotation.weight_preprocess', 'train_utils.v6_model_loader', 'experiments.longbench_v1.runner'):
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(SNAPSHOT),
            'Use the original extended full-rotation snapshot: ' + name)
    paths = [Path(__file__), Path(__file__).with_name('run.sh'), ROOT / 'experiments/cat_recompression/longbench.py',
        DONOR / 'pipeline.json', DONOR / 'native_audit.json', checkpoint / 'checkpoint_meta.json',
        TOPIC / 'qwen_b32_mlp_s3_20261023/summarize_nine.py', TOPIC / 'qwen_b32_mlp_s3_20261023/s3_bindings.py',
        *own_sources(), *api.source_paths()]
    pins = {str(path.resolve()): sha256(path) for path in paths}
    check_hashes(reference['source_sha256'])
    pins.update(config['source_sha256'])
    config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0,
        bit_budget=original['bit_budget'], source_sha256=pins)
    check_config(config)
    check_hashes(preserved)
    return checkpoint, config, pins, preserved, full8, baseline_paths, template


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Use exactly assigned GPU4.')
    check_gpu(GPU)
    checkpoint, config, pins, preserved, full8, baseline_paths, template = prepare()
    SUPPLEMENT.mkdir(parents=True, exist_ok=False)
    config_path = OUTPUT / 'longbench_config.json'
    write_json(config_path, config)
    record = dict(status='running', profile='D', donor='ce50', gpu=GPU, checkpoint_id=UUID,
        checkpoint_dir=str(checkpoint), output=str(OUTPUT), pid=os.getpid(),
        tmux_session='d_ce50_lb21_20261028_gpu4', log=str(DONOR / 'longbench_20261028.log'),
        outer_exit=str(DONOR / 'longbench_20261028.exit'),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(),
        config_path=str(config_path), config_sha256=sha256(config_path), source_sha256=pins,
        reference_source_sha256=read(TOPIC / 'qwen_a0e5_baseline_s400_longbench21/config.json')['source_sha256'],
        preserved_sha256=preserved, checkpoint_weight_stat=stat(config['model']['weight_stat']['path']),
        full8_mean_percent=full8['mean_percent'], full8_sample_count=full8['sample_count'],
        max_runtime_seconds=86400, full8_rerun=False, resume=False, automatic_retry=False,
        original_full8_report_modified=False,
        runtime_note='Original runner checks24h between samples; an official in-flight generation is not truncated.',
        driver=subprocess.check_output(['nvidia-smi', '--id=4',
            '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip(),
        environment_note='Driver590.48.01 after server reboot; unchanged bitvae/PyTorch and original prediction protocol.')
    report_path = SUPPLEMENT / 'report.json'
    write_json(report_path, record)
    try:
        longbench.runner.load_model = longbench.load_model
        result = longbench.runner.run(config_path, OUTPUT / 'longbench', resume=False, max_runtime_seconds=86400)
        record['longbench_run'] = result
        require(result['status'] == 'partition_complete' and result['exit_code'] == 0
            and result['already_present'] == 0 and result['new_predictions'] == 4750,
            'The complete donor LongBench did not finish; do not score partial predictions.')
        check_hashes(preserved)
        check_hashes(pins)
        nine = recovery_eval.summarize_nine(checkpoint, OUTPUT, config_path, baseline_paths, template)
        check_hashes(preserved)
        check_hashes(pins)
        require(stat(record['checkpoint_weight_stat']['path']) == record['checkpoint_weight_stat'],
            'Native donor weights changed.')
        record.update(status='complete_existing_full8_plus_full21_nine', exit_code=0,
            nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'],
            nine=str(OUTPUT / 'nine_task_acceptance.json'), scores=str(OUTPUT / 'longbench/scores.json'))
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        write_json(report_path, record)
    print(dict(status=record['status'], nine_task_score=record['nine_task_score'], passed=record['passed']), flush=True)


if __name__ == '__main__':
    main()

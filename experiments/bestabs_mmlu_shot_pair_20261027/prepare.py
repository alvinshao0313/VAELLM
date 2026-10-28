"""Bind two native bestabs MMLU runs to reusable complete same-B16 BF16 controls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PREVIOUS = RESULT / 'qwen_mmlu_shot_pair_20261009'
BEST = RESULT / 'qwen_b32_sparse_mlp005_probe_20261024'
SNAPSHOT = RESULT / 'rotation_factor_probe/source_snapshot_20261005_01'
CONSUMER = RESULT / 'lm_eval_sample_artifacts_20261010'
CHECKPOINT = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004') / BEST.name / 'native_export/final_model'
SOURCE_ID = 'e01b09b7-a85d-4534-a5d2-5ab3b91312e6'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(), 'Use CPU-only confirmed bitvae.')
    require(not args.output_dir.exists(), 'Keep this source-bound preparation separate from every older result.')
    started = time.time()
    previous = read(PREVIOUS / 'launch.prepared.json')
    spec = importlib.util.spec_from_file_location('_bestabs_existing_shot_validator', PREVIOUS / 'analyze.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    require(sha(PREVIOUS / 'launch.prepared.json') == prior.PREPARED_SHA, 'Original shot preparation identity changed.')
    prior.check_pins(previous['source_sha256'])
    prior.check_pins(previous['hf_source']['metadata_sha256'])
    for item in (*previous['hf_source']['weight_files'], *previous['dataset_cache_stats']):
        prior.check_stat(item)
    versions = {name: importlib.metadata.version(name) for name in previous['package_versions']}
    require(versions == previous['package_versions'], 'Do not reuse BF16 controls across changed evaluation package versions.')
    baselines, configs, task_versions = {}, {}, {}
    for shot in (0, 5):
        directory = PREVIOUS / f'bf16_shot{shot}'
        require((directory / 'outer.exit').read_text().strip() == '0', 'The same-B16 BF16 control did not complete.')
        paths = sorted((directory / 'native').rglob('results_*.json'))
        require(len(paths) == 1, 'Use exactly one original complete BF16 native result.')
        baseline, configs[str(shot)], task_versions[str(shot)] = prior.extract(paths[0], 'bf16', shot, 16, previous)
        baselines[str(shot)] = baseline
    audit_path = BEST / 'native_export/audit.json'
    audit = read(audit_path)
    meta = read(CHECKPOINT / 'checkpoint_meta.json')
    old_source = read(RESULT / 'qwen_sparse005_rte_gold_short2_20261024/prepared.json')
    weight = CHECKPOINT / meta['state_dict_file']
    stat = weight.stat()
    weight_stat = dict(path=str(weight), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    require(weight_stat == old_source['source_weight_stat'], 'The audited native bestabs weights changed.')
    require(meta['checkpoint_id'] == audit['checkpoint_id'] == SOURCE_ID and meta['lora_config'] is None
        and not meta.get('post_norm_head_linear') and audit['status'] == 'verified'
        and audit['fresh_strict_reload_state_exact'] and audit['fresh_strict_tensors_compared'] == 4071
        and audit['original_banks_compared'] == 504 and audit['no_LoRA_optimizer_or_sidecar'], 'Use the actual rank0 independently reloadable bestabs4071 endpoint.')
    budget = audit['bit_budget']
    require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
        and budget['user_linear_bits'] * 5 <= budget['original_linear_weights'] * 13, 'The complete actual logical2.6 budget failed.')
    source_evaluation = read(BEST / 'evaluation_full9/evaluation.json')
    source_mmlu = read(BEST / 'evaluation_full9/mmlu.json')
    source_nine_path = BEST / 'evaluation_full9/longbench/merged/nine_task_acceptance.json'
    source_nine = read(source_nine_path)
    require(source_evaluation['status'] == 'complete_full8_full21_nine'
        and source_evaluation['checkpoint_id'] == source_mmlu['checkpoint_id'] == SOURCE_ID
        and source_mmlu['checkpoint_loader'] == 'v6' and source_mmlu['strict_load']
        and source_mmlu['evals']['cache_warmup']['warmed'] == 252
        and source_mmlu['evals']['lm_eval']['num_fewshot'] == 0
        and source_mmlu['evals']['lm_eval']['batch_size'] == 64
        and abs(source_nine['compressed_nine_task_score'] - 67.02351209443798) < 1e-12, 'The existing bestabs full-nine control changed.')
    consumer_paths = [CONSUMER / 'cat_eval.py', CONSUMER / 'eval_utils_with_samples.py',
        SNAPSHOT / 'train_utils/v6_model_loader.py', SNAPSHOT / 'litebsq/vae_linear.py',
        SNAPSHOT / 'litebsq/sparse_residual.py']
    require(all(source_evaluation['source_sha256'][str(path)] == sha(path) for path in consumer_paths), 'The consumer differs from the actual completed bestabs MMLU loader.')
    commands = {}
    for shot, gpu in ((0, 2), (5, 7)):
        directory = args.output_dir / f'bestabs_shot{shot}'
        commands[str(shot)] = dict(gpu=gpu, cwd=str(SNAPSHOT), argv=[sys.executable, str(CONSUMER / 'cat_eval.py'),
            '--checkpoint_dir', str(CHECKPOINT), '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0',
            '--tasks', 'mmlu', '--num_fewshot', str(shot), '--lm_batch_size', '16', '--tokenizer_use_fast', 'true',
            '--prewarm_group_size', '8', '--eval_log_dir', str(directory / 'native_logs'),
            '--output_json', str(directory / 'mmlu.json')], outer_timeout_seconds=7200, termination_grace_seconds=20)
    evidence = [Path(__file__), PREVIOUS / 'launch.prepared.json', PREVIOUS / 'paired_report.json',
        PREVIOUS / 'analyze.py', audit_path, CHECKPOINT / 'checkpoint_meta.json', source_nine_path,
        BEST / 'evaluation_full9/evaluation.json', BEST / 'evaluation_full9/mmlu.json',
        BEST / 'evaluation_full9/seven.json', BEST / 'evaluation_full9/longbench/merged/scores.json',
        *consumer_paths]
    report = dict(schema='bestabs_same_B16_native_mmlu_shot_pair_20261027_v1', status='prepared_no_GPU_started',
        source_checkpoint=str(CHECKPOINT), source_checkpoint_id=SOURCE_ID, source_weight_stat=weight_stat,
        source_audit=dict(persistent_tensors=4071, main_banks=504, lora_rank=0,
            logical_bpw=budget['user_linear_bpw'], packed_bpw=budget['packed_linear_bpw'],
            budget_boundary='Original logical<=2.6 complete accounting; packed padding is reported separately, not hidden.'),
        reused_BF16=baselines, package_versions=versions, subjects=previous['subjects'],
        protocol=previous['protocol'], length_summary=previous['length_summary'],
        max_length=previous['native_hflm_max_length'], baseline_task_configs=configs, baseline_task_versions=task_versions,
        source_sha256={str(path): sha(path) for path in evidence},
        reusable_baseline_pins=previous['source_sha256'], hf_source=previous['hf_source'],
        dataset_cache_stats=previous['dataset_cache_stats'], commands=commands,
        environment={**previous['environment'], 'PYTHONPATH': str(SNAPSHOT)},
        validation='Prior complete BF160/5 native count/protocol validator reused; installed packages, original task/harness sources, full HF weight stats/metadata and dev/test cache identities verified.',
        comparison_boundary='Two new bestabs B16 runs only. Reuse the same-B16 BF160/5 controls. Preserve original B64/0shot report; replace only MMLU in each same-setting nine-term report, retaining seven+completeLB. Disclose post-observation reasonable-shot selection; never use the old 0shot-only acceptance checker for 5shot.',
        started_unix=started, finished_unix=time.time())
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / 'prepared.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], prepared=str(path), sha256=sha(path),
        BF16_scores={shot: value['score_percent'] for shot, value in baselines.items()},
        source_audit=report['source_audit'], elapsed_seconds=time.time() - started), indent=2), flush=True)


if __name__ == '__main__':
    main()

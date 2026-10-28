"""One original full8 and complete21-task LongBench evaluation per raw Rice bundle."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import sparse_common as c
import sparse_runtime as runtime


def collect_results(paths, inventory, *, checkpoint=None):
    from experiments.w25_eval import acceptance as original
    from experiments.qwen3_w25.summarize_results import TASK_METRICS, _score
    if checkpoint is None:
        return original.collect_results(paths, inventory)
    directory, meta = checkpoint
    _, audit = runtime.check_audit(directory)
    scores, subjects, sources, versions = {}, {}, [], {}
    c.require(len(paths) == 2 and meta['format'] == c.FORMAT, 'Exactly two truthful same-bundle full8 shards.')
    for raw_path in paths:
        path = Path(raw_path).resolve()
        payload = c.read(path)
        c.require(Path(payload['checkpoint_dir']).resolve() == directory and payload['checkpoint_id'] == meta['checkpoint_id']
                  and Path(payload['base_model_path']).resolve() == Path(inventory['base_model_path'])
                  and payload['checkpoint_loader'] == c.FORMAT and payload['strict_load'] is True
                  and payload['adapter_dir'] is None and payload['eval_hif4_act'] is False
                  and payload['tokenizer_class'] == inventory['tokenizer_class'] and payload['tokenizer_is_fast'] is True,
                  'Require actual strict bundle/full8/original tokenizer identity.')
        runtime.check_loaded(payload['bundle_load_audit'], meta['checkpoint_id'], audit['bit_budget'])
        result = payload['evals']['lm_eval']
        c.require(result.get('num_fewshot') == 0 and result.get('limit', 'missing') is None,
                  'The existing full8 protocol is zero-shot with no limit.')
        raw, groups = result['raw_results'], result.get('group_results', {})
        counts, shots, configs = result['n_samples'], result['n_shot'], result['configs']
        current, sub, batch = original.leaf_metrics(raw, counts, shots, configs)
        c.require(str(result['batch_size']) == str(batch) and not set(scores) & set(current), 'Original B64/B32 unique shards required.')
        original._group_agrees(current, raw, groups)
        c.require(set(result['tasks']) == set(current) and len(result['tasks']) == len(current)
                  and set(result['task_metrics']) == set(current) and set(result['task_metric_keys']) == set(current),
                  'Full8 summary inventory differs from original leaves.')
        for task, value in current.items():
            c.require(result['task_metric_keys'][task] == TASK_METRICS[task]
                      and math.isclose(_score(result['task_metrics'][task], task), value, rel_tol=0, abs_tol=1e-12),
                      'Full8 summary differs from the actual original raw metric.')
        scores.update(current)
        subjects.update(sub)
        versions.update({task: result['versions'][task] for task in counts})
        sources.append(dict(path=str(path), sha256=c.binding(path)['sha256'], tasks=list(current),
                            leaf_count=len(counts), sample_count=sum(item['effective'] for item in counts.values()),
                            batch_size=batch, format=c.FORMAT))
    c.require(set(scores) == set(TASK_METRICS) and len(subjects) == 57
              and sum(row['leaf_count'] for row in sources) == 64
              and sum(row['sample_count'] for row in sources) == 24742, 'Require all64 leaves/24742 examples.')
    return dict(task_metrics=scores, task_metric_keys=TASK_METRICS,
                mean_percent=math.fsum(scores[task] for task in TASK_METRICS) * 100 / 8,
                mmlu_subject_metrics=subjects, versions=versions, input_results=sources, leaf_count=64, sample_count=24742)


def make_config(checkpoint, pins):
    from experiments.longbench_v1.models import make_config as canonical_config, check_config
    from experiments.longbench_v1.common import file_stat
    api = c.module(c.RESULT / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_raw_sparse_config_helpers')
    reference = c.read(api.REFERENCE_CONFIG)
    meta, audit = runtime.check_audit(checkpoint)
    # The canonical constructor validates the actual native component; outer identity is explicit.
    config = canonical_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
                              checkpoint=checkpoint / 'base', mode='full')
    c.require(all(config[key] == reference[key] for key in reference if key != 'model'),
              'Keep the existing archive,32K original non-thinking generation and package policy.')
    config['model'].update(checkpoint=str(checkpoint), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=c.binding(checkpoint / 'checkpoint_meta.json')['sha256'],
        weight_stat=file_stat(checkpoint / 'base/pytorch_model.bin'),
        raw_abs005_Rice7=dict(format=c.FORMAT, max_linear_bpw=2.6, max_lora_rank=0,
                              bit_budget=audit['bit_budget'], source_sha256=pins))
    check_config(config)
    return config


def evaluate(checkpoint, output, gpu, stage):
    c.cpu_only()
    c.require(not output.exists(), 'One fresh full9 output; no retry or resume.')
    meta, audit = runtime.check_audit(checkpoint)
    runtime.check_gpu(checkpoint)
    api = c.module(c.RESULT / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_raw_sparse_full9_helpers')
    pins = {str(path): c.binding(path)['sha256'] for path in
            [*c.HERE.glob('sparse*.py'), *api.source_paths(), checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json']}
    output.mkdir(parents=True)
    config = make_config(checkpoint, pins)
    from experiments.longbench_v1.data import load_archive
    from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS
    manifest, rows = load_archive(config['archive']['path'], check_token_file=False)
    c.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS
              and all(row['input_tokens'] + row['max_new_tokens'] <= 32768 for row in rows), 'Canonical complete21 archive differs.')
    config_path = output / 'config.json'
    c.write(config_path, config)
    record = dict(status='running', checkpoint_id=meta['checkpoint_id'], checkpoint_dir=str(checkpoint),
                  source_checkpoint_id=meta['source_checkpoint_id'], started_unix=time.time(), gpu=gpu,
                  source_sha256=pins, bit_budget=audit['bit_budget'], automatic_retry=False, automatic_resume=False)
    path = output / 'evaluation.json'
    c.write(path, record)
    try:
        for name, tasks, batch in [('mmlu', 'mmlu', '64'), ('seven', api.SEVEN, '32')]:
            stage(name, [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_runtime.py'), 'full8',
                '--checkpoint_dir', str(checkpoint), '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0',
                '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch, '--tokenizer_use_fast', 'true',
                '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')),
                '--output_json', str(output / (name + '.json'))], gpu=gpu, seconds=3600, directory=output)
        from experiments.w25_eval.model_inventory import model_inventory
        inventory = model_inventory(meta['base_model_path'])
        paths = [output / 'mmlu.json', output / 'seven.json']
        full8 = collect_results(paths, inventory, checkpoint=(checkpoint, meta))
        template = c.read(api.TEMPLATE)['nine_task_argv']
        index = template.index('--baseline_results')
        baseline_paths = template[index + 1:index + 3]
        baseline = collect_results(baseline_paths, inventory)
        c.require(full8['versions'] == baseline['versions'], 'Full8 task versions differ from BF16.')
        for candidate, base, count in zip(paths, baseline_paths, (14042, 10700), strict=True):
            result = c.read(candidate)['evals']['lm_eval']
            c.require(result['configs'] == c.read(base)['configs'] and result['sample_artifacts']['total_count'] == count,
                      'The full8 configurations/sample counts differ from the original BF16 protocol.')
            for row in result['sample_artifacts']['tasks'].values():
                raw = Path(row['path'])
                c.require(raw.is_relative_to(output) and c.binding(raw)['sha256'] == row['sha256'], 'Raw question sample binding changed.')
        c.write(output / 'full8_summary.json', dict(checkpoint_id=meta['checkpoint_id'], **full8))
        stage('longbench', [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_runtime.py'), 'longbench',
            '--config', str(config_path), '--output', str(output / 'longbench')], gpu=gpu, seconds=25200, directory=output)
        from experiments.longbench_v1.scoring import score_directory
        scores = score_directory(config['archive']['path'], output / 'longbench', output / 'longbench/scores.json')
        c.require(scores['status'] == 'complete' and scores['sample_count'] == 4750
                  and set(scores['tasks']) == set(DATASETS), 'Require complete official21-task scoring.')
        formula_dir = c.RESULT / 'qwen_b32_mlp_s3_20261023'
        sys.path.insert(0, str(formula_dir))
        formula = c.module(formula_dir / 'summarize_nine.py', '_raw_sparse_original_nine_formula')
        def candidate(directory, architecture):
            c.require(Path(directory).resolve() == checkpoint and architecture['projection_count'] == 252
                      and architecture['original_linear_weights'] == audit['bit_budget']['original_linear_weights'],
                      'Actual full bundle architecture changed.')
            runtime.check_gpu(checkpoint)
            actual_meta, actual_audit = runtime.check_audit(checkpoint)
            return checkpoint, actual_meta, actual_audit['bit_budget']
        formula._candidate, formula.collect_results = candidate, collect_results
        nine = formula.summarize(base_model_path=meta['base_model_path'], checkpoint_dir=str(checkpoint),
            baseline_results=baseline_paths, compressed_results=list(map(str, paths)),
            longbench_manifest=config['archive']['path'],
            baseline_longbench_scores=template[template.index('--baseline_longbench_scores') + 1],
            compressed_longbench_scores=str(output / 'longbench/scores.json'))
        c.require(nine['quality_target_evaluated'] and nine['checkpoint_id'] == meta['checkpoint_id']
                  and abs(nine['baseline_nine_task_score'] - 68.04714491790848) < 1e-12,
                  'The original nine-term formula/reference changed.')
        nine.update(schema='raw_CG_abs005_Rice7_nine_task_acceptance_v1', checkpoint_loader=c.FORMAT)
        c.write(output / 'nine_task_acceptance.json', nine)
        from experiments.longbench_v1.common import check_hashes
        check_hashes(pins)
        record.update(status='complete_full9', full8_mean_percent=full8['mean_percent'], full8_sample_count=24742,
            longbench_sample_count=4750, nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'],
            nine_task_report=str(output / 'nine_task_acceptance.json'))
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(path, record)
    return record

"""Mixed v2 provenance validation plus the existing current transfer formula."""
import argparse
import json
import math
from pathlib import Path

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.compare_longbench_v1 import comparison
from experiments.qwen3_w25.summarize_results import TASK_METRICS
from experiments.w25_eval.acceptance import collect_results, leaf_metrics, _group_agrees
from experiments.w25_eval.transfer_nine_task_acceptance import combine_scores, _sample_artifacts


def collect_mixed(paths, inventory, directory, meta, hashes):
    scores, subjects, versions, sources = {}, {}, {}, []
    for path in paths:
        path = Path(path).resolve()
        payload = b.read(path)
        b.require(Path(payload['checkpoint_dir']).resolve() == directory and payload['checkpoint_id'] == meta['checkpoint_id']
            and payload['checkpoint_loader'] == payload['checkpoint_format'] == b.FORMAT
            and payload['strict_load'] is True and payload['adapter_dir'] is None and payload['eval_hif4_act'] is False
            and payload['checkpoint_files_sha256'] == hashes, 'Wrong mixed-format downstream identity.')
        b.require(Path(payload['base_model_path']).resolve() == b.BASE and payload['tokenizer_class'] == inventory['tokenizer_class']
            and payload['tokenizer_is_fast'] is True, 'Downstream tokenizer differs.')
        audit = payload['load_audit']
        b.require(audit['checkpoint_id'] == meta['checkpoint_id'] and audit['checkpoint_format'] == b.FORMAT
            and not audit['strict_missing'] and not audit['strict_unexpected']
            and audit['persistent_tensors'] == len(meta['tensor_table'])
            and audit['checkpoint_files_sha256'] == hashes
            and payload['inference']['use_cache'] is False
            and payload['inference']['model_seqlen'] == meta['runtime']['seqlen'], 'Incomplete actual mixed loading/inference audit.')
        result = payload['evals']['lm_eval']
        b.require(result['num_fewshot'] == 0 and result['limit'] is None, 'Require complete zero-shot downstream evaluation.')
        current, sub, batch = leaf_metrics(result['raw_results'], result['n_samples'], result['n_shot'], result['configs'])
        _group_agrees(current, result['raw_results'], result['group_results'])
        b.require(str(result['batch_size']) == str(batch) and not set(scores) & set(current)
            and set(result['tasks']) == set(result['task_metrics']) == set(result['task_metric_keys']) == set(current),
            'Wrong shard inventory or actual MMLU64/seven32 batch.')
        for task, value in current.items():
            b.require(result['task_metric_keys'][task] == TASK_METRICS[task]
                and math.isclose(value, result['task_metrics'][task], rel_tol=0, abs_tol=1e-12), 'Raw/task metric mismatch.')
        scores.update(current); subjects.update(sub)
        versions.update({task: result['versions'][task] for task in result['n_samples']})
        sources.append(dict(path=str(path), sha256=b.sha256(path), tasks=list(current),
            leaf_count=len(result['n_samples']), sample_count=sum(row['effective'] for row in result['n_samples'].values()),
            batch_size=batch, format=b.FORMAT))
    b.require(set(scores) == set(TASK_METRICS) and len(subjects) == 57
        and sum(row['leaf_count'] for row in sources) == 64 and sum(row['sample_count'] for row in sources) == 24742,
        'Require complete64 leaves/24742 examples.')
    artifacts = _sample_artifacts(paths)
    return dict(task_metrics=scores, task_metric_keys=TASK_METRICS, mmlu_subject_metrics=subjects,
        mean_percent=math.fsum(scores.values()) * 100 / 8, versions=versions, input_results=sources,
        leaf_count=64, sample_count=24742, sample_artifacts=artifacts)


def full8_summary(config, paths, output):
    record = config['model']
    directory = Path(record['checkpoint'])
    meta = b.read(directory / 'adaptive_model_meta.json')
    inventory = b.model_inventory(b.BASE)
    compressed = collect_mixed(paths, inventory, directory, meta, record['checkpoint_files_sha256'])
    baseline_paths, _ = b.template_inputs()
    baseline = collect_results(baseline_paths, inventory)
    b.require(compressed['versions'] == baseline['versions'], 'Downstream versions differ from BF16.')
    for path, reference in zip(paths, baseline_paths, strict=True):
        b.require(b.read(path)['evals']['lm_eval']['configs'] == b.read(reference)['configs'], 'Downstream task configurations differ from BF16.')
    b.write_json(output, dict(checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT, **compressed))
    return compressed


def summarize(checkpoint_dir, pipeline_report, compressed_results, compressed_longbench_scores):
    binding, meta = b.source(checkpoint_dir, pipeline_report)
    directory = Path(binding['checkpoint_dir'])
    inventory = b.model_inventory(b.BASE)
    baseline_paths, baseline_lb = b.template_inputs()
    baseline = collect_results(baseline_paths, inventory)
    compressed = collect_mixed(compressed_results, inventory, directory, meta, binding['checkpoint_files_sha256'])
    b.require(baseline['versions'] == compressed['versions'], 'Task versions differ from BF16.')
    for path, reference in zip(compressed_results, baseline_paths, strict=True):
        b.require(b.read(path)['evals']['lm_eval']['configs'] == b.read(reference)['configs'], 'Task configurations differ from BF16.')
    lb = comparison(baseline_lb, compressed_longbench_scores, b.read(b.REFERENCE)['archive']['path'])
    low_model = lb['models']['compressed']
    b.require(Path(low_model['checkpoint']).resolve() == directory and low_model['checkpoint_id'] == meta['checkpoint_id']
        and low_model['checkpoint_format'] == b.FORMAT and low_model['metadata_sha256'] == binding['metadata_sha256']
        and low_model['checkpoint_files_sha256'] == binding['checkpoint_files_sha256'], 'LongBench and full8 are different actual models.')
    for model in lb['models'].values():
        b.require(Path(model['hf_path']).resolve() == b.BASE, 'Wrong BF16/tokenizer identity.')
        b.check_hashes(model['hf_metadata_sha256'])
        for stat in model['hf_weight_stats']:
            b.check_stat(stat)
    b.check_hashes(low_model['mixed_runtime']['source_sha256'])
    b.check_stat(low_model['weight_stat'])
    budget = binding['bit_budget']
    b.require(all(budget[key] == low_model['mixed_runtime']['bit_budget'][key] for key in b.BUDGET_FIELDS), 'Actual storage changed after inference.')
    result = combine_scores(inventory['model_type'], baseline['task_metrics'], compressed['task_metrics'], lb, budget)
    result['protocol']['max_lora_rank'] = 0
    result.update(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        checkpoint_files_sha256=binding['checkpoint_files_sha256'], checkpoint_meta_sha256=binding['metadata_sha256'],
        pipeline_report=binding['pipeline_report'], pipeline_provenance=binding['pipeline_provenance'],
        base_model_path=str(b.BASE), architecture=inventory, downstream_baseline=baseline,
        downstream_compressed=compressed, downstream_sample_artifacts=compressed['sample_artifacts'],
        longbench_inputs=lb['inputs'], longbench_model_identities=lb['models'],
        implementation_sha256=b.sha256(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pipeline-report', required=True)
    parser.add_argument('--compressed-results', required=True, nargs=2)
    parser.add_argument('--compressed-longbench-scores', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.cpu_environment()
    b.require(not args.output.exists(), 'Never overwrite acceptance evidence.')
    try:
        result = summarize(args.checkpoint, args.pipeline_report, args.compressed_results, args.compressed_longbench_scores)
        code = 0 if result['passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(status='invalid', passed=False, quality_target_evaluated=False,
                      transferability_evaluated=False, error=repr(error))
        code = 2
    b.write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'checkpoint_id', 'compressed_nine_task_score', 'error') if key in result}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

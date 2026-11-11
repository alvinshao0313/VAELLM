"""Fixed64 MMLU, fixed128 full-nine, and actual saved-short consumer validation."""
import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback

from . import evaluation_bindings as e
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def annotate_acceptance():
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.summarize')
    original = module.summarize

    def summarize(checkpoint_dir, pipeline_report, compressed_results, compressed_longbench_scores):
        binding, _meta = e.snapshot(checkpoint_dir, pipeline_report, updates=128)
        result = original(checkpoint_dir, pipeline_report, compressed_results, compressed_longbench_scores)
        result.update(e.annotations(binding))
        b.require(not binding['diagnostic_overbudget'] or not result['passed'],
            'An overbudget diagnostic cannot pass the compression goal.')
        return result

    module.summarize = summarize
    return module


def endpoint_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pipeline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser


def midpoint(arguments):
    from experiments.adaptive_class_evaluation_20261030.evaluate import canonical
    from experiments.cmcc_runtime_20261030 import sequence as execution
    from experiments.qwen3_w25.step_export_callback import run_logged_stage
    from experiments.w25_eval.acceptance import collect_results, leaf_metrics, _group_agrees

    parser = endpoint_parser()
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args(arguments)
    b.cpu_environment()
    binding, meta = e.snapshot(args.checkpoint, args.pipeline_report, updates=64)
    output = args.output.resolve()
    b.require(not output.exists() and output.is_relative_to(b.TOPIC),
        'Use a new midpoint directory under the topic result root.')
    resources = canonical().check_resources([str(args.gpu)])
    pins = b.source_pins(binding)
    config = b.make_config(binding, pins)
    output.mkdir(parents=True)
    config_path = output / 'config.json'
    b.write_json(config_path, config)
    result_path = output / 'mmlu.json'
    command = b.command('full8', '--config', str(config_path), '--shard', 'mmlu', '--output', str(result_path))
    record = dict(status='running_fixed64_mmlu', checkpoint_id=meta['checkpoint_id'],
        **e.annotations(binding), started_unix=time.time(), argv=command, resources=resources)
    try:
        run_logged_stage(['timeout', '--signal=TERM', '--kill-after=20s', '3600s', *command],
            name='mmlu', output_dir=output, env=execution.environment(str(args.gpu)))
        payload = b.read(result_path)
        lm = payload['evals']['lm_eval']
        b.require(payload['checkpoint_id'] == meta['checkpoint_id']
            and payload['checkpoint_files_sha256'] == binding['checkpoint_files_sha256']
            and payload['strict_load'] and lm['limit'] is None and lm['num_fewshot'] == 0,
            'Require this snapshot and a complete zero-shot MMLU readout.')
        scores, subjects, batch = leaf_metrics(lm['raw_results'], lm['n_samples'], lm['n_shot'], lm['configs'])
        _group_agrees(scores, lm['raw_results'], lm['group_results'])
        samples = lm['sample_artifacts']
        b.require(set(scores) == {'mmlu'} and len(subjects) == len(samples['tasks']) == 57
            and str(batch) == str(lm['batch_size']) == '64' and samples['total_count'] == 14042
            and set(samples['tasks']) == set(lm['n_samples'])
            and sum(row['effective'] for row in lm['n_samples'].values()) == 14042,
            'Require all57 MMLU subjects/14042 questions.')
        for task, row in samples['tasks'].items():
            path = Path(row['path']).resolve()
            b.require(path.is_relative_to(output) and row['count'] == lm['n_samples'][task]['effective']
                and b.sha256(path) == row['sha256'], 'MMLU raw samples changed: ' + task)
        baseline_paths, _lb = b.template_inputs()
        baseline = collect_results(baseline_paths, b.model_inventory(b.BASE))
        b.require(lm['configs'] == b.read(baseline_paths[0])['configs']
            and {task: lm['versions'][task] for task in lm['n_samples']}
                == {task: baseline['versions'][task] for task in lm['n_samples']},
            'MMLU configuration or task versions differ from BF16.')
        b.check_config(config)
        record.update(status='complete_fixed64_mmlu_observation', exit_code=0,
            mmlu_percent=scores['mmlu'] * 100,
            mmlu_minus_BF16_points=(scores['mmlu'] - baseline['task_metrics']['mmlu']) * 100,
            sample_count=14042, subject_count=57, result=str(result_path), result_sha256=b.sha256(result_path),
            boundary='Fixed64 observation only; no checkpoint selection, early stopping, or complete-nine claim.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write_json(output / 'midpoint.json', record)
    return 0


def consumer(arguments):
    import torch
    from experiments.adaptive_class_evaluation_20261030.loading import load_model
    from experiments.mlp_learnability_cmcc_20261030 import lifecycle as life

    args = endpoint_parser().parse_args(arguments)
    binding, _meta = e.snapshot(args.checkpoint, args.pipeline_report, validation=True, updates=2)
    output = args.output.resolve()
    b.require(not output.exists() and output.is_relative_to(b.TOPIC),
        'Use a fresh diagnostic directory under the topic result root.')
    report = b.read(args.pipeline_report)
    fixture_path = Path(report['export']['fixture']['path'])
    expected_path = Path(report['export']['expected_state']['path'])
    original_pins = b.source_pins
    b.source_pins = lambda value: {**original_pins(value), **binding['checkpoint_files_sha256']}

    def actual_load(config, logger):
        model, audit = load_model(config, logger)
        fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
        with torch.inference_mode():
            logits = model(input_ids=fixture['input_ids'].to('cuda:0'),
                attention_mask=fixture['attention_mask'].to('cuda:0'),
                use_cache=False, logits_to_keep=1).logits.detach().cpu()
        reference = fixture['logits']
        b.require(logits.shape == reference.shape and logits.dtype == reference.dtype == torch.bfloat16
            and torch.isfinite(logits).all() and torch.isfinite(reference).all(),
            'The complete saved fixture must yield finite BF16 logits on the single-GPU consumer.')
        difference = logits.double() - reference.double()
        norm = float(reference.double().norm())
        state = life.state_records(model)
        b.require(not state['extra'] and state['tensors'] == b.read(expected_path),
            'Single-GPU prewarm and forward changed a persistent snapshot field.')
        comparison = dict(exact=torch.equal(logits, reference), max_abs=float(difference.abs().max()),
            relative_L2=float(difference.norm()) / norm if norm else None,
            changed_top1_positions=int((logits.argmax(-1) != reference.argmax(-1)).sum()),
            full_persistent_state_exact=True, finite=True,
            boundary='Compare identical saved weights across two-card and single-card deployment groups; '
                'normal group8 floating-point differences are measured without compensation or a bitwise gate.')
        audit['two_card_fixture_comparison'] = comparison
        b.write_json(output / 'consumer_comparison.json', dict(checkpoint_id=binding['checkpoint_id'],
            **e.annotations(binding), comparison=comparison))
        return model, audit

    if binding['arm'] == 'dense':
        module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.validate_gpu')
        module.load_model = actual_load
        sys.argv = [sys.argv[0], '--checkpoint', str(args.checkpoint), '--output', str(output)]
        module.main()
        record = b.read(output / 'report.json')
        b.require(record['status'] == 'verified_gpu_mixed_inference_diagnostic', 'The original dense diagnostic failed.')
        evidence = 'RTE2 plus the original three LongBench validation requests, including longest32K/KV and summary512.'
    else:
        output.mkdir(parents=True)
        config = b.make_config(binding, b.source_pins(binding), mode='validation')
        b.write_json(output / 'config.json', config)
        model, audit = actual_load(config, logging.getLogger('mlp_snapshot_single_GPU'))
        b.check_config(config)
        record = dict(checkpoint_id=binding['checkpoint_id'], load_audit=audit, quality_target_evaluated=False)
        del model
        evidence = 'New snapshot binding and single-GPU fixture; the same D/C0 full9 kernels already have complete evidence.'
    record.update(status='verified_mlp_single_GPU_consumer', **e.annotations(binding),
        source_snapshot=binding, consumer_scope=evidence)
    b.write_json(output / 'consumer.json', record)
    print(json.dumps(dict(status=record['status'], arm=binding['arm'], checkpoint_id=binding['checkpoint_id'])), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', required=True, type=Path)
    parser.add_argument('--component', default='evaluate',
        choices=('evaluate', 'midpoint', 'consumer', 'full8', 'predict', 'summarize', 'merge'))
    args, remaining = parser.parse_known_args()
    e.setup(args.support.resolve())
    b.CLI_ARGUMENTS = list(sys.argv)
    if args.component == 'midpoint':
        return midpoint(remaining)
    if args.component == 'consumer':
        return consumer(remaining)
    if args.component == 'summarize':
        module = annotate_acceptance()
    else:
        module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
    if args.component == 'evaluate':
        endpoint, _other = endpoint_parser().parse_known_args(remaining)
        binding, _meta = e.snapshot(endpoint.checkpoint, endpoint.pipeline_report, updates=128)
    sys.argv = [sys.argv[0], *remaining]
    code = module.main()
    if args.component == 'evaluate' and not '--check-only' in remaining:
        for name in ('evaluation.json', 'current_goal_result.json'):
            path = endpoint.output.resolve() / name
            record = b.read(path)
            record.update(e.annotations(binding))
            b.require(not binding['diagnostic_overbudget'] or not record.get('passed', False),
                'The dense diagnostic must remain outside the low-bit acceptance budget.')
            b.write_json(path, record)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

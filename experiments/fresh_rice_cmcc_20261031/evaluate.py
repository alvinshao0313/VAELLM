"""Actual fresh C1 Rice audit and unchanged official full8/full21 evaluation."""
import argparse
import importlib
from pathlib import Path
import sys

from . import sources as s

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc

SUPPORT = s.common.ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'
_READERS_CONFIGURATION = None


def configure_readers(support=SUPPORT):
    """Initialize cmcc once; later source reads preserve installed C1 dispatch."""
    global _READERS_CONFIGURATION
    support = Path(support).resolve(strict=True)
    b.require(support == SUPPORT.resolve(strict=True), 'Use the reviewed local cmcc evaluation support.')
    prepared = support / 'prepared.json'
    binding = dict(path=str(prepared), sha256=b.sha256(prepared))
    if _READERS_CONFIGURATION is None:
        cmcc.setup(support)
        _READERS_CONFIGURATION = binding
    b.require(_READERS_CONFIGURATION == binding and b.BASE.resolve(strict=True) == s.HF.resolve(strict=True)
        and b.REFERENCE == support / 'config.json' and b.TEMPLATE == support / 'template.json',
        'The initialized local HF/support reader identity changed.')
    return binding.copy()


def validate_report(report):
    b.require(report.get('schema') == 'fresh_C1_Rice_pipeline_v1' and report.get('arm') == s.ARM
        and report.get('status') == 'completed_fresh_C1_Rice_pipeline' and report.get('exit_code') == 0
        and report.get('phase') == 'formal', 'Require the completed formal fresh C1_Rice report; old/short/running cannot masquerade as final.')
    b.require(report['source_after_o'] == s.AFTER_O_ID and report['source_raw2'] == s.RAW2_ID
        and report['planned_recovery_steps_per_category'] == report['actual_recovery_steps_per_category'] == 500
        and report['planned_native_fit_updates_per_stage'] == report['actual_native_fit_updates_per_stage'] == 10000
        and report['additional_gate_fit_updates'] == report['third_stage_fit_updates'] == 0
        and report['rice_gate_count'] == 36
        and 'source_raw_manifest' not in report and 'stage3_fit_actual_updates' not in report,
        'C1 must use actual fresh-source exposure, all36 Rice gates and zero third-stage fitting.')


def source(directory, pipeline_report):
    configure_readers()
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = b.read(report_path)
    validate_report(report)
    meta, budget, hashes = b.inspect(directory)
    binding = s.bind(Path(report['source_binding']['common_directory']))
    b.require(binding == report['source_binding'] and report['final_checkpoint_id'] == meta['checkpoint_id']
        and Path(report['final_model']).resolve() == directory, 'Actual new source/final UUID differs.')
    provenance = meta['provenance']
    b.require(provenance['schema'] == 'fresh_C1_Rice_provenance_v1' and provenance['arm'] == s.ARM
        and provenance['phase'] == 'formal' and provenance['actual_category'] == 'down_proj'
        and provenance['actual_recovery_steps'] == provenance['planned_recovery_schedule'] == 500
        and provenance['source_binding'] == binding
        and provenance['source_raw2'] == s.RAW2_ID and provenance['source_after_o'] == s.AFTER_O_ID
        and provenance['additional_gate_fit_updates'] == provenance['third_stage_fit_updates'] == 0
        and provenance['rice_gate_count'] == 36 and provenance['suffix_fit_stages'] == 2
        and provenance['suffix_fit_updates_per_stage'] == 10000 and provenance['future_VAE_seed'] == 31
        and 'source_raw_manifest' not in provenance and 'stage3_fit_actual_updates' not in provenance,
        'Require genuine C1_Rice provenance with no old source or fictitious stage3 exposure.')
    raw = report['raw_source']
    b.require(raw['original_non_gate_fields_exact'] and raw['imported_gate_count'] == 36
        and raw['additional_gate_fit_updates'] == raw['third_stage_fit_updates'] == 0
        and raw['rice_recipe'] == provenance['rice_recipe'] == dict(rank_metric='sparse_residual_abs', top_p=.005,
            value_bits=8, index_bits=8, block_shape=[256, 256], rice_k=7, protected_columns_excluded=True)
        and {row['name'] for row in raw['rice_payloads']} == {f'model.layers.{i}.mlp.gate_proj' for i in range(36)}
        and all(row['source_raw2'] == s.RAW2_ID and row['source_after_o'] == s.AFTER_O_ID and row['nnz'] == 251659
            and row['sparse_persistent_fields'] == len(s.sparse.FIELDS) and row['sparse_fields_digest']
            for row in raw['rice_payloads']), 'All36 payloads must be freshly built from these own actual sources.')
    b.require([row['category'] for row in report['stages']] == ['gate_proj', 'up_proj', 'down_proj']
        and all(row['recovery']['actual_steps'] == 500 and row['recovery']['frozen_persistent_fields_exact']
            for row in report['stages']), 'The three real original500 recoveries did not complete.')
    for row, category, shuffle in zip(report['stages'][:-1], ('up_proj', 'down_proj'), (500031, 600031), strict=True):
        fit = row['next_live_category_fit']
        b.require(fit['category'] == category and fit['actual_stages'] == 2
            and fit['actual_updates_per_stage'] == fit['schedule_steps'] == 10000
            and fit['fresh_VAE_initialization_seed'] == 31 and fit['shuffle_seed'] == shuffle
            and len(fit['actual_optimizer_initializations']) == 2
            and all(item['initial_parameter_digest'] for item in fit['actual_optimizer_initializations']),
            'C1 needs both actual own-live suffix fits with original seed/shuffle and stage initialization.')
    b.require(all(item['spec'].get('consumer') == 'fixed2_rice7' and item['spec']['implicit_length'] == 2
        and item['spec']['sparse_fields'] == list(s.sparse.FIELDS)
        and len(item['spec']['stages']) == 2 and item['spec']['vector_dim'] == 64
        and all(stage['decoder_spec']['in_dim'] == stage['decoder_spec']['out_dim'] == 64
            and stage['decoder_spec']['decoder_type'] == 'asymmetric' for stage in item['spec']['stages'])
        and budget['compact_targets'][item['name']]['length_histogram'] == [0, 0, item['spec']['n_vectors']]
        and budget['compact_targets'][item['name']]['components_bytes']['rice7_sparse_residual'] > 0
        and 'route_frame' not in budget['compact_targets'][item['name']]['components_bytes']
        for item in meta['compact_modules']), 'Deploy all36 real two-stage Rice consumers, fully charged, with implicit length2.')
    checked = report['parent_loading_validation']
    b.require(checked['status'] == 'verified_actual_fresh_raw2_Rice_parent'
        and checked['full_state_and_same_input_BF16_logits_exact'] and checked['extra_training_updates'] == 0
        and checked['CPU_CUDA_RNG_preserved'], 'New Rice finite/save/fresh/RNG gap has not passed.')
    strict_path = Path(report['final_strict_reload_report'])
    strict = b.read(strict_path)
    b.require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
        and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
        and strict['persistent_fields'] == len(meta['tensor_table'])
        and strict['effective_weight_files'] == [str(directory / b.checkpoint.STATE_FILE)]
        and all(strict['actual_budget'][key] == budget[key] for key in b.BUDGET_FIELDS), 'Final actual independent strict evidence missing.')
    b.check_hashes(report['pipeline_source_sha256'])
    hashes.update(binding['common_files_sha256'])
    for event in (binding['source_after_o'], binding['source_raw2']):
        parent = Path(event['path'])
        hashes[str(parent / 'checkpoint_meta.json')] = event['metadata_sha256']
        hashes[str(parent / 'cat_runtime_state.pt')] = event['runtime_sha256']
    for path in (strict_path, report_path):
        hashes[str(path)] = b.sha256(path)
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory / b.checkpoint.META_FILE)], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory / b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report_path), pipeline_provenance=provenance), meta


def setup(support):
    configure_readers(support)
    original_pins = b.source_pins

    def source_pins(binding):
        return {**original_pins(binding), **s.source_pins()}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m', 'experiments.fresh_rice_cmcc_20261031.evaluate',
            '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--component', choices=('audit', 'evaluate', 'full8', 'predict', 'summarize', 'merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    setup(args.support.resolve(strict=True))
    b.CLI_ARGUMENTS = list(sys.argv)
    if args.component == 'audit':
        audit = argparse.ArgumentParser()
        audit.add_argument('--checkpoint', type=Path, required=True)
        audit.add_argument('--pipeline-report', type=Path, required=True)
        audit.add_argument('--output', type=Path, required=True)
        values = audit.parse_args(remaining)
        b.cpu_environment()
        b.require(not values.output.exists(), 'Never overwrite a real C1 final audit.')
        model, meta = source(values.checkpoint, values.pipeline_report)
        b.write_json(values.output, dict(status='verified_actual_fresh_C1_Rice_final', exit_code=0,
            checkpoint_id=meta['checkpoint_id'], binding=model, source_sha256=b.source_pins(model)))
        return
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
    sys.argv = [sys.argv[0], *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

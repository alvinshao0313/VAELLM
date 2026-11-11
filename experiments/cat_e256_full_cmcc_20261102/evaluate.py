"""Official full8/full21 evaluation with genuine fresh ordinary-fixed2 provenance."""
import argparse
import importlib
from pathlib import Path
import sys

from . import sources as s

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def source(directory, pipeline_report):
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = b.read(report_path)
    meta, budget, hashes = b.inspect(directory)
    b.require(report['schema'] == 'same_parent_E256_fixed2_pipeline_v1' and report['arm'] == 'E256_fixed2'
        and report['status'] == 'completed_same_parent_E256_fixed2_pipeline' and report['exit_code'] == 0
        and report['phase'] == 'formal' and report['final_checkpoint_id'] == meta['checkpoint_id']
        and Path(report['final_model']).resolve() == directory
        and report['planned_recovery_steps_per_category'] == report['actual_recovery_steps_per_category'] == 500
        and report['planned_native_fit_updates_per_stage'] == report['actual_native_fit_updates_per_stage'] == 10000
        and report['additional_gate_fit_updates'] == report['third_stage_fit_updates'] == report['rice_sparse_fields'] == 0,
        'Require the genuine completed fresh C0 ordinary fixed2 endpoint and original suffix budget.')
    binding = s.bind(Path(report['source_binding']['common_directory']))
    b.require(binding == report['source_binding'], 'The actual new shared raw2/after_o source changed.')
    provenance = meta['provenance']
    b.require(provenance['schema'] == 'same_parent_E256_fixed2_provenance_v1'
        and provenance['arm'] == 'E256_fixed2' and provenance['phase'] == 'formal'
        and provenance['actual_category'] == 'down_proj' and provenance['actual_recovery_steps'] == 500
        and provenance['source_binding'] == binding
        and provenance['source_raw2'] == report['source_raw2'] == binding['source_raw2']['checkpoint_id']
        and provenance['source_after_o'] == report['source_after_o'] == binding['source_after_o']['checkpoint_id']
        and provenance['additional_gate_fit_updates'] == provenance['third_stage_fit_updates'] == provenance['rice_sparse_fields'] == 0
        and provenance['suffix_fit_stages'] == 2 and provenance['suffix_fit_updates_per_stage'] == 10000,
        'Final C0 provenance must bind this new raw2; no fictitious stage3/Rice training fields.')
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
            'Each own live suffix requires both actual seed31 optimizer initializations and its original shuffle.')
    b.require(all('consumer' not in item['spec'] and len(item['spec']['stages']) == 2
        and item['spec']['vector_dim'] == 64
        and all(stage['decoder_spec']['in_dim'] == stage['decoder_spec']['out_dim'] == 64
            and stage['decoder_spec']['decoder_type'] == 'asymmetric' for stage in item['spec']['stages'])
        and budget['compact_targets'][item['name']]['length_histogram'] == [0, 0, item['spec']['n_vectors']]
        and 'rice7_sparse_residual' not in budget['compact_targets'][item['name']]['components_bytes']
        for item in meta['compact_modules']), 'C0 must deploy all36 actual ordinary fixed2 gates, with no third decoder or Rice fields.')
    parent_check = report['parent_loading_validation']
    initializations = s.verify_suffix_initializations(binding, report['stages'])
    b.require(len(initializations) == 2 and initializations == report['suffix_initialization_comparison'],
        'Require both actual own live suffix initializations to match genuine C0.')
    b.require(parent_check['status'] == 'verified_actual_fresh_raw2_fixed2_parent'
        and parent_check['full_state_and_same_input_BF16_logits_exact'] and parent_check['extra_training_updates'] == 0
        and parent_check['CPU_CUDA_RNG_preserved'],
        'The new-parent fixed2 loading gap has not been verified.')
    strict_path = Path(report['final_strict_reload_report'])
    strict = b.read(strict_path)
    b.require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
        and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
        and strict['persistent_fields'] == len(meta['tensor_table'])
        and strict['effective_weight_files'] == [str(directory / b.checkpoint.STATE_FILE)]
        and all(strict['actual_budget'][key] == budget[key] for key in b.BUDGET_FIELDS),
        'Final independent strict C0 full-state/BF16 evidence is missing.')
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
    cmcc.setup(support)
    original_pins = b.source_pins

    def source_pins(binding):
        return {**original_pins(binding), **s.source_pins()}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m', 'experiments.cat_e256_full_cmcc_20261102.evaluate',
            '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--component', choices=('evaluate', 'full8', 'predict', 'summarize', 'merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    setup(args.support.resolve(strict=True))
    b.CLI_ARGUMENTS = list(sys.argv)
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
    sys.argv = [sys.argv[0], *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

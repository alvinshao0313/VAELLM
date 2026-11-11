"""Official full8/full21 accepting only the independent original-C1 Muon32 endpoint."""
import argparse
import importlib
from pathlib import Path
import sys

from . import bindings as r
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def validate_header(report):
    r.require(report.get('schema') == 'C1_M_target_Muon_execution_v1'
        and report.get('validation_only') is False and report.get('phase') == 'formal32'
        and report.get('steps') == 32 and report.get('arm') in r.ARMS
        and report.get('status') == 'C1_M_target_training_export_complete_fresh_pending',
        'Require the actual complete fresh-C1 Rice Muon32; no old C0, source, running or engineering2 endpoint.')


def source(directory, pipeline_report):
    r.configure()
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = r.read(report_path)
    validate_header(report)
    prepared_path = Path(report['prepared']['path'])
    prepared = r.read_prepared(prepared_path)
    short = prepared['Rice_lifecycle_evidence']
    meta, budget, hashes = b.inspect(directory)
    audit_path = report_path.parent / 'fresh_audit.json'
    audit = r.read(audit_path)
    r.require(report['group_indices'] == r.GROUPS and report['contract'] == r.CONTRACT
        and report['fresh_optimizer'] and report['original_schedule_steps'] == 500 and report['seed'] == 137
        and report['overall_LR_multiplier'] == r.LR_MULTIPLIERS[report['arm']]
        and report['Rice_lifecycle_evidence'] == short
        and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
        and report['Rice_frozen_exact'] and (report_path.parent / 'run.exit').read_text().strip() == '0'
        and Path(report['export']['directory']).resolve() == directory
        and report['export']['checkpoint_id'] == meta['checkpoint_id']
        and audit['schema'] == 'C1_M_target_Muon_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['arm'] == report['arm']
        and audit['phase'] == 'formal32' and audit['validation_only'] is False and audit['steps'] == 32
        and audit['training_report'] == r.binding(report_path) and audit['prepared'] == r.binding(prepared_path)
        and audit['export_meta'] == r.binding(directory / 'adaptive_model_meta.json')
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['full_state_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite'] and audit['Rice_frozen_exact'],
        'Require genuine original-source32, actual saved deployment and its independent Rice/frozen/BF16 audit.')
    parent, parent_meta = r.original.source(prepared['source_binding']['checkpoint_dir'], prepared['source_binding']['pipeline_report'])
    r.require(parent == prepared['source_binding'] == report['source_binding']
        and meta['checkpoint_id'] not in (parent['checkpoint_id'], short['checkpoint_id'])
        and report['source_checkpoint_id'] == audit['source_checkpoint_id'] == short['source_checkpoint_id'] == parent['checkpoint_id'],
        'Actual C1 lineage changed, or original/short checkpoint was substituted for the formal32 endpoint.')
    provenance = meta['provenance']
    r.require(provenance['schema'] == 'C1_M_target_Muon_provenance_v1' and provenance['source_binding'] == parent
        and provenance['arm'] == report['arm']
        and provenance['overall_LR_multiplier'] == r.LR_MULTIPLIERS[report['arm']]
        and provenance['source_checkpoint_id'] == parent['checkpoint_id'] and provenance['prepared'] == r.binding(prepared_path)
        and provenance['actual_updates'] == 32 and provenance['planned_schedule'] == 500
        and provenance['group_indices'] == r.GROUPS and provenance['contract'] == r.CONTRACT
        and provenance['seed'] == 137 and provenance['phase'] == 'formal32' and provenance['validation_only'] is False
        and provenance['Rice_lifecycle_evidence'] == short, 'Formal32 must retain genuine C1/short-evidence lineage and unchanged recipe.')
    r.require(meta['native_modules'] == parent_meta['native_modules'] and meta['compact_modules'] == parent_meta['compact_modules']
        and audit['frozen_state'] == prepared['record']['validated_frozen_state']
        and report['training']['frozen_state_before'] == report['training']['frozen_state_after']
            == prepared['record']['validated_frozen_state']
        and audit['Rice_state'] == report['Rice_state'] == prepared['record']['validated_Rice_state']
        and all(budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS)
        and all(budget[key] == prepared['source_budget'][key] for key in
            ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'full_model_tensor_bytes')),
        'Actual geometry, Rice/non-D frozen fields or the independently verified tensor budget changed.')
    trajectory = report['training']['trajectory']
    r.require(len(trajectory) == report['training']['steps'] == 32
        and [row['group_index'] for row in trajectory] == r.GROUPS
        and all(row['retention_lambda'] == 0 and row['mmlu_execution_batch_size'] == 32
            and row['M_target_arm'] == report['arm'] for row in trajectory),
        'The actual H96/original500 prefix32 trajectory is incomplete.')
    hashes.update(parent['checkpoint_files_sha256'])
    hashes.update({str(path):b.sha256(path) for path in (report_path, audit_path, prepared_path,
        Path(short['training']['path']), Path(short['fresh']['path']))})
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory / 'adaptive_model_meta.json')], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory / b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report_path), pipeline_provenance=provenance), meta


def setup(support):
    r.original.c1.configure_readers(support)
    r.configure()
    original = b.source_pins

    def source_pins(binding):
        return {**original(binding), **{path:item['sha256'] for path,item in r.code_bindings().items()}}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m', 'experiments.c1_m_target_muon_cmcc_20261102.evaluate',
            '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', required=True, type=Path)
    parser.add_argument('--component', choices=('evaluate', 'full8', 'predict', 'summarize', 'merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    setup(args.support.resolve())
    b.CLI_ARGUMENTS = list(sys.argv)
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
    sys.argv = [sys.argv[0], *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

"""Official full8 plus LB21 evaluation, accepting only genuine completed BR->E32."""
import argparse
import importlib
from pathlib import Path
import sys

from . import recovery_bindings as r
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def source(directory, pipeline_report):
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = r.read(report_path)
    r.require(report.get('schema') == 'BR_E32_execution_v1' and report.get('validation_only') is False,
        'BR->E full9 requires its own real complete32 execution, never a short, C0 or BR midpoint report.')
    prepared_path = Path(report['prepared']['path'])
    prepared = r.read_prepared(prepared_path)
    meta, budget, hashes = b.inspect(directory)
    audit_path = report_path.parent/'fresh_audit.json'
    audit = r.read(audit_path)
    r.require(report['status'] == 'BR_E32_training_export_complete_fresh_pending'
        and report['phase'] == 'formal32' and report['steps'] == 32 and report['seed'] == 137
        and report['group_indices'] == r.GROUPS and report['contract'] == r.CONTRACT
        and report['fresh_optimizer'] and report['original_schedule_steps'] == 500
        and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
        and (report_path.parent/'run.exit').read_text().strip() == '0'
        and Path(report['export']['directory']).resolve() == directory
        and report['export']['checkpoint_id'] == meta['checkpoint_id']
        and audit['schema'] == 'BR_E32_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['training_report'] == r.binding(report_path)
        and audit['prepared'] == r.binding(prepared_path)
        and audit['export_meta'] == r.binding(directory/'adaptive_model_meta.json')
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['full_state_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite'],
        'Require completed genuine-BR32, the actual saved deployment and its independent verified fresh audit.')
    parent, parent_meta = r.source(prepared['source_binding']['checkpoint_dir'], prepared['source_binding']['pipeline_report'])
    r.require(parent == prepared['source_binding']
        and meta['checkpoint_id'] not in (parent['checkpoint_id'], parent['source_C0_binding']['checkpoint_id'])
        and report['source_checkpoint_id'] == audit['source_checkpoint_id'] == parent['checkpoint_id']
        and report['source_C0_binding'] == audit['source_C0_binding'] == parent['source_C0_binding']
        and report['BR_record'] == audit['BR_record'] == parent['BR_record'],
        'The actual BR/C0 lineage changed, or an ancestor was substituted for the recovered endpoint.')
    provenance = meta['provenance']
    r.require(provenance['schema'] == 'BR_E32_provenance_v1' and provenance['source_binding'] == parent
        and provenance['source_C0_binding'] == parent['source_C0_binding'] and provenance['BR_record'] == parent['BR_record']
        and provenance['prepared'] == r.binding(prepared_path) and provenance['source_checkpoint_id'] == parent['checkpoint_id']
        and provenance['actual_updates'] == 32 and provenance['planned_schedule'] == 500
        and provenance['group_indices'] == r.GROUPS and provenance['contract'] == r.CONTRACT
        and provenance['seed'] == 137 and provenance['phase'] == 'formal32' and provenance['validation_only'] is False,
        'Do not evaluate another recipe or substitute old C0 schema for the BR->E lineage.')
    r.require(meta['native_modules'] == parent_meta['native_modules']
        and meta['compact_modules'] == parent_meta['compact_modules']
        and audit['frozen_state'] == prepared['record']['validated_frozen_state']
        and report['training']['frozen_state_before'] == report['training']['frozen_state_after']
            == prepared['record']['validated_frozen_state']
        and all(budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
        'Fixed-code geometry, frozen persistent fields or the independently verified actual storage ledger changed.')
    trajectory = report['training']['trajectory']
    r.require(len(trajectory) == report['training']['steps'] == 32
        and [row['group_index'] for row in trajectory] == r.GROUPS
        and report['training']['teacher_unchanged']
        and all(row['retention_lambda'] == 0 and row['mmlu_execution_batch_size'] == 32 for row in trajectory)
        and trajectory[0]['Muon_calibration']['real_optimizer_still_fresh']
        and trajectory[0]['Muon_calibration']['calibration_step'] == 1,
        'The actual complete H96 trajectory and fresh first-gradient Muon calibration are required.')
    hashes.update(parent['checkpoint_files_sha256'])
    hashes.update({str(path): b.sha256(path) for path in (report_path, audit_path, prepared_path)})
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory/'adaptive_model_meta.json')], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory/b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report_path), pipeline_provenance=provenance,
        source_C0_binding=parent['source_C0_binding'], BR_record=parent['BR_record']), meta


def setup(support):
    from . import bindings as formal_BR
    formal_BR.configure_readers(support)
    original = b.source_pins

    def source_pins(binding):
        return {**original(binding), **{path:item['sha256'] for path,item in r.code_bindings().items()}}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m',
            'experiments.block_reconstruction_formal_cmcc_20261031.recovery_evaluate',
            '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', required=True, type=Path)
    parser.add_argument('--component', choices=('evaluate','full8','predict','summarize','merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    setup(args.support.resolve())
    b.CLI_ARGUMENTS = list(sys.argv)
    module = importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component)
    sys.argv = [sys.argv[0], *remaining]
    return module.main()


if __name__ == '__main__':
    main()

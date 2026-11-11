"""The unchanged complete-nine evaluator with actual C0-Muon32 provenance."""
import argparse
import importlib
from pathlib import Path
import sys

from . import bindings as r
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def source(directory, pipeline_report):
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = r.read(report_path)
    prepared_path = Path(report['prepared']['path'])
    prepared = r.read_prepared(prepared_path)
    meta, budget, hashes = b.inspect(directory)
    audit_path = report_path.parent/'fresh_audit.json'
    audit = r.read(audit_path)
    r.require(report['schema'] == 'fresh_C0_Muon32_execution_v1'
        and report['status'] == 'formal32_training_export_complete_fresh_pending'
        and report['phase'] == 'formal32' and report['steps'] == 32
        and report['group_indices'] == r.GROUPS and report['contract'] == r.CONTRACT
        and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
        and (report_path.parent/'run.exit').read_text().strip() == '0'
        and Path(report['export']['directory']).resolve() == directory
        and report['export']['checkpoint_id'] == meta['checkpoint_id']
        and audit['schema'] == 'fresh_C0_Muon32_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['training_report'] == r.binding(report_path)
        and audit['prepared'] == r.binding(prepared_path)
        and audit['export_meta'] == r.binding(directory/'adaptive_model_meta.json')
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['full_state_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite'],
        'Require completed32, real saved deployment and its independent verified fresh audit.')
    parent, parent_meta = r.source(prepared['source_binding']['checkpoint_dir'], prepared['source_binding']['pipeline_report'])
    r.require(parent == prepared['source_binding']
        and meta['checkpoint_id'] != parent['checkpoint_id']
        and report['source_checkpoint_id'] == audit['source_checkpoint_id'] == parent['checkpoint_id'],
        'The actual parent identity changed or the source itself was substituted for the endpoint.')
    provenance = meta['provenance']
    r.require(provenance['schema'] == 'fresh_C0_Muon32_provenance_v1'
        and provenance['source_binding'] == parent and provenance['prepared'] == r.binding(prepared_path)
        and provenance['actual_updates'] == 32 and provenance['planned_schedule'] == 500
        and provenance['group_indices'] == r.GROUPS and provenance['contract'] == r.CONTRACT
        and not provenance['validation_only'], 'Do not evaluate a short or another recovery recipe.')
    r.require(meta['native_modules'] == parent_meta['native_modules']
        and meta['compact_modules'] == parent_meta['compact_modules']
        and audit['frozen_state'] == prepared['record']['validated_frozen_state']
        and report['training']['frozen_state_before'] == report['training']['frozen_state_after']
            == prepared['record']['validated_frozen_state']
        and all(budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
        'Fixed-code geometry/frozen fields or the actual fresh storage ledger changed.')
    trajectory = report['training']['trajectory']
    r.require(len(trajectory) == 32 and [row['group_index'] for row in trajectory] == r.GROUPS
        and all(row['retention_lambda'] == 0 and row['mmlu_execution_batch_size'] == 32 for row in trajectory),
        'The actual complete H96 update trajectory is missing.')
    hashes.update(parent['checkpoint_files_sha256'])
    hashes.update({str(path): b.sha256(path) for path in (report_path, audit_path, prepared_path)})
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        metadata_sha256=hashes[str(directory/'adaptive_model_meta.json')], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=b.file_stat(directory/b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report_path), pipeline_provenance=provenance), meta


def setup(support):
    r.shared.setup(support)
    original = b.source_pins

    def source_pins(binding):
        return {**original(binding), **{path:item['sha256'] for path,item in r.code_bindings().items()}}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m', 'experiments.fresh_muon_cmcc_20261030.evaluate',
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

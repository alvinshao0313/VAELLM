"""Real snapshot source ownership, strict publication and unchanged C0 ancestry."""
from pathlib import Path

from . import bindings as b
from experiments.adaptive_class_evaluation_20261030 import bindings as official


def readers():
    # A consumer may already have installed its dispatch. Configure only when
    # its actual cmcc BASE/support paths are not initialized yet.
    if not (official.BASE == b.HF and official.REFERENCE == b.SUPPORT/'config.json'
            and official.TEMPLATE == b.SUPPORT/'template.json'):
        from experiments.cmcc_runtime_20261030 import adaptive_evaluation
        adaptive_evaluation.setup(b.SUPPORT)


def execution(directory, report_path, meta):
    directory, report_path = Path(directory).resolve(strict=True), Path(report_path).resolve(strict=True)
    report = b.read(report_path)
    short = report['validation_only']
    b.require(short is False and report['arm']=='decoder'
        and report['data_contract']==b.data_binding(), 'Require this actual fixed128 decoder-only M4096 data contract.')
    root = report_path.parent.parents[1]
    training = b.read(root/'report.json')
    b.require((root/'run.exit').read_text().strip() == '0'
        and training['schema'] == 'all_MLP_learning_execution_v1'
        and training['status'] == 'all_MLP_learning_training_complete_fresh_pending'
        and training['exit_code'] == 0 and training['seed'] == 137
        and training['training']['steps'] == training['planned_updates']
        and training['training']['group_indices'] == b.groups(short)
        and training['data_contract']==report['data_contract']
        and b.binding(report_path) in training['training']['snapshots'],
        'Require an actual completed continuous learning process and its bound snapshot.')
    b.require(type(short) is bool and report['schema'] == 'all_MLP_learning_snapshot_v1'
        and report['status'] == 'BF16_snapshot_saved_fresh_pending' and report['arm'] in ('decoder','dense')
        and report['planned_updates'] == (2 if short else 128)
        and report['actual_updates'] in b.snapshot_steps(short) and report['training_contract'] == b.CONTRACT
        and report['diagnostic_overbudget'] == (report['arm']=='dense')
        and Path(report['export']['directory']).resolve(strict=True) == directory
        and report['export']['checkpoint_id'] == meta['checkpoint_id']
        and report['live_master_and_Adam_unchanged'] and report['CPU_CUDA_Python_NumPy_RNG_preserved']
        and report['live_training_mode_preserved'], 'Require this actual committed learning snapshot and continuity proof.')
    b.executor.check_pins(report['source_sha256'])
    parent = report['source_binding']
    b.require(parent['checkpoint_id'] == b.C0_ID and not parent['prototype_only']
        and parent['fresh_C0'] == report['source_C0_binding'], 'The actual C0 parent binding changed.')
    for name in ('manifest','config','weight'):
        b.require(b.binding(parent[name]['path']) == parent[name], 'The actual compressed C0 source changed: '+name)
    b.executor.check_pins(parent['runtime_sha256'])
    readers()
    from experiments.fresh_fixed2_cmcc_20261030.evaluate import source as C0_source
    actual,_parent_meta = C0_source(b.C0,b.C0.parent/'report.json')
    b.require(actual == report['source_C0_binding'], 'Require the genuine complete same C0, not a relabeled model.')
    provenance = meta['provenance']
    b.require(provenance['schema'] == 'all_MLP_learning_snapshot_provenance_v1'
        and all(provenance[name] == report[name] for name in ('arm','actual_updates','planned_updates',
            'validation_only','diagnostic_overbudget','source_binding','source_C0_binding',
            'training_contract','source_non_mlp','protection_records','data_contract')),
        'Serialized snapshot provenance differs from the actual report.')
    for key,name in (('expected_state','expected_state.json'),('fixture','same_input_output.pt')):
        b.require(b.binding(report_path.parent/name) == report['export'][key], 'A saved-state/fixture binding changed.')
    expected = b.read(report_path.parent/'expected_state.json')
    b.require(set(expected) == set(meta['tensor_table']), 'Every actual persistent snapshot field must be bound.')
    return report, expected


def source(directory, report_path):
    readers()
    directory, report_path = Path(directory).resolve(strict=True), Path(report_path).resolve(strict=True)
    meta,budget,hashes = official.inspect(directory,final=False)
    report,expected = execution(directory,report_path,meta)
    audit_path = report_path.parent/'fresh_audit.json'
    audit = b.read(audit_path)
    b.require(audit['schema'] == 'all_MLP_learning_snapshot_fresh_v1' and audit['status'] == 'verified'
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['source_checkpoint_id'] == b.C0_ID
        and audit['training_report'] == b.binding(report_path) and audit['export_meta'] == b.binding(directory/b.checkpoint.META_FILE)
        and audit['full_state_exact'] and audit['non_MLP_source_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite']
        and audit['fixture'] == report['export']['fixture'] and audit['bit_budget'] == budget
        and audit['persistent_tensors_compared'] == len(expected)
        and (report_path.parent/'fresh.exit').read_text().strip() == '0', 'Require this snapshot independent strict/frozen/probe readback.')
    arm = report['arm']
    counts = [len(meta[key]) for key in ('native_modules','compact_modules','pending_dense_modules')]
    b.require(counts == ([216,36,0] if arm == 'decoder' else [144,0,108])
        and budget['budget_passed'] is (arm=='decoder') and budget['final_budget_applicable'] is (arm=='decoder'),
        'Actual D/dense inventory or honest final budget semantics differ.')
    if arm == 'dense':
        b.require(all('.self_attn.' in row['name'] for row in meta['native_modules'])
            and all('.mlp.' in row['name'] for row in meta['pending_dense_modules']), 'Only all108 MLP may be dense.')
    hashes.update(report['source_C0_binding']['checkpoint_files_sha256'])
    for path in (report_path,audit_path,report_path.parent/'expected_state.json',report_path.parent/'same_input_output.pt'):
        hashes[str(path)] = b.sha(path)
    return dict(checkpoint_dir=str(directory),checkpoint_id=meta['checkpoint_id'],checkpoint_format=b.checkpoint.FORMAT,
        metadata_sha256=hashes[str(directory/b.checkpoint.META_FILE)],checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=official.file_stat(directory/b.checkpoint.STATE_FILE),bit_budget=budget,
        pipeline_report=str(report_path),pipeline_provenance=meta['provenance'],arm=arm,
        validation_only=report['validation_only'],planned_updates=report['planned_updates'],
        actual_updates=report['actual_updates'],diagnostic_overbudget=arm=='dense',
        data_contract=report['data_contract'],
        source_C0_binding=report['source_C0_binding']),meta

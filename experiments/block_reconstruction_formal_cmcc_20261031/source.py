"""Validate real C0 ancestry, all36 BR32 records, frozen fields and fresh output."""
import math
from pathlib import Path

from . import bindings as b
from . import export


def bound_file(binding, expected=None):
    path = Path(binding['path']).resolve(strict=True)
    b.require((expected is None or path == Path(expected).resolve(strict=True))
        and b.file_binding(path) == binding, 'A bound BR evidence file changed or points outside its run: '+str(path))
    return path


def transition(previous, current, previous_key='output', current_key='input'):
    def records(rows, key):
        return {row['entry_index']: (row['entry_id'], row[key]) for row in rows}
    b.require(len(previous) == len(current) == 160
        and len(records(previous, previous_key)) == 160
        and records(previous, previous_key) == records(current, current_key),
        'All160 independent CPU entries must preserve the exact actual teacher/student prefix relation.')


def validate_block(row, index, pool, allowed, expected, source_state, previous):
    training = row['training']
    names = {name for name in allowed if name.startswith(f'model.layers.{index}.')}
    projections = {f'model.layers.{index}.{name}' for name in b.rolling.PROJECTIONS}
    b.require(row['schema'] == 'BR36_H96_32_block_v1' and row['layer'] == index
        and row['validation_only'] is False and row['completed_prefix'] == list(range(index+1))
        and training['layer'] == index and training['steps'] == 32
        and training['group_indices'] == list(b.GROUPS) and set(training['active_names']) == names
        and training['trainable_leaves'] == len(names) and training['fresh_optimizer'] is True
        and training['frozen_before'] == training['frozen_after']
        and all(training[key] == b.CONTRACT[key] for key in
            ('base_lr', 'schedule_steps', 'warmup_steps', 'optimizer', 'betas', 'eps', 'weight_decay', 'foreach', 'joint_clip')),
        'The actual full block32 update contract, decoder ownership or frozen state differs.')
    updates = training['updates']
    b.require(len(updates) == 32, 'Every block must contain32 actual updates; no short counter substitution.')
    for step, (update, group) in enumerate(zip(updates, pool.groups, strict=True), 1):
        entries = update['entries']
        gradients, motion = update['projection_gradients'], update['parameter_motion']
        b.require(update['step'] == step and update['group_index'] == group.group_index
            and update['lr'] == 1e-6*(1-(step-1)/500) and len(entries) == 9
            and update['valid_tokens'] == group.valid_tokens
            and update['denominator'] == group.denominator(4096)
            and [entry['valid_tokens'] for entry in entries] == [pool.entries[i].valid_tokens for i in group.entry_indices]
            and all(math.isfinite(entry['squared_error']) and entry['squared_error'] >= 0 for entry in entries)
            and update['MSE'] == sum(entry['squared_error'] for entry in entries)/update['denominator']
            and math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip'] > 0
            and update['gradient_norm_after_clip'] == min(1.5, update['gradient_norm_before_clip'])
            and set(gradients) == set(motion) == projections
            and sum(value['leaves'] for value in gradients.values()) == len(names)
            and all(math.isfinite(value['norm']) and value['norm'] > 0 for value in gradients.values())
            and all(value['fp32_delta_L2'] > 0 and value['bf16_changed_elements'] > 0 for value in motion.values())
            and update['Adam']['step'] == step and update['Adam']['initialized_leaves'] == len(names)
            and update['Adam']['every_state_nonempty'] and not update['Adam']['foreach']
            and update['frozen_persistent_fields_exact'] and update['master_and_Adam_state_identity_continuous'],
            'A real BR group, token denominator, Adam/LR trajectory, gradient or BF16 update is missing.')
    commit = row['commit']
    b.require(commit['all_D_BF16_cast_exact'] and set(commit['committed_D']) == names
        and all(commit['committed_D'][name] == expected[name] for name in names),
        'The endpoint D fields must equal each actual block final FP32-master BF16 commit.')
    completed = {name for name in allowed if any(name.startswith(f'model.layers.{i}.') for i in range(index+1))}
    b.require(row['frozen_complement'] == b.fit.proof_from_records(source_state, completed),
              'Each completed prefix must retain its exact C0 persistent complement.')
    for key in ('teacher_advance', 'student_advance'):
        advance = row[key]
        schema = 'BR_independent_teacher_block_advance_v1' if key == 'teacher_advance' else 'BR_committed_student_block_advance_v1'
        b.require(advance['schema'] == schema and advance['layer'] == index and advance['each_unique_entry_advanced_once']
            and advance['CPU_hidden_storage_preserved'] and advance['ordinary_detached_CPU_hidden']
            and len(advance['entries']) == len(pool.entries) == 160,
            'Each teacher/student layer must advance all160 unique detached entries in place exactly once.')
        for entry_index, (entry, actual) in enumerate(zip(advance['entries'], pool.entries, strict=True)):
            shape = [*actual.valid_mask.shape, 4096]
            b.require(entry['entry_index'] == entry_index and entry['entry_id'] == actual.entry_id
                and entry['valid_tokens'] == actual.valid_tokens and entry['padded_tokens'] == actual.padded_tokens
                and entry['input']['shape'] == entry['output']['shape'] == shape
                and entry['input']['dtype'] == entry['output']['dtype'] == 'torch.bfloat16'
                and entry['CPU_storage_preserved'], 'A stream entry identity, shape, dtype or CPU storage changed.')
    b.require(row['after_deployed'] == row['student_advance']['after_deployed']
        == b.rolling._group_statistics(pool, row['student_advance']['entries'], 4096),
        'After-deployment group statistics must come from the same160 actual BF16 student advances.')
    transition(row['teacher_advance']['entries'], row['student_advance']['entries'], 'output', 'teacher_target')
    if previous is not None:
        transition(previous['teacher_advance']['entries'], row['teacher_advance']['entries'])
        transition(previous['student_advance']['entries'], row['student_advance']['entries'])
    else:
        transition(row['teacher_advance']['entries'], row['student_advance']['entries'], 'input', 'input')
    for warm in ('deployment_warmup_before', 'deployment_warmup_after'):
        b.require(row[warm]['native'] == dict(total=216, warmed=216, skipped=0, failed=0)
            and row[warm]['native_group_size'] == 8 and row[warm]['current_compact'] == 1,
            'Every block deployment requires the tested complete same-device native group8 set.')


def execution(directory, report_path, meta):
    """Endpoint-only evidence validation; deliberately never load parent weights."""
    directory, report_path = Path(directory).resolve(strict=True), Path(report_path).resolve(strict=True)
    report = b.read(report_path)
    root = report_path.parent
    b.require(report['schema'] == 'BR36_H96_32_execution_v1' and report['phase'] == 'BR36'
        and report['status'] == 'BR36_training_BF16_export_complete_fresh_pending' and report['exit_code'] == 0
        and report['validation_only'] is report['prototype_only'] is False
        and report['contract'] == b.CONTRACT and report['layer_ids'] == report['completed_layers'] == list(b.LAYERS)
        and report['group_indices'] == list(b.GROUPS) and report['updates_per_layer'] == 32
        and report['total_updates'] == 1152 and report['teacher_frozen_exact']
        and report['teacher_frozen_before'] == report['teacher_frozen_after']
        and (root/'run.exit').read_text().strip() == '0'
        and Path(report['export']['directory']).resolve(strict=True) == directory
        and report['export']['checkpoint_id'] == meta['checkpoint_id'],
        'Require an actual complete formal36x32 BR endpoint, never a prototype, short or intermediate.')
    b.sequence.check_pins(report['source_sha256'])
    b.require(b.verified_short() == report['reused_short'], 'The reused tested numerical core or actual short evidence changed.')
    expected = b.read(bound_file(report['export']['expected_state'], root/'expected_state.json'))
    source_state = b.read(bound_file(report['source_state'], root/'source_state.json'))
    bound_file(report['export']['fixture'], root/'same_input_output.pt')
    allowed = export.decoder_names(meta)
    b.require(set(report['trained_decoder_names']) == allowed and set(expected) == set(source_state)
        and set(expected) == set(meta['tensor_table'])
        and {name: value for name, value in expected.items() if name not in allowed}
            == {name: value for name, value in source_state.items() if name not in allowed}
        and b.fit.proof_from_records(expected, allowed) == report['source_frozen_complement']
        and export.geometry(meta) == report['source_geometry_sha256'],
        'Codes, routes, protection, rotations, nondecoder fields, geometry or actual decoder ownership changed.')
    pool, data_evidence = b.data.load(report['data']['pad_token_id'])
    b.require(report['data'] == data_evidence and report['full_data'] == b.data.summary(pool),
              'The actual original full H96 data sources, identities, shapes or32 groups changed.')
    blocks = report['block_reports']
    b.require(len(blocks) == 36, 'Require36 independently bound completed block records.')
    previous = None
    for index, binding in enumerate(blocks):
        row = b.read(bound_file(binding, root/f'blocks/layer_{index:02d}.json'))
        validate_block(row, index, pool, allowed, expected, source_state, previous)
        previous = row
    provenance = meta['provenance']
    b.require(provenance['schema'] == 'BR36_H96_32_provenance_v1'
        and provenance['validation_only'] is provenance['prototype_only'] is False
        and provenance['execution'] == {key: report[key] for key in provenance['execution']}
        and provenance['contract'] == b.CONTRACT and provenance['source_C0_binding'] == report['source_C0_binding']
        and provenance['completed_layers'] == list(b.LAYERS) and provenance['updates_per_layer'] == 32
        and provenance['total_updates'] == 1152 and provenance['group_indices'] == list(b.GROUPS)
        and set(provenance['trained_decoder_names']) == allowed
        and provenance['source_frozen_complement'] == report['source_frozen_complement']
        and provenance['source_geometry_sha256'] == report['source_geometry_sha256']
        and provenance['source_state'] == report['source_state'] and provenance['block_reports'] == blocks,
        'The real BR trajectory and C0 ancestry must be bound to the saved canonical endpoint.')
    b.deployment.budget_contract(report['export']['bit_budget'], report['source_C0_binding']['bit_budget'])
    return report, expected


def source(directory, report):
    b.configure_readers()
    from experiments.adaptive_class_evaluation_20261030 import bindings as official
    directory, report = Path(directory).resolve(), Path(report).resolve()
    value = b.read(report)
    b.require(value.get('schema') == 'BR36_H96_32_execution_v1'
        and value.get('validation_only') is False and value.get('prototype_only') is False,
        'Only a real formal BR36 endpoint is eligible; B, C0 and short reports are not BR ancestry.')
    meta, budget, hashes = official.inspect(directory)
    value, _expected = execution(directory, report, meta)
    parent, parent_meta = b.bind_C0(value['source_C0_binding']['checkpoint_dir'], value['source_C0_binding']['pipeline_report'])
    b.require(parent == value['source_C0_binding'] and export.geometry(parent_meta) == export.geometry(meta)
        and parent['checkpoint_id'] != meta['checkpoint_id'], 'The actual full C0 ancestry or source geometry changed.')
    b.deployment.budget_contract(budget, parent['bit_budget'])
    audit_path = report.parent/'fresh_audit.json'
    audit = b.read(audit_path)
    b.require((report.parent/'fresh.exit').read_text().strip() == '0'
        and audit['schema'] == 'BR36_H96_32_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['training_report'] == b.file_binding(report)
        and audit['export_meta'] == b.file_binding(directory/b.checkpoint.META_FILE)
        and audit['full_state_exact'] and audit['frozen_complement_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite']
        and audit['bit_budget'] == budget and audit['source_frozen_complement'] == value['source_frozen_complement']
        and audit['source_state'] == value['source_state']
        and audit['expected_state'] == value['export']['expected_state'] and audit['fixture'] == value['export']['fixture']
        and audit['block_reports'] == value['block_reports']
        and audit['source_geometry_sha256'] == value['source_geometry_sha256'],
        'Require this same real BR endpoint independent full-state/frozen/BF16 fresh audit.')
    hashes.update(parent['checkpoint_files_sha256'])
    paths = [report, audit_path, Path(value['source_state']['path']),
        Path(value['export']['expected_state']['path']), Path(value['export']['fixture']['path'])]
    paths.extend(Path(item['path']) for item in value['block_reports'])
    hashes.update({str(path): b.sha(path) for path in paths})
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=official.FORMAT,
        metadata_sha256=hashes[str(directory/b.checkpoint.META_FILE)], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=official.file_stat(directory/b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report), pipeline_provenance=meta['provenance'],
        source_C0_binding=parent, BR_record=b.file_binding(report)), meta

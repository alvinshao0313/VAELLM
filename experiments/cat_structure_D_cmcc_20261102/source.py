"""Read-only completed original-D native identity, real ledger and A-control binding."""
from pathlib import Path

from . import bindings as b


def projection(spec):
    category = spec['name'].rsplit('.', 1)[1]
    bits = dict(q_proj=38, k_proj=38, v_proj=32, o_proj=48, gate_proj=32, up_proj=32, down_proj=48)[category]
    b.require(spec['codebook_dim'] == 32 and spec['stage_codebook_dims'] == [32, 32]
        and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1 and spec['parallel_stage_decode']
        and not spec['has_original_weight'] and not spec['always_use_original']
        and len(spec['stage_decoders']) == 2 and all(item['in_dim'] == bits and item['out_dim'] == 32
            and item['hidden_dim'] == (256 if category == 'gate_proj' else 128)
            and item['num_res_blocks'] == 0 and item['norm_type'] == 'layer'
            and item['activation_type'] == 'swish'
            and item['decoder_type'] == ('asymmetric' if category == 'gate_proj' else 'symmetric')
            for item in spec['stage_decoders']), 'Incorrect original D deployed projection geometry: ' + spec['name'])


def inspect(directory, report_path):
    directory, report_path = Path(directory).resolve(), Path(report_path).resolve()
    report = b.read(report_path)
    b.require(report['schema'] == 'cat_structure_D_native_pipeline_v1'
        and report['status'] == 'completed_D_native_pipeline' and report['exit_code'] == 0
        and report['phase'] == 'formal' and report['validation_only'] is False
        and report['training_contract'] == b.CONTRACT and Path(report['final_model']).resolve() == directory
        and report_path.parent == b.OUTPUT / 'formal'
        and (report_path.parent / 'run.exit').read_text().strip() == '0',
        'Require the actual completed formal original D native endpoint, never a short or running source.')
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified' and report['prepared_sha256'] == b.sha(b.RESULT / 'prepared.json')
        and report['source_binding'] == prepared['source_binding'] == b.source_binding()
        and report['originalA_control'] == b.control(), 'The original actual source or A-control binding changed.')
    b.check_pins(report['pipeline_source_sha256'])
    b.require([row['category'] for row in report['actual_groups']] == list(b.CATEGORIES[4:])
        and all(row['status'] == 'fitted_and_encoded' and row['actual_updates_per_stage'] == 10000
            and row['stages'] == 2 and row['linears'] == 36 and row['vector_batch_per_linear'] == 8192
            and len(row['actual_optimizer_initializations']) == 2 for row in report['actual_groups'])
        and [row['category'] for row in report['recoveries']] == list(b.CATEGORIES[4:])
        and all(row['status'] == 'completed' and row['actual_steps'] == row['actual_schedule_max_steps'] == 500
            for row in report['recoveries']), 'The original M36 stages or true500 recoveries did not complete.')
    meta = b.read(directory / 'checkpoint_meta.json')
    event = report['final_native_source']
    current = b.event(directory)
    # Filesystem extent allocation can settle after close; content identity uses
    # stable path/size/mtime/inode and the actual metadata/runtime hashes.
    saved = {key:value for key,value in event.items() if key != 'state'}
    for value in (current, saved):
        value['weight_stat'] = {key:item for key,item in value['weight_stat'].items() if key != 'allocated_bytes'}
    b.require(meta['format'] == 'vaellm_model_checkpoint_v6' and meta['checkpoint_kind'] == 'final_model'
        and meta['checkpoint_id'] == report['final_checkpoint_id'] == event['checkpoint_id']
        and meta['completed_categories'] == meta['compression_categories'] == list(b.CATEGORIES)
        and meta['converted_module_count'] == len(meta['converted_modules']) == 252
        and not meta['pending_dense_targets'] and not meta['skip_targets'] and meta['lora_config'] is None
        and meta['finalized_status']['inference_ready']
        and current == saved,
        'The actual native final save identity changed.')
    provenance = meta['extra_meta']['cat_structure_D']
    b.require(provenance['schema'] == 'original_D_native_provenance_v1' and provenance['validation_only'] is False
        and provenance['source_binding'] == report['source_binding']
        and provenance['training_contract'] == b.CONTRACT
        and provenance['actual_fit_updates_per_stage'] == 10000
        and provenance['actual_recovery_steps_per_category'] == 500, 'Final native provenance is inconsistent.')
    for spec in meta['converted_modules']:
        projection(spec)
    fresh_path = Path(report['final_strict_reload_report'])
    fresh = b.read(fresh_path)
    b.require(fresh['status'] == 'verified_native_full_state_strict' and fresh['exit_code'] == 0
        and fresh['checkpoint_id'] == meta['checkpoint_id'] and fresh['expected_state_equal']
        and fresh['full_state'] == event['state'] and fresh['persistent_fields'] == 3423,
        'Final full native strict reload has not completed.')
    from train_utils import checkpoint_bpw as ledger
    state = b.torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = {name for spec in meta['converted_modules'] for name, _ in ledger._code_specs(spec)}
    b.require(len(state) == 3423 and len(banks) == 504 and banks <= state.keys(), 'Incomplete native3423 fields/504 code banks.')
    budget = ledger.audit_state_dict(meta, state, max_bpw=2.6, max_lora_rank=0)
    b.require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
        and budget['target_count'] == budget['compressed_target_count'] == 252
        and all(not budget['components'].get(name, {}).get('storage_bits', 0)
            for name in ('dense_weights', 'lora', 'residual_lora')), 'Original complete native rank0/2.6 ledger failed.')
    evidence = dict(report['pipeline_source_sha256'])
    for path in (report_path, fresh_path, report_path.parent / 'run.exit', b.RESULT / 'prepared.json',
            b.RESULT / 'short_validation.json', b.TERMINAL, b.BLUEPRINT,
            b.SOURCE / 'checkpoint_meta.json', b.SOURCE / 'cat_runtime_state.pt',
            directory / 'checkpoint_meta.json', directory / 'cat_runtime_state.pt'):
        evidence[str(path)] = b.sha(path)
    return dict(profile='D', stage='CAT_final', checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'],
        checkpoint_format=meta['format'], metadata_sha256=b.sha(directory / 'checkpoint_meta.json'),
        checkpoint_files_sha256=evidence, evidence_sha256=evidence,
        checkpoint_weight_stat=dict(path=str(directory / meta['state_dict_file']),
            size=event['weight_stat']['bytes'], mtime_ns=event['weight_stat']['mtime_ns'], inode=event['weight_stat']['inode']),
        bit_budget=budget, pipeline_report=str(report_path), pipeline_provenance=provenance,
        source_binding=report['source_binding'], originalA_control=report['originalA_control'],
        training_contract=b.CONTRACT, validation_only=False), meta


def publish(report_path):
    report_path = Path(report_path)
    report = b.read(report_path)
    binding, meta = inspect(report['final_model'], report_path)
    destination = report_path.parent / 'native_audit.json'
    b.require(not destination.exists(), 'Never overwrite a native terminal audit.')
    b.write(destination, dict(schema='original_D_native_source_v1', status='verified',
        serialized_fields=3423, code_bank_count=504, binding=binding))
    return destination


def source(directory, pipeline_report):
    binding, meta = inspect(directory, pipeline_report)
    path = Path(pipeline_report).parent / 'native_audit.json'
    audit = b.read(path)
    b.require(audit['status'] == 'verified' and audit['binding'] == binding, 'The real native terminal audit changed.')
    binding.update(native_audit=str(path), native_audit_sha256=b.sha(path))
    return binding, meta

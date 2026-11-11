"""Import immutable historical evidence without weakening the native model reader."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha, public_consumer_identity
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, inventory_id, require, validate_ledger
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id, semantic_seed
from experiments.cat_auto_protocol_cmcc_20261104.nodes import load_complete_eight_score
from experiments.cat_offline_execution_cmcc_20261104.protocol import binding, collect_native, read, verify_files, write
from experiments.cat_offline_execution_cmcc_20261104.results import extract_bsq_diagnostics


def _path(reference):
    if isinstance(reference, dict):
        path = Path(reference['path']).resolve()
        require(file_sha(path) == reference['sha256'], 'Bound evidence changed: ' + str(path))
        return path
    return Path(reference).resolve()


def normalized_transpose(value, names):
    names = tuple(names)
    require(names and len(names) == len(set(names)), 'A complete distinct projection inventory is required.')
    if type(value) is bool:
        return value
    require(isinstance(value, dict) and set(value) == set(names) and all(type(item) is bool for item in value.values()),
        'Transpose mapping must contain every actual projection and actual booleans.')
    values = set(value.values())
    return next(iter(values)) if len(values) == 1 else {name: value[name] for name in names}


def normalized_recipe(recipe):
    value = deepcopy(recipe)
    value['transpose'] = normalized_transpose(value['transpose'], value['group_signature'])
    return value


def equivalent_recipe(first, second):
    return normalized_recipe(first) == normalized_recipe(second)


def _stat(path):
    path = Path(path)
    if not path.is_file():
        return None
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev)


def payload_availability(row):
    endpoint = row['endpoint']; directory = Path(endpoint['path'])
    meta = read(directory / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == endpoint['checkpoint_id'] and file_sha(directory / 'checkpoint_meta.json') == endpoint['metadata_sha256'],
        'Original metadata/UUID changed; missing raw state is never a replacement for metadata.')
    state = directory / meta['state_dict_file']
    runtime_name = meta.get('extra_meta', {}).get('cat_runtime_state_file')
    runtime = directory / runtime_name if runtime_name else None
    bank = Path(row['bank']['state_file'])
    state_stat, bank_stat = _stat(state), _stat(bank)
    runtime_stat = _stat(runtime) if runtime else None
    return dict(bank_available=bool(bank_stat and bank_stat['bytes'] > 0),
        raw_payload_available=bool(state_stat and state_stat['bytes'] > 0 and (runtime is None or runtime_stat)),
        raw_loaded_in_this_check=False, strict_native_loader_unchanged=True,
        bank=dict(path=str(bank), certified_sha256=row['bank']['state_sha256'], stat=bank_stat),
        raw_state=dict(path=str(state), certified_sha256=endpoint['state_sha256'], stat=state_stat),
        raw_runtime=dict(path=str(runtime), stat=runtime_stat) if runtime else None,
        boundary='Availability is separate from a new strict load; loading/evaluation still verifies the real complete native payload.')


def require_native_payload(row):
    actual = payload_availability(row)
    require(actual['raw_payload_available'], 'Historical quality is valid, but the original raw model is unavailable for strict loading/evaluation.')
    return row['endpoint']


def _contract(row, request, completed, inventory, manifest):
    identity, recipe = manifest['run_identity'], row['recipe']
    require(row['run_identity'] == request['run_identity'] == identity and row['candidate_id'] == candidate_id(identity, recipe),
        'Original candidate ID/full resolved/source9 identity mismatch.')
    require(row['category_id'] == recipe['category_id']
        and set(recipe['group_signature']) == {value['actual_name'] for value in inventory['projections'] if value['category_id'] == recipe['category_id']},
        'The resolved recipe must cover exactly every actual projection in its own semantic category.')
    require(request['parent'] == manifest['parent'] and request['parent']['kind'] == 'BF16'
        and request['parent']['uuid'] == identity['parent_uuid']
        and request['parent']['weights_digest'] == identity['parent_weights_digest'] == digest(identity['model']['weight_shards_sha256'])
        and Path(request['parent']['path']).resolve() == Path(inventory['source_evidence']['path']).resolve(),
        'The actual original BF16 parent path/content changed.')
    require(request['inventory_receipt_sha256'] == manifest['inventory']['sha256']
        and Path(request['inventory_receipt']).resolve() == _path(manifest['inventory'])
        and request['data_plan_sha256'] == manifest['data_plan']['sha256']
        and Path(request['data_plan']).resolve() == Path(manifest['data_plan']['path']).resolve(),
        'Actual inventory or frozen DataPlan source differs.')
    require(completed['status'] == 'completed' and completed['exit_code'] == 0
        and completed['phase'] == request['phase'] == row['phase'] == 'formal'
        and completed['request_identity'] == digest(request)
        and completed['model_identity'] == inventory['model_identity'] == identity['model']
        and completed['inventory_identity'] == inventory_id(inventory)
        and completed['data_identity'] == identity['data_identity']
        and completed['fit_code_identity'] == identity['code_identity']
        and completed['consumer_identity'] == identity['consumer_identity'] == public_consumer_identity(),
        'Incomplete producer or actual model/data/source9/PUBLIC consumer differs.')
    require(len(identity['code_identity']['source_sha256']) == 9, 'Preserve the reviewed source9 fitting identity.')
    verify_files(identity['code_identity']['source_sha256']); verify_files(request['source_sha256'])
    planned = [value for value in request['candidates'] if value['candidate_id'] == row['candidate_id']]
    actuals = [value for value in completed['candidates'] if value['candidate_id'] == row['candidate_id']]
    require(len(planned) == len(actuals) == 1, 'Require exactly the original requested/completed candidate.')
    actual = actuals[0]; fit, bank, fresh = actual['fit_report'], actual['bank'], actual['fresh_evidence']
    require(actual['status'] == fit['status'] == 'completed' and actual['exit_code'] == fit['exit_code'] == 0
        and planned[0]['recipe'] == fit['resolved_recipe'] == bank['recipe'] == recipe
        and planned[0]['purpose'] == actual['purpose'] == row['purpose'] in ('profile', 'offline')
        and fit['candidate_id'] == bank['candidate_id'] == row['candidate_id']
        and fit['parent_identity'] == bank['parent_identity'] == identity
        and bank == row['bank'] and bank['tensor_digest'] == row['bank_digest']
        and bank['inventory_identity'] == inventory_id(inventory) and bank['model_identity'] == identity['model']
        and bank['target_before_digest'] == fit['target_before_digest'] and bank['artifact_id'] == fit['artifact_id'], 'Actual requested/fit/bank identity differs.')
    require(row['formal_eligible'] and actual['formal_result_eligibility'] and fit['formal_eligible'] and bank['formal_eligible']
        and recipe['S'] == 2 and recipe['linear_group_size'] == len(recipe['group_signature']) == fit['M']
        and set(fit['actual_targets']) == set(recipe['group_signature'])
        and fit['technical_steps'] is None and fit['actual_stage_steps'] == fit['original_schedule_steps'] == 10000
        and len(fit['stages']) == 2 and all(stage['actual_steps'] == 10000 for stage in fit['stages'])
        and fit['frozen_prefix_before_digest'] == fit['frozen_prefix_after_digest'], 'Full pooled S2x10K or frozen complement contract failed.')
    seeds = []
    for index, stage in enumerate(fit['stages']):
        require(stage['stage_index'] == index and stage['initialization_digest']
            and stage['init_seed'] == semantic_seed(identity['model'], recipe['category_id'], 'residual_stage_' + str(index + 1), 'vae_initialization', identity['search_seed'])
            and stage['shuffle_seed'] == semantic_seed(identity['model'], recipe['category_id'], 'residual_stage_' + str(index + 1), 'vae_shuffle', identity['search_seed']),
            'Actual stage initialization/shuffle differs from semantic seed contract.')
        seeds.append({key: stage[key] for key in ('stage_index', 'init_seed', 'shuffle_seed')})
    require(row['init_contract'] == dict(seed_rule='semantic_uint32_model_category_stage_purpose_v0', search_seed=identity['search_seed'], stage_seeds=seeds)
        and row['BSQ_diagnostics'] == extract_bsq_diagnostics(fit, True), 'Canonical initialization/BSQ diagnostics differ from actual fit.')
    ledger = validate_ledger(row['ledger']); endpoint = row['endpoint']; fee = ledger['per_category'][recipe['category_id']]
    require(ledger == actual['actual_ledger'] == endpoint['ledger'] and ledger['N'] == inventory['total_target_numel']
        and set(ledger['per_category']) == set(inventory['category_ids'])
        and ledger['source']['kind'] == 'actual_native_v6_tensor_storage'
        and all(sum(value[key] for value in ledger['per_category'].values()) == ledger[key] for key in ('N', 'logical_bits', 'packed_bits'))
        and row['logical_bits'] == fee['logical_bits'] and row['packed_bits'] == fee['packed_bits']
        and all(sum(value[key] for value in row['per_projection_fees'].values()) == fee[key] for key in ('logical_bits', 'packed_bits'))
        and set(row['per_projection_fees']) == set(recipe['group_signature']), 'Actual measured category/projection/full-model ledger does not close.')
    require(all(fresh[key] for key in ('state_exact', 'forward_executed', 'gpu_validation', 'same_input_same_BF16_path_logits_exact'))
        and fresh['checkpoint_id'] == actual['raw_checkpoint']['checkpoint_id'] == endpoint['checkpoint_id'] == ledger['source']['checkpoint_id']
        and fresh['weight_digest'] == endpoint['state_sha256'] == ledger['source']['state_file_sha256']
        and row['artifact_evidence']['bank_state']['sha256'] == bank['state_sha256']
        and Path(row['artifact_evidence']['bank_state']['path']).resolve() == Path(bank['state_file']).resolve(),
        'Original full-state/fresh/certified payload bindings differ.')
    meta = read(Path(endpoint['path']) / 'checkpoint_meta.json')
    require(file_sha(Path(endpoint['path']) / 'checkpoint_meta.json') == endpoint['metadata_sha256'] == ledger['source']['metadata_sha256']
        and meta['checkpoint_id'] == endpoint['checkpoint_id']
        and {value['name']: value for value in meta['converted_modules']} == {value['name']: value for value in bank['converted_modules']},
        'Original native metadata/UUID/topology differs.')
    score = actual['joint_readout']
    from experiments.cat_auto_protocol_cmcc_20261104.proposal_readout import LOSS_CONTRACT
    selection = {key: score[key] for key in ('data_identity', 'teacher_cache_identity', 'record_order_digest', 'loss_contract')}
    require(selection == row['selection_data_identity'] and score['data_identity'] == identity['data_identity']
        and score['loss_contract'] == LOSS_CONTRACT
        and score['full']['records'] == 512 and score['full']['effective_token_weight'] > 0
        and score['full']['KL'] == row['joint_KL'] == row['select_KL']
        and math.isfinite(row['joint_KL']) and score['full']['KL'] == score['full']['numerator'] / score['full']['effective_token_weight'],
        'The original common512 teacher/data/token/pooled KL identity differs.')
    require(actual['layer_functional_readout'] == row['layer_functional_readout'], 'Canonical local readout differs from completed original.')
    return actual


def _full8(row, manifest):
    # Ordinary candidates declare the completed formal directory; the separate
    # up-profile full8 manifest below owns its actual evaluation destination.
    candidates = [Path(row['artifact_evidence']['completed']['path']).parent.parent / 'full8/full8_summary.json']
    source_shards = []
    for item in candidates:
        if item.is_file():
            summary = read(item)
            if summary.get('checkpoint_id') == row['endpoint']['checkpoint_id']:
                source_shards.append(item.resolve())
    # Up profiles store evaluation separately; the audit contains its exact original summary binding.
    if not source_shards:
        from experiments import cat_profile_coverage_cmcc_20261104 as coverage
        snapshot = coverage.Snapshot(); profiles, _ = coverage.expected_sources(snapshot, coverage.DEFAULT_UP, coverage.DEFAULT_EIGHT, coverage.DEFAULT_NIGHT)
        matches = [value for value in profiles if value['candidate_id'] == row['candidate_id']]
        require(len(matches) == 1, 'No explicit original full8 directory for the imported candidate.')
        source_shards = [Path(matches[0]['full8_dir']) / 'full8_summary.json']
    require(len(set(source_shards)) == 1, 'Ambiguous original full8 summary source.')
    summary_path = source_shards[0]; root = summary_path.parent
    summary, evaluation, plan = (read(root / name) for name in ('full8_summary.json', 'evaluation.json', 'plan.json'))
    baseline = read(_path(manifest['baseline']))
    require(baseline['quality_reference_verified'] and baseline['protocol_identity'] == digest(baseline['protocol']) == manifest['protocol_identity']
        and plan['schema'] == 'cat_PUBLIC_native_full8_plan_v0' and plan['endpoint'] == row['endpoint']
        and plan['baseline_receipt'] == manifest['baseline'] and plan['phase'] == 'formal' and plan['formal_eligible']
        and plan['protocol_identity'] == manifest['protocol_identity'] and plan['expected'] == dict(leaf_count=64, sample_count=24742),
        'Original full8 plan/endpoint/common execution protocol differs.')
    verify_files(plan['source_sha256'])
    paths = [_path(value) for value in summary['input_results']]
    require(paths == [(root / name).resolve() for name in ('mmlu.json', 'seven.json')], 'Full8 must retain its original two complete shards.')
    protocol_manifest = dict(plan_path=str(root / 'plan.json'), plan_sha256=file_sha(root / 'plan.json'),
        checkpoint_meta_sha256=row['endpoint']['metadata_sha256'], protocol_id=manifest['protocol_identity'],
        leaf_count=64, sample_count=24742, task_metric_keys=baseline['task_metric_keys'], expected_stages=plan['expected_stages'])
    load_complete_eight_score(summary_path=summary_path, evaluation_path=root / 'evaluation.json', checkpoint_dir=row['endpoint']['path'],
        source_checkpoint_id=row['endpoint']['checkpoint_id'], protocol_manifest=protocol_manifest)
    actual = collect_native(paths, endpoint=row['endpoint'], baseline=baseline)
    require(summary == actual and row['complete_full8_evaluated'] and row['full8_mean'] == actual['mean_percent']
        and row['full8_evidence'] == {key: actual[key] for key in ('complete', 'protocol_identity', 'checkpoint_id', 'leaf_count', 'sample_count',
            'evaluation_digest', 'task_metrics', 'ordered_sample_identity_evidence')}, 'Canonical/summary/raw64-leaf24742 scores differ.')
    return [binding(root / name) for name in ('plan.json', 'evaluation.json', 'full8_summary.json')] + [binding(path) for path in paths], actual


def import_result(reference, manifest, *, expected_recipe=None, existing_audit=None):
    path = _path(reference); row = read(path)
    require(expected_recipe is None or equivalent_recipe(row['recipe'], expected_recipe), 'Reused candidate differs from full resolved recipe.')
    target = Path(manifest['output_root']) / 'imports' / (row['candidate_id'] + '.json')
    if target.is_file():
        receipt = load_receipt(target, manifest, expected_recipe=expected_recipe)
        require(receipt['canonical_reference'] == binding(path), 'Imported original candidate source was replaced.')
        return receipt
    inventory = read(_path(manifest['inventory']))['inventory']
    _path(manifest['data_plan'])
    request_path, completed_path = (_path(row['artifact_evidence'][key]) for key in ('request', 'completed'))
    request, completed = read(request_path), read(completed_path)
    _contract(row, request, completed, inventory, manifest)
    require(row['logical_bits'] <= manifest['caps'][row['category_id']]['cap_bits'], 'Historical measured class fee exceeds frozen cap.')
    availability = payload_availability(row)
    require(availability['bank_available'] and availability['raw_payload_available'],
        'First-time evidence sealing requires the original bank and full raw payload before any controlled release.')
    sources, full8 = _full8(row, manifest)
    sources += [binding(path), binding(request_path), binding(completed_path), binding(Path(row['endpoint']['path']) / 'checkpoint_meta.json')]
    samples = full8['ordered_sample_identity_evidence']['files_sha256']
    value = dict(schema='sc_cat_historical_evidence_import_v1', candidate_id=row['candidate_id'], category_id=row['category_id'],
        canonical_reference=binding(path), original_candidate_id_preserved=True, run_identity=manifest['run_identity'],
        protocol_identity=manifest['protocol_identity'], normalized_recipe_identity=digest(normalized_recipe(row['recipe'])),
        historical_quality_verified=True, source_references=sources,
        sample_files={name: dict(sha256=sha, stat=_stat(name)) for name, sha in samples.items()},
        complete_full8=dict(leaf_count=64, sample_count=24742, mean_percent=row['full8_mean'], evaluation_digest=full8['evaluation_digest']),
        certified_payloads=availability, sealed_epoch=time.time(),
        large_state_bytes_rehashed=False, large_state_proof='Original successful worker strict state/fresh/ledger SHA receipts retained; not a new tensor-byte audit.',
        sample_bytes_reverified=True, bank_installation='Actual install_category_bank still verifies bank SHA/tensors and complete topology.',
        boundary='Historical evidence import is an explicit reader. Native model loading/evaluation is unchanged and never accepts a missing raw state.')
    value['receipt_identity'] = digest(value); write(target, value)
    return dict(value, receipt_reference=binding(target), availability=availability)


def load_receipt(reference, manifest, *, expected_recipe=None, require_bank=True):
    path = _path(reference); receipt = read(path)
    require(receipt.get('schema') == 'sc_cat_historical_evidence_import_v1'
        and receipt['receipt_identity'] == digest({key: value for key, value in receipt.items() if key != 'receipt_identity'})
        and receipt['run_identity'] == manifest['run_identity'] and receipt['protocol_identity'] == manifest['protocol_identity']
        and receipt['historical_quality_verified'], 'Historical evidence receipt identity/source/protocol differs.')
    for source in receipt['source_references']:
        _path(source)
    for name, sample in receipt['sample_files'].items():
        require(_stat(name) == sample['stat'], 'Sealed original per-question sample artifact changed or disappeared: ' + name)
    row = read(_path(receipt['canonical_reference']))
    require(row['candidate_id'] == receipt['candidate_id'] and digest(normalized_recipe(row['recipe'])) == receipt['normalized_recipe_identity']
        and (expected_recipe is None or equivalent_recipe(row['recipe'], expected_recipe)), 'Sealed original recipe/candidate differs.')
    availability = payload_availability(row)
    require(not require_bank or availability['bank_available'], 'Verified historical score exists but original installable bank is unavailable.')
    for name in ('bank', 'raw_state', 'raw_runtime'):
        current, sealed = availability.get(name), receipt['certified_payloads'].get(name)
        if current and current['stat'] is not None:
            require(current['stat'] == sealed['stat'], 'Sealed payload was replaced/modified: ' + current['path'])
    return dict(receipt, receipt_reference=binding(path), availability=availability)


def checked_result(path, manifest, *, expected_recipe=None):
    receipt = import_result(path, manifest, expected_recipe=expected_recipe)
    return read(receipt['canonical_reference']['path'])


def lookup_result(references, manifest, recipe):
    matches = []
    for reference in references:
        row = read(_path(reference))
        if equivalent_recipe(row['recipe'], recipe):
            checked_result(reference, manifest, expected_recipe=recipe)
            matches.append(binding(_path(reference)))
    return sorted(matches, key=lambda value: (read(value['path'])['candidate_id'], value['path']))


def cleanup_completed(manifest, state):
    """Return precise proposals only; undeclared reader dependencies keep payloads."""
    retention = state.get('payload_retention', {})
    protected = set(retention.get('protected_candidate_ids', [])) | set(retention.get('pending_readout_candidate_ids', [])) | set(retention.get('validation_candidate_ids', []))
    for value in manifest['anchors'].values():
        protected.add(value['candidate_id'])
    proposals = []
    for identifier in retention.get('no_future_raw_readers', []):
        if identifier in protected:
            continue
        path = Path(manifest['output_root']) / 'imports' / (identifier + '.json')
        if not path.is_file():
            continue
        receipt = load_receipt(path, manifest); row = read(receipt['canonical_reference']['path'])
        canonical = Path(receipt['canonical_reference']['path']).resolve()
        if row['purpose'] == 'profile' or 'joint' in canonical.parts:
            continue
        raw = Path(row['endpoint']['path']).resolve()
        allowed = (Path(manifest['output_root']) / 'candidates').resolve()
        if not raw.is_relative_to(allowed):
            continue
        candidate_state = read(allowed / identifier / 'candidate.json')
        require(candidate_state.get('status') == 'completed' and candidate_state.get('exit_code') == 0
            and candidate_state.get('candidate_id') == identifier and candidate_state.get('finished_epoch'),
            'Only a genuinely terminal new candidate with no active raw reader can be released.')
        for item in (receipt['availability']['raw_state'], receipt['availability'].get('raw_runtime')):
            if item is None or item['stat'] is None:
                continue
            target = Path(item['path'])
            require(not target.is_symlink() and target.resolve().is_relative_to(raw), 'Never release a symlink or another task payload.')
            proposals.append(dict(candidate_id=identifier, path=str(target), stat=item['stat'],
                allocated_bytes=target.stat().st_blocks * 512, receipt=receipt['receipt_reference'],
                reason='Complete evidence sealed; explicit dependency graph declares no future raw reader; bank and original metrics/metadata retained.'))
    return proposals


def execute_release_plan(manifest, plan, authorization_reference):
    """Release only own reviewed dependency-free raw payloads, after journaling."""
    require(authorization_reference and authorization_reference == manifest['rules']['authorization_reference'],
        'Own-payload release must cite this complete batch authorization and AGENTS standing cleanup authorization.')
    root = Path(manifest['output_root']).resolve()
    journal = root / 'releases' / (digest(plan) + '.json')
    require(not journal.exists(), 'A release plan is immutable and cannot execute twice.')
    state = read(root / 'search.json')
    allowed = cleanup_completed(manifest, state)
    require(all(item in allowed for item in plan) and len({item['path'] for item in plan}) == len(plan),
        'Release plan differs from the latest persisted dependency/retention graph.')
    result = []
    # Recheck all items before the first irreversible action.
    for item in plan:
        target = Path(item['path'])
        require(target.resolve().is_relative_to(root / 'candidates') and not target.is_symlink() and _stat(target) == item['stat'],
            'Release target changed after review or belongs to an old/profile/joint batch.')
        receipt = load_receipt(item['receipt'], manifest)
        require(item['candidate_id'] == receipt['candidate_id'] and item['path'] in {
            value['path'] for key, value in receipt['availability'].items() if key in ('raw_state', 'raw_runtime') and value},
            'Release may only remove its sealed raw state/runtime, never bank/profile/joint/source.')
    write(journal, dict(schema='sc_cat_own_payload_release_journal_v1', status='planned', plan_identity=digest(plan), plan=plan,
        authorization_reference=authorization_reference, AGENTS_standing_cleanup_authorization='2026-09-23', planned_epoch=time.time()))
    try:
        for item in plan:
            target = Path(item['path']); target.unlink()
            require(not target.exists(), 'Reviewed payload release did not complete.')
            result.append(dict(path=item['path'], bytes=item['stat']['bytes'], allocated_bytes=item['allocated_bytes'], reason=item['reason']))
    finally:
        record = read(journal)
        record.update(status='completed' if len(result) == len(plan) else 'incomplete', released=result, finished_epoch=time.time())
        journal.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result

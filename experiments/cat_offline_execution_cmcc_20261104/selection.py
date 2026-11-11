"""v0.2 selection over explicitly frozen real artifacts; never generate a search."""
import argparse
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from experiments.cat_auto_protocol_cmcc_20261104.contracts import (
    digest, inventory_id, require, validate_inventory, validate_recipe, validate_run_identity,
)
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
from experiments.cat_auto_protocol_cmcc_20261104.offline import geometry_key, _resolved_like
from .protocol import binding, code_identity, load_baseline, read, read_native_eight, verify_files, write
from .results import worker_result

VERSION = 'cat_offline_full8_selection_v0.2'


def quality_key(mean_percent, logical_bits, packed_bits, stable_id):
    """Pure ordering, also usable with explicitly labelled mathematical test values."""
    require(isinstance(mean_percent, (float, int)) and not isinstance(mean_percent, bool)
        and math.isfinite(mean_percent) and 0 <= mean_percent <= 100, 'Require an unrounded finite percentage full8 score.')
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (logical_bits, packed_bits)) and logical_bits <= packed_bits, 'Require actual logical/packed integer fees.')
    require(bool(stable_id), 'A stable candidate ID or rate vector is required.')
    return -mean_percent, logical_bits, packed_bits, stable_id


def capacity_improved(current_score, current_fees, candidate_score, candidate_fees):
    quality_key(current_score, *current_fees, 'current')
    quality_key(candidate_score, *candidate_fees, 'candidate')
    return candidate_score > current_score or (candidate_score == current_score and candidate_fees < current_fees)


def _seal(value, key):
    return dict(value, **{key: digest(value)})


def _verify(value, key):
    require(value.get('schema') == VERSION and value.get(key) == digest({name: item for name, item in value.items() if name != key}),
        'Frozen selection set or its protocol version changed.')


def _path(path):
    return str(Path(path).resolve())


def _reference(reference):
    require(file_sha(reference['path']) == reference['sha256'], 'Bound selection source changed: ' + reference['path'])
    return read(reference['path'])


def _context(inventory_receipt, baseline_path, run_identity):
    inventory = validate_inventory(read(inventory_receipt)['inventory'])
    baseline = load_baseline(baseline_path)
    identity = validate_run_identity(run_identity)
    require(identity['model'] == inventory['model_identity'] == baseline['model_identity']
        and identity['parent_weights_digest'] == digest(inventory['model_identity']['weight_shards_sha256']),
        'Offline selection requires this complete original BF16 model/content parent.')
    require(binding(inventory_receipt) == baseline['source_evidence']['inventory_receipt'], 'Selection inventory differs from the actual BF16 reference.')
    return dict(inventory_receipt=binding(inventory_receipt), inventory_identity=inventory_id(inventory),
        baseline_receipt=binding(baseline_path), baseline_reference_id=baseline['reference_id'],
        protocol_identity=baseline['protocol_identity'], run_identity=identity, source_sha256=code_identity())


def require_same_context(actual, expected):
    require(all(actual[key] == expected[key] for key in ('run_identity', 'inventory_receipt', 'baseline_receipt', 'protocol_identity')),
        'Budget/evaluation evidence belongs to another original model/parent/inventory/BF16/protocol context.')


def _verify_context(frozen):
    _reference(frozen['inventory_receipt']); _reference(frozen['baseline_receipt'])
    verify_files(frozen['source_sha256'])
    expected = _context(frozen['inventory_receipt']['path'], frozen['baseline_receipt']['path'], frozen['run_identity'])
    require(all(frozen[key] == value for key, value in expected.items()), 'Frozen model, source code, parent or common protocol differs.')
    return validate_inventory(read(frozen['inventory_receipt']['path'])['inventory'])


def _expected(row, identity, category, purpose):
    recipe = validate_recipe(row['recipe'])
    require(recipe['S'] == 2, 'SC-CAT profile/offline/live/replay baseline is fixed S2; residual-stage search is excluded.')
    require(recipe['category_id'] == category and row['candidate_id'] == candidate_id(identity, recipe),
        'Frozen candidate ID/recipe/category differs from its full planned identity.')
    request = row['worker']['request']
    actual_request = _reference(request)
    planned = [item for item in actual_request['candidates'] if item['candidate_id'] == row['candidate_id']]
    require(actual_request['phase'] == 'formal' and actual_request['parent']['kind'] == 'BF16'
        and actual_request['run_identity'] == identity and len(planned) == 1
        and planned[0]['recipe'] == recipe and planned[0]['purpose'] == purpose,
        'Expected endpoint must be present in its actual formal same-BF16 request with the declared purpose.')
    return dict(candidate_id=row['candidate_id'], recipe=recipe, recipe_digest=digest(recipe),
        worker=dict(completed=_path(row['worker']['completed']), request=dict(path=_path(request['path']), sha256=request['sha256'])))


def _full8_paths(value):
    return {name: _path(value[name]) for name in ('summary', 'evaluation')}


def validate_endpoint_set(rows):
    """Count explicit distinct endpoints; this does not certify an entire executor."""
    require(1 <= len(rows) <= 19 and all(isinstance(row['slot'], int) and not isinstance(row['slot'], bool)
        and 1 <= row['slot'] <= 20 and row['slot'] != 16 for row in rows) and len({row['slot'] for row in rows}) == len(rows)
        and len({row['candidate_id'] for row in rows}) == len(rows)
        and len({row['recipe_digest'] for row in rows}) == len(rows), 'Freeze1..19 distinct endpoints in original slots1..20 excluding fixed-empty16; no transferred quota or repeated fits.')


def freeze_block(*, inventory_receipt, baseline_path, run_identity, category_id, stage, class_cap_bits, expected):
    """Freeze only announced valid distinct endpoints; skipped/illegal slots stay with the caller."""
    context = _context(inventory_receipt, baseline_path, run_identity)
    inventory = read(inventory_receipt)['inventory']
    require(category_id in inventory['category_ids'] and stage in ('S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10', 'S11', 'S12'),
        'Declare the actual category and existing offline decision stage.')
    require(isinstance(class_cap_bits, int) and not isinstance(class_cap_bits, bool) and class_cap_bits > 0,
        'Use an explicitly allocated positive category logical cap.')
    rows = []
    for row in expected:
        require(isinstance(row['slot'], int) and not isinstance(row['slot'], bool) and 1 <= row['slot'] <= 20 and row['slot'] != 16,
            'Each distinct endpoint uses its original slot1..20, excluding the fixed-empty slot16.')
        actual = _expected(row, context['run_identity'], category_id, 'offline')
        request = _reference(actual['worker']['request'])
        planned = next(item for item in request['candidates'] if item['candidate_id'] == actual['candidate_id'])
        require(planned['class_cap_bits'] == class_cap_bits, 'Frozen block cap differs from its actual fit request.')
        cap = _reference(planned['class_cap_evidence'])
        _verify(cap, 'certificate_identity')
        require_same_context(cap, context)
        require(cap['kind'] == 'actual_joint_full8_cap_certificate' and cap['formal_eligible']
            and cap['categories'][category_id]['cap_bits'] == class_cap_bits,
            'Offline slots need actual v0.2 full8 cap provenance; old KL-only certificates cannot enter.')
        rows.append(dict(actual, slot=row['slot'], full8=_full8_paths(row['full8']), class_cap_evidence=planned['class_cap_evidence']))
    validate_endpoint_set(rows)
    require(len({digest(row['class_cap_evidence']) for row in rows}) == 1, 'One offline block must use one frozen actual joint/cap certificate.')
    rows.sort(key=lambda row: row['slot'])
    return _seal(dict(schema=VERSION, kind='offline_block', **context, category_id=category_id, stage=stage,
        class_cap_bits=class_cap_bits, expected=rows, generation_policy='external_explicit_frozen_candidates_only',
        coverage_scope='explicit_frozen_set_only_not_full19_search_slot16_fixed_empty'), 'block_identity')


def _worker(row, frozen):
    result = worker_result(row['worker']['completed'], row['candidate_id'], request_reference=row['worker']['request'])
    require(result['recipe']['S'] == row['recipe']['S'] == 2, 'Actual SC-CAT measured bank and frozen recipe must both use fixed S2.')
    require(result['recipe'] == row['recipe'] and digest(result['recipe']) == row['recipe_digest']
        and result['run_identity'] == frozen['run_identity'], 'Actual candidate/recipe/model/parent/data/code/consumer or semantic seed identity differs.')
    return result


def _pending_worker(row):
    path = Path(row['worker']['completed'])
    if not path.is_file():
        return 'fit_pending'
    completed = read(path)
    if completed.get('status') != 'completed' or completed.get('exit_code') != 0:
        return 'fit_dependency_' + str(completed.get('status', 'unknown'))
    return None


def _pending_full8(row):
    if not all(Path(path).is_file() for path in row['full8'].values()):
        return 'evaluation_pending'
    evaluation = read(row['full8']['evaluation'])
    if evaluation.get('status') != 'complete_original_full8' or evaluation.get('exit_code') != 0:
        return 'evaluation_dependency_' + str(evaluation.get('status', 'unknown'))
    return None


def _attach_candidate(result, row, frozen):
    require(result['purpose'] == 'offline', 'Offline slots require actual offline-purpose artifacts, not profile/live/technical substitutions.')
    score, receipt = read_native_eight(summary_path=row['full8']['summary'], evaluation_path=row['full8']['evaluation'])
    require(receipt['complete'] and receipt['formal_eligible'] and receipt['phase'] == 'formal'
        and receipt['endpoint'] == result['endpoint'] and receipt['checkpoint_id'] == score.evidence['checkpoint_id']
        and receipt['protocol_identity'] == frozen['protocol_identity'], 'Actual full8 endpoint UUID/weights/complete protocol differs from its own formal fit.')
    fee = result['ledger']['per_category'][result['category_id']]
    require(result['logical_bits'] == fee['logical_bits'] <= frozen['class_cap_bits']
        and result['packed_bits'] == fee['packed_bits'], 'Actual category fees exceed the frozen cap or differ from the complete ledger.')
    require(math.isclose(float(score.mean) * 100, receipt['mean_percent'], rel_tol=0, abs_tol=1e-12), 'Reader full8 score arithmetic differs.')
    return dict(result, full8_mean=receipt['mean_percent'], full8_receipt=receipt,
        selection_sources=dict(worker=result['artifact_evidence'], full8={name: binding(path) for name, path in row['full8'].items()}))


def load_candidate(row, frozen):
    """No preadapted score or caller-supplied verified flag is accepted here."""
    return _attach_candidate(_worker(row, frozen), row, frozen)


def _rank_record(result):
    return dict(candidate_id=result['candidate_id'], full8_mean=result['full8_mean'], logical_bits=result['logical_bits'],
        packed_bits=result['packed_bits'], diagnostic_KL=result['joint_KL'], recipe_digest=digest(result['recipe']),
        checkpoint_id=result['endpoint']['checkpoint_id'], evaluation_digest=result['full8_receipt']['evaluation_digest'],
        task_metrics=result['full8_receipt']['task_metrics'], sources=result['selection_sources'])


def _request_cap(row, frozen):
    request = _reference(row['worker']['request'])
    planned = [item for item in request['candidates'] if item['candidate_id'] == row['candidate_id']]
    require(request['phase'] == 'formal' and request['run_identity'] == frozen['run_identity']
        and len(planned) == 1 and planned[0]['purpose'] == 'offline', 'Only its actual formal offline request can bind a candidate cap.')
    planned = planned[0]
    require(planned['class_cap_evidence'] == row['class_cap_evidence']
        and planned['class_cap_bits'] == frozen['class_cap_bits'], 'Selection cap source/bits differs from the actual frozen worker request.')
    cap = _reference(row['class_cap_evidence']); _verify(cap, 'certificate_identity')
    require_same_context(cap, frozen)
    require(cap['kind'] == 'actual_joint_full8_cap_certificate' and cap['formal_eligible']
        and cap['categories'][frozen['category_id']]['cap_bits'] == frozen['class_cap_bits'],
        'Imported selection input cannot replace its own actual v0.2 full8 cap evidence.')
    return cap


def _load_block(frozen):
    _verify(frozen, 'block_identity'); _verify_context(frozen)
    require(frozen['kind'] == 'offline_block', 'An offline candidate block is required.')
    validate_endpoint_set(frozen['expected'])
    require(len({digest(row['class_cap_evidence']) for row in frozen['expected']}) == 1, 'Imported offline blocks cannot mix cap certificates.')
    pending, results, cap_receipts = [], [], set()
    for row in frozen['expected']:
        proof = row['class_cap_evidence']; cap = _request_cap(row, frozen)
        if proof['sha256'] not in cap_receipts:
            require(select_joint(cap['frozen_joint_set']) == cap, 'Actual full8 cap certificate no longer matches all of its compared source artifacts.')
            cap_receipts.add(proof['sha256'])
        reason = _pending_worker(row)
        if reason is None:
            # Reject bad/technical fit evidence even while its evaluation is pending.
            actual = _worker(row, frozen)
            require(actual['purpose'] == 'offline', 'Offline blocks cannot rank profile/live worker artifacts.')
            reason = _pending_full8(row)
        if reason is not None:
            pending.append(dict(candidate_id=row['candidate_id'], slot=row['slot'], status=reason))
        else:
            results.append(_attach_candidate(actual, row, frozen))
    if pending:
        return dict(status='pending', block_identity=frozen['block_identity'], pending=pending,
            completed_candidate_ids=[row['candidate_id'] for row in results], winner_candidate_id=None, formal_eligible=False), None
    require(len({digest(row['selection_data_identity']) for row in results}) == 1
        and len({digest({key: row['init_contract'][key] for key in ('seed_rule', 'search_seed')}) for row in results}) == 1
        and len({row['full8_receipt']['ordered_sample_identity_evidence']['ordered_input_identity'] for row in results}) == 1,
        'Block candidates require the same real512/teacher/loss, semantic initialization rule and ordered full8 inputs.')
    results.sort(key=lambda row: quality_key(row['full8_mean'], row['logical_bits'], row['packed_bits'], row['candidate_id']))
    certificate = _seal(dict(schema=VERSION, kind='complete_offline_block_selection', status='complete', formal_eligible=True,
        block_identity=frozen['block_identity'], category_id=frozen['category_id'], stage=frozen['stage'],
        expected_candidate_ids=[row['candidate_id'] for row in frozen['expected']], protocol_identity=frozen['protocol_identity'],
        run_identity_digest=digest(frozen['run_identity']), ranking=[_rank_record(row) for row in results],
        winner_candidate_id=results[0]['candidate_id'], quality_rule='(-S8,class_logical_bits,class_packed_bits,candidate_id)',
        coverage_scope=frozen['coverage_scope'],
        scope='This frozen block only; not a completed S0-S13 search or a formal quality acceptance.'), 'selection_identity')
    return certificate, results


def select_block(frozen):
    return _load_block(frozen)[0]


def _capacity_decision(current, candidate):
    accepted = capacity_improved(current['full8_mean'], (current['logical_bits'], current['packed_bits']),
        candidate['full8_mean'], (candidate['logical_bits'], candidate['packed_bits']))
    return dict(current_candidate_id=current['candidate_id'], candidate_id=candidate['candidate_id'], accepted=accepted,
        selected_candidate_id=(candidate if accepted else current)['candidate_id'],
        reason='strict_full8_improvement' if candidate['full8_mean'] > current['full8_mean'] else
            ('equal_full8_lower_actual_fee' if accepted else 'no_full8_or_fee_improvement'))


def _conditional_queue(reference, frozen, base):
    queue = _reference(reference)
    require(queue['schema'] == VERSION and queue['kind'] == 'explicit_conditional_queue'
        and queue['authorization_reference'] and queue['base_candidate_id'] == base['candidate_id']
        and queue['run_identity_digest'] == digest(frozen['run_identity'])
        and queue['protocol_identity'] == frozen['protocol_identity'] and 1 <= len(queue['instructions']) <= 3,
        'Require an explicitly confirmed frozen queue bound to this actual base/parent/protocol; no trigger defaults.')
    if any(item['kind'] == 'BSQ' for item in queue['instructions']):
        # Only the rule can be frozen at segment entry. Numeric directions and
        # current-result diagnostics belong to a later per-slot decision;
        # that generator remains unreviewed, so an opaque rule reference is
        # insufficient to make this queue executable.
        return dict(queue, configuration_status='unconfigured_BSQ_generation')
    return queue


def _current_bsq_diagnostics(current, instruction):
    if instruction['kind'] == 'BSQ':
        require(current.get('BSQ_diagnostics') and instruction['diagnostics_source_candidate_id'] == current['candidate_id']
            and instruction['BSQ_diagnostics_digest'] == digest(current['BSQ_diagnostics']),
            'BSQ must read the real current accepted baseline diagnostics at this slot, not the initial capacity base.')


def _conditional_transition(inventory, current, candidate, instruction):
    base = current['recipe']; kind = instruction['kind']; changes = {}
    require(base['S'] == candidate['recipe']['S'] == 2, 'SC-CAT fixed S2 rejects the old S4 condition/slot16 path.')
    if kind == 'S4':
        raise ValueError('SC-CAT slot16 is fixed empty; S4 is not an eligible baseline instruction.')
    elif kind in ('encoder256', 'decoder256'):
        key = 'encoder' if kind == 'encoder256' else 'decoder'
        require(base[key]['hidden_dim'] == 128, 'Width instruction must change128 to256 once.')
        changes[key+'_width'] = 256
    elif kind == 'BSQ':
        require(instruction['field'] in ('commitment_loss_weight', 'entropy_loss_weight'), 'Explicit BSQ instruction changes one registered contribution.')
        changes['loss_overrides'] = {key: value for key, value in base['loss'].items() if key in ('commitment_loss_weight', 'entropy_loss_weight')}
        changes['loss_overrides'][instruction['field']] = instruction['value']
    elif kind in ('protection_neighbor', 'output_axis'):
        changes['protection_counts'] = instruction['counts']
        if kind == 'output_axis':
            changes['protection_axis'] = 'output'
    else:
        raise ValueError('Conditional instruction remains unconfigured: ' + str(kind))
    expected = _resolved_like(inventory, base, **changes)
    require(candidate['recipe'] == expected and candidate['recipe'] != current['recipe'],
        'Conditional candidate is not exactly one frozen instruction applied to the current accepted recipe.')


def capacity_decision(frozen, *, current_id, candidate_id, queue_reference=None, instruction_index=None, prior_trial_ids=()):
    certificate, results = _load_block(frozen)
    if results is None:
        return certificate
    by_id = {row['candidate_id']: row for row in results}
    require(current_id != candidate_id and {current_id, candidate_id} <= set(by_id), 'Both actual distinct capacity endpoints must be in the complete frozen block.')
    if queue_reference is None:
        return dict(status='unconfigured_conditional_queue', selection_identity=certificate['selection_identity'], accepted=None)
    queue_source = _reference(queue_reference)
    require(queue_source['base_candidate_id'] in by_id, 'The frozen conditional base must be in this complete block.')
    queue = _conditional_queue(queue_reference, frozen, by_id[queue_source['base_candidate_id']])
    if queue.get('configuration_status'):
        return dict(status=queue['configuration_status'], accepted=None, selection_identity=certificate['selection_identity'])
    require(isinstance(instruction_index, int) and not isinstance(instruction_index, bool) and 0 <= instruction_index < len(queue['instructions']), 'Declare the actual frozen instruction index.')
    require(len(prior_trial_ids) == instruction_index and len(set(prior_trial_ids)) == len(prior_trial_ids)
        and set(prior_trial_ids) <= set(by_id), 'Bind every earlier frozen capacity trial before applying this instruction.')
    prior = by_id[queue['base_candidate_id']]
    inventory = read(frozen['inventory_receipt']['path'])['inventory']
    for trial, instruction in zip(prior_trial_ids, queue['instructions'][:instruction_index], strict=True):
        _current_bsq_diagnostics(prior, instruction)
        _conditional_transition(inventory, prior, by_id[trial], instruction)
        prior = by_id[_capacity_decision(prior, by_id[trial])['selected_candidate_id']]
    require(prior['candidate_id'] == current_id, 'Current capacity recipe is not the actual accepted predecessor of this instruction.')
    _current_bsq_diagnostics(by_id[current_id], queue['instructions'][instruction_index])
    _conditional_transition(inventory, by_id[current_id], by_id[candidate_id], queue['instructions'][instruction_index])
    return dict(_capacity_decision(by_id[current_id], by_id[candidate_id]), status='complete',
        selection_identity=certificate['selection_identity'], block_identity=frozen['block_identity'],
        queue_reference=queue_reference, instruction_index=instruction_index, prior_trial_ids=list(prior_trial_ids))


def retained_geometries(frozen, *, count=2):
    certificate, results = _load_block(frozen)
    if results is None:
        return certificate
    require(count == 2, 'The confirmed core retains two distinct geometries.')
    retained = []
    for row in results:
        if geometry_key(row['recipe']) not in {geometry_key(old['recipe']) for old in retained}:
            retained.append(row)
        if len(retained) == count:
            break
    return dict(status='complete', selection_identity=certificate['selection_identity'],
        retained_candidate_ids=[row['candidate_id'] for row in retained], coverage_sufficient=len(retained) == count)


def interaction_basis(frozen, *, capacity_base_id, capacity_trial_ids, retained_ids, queue_reference=None):
    """Return the slot20 dependency decision; the caller supplies any new resolved recipe."""
    certificate, results = _load_block(frozen)
    if results is None:
        return certificate
    by_id = {row['candidate_id']: row for row in results}
    require(len(capacity_trial_ids) <= 3 and len(set(capacity_trial_ids)) == len(capacity_trial_ids)
        and len(set(retained_ids)) == len(retained_ids) <= 2
        and {capacity_base_id, *capacity_trial_ids, *retained_ids} <= set(by_id), 'Use the actual frozen capacity sequence and at most two retained geometries.')
    base = current = by_id[capacity_base_id]
    if queue_reference is None:
        return dict(status='unconfigured_conditional_queue', slot=20, selection_identity=certificate['selection_identity'],
            generation_status='unconfigured', other_geometry_id=None)
    queue = _conditional_queue(queue_reference, frozen, base)
    if queue.get('configuration_status'):
        return dict(status=queue['configuration_status'], slot=20, generation_status='unconfigured', other_geometry_id=None)
    require(len(capacity_trial_ids) == len(queue['instructions']), 'Finish every frozen sequential capacity instruction before the interaction dependency.')
    decisions = []
    inventory = read(frozen['inventory_receipt']['path'])['inventory']
    for candidate, instruction in zip(capacity_trial_ids, queue['instructions'], strict=True):
        require(candidate != current['candidate_id'], 'An already accepted endpoint is not a new capacity trial.')
        _current_bsq_diagnostics(current, instruction)
        _conditional_transition(inventory, current, by_id[candidate], instruction)
        decision = _capacity_decision(current, by_id[candidate]); decisions.append(decision)
        current = by_id[decision['selected_candidate_id']]
    others = [row for row in results if row['candidate_id'] in retained_ids and geometry_key(row['recipe']) != geometry_key(base['recipe'])]
    accepted = [row['candidate_id'] for row in decisions if row['accepted']]
    return dict(status='ready_for_explicit_recipe' if others else 'empty_no_other_preserved_geometry', slot=20,
        selection_identity=certificate['selection_identity'], capacity_decisions=decisions, capacity_base_id=capacity_base_id,
        queue_reference=queue_reference,
        final_capacity_id=current['candidate_id'], other_geometry_id=others[0]['candidate_id'] if others else None,
        role='accepted_bundle_other_preserved_geometry' if accepted else 'optimizer_multiplier_other_preserved_geometry',
        optimizer=deepcopy(base['recipe']['optimizer']), accepted_candidate_ids=accepted,
        generation_policy='No neighbor, BSQ, layer-group or resolved-recipe generation; legality/dedup/cap still required.')


def best_different_from_anchor(ranked_ids, anchor_id):
    require(anchor_id in ranked_ids, 'The declared actual anchor must be in the complete ranked set.')
    return next((candidate for candidate in ranked_ids if candidate != anchor_id), None)


def close_category(frozen, *, anchor_id):
    certificate, results = _load_block(frozen)
    if results is None:
        return certificate
    require(frozen['stage'] == 'S12' and anchor_id in {row['candidate_id'] for row in results}, 'Close only the explicit complete category set with its actual anchor.')
    return _seal(dict({key: value for key, value in certificate.items() if key != 'selection_identity'}, anchor_candidate_id=anchor_id,
        best_different_candidate_id=best_different_from_anchor([row['candidate_id'] for row in results], anchor_id),
        raw_full8_endpoint_count=len(results), maximum_distinct_endpoints=19, fixed_empty_slots=[16]), 'selection_identity')


def freeze_order(category_blocks, *, baseline_path):
    """S13 keeps the confirmed cross-category d_c/KL/ID tie rule."""
    baseline = load_baseline(baseline_path)
    inventory = read(baseline['source_evidence']['inventory_receipt']['path'])['inventory']
    require(set(category_blocks) == set(inventory['category_ids']), 'Order freezing needs every actual category.')
    selected, certificates, pending = [], {}, {}
    for category, block in category_blocks.items():
        require(block['category_id'] == category and block['stage'] == 'S12'
            and block['baseline_receipt'] == binding(baseline_path), 'Order needs complete S12 sets and the identical actual BF16 reference.')
        certificate, results = _load_block(block); certificates[category] = certificate
        if results is None:
            pending[category] = certificate['pending']
        else:
            selected.append(results[0])
    if pending:
        return dict(status='pending', categories=pending, frozen_order_identity=None, formal_eligible=False)
    require(len({digest(row['run_identity']) for row in selected}) == 1
        and len({digest(row['selection_data_identity']) for row in selected}) == 1
        and len({row['full8_receipt']['ordered_sample_identity_evidence']['ordered_input_identity'] for row in selected}) == 1,
        'Cross-class order requires one original BF16/data/code/consumer/512/protocol and actual ordered candidate inputs.')
    evidence = [dict(category_id=row['category_id'], selected_candidate_id=row['candidate_id'],
        full8_mean=row['full8_mean'], d_c=baseline['mean_percent']-row['full8_mean'], diagnostic_KL=row['joint_KL'],
        sources=row['selection_sources'], selection_identity=certificates[row['category_id']]['selection_identity']) for row in selected]
    evidence.sort(key=lambda row: (row['d_c'], row['diagnostic_KL'], row['category_id']))
    return _seal(dict(schema=VERSION, kind='frozen_class_order', status='complete', formal_eligible=True,
        ordered_categories=[row['category_id'] for row in evidence], evidence=evidence,
        baseline_receipt=binding(baseline_path), protocol_identity=baseline['protocol_identity'],
        BF16_ordered_pairing_verified=False, scope='Explicit complete category sets; independent BF16 losses are not additive.'), 'frozen_order_identity')


def freeze_joint(*, inventory_receipt, baseline_path, run_identity, proposals, cap_allocation=None):
    """Freeze actual proposed vectors; no implicit 4+4 limit or proposal generator."""
    context = _context(inventory_receipt, baseline_path, run_identity)
    categories = sorted(read(inventory_receipt)['inventory']['category_ids'])
    rows = []
    for proposal in proposals:
        require(set(proposal['banks']) == set(categories), 'Joint proposals must name every actual category bank.')
        banks = {category: dict(_expected(proposal['banks'][category], context['run_identity'], category, 'profile'),
            bank_digest=proposal['banks'][category]['bank_digest']) for category in categories}
        rates = [[category, str(Fraction(banks[category]['recipe']['nominal_main_rate']))] for category in categories]
        require(proposal['rate_vector'] == rates, 'Joint rate vector must match its own explicit complete recipes.')
        expected_id = digest(dict(rates=rates, banks={category: row['bank_digest'] for category, row in banks.items()}))
        require(proposal['proposal_id'] == expected_id, 'Joint proposal ID must bind its full stable vector and actual banks.')
        rows.append(dict(proposal_id=expected_id, rate_vector=rates, banks=banks, full8=_full8_paths(proposal['full8'])))
    require(rows and len({row['proposal_id'] for row in rows}) == len(rows), 'Freeze the nonempty exact distinct announced joint set.')
    if cap_allocation is not None:
        require(cap_allocation.get('authorization_reference') and set(cap_allocation['categories']) == set(categories),
            'Category allocation needs an explicit confirmed reference and every category; no default reserve rule.')
        require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in cap_allocation['categories'].values()),
            'Explicit category caps must be positive logical integer bits.')
    return _seal(dict(schema=VERSION, kind='joint_set', **context, proposals=sorted(rows, key=lambda row: row['proposal_id']),
        cap_allocation=deepcopy(cap_allocation), proposal_generation='external_explicit_frozen_proposals_only'), 'joint_set_identity')


def _load_joint(proposal, frozen, inventory):
    import torch
    from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
    from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
    banks = {category: _worker(row, frozen) for category, row in proposal['banks'].items()}
    require(all(row['purpose'] == 'profile' and row['bank_digest'] == proposal['banks'][category]['bank_digest']
        for category, row in banks.items()), 'Joint profile-bank origins or actual tensor identities differ.')
    require(len({digest(row['selection_data_identity']) for row in banks.values()}) == 1, 'Every bank must share the actual fixed512/teacher/loss identity.')
    score, receipt = read_native_eight(summary_path=proposal['full8']['summary'], evaluation_path=proposal['full8']['evaluation'])
    endpoint = receipt['endpoint']; directory = Path(endpoint['path']); meta = read(directory/'checkpoint_meta.json')
    require(receipt['complete'] and receipt['formal_eligible'] and receipt['phase'] == 'formal'
        and receipt['protocol_identity'] == frozen['protocol_identity'] and endpoint['model_identity'] == inventory['model_identity'],
        'Joint needs its actual complete whole-model formal full8 and common source/protocol.')
    names = {row['actual_name'] for row in inventory['projections']}
    specs = {spec['name']: spec for row in banks.values() for spec in row['bank']['converted_modules']}
    require(set(specs) == names and specs == {spec['name']: spec for spec in meta['converted_modules']}
        and not meta['pending_dense_targets'] and not meta['skip_targets'], 'Joint deployment topology is not exactly the complete bank union.')
    state = torch.load(directory/meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    owned = set()
    for category, row in banks.items():
        group = set(row['recipe']['group_signature'])
        keys = {key for key in state if any(key.startswith(name+'.') for name in group)}
        bank = torch.load(row['bank']['state_file'], map_location='cpu', weights_only=True, mmap=True)
        require(keys == set(bank) and tensor_digest({key: state[key] for key in keys}) == row['bank_digest'],
            'Whole-model deployment differs from an actual fitted bank; additive scores/boolean installation flags cannot substitute.')
        owned.update(keys)
    reference = next(iter(banks.values()))['endpoint']
    source_meta = read(Path(reference['path'])/'checkpoint_meta.json')
    original = torch.load(Path(reference['path'])/source_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    outside = {key: value for key, value in state.items() if key not in owned}
    source_outside = {key: value for key, value in original.items() if not any(key.startswith(name+'.') for name in names)}
    require(tensor_digest(outside) == tensor_digest(source_outside), 'Joint installation changed the full original BF16 complement.')
    ledger = measure_checkpoint(directory, inventory)
    require(ledger['final_budget_applicable'] and ledger['final_budget_passed'] and ledger['future_dense_bits'] == 0,
        'Joint actual full-inventory deployment exceeds logical2.6 or still contains future dense targets.')
    for category, row in banks.items():
        require(ledger['per_category'][category] == row['ledger']['per_category'][category], 'Joint actual category fees differ from its own measured bank.')
    require(all(ledger[key] == endpoint['ledger'][key] for key in ('N', 'logical_bits', 'packed_bits', 'per_category', 'component_fees')),
        'Joint endpoint ledger differs from remeasured actual complete deployment.')
    require(math.isclose(float(score.mean)*100, receipt['mean_percent'], rel_tol=0, abs_tol=1e-12), 'Actual joint full8 arithmetic differs.')
    return dict(proposal_id=proposal['proposal_id'], rate_vector=proposal['rate_vector'], full8_mean=receipt['mean_percent'],
        logical_bits=ledger['logical_bits'], packed_bits=ledger['packed_bits'], ledger=ledger, endpoint=endpoint,
        selection_data_identity=next(iter(banks.values()))['selection_data_identity'],
        full8_receipt=receipt, bank_sources={category: dict(candidate_id=row['candidate_id'], bank_digest=row['bank_digest'],
            artifact_evidence=row['artifact_evidence'], recipe_digest=digest(row['recipe'])) for category, row in banks.items()},
        full8_sources={name: binding(path) for name, path in proposal['full8'].items()}, actual_complete_bank_union_verified=True)


def select_joint(frozen):
    _verify(frozen, 'joint_set_identity'); inventory = _verify_context(frozen)
    require(frozen['kind'] == 'joint_set', 'An explicit frozen joint set is required.')
    pending, results = [], []
    for proposal in frozen['proposals']:
        for row in proposal['banks'].values():
            _reference(row['worker']['request'])
        missing = {category: reason for category, row in proposal['banks'].items() if (reason := _pending_worker(row)) is not None}
        reason = _pending_full8(proposal)
        if missing or reason:
            pending.append(dict(proposal_id=proposal['proposal_id'], bank_dependencies=missing, full8_dependency=reason))
        else:
            results.append(_load_joint(proposal, frozen, inventory))
    if pending:
        return dict(status='pending', joint_set_identity=frozen['joint_set_identity'], pending=pending,
            winner_proposal_id=None, formal_eligible=False)
    require(len({row['full8_receipt']['ordered_sample_identity_evidence']['ordered_input_identity'] for row in results}) == 1
        and len({digest(row['selection_data_identity']) for row in results}) == 1,
        'Actual joint proposals used different ordered full8 inputs or fixed512/teacher/loss identities.')
    results.sort(key=lambda row: quality_key(row['full8_mean'], row['logical_bits'], row['packed_bits'],
        tuple((category, Fraction(rate)) for category, rate in row['rate_vector'])))
    winner = results[0]; hard = math.floor(Fraction(13, 5)*inventory['total_target_numel'])
    allocation = frozen['cap_allocation']
    caps = None
    if allocation is not None:
        require(sum(allocation['categories'].values()) <= hard
            and all(winner['ledger']['per_category'][category]['logical_bits'] <= value for category, value in allocation['categories'].items()),
            'Confirmed category allocation cannot contain the measured winner or exceeds complete logical2.6; no borrowing.')
        caps = {category: dict(anchor_bits=winner['ledger']['per_category'][category]['logical_bits'], cap_bits=value,
            reserve_bits=value-winner['ledger']['per_category'][category]['logical_bits'], N=winner['ledger']['per_category'][category]['N'])
            for category, value in allocation['categories'].items()}
    return _seal(dict(schema=VERSION, kind='actual_joint_full8_cap_certificate',
        status='complete' if caps else 'budget_allocation_pending', formal_eligible=caps is not None,
        joint_set_identity=frozen['joint_set_identity'], frozen_joint_set=frozen, protocol_identity=frozen['protocol_identity'],
        inventory_receipt=frozen['inventory_receipt'], baseline_receipt=frozen['baseline_receipt'],
        run_identity=frozen['run_identity'], N=inventory['total_target_numel'], hard_cap_bits=hard,
        source_proposal_id=winner['proposal_id'], winner_proposal_id=winner['proposal_id'],
        winner=winner, compared_proposals=results, categories=caps,
        allocation=allocation, sum_cap_bits=sum(row['cap_bits'] for row in caps.values()) if caps else None,
        borrowing_between_categories=False, joint_evidence_identity=digest(results),
        quality_rule='(-actual_joint_S8,actual_logical_bits,actual_packed_bits,stable_rate_vector)',
        scope='All explicitly frozen actual joint proposals only; profile8 and proposal/allocation generation remain unconfigured.'), 'certificate_identity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('freeze-block', 'select-block', 'freeze-joint', 'select-joint'), required=True)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); value = read(args.input)
    if args.component == 'freeze-block':
        result = freeze_block(**value)
    elif args.component == 'freeze-joint':
        result = freeze_joint(**value)
    else:
        result = select_block(value) if args.component == 'select-block' else select_joint(value)
    write(args.output, result)
    print({key: result[key] for key in ('status', 'block_identity', 'joint_set_identity', 'winner_candidate_id', 'winner_proposal_id') if key in result}, flush=True)


if __name__ == '__main__':
    main()

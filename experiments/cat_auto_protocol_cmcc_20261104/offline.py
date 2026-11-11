"""Deterministic bounded20-slot library; result-dependent slots require evidence."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import itertools
import math

from .adapter import category_projections
from .config import resolve_recipe
from .contracts import COVERAGE_STATUSES, digest, require, validate_recipe
from .geometry import DIMENSIONS, ROTATIONS, component_plan
from .identity import candidate_id


def geometry_key(recipe):
    return (digest(recipe['transpose']), digest(recipe['rotation']), recipe['D'])


def rank_joint(result):
    require(math.isfinite(result['joint_KL']) and result.get('joint_evidence_identity'), 'Offline ranking needs the actual fixed whole-model selection KL evidence.')
    return result['joint_KL'], result['logical_bits'], result['packed_bits'], result['candidate_id']


def full8_key(result):
    evidence = result['full8_evidence']
    require(evidence.get('complete') and math.isfinite(result['full8_mean']) and evidence.get('protocol_identity'), 'Only a complete, identity-bound full8 result is selectable.')
    return -result['full8_mean'], result['logical_bits'], result['packed_bits'], result['candidate_id']


def _levels(recipe):
    require(isinstance(recipe['transpose'], bool) and not recipe['rotation'].get('per_layer'), 'Core coverage uses uniform category geometry.')
    return frozenset((('T', str(recipe['transpose'])), ('R', recipe['rotation']['level']), ('D', str(recipe['D']))))


def _distance(left, right):
    return sum(a != b for a, b in zip(geometry_key(left), geometry_key(right), strict=True))


def _hash(identity, recipe):
    return digest(dict(seed=identity['search_seed'], category=recipe['category_id'], recipe=recipe))


def _candidate(inventory, identity, recipe, slot, role, cap, generation=None):
    plan = component_plan(inventory, recipe)
    status = plan['capability']['status']
    coverage = 'math_invalid' if status == 'math_invalid' else 'impl_blocked' if not plan['capability']['constructible'] else 'budget_deferred'
    feasible = plan['capability']['constructible'] and plan['logical_bits'] <= cap
    return dict(slot=slot, role=role, candidate_id=candidate_id(identity, recipe), recipe=recipe,
        generation=generation or {}, coverage_status=coverage, constructible=plan['capability']['constructible'],
        within_class_cap=bool(feasible), class_cap_bits=cap, fee_quote=plan,
        status='planned_requires_complete_fit' if feasible else 'class_cap_exceeded' if plan['capability']['constructible'] else status,
        formal_eligible=False, prediction_is_actual_fee=False)


def geometry_space(inventory, category, rate, identity, cap):
    rows = []
    for transpose, rotation, dim in itertools.product((False, True), ROTATIONS, DIMENSIONS):
        generation = dict(transpose=transpose, rotation=rotation, D=dim, requested_protection_fraction='1/64')
        try:
            recipe = resolve_recipe(inventory, category, rate, dim=dim, transpose=transpose, rotation_level=rotation)
            row = _candidate(inventory, identity, recipe, None, 'geometry_enumeration', cap, generation)
        except ValueError as error:
            text = str(error)
            row = dict(generation=generation, coverage_status='math_invalid' if text.startswith('math_invalid:') else 'impl_blocked',
                status='rejected', reason=text, constructible=False, within_class_cap=False, formal_eligible=False)
        rows.append(row)
    return rows


def balanced_first_seven(rows, anchor_id, identity):
    pool = {row['candidate_id']: row for row in rows if row.get('within_class_cap')}
    require(anchor_id in pool, 'The registered anchor is not constructible within its fixed class cap.')
    anchor = pool[anchor_id]
    ordered = sorted((row for key, row in pool.items() if key != anchor_id), key=lambda row: (_hash(identity, row['recipe']), row['candidate_id']))
    universe = sorted(set().union(*(_levels(row['recipe']) for row in pool.values())))
    bits = {level: 1 << index for index, level in enumerate(universe)}
    mask = lambda row: sum(bits[level] for level in _levels(row['recipe']))
    start = mask(anchor)
    # Exact finite set-cover DP, with anchor forced and lexicographic hash tie.
    best = {start: ()}
    for index, row in enumerate(ordered):
        for old_mask, chosen in list(best.items()):
            if len(chosen) >= 6:
                continue
            new_mask, proposal = old_mask | mask(row), chosen + (index,)
            old = best.get(new_mask)
            if old is None or (len(proposal), proposal) < (len(old), old):
                best[new_mask] = proposal
    target = (1 << len(universe)) - 1
    selected_mask = target if target in best else min(best, key=lambda value: (-value.bit_count(), len(best[value]), best[value]))
    chosen = [anchor] + [ordered[index] for index in best[selected_mask]]
    while len(chosen) < min(7, len(pool)):
        remaining = [row for row in pool.values() if row['candidate_id'] not in {old['candidate_id'] for old in chosen}]
        def fill_key(row):
            counts = Counter(level for old in [*chosen, row] for level in _levels(old['recipe']))
            return (-min(_distance(row['recipe'], old['recipe']) for old in chosen), sum(value * value for value in counts.values()),
                _hash(identity, row['recipe']), row['candidate_id'])
        chosen.append(min(remaining, key=fill_key))
    actual = set().union(*(_levels(row['recipe']) for row in chosen))
    return dict(slots=[dict(row, slot=index + 1, role='anchor' if index == 0 else 'balanced_joint_geometry') for index, row in enumerate(chosen)],
        complete_runnable_level_coverage=actual == set(universe), omitted_levels=[list(value) for value in universe if value not in actual],
        minimum_cover_size=1 + len(best[selected_mask]), exact_set_cover=True,
        no_single_factor_attribution=True)


def local_slots(first_seven_results, space):
    base = min(first_seven_results, key=rank_joint)
    recipe = base['recipe']
    require(isinstance(recipe['transpose'], bool), 'Uniform geometry expected before slot20.')
    available = {geometry_key(row['recipe']): row for row in space if row.get('within_class_cap')}
    used = {row['candidate_id'] for row in first_seven_results}
    requests = []
    changed = deepcopy(recipe); changed['transpose'] = not recipe['transpose']; requests.append(changed)
    di, ri = DIMENSIONS.index(recipe['D']), ROTATIONS.index(recipe['rotation']['level'])
    for distance in range(1, max(len(DIMENSIONS), len(ROTATIONS))):
        for name, index, levels in (('D', di, DIMENSIONS), ('R', ri, ROTATIONS)):
            for next_index in (index - distance, index + distance):
                if 0 <= next_index < len(levels):
                    requests.append(dict(recipe, D=levels[next_index]) if name == 'D' else dict(recipe, rotation=dict(recipe['rotation'], level=levels[next_index])))
    chosen = []
    for request in requests:
        key = (digest(request['transpose']), digest(request['rotation']), request['D'])
        # Rotation fields must be canonical, including kind/block, not level-only.
        if request['rotation']['level'] != recipe['rotation']['level']:
            from .geometry import rotation
            key = (digest(request['transpose']), digest(rotation(request['rotation']['level'])), request['D'])
        row = available.get(key)
        if row and row['candidate_id'] not in used:
            chosen.append(dict(row, slot=8 + len(chosen), role='predefined_local_geometry'))
            used.add(row['candidate_id'])
            if len(chosen) == 2:
                break
    return dict(base_candidate_id=base['candidate_id'], slots=chosen, empty_slots=list(range(8 + len(chosen), 10)))


def optimizer_slots(inventory, identity, first_nine, cap):
    retained = []
    for row in sorted(first_nine, key=rank_joint):
        if geometry_key(row['recipe']) not in {geometry_key(old['recipe']) for old in retained}:
            retained.append(row)
        if len(retained) == 2:
            break
    slots = []
    for row in retained:
        recipe = deepcopy(row['recipe'])
        recipe['optimizer']['kind'] = 'vae_muon'
        recipe['optimizer']['kappa_contract']['applicable'] = True
        slots.append(_candidate(inventory, identity, recipe, 10 + len(slots), 'other_optimizer_same_geometry', cap))
    return dict(retained_geometry_ids=[row['candidate_id'] for row in retained], slots=slots, empty_slots=list(range(10 + len(slots), 12)))


def multiplier_slots(inventory, identity, first_eleven, cap):
    base = min(first_eleven, key=rank_joint)
    slots = []
    for multiplier in (.25, .5, 2, 4):
        recipe = deepcopy(base['recipe']); recipe['optimizer']['multiplier'] = multiplier
        slots.append(_candidate(inventory, identity, recipe, 12 + len(slots), 'actual_multiplier_after_fixed_reference', cap))
    return dict(base_candidate_id=base['candidate_id'], slots=slots, reference_calibration_unchanged=True)


def bsq_condition(diagnostics):
    if not diagnostics:
        return None
    require([row['step'] for row in diagnostics] == [5000, 10000], 'Conditional BSQ uses exactly the two actual stage1 batches, not interpolated logs.')
    for name, field, fixed in (('commitment', 'commitment_loss_weight', .25), ('entropy', 'entropy_loss_weight', .01)):
        rows = [row['contributions'][name] for row in diagnostics]
        if all(row['ratio'] > 10 and row['cosine'] < 0 for row in rows):
            return dict(field=field, value=fixed * .5, reason='Both actual contribution gradients oppose reconstruction and exceed10x')
        if all(row['ratio'] < .1 for row in rows) and all(row['bit_marginal_extreme_fraction'] > .1 for row in diagnostics):
            return dict(field=field, value=fixed * 2, reason='Both contribution gradients below.1x and bit marginal extreme fraction above.1')
    return None


def _resolved_like(inventory, base, **changes):
    require(isinstance(base['transpose'], bool) and not base['rotation'].get('per_layer'), 'Conditional base before slot20 is uniform geometry.')
    options = dict(dim=base['D'], stages=base['S'], transpose=base['transpose'], rotation_level=base['rotation']['level'],
        encoder_width=base['encoder']['hidden_dim'], decoder_width=base['decoder']['hidden_dim'],
        optimizer=base['optimizer']['kind'], multiplier=base['optimizer']['multiplier'], protection_axis=base['protection']['axis'],
        protection_counts=base['protection']['counts'], loss_overrides={key: value for key, value in base['loss'].items() if key in ('commitment_loss_weight', 'entropy_loss_weight')})
    options.update(changes)
    return resolve_recipe(inventory, base['category_id'], base['nominal_main_rate'], **options)


def conditional_queue(inventory, base_result, identity, cap):
    base = base_result['recipe']
    queue = [dict(kind='S4'), dict(kind='encoder256'), dict(kind='decoder256'), dict(kind='protection_neighbor')]
    diagnostic = bsq_condition(base_result.get('BSQ_diagnostics'))
    if diagnostic:
        queue[3] = dict(kind='BSQ', **diagnostic)
    # Input/output eligibility is geometry, not an error-size preference.
    input_legal, output = False, []
    for fraction in ('0', '1/64', '1/32'):
        for axis in ('input', 'output'):
            try:
                candidate = _resolved_like(inventory, base, protection_counts=None, protection_fraction=fraction, protection_axis=axis)
                plan = component_plan(inventory, candidate)
                if plan['capability']['constructible']:
                    if axis == 'input':
                        input_legal = True
                    elif plan['logical_bits'] <= cap:
                        output.append((candidate, plan))
            except ValueError:
                pass
    if not input_legal and output:
        control = component_plan(inventory, base)
        protection_fee = lambda plan: sum(row['fees'][key] for row in plan['rows'] for key in ('protected_weights_bits', 'protected_indices_bits', 'protected_scales_bits'))
        winner, quote = min(output, key=lambda pair: (abs(protection_fee(pair[1]) - protection_fee(control)), protection_fee(pair[1]), tuple(pair[0]['protection']['counts'].values())))
        queue[2 if diagnostic else 3] = dict(kind='output_axis', resolved_recipe=winner,
            protection_fee_difference_bits=protection_fee(quote) - protection_fee(control))
    return dict(base_candidate_id=base_result['candidate_id'], frozen_diagnostics=digest(base_result.get('BSQ_diagnostics')),
        queue=queue, output_axis_fixed_uncovered=input_legal)


def conditional_candidate(inventory, current, instruction, identity, slot, cap):
    base = current['recipe']
    kind = instruction['kind']
    if kind == 'S4':
        recipe = _resolved_like(inventory, base, stages=4)
    elif kind in ('encoder256', 'decoder256'):
        recipe = _resolved_like(inventory, base, **{'encoder_width' if kind == 'encoder256' else 'decoder_width': 256})
    elif kind == 'BSQ':
        recipe = _resolved_like(inventory, base, loss_overrides={instruction['field']: instruction['value']})
    elif kind == 'output_axis':
        recipe = _resolved_like(inventory, base, protection_axis='output', protection_counts=instruction['resolved_recipe']['protection']['counts'])
    else:
        require(kind == 'protection_neighbor', 'Unregistered conditional variable.')
        unique = {}
        for fraction in ('0', '1/64', '1/32'):
            try:
                other = _resolved_like(inventory, base, protection_counts=None, protection_fraction=fraction)
                if other['protection']['counts'] != base['protection']['counts']:
                    plan = component_plan(inventory, other)
                    if plan['capability']['constructible'] and plan['logical_bits'] <= cap:
                        unique[digest(other['protection']['counts'])] = (other, fraction)
            except ValueError:
                pass
        if not unique:
            return dict(slot=slot, status='empty_no_registered_neighbor', coverage_status='v0_fixed', formal_eligible=False)
        # Adjacent actual count mappings first; then fixed hash among those.
        distance = lambda pair: sum(abs(pair[0]['protection']['counts'][name] - count) for name, count in base['protection']['counts'].items())
        nearest = min(distance(pair) for pair in unique.values())
        recipe, fraction = min((pair for pair in unique.values() if distance(pair) == nearest), key=lambda pair: _hash(identity, pair[0]))
        instruction = dict(instruction, requested_protection_fraction=fraction)
    return _candidate(inventory, identity, recipe, slot, kind, cap, instruction)


def accept_conditional(current, candidate):
    require(current['full8_evidence']['protocol_identity'] == candidate['full8_evidence']['protocol_identity'], 'Conditional full8 must use the same complete protocol.')
    full8_key(current); full8_key(candidate)
    improved = candidate['full8_mean'] > current['full8_mean'] or (candidate['full8_mean'] == current['full8_mean']
        and (candidate['logical_bits'], candidate['packed_bits']) < (current['logical_bits'], current['packed_bits']))
    return (candidate if improved else current), improved


def interaction_slot(inventory, identity, original_base, final_conditional, retained, accepted_changes, cap):
    if not accepted_changes:
        return dict(slot=20, status='empty_no_accepted_conditional', coverage_status='v0_fixed', formal_eligible=False)
    others = [row for row in sorted(retained, key=rank_joint) if geometry_key(row['recipe']) != geometry_key(original_base['recipe'])]
    if not others:
        return dict(slot=20, status='empty_no_other_preserved_geometry', coverage_status='v0_fixed', formal_eligible=False)
    geometry, final = others[0]['recipe'], final_conditional['recipe']
    try:
        # Apply the accepted bundle to the other geometry, rederive K/B/count
        # legality rather than copying decoder weights or an old dimensionalK.
        fraction = accepted_changes.get('requested_protection_fraction', '1/64') if isinstance(accepted_changes, dict) else '1/64'
        recipe = _resolved_like(inventory, geometry, stages=final['S'], encoder_width=final['encoder']['hidden_dim'],
            decoder_width=final['decoder']['hidden_dim'], optimizer=original_base['recipe']['optimizer']['kind'],
            multiplier=original_base['recipe']['optimizer']['multiplier'], protection_axis=final['protection']['axis'],
            protection_counts=None, protection_fraction=fraction, loss_overrides={key: value for key, value in final['loss'].items()
                if key in ('commitment_loss_weight', 'entropy_loss_weight')})
        return _candidate(inventory, identity, recipe, 20, 'accepted_bundle_other_preserved_geometry', cap, dict(accepted_changes=accepted_changes))
    except ValueError as error:
        return dict(slot=20, status='empty_conflict', reason=str(error), coverage_status='math_invalid', formal_eligible=False)


def _family_values(recipe):
    return [recipe['transpose'], recipe['rotation'], recipe['D'], recipe['S'], recipe['encoder'], recipe['decoder'],
        recipe['optimizer']['kind'], recipe['optimizer']['multiplier'], recipe['protection'], recipe['loss']]


def raw_full8_coverage(results, *, anchor_id, capacity_base_id, final_capacity_id, slot20_id, identity):
    by_id = {row['candidate_id']: row for row in results}
    selected = []
    for key in (anchor_id, capacity_base_id, final_capacity_id, slot20_id):
        if key is not None and key in by_id and key not in selected:
            selected.append(key)
    while len(selected) < min(6, len(by_id)):
        left = [key for key in by_id if key not in selected]
        def order(key):
            values = _family_values(by_id[key]['recipe'])
            distance = min(sum(a != b for a, b in zip(values, _family_values(by_id[old]['recipe']), strict=True)) for old in selected) if selected else len(values)
            return -distance, _hash(identity, by_id[key]['recipe']), key
        selected.append(min(left, key=order))
    already = {row['candidate_id'] for row in results if row.get('capacity_full8_evaluated')}
    require(len(already) <= 5 and capacity_base_id in already, 'Capacity full8 requires its base plus at most4 candidates.')
    union = already | set(selected)
    require(len(union) <= 10, 'Offline full8 union exceeds10; do not silently expand.')
    return dict(coverage_candidate_ids=selected, all_full8_candidate_ids=sorted(union), maximum_union=10,
        every_completed_full8_selectable=True)


def freeze_class_order(inventory, per_category, bf16_score):
    require(set(per_category) == set(inventory['category_ids']), 'Offline order needs all actual model categories.')
    rows = []
    for category, results in per_category.items():
        eligible = [row for row in results if row.get('full8_evidence')]
        require(eligible and len({row['full8_evidence']['protocol_identity'] for row in eligible}) == 1, 'Class full8 missing or protocol changed.')
        best = min(eligible, key=full8_key)
        rows.append(dict(category_id=category, selected_candidate_id=best['candidate_id'],
            d_c=bf16_score - best['full8_mean'], select_KL=best['joint_KL']))
    ordered = sorted(rows, key=lambda row: (row['d_c'], row['select_KL'], row['category_id']))
    return dict(ordered_categories=[row['category_id'] for row in ordered], evidence=ordered, frozen_order_identity=digest(ordered),
        score_scope='All classes relative to the same original BF16 full8 reference; no additive quality claim')


def layer_group_nominations(inventory, category_results):
    nominations = []
    for category, results in category_results.items():
        projections = category_projections(inventory, category)
        require(len(projections) >= 3, 'Three contiguous depth groups need at least3 actual layers.')
        q, remainder = divmod(len(projections), 3)
        sizes = [q + (group < remainder) for group in range(3)]
        boundaries = [0, sizes[0], sizes[0] + sizes[1], len(projections)]
        groups = [projections[boundaries[group]:boundaries[group + 1]] for group in range(3)]
        matched = {}
        for result in results:
            evidence = result.get('layer_functional_readout')
            if not evidence:
                continue
            require(evidence['measurement'] == 'actual_parent_input_BF16_functional_NMSE' and evidence['selection_halves'] == [256, 256]
                and evidence.get('parent_input_identity'), 'Layer-group triggers need actual same-parent proposal512 input halves.')
            fixed = deepcopy(result['recipe']); fixed.pop('transpose'); fixed.pop('rotation')
            key = digest(dict(recipe=fixed, input_identity=evidence['parent_input_identity'],
                run_identity=result['run_identity'], init_contract=result['init_contract']))
            matched.setdefault(key, []).append(result)
        candidates = []
        for values in matched.values():
            if len({geometry_key(row['recipe']) for row in values}) < 2:
                continue
            weights = [sum(row['numel'] for row in group) for group in groups]
            def error(result, group, half):
                halves = (0, 1) if half == 'full' else (half,)
                stats = result['layer_functional_readout']['sufficient_statistics']
                numer = sum(stats[row['actual_name']][index]['error_squared'] for row in groups[group] for index in halves)
                denom = sum(stats[row['actual_name']][index]['target_squared'] for row in groups[group] for index in halves)
                require(denom > 0 and math.isfinite(numer) and math.isfinite(denom), 'Layer-group NMSE needs finite positive-energy actual sufficient statistics.')
                return numer / denom
            winners = []
            for group in range(3):
                minima = [min(error(row, group, half) for row in values) for half in (0, 1)]
                common = [row for row in values if all(error(row, group, half) == minima[half] for half in (0, 1))]
                if not common:
                    break
                def group_cost(row):
                    fees = row['per_projection_fees']
                    return (sum(fees[item['actual_name']]['logical_bits'] for item in groups[group]),
                        sum(fees[item['actual_name']]['packed_bits'] for item in groups[group]))
                winners.append(min(common, key=lambda row: (error(row, group, 'full'), *group_cost(row), row['candidate_id'])))
            if len(winners) != 3 or len({geometry_key(row['recipe']) for row in winners}) < 2:
                continue
            uniform = min(values, key=lambda row: (sum(weights[group] * error(row, group, 'full') for group in range(3)) / sum(weights),
                row['logical_bits'], row['packed_bits'], row['candidate_id']))
            regrets = {str(half): sum(weights[group] / sum(weights) * max(0., error(uniform, group, half) - error(winners[group], group, half))
                for group in range(3)) for half in (0, 1, 'full')}
            if not (regrets['0'] > 0 and regrets['1'] > 0):
                continue
            score = sum(weights) / inventory['total_target_numel'] * regrets['full']
            candidates.append(dict(category_id=category, regret=regrets['full'], regrets=regrets, weighted_regret_score=score,
                uniform_candidate_id=uniform['candidate_id'], group_candidate_ids=[row['candidate_id'] for row in winners],
                groups=[[row['actual_name'] for row in group] for group in groups], matched_recipe_identity=digest(values[0]['recipe']),
                cohort_identity=digest(dict(fixed_recipe={key: value for key, value in values[0]['recipe'].items() if key not in ('transpose', 'rotation')},
                    candidates=sorted(row['candidate_id'] for row in values))),
                parent_input_identity=uniform['layer_functional_readout']['parent_input_identity']))
        if candidates:
            nominations.append(min(candidates, key=lambda row: (-row['weighted_regret_score'], row['cohort_identity'])))
    return sorted(nominations, key=lambda row: (-row['weighted_regret_score'], row['category_id']))[:2]


def layer_group_slot(inventory, nomination, results, identity, cap):
    by_id = {row['candidate_id']: row for row in results}
    uniform = by_id[nomination['uniform_candidate_id']]
    recipe = deepcopy(uniform['recipe'])
    transpose, rotations = {}, {}
    for names, key in zip(nomination['groups'], nomination['group_candidate_ids'], strict=True):
        source = by_id[key]['recipe']
        for name in names:
            transpose[name] = source['transpose'][name] if isinstance(source['transpose'], dict) else source['transpose']
            rotations[name] = source['rotation'].get('per_layer', {}).get(name, source['rotation']['level'])
    recipe['transpose'] = transpose
    recipe['rotation']['per_layer'] = rotations
    row = _candidate(inventory, identity, validate_recipe(recipe), 20, 'same_M_three_contiguous_depth_geometries', cap,
        dict(nomination=nomination, uniform_control_reused=uniform['candidate_id']))
    row['fresh_full_M_required'] = True
    return row


class SlotBook:
    """Distinct planned IDs own slots before fitting; fitted κ never renames them."""
    def __init__(self, category_id):
        self.category_id, self.slots, self.by_id = category_id, {}, {}

    def register(self, row):
        require(row['slot'] in range(1, 21) and row['slot'] not in self.slots, 'Slot is already fixed or outside1..20.')
        if row.get('candidate_id') in self.by_id:
            row = dict(row, coverage_status='matched_reuse', reused_slot=self.by_id[row['candidate_id']], status='reuse_no_new_fit')
        elif row.get('candidate_id') and row.get('within_class_cap'):
            self.by_id[row['candidate_id']] = row['slot']
        require(row['coverage_status'] in COVERAGE_STATUSES, 'Every slot needs one of the six protocol coverage states.')
        self.slots[row['slot']] = row
        return row

"""Actual model geometry plus explicitly mathematical search-rule fixtures."""
import argparse
from copy import deepcopy
from fractions import Fraction
import itertools
import json
from pathlib import Path
import time

import torch
from .adapter import file_sha
from .budget import build_task_graph, candidate_affordable, scenario
from .config import resolve_recipe
from .contracts import digest, inventory_id, require
from .geometry import component_plan
from .identity import candidate_id
from .offline import (SlotBook, balanced_first_seven, geometry_space, local_slots, optimizer_slots,
    multiplier_slots, conditional_queue, conditional_candidate, accept_conditional,
    layer_group_nominations, layer_group_slot, raw_full8_coverage)
from .proposals import category_caps, round_one, round_two, select_joint
from .validate_fit_cpu import tiny_fixture


def proposals_fixture():
    inventory = dict(model_identity=dict(scope='CPU_math_fixture'), adapter_id='math_rules_only', adapter_version=0,
        category_ids=['large', 'middle', 'small'], projections=[dict(actual_name=category, category_id=category,
            layer_id=0, shape=[size, 100], dtype='bfloat16', numel=size * 100) for category, size in
            (('large', 6), ('middle', 3), ('small', 1))], total_target_numel=1000)
    identity = dict(model=inventory['model_identity'], parent_uuid='CPU_fixture', parent_weights_digest='fixture_digest',
        data_identity='fixture_data', code_identity='fixture_code', consumer_identity='fixture_consumer', search_seed=31)
    profiles = []
    for category, size in (('large', 600), ('middle', 300), ('small', 100)):
        for rate in ('2', '5/2', '3'):
            per_category = {name: dict(N=amount, logical_bits=amount * 16, packed_bits=amount * 16) for name, amount in
                (('large', 600), ('middle', 300), ('small', 100))}
            bits = int(Fraction(rate) * size)
            per_category[category].update(logical_bits=bits, packed_bits=bits)
            ledger = dict(N=1000, logical_bits=sum(row['logical_bits'] for row in per_category.values()),
                packed_bits=sum(row['packed_bits'] for row in per_category.values()), tensor_bytes=2000,
                directory_bytes=None, per_category=per_category, future_dense_bits=(1000 - size) * 16,
                component_fees={}, assumptions=['Mathematical fee fixture only'], source=dict(kind='CPU_math_fixture'))
            profiles.append(dict(category_id=category, rate=rate, ledger=ledger, select_KL=3 - float(Fraction(rate)),
                bank_digest=category + rate, run_identity=identity, selection_data_identity='fixed_fixture_select',
                formal_eligible=False, phase='cpu_fixture'))
    first = round_one(inventory, profiles, 31)
    assert first['working_set_used'] and all(row['logical_bits'] <= 2500 for row in first['proposals'])
    def measured(row, value):
        return dict(proposal_id=row['proposal_id'], whole_model_installation_verified=True,
            measurement='CPU_math_fixture', joint_KL=value, bank_digests=row['bank_digests'],
            selection_data_identity=row['selection_data_identity'], formal_eligible=False, phase='cpu_fixture')
    balance = next(row for row in first['proposals'] if 'A_balance' in row['nominators'])
    winner = select_joint(first['proposals'], [measured(row, 0. if row['proposal_id'] == balance['proposal_id'] else 1.) for row in first['proposals']])
    second = round_two(inventory, profiles, winner, {row['proposal_id'] for row in first['proposals']})
    assert any(2500 < row['logical_bits'] <= 2600 for row in second['proposals'])
    all_rows = [winner, *second['proposals']]
    final = select_joint(all_rows, [measured(row, 0. if row['proposal_id'] == winner['proposal_id'] else 2.) for row in all_rows])
    assert final['proposal_id'] == winner['proposal_id'] and not final['formal_eligible']
    caps = category_caps(inventory, final)
    assert caps['sum_cap_bits'] == 2600 and not caps['formal_eligible']
    return dict(scope='CPU_math_fixture_only', R1_prefers_working_2_5=True, R2_allows_above2_5_under2_6=True,
        original_winner_retained=True, fixture_formal_false_through_caps=True, caps=caps)


def layer_fixture():
    _, inventory, identity, _ = tiny_fixture(5)
    category = 'up_proj'
    results = []
    for index, transpose in enumerate((False, True)):
        recipe = resolve_recipe(inventory, category, '2.5', dim=8, rotation_level='none', protection_fraction='0', transpose=transpose)
        plan = component_plan(inventory, recipe)
        sufficient = {}
        for layer, name in enumerate(recipe['group_signature']):
            group = 0 if layer < 2 else 1 if layer < 4 else 2
            preferred = 1 if group == 1 else 0
            error = .1 if index == preferred else .3
            sufficient[name] = [dict(error_squared=error * (layer + 1) * 100, target_squared=(layer + 1) * 100,
                records=256, valid_tokens=512) for _ in range(2)]
        results.append(dict(candidate_id=candidate_id(identity, recipe), recipe=recipe,
            joint_KL=.2 if index else .3, joint_evidence_identity='CPU_math_fixture', logical_bits=plan['logical_bits'],
            packed_bits=plan['packed_bits'], per_projection_fees={row['name']: row for row in plan['rows']},
            run_identity=identity, init_contract='semantic_stage_seed_contract',
            layer_functional_readout=dict(measurement='actual_parent_input_BF16_functional_NMSE', selection_halves=[256, 256],
                parent_input_identity='explicit_CPU_math_fixture_not_actual_readout', sufficient_statistics=sufficient)))
    nominations = layer_group_nominations(inventory, {category: results})
    assert len(nominations) == 1 and [len(group) for group in nominations[0]['groups']] == [2, 2, 1]
    selected = layer_group_slot(inventory, nominations[0], results, identity, 10 ** 12)
    assert selected['recipe']['linear_group_size'] == 5 and selected['recipe']['transpose'] == {
        name: index in (2, 3) for index, name in enumerate(selected['recipe']['group_signature'])}
    # Opposite winners in the halves remove the nomination rather than averaging.
    changed = deepcopy(results)
    name = changed[0]['recipe']['group_signature'][0]
    changed[0]['layer_functional_readout']['sufficient_statistics'][name][1]['error_squared'] = 1000
    assert not layer_group_nominations(inventory, {category: changed})
    return dict(scope='CPU_math_fixture_only', depth_partition=[2, 2, 1], nomination=nominations,
        same_M_heterogeneous_recipe_constructible=selected['constructible'], disagreement_rejected=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inventory', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); began = time.monotonic(); torch.set_num_threads(2)
    inventory = json.loads(Path(args.inventory).read_text())['inventory']
    identity = dict(model=inventory['model_identity'], parent_uuid='CPU_plan_not_fitted_BF16', parent_weights_digest=digest(inventory['model_identity']['weight_shards_sha256']),
        data_identity='CPU_geometry_no_data_execution', code_identity='CPU_geometry_fixture', consumer_identity='PUBLIC_v6_CPU_component_quote', search_seed=31)
    category = 'up_proj'; cap = 16 * sum(row['numel'] for row in inventory['projections'] if row['category_id'] == category)
    space = geometry_space(inventory, category, '2.5', identity, cap)
    anchor = resolve_recipe(inventory, category, '2.5')
    first = balanced_first_seven(space, candidate_id(identity, anchor), identity)
    assert len(first['slots']) == 7 and first['complete_runnable_level_coverage']
    reverse = balanced_first_seven(list(reversed(space)), candidate_id(identity, anchor), identity)
    assert [row['candidate_id'] for row in first['slots']] == [row['candidate_id'] for row in reverse['slots']]
    book = SlotBook(category)
    for row in first['slots']:
        book.register(row)
    results = [dict(row, joint_KL=float(index), joint_evidence_identity='CPU_math_fixture', logical_bits=row['fee_quote']['logical_bits'],
        packed_bits=row['fee_quote']['packed_bits']) for index, row in enumerate(first['slots'])]
    local = local_slots(results, space)
    for row in local['slots']:
        book.register(row)
        results.append(dict(row, joint_KL=8., joint_evidence_identity='CPU_math_fixture', logical_bits=row['fee_quote']['logical_bits'], packed_bits=row['fee_quote']['packed_bits']))
    optim = optimizer_slots(inventory, identity, results, cap)
    mult = multiplier_slots(inventory, identity, results, cap)
    for row in optim['slots'] + mult['slots']:
        book.register(row)
    base = dict(results[0], full8_mean=.6, full8_evidence=dict(complete=True, protocol_identity='CPU_math_fixture'), capacity_full8_evaluated=True)
    frozen_queue = conditional_queue(inventory, base, identity, cap)
    current = base
    for slot, instruction in enumerate(frozen_queue['queue'], 16):
        row = conditional_candidate(inventory, current, instruction, identity, slot, cap)
        book.register(row)
        if row.get('within_class_cap'):
            measured = dict(row, full8_mean=.59, full8_evidence=base['full8_evidence'], logical_bits=row['fee_quote']['logical_bits'], packed_bits=row['fee_quote']['packed_bits'])
            current, accepted = accept_conditional(current, measured)
            assert not accepted
    cover = raw_full8_coverage([base, *results[1:]], anchor_id=base['candidate_id'], capacity_base_id=base['candidate_id'],
        final_capacity_id=base['candidate_id'], slot20_id=None, identity=identity)
    quote = build_task_graph(inventory)
    assert quote['limits']['profile_fits'] == 21 and quote['limits']['offline_fits'] == 140 and quote['limits']['live_fits'] == 14
    assert quote['limits']['recovery_trajectories'] == 23 and quote['limits']['recovery_node8'] == 100 and quote['total_upper_gpu_hours'] is None
    affordability = candidate_affordable(quote, ['live:up_proj:0'], remaining_budget=16,
        protected_reservation_ids=['search:nine', 'replay:nine'])
    assert not affordability['affordable'] and affordability['unknown_nodes']
    illustrative = scenario(quote, {node: 1. for node in quote['unknown_nodes']}, label='Every unknown task hypothetically1GPUh; explicitly not measurement')
    assert illustrative['total_upper_gpu_hours'] > 175 and not illustrative['GPU_execution_authorized']
    moved = deepcopy(inventory); moved['source_evidence']['path'] = '/different_machine/content_identical'
    assert inventory_id(moved) == inventory_id(inventory)
    report = dict(exit_code=0, cuda_initialized=torch.cuda.is_initialized(), elapsed_seconds=time.monotonic() - began,
        formal_eligible=False, model_inventory_identity=inventory_id(inventory), actual_Qwen_geometry_enumeration=space,
        core_first_seven=first, local_slots=local, optimizer_slots=optim, multiplier_slots=mult, conditional_queue=frozen_queue,
        raw8_coverage=cover, proposals_math=proposals_fixture(), layer_group_math=layer_fixture(), budget_DAG=quote,
        unknown_budget_refuses_execution=affordability, illustrative_scenario_total_gpu_hours=illustrative['total_upper_gpu_hours'],
        source_sha256={str(path): file_sha(path) for path in Path(__file__).parent.glob('*.py')},
        scope='Real Qwen inventory/component geometry and pure CPU rule fixtures; no VAE training/full-model GPU/quality claim')
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(dict(exit_code=0, elapsed_seconds=report['elapsed_seconds'], cuda_initialized=report['cuda_initialized']), flush=True)


if __name__ == '__main__':
    main()

"""Bounded category-budget proposals; additive KL is never a quality evaluator."""
from fractions import Fraction
import itertools
import math

from .adapter import category_projections
from .contracts import digest, require, validate_inventory, validate_ledger, validate_run_identity

RATES = (Fraction(2), Fraction(5, 2), Fraction(3))


def _profile_map(inventory, profiles):
    inventory = validate_inventory(inventory)
    categories = sorted(inventory['category_ids'])
    options = {category: {} for category in categories}
    identities, selections = set(), set()
    for row in profiles:
        category, rate = row['category_id'], Fraction(str(row['rate']))
        require(category in options and rate in RATES and rate not in options[category], 'Duplicate or unregistered category rate profile.')
        ledger = validate_ledger(row['ledger'])
        require(ledger['N'] == inventory['total_target_numel'] and set(ledger['per_category']) == set(categories),
            'Profile must carry the full-inventory actual partial ledger, never nominal-only or active-onlyN.')
        require(ledger['source']['kind'] in ('actual_native_v6_tensor_storage', 'CPU_math_fixture'), 'Forecasts cannot masquerade as measured profile-bank fees.')
        require(math.isfinite(row['select_KL']) and row['bank_digest'], 'Require an actual profile result and bank identity.')
        identities.add(digest(validate_run_identity(row['run_identity'])))
        require(row.get('selection_data_identity'), 'The profile-select stream must have its own actual identity.')
        selections.add(digest(row['selection_data_identity']))
        fee = ledger['per_category'][category]
        expected_n = sum(item['numel'] for item in category_projections(inventory, category))
        require(fee['N'] == expected_n and isinstance(fee['logical_bits'], int) and isinstance(fee['packed_bits'], int), 'Profile category costs must come from real complete component ownership.')
        options[category][rate] = dict(row, rate=str(rate), logical_bits=fee['logical_bits'], packed_bits=fee['packed_bits'])
    require(len(identities) == len(selections) == 1 and all(options.values()), 'All category profiles need one common BF16/data/code/consumer/select identity and explicit coverage.')
    return categories, options


def _vector(categories, chosen):
    rates = [[category, row['rate']] for category, row in zip(categories, chosen, strict=True)]
    banks = {category: row['bank_digest'] for category, row in zip(categories, chosen, strict=True)}
    return dict(proposal_id=digest(dict(rates=rates, banks=banks)), rate_vector=rates, bank_digests=banks,
        logical_bits=sum(row['logical_bits'] for row in chosen), packed_bits=sum(row['packed_bits'] for row in chosen),
        additive_profile_KL=sum(row['select_KL'] for row in chosen),
        class_fees={category: dict(logical_bits=row['logical_bits'], packed_bits=row['packed_bits'])
            for category, row in zip(categories, chosen, strict=True)},
        formal_eligible=all(row.get('formal_eligible', True) and row.get('phase') != 'cpu_fixture' and row['ledger']['source']['kind'] != 'CPU_math_fixture' for row in chosen),
        selection_data_identity=chosen[0]['selection_data_identity'], run_identity_digest=digest(chosen[0]['run_identity']),
        additive_score_scope='Proposal ordering only; must install all banks and measure actual whole-model selectKL')


def stable_vector(row):
    return tuple((category, Fraction(rate)) for category, rate in row['rate_vector'])


def enumerate_vectors(inventory, profiles):
    categories, options = _profile_map(inventory, profiles)
    hard = math.floor(Fraction(13, 5) * inventory['total_target_numel'])
    vectors = [_vector(categories, chosen) for chosen in itertools.product(*(list(options[category].values()) for category in categories))]
    return [dict(row, actual_logical=row['logical_bits'] / inventory['total_target_numel'],
        working_target_deviation=row['logical_bits'] / inventory['total_target_numel'] - 2.5,
        hard_headroom_bits=hard - row['logical_bits']) for row in vectors if row['logical_bits'] <= hard]


def round_one(inventory, profiles, search_seed):
    require(all(row['run_identity']['search_seed'] == search_seed for row in profiles), 'Proposal hash seed must be the fixed search identity seed.')
    candidates = enumerate_vectors(inventory, profiles)
    n = inventory['total_target_numel']
    working, hard = Fraction(5, 2) * n, math.floor(Fraction(13, 5) * n)
    if not candidates:
        return dict(status='infeasible_hard_cap', proposals=[], hard_cap_bits=hard, no_automatic_limit_increase=True)
    pool = [row for row in candidates if row['logical_bits'] <= working]
    use_working = bool(pool)
    pool = pool or candidates
    sizes = {category: sum(row['numel'] for row in category_projections(inventory, category)) for category in inventory['category_ids']}

    def balance(row):
        return sum(Fraction(sizes[category], n) * (Fraction(rate) - Fraction(5, 2)) ** 2 for category, rate in row['rate_vector'])

    chosen = []

    def add(row, role):
        found = next((item for item in chosen if item['proposal_id'] == row['proposal_id']), None)
        if found is None:
            chosen.append(dict(row, nominators=[role]))
        else:
            found['nominators'].append(role)

    a = min(pool, key=lambda row: (balance(row), abs(Fraction(row['logical_bits']) - working), stable_vector(row)))
    b = min(pool, key=lambda row: (row['additive_profile_KL'], stable_vector(row)))
    add(a, 'A_balance')
    add(b, 'B_additive_proposal_only')
    rest = [row for row in pool if row['proposal_id'] not in {item['proposal_id'] for item in chosen}]
    if rest:
        def distance(row):
            return min(sum(left[1] != right[1] for left, right in zip(row['rate_vector'], old['rate_vector'], strict=True)) for old in chosen)
        add(min(rest, key=lambda row: (-distance(row), stable_vector(row))), 'C_max_min_category_Hamming')
    rest = [row for row in pool if row['proposal_id'] not in {item['proposal_id'] for item in chosen}]
    if rest:
        add(min(rest, key=lambda row: (digest(dict(seed=search_seed, vector=row['rate_vector'])), stable_vector(row))), 'D_fixed_hash')
    return dict(status='proposals_require_actual_joint_readout', proposals=chosen, hard_cap_bits=hard,
        work_target_bits=str(working), working_set_used=use_working, deviation_from_work_target=not use_working,
        hard_feasible_count=len(candidates), selection_set_count=len(pool), maximum_proposals=4)


def select_joint(proposals, measurements):
    by_id = {row['proposal_id']: row for row in proposals}
    require(len(measurements) == len(by_id) and len({row['proposal_id'] for row in measurements}) == len(by_id), 'Every proposed full bank installation must receive one own joint readout.')
    rows = []
    for measured in measurements:
        require(measured['proposal_id'] in by_id and measured['whole_model_installation_verified']
            and measured['measurement'] in ('actual_joint_model_select_KL', 'CPU_math_fixture')
            and math.isfinite(measured['joint_KL']), 'Additive per-class scores cannot substitute for whole-model joint KL.')
        proposed = by_id[measured['proposal_id']]
        require(measured['bank_digests'] == proposed['bank_digests'] and measured['selection_data_identity'] == proposed['selection_data_identity'], 'Joint readout used different banks or a different selection data stream.')
        rows.append(dict(proposed, joint_KL=measured['joint_KL'], joint_evidence=measured,
            formal_eligible=proposed['formal_eligible'] and measured.get('formal_eligible', True)
                and measured.get('phase') != 'cpu_fixture' and measured['measurement'] == 'actual_joint_model_select_KL'))
    require(len({digest(row['joint_evidence']['selection_data_identity']) for row in rows}) == 1, 'All joint proposals must use exactly the same proposal-select stream.')
    return min(rows, key=lambda row: (row['joint_KL'], row['logical_bits'], row['packed_bits'], stable_vector(row)))


def round_two(inventory, profiles, winner, measured_proposal_ids):
    categories, options = _profile_map(inventory, profiles)
    base = {category: Fraction(rate) for category, rate in winner['rate_vector']}
    require(set(base) == set(categories), 'Winner rate vector omits a category.')
    all_valid = {tuple(row['rate_vector'][index][1] for index in range(len(categories))): row for row in enumerate_vectors(inventory, profiles)}
    neighbours = {}
    for up, down in itertools.permutations(categories, 2):
        ui, di = RATES.index(base[up]), RATES.index(base[down])
        if ui == len(RATES) - 1 or di == 0:
            continue
        changed = dict(base)
        changed[up], changed[down] = RATES[ui + 1], RATES[di - 1]
        key = tuple(str(changed[category]) for category in categories)
        if key in all_valid:
            row = all_valid[key]
            if row['proposal_id'] not in measured_proposal_ids:
                neighbours[row['proposal_id']] = dict(row, upgraded_category=up, downgraded_category=down)
    target = Fraction(5, 2) * inventory['total_target_numel']
    ordered = sorted(neighbours.values(), key=lambda row: (row['additive_profile_KL'], abs(Fraction(row['logical_bits']) - target), stable_vector(row)))
    return dict(proposals=ordered[:4], maximum_new_joint_readouts=4, include_original_winner_in_final_comparison=True,
        not_equal_bits_exchange=True, no_seed_resampling=True)


def category_caps(inventory, winner):
    categories = sorted(inventory['category_ids'])
    n = inventory['total_target_numel']
    hard = math.floor(Fraction(13, 5) * n)
    require(set(winner['class_fees']) == set(categories), 'Winner needs actual complete per-category fees.')
    anchor = {category: winner['class_fees'][category]['logical_bits'] for category in categories}
    remaining = hard - sum(anchor.values())
    require(remaining >= 0, 'No feasible hard-cap allocation; do not raise2.6.')
    sizes = {category: sum(row['numel'] for row in category_projections(inventory, category)) for category in categories}
    reserve = {category: remaining * sizes[category] // n for category in categories}
    remainder = remaining - sum(reserve.values())
    for category in categories[:remainder]:
        reserve[category] += 1
    caps = {category: dict(anchor_bits=anchor[category], reserve_bits=reserve[category], cap_bits=anchor[category] + reserve[category], N=sizes[category]) for category in categories}
    require(sum(row['cap_bits'] for row in caps.values()) <= hard, 'Category reserves exceed the fixed complete hard cap.')
    return dict(N=n, hard_cap_bits=hard, categories=caps, sum_cap_bits=sum(row['cap_bits'] for row in caps.values()),
        allocation='Parameter-weighted floor reserve; integer remainder assigned by stable semantic category ID', borrowing_between_categories=False,
        source_proposal_id=winner['proposal_id'], joint_evidence_identity=digest(winner.get('joint_evidence')),
        formal_eligible=bool(winner.get('formal_eligible')) and winner.get('joint_evidence', {}).get('measurement') == 'actual_joint_model_select_KL')

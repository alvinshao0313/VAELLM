"""Full-model GPU-hour task graph; unknown measurements remain unknown."""
from copy import deepcopy
import math
from .contracts import digest, require


def _cost(prices, key, *, recipe_id=None):
    quote = prices.get(key)
    if quote is None:
        return None, dict(key=key, status='unknown_unmeasured', recipe_id=recipe_id)
    require(quote.get('basis') and quote.get('source') and quote['upper_gpu_hours'] >= 0
        and math.isfinite(quote['upper_gpu_hours']), 'A GPU-hour bound needs actual source and explicit basis.')
    require(recipe_id is None or quote.get('recipe_id') == recipe_id, 'Do not apply an anchor fit quote to S4/width/another resolved candidate.')
    return quote['upper_gpu_hours'], dict(key=key, **quote)


def build_task_graph(inventory, *, prices=None, offline_recipes=None, profile_recipes=None,
        selected_recipes=None, reference_identity=None, replay_identity=None):
    prices, nodes = prices or {}, []
    categories = list(inventory['category_ids'])
    offline_recipes, profile_recipes = offline_recipes or {}, profile_recipes or {}
    selected_recipes = selected_recipes or {}

    def add(identifier, kind, category=None, dependencies=(), recipe=None, phase=None, key=None, cost=None, basis=None):
        if basis is None:
            cost, basis = _cost(prices, key or identifier, recipe_id=None if recipe is None else digest(recipe))
        nodes.append(dict(id=identifier, kind=kind, parent_dependencies=list(dependencies),
            candidate_id=None if recipe is None else digest(dict(category=category, phase=phase, recipe=recipe)), category=category,
            phase=phase or kind, cost_upper_gpu_hours=cost, cost_basis=basis, reuse_key=digest(dict(model=inventory['model_identity'],
                category=category, recipe=recipe, phase=phase or kind, reference=reference_identity if phase == 'reference' else replay_identity if phase == 'replay' else None)),
            status='planned_CPU_only', conditional=kind in ('conditional_long_increment', 'interaction500', 'optional_node8')))
        return identifier

    profiles = []
    for category in categories:
        for index in range(3):
            recipe = profile_recipes.get(category, [None] * 3)[index]
            profiles.append(add(f'profile:{category}:{index}', 'profile_fit', category, recipe=recipe, phase='profile',
                key=f'F_off:{category}:{digest(recipe) if recipe is not None else "profile_" + str(index)}'))
    joint = [add(f'joint_select:{index}', 'joint_select', dependencies=profiles, key='joint_select') for index in range(8)]
    offline = {}
    for category in categories:
        fits = []
        recipes = offline_recipes.get(category, [])
        require(len(recipes) <= 20, 'At most20 distinct offline fits; profiles are separately charged.')
        for index in range(20):
            recipe = recipes[index] if index < len(recipes) else None
            fits.append(add(f'offline:{category}:{index + 1}', 'offline_fit', category, joint, recipe, 'offline',
                key=f'F_off:{category}:{digest(recipe) if recipe is not None else "unresolved_slot_" + str(index + 1)}'))
        offline[category] = [add(f'offline8:{category}:{index}', 'offline_full8', category, fits, phase='offline8', key=f'E_off:{category}') for index in range(10)]
    order = add('freeze_class_order', 'CPU_order', dependencies=[node for values in offline.values() for node in values], cost=0.,
        basis=dict(status='CPU_only_exact_zero_GPU', source='Deterministic ordering of completed actual scores'))
    previous = order
    for category in categories:
        live = []
        for index in range(2):
            recipe = selected_recipes.get(category, [None, None])[index]
            fit = add(f'live:{category}:{index}', 'live_fit', category, (previous,), recipe, 'live',
                key=f'F_live:{category}:{digest(recipe) if recipe is not None else "unresolved_raw_" + str(index)}')
            live.append(add(f'live_raw8:{category}:{index}', 'live_raw_full8', category, (fit,), phase='live_raw8', key=f'E_live:{category}'))
        recovery = [add(f'recovery500:{category}:{index}', 'recovery500', category, live, phase='recovery', key=f'R500:{category}:{index}') for index in range(3)]
        node8 = [add(f'recovery_node8:{category}:{index}', 'node_full8', category, recovery, phase='node8', key=f'E_live:{category}') for index in range(12)]
        previous = add(f'boundary:{category}', 'CPU_boundary_selection', category, node8, cost=0.,
            basis=dict(status='CPU_only_exact_zero_GPU', source='Only one selected complete mutable node continues'))
    # Shared optional quotas: interaction2×500; long2 differences; node8 extra16.
    def maximum(family):
        values = []
        for category in categories:
            value, basis = _cost(prices, family + ':' + category)
            if value is None:
                return None, dict(status='unknown_unmeasured_model_maximum', family=family, categories=categories)
            values.append((value, category, basis))
        return max(value[0] for value in values), dict(status='max_actual_class_upper_bound', family=family, class_bounds=values)
    for index in range(2):
        value, basis = maximum('R500_interaction')
        add(f'interaction500:{index}', 'interaction500', dependencies=(order,), cost=value, basis=basis)
    differences = []
    for category in categories:
        small, _ = _cost(prices, 'R500_long_reference:' + category)
        large, _ = _cost(prices, 'R1000:' + category)
        differences.append(None if small is None or large is None else max(0., large - small))
    for index in range(2):
        known = all(value is not None for value in differences)
        value = sorted(differences, reverse=True)[index] if known and index < len(categories) else 0. if known else None
        add(f'long_increment:{index}', 'conditional_long_increment', dependencies=(order,), cost=value,
            basis=dict(status='two_largest_class_differences' if known else 'unknown_unmeasured_R1000_minus_R500',
                class_differences=dict(zip(categories, differences, strict=True)), replaces_third500=True))
    for index in range(16):
        value, basis = maximum('E_live')
        add(f'optional_node8:{index}', 'optional_node8', dependencies=(order,), cost=value, basis=basis)
    for phase in ('reference', 'replay'):
        parent = 'source_BF16'
        for category in categories:
            chain = []
            for family in ('F', 'R', 'E'):
                chain.append(add(f'{phase}:{family}:{category}', phase + '_' + family, category,
                    () if parent == 'source_BF16' else (parent,), selected_recipes.get(category, [None])[0] if phase == 'replay' else None,
                    phase, key=f'{family}_{phase}:{category}'))
                parent = chain[-1]
        add(phase + ':nine', 'final_nine', dependencies=(parent,), phase=phase, key='J:' + phase)
    add('search:nine', 'final_nine', dependencies=(previous,), phase='search', key='J:search')
    add('BF16:reference', 'BF16_reference', key='BF16_reference')
    add('shared_GPU_overhead', 'shared_GPU_overhead', key='shared_GPU_overhead')
    ids = {row['id'] for row in nodes}
    require(len(ids) == len(nodes) and all(set(row['parent_dependencies']) <= ids for row in nodes), 'Malformed task dependency graph.')
    unknown = [row['id'] for row in nodes if row['cost_upper_gpu_hours'] is None]
    return dict(schema='cat_auto_complete_budget_DAG_v0', model_identity=inventory['model_identity'], nodes=nodes,
        known_upper_gpu_hours=sum(row['cost_upper_gpu_hours'] or 0 for row in nodes), unknown_nodes=unknown,
        total_upper_gpu_hours=None if unknown else sum(row['cost_upper_gpu_hours'] for row in nodes),
        model_budget_confirmed=False, GPU_execution_authorized=False,
        limits=dict(C=len(categories), profile_fits=3 * len(categories), offline_fits=20 * len(categories), live_fits=2 * len(categories),
            offline8=10 * len(categories), live_raw8=2 * len(categories), recovery_trajectories=3 * len(categories) + 2,
            conditional_long_replacements=2, recovery_node8=12 * len(categories) + 16, final_nine_minimum=3),
        accounting_notes=['Unknown per-class/per-structure costs are not zero', 'All F includes encode/export/fresh/proxy; R includes actual saves/exports',
            'Replay R preserves original schedule_max and stops at selected_step; requires its own actual quote',
            'CPU overhead separately reported, no CPU fixture credited against GPU work', 'No anchor quote silently applied to S4 or wider candidates'])


def scenario(graph, replacement_bounds, *, label):
    require(label and not graph.get('GPU_execution_authorized'), 'Scenarios are labeled hypothetical CPU calculations only.')
    result = deepcopy(graph)
    for node in result['nodes']:
        if node['cost_upper_gpu_hours'] is None and node['id'] in replacement_bounds:
            value = replacement_bounds[node['id']]
            require(isinstance(value, (int, float)) and value > 0 and math.isfinite(value), 'An unmeasured scenario must use explicit positive hypotheses, not zero.')
            node.update(cost_upper_gpu_hours=value, cost_basis=dict(status='hypothetical_scenario_not_measurement', scenario_label=label))
    result['scenario_label'] = label
    result['unknown_nodes'] = [row['id'] for row in result['nodes'] if row['cost_upper_gpu_hours'] is None]
    result['total_upper_gpu_hours'] = None if result['unknown_nodes'] else sum(row['cost_upper_gpu_hours'] for row in result['nodes'])
    result['not_an_execution_budget'] = True
    return result


def candidate_affordable(graph, candidate_node_ids, *, remaining_budget, protected_reservation_ids):
    by_id = {node['id']: node for node in graph['nodes']}
    needed = set(candidate_node_ids) | set(protected_reservation_ids)
    require(needed <= set(by_id), 'Reservation omitted an actual dependency node.')
    stack = list(needed)
    while stack:
        for dependency in by_id[stack.pop()]['parent_dependencies']:
            if dependency not in needed and by_id[dependency]['status'] != 'completed':
                needed.add(dependency); stack.append(dependency)
    unknown = sorted(key for key in needed if by_id[key]['cost_upper_gpu_hours'] is None)
    amount = None if unknown else sum(by_id[key]['cost_upper_gpu_hours'] for key in needed if by_id[key]['status'] != 'completed')
    return dict(affordable=not unknown and amount <= remaining_budget, dependency_upper_gpu_hours=amount,
        unknown_nodes=unknown, reserved_node_ids=sorted(needed), reason='unknown_or_insufficient_budget' if unknown or amount > remaining_budget else 'complete_dependency_budget_fits')

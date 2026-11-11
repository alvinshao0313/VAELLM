"""Persistent per-round decisions; only the default winner receives further HPO."""
from dataclasses import asdict

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.sc_cat_live_cmcc_20261106.bindings import write
from .selection import compare, calibration, screen
from .settings import ROUNDS, CALIBRATED_ROUNDS, round_candidates, trajectory_id


def new_state(handoff, run_id):
    return dict(schema='sc_cat_recovery_search_v1', run_id=run_id, status='awaiting_first_class_budget',
        frozen_order=handoff['frozen_order'], current_category=handoff['frozen_order'][0],
        root_parent=handoff['parent'], categories={}, short_screen={name:dict(status='uncalibrated') for name in CALIBRATED_ROUNDS},
        long_categories=[], cumulative_GPUh=0., approved_categories=[], boundary=None,
        final_nine_complete=False, actual_bits_accepted=False, BF16_replay_complete=False)


def begin_round(state, *, category, parameter, base, raw_id, inventory, data_identity, code_identity, completed):
    require(category == state['current_category'] and category in state['approved_categories'], 'This class has no approved budget or true boundary.')
    block = state['categories'][category]
    require(block.get('default_winner_raw') == raw_id, 'Only the fully compared default winner receives HPO.')
    index = len(block.setdefault('rounds', []))
    require(index < len(ROUNDS) and parameter == ROUNDS[index] and all(r['status']=='complete' for r in block['rounds']),
        'Do not generate a later parameter before the previous full comparison.')
    candidates = round_candidates(parameter, base, **inventory)
    ids = [trajectory_id(raw_id, s, data_identity, code_identity) for s in candidates]
    existing = [identifier for identifier in ids if identifier in completed]
    require(all(completed[i].status == 'complete' and completed[i].raw_id == raw_id for i in existing), 'Reuse only matching complete controls.')
    calibration_state = state['short_screen'].get(parameter, dict(status='not_applicable'))
    first_calibration = len(ids)>=3 and calibration_state['status']=='uncalibrated'
    do_screen = len(ids)>=3 and calibration_state['status']=='passed'
    row = dict(parameter=parameter, status='pending', raw_id=raw_id, base=asdict(base),
        candidates=[dict(trajectory_id=i, settings=asdict(s), reuse=i in existing,
            first_stop=250 if do_screen and i not in existing else 500) for i,s in zip(ids,candidates)],
        existing_complete_ids=existing, first_calibration=first_calibration, short_screen=do_screen,
        continuation_ids=None, winner=None)
    block['rounds'].append(row)
    return row


def finish_round(state, category, results):
    row = state['categories'][category]['rounds'][-1]
    require(row['status']=='pending', 'A round decision is immutable once complete.')
    ids = [r['trajectory_id'] for r in row['candidates']]
    require(set(results)==set(ids), 'Wait for every announced legal candidate; failures are not zero scores.')
    if not ids:
        row.update(status='complete', disposition='inactive', winner=None)
        return None
    values = [results[i] for i in ids]
    if row['short_screen'] and row['continuation_ids'] is None:
        row['continuation_ids'] = screen(values, row['existing_complete_ids'], state['short_screen'][row['parameter']])
    survivors = row['continuation_ids'] if row['short_screen'] else ids
    if any(results[i].status != 'complete' for i in survivors):
        return dict(continue_to500=[i for i in survivors if results[i].status != 'complete'])
    if row['first_calibration']:
        require(state['short_screen'][row['parameter']]['status']=='uncalibrated', 'Do not recalibrate a decided item.')
        state['short_screen'][row['parameter']] = calibration(values, row['existing_complete_ids'])
    winner, node = compare(values)[0]
    row.update(status='complete', winner=winner.trajectory_id, selected_step=node.step,
        winner_settings=asdict(winner.settings), disposition='selected_from_all_measured_nodes')
    return winner.settings


def commit_first_batch(state, *, boundary, selected, cost, long_categories):
    require(state['current_category']==state['frozen_order'][0] and all(
        r['status']=='complete' for r in state['categories'][state['current_category']]['rounds'])
        and len(state['categories'][state['current_category']]['rounds'])==len(ROUNDS), 'First-class comparisons are incomplete.')
    require(boundary['formal_eligible'] and boundary['completed_prefix']==state['frozen_order'][:1]
        and boundary['export_evidence']['source_node_id']==selected['selected_node_id'], 'Commit the actual selected boundary.')
    require(len(set(long_categories))==len(long_categories)<=2, 'Preserve used long-check slots.')
    state.update(status='first_class_complete_awaiting_second_budget', boundary=boundary,
        long_categories=list(long_categories), cumulative_GPUh=cost,
        current_category=state['frozen_order'][1], approved_categories=[])


def save(path, state):
    write(path, state, replace=True)

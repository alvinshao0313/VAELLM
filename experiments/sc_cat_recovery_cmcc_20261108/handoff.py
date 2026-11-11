"""Read sealed offline evidence and rebuild Top2 without writing to the old run."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_offline_execution_cmcc_20261104.protocol import binding, read, write
from experiments.sc_cat_search_cmcc_20261105.evidence import load_receipt, normalized_recipe


def choose_top2(rows, champion_id):
    require(rows, "No complete legal configuration remains.")
    unique = {}
    for row in rows:
        key = digest(dict(recipe=normalized_recipe(row['recipe']), run_identity=row['run_identity']))
        if key in unique:
            require(unique[key]['full8_mean'] == row['full8_mean'], "Duplicate configuration has conflicting evidence.")
            if row['candidate_id'] == champion_id:
                unique[key] = row
        else:
            unique[key] = row
    candidates = list(unique.values())
    champions = [row for row in candidates if row['candidate_id'] == champion_id]
    require(len(champions) == 1, "The final sequential champion must remain in the complete pool.")
    first = champions[0]
    quality_key = lambda row: (-row['full8_mean'], row['logical_bits'], row['packed_bits'])
    require(quality_key(first) == min(map(quality_key, candidates)), "Final champion conflicts with the full pool.")
    rest = sorted((row for row in candidates if row is not first),
        key=lambda row: (*quality_key(row), row['candidate_id']))
    return [first, *rest[:1]], candidates


def audit(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), "Never replace a handoff audit.")
    manifest, state = read(source / 'manifest.json'), read(source / 'search.json')
    require(state['status'] == 'completed' and state['exit_code'] == 0, "Offline search must be complete.")
    baseline = read(manifest['baseline']['path'])
    categories, evidence, exclusions = {}, {}, []
    for name, category in state['categories'].items():
        require(category['status'] == 'completed' and len(category['rounds']) == 8
            and all(r['status'] == 'completed' for r in category['rounds']), "Incomplete category: " + name)
        refs = [category['anchor']['result']]
        for stage in category['rounds']:
            for candidate in stage['candidates']:
                if candidate.get('result'):
                    refs.append(candidate['result'])
                else:
                    require(candidate['status'] in ('uncovered_cap', 'math_invalid'), "Unresolved candidate: " + name)
                    exclusions.append(dict(category=name, parameter=stage['parameter'], candidate=candidate))
        rows = []
        for ref in {v['path']: v for v in refs}.values():
            require(binding(ref['path']) == ref, "Canonical source changed.")
            row = read(ref['path'])
            receipt = load_receipt(source / 'imports' / (row['candidate_id'] + '.json'), manifest, require_bank=False)
            require(row['run_identity'] == manifest['run_identity'] and row['category_id'] == name
                and row['complete_full8_evaluated'] and row['logical_bits'] <= manifest['caps'][name]['cap_bits']
                and row['full8_evidence']['protocol_identity'] == manifest['protocol_identity']
                and row['full8_evidence']['leaf_count'] == 64 and row['full8_evidence']['sample_count'] == 24742,
                "Candidate identity, coverage or class cap differs.")
            require(all(count % row['recipe']['D'] == 0 for count in row['recipe']['protection']['counts'].values()),
                "Protection count is not aligned to D.")
            evidence[row['candidate_id']] = dict(canonical=ref, sealed_receipt=receipt['receipt_reference'],
                availability=receipt['availability'], recipe=row['recipe'], full8_mean=row['full8_mean'],
                task_metrics=row['full8_evidence']['task_metrics'], logical_bits=row['logical_bits'], packed_bits=row['packed_bits'])
            rows.append(row)
        top, pool = choose_top2(rows, category['current']['candidate_id'])
        categories[name] = dict(top2=[r['candidate_id'] for r in top], anchor_id=category['anchor']['candidate_id'],
            distinct_legal_complete=len(pool), cap=manifest['caps'][name],
            ranking=[r['candidate_id'] for r in sorted(pool, key=lambda r: (-r['full8_mean'], r['logical_bits'], r['packed_bits'], r['candidate_id']))],
            champion_preserves_sequential_tie=True, full8_mean=top[0]['full8_mean'], KL=top[0]['joint_KL'])
    order = sorted(categories, key=lambda name: (-categories[name]['full8_mean'], categories[name]['KL'], name))
    require(order == state['frozen_order']['ordered_categories'], "Recomputed order differs from the frozen order.")
    report = dict(schema='sc_cat_top2_handoff_v1', status='offline_evidence_verified', source_manifest=binding(source / 'manifest.json'),
        source_state=binding(source / 'search.json'), source_selected_settings=binding(source / 'selected_settings.json'),
        parent=manifest['parent'], run_identity=manifest['run_identity'], inventory=manifest['inventory'],
        data_plan=manifest['data_plan'], baseline=manifest['baseline'], cap_evidence=manifest['cap_evidence'],
        protocol_identity=manifest['protocol_identity'], frozen_order=order, categories=categories,
        candidates=evidence, exclusions=exclusions, historical_cost=dict(
            offline_candidate_GPUh=state['candidate_GPU_hours'], offline_readout_GPUh=state['readout_GPU_hours']),
        new_GPUh=0, GPU_started=False, formal_chain_complete=False,
        evidence_scope='Sealed full8 receipts, source SHA and unchanged sample stats verified; large weights not rehashed.',
        reuse_boundary='Only first-class BF16-equivalent banks may be installed after full identity validation; later classes refit on own live parent.')
    report['identity'] = digest(report)
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit(args.source, args.output)
    for name in report['frozen_order']:
        row = report['categories'][name]
        print(name, row['distinct_legal_complete'], [(v[:12], report['candidates'][v]['full8_mean'],
            report['candidates'][v]['availability']['bank_available'],
            report['candidates'][v]['availability']['raw_payload_available']) for v in row['top2']], flush=True)


if __name__ == '__main__':
    main()

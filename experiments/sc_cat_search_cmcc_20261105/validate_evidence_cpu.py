"""CPU checks against complete real historical candidates; no invented scores."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_offline_execution_cmcc_20261104.protocol import binding, read, write
from . import evidence, layerwise

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / 'artifacts/cat_sequential_search_cmcc_20261105/confirmed_order_20261105_1049'


def rejects(call):
    try:
        call()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError('A deliberately mismatched CPU counterexample was accepted.')


def validate(output):
    import torch
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use caller bitvae with CUDA hidden.')
    started = time.time(); manifest = read(OLD / 'manifest.json'); manifest['output_root'] = str(output.resolve())
    state = read(OLD / 'search.json'); category = 'q_proj'
    Aref = manifest['anchors'][category]['result']
    round0 = state['categories'][category]['rounds'][0]
    Bref = next(row['result'] for row in round0['candidates'] if row['requested_value'] is True)
    receipts = [evidence.import_result(reference, manifest) for reference in (Aref, Bref)]
    A, B = (evidence.checked_result(reference, manifest) for reference in (Aref, Bref))
    require(A['full8_mean'] != B['full8_mean'] and all(receipt['historical_quality_verified'] for receipt in receipts), 'Require two real complete independently scored candidates.')
    require(all(receipt['complete_full8']['leaf_count'] == 64 and receipt['complete_full8']['sample_count'] == 24742
        and receipt['availability']['raw_payload_available'] and receipt['availability']['bank_available'] for receipt in receipts), 'Real imports must retain all64/24742 and separate raw/bank availability.')
    inventory = read(manifest['inventory']['path'])['inventory']; _, _, projections = layerwise.paired_sources(A, B, inventory)
    legacy = [layerwise.merge_output_statistics(row['layer_functional_readout'], row, projections, require_fp32=False) for row in (A, B)]
    for index, row in enumerate((A, B)):
        for projection in projections:
            value = legacy[index][projection['actual_name']]
            halves = row['layer_functional_readout']['sufficient_statistics'][projection['actual_name']]
            require(value['elements'] == sum(half['valid_tokens'] for half in halves) * projection['shape'][0]
                and value['MSE'] == value['SSE'] / value['elements'], 'Actual sufficient SSE/valid-output-element CPU merge differs.')
        rejects(lambda row=row: layerwise.merge_output_statistics(row['layer_functional_readout'], row, projections))
    require(all(legacy[0][row['actual_name']]['half_valid_tokens'] == legacy[1][row['actual_name']]['half_valid_tokens'] for row in projections), 'Real A/B valid token halves differ.')
    for key in ('run_identity', 'recipe'):
        wrong = deepcopy(A)
        if key == 'run_identity':
            wrong[key]['parent_weights_digest'] = 'CPU_deliberately_wrong_parent'
        else:
            wrong[key]['encoder']['hidden_dim'] = 256
        rejects(lambda wrong=wrong: layerwise.paired_sources(wrong, B, inventory))
    for key in ('data_identity', 'parent_input_identity'):
        local = deepcopy(A['layer_functional_readout']); local[key] = 'CPU_deliberately_wrong_identity'
        if key == 'data_identity':
            rejects(lambda local=local: layerwise.merge_output_statistics(local, A, projections, require_fp32=False))
        else:
            wrong = deepcopy(A); wrong['layer_functional_readout'] = local
            rejects(lambda wrong=wrong: layerwise.build_mixed(wrong, B, inventory, False))
    wrong = deepcopy(A['layer_functional_readout']); wrong['sufficient_statistics'][projections[0]['actual_name']][0]['records'] = 255
    rejects(lambda: layerwise.merge_output_statistics(wrong, A, projections, require_fp32=False))
    normalized = deepcopy(A['recipe']); normalized['transpose'] = {name: False for name in normalized['group_signature']}
    require(evidence.equivalent_recipe(A['recipe'], normalized) and evidence.normalized_transpose(normalized['transpose'], normalized['group_signature']) is False,
        'Uniform maps must reuse original boolean identity without fabricating another ID.')
    require(evidence.lookup_result([Aref, Bref], manifest, normalized) == [Aref], 'Full resolved lookup failed on a real uniform-map equivalent.')
    require(layerwise.choose_direction(3., 3., True) is True and layerwise.choose_direction(3., 3., False) is False
        and layerwise.choose_direction(3., 2., False) is True and layerwise.choose_direction(2., 3., True) is False, 'Exact local tie must retain entering direction.')
    rejects(lambda: layerwise.choose_direction(float('nan'), 1., False))
    request, completed = (read(A['artifact_evidence'][key]['path']) for key in ('request', 'completed'))
    wrong_completed = deepcopy(completed); wrong_completed['candidates'][0]['fit_report']['frozen_prefix_after_digest'] = 'CPU_wrong_frozen_digest'
    rejects(lambda: evidence._contract(A, request, wrong_completed, inventory, manifest))
    wrong_completed = deepcopy(completed); wrong_completed['candidates'][0]['fit_report']['stages'][0]['init_seed'] += 1
    rejects(lambda: evidence._contract(A, request, wrong_completed, inventory, manifest))
    rejects(lambda: evidence.execute_release_plan(manifest, [], 'CPU_unauthorized_release'))
    require(evidence.cleanup_completed(manifest, dict(payload_retention=dict(no_future_raw_readers=[A['candidate_id'], B['candidate_id']]))) == [],
        'The CPU scope must never propose release of original profile/old19 payloads.')
    # A missing raw state is represented in an isolated metadata-only directory;
    # original source files and model scores are neither copied nor changed.
    unavailable = deepcopy(A); temporary = output / 'unavailable_raw_counterexample'; temporary.mkdir()
    meta = read(Path(A['endpoint']['path']) / 'checkpoint_meta.json')
    write(temporary / 'checkpoint_meta.json', meta); unavailable['endpoint']['path'] = str(temporary)
    availability = evidence.payload_availability(unavailable)
    require(availability['bank_available'] and not availability['raw_payload_available'], 'Historical/bank evidence must not imply a directly loadable raw model.')
    rejects(lambda: evidence.require_native_payload(unavailable))
    (temporary / 'checkpoint_meta.json').unlink(); temporary.rmdir()
    # Load actual PUBLIC bank topology and state on CPU, without a full raw-state
    # rehash or any fabricated model/inference fixture.
    modules = layerwise.load_bank_modules(A['bank'], 'cpu')
    require(set(modules) == set(A['recipe']['group_signature']) and all(module.training is False for module in modules.values()), 'Actual complete PUBLIC bank CPU strict load failed.')
    native_fields = {name: digest({key: list(value.shape) for key, value in module.state_dict().items()}) for name, module in modules.items()}
    require(len(native_fields) == len(projections) and all(module.transpose is False for module in modules.values()), 'Actual bank transpose/topology differed from original recipe.')
    require(not torch.cuda.is_initialized(), 'CPU evidence validation must never initialize CUDA.')
    value = dict(schema='sc_cat_actual_evidence_CPU_validation_v1', status='passed', exit_code=0, actual_sources=[Aref, Bref], imports=[row['receipt_reference'] for row in receipts],
        real_complete_candidates=2, original_full8_leaves_each=64, original_full8_samples_each=24742,
        original_candidate_IDs_and_paths_preserved=True, actual_public_bank_CPU_strict_modules=len(modules),
        original_full512_legacy_SSE_merge_verified=True, legacy_FP64_not_accepted_as_formal_FP32=True,
        exact_tie_retains_entering_direction=True, uniform_boolean_map_deduplicated=True,
        parent_recipe_data_records_init_frozen_mismatches_rejected=True, missing_raw_distinguished_from_quality_and_bank=True,
        no_original_payload_deleted=True, no_model_scores_fabricated=True, CUDA_initialized=False, CPU_seconds=time.time() - started,
        limitation='CPU-only source/import/load and arithmetic-contract checks; real paired FP32/W-MSE and mixed full-M path remain GPU work.')
    write(output / 'validation.json', value)
    print({key: value[key] for key in ('status', 'real_complete_candidates', 'actual_public_bank_CPU_strict_modules', 'CPU_seconds')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / 'validation.json').exists(), 'Use a new CPU verification output; preserve the previous receipt.')
    validate(args.output)


if __name__ == '__main__':
    main()

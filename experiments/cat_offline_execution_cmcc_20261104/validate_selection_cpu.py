"""Re-read genuine historical full8 evidence; separately test pure v0.2 decisions."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.offline import _resolved_like
from .protocol import binding, code_identity, read, read_native_eight, sample_evidence, write
from .results import load_worker_request
from .selection import (
    VERSION, _capacity_decision, _conditional_queue, _conditional_transition, _context, _current_bsq_diagnostics, _expected, _pending_full8, _pending_worker, _request_cap,
    _seal, _verify, best_different_from_anchor, capacity_improved, freeze_joint, load_candidate,
    quality_key, require_same_context, validate_endpoint_set,
)
from .validate_cpu import INVENTORY, PILOT, ROOT, rejects

HISTORICAL = ROOT/'result/qwen3_8b_mmlu65_20261004/up_vae_muon_cmcc_20261104'
PROFILE = ROOT/'artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538/k32/formal/completed.json'


def historical_evidence():
    from experiments.adaptive_class_evaluation_20261030 import bindings as original
    from experiments.adaptive_class_evaluation_20261030.summarize import collect_mixed
    from experiments.w25_eval.acceptance import collect_results
    from experiments.cmcc_runtime_20261030.adaptive_evaluation import setup
    from experiments.up_vae_muon_cmcc_20261104.bindings import SUPPORT
    setup(SUPPORT)
    closure_path = HISTORICAL/'closure_20261104.json'; closure = read(closure_path)
    require(closure['status'] == 'complete_terminal_CPU_verified', 'Use the retained complete same-parent closure.')
    inventory = original.model_inventory(original.BASE)
    baseline_paths, _ = original.template_inputs()
    baseline = collect_results(baseline_paths, inventory)
    records, common_protocol, common_inputs = [], set(), set()
    for arm in ('control', 'muon', 'd16'):
        directory = HISTORICAL/'control' if arm == 'control' else HISTORICAL/'runtime_repair_v2'/arm
        full8 = directory/'raw_full8'; config_path = full8/'config.json'
        config = read(config_path); model = config['model']; native = Path(model['checkpoint']).resolve()
        meta_path = native/'adaptive_model_meta.json'; meta = read(meta_path)
        require(binding(meta_path)['sha256'] == model['metadata_sha256'], 'Retained actual historical metadata changed.')
        paths = [full8/'mmlu.json', full8/'seven.json']
        actual = collect_mixed(paths, inventory, native, meta, model['checkpoint_files_sha256'])
        summary_path = full8/'full8_summary.json'; summary = read(summary_path); run = read(full8/'run.json')
        require(summary == dict(checkpoint_id=meta['checkpoint_id'], checkpoint_format=original.FORMAT, **actual)
            and run['checkpoint_id'] == meta['checkpoint_id']
            and all(row['exit_code'] == 0 for row in run['stages'].values()), 'Historical complete original summary/shards changed.')
        if arm == 'control':
            cpu_closure_path = HISTORICAL/'runtime_repair_v2/control/full8_cpu_closure.json'
            cpu_closure = read(cpu_closure_path); controller = read(HISTORICAL/'runtime_repair_v2/control/run.json')
            require(run['status'] == 'failed' and run['exit_code'] == 1
                and cpu_closure['status'] == 'existing_complete_full8_CPU_verified'
                and cpu_closure['checkpoint_id'] == meta['checkpoint_id'] and cpu_closure['task_metrics'] == actual['task_metrics']
                and cpu_closure['mean_percent'] == actual['mean_percent']
                and controller['status'] == 'completed_up_raw_full8_and_readout' and controller['exit_code'] == 0,
                'Require the actual documented control bookkeeping-failure closure; never rewrite it as an exit0 evaluator run.')
        else:
            require(run['status'] == 'completed_raw_full8' and run['exit_code'] == 0
                and run['mean_percent'] == actual['mean_percent'], 'Historical candidate original evaluator failed or changed.')
        samples = sample_evidence(paths)
        source = closure['records'][arm]
        readout_path = HISTORICAL/'runtime_repair_v2'/arm/'raw_readout.json'; readout = read(readout_path)
        require(source['checkpoint_id'] == meta['checkpoint_id'] == readout['checkpoint_id']
            and source['source_after_gate_uuid'] == readout['parent_checkpoint_id']
            and source['task_metrics'] == actual['task_metrics']
            and source['mean_percent'] == actual['mean_percent'], 'Historical source parent/endpoint/metrics differs from the retained closure.')
        protocol = []
        for path, reference in zip(paths, baseline_paths, strict=True):
            payload = read(path); result = payload['evals']['lm_eval']; expected = read(reference)
            require(result['configs'] == expected['configs'] and result['versions'] == expected['versions']
                and actual['versions'] == baseline['versions'], 'Historical tasks/prompt/version differ from original BF16.')
            protocol.append({key: result[key] for key in ('configs', 'versions', 'n_shot', 'n_samples', 'task_metric_keys', 'batch_size', 'num_fewshot', 'limit')})
        common_protocol.add(digest(dict(shards=protocol, policy=config['model_policy'], packages=config['package_versions'])))
        common_inputs.add(samples['ordered_input_identity'])
        for path in [*paths, summary_path, config_path, meta_path, readout_path]:
            require(binding(path)['sha256'] == closure['evidence_sha256'][str(path.resolve())], 'Original retained historical evidence SHA differs.')
        rejects(collect_mixed, paths[:1], inventory, native, meta, model['checkpoint_files_sha256'])
        foreign = deepcopy(meta); foreign['checkpoint_id'] = 'foreign_CPU_counterexample_UUID'
        rejects(collect_mixed, paths, inventory, native, foreign, model['checkpoint_files_sha256'])
        wrong = deepcopy(model['checkpoint_files_sha256']); wrong[next(iter(wrong))] = '0'*64
        rejects(collect_mixed, paths, inventory, native, meta, wrong)
        records.append(dict(candidate_id=arm, checkpoint_id=meta['checkpoint_id'], parent_uuid=readout['parent_checkpoint_id'],
            full8_mean=actual['mean_percent'], task_metrics=actual['task_metrics'],
            logical_bits=source['actual_partial_ledger']['user_linear_bits'], packed_bits=source['actual_partial_ledger']['packed_linear_bits'],
            original_files=[binding(path) for path in [*paths, summary_path, full8/'run.json', config_path, meta_path, readout_path]],
            leaf_count=actual['leaf_count'], sample_count=actual['sample_count'],
            original_full8_wrapper_exit_code=run['exit_code'],
            control_bookkeeping_CPU_closure=binding(cpu_closure_path) if arm == 'control' else None,
            ordered_input_identity=samples['ordered_input_identity'], weights_currently_available=(native/meta['state_dict_file']).is_file()))
    require(len(common_protocol) == len(common_inputs) == len({row['parent_uuid'] for row in records}) == 1,
        'Historical same-parent cohort does not share common actual full8 conditions/ordered inputs.')
    ranked = sorted(records, key=lambda row: quality_key(row['full8_mean'], row['logical_bits'], row['packed_bits'], row['candidate_id']))
    require([row['candidate_id'] for row in ranked] == ['control', 'muon', 'd16'], 'Genuine historical complete full8 ranking differs.')
    # This historical mixed/N producer has no new PUBLIC plan and must remain ineligible.
    rejects(read_native_eight, summary_path=HISTORICAL/'control/raw_full8/full8_summary.json',
        evaluation_path=HISTORICAL/'control/raw_full8/run.json')
    return dict(status='historical_same_parent_complete_full8_ranking_only', records=ranked,
        source_closure=binding(closure_path), common_protocol_identity=next(iter(common_protocol)),
        reader_source_sha256={str(Path(path).resolve()): binding(path)['sha256'] for path in
            (original.__file__, collect_mixed.__code__.co_filename, collect_results.__code__.co_filename,
                sample_evidence.__code__.co_filename, setup.__code__.co_filename)},
        missing_shard_rejected=True, foreign_UUID_rejected=True, wrong_recorded_source_hash_rejected=True,
        legacy_producer_rejected_by_new_PUBLIC_reader=True,
        fee_scope='Actual recorded whole partial-model fees; not a new offline per-class fee receipt.',
        boundary='Original weights were cleaned after branch closure. Re-read retained metadata/results/all real samples only; no current model-state or PUBLIC full8 execution claim.')


def mathematical_rules():
    # These values are abstract arithmetic inputs, never serialized as model/evaluator evidence.
    endpoints = [dict(slot=index, candidate_id='CPU_math_'+str(index), recipe_digest='CPU_math_recipe_'+str(index)) for index in range(1, 21) if index != 16]
    validate_endpoint_set(endpoints)
    rejects(validate_endpoint_set, endpoints+[dict(slot=16, candidate_id='CPU_math_16', recipe_digest='CPU_math_recipe_16')])
    rejects(validate_endpoint_set, [dict(slot=16, candidate_id='CPU_math_16', recipe_digest='CPU_math_recipe_16')])
    rejects(validate_endpoint_set, endpoints+[dict(slot=21, candidate_id='CPU_math_21', recipe_digest='CPU_math_recipe_21')])
    rejects(validate_endpoint_set, endpoints[:-1]+[endpoints[0]])
    require(quality_key(60.00000000000001, 110, 120, 'b') < quality_key(60., 100, 110, 'a'), 'Do not round away full8 improvements.')
    require(quality_key(60., 99, 120, 'b') < quality_key(60., 100, 110, 'a'), 'Logical fees precede packed fees.')
    require(quality_key(60., 100, 109, 'b') < quality_key(60., 100, 110, 'a'), 'Packed fees break equal-score/logical ties.')
    require(quality_key(60., 100, 110, 'a') < quality_key(60., 100, 110, 'b'), 'Only group ranking uses stable ID tie-breaking.')
    require(capacity_improved(60., (100, 110), 61., (105, 120)), 'Strict score gain may spend the already assigned cap.')
    require(capacity_improved(60., (100, 110), 60., (99, 120)), 'Lexicographic fee reduction can be accepted at equal quality.')
    require(not capacity_improved(60., (100, 110), 59., (90, 100))
        and not capacity_improved(60., (100, 110), 60., (100, 110)), 'Quality loss or same-score/same-fee cannot update capacity.')
    current = dict(candidate_id='b', full8_mean=60., logical_bits=100, packed_bits=110, joint_KL=100.)
    candidate = dict(candidate_id='a', full8_mean=60., logical_bits=100, packed_bits=110, joint_KL=0.)
    require(not _capacity_decision(current, candidate)['accepted'], 'ID/KL must not update equal-score/equal-fee capacity.')
    rejects(quality_key, float('nan'), 1, 1, 'CPU_math_invalid')
    rejects(quality_key, 60., 2, 1, 'CPU_math_invalid')
    sealed = _seal(dict(schema=VERSION, kind='CPU_math_digest_only'), 'block_identity')
    _verify(sealed, 'block_identity'); changed = dict(sealed, kind='foreign_CPU_counterexample')
    rejects(_verify, changed, 'block_identity')
    rejects(_verify, dict(schema='legacy_KL_only', formal_eligible=True), 'certificate_identity')
    require(best_different_from_anchor(['winner', 'anchor', 'third'], 'anchor') == 'winner'
        and best_different_from_anchor(['anchor', 'second'], 'anchor') == 'second'
        and best_different_from_anchor(['anchor'], 'anchor') is None, 'Live nominee must be the best different from anchor, retaining a non-anchor champion.')
    current = dict(candidate_id='CPU_current_accepted', BSQ_diagnostics=[dict(CPU_rule_only=1)])
    _current_bsq_diagnostics(current, dict(kind='BSQ', diagnostics_source_candidate_id=current['candidate_id'], BSQ_diagnostics_digest=digest(current['BSQ_diagnostics'])))
    rejects(_current_bsq_diagnostics, current, dict(kind='BSQ', diagnostics_source_candidate_id='CPU_initial_base', BSQ_diagnostics_digest=digest(current['BSQ_diagnostics'])))
    return dict(scope='Pure CPU arithmetic/set/digest rules, no model metrics fabricated', maximum19_allowed=True,
        twentieth_slot16_or_duplicate_rejected=True, fixed_empty_slot16_rejected_alone=True, BSQ_current_accepted_source_binding_verified=True,
        unrounded_score_order_verified=True,
        capacity_quality_then_actual_fee_verified=True, same_score_fee_lower_ID_or_KL_not_accepted=True,
        NaN_and_invalid_fee_rejected=True, frozen_set_mutation_and_old_KL_certificate_rejected=True,
        best_different_from_anchor_rule_verified=True)


def source_boundaries(baseline_path):
    request_path, request = load_worker_request(PROFILE)
    planned = request['candidates'][0]
    reference = dict(candidate_id=planned['candidate_id'], recipe=planned['recipe'],
        worker=dict(completed=str(PROFILE), request=binding(request_path)))
    context = _context(INVENTORY, baseline_path, request['run_identity'])
    require_same_context(context, context)
    foreign_context = deepcopy(context); foreign_context['run_identity']['parent_uuid'] = 'foreign_CPU_parent_counterexample'
    rejects(require_same_context, foreign_context, context)
    foreign_context = deepcopy(context); foreign_context['protocol_identity'] = 'foreign_CPU_protocol_counterexample'
    rejects(require_same_context, foreign_context, context)
    _expected(reference, context['run_identity'], planned['recipe']['category_id'], 'profile')
    rejects(_request_cap, reference, context)
    rejects(_expected, reference, context['run_identity'], planned['recipe']['category_id'], 'offline')
    bad = deepcopy(reference); bad['worker']['request']['sha256'] = '0'*64
    rejects(_expected, bad, context['run_identity'], planned['recipe']['category_id'], 'profile')
    foreign = deepcopy(reference); foreign['candidate_id'] = 'foreign_CPU_candidate'
    rejects(_expected, foreign, context['run_identity'], planned['recipe']['category_id'], 'profile')
    rejects(freeze_joint, inventory_receipt=INVENTORY, baseline_path=baseline_path, run_identity=request['run_identity'],
        proposals=[dict(banks={planned['recipe']['category_id']: dict(reference, bank_digest='not_an_actual_bank')})])
    completed = read(PILOT); candidate = completed['candidates'][0]['candidate_id']
    command = read(PILOT.parent/'process.json')['command']; actual_request = binding(command[command.index('--request')+1])
    row = dict(candidate_id=candidate, worker=dict(completed=str(PILOT), request=actual_request))
    rejects(load_candidate, row, context)
    # Recipe transition checks use the actual planned geometry, without claiming fitted models or scores.
    inventory = read(INVENTORY)['inventory']; base_recipe = deepcopy(planned['recipe'])
    encoder = _resolved_like(inventory, base_recipe, encoder_width=256)
    decoder_independent = _resolved_like(inventory, base_recipe, decoder_width=256)
    s4 = _resolved_like(inventory, base_recipe, stages=4)
    _conditional_transition(inventory, dict(recipe=base_recipe), dict(recipe=encoder), dict(kind='encoder256'))
    _conditional_transition(inventory, dict(recipe=base_recipe), dict(recipe=decoder_independent), dict(kind='decoder256'))
    rejects(_conditional_transition, inventory, dict(recipe=base_recipe), dict(recipe=s4), dict(kind='S4'))
    rejects(_conditional_transition, inventory, dict(recipe=encoder), dict(recipe=decoder_independent), dict(kind='decoder256'))
    with tempfile.TemporaryDirectory(prefix='cat_full8_selection_CPU_') as directory:
        queue_path = Path(directory)/'explicit_unconfigured_queue.json'
        write(queue_path, dict(schema=VERSION, kind='explicit_conditional_queue', authorization_reference='CPU_unconfigured_counterexample_only',
            base_candidate_id=planned['candidate_id'], run_identity_digest=digest(context['run_identity']),
            protocol_identity=context['protocol_identity'], instructions=[dict(kind='BSQ')]))
        queue = _conditional_queue(binding(queue_path), context, dict(candidate_id=planned['candidate_id']))
        require(queue['configuration_status'] == 'unconfigured_BSQ_generation', 'Pending BSQ generation must remain unconfigured.')
        opaque = read(queue_path); opaque['BSQ_rule_reference'] = binding(INVENTORY)
        opaque['BSQ_rule_authorization_reference'] = 'CPU_opaque_reference_is_not_a_reviewed_generator'
        opaque['instructions'][0].update(field='commitment_loss_weight', value=.125,
            diagnostics_source_candidate_id=planned['candidate_id'], BSQ_diagnostics_digest='CPU_pre_frozen_diagnostics_counterexample')
        opaque_path = Path(directory)/'opaque_rule_queue.json'; write(opaque_path, opaque)
        require(_conditional_queue(binding(opaque_path), context, dict(candidate_id=planned['candidate_id']))['configuration_status']
            == 'unconfigured_BSQ_generation', 'An opaque rule reference cannot configure pending BSQ direction generation.')
        missing = Path(directory)/'unproduced.json'
        require(_pending_worker(dict(worker=dict(completed=str(missing)))) == 'fit_pending', 'Missing fit must remain pending.')
        require(_pending_full8(dict(full8=dict(summary=str(missing), evaluation=str(missing)))) == 'evaluation_pending', 'Missing evaluation must remain pending.')
        failed = Path(directory)/'failed.json'; write(failed, dict(status='failed', exit_code=1))
        require(_pending_worker(dict(worker=dict(completed=str(failed)))) == 'fit_dependency_failed', 'Failed fit is a dependency, never a low score.')
    return dict(actual_formal_profile_registry=binding(request_path), static_formal_profile_identity_verified=True,
        profile_cannot_masquerade_as_offline=True, wrong_request_SHA_rejected=True, foreign_candidate_rejected=True,
        imported_profile_request_cannot_supply_offline_cap=True,
        incomplete_joint_category_sources_rejected=True, actual_technical_worker_rejected=True,
        foreign_parent_or_protocol_cap_context_rejected=True, independently_fitted_same_base_points_cannot_form_capacity_sequence=True,
        resolved_encoder_decoder_and_derived_owner_contracts_verified=True, old_S4_condition_rejected=True,
        BSQ_generation_unconfigured_even_with_opaque_rule_reference=True,
        missing_fit_full8_and_failed_fit_stay_pending=True,
        new_formal_offline_full8_positive=False, actual_joint_full8_positive=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical-only', action='store_true')
    parser.add_argument('--source-only', action='store_true'); parser.add_argument('--reused-report', type=Path); args = parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use caller bitvae with CUDA hidden.')
    import torch
    torch.set_num_threads(2); started = time.time()
    baseline_path = ROOT/'result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/current/bf16_quality_receipt.json'
    if args.source_only:
        require(args.reused_report is not None, 'A focused added-boundary check must bind the genuine prior complete CPU report.')
        reused = read(args.reused_report)
        require(reused['status'] == 'CPU_v0.2_selection_rules_and_genuine_historical_full8_verified' and reused['CUDA_initialized'] is False,
            'Reuse only the actual successful historical/rule CPU evidence.')
        report = dict(status='CPU_SC_CAT_S2_selection_and_source_boundaries_verified', mathematical_rules=mathematical_rules(),
            source_boundaries=source_boundaries(baseline_path), reused_complete_CPU_report=binding(args.reused_report),
            source_sha256=code_identity(), CUDA_initialized=torch.cuda.is_initialized(), elapsed_seconds=time.time()-started,
            scope='Only the added imported request/cap source checks and affected source/recipe boundaries; no repeated historical evaluator read or GPU inference.')
        require(not report['CUDA_initialized'], 'Focused source check must not initialize CUDA.')
        write(args.output, report); print({key: report[key] for key in ('status', 'CUDA_initialized', 'elapsed_seconds')}, flush=True)
        return
    if args.historical_only:
        historical = historical_evidence()
        require(not torch.cuda.is_initialized(), 'Historical CPU reader must never initialize CUDA.')
        write(args.output, dict(historical, CUDA_initialized=False, consumer_family_isolation='separate_actual_historical_reader_process'))
        return
    # The original mixed reader binds its N family at import. Keep that real
    # consumer in a separate CPU process rather than relabelling it PUBLIC.
    with tempfile.TemporaryDirectory(prefix='cat_historical_full8_CPU_') as directory:
        path = Path(directory)/'historical_reader.json'
        subprocess.run([sys.executable, '-B', '-u', '-m', __package__+'.validate_selection_cpu',
            '--historical-only', '--output', str(path)], check=True)
        historical = read(path)
    report = dict(status='CPU_v0.2_selection_rules_and_genuine_historical_full8_verified',
        mathematical_rules=mathematical_rules(), historical_full8=historical, source_boundaries=source_boundaries(baseline_path),
        elapsed_seconds=time.time()-started, source_sha256=code_identity(), CUDA_initialized=torch.cuda.is_initialized(),
        new_GPU_evaluation=False, full_S0_S13_executor_implemented=False,
        limitations='No actual new PUBLIC formal offline/joint full8 positive. No new profile8 policy, proposal generation/4+4, neighbor8-9, BSQ or layer-group trigger policy, cap allocation rule, GPU budget or search launch.')
    require(not report['CUDA_initialized'], 'CPU selection validation must never initialize CUDA.')
    write(args.output, report)
    print({key: report[key] for key in ('status', 'CUDA_initialized', 'elapsed_seconds')}, flush=True)


if __name__ == '__main__':
    main()

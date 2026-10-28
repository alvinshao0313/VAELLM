"""Finite native short check, common128 anchor, or one fixed continuation arm."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import deosc_bindings as b
import deosc_runtime as runtime
from controller import matched_masks


def verify_frozen(model, frozen, excluded, caches):
    b.require(runtime.frozen_stamp(model, excluded) == frozen,
              'A persistent source field outside the six decoders/two code banks changed.')
    runtime.assert_caches(model, caches)


def finish_windows(manager, trackers):
    return {spec.canonical_key: trackers[spec.canonical_key].finish_window(
        manager.score_module.score_view(spec)) for spec in manager.bank_specs}


def select_masks(manager, trackers):
    masks, report = {}, {}
    for spec in manager.bank_specs:
        key = spec.canonical_key
        score = manager.score_module.score_view(spec)
        risk = trackers[key].risk_mask(b.THRESHOLD)
        selected, control, info = matched_masks(score, trackers[key].scale, risk,
            distance_bins=b.DISTANCE_BINS, seed=b.MATCH_SEED + spec.stage_idx)
        masks[key] = {'risk': selected.cpu(), 'matched': control.cpu()}
        report[key] = info
    return masks, report


def short_check(model, target, active, manager, trackers, optimizer, output):
    saved = runtime.anchor(active, manager, trackers, optimizer, step=2)
    packed = manager.checkpoint_packed_snapshot()
    before = runtime.value_proof(active, manager, optimizer)
    intervention = {}
    for spec in manager.bank_specs:
        mask = b.t.zeros(spec.n_active, dtype=b.t.bool, device=spec.device)
        mask[:128] = True
        intervention[spec.canonical_key] = trackers[spec.canonical_key].reanchor(
            manager.score_module.score_view(spec), mask)
    b.require(sum(row['score_values_changed'] for row in intervention.values()) > 0,
              'The diagnostic must move genuine updated scores, not only initialized centers.')
    current = manager.checkpoint_packed_snapshot()
    actual_moments = manager.bit_optimizer.state_dict()['chunks']
    expected_moments = saved['bit']['bit_optimizer']['chunks']
    b.require(all(b.t.equal(current[key], value) for key, value in packed.items()),
              'Selective score snapping changed actual native packed codes.')
    b.require(all(b.t.equal(actual_moments[key][field], values[field]) for key, values in expected_moments.items()
                  for field in ('exp_avg', 'exp_avg_sq')), 'Selective score snapping changed real Adam moments.')
    after = runtime.value_proof(active, manager, optimizer)
    b.require(before['decoder'] == after['decoder'] and before['decoder_Adam'] == after['decoder_Adam'],
              'Score-only snapping changed actual decoder values or decoder Adam.')
    runtime.restore_anchor(saved, active, manager, trackers, optimizer)
    b.require(all(b.t.equal(p.detach().cpu(), saved['decoder'][key]) for key, p in active.items())
        and all(b.t.equal(score.detach().cpu(), expected) for score, expected in
                zip(manager.score_module.score_chunks, saved['bit']['score_chunks'])),
        'The real common-state restore failed to recover original FP32 values.')
    actual_moments = manager.bit_optimizer.state_dict()['chunks']
    b.require(all(b.t.equal(actual_moments[key][field], values[field]) for key, values in expected_moments.items()
                  for field in ('exp_avg', 'exp_avg_sq')), 'The real common-state restore changed bit Adam values.')
    b.require(runtime.rng_state()['python'] == saved['rng']['python'], 'The common RNG restore failed.')
    proof = {'diagnostic_intervention': intervention, 'diagnostic_mask_is_not_a_risk_selection': True,
        'actual_packed_codes_unchanged': True, 'actual_bit_Adam_m_v_exact': True,
        'actual_decoder_and_decoder_Adam_exact': True, 'actual_anchor_score_decoder_moments_restored': True,
        'actual_optimizer_steps': 2, 'maximum_prompt_tokens': 32561}
    del saved, actual_moments, expected_moments
    proof['endpoint'] = runtime.endpoint(output / 'step_0002', model, target, active, manager,
                                         optimizer, step=2, arm='short')
    return proof


def execute(args):
    b.require(os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True',
              'Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before interpreter startup.')
    b.require(os.environ.get('FLASH_ATTENTION_DETERMINISTIC') == '1',
              'Set FLASH_ATTENTION_DETERMINISTIC=1 identically for every paired trajectory.')
    b.require(Path(sys.executable).resolve() == b.previous.PYTHON.resolve()
        and b.t.cuda.is_available() and b.t.cuda.device_count() == 1,
        'Use the caller-activated bitvae and exactly one assigned GPU.')
    output = args.output_root.resolve()
    b.require(not (output / 'report.json').exists() and not (output / 'training.jsonl').exists(),
              'Use a fresh run directory.')
    output.mkdir(parents=True, exist_ok=True)
    started, deadline = time.monotonic(), time.time() + args.max_seconds
    report = {'status': 'preparing', 'phase': args.phase, 'arm': args.arm,
        'pid': os.getpid(), 'process_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'source_checkpoint_id': b.SOURCE_ID, 'source_dir': str(b.SOURCE), 'target': b.TARGET,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'source_pins': b.own_bindings(),
        'environment': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF', 'FLASH_ATTENTION_DETERMINISTIC')},
        'parameters': {'common_steps': b.COMMON_STEPS, 'final_steps': b.FINAL_STEPS, 'window': b.WINDOW,
            'risk_threshold': b.THRESHOLD, 'risk_min_window_flips': 2, 'distance_bins': b.DISTANCE_BINS,
            'match_seed': b.MATCH_SEED, 'bit_lr': b.BIT_LR, 'bit_Adam_betas': [.9, .999],
            'decoder_lr': b.DECODER_LR, 'decoder_linear_schedule_denominator': 500,
            'decoder_AdamW_betas': [.9, .95], 'decoder_clip': 1.5, 'score_gradient_clip': None,
            'ordinary_round_and_stability_steps': b.FINAL_STEPS + 1, 'intervention_steps': [b.COMMON_STEPS],
            'flipback_intervals_cross_window_boundaries': True, 'risk_and_return_pair_counts_are_window_local': True},
        'completed_updates': 0, 'endpoints': {}, 'windows': {}}
    path = output / 'report.json'
    b.write(path, report)
    def emit(event, **values):
        print(json.dumps({'event': event, **values}, ensure_ascii=False, allow_nan=False), flush=True)
    model = training = manager = trackers = optimizer = active = None
    try:
        training, record, evidence = b.inputs()
        report['data_evidence'] = evidence
        report['original_training_record'] = b.binding(b.PREVIOUS / 'prepared.json')
        b.t.manual_seed(0)
        b.t.cuda.manual_seed_all(0)
        random_seed = 0
        import random
        import numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        model, audit = b.previous.g.native.load_source(record)
        report['load_audit'], report['load_memory'] = audit, b.memory()
        target, active, manager, trackers, optimizer, frozen, excluded, caches = runtime.install(model)
        report['coordinate_scales'] = dict(manager.score_module.coordinate_scales)
        report['source_mutability'] = {'frozen_persistent_fields': len(frozen),
            'mutable_decoder_fields': list(active), 'mutable_code_fields': [b.TARGET + '.vq_weight', b.TARGET + '.vq_weight_s1'],
            'frozen_native_main_cache_modules': len(caches), 'all108_sparse_payloads_frozen': True}
        start_step = 0
        masks = None
        if args.phase == 'arm':
            b.require(args.anchor is not None and args.arm in b.ARMS, 'A continuation requires its common anchor and fixed arm.')
            saved = b.t.load(args.anchor, map_location='cpu', weights_only=True, mmap=True)
            b.require(saved['source_pins'] == report['source_pins'] and saved['data_evidence'] == evidence,
                      'All continuation arms must use the actual same core and train evidence.')
            start_step = runtime.restore_anchor(saved, active, manager, trackers, optimizer)
            b.require(start_step == b.COMMON_STEPS, 'Only the actual common128 anchor is supported.')
            masks = saved['masks']
            report['shared_anchor'] = b.binding(args.anchor)
            report['anchor_selection'] = saved['selection']
            report['anchor_state_restored'] = True
            if args.arm in ('risk', 'matched'):
                report['intervention'] = {spec.canonical_key: trackers[spec.canonical_key].reanchor(
                    manager.score_module.score_view(spec), masks[spec.canonical_key][args.arm].to(spec.device))
                    for spec in manager.bank_specs}
            elif args.arm == 'half_lr':
                manager.bit_optimizer.lr = b.BIT_LR / 2
            finish_windows(manager, trackers)
            del saved, masks
        elif args.phase != 'short':
            b.require(args.phase == 'anchor' and args.arm is None and args.anchor is None,
                      'The common prefix is not an independently initialized arm.')
        report.update(status='training', completed_updates=start_step)
        b.write(path, report)
        stop = 2 if args.phase == 'short' else (b.COMMON_STEPS if args.phase == 'anchor' else b.FINAL_STEPS)
        step = start_step
        with (output / 'training.jsonl').open('x', encoding='utf-8') as log:
            while step < stop:
                b.previous.tick(deadline)
                step += 1
                tick = time.monotonic()
                b.t.cuda.reset_peak_memory_stats()
                row = runtime.update(model, active, manager, trackers, optimizer, training, record, step,
                    freeze=args.arm == 'freeze_codes', deadline=deadline, emit=emit)
                b.t.cuda.synchronize()
                row['seconds'] = time.monotonic() - tick
                verify_frozen(model, frozen, excluded, caches)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                report['completed_updates'] = step
                report['last_update_seconds'] = row['seconds']
                b.write(path, report)
                emit('native_update', step=step, bit_state_step=manager.bit_round_step,
                     domain_losses=row['domain_losses'], seconds=row['seconds'], memory=row['memory'])
                if args.phase == 'short' and step == 1:
                    native, proof = runtime.native_weight_record(model, target, active, manager, optimizer)
                    report['mid_short_native_transition'] = {'native_weight_record': native, **proof,
                        'real_decoder_Adam_step': 1, 'real_bit_Adam_step': manager.bit_round_step}
                if args.phase == 'anchor' and step == b.COMMON_STEPS:
                    masks, selection = select_masks(manager, trackers)
                    report['selection'] = selection
                    raw = sum(info['raw_risk_count'] for info in selection.values())
                    matched = sum(info['matched_pair_count'] for info in selection.values())
                    report['raw_risk_bits'], report['matched_risk_bits'] = raw, matched
                    report['windows'][str(step)] = finish_windows(manager, trackers)
                    report['endpoints']['128'] = runtime.endpoint(output / 'step_0128', model, target,
                        active, manager, optimizer, step=step, arm='common')
                    if matched:
                        saved = runtime.anchor(active, manager, trackers, optimizer, step=step)
                        saved.update(masks=masks, selection=selection, source_pins=report['source_pins'], data_evidence=evidence)
                        b.t.save(saved, output / 'anchor128.pt')
                        report['shared_anchor'] = b.binding(output / 'anchor128.pt')
                        report['status'] = 'anchor_ready'
                        del saved, masks
                    else:
                        report['comparison_not_triggered'] = 'no_window_risk' if not raw else 'no_matched_nonrisk_pairs'
                        report['only_baseline_continues'] = True
                        stop = b.FINAL_STEPS
                        args.arm = 'baseline'
                        report['arm'] = 'baseline'
                elif step % b.WINDOW == 0:
                    report['windows'][str(step)] = finish_windows(manager, trackers)
                    if step == 192:
                        report['endpoints']['192'] = runtime.endpoint(output / 'step_0192', model, target,
                            active, manager, optimizer, step=step, arm=args.arm)
                b.write(path, report)
        if args.phase == 'short':
            report['short_verification'] = short_check(model, target, active, manager, trackers, optimizer, output)
            report['status'] = 'short_passed'
        elif step == b.FINAL_STEPS:
            report['endpoints']['256'] = runtime.endpoint(output / 'step_0256', model, target,
                active, manager, optimizer, step=step, arm=args.arm)
            report['status'] = 'training_complete'
        verify_frozen(model, frozen, excluded, caches)
        b.require(report['coordinate_scales'] == manager.score_module.coordinate_scales, 'The fixed sensitivity scale changed.')
        report.update(frozen4063_unchanged=True, all108_sparse_payloads_unchanged=True,
                      non_target_native_caches_unchanged=True, ordinary_round_resets=0)
        for info in report['source_pins']:
            b.previous.check_binding(info)
    except BaseException as error:
        report.update(status='failed_or_incomplete', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, finished_unix=time.time(),
                      final_memory=b.memory() if b.t.cuda.is_initialized() else {})
        b.write(path, report)
        model = training = manager = trackers = optimizer = active = None
        gc.collect()
        b.t.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('short', 'anchor', 'arm'))
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--arm', choices=b.ARMS)
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    b.require(args.max_seconds > 0, 'A finite positive runtime budget is required.')
    execute(args)


if __name__ == '__main__':
    main()

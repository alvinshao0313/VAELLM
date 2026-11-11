"""Fit one authorized actual live recipe; no scheduling, recovery or retries."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time
import traceback

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
from train_utils import checkpoint_v6 as v6
from tools.cat_eval import _prepare_model_for_eval

from . import live_io as io


def probe(model, input_ids):
    import logging
    model.eval().to('cuda:0')
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    require(not warmup['failed'] and warmup['warmed'] == warmup['total'], 'Every actual compressed module must prewarm through the deployment path.')
    with torch.no_grad():
        logits = model(input_ids=input_ids.to('cuda:0'), use_cache=False).logits.detach().cpu()
    require(torch.isfinite(logits).all().item(), 'The real live raw produced nonfinite deployment logits.')
    return logits, warmup


def execute(request, *, request_path):
    started = time.time()
    io.validate_execution(request)
    manifest = io.source_manifest(request)
    pins = io.verify_fit_sources(manifest)
    inventory, prepared, _sources, tokenizer = io.load_inputs(request, with_sources=False)
    parent = io.actual_parent(request)
    identity = io.live_identity(manifest, parent)
    attention = request['execution_policy']['attention_implementation']
    require(attention == 'sdpa', 'Keep the actual source9 SDPA fitting/statistics policy; full8 has its separately frozen FA2 consumer.')
    recipe = deepcopy(request['recipe'])
    require(recipe['S'] == 2 and recipe['B'] * recipe['D'] == 262144, 'Keep the actual selected S2 and coupled formal B/D.')
    selected = io.read(Path(manifest['output_root']) / 'selected_settings.json')
    category = next(row for row in selected['categories'] if row['category_id'] == request['category_id'])
    allowed = [category['anchor'], category['best_different']['result']]
    originals = [io.read(io.bound_path(reference)) for reference in allowed]
    require(any(row['candidate_id'] == request['candidate_id'] and row['recipe'] == recipe for row in originals), 'Fit exactly the approved anchor/best-different complete recipe, preserving offline source provenance.')
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    planned = candidate_id(identity, recipe)
    registry = dict(phase=request['phase'], run_identity=identity, technical_steps=None if request['phase'] == 'formal' else 2,
        data_plan=str(io.bound_path(request['data_plan_path'])), data_plan_sha256=io.binding(io.bound_path(request['data_plan_path']))['sha256'],
        candidates=[dict(candidate_id=planned, recipe=recipe)], source_request=io.binding(request_path), source_pins=pins)
    registry_path = output / 'formal_registry.json'
    io.write(registry_path, registry)
    calibration = dict(input_ids=prepared['calibration'], manifest=prepared['manifest']['calibration'],
        data_identity=prepared['data_identity'], model_path=inventory['source_evidence']['path'], dataset_spec='alpaca=1')
    if request['phase'] == 'formal':
        calibration['formal_recipe_certificate'] = dict(phase='formal', formal_eligible=True,
            registry_path=str(registry_path), registry_sha256=io.binding(registry_path)['sha256'])
    result = io.source9_live_fit(boundary_reference=parent, run_identity=identity, resolved_recipe=recipe,
        inventory=inventory, calibration_bundle=calibration, frozen_order=io.ORDER, device='cuda:0',
        output_dir=output / 'fitted', attention_implementation=attention, phase=request['phase'],
        technical_steps=None if request['phase'] == 'formal' else 2)
    model, bank = result.pop('_fitted_model'), result.pop('_category_bank')
    activation = result.pop('activation_bundle')
    result['activation_evidence'] = {key: activation[key] for key in ('source_parent_uuid', 'source_weights_digest',
        'target_digest', 'calibration_token_digest', 'calibration_data_identity', 'statistics_identity')}
    del activation
    raw_dir = Path(result['round_base_dir'])
    meta = v6.load_v6_meta(str(raw_dir))
    saved_state = torch.load(raw_dir / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require(tensor_digest(model.state_dict()) == tensor_digest(saved_state), 'The saved complete live round_base differs from its actual fitted source model.')
    del saved_state
    input_ids = prepared['calibration'][0][:, :8].contiguous()
    expected, warmup = probe(model, input_ids)
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    fresh, evidence = fresh_native(raw_dir, model_factory=lambda directory: architecture(directory, attn_implementation=attention))
    actual, fresh_warmup = probe(fresh, input_ids)
    require(torch.equal(expected, actual), 'Independent fresh same-input BF16 deployment logits changed.')
    evidence.update(forward_executed=True, gpu_validation=True, same_input_same_BF16_path_logits_exact=True,
        input_identity=tensor_digest({'input_ids': input_ids}), source_model_state_exact=True,
        source_warmup=warmup, fresh_warmup=fresh_warmup)
    del fresh, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    class_fee = result['actual_ledger']['per_category'][request['category_id']]
    require(class_fee['logical_bits'] <= manifest['caps'][request['category_id']]['cap_bits'], 'Actual live category fee exceeds its fixed cap.')
    require(time.time() < request['deadline_epoch'], 'The allocated fit/export/fresh stage window expired; no continuation.')
    formal = request['phase'] == 'formal' and result['fit_report']['formal_eligible'] and evidence['gpu_validation']
    result.update(status='completed', exit_code=0, formal_eligible=bool(formal), request=io.binding(request_path),
        request_identity=digest(request), run_identity=identity, recipe=recipe, offline_candidate_id=request['candidate_id'],
        fresh_evidence=evidence, bank=bank, source_pins=pins, execution_policy=request['execution_policy'], elapsed_seconds=time.time() - started,
        bank_reuse=dict(reused=False, reason='The shared native BF16 wrapper has a new parent UUID/payload identity; the existing installer requires exact source-parent identity. No metadata-only equivalence claim or donor installation was used.'))
    result['endpoint'] = dict(kind='native_v6', path=str(raw_dir), checkpoint_id=meta['checkpoint_id'],
        model_identity=inventory['model_identity'], metadata_sha256=io.binding(raw_dir / 'checkpoint_meta.json')['sha256'],
        state_sha256=io.binding(raw_dir / meta['state_dict_file'])['sha256'], formal_eligible=bool(formal), phase=request['phase'])
    io.write(output / 'live_raw.json', result)
    io.write(output / 'completed.json', dict(status='completed', exit_code=0, live_raw=io.binding(output / 'live_raw.json'),
        request_identity=digest(request), elapsed_seconds=result['elapsed_seconds'], phase=request['phase']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    request = io.read(request_path)
    try:
        result = execute(request, request_path=request_path)
    except Exception:
        output = Path(request['output'])
        output.mkdir(parents=True, exist_ok=True)
        if not (output / 'failed.json').exists():
            io.write(output / 'failed.json', dict(status='failed', exit_code=1, request=io.binding(request_path), traceback=traceback.format_exc()))
        raise
    print(dict(status=result['status'], round_base_uuid=result['round_base_uuid'], candidate_id=result['candidate_id']), flush=True)


if __name__ == '__main__':
    main()

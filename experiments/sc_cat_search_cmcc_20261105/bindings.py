"""Bind the approved offline budget to the unchanged source9 model fitter."""
from copy import deepcopy
from pathlib import Path
import math
import os
import shutil
import time

from experiments.cat_sequential_search_cmcc_20261105 import bindings as old
from experiments.cat_auto_protocol_cmcc_20261104.config import resolve_recipe
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.geometry import component_plan
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
from experiments.cat_offline_execution_cmcc_20261104.protocol import load_baseline, verify_files

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORDER = ('protection', 'transpose', 'rotation', 'dimension', 'encoder', 'decoder', 'optimizer', 'multiplier')
LIBRARY = dict(protection=['1/32'], transpose=[False, True], rotation=['none', '32', '64', '128', '256', 'full'],
    dimension=[8, 16, 32, 64, 128], encoder=[128, 256], decoder=[128, 256],
    optimizer=['adamw', 'vae_muon'], multiplier=[.25, .5, 1, 2, 4])
DEFAULT_SETTINGS = deepcopy(old.DEFAULT_SETTINGS)
read, write, write_new, bound, sha = old.read, old.write, old.write_new, old.bound, old.sha
PHASE_RULES = dict(formal_base_seconds={'8':7200, '16':5400, '32':3600, '64':3600, '128':3600},
    full8_seconds=900, technical_base_seconds=900, technical_large_seconds=1800)


def normalize_transpose(value):
    if isinstance(value, dict) and value and len(set(value.values())) == 1:
        return next(iter(value.values()))
    return value


def normalized_recipe(recipe):
    value = deepcopy(recipe)
    value['transpose'] = normalize_transpose(value['transpose'])
    return value


def recipe_key(recipe):
    return digest(normalized_recipe(recipe))


def pins():
    return {**old.pins(), **{str(path.resolve()):sha(path) for path in HERE.glob('*.py')}}


def validate_authorization(value):
    require(value.get('confirmed') is True and value.get('authorization_reference'), 'Bind the actual whole-batch user approval.')
    require(value.get('candidate_GPU_hours') == 350 and value.get('readout_GPU_hours') == 3.5,
        'Only the approved350 candidate plus3.5 readout GPUh envelope is supported.')
    require(value.get('gpus') == list(range(6)) and value.get('protected_gpus') == [6,7]
        and {'formal live CAT fits','category recovery/KD','LongBench','BF16 recipe replay'} <= set(value.get('exclusions',[])),
        'This approval is offline search onGPU0..5 only, with no live/KD/LongBench/replay.')
    require(value.get('maximum_distinct_per_category') == 20 and value.get('order') == list(ORDER),
        'Bind protection-first twenty-configuration rules, including one dedicated mixed candidate.')
    require(value.get('authorization_evidence',{}).get('question_id') == 'call_81e0e6704b6f430680438a62984cc201'
        and value.get('prior_controller',{}).get('pid') == 3252579 and value.get('total_GPU_hours') == 353.5
        and all(value.get(key) == item for key,item in PHASE_RULES.items())
        and value.get('readout_per_category_seconds') == 1800 and 'no hidden extra candidate, retry' in value.get('budget_accounting',''),
        'No retry, old-code mutation, signal handoff or concurrent GPU ownership.')
    return value


def phase_windows(manifest, recipe, *, needs_technical=True):
    rules = manifest['rules']
    formal = rules['formal_base_seconds'][str(recipe['D'])] * (recipe['encoder']['hidden_dim'] + recipe['decoder']['hidden_dim']) // 256
    technical = rules['technical_large_seconds'] if recipe['D'] == 8 or max(recipe['encoder']['hidden_dim'],recipe['decoder']['hidden_dim']) == 256 else rules['technical_base_seconds']
    return dict(technical_validation=technical if needs_technical else 0, formal=formal, full8=rules['full8_seconds'])


def worst_case_GPU_hours(category_count):
    # Widths are selected before the optimizer and four non-reference multipliers.
    per_category = 1.5 + 1.5 + 1.5 + 7.5 + 7.75 + 3.75 + 4.75 + 4.75 + 17
    return per_category * category_count


def recipe(inventory, category, rate, settings):
    return resolve_recipe(inventory, category, rate, dim=settings['dimension'], stages=2,
        transpose=normalize_transpose(settings['transpose']), rotation_level=settings['rotation'],
        protection_fraction=settings['protection'], encoder_width=settings['encoder'], decoder_width=settings['decoder'],
        optimizer=settings['optimizer'], multiplier=settings['multiplier'])


def make_manifest(output, authorization_path, old_manifest_path, result_document):
    authorization = validate_authorization(read(authorization_path))
    prior = read(old_manifest_path)
    approved_prior = authorization['prior_manifest']
    require(Path(old_manifest_path).resolve() == Path(approved_prior['path']).resolve() and sha(old_manifest_path) == approved_prior['sha256'], 'The approved prior manifest must match its exact actual SHA.')
    old.integrity(prior)
    require(prior['manifest_identity'] == digest({k:v for k,v in prior.items() if k != 'manifest_identity'}), 'Original frozen manifest identity changed.')
    inventory = read(prior['inventory']['path'])['inventory']
    baseline = load_baseline(prior['baseline']['path'])
    require(prior['protocol_identity'] == baseline['protocol_identity'] and prior['run_identity']['model'] == inventory['model_identity'], 'The source BF16 and common protocol differ.')
    require(len(inventory['category_ids']) == 7 and all(len([p for p in inventory['projections'] if p['category_id'] == c]) == 36 for c in inventory['category_ids']),
        'This approved concrete batch is Qwen3-8B seven fullM36 categories; another model needs its own budget.')
    output, result_document = Path(output).absolute(), Path(result_document).absolute()
    require(str(output) == authorization['output_root'] and str(result_document) == authorization['result_document'], 'Authorization binds this actual output and unique result document.')
    require(output.name == '2026-11-05_184620_s31' and output.parent.name == 'qwen--qwen3-8b'
        and result_document == REPO/'docs/experience/records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md',
        'Use this approved model/run and its unique result document.')
    anchors = deepcopy(prior['anchors'])
    for category, anchor in anchors.items():
        expected = recipe(inventory, category, prior['rates'][category], DEFAULT_SETTINGS)
        old.checked_result(anchor['result']['path'], prior, expected_recipe=expected)
    rules = dict(PHASE_RULES, order=list(ORDER), maximum_distinct_per_category=20, maximum_mixed_per_category=1,
        gpus=list(range(6)), candidate_GPU_hours=350., readout_GPU_hours=3.5, readout_per_category_seconds=1800,
        authorization_reference=authorization['authorization_reference'])
    maximum = worst_case_GPU_hours(len(inventory['category_ids']))
    require(maximum == rules['candidate_GPU_hours'], 'The full offline candidate envelope must close at350GPUh.')
    value = dict(schema='sc_cat_protection_first_offline_v1', output_root=str(output), result_document=str(result_document),
        authorization=bound(authorization_path), old_manifest=bound(old_manifest_path), old_output_root=prior['output_root'],
        run_identity=prior['run_identity'], parent=prior['parent'], inventory=prior['inventory'], data_plan=prior['data_plan'],
        baseline=prior['baseline'], cap_evidence=prior['cap_evidence'], caps=prior['caps'], rates=prior['rates'],
        protocol_identity=prior['protocol_identity'], anchors=anchors, rules=rules, source_sha256=pins(),
        protocol=authorization['protocol'], execution_protocol=authorization['execution_protocol'],
        maximum_candidate_GPU_hours=maximum, fit_identity_scope='Original source9 fitter; new orchestration pinned separately',
        scope='Full declared offline single-BF16/category search only; no live fit, KD, LongBench or BF16 replay')
    return dict(value, manifest_identity=digest(value))


def integrity(manifest):
    require(manifest['manifest_identity'] == digest({k:v for k,v in manifest.items() if k != 'manifest_identity'}), 'Frozen new manifest changed.')
    verify_files(manifest['source_sha256'])
    for name in ('authorization','old_manifest','inventory','data_plan','baseline','cap_evidence','protocol','execution_protocol'):
        reference = manifest[name]
        require(sha(reference['path']) == reference['sha256'], 'Frozen '+name+' changed.')
    validate_authorization(read(manifest['authorization']['path']))


def prior_complete(manifest):
    path = Path(manifest['old_output_root'])/'search.json'
    state = read(path)
    require(state['status'] in ('completed','completed_with_failures') and type(state.get('exit_code')) is int,
        'The old batch must finish naturally before taking its GPU domain; do not poll or signal it.')
    pid = state['controller_pid']
    require(not Path('/proc',str(pid)).exists(), 'Old controller is still present; its domain has not been returned.')
    require(not any(row['status'] == 'running' for c in state['categories'].values() for s in c['rounds'] for row in s['candidates']),
        'Old state still has running candidates; no takeover.')
    for record_path in (Path(manifest['old_output_root'])/'candidates').glob('*/candidate.json'):
        record = read(record_path)
        require(record['status'] in ('completed','failed'), 'An old candidate is still running/incomplete.')
        for phase in record.get('stages',[]):
            if phase['phase'] == 'full8':
                evaluation = read(record_path.parent/'full8/evaluation.json')
                workers = evaluation['stages']
            else:
                workers = [read(record_path.parent/phase['phase']/'process.json')]
            require(all(not Path('/proc',str(row['pid'])).exists() for row in workers), 'An old phase worker is still present; no GPU domain takeover.')
    return bound(path), state


def validate_resource_release(manifest, path=None):
    old_reference, old_state = prior_complete(manifest)
    from experiments.cmcc_runtime_20261030.sequence import check_resources
    resources = check_resources([str(gpu) for gpu in manifest['rules']['gpus']])
    return dict(old_search=old_reference,old_exit_code=old_state['exit_code'],actual_resource_check=resources,
        checked_epoch=time.time(),gpus=manifest['rules']['gpus'],external_receipt=bound(path) if path else None)


def freeze_source_cache(manifest):
    path = Path(manifest['output_root'])/'source_cache.json'
    if path.exists():
        value = read(path)
        require(value['cache_identity'] == digest({k:v for k,v in value.items() if k != 'cache_identity'}), 'Frozen source cache changed.')
        require(value['manifest_identity'] == manifest['manifest_identity'], 'Cache belongs to another new run.')
        return value
    old_reference, state = prior_complete(manifest)
    results, failures = {}, {}
    for category, value in state['categories'].items():
        for reference in value['known'].values():
            row = read(reference['path'])
            require(sha(reference['path']) == reference['sha256'] and row['category_id'] == category, 'Frozen old canonical reference changed.')
            key = recipe_key(row['recipe'])
            if key in results:
                require(results[key] == reference, 'Two distinct historical artifacts claim one exact configuration; explicit resolution required.')
            results[key] = reference
        for stage in value['rounds']:
            for row in stage['candidates']:
                if row['status'] == 'failed' and row.get('candidate_id'):
                    failures[row['candidate_id']] = dict(category_id=category, parameter=stage['parameter'], error=row.get('error'))
    for value in manifest['anchors'].values():
        reference = value['result']; row = read(reference['path']); results[recipe_key(row['recipe'])] = reference
    value = dict(schema='sc_cat_immutable_source_cache_v1',manifest_identity=manifest['manifest_identity'],old_search=old_reference,
        results=results, failures=failures, purpose='Exact resolved recipe evidence only; old winners are not new choices')
    value['cache_identity'] = digest(value)
    write_new(path,value)
    return value


def checked_result(path, manifest, *, expected_recipe=None):
    from .evidence import checked_result as check
    row = check(path, manifest, expected_recipe=expected_recipe)
    require(row['complete_full8_evaluated'] and row['formal_eligible'] and row['recipe']['S'] == 2
        and row['logical_bits'] <= manifest['caps'][row['category_id']]['cap_bits'], 'Only complete same-contract cap-compliant S2 results may compete.')
    return row


def candidate_row(manifest, category, parameter, settings, requested_value, cache):
    inventory = read(manifest['inventory']['path'])['inventory']
    cap = manifest['caps'][category]['cap_bits']
    settings = deepcopy(settings); settings['transpose'] = normalize_transpose(settings['transpose'])
    item = dict(parameter=parameter, requested_value=requested_value, settings=settings)
    try:
        resolved = recipe(inventory,category,manifest['rates'][category],settings)
        quote = component_plan(inventory,resolved)
        if parameter == 'protection' and quote['capability']['constructible'] and quote['logical_bits'] > cap:
            settings['protection'] = '0'; item['settings'] = deepcopy(settings)
            item['protection_fallback_reason'] = '1/32 forecast exceeds the frozen cap before fitting'
            resolved = recipe(inventory,category,manifest['rates'][category],settings)
            quote = component_plan(inventory,resolved)
            require(set(resolved['protection']['counts'].values()) == {0}, 'No-protection fallback must have real zero counts.')
        item.update(recipe=resolved,candidate_id=candidate_id(manifest['run_identity'],resolved),
            predicted_logical_bits=quote.get('logical_bits'),predicted_packed_bits=quote.get('packed_bits'),capability=quote['capability'])
        require(all(count % resolved['D'] == 0 for count in resolved['protection']['counts'].values()), 'Every actual protected count must be a D multiple.')
        if not quote['capability']['constructible']:
            item.update(status='uncovered_illegal',reason=quote['capability']['reason'])
        elif quote['logical_bits'] > cap:
            item.update(status='uncovered_cap',reason='Forecast actual components exceed this frozen category cap; no borrowing')
        elif item['candidate_id'] in cache.get('failures',{}):
            item.update(status='failed',error='Historical exact candidate failed; automatic duplicate/retry is forbidden: '+str(cache['failures'][item['candidate_id']]))
        elif recipe_key(resolved) in cache['results']:
            reference = cache['results'][recipe_key(resolved)]
            source = checked_result(reference['path'],manifest,expected_recipe=resolved)
            item.update(status='reused',result=reference,candidate_id=source['candidate_id'],
                planned_candidate_id=item['candidate_id'],reuse_kind='exact_resolved_recipe_original_candidate_ID')
        else:
            item['status'] = 'declared_pending'
    except ValueError as error:
        if str(error).startswith(('math_invalid:','impl_blocked:')):
            item.update(status='uncovered_illegal',reason=str(error))
        else:
            raise
    return item


def proposals(manifest, category, parameter, settings, cache):
    return [candidate_row(manifest,category,parameter,dict(settings,**{parameter:value}),value,cache) for value in LIBRARY[parameter]]


def proof_for(manifest,candidate,candidates_root):
    paths = [Path(manifest['old_output_root'])/'candidates',Path(candidates_root)]
    native = old.search.batched.native
    inventory = read(manifest['inventory']['path'])['inventory']
    existing = old.proof_for(manifest,candidate,candidates_root)
    if existing:return existing
    for root in paths:
        for path in root.glob('*/technical_validation/completed.json'):
            exit_path = path.parent/'exit.json'
            if not exit_path.is_file() or read(exit_path).get('exit_code') != 0:continue
            try:native._validate_model_path_evidence(read(path),inventory,manifest['run_identity'],[candidate])
            except (ValueError,KeyError):continue
            return bound(path)
    return None


def make_request(manifest,candidate,directory,gpu,phase,deadline,proof=None):
    request = old.make_request(manifest,candidate,directory,gpu,phase,deadline,proof)
    request['sc_cat_manifest_identity'] = manifest['manifest_identity']
    old.search.validate_request(request)
    return request


def require_space(manifest,reserve_candidates):
    # Reserve a full raw plus technical raw and serialization working room per new slot.
    required = (40 * reserve_candidates + 16) * 1024**3
    available = shutil.disk_usage(manifest['output_root']).free
    require(available >= required, 'Insufficient free disk for complete pending candidate payloads: '+str(dict(required_bytes=required,available_bytes=available)))
    return dict(required_bytes=required,available_bytes=available)

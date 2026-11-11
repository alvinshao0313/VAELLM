"""Bind confirmed search rules, actual joint caps and the unchanged source9 fit."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import time

from experiments.cat_night_profiles_cmcc_20261104 import bindings as prior, search
from experiments.cat_auto_protocol_cmcc_20261104.config import resolve_recipe
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.geometry import component_plan
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
from experiments.cat_offline_execution_cmcc_20261104 import selection
from experiments.cat_offline_execution_cmcc_20261104.protocol import verify_files, load_baseline
from experiments import cat_profile_coverage_cmcc_20261104 as coverage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARAMETERS = ('transpose', 'rotation', 'dimension', 'optimizer', 'multiplier', 'encoder', 'decoder', 'protection')
LIBRARY = dict(transpose=[False, True], rotation=['none', '32', '64', '128', '256', 'full'],
    dimension=[8, 16, 32, 64, 128], optimizer=['adamw', 'vae_muon'], multiplier=[.25, .5, 1, 2, 4],
    encoder=[128, 256], decoder=[128, 256], protection=['1/64', '1/32'])
DEFAULT_SETTINGS = dict(transpose=False, rotation='32', dimension=32, optimizer='adamw', multiplier=1,
    encoder=128, decoder=128, protection='1/64')
read, write, write_new, bound, sha = prior.read, prior.write, prior.write_new, prior.bound, prior.sha


def pins():
    return {**prior.pins(), str(Path(coverage.__file__).resolve()):sha(coverage.__file__),
        **{str(path.resolve()): sha(path) for path in HERE.glob('*.py')}}


def validate_rules(rules):
    require(rules.get('confirmed') is True and rules.get('authorization_reference'), 'Bind the actual confirmed order and finite resource budget.')
    require(len(rules['order']) == 8 and set(rules['order']) == set(PARAMETERS), 'Each registered parameter must occur exactly once.')
    require(rules.get('maximum_distinct_per_category') == 19 and rules.get('extra_layerwise_or_BSQ_or_revisit') is False,
        'Exactly the bounded19 library, with no extra layerwise/BSQ/revisit branch.')
    require(rules['gpus'] and all(type(gpu) is int for gpu in rules['gpus']) and len(set(rules['gpus'])) == len(rules['gpus'])
        and set(rules['gpus']) <= set(range(6)), 'Only actual non-bool integer authorized GPU0..5.')
    require(rules['formal_base_seconds'] == {'8':7200, '16':5400, '32':3600, '64':3600, '128':3600}
        and rules['full8_seconds'] == 900 and rules['technical_base_seconds'] == 900 and rules['technical_large_seconds'] == 1800,
        'Use the reviewed D/width-dependent finite stage caps; no implicit shortening.')
    require(type(rules['total_GPU_hours']) in (float,int) and math.isfinite(rules['total_GPU_hours']) and rules['total_GPU_hours'] > 0,
        'A finite numeric complete GPUh ceiling is required.')
    return rules


def phase_windows(manifest, resolved, *, needs_technical=True):
    rules = manifest['rules']; encoder = resolved['encoder']['hidden_dim']; decoder = resolved['decoder']['hidden_dim']
    formal = rules['formal_base_seconds'][str(resolved['D'])] * (encoder+decoder)//256
    technical = rules['technical_large_seconds'] if resolved['D'] == 8 or max(encoder, decoder) == 256 else rules['technical_base_seconds']
    return dict(technical_validation=technical if needs_technical else 0, formal=formal, full8=rules['full8_seconds'])


def checked_cap(path, identity, inventory, *, inventory_reference, baseline_reference):
    cap = read(path); selection._verify(cap, 'certificate_identity')
    require(cap['kind'] == 'actual_joint_full8_cap_certificate' and cap.get('formal_eligible') and cap['status'] == 'complete'
        and cap['run_identity'] == identity and set(cap['categories']) == set(inventory['category_ids']),
        'A complete actual joint full8 certificate for this same BF16/source9/data is required.')
    require(cap['inventory_receipt']==inventory_reference and cap['baseline_receipt']==baseline_reference
        and cap['N']==inventory['total_target_numel']==cap['winner']['ledger']['N'],
        'Actual cap inventory receipt SHA, target N and BF16 baseline must match this run exactly.')
    require(cap['borrowing_between_categories'] is False and cap['sum_cap_bits'] == sum(row['cap_bits'] for row in cap['categories'].values())
        and cap['sum_cap_bits'] <= cap['hard_cap_bits'], 'Actual fixed category caps must close without borrowing.')
    return cap


def profile_paths():
    night = REPO/'artifacts/cat_night_profiles_cmcc_20261104/night_20261104_1918'
    return sorted(night.glob('gpu*/*/formal/profile_full8_result.json')) + sorted(
        (REPO/'artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837').glob('*/profile_full8_result.json'))


def checked_result(path, manifest, *, expected_recipe=None):
    row = read(path)
    require(row.get('formal_eligible') and row.get('complete_full8_evaluated') and row['phase'] == 'formal'
        and row['run_identity'] == manifest['run_identity'] and row['candidate_id'] == candidate_id(row['run_identity'], row['recipe'])
        and row['full8_evidence']['complete'] and row['full8_evidence']['leaf_count'] == 64 and row['full8_evidence']['sample_count'] == 24742
        and row['full8_evidence']['protocol_identity'] == manifest['protocol_identity'], 'Require a complete canonical source9/full8 result, never a partial or KL substitute.')
    require(expected_recipe is None or row['recipe'] == expected_recipe, 'Reused endpoint differs from the full resolved candidate recipe.')
    selection.quality_key(row['full8_mean'], row['logical_bits'], row['packed_bits'], row['candidate_id'])
    require(row['logical_bits'] <= manifest['caps'][row['category_id']]['cap_bits'], 'Measured category endpoint exceeds its fixed cap.')
    return row


def recipe(inventory, category, rate, settings):
    return resolve_recipe(inventory, category, rate, dim=settings['dimension'], stages=2, transpose=settings['transpose'],
        rotation_level=settings['rotation'], protection_fraction=settings['protection'], encoder_width=settings['encoder'],
        decoder_width=settings['decoder'], optimizer=settings['optimizer'], multiplier=settings['multiplier'])


def make_manifest(output, rules_path, cap_path):
    rules = validate_rules(read(rules_path)); original, inventory, identity = prior.prior.common()
    baseline = load_baseline(prior.BASELINE)
    cap = checked_cap(cap_path, identity, inventory, inventory_reference=bound(original['inventory_receipt']), baseline_reference=bound(prior.BASELINE))
    require(cap['protocol_identity'] == baseline['protocol_identity'], 'Actual joint cap and BF16 full8 protocol changed.')
    value = dict(schema='cat_sequential_parameter_search_v1', output_root=str(Path(output).resolve()),
        rules=rules, rules_file=bound(rules_path), cap_evidence=bound(cap_path), caps=cap['categories'],
        rates=dict(cap['winner']['rate_vector']), run_identity=identity, parent=original['parent'],
        inventory=bound(original['inventory_receipt']), data_plan=bound(original['data_plan']), baseline=bound(prior.BASELINE),
        protocol_identity=baseline['protocol_identity'], source_sha256=pins(), anchors={})
    snapshot=coverage.Snapshot()
    expected,context=coverage.expected_sources(snapshot,coverage.DEFAULT_UP,coverage.DEFAULT_EIGHT,coverage.DEFAULT_NIGHT)
    profiles={(row['category_id'],row['rate']):row for row in expected}
    for category in inventory['category_ids']:
        profile=profiles[(category,value['rates'][category])]
        verified=coverage.verify_complete(snapshot,profile,context)
        path=Path(verified['sources']['canonical']['path'])
        resolved = recipe(inventory, category, value['rates'][category], DEFAULT_SETTINGS)
        row = checked_result(path, value, expected_recipe=resolved)
        source = cap['winner']['bank_sources'][category]
        require(row['candidate_id'] == source['candidate_id'] and row['bank_digest'] == source['bank_digest']
            and row['artifact_evidence'] == source['artifact_evidence'], 'Anchor must be the actual selected joint source bank.')
        value['anchors'][category] = dict(result=bound(path), settings=deepcopy(DEFAULT_SETTINGS), candidate_id=row['candidate_id'],
            role='reused_profile_anchor', actual_small_source_verification=verified)
    maximum = worst_case_GPU_hours(value)
    value.update(maximum_declared_candidate_GPU_hours=maximum,
        scope='Fresh BF16 single-category recipes; per-parameter observed best only, not global or final live-chain optimum',
        failures='A failed announced legal candidate blocks this category successors; independent categories may continue',
        technical_reuse='Original source9 same_resource_path only; new geometry/optimizer/resource paths require technical2',
        source_identity='Existing source9 fit identity retained; actual new orchestration is additionally pinned in each request')
    require(rules['total_GPU_hours'] >= maximum, 'Declared whole-run ceiling cannot reserve the finite worst-case distinct-candidate windows.')
    return dict(value, manifest_identity=digest(value))


def worst_case_GPU_hours(manifest):
    """Enumerate five possible accepted D values; no real HPO recipes are generated."""
    order = manifest['rules']['order']; maximum = 0
    for chosen_dim in (8,16,32,64,128):
        current = deepcopy(DEFAULT_SETTINGS); total = 0
        for parameter in order:
            choices = LIBRARY[parameter] if parameter != 'protection' else ['1/32']
            for value in choices:
                if value == current[parameter]: continue
                settings = dict(current, **{parameter:value})
                geometry = dict(D=settings['dimension'], encoder=dict(hidden_dim=settings['encoder']), decoder=dict(hidden_dim=settings['decoder']))
                total += sum(phase_windows(manifest, geometry, needs_technical=parameter != 'multiplier').values())
            if parameter == 'dimension': current['dimension'] = chosen_dim
            elif parameter in ('encoder', 'decoder'): current[parameter] = 256
        maximum = max(maximum,total)
    return maximum*len(manifest['caps'])/3600


def integrity(manifest):
    verify_files(manifest['source_sha256'])
    for key in ('rules_file', 'cap_evidence', 'inventory', 'data_plan', 'baseline'):
        item = manifest[key]; require(sha(item['path']) == item['sha256'], 'Frozen '+key+' changed.')
    for row in manifest['anchors'].values():
        item = row['result']; require(sha(item['path']) == item['sha256'], 'Frozen reused profile anchor changed.')


def proposals(manifest, category, parameter, current_settings, known):
    """Freeze one parameter's legal explicit candidates; preserve all rejected coverage."""
    inventory = read(manifest['inventory']['path'])['inventory']; cap = manifest['caps'][category]['cap_bits']
    rows = []
    values = LIBRARY[parameter] if parameter != 'protection' else ['1/32']
    for value in values:
        settings = dict(current_settings, **{parameter: value})
        item = dict(parameter=parameter, requested_value=value, settings=settings)
        try:
            resolved = recipe(inventory, category, manifest['rates'][category], settings)
            quote = component_plan(inventory, resolved)
            if parameter == 'protection' and quote['capability']['constructible'] and quote['logical_bits'] > cap:
                item['protection_fallback_reason'] = 'requested1/32 forecast exceeds frozen class cap before fitting'
                settings = dict(settings, protection='0'); item['settings'] = settings
                resolved = recipe(inventory, category, manifest['rates'][category], settings); quote = component_plan(inventory, resolved)
                require(set(resolved['protection']['counts'].values()) == {0}, 'Cancellation must be real full-zero protection.')
            item.update(recipe=resolved, candidate_id=candidate_id(manifest['run_identity'], resolved),
                predicted_logical_bits=quote.get('logical_bits'), predicted_packed_bits=quote.get('packed_bits'), capability=quote['capability'])
            if not quote['capability']['constructible']:
                item.update(status='uncovered_illegal', reason=quote['capability']['reason'])
            elif quote['logical_bits'] > cap:
                item.update(status='uncovered_cap', reason='Actual-component forecast exceeds this fixed class cap; no borrowing or replacement quota')
            elif item['candidate_id'] in known:
                item.update(status='reused', result=known[item['candidate_id']])
            else:
                item['status'] = 'declared_pending'
        except ValueError as error:
            if str(error).startswith(('math_invalid:', 'impl_blocked:')):
                item.update(status='uncovered_illegal', reason=str(error))
            else:
                raise
        rows.append(item)
    return rows


def proof_for(manifest, candidate, candidates_root):
    native = search.batched.native
    paths = list((REPO/'artifacts/cat_night_profiles_cmcc_20261104/night_20261104_1918').glob('gpu*/*/technical_validation/completed.json'))
    paths += list((REPO/'artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759').glob('*/technical_validation/completed.json'))
    paths += list(Path(candidates_root).glob('*/technical_validation/completed.json'))
    inventory = read(manifest['inventory']['path'])['inventory']
    for path in paths:
        exit_path = path.parent/'exit.json'
        if not exit_path.is_file() or read(exit_path).get('exit_code') != 0:
            continue
        completed = read(path)
        try: native._validate_model_path_evidence(completed, inventory, manifest['run_identity'], [candidate])
        except (ValueError, KeyError): continue
        return bound(path)
    return None


def make_request(manifest, candidate, directory, gpu, phase, deadline, proof=None):
    now = time.time(); seconds = phase_windows(manifest,candidate['recipe'])[phase]
    row = {key: candidate[key] for key in ('candidate_id', 'recipe')}
    row.update(purpose='offline', class_cap_bits=manifest['caps'][candidate['recipe']['category_id']]['cap_bits'],
        class_cap_evidence=manifest['cap_evidence'], complete_dependency_gpu_hours=seconds/3600,
        cost_basis=dict(kind='confirmed_finite_sequential_parameter_stage', reference=manifest['rules']['authorization_reference'],
            includes='complete source/teacher/fit/encode/readout/save/fresh stage'))
    request = dict(schema='cat_auto_explicit_candidates_v0', phase=phase, inventory_receipt=manifest['inventory']['path'],
        inventory_receipt_sha256=manifest['inventory']['sha256'], data_plan=manifest['data_plan']['path'], data_plan_sha256=manifest['data_plan']['sha256'],
        run_identity=manifest['run_identity'], parent=manifest['parent'], candidates=[row], output_dir=str(directory/phase),
        absolute_deadline_epoch=min(deadline, now+seconds), authorized_start_epoch=now, authorized_gpu_hours=seconds/3600,
        physical_GPU=gpu, authorization_reference=manifest['rules']['authorization_reference'], source_sha256=manifest['source_sha256'],
        shared_overhead_gpu_hours=0., shared_cost_scope='All overhead within each single-card stage reservation',
        technical_steps=2 if phase == 'technical_validation' else None, required_model_path_evidence=proof,
        full8=False, KD=False, automatic_retry=False, sequential_manifest_identity=manifest['manifest_identity'],
        formal_initialization='Fresh original BF16; only selected hyperparameters are inherited')
    search.validate_request(request)
    return request

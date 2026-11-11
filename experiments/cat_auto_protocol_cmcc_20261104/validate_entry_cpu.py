"""Actual data/request binding and static entry contracts; no GPU worker runs."""
import argparse
from copy import deepcopy
import ast
import json
from pathlib import Path
import time

import torch
from .adapter import file_sha, public_consumer_identity
from .config import resolve_recipe
from .contracts import digest
from .fit_adapter import fit_code_identity
from .identity import candidate_id
from .search import (validate_request, _path_covered, _validate_model_path_evidence,
    _two_step_fit_completed, _completion_scope)


def resource_reuse_checks(identity, inventory, recipe):
    """CPU decision-rule fixtures; none is a model GPU evidence certificate."""
    observed = dict(candidate_id=candidate_id(identity, recipe), fit_report=dict(resolved_recipe=recipe))
    positives = []
    for label, change in (('actual_LR_multiplier', ('optimizer', 'multiplier', 2)),
            ('commitment_coefficient', ('loss', 'commitment_loss_weight', .125)),
            ('entropy_coefficient', ('loss', 'entropy_loss_weight', .02))):
        changed = deepcopy(recipe); changed[change[0]][change[1]] = change[2]
        target = dict(candidate_id=candidate_id(identity, changed), recipe=changed)
        assert target['candidate_id'] != observed['candidate_id'] and _path_covered(target, observed)
        positives.append(label)
    assert _path_covered(dict(candidate_id=observed['candidate_id'], recipe=recipe), observed)
    negatives = []
    variants = [
        ('new_B_and_D', resolve_recipe(inventory, 'up_proj', '2.5', dim=16, optimizer='vae_muon')),
        ('new_network', resolve_recipe(inventory, 'up_proj', '2.5', encoder_width=256, optimizer='vae_muon')),
        ('new_optimizer_algorithm', resolve_recipe(inventory, 'up_proj', '2.5', optimizer='adamw')),
        ('new_transpose', resolve_recipe(inventory, 'up_proj', '2.5', transpose=True, optimizer='vae_muon')),
        ('new_rotation', resolve_recipe(inventory, 'up_proj', '2.5', rotation_level='none', optimizer='vae_muon')),
        ('new_protection', resolve_recipe(inventory, 'up_proj', '2.5', protection_fraction='0', optimizer='vae_muon'))]
    changed = deepcopy(recipe)
    removed = changed['group_signature'].pop(); changed['linear_group_size'] -= 1; changed['protection']['counts'].pop(removed)
    variants.append(('new_M', changed))
    for label, changed in variants:
        assert not _path_covered(dict(candidate_id=candidate_id(identity, changed), recipe=changed), observed)
        negatives.append(label)
    # Reuse actual CPU stage evidence for the step-count predicate only. Its CPU
    # provenance must still fail the independent full GPU-evidence requirement.
    actual_cpu = json.loads((Path(__file__).resolve().parents[2] /
        'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104/fit_cpu_final.json').read_text())
    fitted = actual_cpu['fit']['reports'][0]
    assert _two_step_fit_completed(fitted)
    one = deepcopy(fitted); one['technical_steps'] = one['actual_stage_steps'] = 1
    for stage in one['stages']:
        stage['actual_steps'] = 1
    assert not _two_step_fit_completed(one)
    malformed = deepcopy(fitted); malformed['stages'].pop()
    assert not _two_step_fit_completed(malformed)
    malformed = deepcopy(fitted); malformed['stages'][0]['actual_steps'] = 1
    assert not _two_step_fit_completed(malformed)
    assert _completion_scope('technical_validation', 1).endswith('technical1')
    assert _completion_scope('technical_validation', 2).endswith('technical2')
    try:
        _validate_model_path_evidence(actual_cpu, inventory, identity, [dict(candidate_id=observed['candidate_id'], recipe=recipe)])
    except ValueError:
        pass
    else:
        raise AssertionError('Actual CPU evidence acquired formal GPU eligibility.')
    return dict(scope='CPU_decision_rules_only_not_GPU_evidence', exact_candidate_sufficient=True,
        same_path_reuses=positives, incompatible_paths_rejected=negatives, actual_CPU_evidence_rejected=True,
        one_step_and_incomplete_stage_evidence_rejected=True, one_step_scope_truthful=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inventory', required=True); parser.add_argument('--data-plan', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); began = time.monotonic()
    inventory = json.loads(Path(args.inventory).read_text())['inventory']
    prepared = torch.load(args.data_plan, map_location='cpu', weights_only=False)
    assert digest(prepared['manifest']) == prepared['data_identity']
    assert prepared['manifest']['model_identity'] == inventory['model_identity']
    assert len(prepared['calibration']) == 512 and all(tuple(value.shape) == (1, 256) for value in prepared['calibration'])
    code = fit_code_identity(); consumer = public_consumer_identity()
    identity = dict(model=inventory['model_identity'], parent_uuid='BF16:' + digest(inventory['model_identity']),
        parent_weights_digest=digest(inventory['model_identity']['weight_shards_sha256']), data_identity=prepared['data_identity'],
        code_identity=code, consumer_identity=consumer, search_seed=prepared['manifest']['search_seed'])
    recipe = resolve_recipe(inventory, 'up_proj', '2.5', optimizer='vae_muon')
    sources = {**consumer['source_sha256'], **{str(path.resolve()): file_sha(path) for path in Path(__file__).parent.glob('*.py')}}
    now = time.time()
    request = dict(schema='cat_auto_explicit_candidates_v0', phase='technical_validation', inventory_receipt=args.inventory,
        inventory_receipt_sha256=file_sha(args.inventory), data_plan=args.data_plan, data_plan_sha256=file_sha(args.data_plan),
        run_identity=identity, parent=dict(kind='BF16', path=inventory['source_evidence']['path'], uuid=identity['parent_uuid'], weights_digest=identity['parent_weights_digest']),
        candidates=[dict(candidate_id=candidate_id(identity, recipe), recipe=recipe, purpose='offline', class_cap_bits=10 ** 12, complete_dependency_gpu_hours=.5,
            cost_basis='CPU_request_rule_fixture_not_a_GPU_price_or_authorization')], output_dir='/nonexecuted_CPU_request_fixture',
        absolute_deadline_epoch=now + 3599, authorized_start_epoch=now, authorized_gpu_hours=1., physical_GPU=1,
        authorization_reference='CPU_request_fixture_has_no_actual_execution_authorization', shared_overhead_gpu_hours=.1,
        source_sha256=sources, technical_steps=2, required_model_path_evidence=None)
    validate_request(request)
    rejected = []
    cases = []
    for key, value in (('physical_GPU', 0), ('physical_GPU', 2), ('physical_GPU', 6), ('data_plan_sha256', 'changed'),
            ('authorized_gpu_hours', .1), ('phase', 'formal')):
        bad = deepcopy(request); bad[key] = value; cases.append((key + '=' + str(value), bad))
    bad = deepcopy(request); bad['candidates'] *= 5; cases.append(('no_automatic_full_grid', bad))
    bad = deepcopy(request); bad['parent']['kind'] = 'native_v6'; cases.append(('offline_not_old_live_parent', bad))
    for label, bad in cases:
        try:
            validate_request(bad)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Invalid request accepted: ' + label)
    syntax = []
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); syntax.append(str(path))
    report = dict(exit_code=0, elapsed_seconds=time.monotonic() - began, cuda_initialized=torch.cuda.is_initialized(),
        scope='CPU request/data/static contracts only; no worker launched, no CUDA model/quality evidence', formal_eligible=False,
        actual_data_identity=prepared['data_identity'], actual_calibration_shape=[512, 1, 256],
        fixture_request_validation_only=True, rejected_cases=rejected, parsed_sources=syntax, source_sha256=sources,
        runner_gpu_path='implemented_but_not_model_GPU_executed', authorization=False,
        resource_path_reuse=resource_reuse_checks(identity, inventory, recipe))
    report['elapsed_seconds'] = time.monotonic() - began
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(dict(exit_code=0, elapsed_seconds=report['elapsed_seconds'], cuda_initialized=report['cuda_initialized']), flush=True)


if __name__ == '__main__':
    main()

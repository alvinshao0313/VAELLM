"""Real tiny native S2 fit comparison; never formal model or CUDA evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104 import fit_adapter as native
from experiments.cat_auto_protocol_cmcc_20261104.config import resolve_recipe
from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
from experiments.cat_auto_protocol_cmcc_20261104.validate_fit_cpu import tiny_fixture
from .fit import fit_category, fit_code_identity, _validate_identity


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new CPU fit verification output.')
    started = time.perf_counter()
    torch.set_num_threads(2)
    model, inventory, identity, calibration = tiny_fixture()
    old_identity = deepcopy(identity)
    identity['code_identity'] = fit_code_identity()
    try:
        _validate_identity(old_identity)
    except ValueError:
        pass
    else:
        raise AssertionError('The new fit accepted an identity omitting its changed sources.')
    recipe = resolve_recipe(inventory, 'up_proj', '2', dim=32, rotation_level='32', optimizer='adamw')
    original_globals = dict(native.fit_category.__globals__)
    results = []
    before_rng = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory(prefix='cat_batch_gather_fit_CPU_') as temporary:
        for label, function in (('native', native.fit_category), ('batched', fit_category)):
            torch.set_rng_state(before_rng)
            result = function(deepcopy(model), inventory=inventory, recipe=recipe,
                run_identity=identity, calibration_bundle=calibration, device='cpu',
                output_dir=Path(temporary) / label, technical_steps=2)
            report = result['fit_report']
            assert report['formal_eligible'] is False and report['technical_steps'] == 2
            assert all(stage['actual_steps'] == 2 for stage in report['stages'])
            assert report['frozen_prefix_before_digest'] == report['frozen_prefix_after_digest']
            updates = [stage['updates'] for stage in report['stages']]
            assert all([u['actual_learning_rates'] for u in stage] == [[.003], [.0029997]] for stage in updates)
            results.append(dict(state=tensor_digest(result['model'].state_dict()),
                bank=result['category_bank']['tensor_digest'],
                target_before=report['target_before_digest'], target_after=report['target_after_digest'],
                stage_initializations=[stage['initialization_digest'] for stage in report['stages']],
                stage_seeds=[{key: stage[key] for key in ('init_seed', 'shuffle_seed')} for stage in report['stages']],
                actual_updates=updates, host_rng=torch.get_rng_state().tolist(),
                formal_eligible=report['formal_eligible']))
        assert results[0] == results[1], 'Native versus batched tiny fits differ.'
    assert not torch.cuda.is_initialized()
    assert all(native.fit_category.__globals__[key] is value for key, value in original_globals.items())
    assert identity['code_identity'] == fit_code_identity()
    report = dict(status='CPU_actual_native_S2_fit_batched_equivalence_verified', exit_code=0,
        elapsed_seconds=time.perf_counter() - started, CUDA_initialized=False,
        scope='Real tiny Qwen3 CPU S2x2 fit only; not 8B shape, GPU throughput or formal profile',
        exact_state_packed_bank_stage_seeds_initializations_updates_and_host_RNG=True,
        unchanged_shared_native_globals=True, missing_new_source_identity_rejected=True,
        formal_eligible=False, fit_code_identity=fit_code_identity(),
        source_sha256={str(Path(__file__).resolve()): source_sha(__file__)},
        actual_comparison={key: value for key, value in results[1].items() if key != 'host_rng'})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seconds=report['elapsed_seconds'], CUDA_initialized=False)))


if __name__ == '__main__':
    main()

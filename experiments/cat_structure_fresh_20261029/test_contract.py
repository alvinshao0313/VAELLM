"""CPU contracts: real parsed recipes and real saved metadata, no fake model/save."""
from copy import deepcopy
from pathlib import Path

import pytest

import fresh_support as s
import recipe
import run
from validate_cpu import EVENT_META, compare_template

HF = Path('/home/star/models/Qwen/Qwen3-8B')


@pytest.fixture(scope='module')
def plans():
    return {arm: recipe.plan(arm, HF, Path('/not_executed') / arm) for arm in ('A', 'B')}


@pytest.mark.parametrize('arm', ('A', 'B'))
def test_seven_resolved_categories_original_recovery_and_exposure(plans, arm):
    plan = plans[arm]
    assert list(plan['category_configs']) == list(s.CATEGORIES)
    compare_template(arm, plan)
    assert '--resume_from_checkpoint' not in plan['native_argv']
    for index, name in enumerate(s.CATEGORIES):
        cfg = plan['category_configs'][name]
        recovery = plan['recovery_configs'][name]
        dim = 64 if arm == 'B' and name == 'gate_proj' else 32
        assert cfg['codebook_dim'] == dim and cfg['residual_stages'] == 2 and cfg['steps'] == 10000
        assert cfg['weight_rotation_block_size'] == (0 if index < 2 else 32)
        assert cfg['decoder_type'] == ('asymmetric' if name == 'gate_proj' else 'symmetric')
        assert s.policy(arm)['category_vector_batches'][name] * dim == 262144
        assert s.policy(arm)['category_shuffle_seeds'][name] == 31 + 100000 * index
        assert recovery['opt']['steps'] == 500 and recovery['opt']['batch_size'] == 4
        assert recovery['data']['model_max_length'] == 1024 and recovery['data']['dynamic_padding']
        assert recovery['lora']['rank'] == 12 and recovery['lora']['alpha'] == 24
    assert s.policy(arm)['category_vector_batches']['up_proj'] == s.policy(arm)['category_vector_batches']['down_proj'] == 8192


@pytest.mark.parametrize('arm', ('A', 'B'))
def test_restart_keeps_identity_and_explicitly_loads_only_handoff(plans, arm):
    # Parsing a future path is a config-only check, not a generated checkpoint.
    handoff = Path('/not_executed') / arm / 'prefix/native/new/after_o_proj'
    values = recipe.argv(arm, HF, Path('/not_executed') / arm / 'suffix/native', after_o=handoff)
    cat, _, training, _, _, identity, _ = recipe.parse(arm, values)
    assert cat.resume_from_checkpoint == str(handoff) and cat.seed == training.seed == 31
    assert identity == plans[arm]['runtime_identity']
    assert cat.compression_categories == ','.join(s.CATEGORIES)
    for name in s.CATEGORIES[:4]:
        assert cat.resolve_after_category_config(name).lora.rank == 12


def test_controller_uses_two_sequential_native_process_commands():
    request = Path('/not_executed/request.json')
    first = run.worker_command(request, 'prefix')
    second = run.worker_command(request, 'suffix')
    assert first[:-1] == second[:-1] and first[-1] == 'prefix' and second[-1] == 'suffix'
    assert first[0] == str(s.PYTHON) and first[3] == str(s.HERE / 'worker.py')
    assert s.worker_environment(3)['PYTHONHASHSEED'] == s.worker_environment(4)['PYTHONHASHSEED'] == '31'
    assert s.worker_environment(4)['PYTHONPATH'] == str(s.SNAPSHOT)


def test_worker_and_actual_native_imports_resolve_to_N():
    import worker
    from train_utils import cat_train_pipeline as native
    s.enable_snapshot()
    assert Path(worker.__file__).resolve() == s.HERE / 'worker.py'
    assert Path(native.__file__).resolve().is_relative_to(s.SNAPSHOT)
    assert Path(native.train_group_vae_payload.__code__.co_filename).resolve().is_relative_to(s.SNAPSHOT)
    import torch
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('arm', ('A', 'B'))
def test_identity_matches_actual_pipeline_builder_normalization(plans, arm):
    from train_utils import cat_train_pipeline as native
    cat, _, _, vae, cfgs, expected, _ = recipe.parse(arm, plans[arm]['native_argv'])
    actual = native.build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs,
        compression_categories=native.parse_compression_categories(cat.compression_categories),
        target_layers=native.parse_target_layers(cat.target_layers),
        skip_layers=native.parse_skip_layers(cat.skip_layers),
        transpose_modules=native._split_csv(cat.transpose_modules))
    assert s.with_policy(actual, arm) == expected
    assert expected['target_layers'] == 'all'


def test_real_saved_after_o_is_recognized_as_complete_only():
    s.validate_after_o_meta(s.read(EVENT_META), HF)


@pytest.mark.parametrize('change', ('kind', 'completed', 'all_seven', 'finalization', 'did_train', 'future_scope'))
def test_boundary_rejects_pre_save_or_partial_recovery_contract(change):
    meta = deepcopy(s.read(EVENT_META))
    if change == 'kind':
        meta['checkpoint_kind'] = 'round_base'
    elif change == 'completed':
        meta['completed_categories'] = list(s.CATEGORIES[:3])
    elif change == 'all_seven':
        meta['compression_categories'] = list(s.CATEGORIES[:4])
    elif change == 'finalization':
        meta['finalized_status']['lora_finalized'] = False
    elif change == 'did_train':
        meta['extra_meta']['distill_stage_history'][3]['did_train'] = False
    else:
        meta['extra_meta']['distill_stage']['remaining_lora_target_count'] = 0
    with pytest.raises(ValueError):
        s.validate_after_o_meta(meta, HF)


def test_foreign_real_checkpoint_cannot_be_used_as_fresh_suffix_input():
    with pytest.raises(ValueError):
        s.prefix_input_path(Path('/not_executed'), EVENT_META.parent)

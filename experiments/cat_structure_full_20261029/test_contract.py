"""Real parent recipe checks; negative cases protect completed geometry and scope."""
from copy import deepcopy

import pytest

import common as c
import recipe


@pytest.fixture(scope='module')
def plans():
    _, row, _ = c.source_binding()
    runtime = c.fork.load_v6_cat_runtime_state(str(c.SOURCE), map_location='cpu')
    values = {}
    for arm in recipe.ARMS:
        argv = recipe.training_argv(arm, row)
        values[arm] = recipe.parse_recipe(arm, argv, runtime)
    return runtime, values


@pytest.mark.parametrize('arm', ('A', 'B'))
def test_real_future_gate_recipe_preserves_parent_and_recovery(plans, arm):
    runtime, values = plans
    cat, _, _, _, cfgs, identity, normalized = values[arm]
    changes = recipe.validate_fork(runtime['runtime_identity'], identity, arm)
    assert changes and all(key.startswith('gate_proj.') for key in changes)
    assert identity['experiment_runtime'] == normalized['experiment_runtime'] == recipe.policy(arm)
    assert cfgs['gate_proj'].decoder_type == 'asymmetric'
    assert cfgs['up_proj'].decoder_type == cfgs['down_proj'].decoder_type == 'symmetric'
    for category in c.CATEGORIES[4:]:
        recovery = cat.resolve_after_category_config(category)
        assert recovery.opt.steps == 500 and recovery.opt.batch_size == 4
        assert recovery.opt.learning_rate == 1e-4 and recovery.opt.decoder_lr == 1e-5
        assert recovery.loss.top_k == 100 and recovery.loss.prompt_loss_weight == 0.3
        assert recovery.lora.rank == 12 and recovery.aux.residual_lora_mode == 'none'
        assert cfgs[category].codebook_dim * recipe.policy(arm)['category_vector_batches'][category] == 262144


@pytest.mark.parametrize('category,field,value', [
    ('q_proj', 'codebook_bits', 64), ('o_proj', 'codebook_dim', 64),
    ('up_proj', 'codebook_dim', 64), ('down_proj', 'decoder_type', 'asymmetric'),
    ('gate_proj', 'norm_type', 'rms'), ('gate_proj', 'steps', 2),
])
def test_fork_rejects_unrelated_or_completed_changes(plans, category, field, value):
    runtime, values = plans
    trial = deepcopy(values['B'][5])
    trial['resolved_category_runtime'][category][field] = value
    with pytest.raises(ValueError):
        recipe.validate_fork(runtime['runtime_identity'], trial, 'B')


def test_batch_contract_cannot_silently_reduce_B_successors(plans):
    runtime, values = plans
    trial = deepcopy(values['B'][5])
    trial['experiment_runtime']['category_vector_batches']['up_proj'] = 4096
    with pytest.raises(ValueError):
        recipe.validate_fork(runtime['runtime_identity'], trial, 'B')
    assert values['A'][5]['experiment_runtime']['category_shuffle_seeds'] == {
        'gate_proj': 400031, 'up_proj': 500031, 'down_proj': 600031}


def test_original_exact_resume_rejects_new_experiment_identity(plans):
    runtime, values = plans
    with pytest.raises(ValueError):
        c.fork.validate_cat_runtime_identity(runtime['runtime_identity'], values['B'][5])

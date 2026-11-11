"""Resolve every declared category setting; no positional/default semantic changes."""
from copy import deepcopy
from fractions import Fraction

from .adapter import REPO, category_projections, file_sha
from .contracts import FIXED_VAE_LOSS, require, validate_recipe
from .geometry import DIMENSIONS, ROTATIONS, map_protection, rotation

FIXED_LOSS = FIXED_VAE_LOSS


def resolve_recipe(inventory, category_id, rate, *, dim=32, stages=2, transpose=False, rotation_level='32',
        protection_fraction='1/64', protection_axis='input', encoder_width=128, decoder_width=128,
        optimizer='adamw', multiplier=1, protection_counts=None, per_layer_rotation=None, loss_overrides=None):
    require(dim in DIMENSIONS and stages in (2, 4) and str(rotation_level) in ROTATIONS, 'Use only registered D/S/rotation levels.')
    rate = Fraction(str(rate))
    bits = rate * dim / stages
    require(bits.denominator == 1 and bits > 0, 'math_invalid: K=rD/S must be a positive integer; no rounding.')
    projections = category_projections(inventory, category_id)
    names = [row['actual_name'] for row in projections]
    counts, failures = map_protection(projections, dim, protection_axis, str(rotation_level), protection_fraction)
    if protection_counts is not None:
        require(set(protection_counts) == set(names), 'Explicit counts must resolve every actual projection.')
        counts, failures = dict(protection_counts), {}
    require(not failures, 'impl_blocked: no joint legal protection mapping: ' + str(failures))
    loss = deepcopy(FIXED_LOSS)
    if loss_overrides:
        require(set(loss_overrides) <= {'commitment_loss_weight', 'entropy_loss_weight'}, 'Only the registered conditional BSQ coefficient is variable.')
        loss.update(loss_overrides)
    rot = rotation(str(rotation_level))
    if per_layer_rotation is not None:
        require(set(per_layer_rotation) == set(names) and all(str(value) in ROTATIONS for value in per_layer_rotation.values()), 'Per-layer rotation must name every actual layer.')
        rot['per_layer'] = {name: str(value) for name, value in per_layer_rotation.items()}
    if isinstance(transpose, dict):
        require(set(transpose) == set(names) and all(isinstance(value, bool) for value in transpose.values()), 'Per-layer transpose must name every actual layer.')
    else:
        require(isinstance(transpose, bool), 'Transpose cannot be inferred or cast from strings.')
    result = dict(category_id=category_id, nominal_main_rate=str(rate), transpose=transpose, rotation=rot,
        D=dim, K=int(bits), S=stages, B=262144 // dim,
        encoder=dict(type='asymmetric', hidden_dim=encoder_width, num_res_blocks=0),
        decoder=dict(type='asymmetric', hidden_dim=decoder_width, num_res_blocks=0),
        normalization=dict(normalize_weight=True, new_quant=True, norm_type='layer', activation_type='swish', parameter_dtype='bfloat16', compute_dtype='bfloat16'),
        protection=dict(axis=protection_axis, format='int8', counts=counts),
        optimizer=dict(kind=optimizer, multiplier=multiplier, base_lr=.003, betas=[.9, .95], weight_decay=0.,
            eps=1e-8,
            kappa_contract=dict(version='fixed_reference_Adam003_before_actual_multiplier_v0', reference_lr=.003,
                applicable=optimizer == 'vae_muon',
                reference_implementation=dict(path='experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py',
                    sha256=file_sha(REPO / 'experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py'), schema='cat_down_vae_muon_bf16_v1'),
                owner_layout=dict(M=len(names), D=dim, K=int(bits), encoder_hidden=encoder_width, decoder_hidden=decoder_width,
                    roles=['encoder.linear_in', 'encoder.linear_out', 'decoder.linear_in', 'decoder.linear_out'], group_signature=names),
                precision=dict(parameters='bfloat16', gradients='bfloat16', Muon_matrix_momentum='float32',
                    Muon_NS='bfloat16', auxiliary_Adam_moments='bfloat16', reference_Adam_moments='bfloat16'),
                Muon=dict(momentum=.95, nesterov=True, NS_steps=5, NS_coefficients=[3.4445, -4.7750, 2.0315],
                    normalization_epsilon=1e-7, direction_scale='.2*sqrt(max(rows,cols))'),
                reference_metric='Actual one-cast BF16 matrix-parameter displacement norm',
                maximum_bracket_doublings=2, bisections=12, relative_update_tolerance=.01,
                actual_multiplier_after_reference_calibration=True, seed_rule='semantic_uint32_model_category_stage_purpose_v0')),
        loss=loss, steps=10000, schedule='linear', warmup=0, linear_group_size=len(names), group_signature=names,
        pooled_BSQ='Original marginal entropy across B and all actual category M; never splitM for speed')
    return validate_recipe(result)

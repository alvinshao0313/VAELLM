"""Exact protected geometry through real rotation/Decoder/bitpack components."""
from fractions import Fraction
from functools import lru_cache
from types import SimpleNamespace

from .contracts import capability, require, validate_recipe
from .adapter import category_projections, public_consumer_identity

ROTATIONS = ('none', '32', '64', '128', '256', 'full')
DIMENSIONS = (8, 16, 32, 64, 128)


def rotation(kind):
    require(str(kind) in ROTATIONS, 'Unknown registered rotation level.')
    return dict(kind='none' if str(kind) == 'none' else 'two_sided', block_size=0 if str(kind) in ('none', 'full') else int(kind), level=str(kind))


@lru_cache(maxsize=4096)
def _shape(out, inp, dim, count, axis, level):
    if min(out, inp, dim) < 1 or count < 0 or axis not in ('input', 'output'):
        return capability('math_invalid', 'Nonpositive geometry or invalid protection axis/count.')
    width = inp if axis == 'input' else out
    if count >= width:
        return capability('math_invalid', 'The protected axis leaves no positive main matrix.')
    co, ci = (out, inp - count) if axis == 'input' else (out - count, inp)
    if co * ci % dim:
        return capability('math_invalid', 'Complete D-vectors do not cover the actual protected matrix; no padding.')
    if count % dim:
        return capability('impl_blocked', 'Native protection/audit requires codebook-aligned channel count.')
    spec = rotation(level)
    if spec['kind'] != 'none':
        block = spec['block_size']
        if block and (co % block or ci % block):
            return capability('math_invalid', 'The chosen exact block does not divide both actual unprotected axes.')
        from rotation.weight_preprocess import build_weight_rotation
        try:
            transform = build_weight_rotation(co, ci, dict(type='two_sided_hadamard', version=1, block_size=block, seed=0))
            transform.validate_state()
        except ValueError as error:
            return capability('impl_blocked', 'Actual exact Hadamard consumer rejected this full/block factor: ' + str(error))
    return dict(**capability('constructible', 'Actual protected dimensions, exact Hadamard constructor and D-coverage accepted.', verification_scope='CPU_components'),
        compressed_shape=[co, ci], n_vectors=co * ci // dim, vector_padding=0)


def legal_protection_counts(projection, dim, axis, level):
    out, inp = projection['shape']
    width = inp if axis == 'input' else out
    return [count for count in range(0, width, dim) if _shape(out, inp, dim, count, axis, str(level))['constructible']]


def map_protection(projections, dim, axis, level, fraction):
    fraction = Fraction(str(fraction))
    require(fraction in (Fraction(0), Fraction(1, 64), Fraction(1, 32)), 'Use registered protection fractions only.')
    counts, failures = {}, {}
    for row in projections:
        choices = legal_protection_counts(row, dim, axis, level)
        if not choices:
            failures[row['actual_name']] = dict(axis=axis, level=level, reason='No joint legal count for actual protected geometry/consumer.')
            continue
        width = row['shape'][1 if axis == 'input' else 0]
        counts[row['actual_name']] = min(choices, key=lambda count: (abs(Fraction(count) - fraction * width), count))
    return counts, failures


def architecture_args(recipe):
    return SimpleNamespace(codebook_dim=recipe['D'], codebook_bits=recipe['K'], base_ch=recipe['encoder']['hidden_dim'],
        decoder_base_ch=recipe['decoder']['hidden_dim'], num_res_blocks=0, decoder_num_res_blocks=0,
        decoder_type='asymmetric', norm_type='layer', activation_type='swish', vae_decoder_checkpoint=True)


def component_plan(inventory, recipe):
    import torch
    from litebsq.autoencoder import Decoder
    from litebsq.bitpack import build_bitpack_u8_spec
    from litebsq.vae_args import resolve_autoencoder_arch_spec

    recipe = validate_recipe(recipe)
    public_consumer_identity()
    require(recipe['D'] in DIMENSIONS and recipe['rotation']['level'] in ROTATIONS, 'Unregistered geometry level.')
    require(recipe['normalization'] == dict(normalize_weight=True, new_quant=True, norm_type='layer', activation_type='swish', parameter_dtype='bfloat16', compute_dtype='bfloat16'), 'Fixed v0 normalization/precision contract changed.')
    require(recipe['encoder']['type'] == recipe['decoder']['type'] == 'asymmetric'
        and recipe['encoder']['hidden_dim'] in (128, 256) and recipe['decoder']['hidden_dim'] in (128, 256)
        and recipe['encoder']['num_res_blocks'] == recipe['decoder']['num_res_blocks'] == 0, 'Unsupported v0 VAE architecture.')
    arch = resolve_autoencoder_arch_spec(architecture_args(recipe))
    with torch.device('meta'):
        decoder = Decoder(in_dim=arch.codebook_bits, out_dim=arch.codebook_dim, hidden_dim=arch.decoder_hidden_dim,
            num_res_blocks=0, decoder_type=arch.decoder_type, norm_type=arch.norm_type, activation_type=arch.activation_type).to(dtype=torch.bfloat16)
    decoder_bytes = sum(value.numel() * value.element_size() for value in decoder.state_dict().values())
    rows, failures = [], []
    for projection in category_projections(inventory, recipe['category_id']):
        name = projection['actual_name']
        transpose = recipe['transpose'] if isinstance(recipe['transpose'], bool) else recipe['transpose'][name]
        level = recipe['rotation'].get('per_layer', {}).get(name, recipe['rotation']['level'])
        check = _shape(*projection['shape'], recipe['D'], recipe['protection']['counts'][name], recipe['protection']['axis'], level)
        if not check['constructible']:
            failures.append(dict(name=name, **check))
            continue
        packed = build_bitpack_u8_spec(logical_shape=(check['n_vectors'], 1, recipe['K']))
        count = recipe['protection']['counts'][name]
        other = projection['shape'][0 if recipe['protection']['axis'] == 'input' else 1]
        co, ci = check['compressed_shape']
        fees = dict(code_logical_bits=check['n_vectors'] * recipe['S'] * recipe['K'],
            code_packed_bits=check['n_vectors'] * recipe['S'] * ((recipe['K'] + 7) // 8) * 8,
            decoder_bits=decoder_bytes * recipe['S'] * 8, protected_weights_bits=count * other * 8,
            protected_indices_bits=count * 64, protected_scales_bits=count * 16,
            rotation_bits=0 if level == 'none' else (co + ci) * 8,
            bias_bits=projection.get('bias_numel', 0) * 16)
        rows.append(dict(name=name, shape=projection['shape'], compressed_shape=check['compressed_shape'],
            D=recipe['D'], K=recipe['K'], S=recipe['S'], transpose=transpose, rotation_level=level,
            n_vectors=check['n_vectors'], packed_spec=packed, decoder_bytes_per_stage=decoder_bytes,
            decoder_tensor_shapes={key: list(value.shape) for key, value in decoder.state_dict().items()}, fees=fees,
            logical_bits=fees['code_logical_bits'] + sum(value for key, value in fees.items() if key not in ('code_logical_bits', 'code_packed_bits')),
            packed_bits=fees['code_packed_bits'] + sum(value for key, value in fees.items() if key not in ('code_logical_bits', 'code_packed_bits'))))
    if failures:
        status = 'math_invalid' if any(item['status'] == 'math_invalid' for item in failures) else 'impl_blocked'
        return dict(capability=capability(status, 'At least one actual projection cannot use this declared geometry.', evidence=failures), rows=rows, failures=failures)
    return dict(capability=capability('constructible', 'Every actual layer has real factory/rotation/pack-shape support; semantic execution is separately gated.', verification_scope='CPU_components'),
        rows=rows, failures=[], logical_bits=sum(row['logical_bits'] for row in rows), packed_bits=sum(row['packed_bits'] for row in rows),
        measurement='prediction_from_actual_Decoder_and_bitpack_and_Hadamard_components_not_checkpoint',
        directory_bytes=None, assumptions=['BF16 decoder deployment; normalized statistics fused once', 'Single-axisINT8 and actual protected dimensions', 'No padding or dense backups; actual artifact audit still required'])

"""CPU contracts; real A/B banks, small decoder reads, no quality experiment."""
import io
import itertools
import json
from pathlib import Path

import pytest
import torch

from . import allocation as a, native_prefix as p, objective as o
from .codec import decode_lengths
from .consumer import CompactRVQLinear
from .spec import resolve
from .validate_native_prefix import compact_selected
from experiments.adaptive_vector_rvq.entropy_budget import allocate as row_allocate
from rotation.weight_preprocess import _transform_last, build_weight_rotation


ARCHIVES = Path('/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes')
SOURCE_IDS = {'A': '94c8c25b-19dd-4c6d-852c-19f5b12c4d6b',
              'B': '7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f'}


@pytest.fixture(scope='module', params=('A', 'B'))
def real_prefix(request):
    torch.set_num_threads(4)
    directory = ARCHIVES / request.param
    if not directory.is_dir():
        pytest.skip('Requires the retained actual A/B raw-gate partial archives.')
    meta = json.loads((directory / 'raw_checkpoint_meta.json').read_text())
    assert meta['checkpoint_id'] == SOURCE_IDS[request.param]
    spec = next(item for item in meta['converted_modules'] if item['name'] == 'model.layers.0.mlp.gate_proj')
    state = torch.load(directory / 'gate_state.pt', map_location='cpu', weights_only=True, mmap=True)
    relative = {name[len(spec['name']) + 1:]: tensor for name, tensor in state.items() if name.startswith(spec['name'] + '.')}
    bundle, lengths, resolved = p.from_native(spec, relative,
        dict(checkpoint_id=meta['checkpoint_id'], source_module=spec['name'], use='CPU contracts; not a quality observation'))
    expected = 32 if request.param == 'A' else 64
    assert resolved['vector_dim'] == expected
    assert [stage['decoder_spec']['in_dim'] for stage in bundle['stages']] == [expected, expected]
    print(dict(source=request.param, checkpoint_id=meta['checkpoint_id'], vector_dim=expected,
               rows=bundle['n_vectors'], scope='Actual complete banks; only 17 decoder rows read'), flush=True)
    return bundle, lengths, resolved


def serialize(candidate, persistent_fields, metadata=''):
    stream = io.BytesIO()
    lengths = decode_lengths(candidate['length_frame'], candidate['n_vectors'])
    torch.save(dict(schema=p.SCHEMA, bundle=candidate, resolved=resolve(candidate, lengths),
                    persistent_fields=persistent_fields, container_metadata=metadata), stream)
    return stream.getvalue()


def test_actual_bf16_prefixes_and_coordinate_errors(real_prefix):
    bundle, lengths, resolved = real_prefix
    start, stop = 113, 130
    prefixes = o.decode_prefixes(bundle, start=start, stop=stop, chunk=8)
    layer = CompactRVQLinear(bundle, lengths)
    oracle = compact_selected(layer, torch.arange(start, stop))
    assert torch.equal(prefixes[0], oracle['stage0'])
    assert torch.equal(prefixes[1], oracle['sum'])
    # A deterministic mathematical target tests the layout/error formula; it is
    # not a substitute live checkpoint and its errors are not quality evidence.
    dim = resolved['vector_dim']
    target = torch.arange((stop - start) * dim, dtype=torch.float32).reshape(stop - start, dim) / 1024
    h = torch.arange(1, resolved['geometry']['compressed_shape'][1] + 1, dtype=torch.float64)
    h /= h.mean()
    variance = .37
    for transpose in (False, True):
        geometry = {**resolved['geometry'], 'transpose': transpose}
        errors = o.prefix_errors(target, prefixes, h, geometry, variance, start=start, chunk=5)
        out_features, in_features = geometry['compressed_shape']
        coordinates = torch.arange(start * dim, stop * dim).reshape(stop - start, dim)
        input_ids = coordinates // out_features if transpose else coordinates % in_features
        expected = torch.stack([((target - prefix.float()).double().square() * h[input_ids]).sum(1)
                                / (dim * variance) for prefix in prefixes], dim=1)
        assert torch.equal(errors, expected)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('transpose', (False, True))
def test_layout_matches_full_matrix_weighted_reduction(transpose):
    # This is an arithmetic coordinate fixture, not a deployed-model geometry.
    shape, dim = (6, 10), 4
    matrix = torch.arange(60, dtype=torch.float32).reshape(shape) / 13
    deployed = (matrix / 3).to(torch.bfloat16)
    target = (matrix.t().contiguous() if transpose else matrix).reshape(-1, dim)
    prefix = (deployed.t().contiguous() if transpose else deployed).reshape(-1, dim)
    h = torch.arange(1, 11, dtype=torch.float64)
    h /= h.mean()
    errors = o.prefix_errors(target, [prefix], h, dict(compressed_shape=shape, transpose=transpose), .23)
    matrix_total = ((matrix - deployed.float()).double().square() * h[None, :]).sum() / .23
    torch.testing.assert_close(errors.sum() * dim, matrix_total, rtol=1e-15, atol=1e-12)
    # Non-transpose vectors cross input rows; h cannot be averaged before squaring.
    if not transpose:
        weights = h[torch.arange(60).reshape(-1, dim) % shape[1]]
        wrong = (target - prefix.float()).double().square().mean(1) * weights.mean(1) / .23
        assert not torch.allclose(errors[:, 0], wrong, rtol=1e-4, atol=1e-8)


def test_input_energy_uses_retained_coordinates_and_forward_rotation(real_prefix):
    bundle, _, resolved = real_prefix
    # Numerical input rows only; no calibration split/data-source claim.
    columns = resolved['geometry']['original_shape'][1]
    inputs = torch.arange(3 * columns, dtype=torch.float32).reshape(3, columns).sin().to(torch.bfloat16)
    actual = o.input_energy(bundle, inputs, chunk=2)
    keep = torch.ones(columns, dtype=torch.bool)
    keep[bundle['fixed']['protected_input_indices']] = False
    rotation = build_weight_rotation(*resolved['geometry']['compressed_shape'], resolved['geometry']['rotation'])
    rotation.input_signs.copy_(bundle['fixed']['rotation_input_signs'])
    rotated = _transform_last(inputs[:, keep].float() * rotation.input_signs.float(), block_size=rotation.input_block,
        factor=rotation.input_factor, had=rotation.input_had, inverse=False)
    expected = rotated.double().square().mean(0)
    expected /= expected.mean()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_lambda_same_k_retains_existing_row_solution():
    errors = torch.tensor([[9., 3., 0.], [8., 2., 1.], [7., 6., 2.], [4., 3., 2.]], dtype=torch.float64)
    expected, old = row_allocate(errors, 7)
    for bits in (32, 64):
        lengths, report = a.allocate(errors, 7 * ((bits + 7) // 8), [bits] * 3)
        assert torch.equal(lengths, expected)
        assert report['code_rows'] == old['code_rows']
        assert report['logical_code_bits'] == report['code_rows'] * bits
        assert report['code_bytes'] * 8 == report['logical_code_bits']


def test_unequal_k_cost_math_matches_exact_small_enumeration():
    # Unequal-K allocation math only, not an unequal-K trained/deployed bundle.
    errors = torch.tensor([[4., 1., .5], [3., 2.5, 0.], [2., 1.8, 1.]], dtype=torch.float64)
    bits, cap = [5, 9, 17], 8
    lengths, report = a.allocate(errors, cap, bits)
    costs = torch.tensor([1, 3, 6], dtype=torch.int64)
    choices = list(itertools.product(range(3), repeat=3))
    feasible = [(sum(float(errors[index, stage]) for index, stage in enumerate(choice)), choice)
                for choice in choices if sum(int(costs[stage]) for stage in choice) <= cap]
    assert report['NMSE'] * len(errors) >= min(value for value, _ in feasible) - 1e-12
    values = errors + report['lambda_value'] * costs
    assert torch.equal(lengths.long() - 1, values.argmin(1))
    assert report['code_bytes'] <= cap
    assert report['logical_code_bits'] == sum(sum(bits[:int(length)]) for length in lengths)
    assert report['code_padding_bits'] == report['code_bytes'] * 8 - report['logical_code_bits'] > 0
    assert report['fixed_byte_budget_dual_NMSE'] <= min(value for value, _ in feasible) / len(errors) + 1e-12


def test_real_complete_serializer_costs_and_last_feasible(real_prefix):
    bundle, _, _ = real_prefix
    n = bundle['n_vectors']
    # Mathematical scores deliberately prefer stage1, avoiding a full-layer
    # numerical decode or a 60-bisection scan. They make no quality assertion.
    errors = torch.tensor([1., 2.], dtype=torch.float64).repeat(n, 1)
    extra = {'extra/consumer_revision': torch.tensor([1], dtype=torch.int64)}
    all1 = torch.ones(n, dtype=torch.uint8)
    candidate = a.materialize(bundle, all1)
    measured = a._measure(candidate, all1, serialize, extra)
    larger = a._measure(candidate, all1, lambda item, fields: serialize(item, fields, 'x' * 257), extra)
    assert larger['actual_complete_bytes'] > measured['actual_complete_bytes']
    assert larger['payload_tensor_bytes'] == measured['payload_tensor_bytes']
    assert measured['decoder_bytes'] == sum(value.numel() * value.element_size()
        for stage in bundle['stages'] for value in stage['decoder_state'].values())
    assert measured['extra_persistent_tensor_bytes'] == 8
    assert measured['stage_active_rows'] == [n, 0]
    cap = measured['actual_complete_bytes'] + (n + 3) // 4 + 1024
    lengths, chosen, report = a.iterate(errors, bundle, cap, serialize, persistent_fields=extra)
    feasible = [row for row in report['rounds'] if row['feasible']]
    assert len(feasible) >= 2
    assert report['chosen_round'] == max(row['round'] for row in feasible)
    assert report['chosen_round'] > feasible[0]['round']  # equal-quality rounds still take the last
    assert report['chosen']['actual_complete_bytes'] == len(serialize(chosen, extra)) <= cap
    assert report['actual_decoder_stages'] == 2 and report['reserved_unmaterialized_decoders'] == 0
    assert all(row['measured']['decoder_bytes'] == measured['decoder_bytes'] for row in report['rounds'])
    assert torch.equal(lengths, all1)
    # Raw length bytes are larger than real zlib here; this cap must still pass.
    _, _, tight = a.iterate(errors, bundle, measured['actual_complete_bytes'], serialize, persistent_fields=extra)
    assert tight['chosen']['actual_complete_bytes'] <= measured['actual_complete_bytes']
    with pytest.raises(ValueError, match='No feasible candidate'):
        a.iterate(errors, bundle, 1, serialize, persistent_fields=extra)
    for index, stage in enumerate(chosen['stages']):
        assert stage['decoder_state'] is bundle['stages'][index]['decoder_state']
    print(dict(serializer_contract=dict(K=measured['stage_bits'], measured=measured['actual_complete_bytes'],
        chosen_round=report['chosen_round'], rounds=len(report['rounds']), source_rows=n,
        scope='Complete actual source-bank costs and mathematical scores only; no quality measurement')), flush=True)


def test_invalid_inputs_are_rejected(real_prefix):
    bundle, _, _ = real_prefix
    with pytest.raises(ValueError, match='stage1'):
        a.allocate(torch.ones(2, 2, dtype=torch.float64), 7, [32, 32])
    with pytest.raises(ValueError, match='logical K'):
        a.allocate(torch.ones(2, 2, dtype=torch.float64), 16, [32])
    with pytest.raises(ValueError, match='finite nonnegative'):
        a.allocate(torch.tensor([[float('nan'), 0.]], dtype=torch.float64), 8, [32, 32])
    with pytest.raises(ValueError, match='actual prefix row'):
        o.decode_prefixes(bundle, start=3, stop=2)
    bad = {**bundle, 'n_vectors': 7}
    with pytest.raises(ValueError, match='identity disagrees'):
        a.materialize(bad, torch.ones(7, dtype=torch.uint8))

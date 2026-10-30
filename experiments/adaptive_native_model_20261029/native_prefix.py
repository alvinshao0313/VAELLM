"""Import one actual deployed native two-stage prefix without refitting or fusion."""
from copy import deepcopy
import json
from pathlib import Path
import zlib

import torch
from torch import nn

from experiments.adaptive_vector_rvq import codec as vector_codec
from experiments.adaptive_vector_rvq.common import compact, require
from experiments.adaptive_vector_rvq.length_codec_probe import HEADER, MAGIC, CODEC_ZLIB
from rotation.model_utils import temporarily_disable_torch_weight_init
from train_utils import checkpoint_v6 as native
from train_utils.distill_precision import restore_parameter_dtypes
from .codec import decode_lengths
from .spec import resolve

SCHEMA = 'native_prefix_v1'
DECODER_FIELDS = ('in_dim', 'out_dim', 'hidden_dim', 'num_res_blocks', 'norm_type', 'activation_type', 'decoder_type')


def load_native(native_spec, linear_state):
    """Strictly reconstruct just this real Linear through the existing native builder."""
    require(int(native_spec['parallel_parts']) == 1 and int(native_spec['parallel_rows']) == 1
            and int(native_spec['parallel_cols']) == 1 and int(native_spec['residual_stages']) == 2,
            'The importer supports one actual part and exactly two native stages.')
    require(not any(native_spec.get(name) for name in ('has_bias', 'has_original_weight', 'always_use_original',
                'protect_original_weight', 'low_rank_a', 'low_rank_b', 'protected_input_weight',
                'protected_output_indices', 'protected_output_weight', 'protected_output_qvalues',
                'protected_output_scales', 'protected_residual_stages')),
            'Bias, dense/original payload, LoRA, output protection and protected residuals are unsupported.')
    require(native_spec['protected_channel_quant_format'] == 'int8' and native_spec.get('weight_rotation') is not None,
            'Import the actual INT8 input-protection and rotation topology.')
    require(linear_state and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            for value in linear_state.values()), 'Use CPU relative fields of one real native Linear.')
    require(not any(name.startswith(('sparse_residual_', 'protected_output_', 'protected_residual_', 'low_rank_'))
                    for name in linear_state), 'Sparse/output-protected/residual/LoRA fields must not be discarded.')
    require(all(int(value) == int(native_spec['codebook_dim']) for value in native_spec['stage_codebook_dims'])
            and len(native_spec['stage_codebook_dims']) == 2, 'stage_codebook_dims are the two equal reconstruction D values, not K.')
    host = nn.Module()
    with temporarily_disable_torch_weight_init():
        host.linear = nn.Linear(int(native_spec['in_features']), int(native_spec['out_features']), bias=False)
    rebuilt_spec = {**deepcopy(native_spec), 'name': 'linear'}
    native._rebuild_converted_modules(host, [rebuilt_spec])
    state = {'linear.' + name: value for name, value in linear_state.items()}
    restore_parameter_dtypes(host, state)
    result = host.load_state_dict(state, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, 'The real single-Linear native payload must load strict.')
    native.refresh_vae_linear_runtime_after_state_load(host)
    return host.linear.eval().requires_grad_(False)


def fixed2_frame(n):
    lengths = torch.full((n,), 2, dtype=torch.uint8)
    raw = vector_codec.pack_lengths(lengths).numpy().tobytes()
    frame = HEADER.pack(MAGIC, CODEC_ZLIB, n, len(raw), zlib.crc32(raw)) + zlib.compress(raw, level=6)
    return lengths, torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone()


@torch.no_grad()
def from_native(native_spec, linear_state, provenance):
    require(isinstance(provenance, dict) and provenance, 'Explicit actual source provenance is required.')
    json.dumps(provenance, allow_nan=False)
    module = load_native(native_spec, linear_state)
    stages, n, dim = [], None, None
    for index in range(2):
        decoder = module.get_stage_part_decoder(index, 0)
        spec = {name: getattr(decoder, name) for name in DECODER_FIELDS}
        packed_spec = module.get_stage_part_vq_spec(index, 0)
        packed = module.get_stage_part_vq_storage(index, 0)
        logical = tuple(packed_spec['logical_shape'])
        require(len(logical) == 3 and logical[1] == 1 and logical[2] == int(spec['in_dim'])
                and tuple(packed.shape) == (logical[0], 1, (logical[2] + 7) // 8),
                'Actual packed [N,1,ceil(K/8)] geometry must match this stage decoder K.')
        require(dim is None or dim == int(spec['out_dim']), 'Both native stages must reconstruct the same D.')
        require(n is None or n == logical[0], 'Both native stages must cover every actual vector.')
        n, dim = logical[0], int(spec['out_dim'])
        require(dim == int(native_spec['stage_codebook_dims'][index]), 'Native codebook dimension D disagrees with decoder output.')
        state = {name: value.detach().cpu().contiguous().clone() for name, value in decoder.state_dict().items()}
        require(all(value.dtype == torch.bfloat16 for value in state.values()), 'Copy actual deployed BF16 decoder values without converting their precision.')
        stages.append(dict(decoder_spec=spec, codes=packed[:, 0, :].clone().contiguous(), decoder_state=state,
            normalization=dict(status='already_fused_in_current_native_decoder',
                original_statistics='not_persisted_and_not_recoverable',
                BSQ_input='bool01; signed scale already fused before native checkpoint',
                import_action='copy current restored decoder exactly; no repeated scale or mean/std fusion')))
    fixed = {name: getattr(module, name).detach().cpu().contiguous().clone()
             for name in ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
    fixed.update({f'rotation_{axis}_signs': getattr(module.weight_rotation, axis + '_signs').detach().cpu().contiguous().clone()
                  for axis in ('input', 'output')})
    lengths, frame = fixed2_frame(n)
    bundle = dict(n_vectors=n, stages=stages, fixed=fixed, length_frame=frame, provenance=deepcopy(provenance),
        geometry=dict(name=native_spec['name'], original_shape=[module.out_features, module.in_features],
            compressed_shape=[module.compressed_out_features, module.compressed_in_features],
            transpose=bool(module.transpose), rotation=deepcopy(module.weight_rotation.to_spec()), protected_quant='int8',
            source='actual current deployed native two-stage BF16 prefix; no live target inferred',
            source_key=native_spec['name']))
    return bundle, lengths, resolve(bundle, lengths)


def inventory(bundle):
    values = {'fixed/' + name: tensor for name, tensor in bundle['fixed'].items()}
    values['length_frame'] = bundle['length_frame']
    for index, stage in enumerate(bundle['stages']):
        values[f'stage{index}/codes'] = stage['codes']
        values.update({f'stage{index}/decoder/{name}': value for name, value in stage['decoder_state'].items()})
    return values


def ledger(bundle):
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    spec = resolve(bundle, lengths)
    sizes = {name: compact(value) for name, value in inventory(bundle).items()}
    code_bytes = sum(sizes[f'stage{index}/codes'] for index in range(2))
    logical_code_bits = sum(int(stage['decoder_spec']['in_dim']) * bundle['n_vectors'] for stage in bundle['stages'])
    return dict(payload_tensor_bytes=sum(sizes.values()), code_bytes=code_bytes, logical_code_bits=logical_code_bits,
        packed_code_bits=code_bytes * 8, code_padding_bits=code_bytes * 8 - logical_code_bits,
        decoder_bytes=sum(value for name, value in sizes.items() if '/decoder/' in name),
        fixed_bytes=sum(value for name, value in sizes.items() if name.startswith('fixed/')),
        length_frame_bytes=sizes['length_frame'], length_frame_header_bytes=HEADER.size,
        raw_two_bit_length_bytes=(bundle['n_vectors'] + 3) // 4, vector_dim=spec['vector_dim'], n_vectors=bundle['n_vectors'],
        tensor_bytes=sizes, future_fixed2_control_budget='Length2 may be implicit; this test frame must not be added to a future Rice fairness cap.')


def save_prefix(bundle, path):
    path = Path(path)
    require(not path.exists(), 'Never overwrite an imported prefix or prior evidence.')
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    resolved = resolve(bundle, lengths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(dict(schema=SCHEMA, bundle=bundle, resolved=resolved), stream)
    measured = ledger(bundle)
    measured.update(actual_file_bytes=path.stat().st_size,
                    serialization_and_metadata_bytes=path.stat().st_size - measured['payload_tensor_bytes'])
    return measured


def load_prefix(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['schema'] == SCHEMA, 'Expected the explicit native_prefix_v1 artifact, not an old three-stage schema.')
    bundle = payload['bundle']
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    resolved = resolve(bundle, lengths)
    require(resolved == payload['resolved'] and len(bundle['stages']) == 2 and bool((lengths == 2).all()),
            'Fresh native fixed2 prefix spec/route disagrees with its actual fields.')
    return bundle, lengths, resolved

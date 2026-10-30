"""Frozen existing abs005/Rice7 payload added to the shared BF16 compact path."""
from copy import deepcopy

import torch

from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import fixed2_frame
from experiments.adaptive_vector_rvq import common as c
from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
from train_utils.cat_train_residual_protection import build_sparse_residual_payload
from .sources import rice

FIELDS = (*rice.UNCHANGED, *rice.ENCODED)


@torch.no_grad()
def build(bundle, live_weight, deployed_weight):
    target = live_weight.detach().cpu().clone()
    base = deployed_weight.detach().cpu().contiguous()
    protected = bundle['fixed']['protected_input_indices']
    target[:, protected] = base[:, protected]
    payload, nnz, storage = build_sparse_residual_payload(linear_name=bundle['geometry']['name'],
        target_weight=target, reconstructed_weight=base, activation_weight=None,
        rank_metric='sparse_residual_abs', top_p=.005, min_abs=0., codec='blocked_quantized',
        index_bits=8, value_bits=8, block_shape=(256, 256))
    c.require(payload is not None and nnz == 251659, 'Preserve the original full-matrix ceil(abs005) support size.')
    fields = {name: payload['sparse_residual_' + name].contiguous().clone() for name in (*rice.UNCHANGED, 'local_indices')}
    encoded, bits = rice.encode(fields['local_indices'], fields['block_ptr'])
    c.require(torch.equal(rice.decode({**encoded, 'block_ptr': fields['block_ptr']}), fields['local_indices']),
              'Actual new Rice7 indices must equal original blocked indices exactly.')
    return {**{name: fields[name] for name in rice.UNCHANGED}, **encoded}, dict(nnz=nnz,
        original_codec_bytes=storage['codec_bytes'], rice_bits=bits,
        Rice_tensor_bytes=sum(c.compact(value) for value in (*encoded.values(), *(fields[name] for name in rice.UNCHANGED))))


class Fixed2RiceLinear(CompactRVQLinear):
    def __init__(self, bundle, fields, device='cpu'):
        lengths, frame = fixed2_frame(int(bundle['n_vectors']))
        c.require(len(bundle['stages']) == 2 and set(fields) == set(FIELDS), 'Fixed2 Rice needs exactly two full banks and all eight original Rice fields.')
        super().__init__({**bundle, 'length_frame': frame}, lengths, device)
        # The fixed control has implicit length2. Do not charge or save a length stream.
        self._non_persistent_buffers_set.add('length_frame')
        for name in FIELDS:
            self.register_buffer('sparse_' + name, fields[name].to(device))
        for name, dtype in (('rows', torch.int64), ('cols', torch.int64), ('values', torch.bfloat16)):
            self.register_buffer('sparse_decoded_' + name, torch.empty(0, dtype=dtype, device=device), persistent=False)
        self.refresh_sparse()
        self.register_load_state_dict_post_hook(lambda module, result: module.refresh_sparse())

    @torch.no_grad()
    def refresh_sparse(self):
        fields = {name: getattr(self, 'sparse_' + name).detach().cpu().contiguous() for name in FIELDS}
        decoded = {name: fields[name] for name in rice.UNCHANGED}
        decoded['local_indices'] = rice.decode(fields)
        rows, cols, values = decode_blocked_quantized_sparse_residual(**decoded,
            out_features=self.out_features, in_features=self.in_features,
            block_rows=256, block_cols=256, index_bits=8, value_bits=8,
            value_dtype=torch.bfloat16, device=self.length_frame.device)
        c.require(not bool(torch.isin(cols, self.protected_input_indices).any()), 'Rice support must exclude fixed protected columns.')
        self.sparse_decoded_rows, self.sparse_decoded_cols, self.sparse_decoded_values = rows, cols, values
        self.clear_decoded_weight_cache()

    def decode_weight(self):
        weight = super().decode_weight()
        return weight.index_put((self.sparse_decoded_rows.to(weight.device), self.sparse_decoded_cols.to(weight.device)),
                                self.sparse_decoded_values.to(weight.device), accumulate=True)

    def metadata(self):
        return {**super().metadata(), 'consumer': 'fixed2_rice7', 'implicit_length': 2,
                'sparse_fields': list(FIELDS), 'sparse_recipe': dict(block_rows=256, block_cols=256, index_bits=8, value_bits=8)}

    @classmethod
    def from_state(cls, spec, state, device='cpu'):
        base_spec = {name: value for name, value in spec.items() if name not in ('consumer', 'implicit_length', 'sparse_fields', 'sparse_recipe')}
        _, frame = fixed2_frame(spec['n_vectors'])
        base_state = {name: value for name, value in state.items() if not name.startswith('sparse_')}
        base = CompactRVQLinear.from_state(base_spec, {**base_state, 'length_frame': frame})
        fixed = {name: getattr(base, name) for name in base_spec['fixed_names'] if not name.startswith('rotation_')}
        fixed.update({f'rotation_{axis}_signs': getattr(base.weight_rotation, axis + '_signs') for axis in ('input', 'output')})
        stages = [dict(decoder_spec=item['decoder_spec'], normalization=item['normalization'],
            decoder_state=deepcopy(base.decoders[index].state_dict()), codes=getattr(base, f'codes_{index}'))
            for index, item in enumerate(base_spec['stages'])]
        module = cls(dict(n_vectors=spec['n_vectors'], geometry=spec['geometry'], fixed=fixed, stages=stages),
                     {name: state['sparse_' + name] for name in FIELDS}, device)
        c.require(module.metadata() == spec and set(state) == set(module.state_dict()), 'Actual Rice metadata or persistent inventory changed.')
        return module

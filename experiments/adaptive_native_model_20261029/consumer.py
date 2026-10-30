"""One actual compact Linear, sharing the native full-model runtime contract."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adaptive_vector_rvq.codec import restore_decoder
from experiments.adaptive_vector_rvq.common import CHUNK, require
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import build_weight_rotation
from train_utils.distill_precision import install_precision_runtime
from .codec import decode_lengths
from .spec import resolve


class CompactRVQLinear(nn.Module):
    def __init__(self, bundle, lengths, device='cpu'):
        super().__init__()
        self.resolved = resolve(bundle, lengths)
        self.out_features, self.in_features = self.resolved['geometry']['original_shape']
        self.register_parameter('bias', None)
        self.register_buffer('length_frame', bundle['length_frame'].to(device))
        self.decoders = nn.ModuleList()
        for index, stage in enumerate(bundle['stages']):
            self.decoders.append(restore_decoder(stage, device))
            self.register_buffer(f'codes_{index}', stage['codes'].to(device))
        for name, tensor in bundle['fixed'].items():
            if not name.startswith('rotation_'):
                self.register_buffer(name, tensor.to(device))
        shape = self.resolved['geometry']['compressed_shape']
        self.weight_rotation = build_weight_rotation(*shape, self.resolved['geometry']['rotation']).to(device)
        self.weight_rotation.input_signs.copy_(bundle['fixed']['rotation_input_signs'].to(device))
        self.weight_rotation.output_signs.copy_(bundle['fixed']['rotation_output_signs'].to(device))
        self.register_buffer('lengths', lengths.to(device), persistent=False)
        self.register_buffer('unprotected_indices', torch.empty(0, dtype=torch.int64, device=device), persistent=False)
        self.register_buffer('protected_weight', torch.empty(0, dtype=torch.bfloat16, device=device), persistent=False)
        for index in range(len(self.decoders)):
            self.register_buffer(f'active_{index}', torch.empty(0, dtype=torch.int64, device=device), persistent=False)
        self._decoded_cache = self._cache_versions = None
        self.cache_builds = 0
        self.refresh_runtime()
        self.register_load_state_dict_post_hook(lambda module, incompatible: module.refresh_runtime())

    def refresh_runtime(self):
        device = self.length_frame.device
        self.lengths = decode_lengths(self.length_frame, self.resolved['n_vectors']).to(device)
        for index in range(len(self.decoders)):
            active = torch.nonzero(self.lengths >= index + 1).flatten()
            require(len(active) == len(getattr(self, f'codes_{index}')), 'Fresh route rows do not match the actual code rows.')
            setattr(self, f'active_{index}', active)
        keep = torch.ones(self.in_features, dtype=torch.bool, device=device)
        keep[self.protected_input_indices] = False
        self.unprotected_indices = torch.nonzero(keep).flatten()
        self.protected_weight = decode_protected_channel_weight(self.protected_input_qvalues, self.protected_input_scales,
            quant_format='int8', dtype=torch.bfloat16, device=device).t().contiguous()
        self.weight_rotation.validate_state()
        self.clear_decoded_weight_cache()

    def enable_decoder_training(self):
        for decoder in self.decoders:
            decoder.float().requires_grad_(True)
            decoder.use_checkpoint = True
            install_precision_runtime(decoder, torch.bfloat16)
        self.clear_decoded_weight_cache()

    def clear_decoded_weight_cache(self):
        self._decoded_cache = self._cache_versions = None

    def train(self, mode=True):
        self.clear_decoded_weight_cache()
        return super().train(mode)

    def decode_weight(self):
        vectors = torch.zeros((self.resolved['n_vectors'], self.resolved['vector_dim']),
                              dtype=torch.bfloat16, device=self.length_frame.device)
        for index, (decoder, spec) in enumerate(zip(self.decoders, self.resolved['stages'], strict=True)):
            active, codes = getattr(self, f'active_{index}'), getattr(self, f'codes_{index}')
            pieces = []
            for start in range(0, len(active), CHUNK):
                packed = codes[start:start + CHUNK]
                bits = unpack_uint8_tensor_to_bool(packed, logical_shape=(len(packed), int(spec['decoder_spec']['in_dim'])))
                values = decoder(bits.to(torch.bfloat16).unsqueeze(1)).squeeze(1)
                require(values.dtype == torch.bfloat16, 'Preserve BF16 deployed decoder compute and stage accumulation.')
                pieces.append(values)
            if pieces:
                vectors = vectors.index_copy(0, active, vectors.index_select(0, active) + torch.cat(pieces))
        shape = self.resolved['geometry']['compressed_shape']
        rotated = vectors.reshape(shape[1], shape[0]).t().contiguous() if self.resolved['geometry']['transpose'] else vectors.reshape(shape)
        decoded = self.weight_rotation(rotated, inverse=True)
        weight = torch.zeros((self.out_features, self.in_features), dtype=torch.bfloat16, device=vectors.device)
        weight = weight.index_copy(1, self.unprotected_indices, decoded)
        return weight.index_copy(1, self.protected_input_indices, self.protected_weight)

    def forward(self, inputs):
        if torch.is_grad_enabled():
            weight = self.decode_weight()
        else:
            versions = tuple(value._version for value in (*self.parameters(), *self.buffers()))
            if self._decoded_cache is None or versions != self._cache_versions:
                self._decoded_cache, self._cache_versions = self.decode_weight(), versions
                self.cache_builds += 1
            weight = self._decoded_cache
        return F.linear(inputs, weight)

    def metadata(self):
        return deepcopy(self.resolved)

    @classmethod
    def from_state(cls, spec, state, device='cpu'):
        fixed = {name: state[name] for name in spec['fixed_names'] if not name.startswith('rotation_')}
        fixed.update({f'rotation_{axis}_signs': state[f'weight_rotation.{axis}_signs'] for axis in ('input', 'output')})
        stages = []
        for index, item in enumerate(spec['stages']):
            stages.append(dict(decoder_spec=item['decoder_spec'], normalization=item['normalization'],
                codes=state[f'codes_{index}'], decoder_state={name: state[f'decoders.{index}.{name}'] for name in item['decoder_tensor_names']}))
        bundle = dict(geometry=spec['geometry'], n_vectors=spec['n_vectors'], stages=stages,
                      fixed=fixed, length_frame=state['length_frame'])
        lengths = decode_lengths(bundle['length_frame'], spec['n_vectors'])
        require(resolve(bundle, lengths) == spec, 'Saved compact spec disagrees with its actual deployment fields.')
        return cls(bundle, lengths, device)

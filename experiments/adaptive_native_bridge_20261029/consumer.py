"""Trainable consumer of the existing compact RVQ files, outside public v6."""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adaptive_sparse_pair_20261028 import sparse_file
from experiments.adaptive_vector_rvq import codec, common as c, third_stage_export
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.protected_channel_quant import decode_protected_channel_weight
from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
from rotation.weight_preprocess import build_weight_rotation
from train_utils.distill_precision import install_precision_runtime


class CompactRVQLinear(nn.Module):
    def __init__(self, path, kind, device):
        super().__init__()
        self.source_path, self.kind = Path(path).resolve(), kind
        if kind == 'fixed2_sparse':
            self.template, self.sparse = sparse_file.load(self.source_path)
            lengths = codec.lengths(self.template)
        elif kind == 'dynamic3':
            self.template, lengths = third_stage_export.load(self.source_path)
            self.sparse = None
        else:
            raise ValueError('Use fixed2_sparse or dynamic3 actual compact files.')
        geometry = self.template['geometry']
        c.require(geometry['transpose'] and tuple(geometry['original_shape']) == (4096, 12288),
                  'This bridge covers the actual transposed Qwen down geometry.')
        self.out_features, self.in_features = geometry['original_shape']
        self.register_parameter('bias', None)
        self.register_buffer('lengths', lengths.to(device), persistent=False)
        self.decoders = nn.ModuleList()
        for index, stage in enumerate(self.template['stages']):
            decoder = codec.restore_decoder(stage, device).float().requires_grad_(True)
            decoder.use_checkpoint = True
            install_precision_runtime(decoder, torch.bfloat16)
            self.decoders.append(decoder)
            self.register_buffer(f'codes_{index}', stage['codes'].to(device))
            self.register_buffer(f'active_{index}', torch.nonzero(self.lengths >= index + 1).flatten(), persistent=False)
        for name, tensor in self.template['fixed'].items():
            self.register_buffer(name, tensor.to(device))
        shape = geometry['compressed_shape']
        self.rotation = build_weight_rotation(*shape, geometry['rotation']).to(device)
        self.rotation.input_signs.copy_(self.rotation_input_signs)
        self.rotation.output_signs.copy_(self.rotation_output_signs)
        keep = torch.ones(self.in_features, dtype=torch.bool, device=device)
        keep[self.protected_input_indices] = False
        self.register_buffer('unprotected_indices', torch.nonzero(keep).flatten(), persistent=False)
        protected = decode_protected_channel_weight(self.protected_input_qvalues, self.protected_input_scales,
            quant_format='int8', dtype=torch.bfloat16, device=torch.device(device)).t().contiguous()
        self.register_buffer('protected_weight', protected, persistent=False)
        if self.sparse is not None:
            for name in sparse_file.FIELDS:
                self.register_buffer('sparse_residual_' + name, self.sparse['sparse_residual_' + name].to(device))
            fields = {name: self.sparse['sparse_residual_' + name] for name in sparse_file.FIELDS}
            spec = {name: self.sparse['sparse_residual_' + name]
                    for name in ('block_rows', 'block_cols', 'index_bits', 'value_bits')}
            rows, columns, values = decode_blocked_quantized_sparse_residual(**fields, **spec,
                out_features=self.out_features, in_features=self.in_features,
                value_dtype=torch.bfloat16, device=torch.device(device))
            self.register_buffer('sparse_rows', rows, persistent=False)
            self.register_buffer('sparse_columns', columns, persistent=False)
            self.register_buffer('sparse_values', values, persistent=False)
        self._decoded_cache = None
        self._cache_versions = None
        self.cache_builds = 0

    def clear_decoded_weight_cache(self):
        self._decoded_cache = self._cache_versions = None

    def train(self, mode=True):
        self.clear_decoded_weight_cache()
        return super().train(mode)

    def decode_weight(self):
        device = self.lengths.device
        vectors = torch.zeros((self.template['n_vectors'], c.DIM), dtype=torch.bfloat16, device=device)
        for index, decoder in enumerate(self.decoders):
            active, codes = getattr(self, f'active_{index}'), getattr(self, f'codes_{index}')
            c.require(len(active) == len(codes), 'Selected code rows must match real routes.')
            pieces = []
            for start in range(0, len(codes), c.CHUNK):
                packed = codes[start:start + c.CHUNK]
                bits = unpack_uint8_tensor_to_bool(packed, logical_shape=(len(packed), c.BITS))
                pieces.append(decoder(bits.to(torch.bfloat16).unsqueeze(1)).squeeze(1))
            if pieces:
                values = torch.cat(pieces)
                # Skip the complete absent stage; a zero code would retain its bias.
                vectors = vectors.index_copy(0, active, vectors.index_select(0, active) + values)
        shape = self.template['geometry']['compressed_shape']
        rotated = vectors.reshape(shape[1], shape[0]).t().contiguous()
        decoded = self.rotation(rotated, inverse=True)
        weight = torch.zeros((self.out_features, self.in_features), dtype=torch.bfloat16, device=device)
        weight = weight.index_copy(1, self.unprotected_indices, decoded)
        weight = weight.index_copy(1, self.protected_input_indices, self.protected_weight)
        if self.sparse is not None:
            weight = weight.index_put((self.sparse_rows, self.sparse_columns), self.sparse_values, accumulate=True)
        return weight

    def forward(self, inputs):
        if torch.is_grad_enabled():
            weight = self.decode_weight()
        else:
            versions = tuple(parameter._version for parameter in self.parameters())
            if self._decoded_cache is None or versions != self._cache_versions:
                self._decoded_cache = self.decode_weight()
                self._cache_versions = versions
                self.cache_builds += 1
            weight = self._decoded_cache
        return F.linear(inputs, weight)

    def frozen_digest(self):
        return c.digest(dict(self.named_buffers()))

    def save_compact(self, path):
        bundle = deepcopy(self.template)
        for stage, decoder in zip(bundle['stages'], self.decoders, strict=True):
            stage['decoder_state'] = {name: tensor.detach().to(device='cpu', dtype=torch.bfloat16).contiguous().clone()
                                      for name, tensor in decoder.state_dict().items()}
        if self.sparse is None:
            return third_stage_export.save(bundle, Path(path))
        return sparse_file.save(bundle, self.sparse, Path(path))['total_bytes']

    @torch.no_grad()
    def reference_weight(self, device):
        from experiments.adaptive_vector_rvq.observe import original_weight
        sizes = self.lengths.cpu()
        vectors = torch.zeros((self.template['n_vectors'], c.DIM), dtype=torch.bfloat16)
        for index, stage in enumerate(self.template['stages']):
            active = torch.nonzero(sizes >= index + 1).flatten()
            codec.add_stage(vectors, stage, active, device)
        weight = original_weight(self.template, vectors, device)
        if self.sparse is not None:
            weight.index_put_((self.sparse_rows, self.sparse_columns), self.sparse_values, accumulate=True)
        return weight

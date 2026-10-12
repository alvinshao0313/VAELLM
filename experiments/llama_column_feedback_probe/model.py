"""Native transposed D16 adapter; only the selected two packed banks can change."""
from __future__ import annotations

import torch

from experiments.llama_column_feedback_probe.algorithm import choices_to_codes
from experiments.rotation_factor_probe.capture import require
from litebsq.bitpack import pack_bool_tensor_to_uint8
from litebsq.fused_multistage_decoder import (
    extract_packed_symmetric_stage_weights, fused_multistage_symmetric_decode,
    packed_symmetric_decoder_supports_fuse,
)
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load

TARGET = 'model.layers.16.mlp.down_proj'
SOURCE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
BANKS = ('vq_weight', 'vq_weight_s1')
INPUTS, OUTPUTS, GROUPS, BITS, WIDTH = 14080, 4096, 256, 24, 16


def validate(layer):
    require(isinstance(layer, VAELinear) and layer.transpose
            and (layer.in_features, layer.out_features, layer.compressed_in_features,
                 layer.compressed_out_features) == (14336, OUTPUTS, INPUTS, OUTPUTS)
            and layer.parallel_parts == 1 and layer.residual_stages == 2 and layer.codebook_dim == WIDTH
            and layer.parallel_stage_decode and layer._parallel_stage_restore_identity
            and list(layer._parallel_stage_layout) == [(0, 0), (1, 0)], 'Require the unchanged e767 L16 down transposed geometry.')
    decoder = layer._parallel_stage_decoder
    require(packed_symmetric_decoder_supports_fuse(decoder)
            and (decoder.in_dim, decoder.out_dim, decoder.hidden_dim, decoder.num_models) == (BITS, WIDTH, 128, 2)
            and all(p.dtype == torch.bfloat16 for p in decoder.parameters()), 'Require the original native BF16 symmetric decoder.')
    require(layer.weight_rotation is not None and layer.weight_rotation.block_size == 32
            and layer.protected_input_indices is not None and layer.protected_input_indices.numel() == 256
            and layer.protected_channel_quant_format == 'int8'
            and layer.protected_output_indices is None and layer.protected_residual_indices is None
            and layer.low_rank_a is None and layer.low_rank_b is None and not layer.has_sparse_residual()
            and layer.original_weight is None and not layer.always_use_original and not layer.trainable_decode
            and layer.bias is None and getattr(layer, '_sparse_bit_binding', None) is None,
            'Keep original input256 protection, block32 rotation, no dense/LoRA/residual/trainable branch.')
    layer.weight_rotation.validate_state()
    for stage in range(2):
        spec = layer.get_stage_part_vq_spec(stage)
        require(spec['logical_shape'] == [INPUTS * GROUPS, 1, BITS]
                and layer.get_stage_part_vq_storage(stage).dtype == torch.uint8, 'Unexpected original packed bank layout.')
    return decoder


def read_codes(layer):
    validate(layer)
    return torch.stack([layer.get_stage_part_vq_weight(stage).reshape(INPUTS, GROUPS, BITS) for stage in range(2)])


def apply_choices(layer, original_codes, choices):
    codes = choices_to_codes(original_codes, choices)
    for stage in range(2):
        logical = layer.get_stage_part_vq_spec(stage)['logical_shape']
        packed = pack_bool_tensor_to_uint8(codes[stage].reshape(logical), logical_shape=logical)
        layer.get_stage_part_vq_storage(stage).copy_(packed)
    refresh_vae_linear_runtime_after_state_load(layer)


class NativeStageOracle:
    """Same stage-isolation kernel and rounding as the existing BlockLDL oracle."""

    def __init__(self, layer):
        weights = extract_packed_symmetric_stage_weights(validate(layer))
        self.stage_weights = []
        for stage in range(2):
            fixed = list(weights)
            fixed[4], fixed[5] = weights[4].clone(), weights[5].clone()
            fixed[4][1 - stage].zero_(); fixed[5][1 - stage].zero_()
            self.stage_weights.append(tuple(fixed))

    @torch.no_grad()
    def __call__(self, stage, bits):
        require(stage in (0, 1) and bits.ndim == 2 and bits.shape[1] == BITS and bits.dtype == torch.bool,
                'Invalid original-stage oracle input.')
        output = torch.empty((len(bits), WIDTH), device=bits.device, dtype=torch.bfloat16)
        for start in range(0, len(bits), 8192):
            selected = bits[start:start + 8192]
            grouped = torch.zeros((len(selected), 2, BITS), device=bits.device, dtype=torch.bfloat16)
            grouped[:, stage] = selected
            output[start:start + len(selected)] = fused_multistage_symmetric_decode(grouped, *self.stage_weights[stage])
        return output


def inventory(model):
    excluded = {TARGET + '.' + key for key in BANKS}
    return {name: [value.data_ptr(), value._version, list(value.shape), str(value.dtype), str(value.device)]
            for name, value in model.state_dict().items() if name not in excluded}

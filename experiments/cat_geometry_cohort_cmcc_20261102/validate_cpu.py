"""Actual resolved recipes, matrix flatten and bitpack contract for this cohort."""
from copy import deepcopy

from . import bindings as b


def validate():
    from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
    from litebsq.llm_vae import MultiLayerVAE
    from train_utils.cat_train_data import reshape_blocks_for_codebook_dim
    binding = b.sources()
    meta = b.read(b.Path(binding['control']['path']) / 'checkpoint_meta.json')
    gates = [spec for spec in meta['converted_modules'] if spec['name'].endswith('.gate_proj')]
    b.require(len(gates) == 36 and all(spec['compressed_out_features'] == 12288
        and spec['compressed_in_features'] == 4032 and spec['parallel_parts'] == 1 and not spec['transpose']
        and spec['protected_input_indices']['shape'] == [64] and spec['weight_rotation']['block_size'] == 32
        for spec in gates), 'The actual shared-parent gate layout changed.')
    scalar_count = 12288*4032
    original = b.torch.arange(scalar_count, dtype=b.torch.int32).reshape(-1, 1, 64)
    rows = {}
    for arm, settings in b.ARMS.items():
        dim = settings['dim']
        reshaped = reshape_blocks_for_codebook_dim(original, codebook_dim=dim)
        b.require(reshaped.shape == (scalar_count//dim, 1, dim) and scalar_count % dim == 0
            and b.torch.equal(reshaped.flatten(), original.flatten()) and settings['batch']*dim == 262144,
            'Native flatten changed actual weight coordinates or added padding.')
        bits = b.torch.arange(7*dim).reshape(7, 1, dim).remainder(3).eq(0)
        packed = pack_bool_tensor_to_uint8(bits, logical_shape=bits.shape)
        unpacked = unpack_uint8_tensor_to_bool(packed, logical_shape=bits.shape)
        b.require(b.torch.equal(bits, unpacked) and packed.shape == (7,1,dim//8), 'Native K bitpack roundtrip failed.')
        argv, parsed, identity, _ = b.recipe(arm, b.OUTPUT / arm / 'formal/native')
        cfg = parsed[-1]['gate_proj']
        args = deepcopy(parsed[3])
        for key, value in dict(parallel_layers=36, codebook_dim=dim, codebook_bits=dim,
                base_ch=settings['encoder'], num_res_blocks=0, decoder_base_ch=128,
                decoder_num_res_blocks=0, decoder_type='asymmetric', norm_type='layer', activation_type='swish').items():
            setattr(args, key, value)
        model = MultiLayerVAE(args)
        rows[arm] = dict(actual_shared_control_gate_shape=[12288,4032], actual_protected_input_count=64,
            native_flatten_verified=True, padding_elements=0, vectors_per_gate=scalar_count//dim,
            D=dim, K=dim, encoder=settings['encoder'], decoder=128, residual_blocks=0,
            vector_batch=settings['batch'], scalar_exposure_per_linear_update=settings['batch']*dim,
            native_bitpack_verified=True, packed_bytes_per_vector_per_stage=dim//8,
            training_parameter_elements=sum(parameter.numel() for parameter in model.parameters()),
            entropy='Original pooled M36; K changes BSQ dimensions/scales, not an equal-FLOPs claim.',
            native_runtime_config=identity['resolved_category_runtime']['gate_proj'])
        del model, reshaped
    del original
    b.require(not b.torch.cuda.is_initialized(), 'CPU geometry validation initialized CUDA.')
    return dict(status='verified_actual_native_geometry', arms=rows,
        boundary='Shape/serialization evidence only; not convergence, downstream benefit or complete stable audit acceptance.')

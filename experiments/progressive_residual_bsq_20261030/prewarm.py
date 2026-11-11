"""Keep native group8 decoding and release each completed group's stage scratch."""
from bindings import s


def prepare(model, logger, expected_weights=None):
    import torch
    from litebsq import vae_linear_prewarm as native
    from tools.cat_eval import _prepare_model_for_eval
    original = native._decode_named_vae_linear_weights_chunk
    evidence = dict(chunks=0, released_scratch_bytes=0, compared_weights=0,
        identical_group_size=8, same_native_kernel_and_order=True)

    def release(*args, **kwargs):
        results, count = original(*args, **kwargs)
        evidence['chunks'] += 1
        for item in results:
            layer = item.base_layer
            scratch = [getattr(layer, name, None) for name in (
                '_parallel_stage_grouped_vq_weight', '_parallel_stage_grouped_vq_runtime',
                '_parallel_stage_grouped_vq_packed_runtime')]
            for tensor in scratch:
                if isinstance(tensor, torch.Tensor):
                    s.require(item.decoded_weight.device != tensor.device
                        or item.decoded_weight.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr(),
                        'Decoded weight must not alias released stage scratch.')
                    evidence['released_scratch_bytes'] += tensor.numel() * tensor.element_size()
            if expected_weights is not None and item.name in expected_weights:
                s.require(torch.equal(item.decoded_weight.cpu(), expected_weights[item.name]),
                    'Same native group8 output changed after per-group scratch release.')
                evidence['compared_weights'] += 1
            layer._clear_parallel_stage_decode_runtime_cache()
        return results, count

    native._decode_named_vae_linear_weights_chunk = release
    try:
        result = _prepare_model_for_eval(model, 'cuda', 8, logger, {})
    finally:
        native._decode_named_vae_linear_weights_chunk = original
    if expected_weights is not None:
        s.require(evidence['compared_weights'] == len(expected_weights), 'Not all actual reference weights were checked.')
    result['per_group_scratch_release'] = evidence
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result

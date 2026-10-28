"""Standalone packed CAT endpoints with scale restoration before inverse rotation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from train_utils.cat_train_pipeline import apply_group_vae_payload
from train_utils.checkpoint_bpw import _code_specs, _component
from train_utils.checkpoint_v6 import (_collect_vae_linear_specs, _rebuild_converted_modules,
    refresh_vae_linear_runtime_after_state_load)

from .scaling import BiaxialScale


class PackedScaledLinear(nn.Module):
    def __init__(self, codec, scale=None):
        super().__init__()
        self.codec = codec
        self.scale = scale

    def decode_weight(self, dtype=torch.bfloat16):
        if self.scale is None:
            return self.codec._decode_weight(dtype=dtype)
        # Native packed stages sum in BF16. Scales and inverse rotation use
        # FP32, followed by the original BF16 channel scatter boundary.
        normalized = self.codec._decode_compressed_weight(dtype=dtype)
        physical = self.scale.restore(normalized)
        return self.codec._finalize_decoded_weight_from_compressed(physical, dtype=dtype)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.decode_weight(x.dtype), self.codec.bias)


def storage_ledger(endpoint, native_spec):
    codes = {key.split('.', 1)[1]: item for key, item in _code_specs({**native_spec, 'name': 'x'})}
    groups, components = {}, defaultdict(lambda: {'logical_bits': 0, 'storage_bits': 0, 'tensors': 0})
    for key, value in endpoint.state_dict().items():
        if not value.numel():
            continue
        component = 'biaxial_scales' if key.startswith('scale.') else _component(key[len('codec.'):])
        relative = key[len('codec.'):] if key.startswith('codec.') else key
        logical = value.numel() * value.element_size() * 8
        if relative in codes:
            component = 'code'
            spec = codes[relative]
            logical = math_product(spec['logical_shape'])
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity in groups:
            raise ValueError('Unexpected standalone persistent storage alias: ' + key)
        groups[identity] = key
        item = components[component]
        item['logical_bits'] += logical
        item['storage_bits'] += storage.nbytes() * 8
        item['tensors'] += 1
    count = native_spec['in_features'] * native_spec['out_features']
    logical = sum(v['logical_bits'] for v in components.values())
    physical = sum(v['storage_bits'] for v in components.values())
    return dict(measurement='actual_standalone_persistent_tensor_storage', original_weights=count,
        logical_bits=logical, storage_bits=physical, logical_bpw=logical / count,
        storage_bpw=physical / count, code_padding_bits=physical - logical,
        components=dict(components), scope='Packed codes, decoder including fused scalar norms, rotation signs, biaxial scales, INT8 weights/scales/indices and bias. Encoder/optimizer excluded.')


def math_product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


@torch.no_grad()
def save_endpoints(host, refs, payload, scales, destination, device):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    apply_group_vae_payload(model=host, group_refs=refs, group_tag='biaxial_probe',
        payload=payload, convert_device=str(device))
    specs = {item['name']: item for item in _collect_vae_linear_specs(host)}
    saved = {}
    for ref, scale in zip(refs, scales):
        endpoint = PackedScaledLinear(host.get_submodule(ref.name), scale).to(device).eval().requires_grad_(False)
        before = endpoint.decode_weight().cpu()
        if not bool(torch.isfinite(before).all()):
            raise FloatingPointError('Nonfinite deployment reconstruction.')
        endpoint.cpu()
        state = {key: value.detach().clone() for key, value in endpoint.state_dict().items()}
        path = destination / (ref.name.replace('.', '_') + '.pt')
        torch.save(dict(format='cat_biaxial_packed_endpoint_v1', name=ref.name,
            native_spec=specs[ref.name], scaled=scale is not None, state_dict=state), path)
        restored = load_endpoint(path, device=device)
        after = restored.decode_weight().cpu()
        if not torch.equal(before, after):
            raise RuntimeError('Packed endpoint reconstruction changed after strict reload.')
        if not all(torch.equal(value.cpu(), state[key]) for key, value in restored.state_dict().items()):
            raise RuntimeError('Persistent packed endpoint state changed after strict reload.')
        saved[ref.name] = dict(path=str(path), strict_state_reload_exact=True,
            deployment_weight_reload_exact=True, ledger=storage_ledger(endpoint, specs[ref.name]))
        del restored, endpoint, before, after, state
    return saved


def load_endpoint(path, *, device='cpu'):
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if value['format'] != 'cat_biaxial_packed_endpoint_v1':
        raise ValueError('Unknown standalone endpoint format.')
    spec = {**value['native_spec'], 'name': 'linear'}
    host = nn.Module()
    host.add_module('linear', nn.Linear(spec['in_features'], spec['out_features'],
        bias=spec.get('has_bias', False), dtype=torch.bfloat16))
    _rebuild_converted_modules(host, [spec])
    scale = BiaxialScale(value['state_dict']['scale.row'], value['state_dict']['scale.column']) if value['scaled'] else None
    endpoint = PackedScaledLinear(host.linear, scale)
    endpoint.load_state_dict(value['state_dict'], strict=True)
    endpoint = endpoint.to(device).eval().requires_grad_(False)
    endpoint.codec._decoder_compute_dtype = torch.bfloat16
    refresh_vae_linear_runtime_after_state_load(endpoint)
    return endpoint

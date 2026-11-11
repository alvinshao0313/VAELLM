"""Future one-pass real after-up train16 input capture; no executable GPU entry."""
import logging

from . import bindings as b
from experiments.adaptive_native_model_20261029 import objective
from tools.cat_eval import _prepare_model_for_eval


@b.torch.no_grad()
def capture(bundles, *, device):
    b.c.require(len(bundles) == 3 and str(device).startswith('cuda'), 'Use the future reviewed three-layer actual after-up forward.')
    binding = b.original.original.bind(b.original.COMMON)
    parsed = b.original.original.recipe(b.OUTPUT / 'unused_recipe_only', binding)
    cache = parsed[-1]['activation_runtime']['cache']
    b.c.require(cache['nsamples'] == len(cache['input_ids']) == 512 and cache['seqlen'] == 256,
        'Reuse the genuine original Alpaca train512x256 calibration, without a new sample rule.')
    model, meta, loaded, _budget = b.original.checkpoint.load(b.LIVE)
    b.c.require(meta['checkpoint_id'] == b.LIVE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
        'Actual common after-up must load strict for live inputs.')
    warm = _prepare_model_for_eval(model, device, 8, logging.getLogger(__name__), {})
    b.c.require(warm['warmed'] == warm['total'] == 180 and not warm['failed'], 'Warm the real180 native after-up projections.')
    inputs = {bundle['geometry']['name']: [] for bundle in bundles}

    def hook(name):
        def collect(_module, values):
            inputs[name].append(values[0].detach().reshape(-1, values[0].shape[-1]).cpu().contiguous())
        return collect

    handles = [model.get_submodule(name).register_forward_pre_hook(hook(name)) for name in inputs]
    tokens = b.torch.cat([value.reshape(1, 256) for value in cache['input_ids'][:16]], dim=0)
    try:
        for begin in range(0, 16, 4):
            model.model(input_ids=tokens[begin:begin + 4].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    inputs = {name: b.torch.cat(parts) for name, parts in inputs.items()}
    b.c.require(all(tuple(value.shape) == (4096, 12288) for value in inputs.values()), 'Keep all sixteen original fixed256 train sequences.')
    h = {bundle['geometry']['name']: objective.input_energy(bundle, inputs[bundle['geometry']['name']][:2048])
        for bundle in bundles}
    return inputs, h, dict(parent_checkpoint_id=b.LIVE_ID, token_digest=b.c.digest({'input_ids': tokens}),
        captured_input_digest=b.c.digest(inputs), original_cache_digest=b.c.digest({'input_ids': tokens}),
        source='Original recorded Alpaca train512x256 calibration cache, first16 without resampling',
        route_documents=[0, 8], functional_readout_only_documents=[8, 16], native_warmup=warm,
        boundary='Both halves remain previously used calibration train pool; the latter is not a blind test')

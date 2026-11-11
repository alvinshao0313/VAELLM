"""One real after-up forward for the original fixed train16, all36 down inputs."""
import gc
import logging

from . import bindings as b
from experiments.adaptive_native_model_20261029 import objective
from tools.cat_eval import _prepare_model_for_eval


@b.torch.no_grad()
def execute(bundles, output):
    tokens = b.calibration()
    old = b.c.read(b.SHORT_RUN / 'report.json')['capture']
    b.c.require(b.c.digest({'input_ids': tokens}) == old['token_digest'], 'Do not change the short/full shared train16 tokens.')
    model, meta, loaded, _budget = b.checkpoint.load(b.LIVE)
    b.c.require(meta['checkpoint_id'] == b.LIVE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Strictly load the true live after-up capture parent.')
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    b.c.require(warm['total'] == warm['warmed'] == 180 and not warm['failed'], 'Warm exactly180 original native projections.')
    inputs = {name: [] for name in b.NAMES}

    def hook(name):
        def collect(_module, args):
            inputs[name].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu().contiguous())
        return collect

    handles = [model.get_submodule(name).register_forward_pre_hook(hook(name)) for name in b.NAMES]
    try:
        for start in range(0, 16, 4):
            model.model(input_ids=tokens[start:start + 4].to('cuda:0'), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    inputs = {name: b.torch.cat(parts).contiguous() for name, parts in inputs.items()}
    b.c.require(all(tuple(value.shape) == (4096, 12288) and value.dtype == b.torch.bfloat16 for value in inputs.values()),
        'Capture every valid coordinate of all36 actual BF16 down inputs.')
    short_inputs = b.torch.load(b.SHORT_RUN / 'capture_inputs.pt', map_location='cpu', weights_only=True, mmap=True)
    b.c.require(all(b.torch.equal(inputs[name], value) for name, value in short_inputs.items()), 'The original three real input tensors changed.')
    del model
    gc.collect()
    b.torch.cuda.empty_cache()
    energies = {name: objective.input_energy(bundles[name], inputs[name][:2048]) for name in b.NAMES}
    short_h = b.torch.load(b.SHORT_RUN / 'route_h.pt', map_location='cpu', weights_only=True, mmap=True)
    b.c.require(all(b.torch.equal(energies[name], value) for name, value in short_h.items()), 'The shared three first8 actual input energies changed.')
    b.torch.save(inputs, output / 'inputs.pt')
    b.torch.save(energies, output / 'h.pt')
    b.torch.save(tokens, output / 'tokens.pt')
    return dict(status='completed_once_live36_train16_capture', exit_code=0, parent_checkpoint_id=b.LIVE_ID,
        token_digest=b.c.digest({'input_ids': tokens}), input_digest=b.c.digest(inputs), h_digest=b.c.digest(energies),
        route_documents=[0, 8], functional_only_documents=[8, 16], source='Original recorded Alpaca train512x256 cache; first16',
        source_short_three_inputs_and_h_exact=True, files_sha256={str(output / name): b.c.sha(output / name) for name in ('inputs.pt', 'h.pt', 'tokens.pt')},
        native_warmup=warm, boundary='Both halves remain already used calibration train pool; no blind-test claim.')

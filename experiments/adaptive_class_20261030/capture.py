"""Stream the fixed first64 FineWeb train documents into actual per-input h."""
import argparse
import gc
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.adaptive_native_model_20261029 import checkpoint, objective
from experiments.adaptive_vector_rvq import common as c
from tools.cat_eval import _prepare_model_for_eval
from rotation.weight_preprocess import _transform_last
from . import sources as s


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c.require(not args.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use a new output and the assigned single GPU.')
    torch.set_num_threads(8)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = c.read(s.TOKEN_MANIFEST)
    c.require(manifest['data_source']['split'] == 'train'
        and c.sha(s.TOKENS) == manifest['archives']['tokens.pt']['sha256'], 'Use the already-audited unchanged FineWeb train tokens.')
    gate, specs, raw, _live = s.raw_inputs()
    rotations, kept, energy = {}, {}, {}
    for spec in specs:
        bundle, _lengths, resolved = s.prefix(gate, spec)
        name = spec['name']
        rotations[name] = objective._rotation(bundle, resolved, 'cuda:0')
        kept[name] = objective._kept(bundle, resolved, 'cuda:0')
        energy[name] = torch.zeros(len(kept[name]), dtype=torch.float64, device='cuda:0')
    del gate, _live, bundle
    gc.collect()
    meta = c.read(s.BASE / 'checkpoint_meta.json')
    model, _state, loaded = checkpoint.load_parent(s.BASE, meta)
    del _state
    c.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Actual after_o must load strict.')
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(warm['warmed'] == warm['total'] == 144 and not warm['failed'], 'Use all144 original after_o native projections.')
    tokens = torch.load(s.TOKENS, map_location='cpu', weights_only=True)
    mask, count = None, 0

    def hook(name):
        def accumulate(module, inputs):
            values = inputs[0].detach()[mask].index_select(1, kept[name]).float()
            rotation = rotations[name]
            with torch.autocast('cuda', enabled=False):
                values = _transform_last(values * rotation.input_signs.float(), block_size=rotation.input_block,
                    factor=rotation.input_factor, had=rotation.input_had, inverse=False)
            energy[name].add_(values.double().square().sum(0))
        return accumulate

    handles = [model.get_submodule(name).register_forward_pre_hook(hook(name)) for name in energy]
    try:
        for begin in range(0, 64, 4):
            batch = {key: value[begin:begin + 4].to('cuda:0') for key, value in tokens.items()}
            mask = batch['attention_mask'].bool()
            count += int(mask.sum())
            model.model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    c.require(count == sum(row['valid_tokens'] for row in manifest['documents'][:64]) == 15183,
              'Use exactly the first64 fixed train documents and valid tokens.')
    values = {name: (value / value.mean()).cpu().contiguous() for name, value in energy.items()}
    c.require(len(values) == 36 and all(torch.isfinite(value).all() and (value >= 0).all() for value in values.values()), 'Invalid complete36gate live input energies.')
    torch.save(values, args.output / 'input_h.pt')
    c.write(args.output / 'report.json', dict(status='completed_live36gate_train64_h', exit_code=0, pid=os.getpid(),
        command=[sys.executable, *sys.argv], gpu=os.environ['CUDA_VISIBLE_DEVICES'], source_after_o=s.BASE_ID,
        raw_B=raw['source_raw']['checkpoint_id'], token_archive=c.stat(s.TOKENS), token_sha256=c.sha(s.TOKENS),
        token_manifest_sha256=c.sha(s.TOKEN_MANIFEST), data_source=manifest['data_source'],
        actual_documents=64, valid_tokens=count, first_document=0, h_digest=c.digest(values),
        h_file_sha256=c.sha(args.output / 'input_h.pt'), native_warmup=warm,
        source_sha256=s.code_identity(), elapsed_seconds=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        boundary='Same after_o true forward inputs; no activation archive, no test selection, no training or new scoring.'))
    print(dict(status='completed_live36gate_train64_h', seconds=time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()

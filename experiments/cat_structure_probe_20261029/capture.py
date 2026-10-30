"""Capture three gate inputs from the true compressed-attention/live-MLP parent."""
import gc
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_structure_capture_common', HERE/'common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
sys.path.insert(0, str(c.SNAPSHOT))

import torch
from train_utils.v6_model_loader import load_v6_model_checkpoint
from tools.cat_eval import _prepare_model_for_eval


@torch.no_grad()
def main():
    output = c.OUTPUT/'prepared'
    record = c.read(output/'prepared.json')
    destination = output/'activations.pt'
    c.require(not destination.exists() and record['source_checkpoint_id'] == c.SOURCE_ID,
              'Capture actual parent activations exactly once into a new file.')
    c.require(c.sha(c.TOKENS) == record['token_archive']['sha256']
              and c.stat(c.SOURCE/c.source()['state_dict_file']) == record['source_weights'], 'Inputs changed after preparation.')
    torch.set_num_threads(4)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model, meta, result = load_v6_model_checkpoint(str(c.SOURCE), map_location='cpu', strict=True,
                                                expected_kind='category_boundary')
    c.require(not result.missing_keys and not result.unexpected_keys and meta['checkpoint_id'] == c.SOURCE_ID,
              'Actual category boundary failed strict loading.')
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    names = list(record['targets'])
    c.require(all(type(model.get_submodule(name)) is torch.nn.Linear for name in names), 'Gate must still be live dense.')
    batch = torch.load(c.TOKENS, map_location='cpu', weights_only=True)
    captured = {name: [] for name in names}
    mask = None

    def hook(name):
        def store(module, args):
            values = args[0].detach()
            c.require(values.dtype == torch.bfloat16, 'Preserve actual native BF16 hidden states.')
            captured[name].append(values[mask].cpu().contiguous())
        return store

    handles = [model.get_submodule(name).register_forward_pre_hook(hook(name)) for name in names]
    try:
        for start in range(0, len(batch['input_ids']), 4):
            current = {key: value[start:start+4].to('cuda:0') for key, value in batch.items()}
            mask = current['attention_mask'].bool()
            model.model(**current, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    values = {name: torch.cat(rows) for name, rows in captured.items()}
    c.require(all(tuple(value.shape) == (record['valid_tokens'], 4096) for value in values.values()),
              'Capture must cover all original valid tokens of the three real gates.')
    torch.save(dict(inputs=values, document_lengths=[row['valid_tokens'] for row in record['documents']],
                    source_checkpoint_id=c.SOURCE_ID), destination)
    capture = dict(status='complete_live_prefix_capture', source_checkpoint_id=c.SOURCE_ID,
        activations=dict(**c.stat(destination), sha256=c.sha(destination)), token_archive=record['token_archive'],
        source_weights=record['source_weights'], native_warmup=warm, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        actual_documents=128, actual_valid_tokens=30675, batch=4, maximum_length=256,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic()-started, loader=str(Path(load_v6_model_checkpoint.__code__.co_filename).resolve()),
        source_sha256={str(path): c.sha(path) for path in (Path(__file__), HERE/'common.py')},
        boundary='Actual after_o native prefix on fixed train text; not the old HF activation archive or downstream scoring.')
    c.write(output/'capture.json', capture)
    del model
    gc.collect(); torch.cuda.empty_cache()
    print(dict(status=capture['status'], seconds=capture['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()

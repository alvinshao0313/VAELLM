"""Check real native loaders against batched gathering; no model/GPU claims."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from train_utils.cat_train_data import build_block_data_loaders as native_loaders
from .data import build_block_data_loaders


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace(factory, data, batch_size, seed):
    torch.manual_seed(31)
    # Preserve both native factory calls: training and evaluation are obtained
    # from different calls in the real residual loop.
    train, unused = factory(data, batch_size, seed)
    del unused
    unused, evaluation = factory(data, batch_size, seed)
    del unused
    rows = []

    def record(batch, name, epoch):
        assert type(batch) is list and len(batch) == 2
        x, index = batch
        assert x.data_ptr() != data.data_ptr()
        assert torch.equal(x, data.index_select(0, index))
        rows.append(dict(kind=name, epoch=epoch, indices=index.tolist(), values=tensor_sha(x),
            shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()),
            global_rng=tensor_sha(torch.get_rng_state()),
            train_rng=tensor_sha(train.generator.get_state()) if train.generator is not None else None))

    for epoch in range(3):
        for step, batch in enumerate(train):
            record(batch, 'train', epoch)
            if step == 0:
                iterator = iter(evaluation)
                record(next(iterator), 'interleaved_evaluation', epoch)
                del iterator
        for batch in evaluation:
            record(batch, 'complete_evaluation', epoch)
    return rows, tensor_sha(torch.get_rng_state())


def benchmark(factory, data, batch_size):
    train, _ = factory(data, batch_size, 104729)
    next(iter(train))
    started = time.perf_counter()
    count = steps = 0
    for _ in range(2):
        for x, _indices in train:
            count += x.shape[0]
            steps += 1
    return dict(seconds=time.perf_counter() - started, batches=steps, rows=count,
        scope='CPU sampler and gathering only; excludes GPU transfer, VAE and model work')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new CPU validation output.')
    started = time.perf_counter()
    cases = []
    for n, batch_size, models, dim, seed, layout in (
        (13, 5, 3, 8, 104729, 'contiguous'),
        (13, 5, 3, 8, None, 'contiguous'),
        (16384, 8192, 36, 32, 104729, 'contiguous'),
        (16401, 8192, 36, 32, 104729, 'contiguous'),
        (19, 7, 3, 8, 104729, 'strided'),
    ):
        data = torch.arange(n * models * dim, dtype=torch.int32).remainder_(251).to(torch.bfloat16)
        data = data.reshape(n, models, dim)
        if layout == 'strided':
            data = data.transpose(1, 2)
        expected = trace(native_loaders, data, batch_size, seed)
        actual = trace(build_block_data_loaders, data, batch_size, seed)
        assert actual == expected, 'Native batches, values, layout or RNG stream changed.'
        row = dict(source_shape=list(data.shape), batch_size=batch_size, shuffle_seed=seed,
            source_layout=layout, observed_batches=len(actual[0]),
            exact_indices_values_layout_global_and_sampler_RNG=True,
            cross_epoch_and_interleaved_evaluation=True)
        if batch_size == 8192:
            row['native_CPU_gather'] = benchmark(native_loaders, data, batch_size)
            row['batched_CPU_gather'] = benchmark(build_block_data_loaders, data, batch_size)
        cases.append(row)
    assert not torch.cuda.is_initialized()
    from train_utils import cat_train_data
    sources = [Path(cat_train_data.__file__), Path(__file__), Path(__file__).with_name('data.py')]
    report = dict(status='CPU_native_batch_gather_verified', exit_code=0,
        elapsed_seconds=time.perf_counter() - started, torch_version=torch.__version__,
        python_version=platform.python_version(), CUDA_initialized=False,
        cases=cases, source_sha256={str(path.resolve()): source_sha(path) for path in sources},
        scope='Real native DataLoader comparison only; not a VAE fit or GPU throughput result',
        formal_fit_integration_verified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seconds=report['elapsed_seconds'], cases=len(cases))))


if __name__ == '__main__':
    main()

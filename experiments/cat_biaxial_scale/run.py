"""Run one preselected category/normalization/protection job on an assigned GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from train_utils.utils import configure_deterministic_mode

from .runtime import save_endpoints
from .training import BATCH, SEED, fit


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scale_statistics(scale):
    if scale is None:
        return None
    result = {}
    for axis in ('row', 'column'):
        value = getattr(scale, axis).float()
        quantiles = torch.quantile(value, torch.tensor([0., .5, .99, 1.]))
        result[axis] = dict(zip(('min', 'median', 'p99', 'max'), quantiles.tolist()))
        result[axis].update(rms=float(value.square().mean().sqrt()),
            stored_dtype=str(getattr(scale, axis).dtype), bytes=getattr(scale, axis).numel() * 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--category', choices=('gate_proj', 'up_proj', 'down_proj'), required=True)
    parser.add_argument('--normalization', choices=('current', 'biaxial', 'learned'), required=True)
    parser.add_argument('--protection', choices=('int8', 'none'), required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.output_dir.exists():
        raise ValueError('Require positive stage steps and a new output directory.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one explicitly assigned GPU.')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    configure_deterministic_mode(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    source = torch.load(args.input_file, map_location='cpu', weights_only=True, mmap=True)
    targets = {name: item for name, item in source['targets'].items() if name.endswith('.' + args.category)}
    expected = {f'model.layers.{layer}.mlp.{args.category}' for layer in (0, 18, 35)}
    if set(targets) != expected or any(item['reference_weight'].dtype != torch.bfloat16 for item in targets.values()):
        raise ValueError('Require all three preselected real BF16 Qwen matrices.')
    shapes = {tuple(item['reference_weight'].shape) for item in targets.values()}
    if len(shapes) != 1:
        raise ValueError('Native parallel category requires matching matrix shapes.')
    args.output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    sources = [*Path(__file__).parent.glob('*.py'), *[root / name for name in (
        'train_utils/cat_train_pipeline.py', 'train_utils/cat_data_prep.py',
        'train_utils/cat_train_data.py', 'train_utils/cat_category_runtime.py',
        'train_utils/cat_runtime_adapter.py', 'train_utils/train_args.py',
        'train_utils/checkpoint_v6.py', 'train_utils/checkpoint_bpw.py',
        'litebsq/autoencoder.py', 'litebsq/bsq.py', 'litebsq/parallel_layers.py',
        'litebsq/vae_linear.py', 'litebsq/vae_args.py', 'litebsq/bitpack.py',
        'rotation/weight_preprocess.py')]]
    report = dict(status='running', argv=sys.argv, pid=os.getpid(), started_unix=time.time(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), category=args.category,
        normalization=args.normalization, protection=args.protection, seed=SEED,
        input_file=str(args.input_file.resolve()), input_file_stat=dict(size=args.input_file.stat().st_size,
            mtime_ns=args.input_file.stat().st_mtime_ns),
        source_sha256={str(path.resolve()): sha256(path) for path in sources},
        precision=dict(torch_version=torch.__version__, parameters='BF16', autoencoder_autocast='BF16',
            reconstruction='FP32', scale_master='FP32 only in learned arm', deployed_scales='BF16'),
        initialization=dict(codec='Fresh native MultiLayerVAE at each residual stage; no historical encoder/decoder state.',
            global_seed_before_group=SEED,
            stage2='Continues the native CPU initialization stream after stage1 decoder extraction; only the sampler resets to seed32.',
            learned_frozen_decode='Extra native stage readout is wrapped in fork_rng so it preserves codec initialization order.'),
        sampler=dict(batch_vectors_per_linear=BATCH, shuffle_seed=SEED,
            schedule='Fresh CPU torch.Generator seed31+stage; randperm at each exhausted epoch, shared vector indices across the three parallel linears.',
            physical_payload_diff='Protection changes vector population/epoch boundaries; no exact batch-content claim across protection arms.'),
        auxiliary='Unmodified native pooled BSQ; lfq_weight2.5, l1_weight1 and native num_models=M3 factors.',
        scope='Single-category three-layer proxy; packed local deployment only, no full-model quality or 2.5bpw acceptance claim.')
    exit_code = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        host, refs, payload, scales, training = fit(targets, source['model_path'], args.category,
            args.normalization, args.protection, args.steps, torch.device('cuda:0'))
        report['training'] = training
        report['scale_statistics'] = {ref.name: scale_statistics(scale) for ref, scale in zip(refs, scales)}
        report['endpoints'] = save_endpoints(host, refs, payload, scales, args.output_dir / 'endpoints', 'cuda:0')
        report.update(status='completed', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        exit_code = 0
    except BaseException as error:
        report.update(status='failed', error=repr(error)); traceback.print_exc()
        raise
    finally:
        report.update(exit_code=exit_code, finished_unix=time.time())
        (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        (args.output_dir / 'run.exit').write_text(str(exit_code) + '\n')


if __name__ == '__main__':
    main()

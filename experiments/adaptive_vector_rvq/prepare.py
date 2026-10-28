"""Prepare three full fresh HF matrices using G's fixed protection IDs and rotation."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import torch
from safetensors import safe_open

from . import common as c
from .budget import counts_for_budget
from litebsq.autoencoder import Decoder
from rotation.weight_preprocess import build_weight_rotation
from train_utils.cat_data_prep import LinearPrepRef, prepare_group_linear_entries


def prepare(output):
    c.require(not output.exists() and not torch.cuda.is_initialized(), 'Use a fresh CPU preparation directory.')
    torch.set_num_threads(4)
    started = time.monotonic()
    native = c.read(c.G_REPORT)
    c.require(native['status'] == 'completed' and native['exit_code'] == 0, 'The real G source must have completed.')
    final = [row for row in native['saved_boundaries'] if row['kind'] == 'final_model']
    c.require(len(final) == 1, 'One actual final G source is required.')
    source = Path(final[0]['path'])
    meta = c.read(source / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == final[0]['checkpoint_id'] == '956b5a70-f830-40cb-9d14-f155de316344', 'Wrong actual G protection/rotation donor.')
    saved = torch.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    index_path = c.HF / 'model.safetensors.index.json'
    index = c.read(index_path)['weight_map']
    specs = {row['name']: row for row in meta['converted_modules']}
    output.mkdir(parents=True)
    rows, input_stats = {}, [c.stat(source / meta['state_dict_file'])]
    with torch.device('meta'):
        decoder = Decoder(in_dim=c.BITS, out_dim=c.DIM, hidden_dim=c.HIDDEN, num_res_blocks=0,
            norm_type='layer', activation_type='swish', decoder_type='symmetric', num_models=1)
    decoder_bytes = sum(value.numel() * 2 for value in decoder.state_dict().values())
    for layer in c.LAYERS:
        name = f'model.layers.{layer}.mlp.down_proj'
        spec, key = specs[name], name + '.weight'
        c.require(spec['transpose'] and spec['parallel_parts'] == 1 and spec['codebook_dim'] == 32
            and spec['protected_channel_quant_format'] == 'int8', 'Use the real single-part down topology.')
        protected = saved[name + '.protected_input_indices'].detach().clone().contiguous()
        c.require(protected.numel() == 256 and protected.dtype == torch.int64, 'Use G actual256 protected input IDs.')
        shard = c.HF / index[key]
        with safe_open(str(shard), framework='pt', device='cpu') as stream:
            weight = stream.get_tensor(key).clone()
        c.require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == (4096, 12288), 'Use fresh original HF BF16 down weights, not G decoded or residual-dense weights.')
        ref = LinearPrepRef(name, weight, 12288, 4096, True)
        entry = prepare_group_linear_entries(group_refs=[ref], activation_weight_by_linear=None,
            channel_protect_count=256, channel_axis='input', channel_quant='int8', recon_loss_type='mse',
            channel_plan={name: protected})[0].prepared_weight
        transform = build_weight_rotation(entry.compressed_out_features, entry.compressed_in_features, spec['weight_rotation'])
        for axis in ('input', 'output'):
            c.require(torch.equal(getattr(transform, axis + '_signs'), saved[name + '.weight_rotation.' + axis + '_signs']), 'Actual G rotation signs differ from the declared source metadata.')
        with torch.no_grad():
            rotated = transform(entry.split_weight.float()).contiguous()
        vectors = rotated.t().contiguous().reshape(-1, c.DIM)
        fixed = dict(protected_input_indices=protected, protected_input_qvalues=entry.protected_input_qvalues.clone().contiguous(),
            protected_input_scales=entry.protected_input_scales.clone().contiguous(),
            rotation_input_signs=transform.input_signs.clone().contiguous(), rotation_output_signs=transform.output_signs.clone().contiguous())
        geometry = dict(name=name, original_shape=list(weight.shape), compressed_shape=list(rotated.shape), transpose=True,
            rotation=spec['weight_rotation'], protected_quant='int8', source='fresh original HF BF16', source_key=key)
        source_variance = float(vectors.double().var(unbiased=False))
        c.require(source_variance > 0 and bool(torch.isfinite(vectors).all()), 'Actual source vectors must be finite with positive variance.')
        budget = counts_for_budget(len(vectors), decoder_bytes, sum(c.compact(value) for value in fixed.values()))
        path = output / f'layer_{layer}.pt'
        torch.save(dict(vectors=vectors, fixed=fixed, geometry=geometry, source_variance=source_variance), path)
        rows[str(layer)] = dict(path=str(path), input=c.stat(path), vector_count=len(vectors), geometry=geometry,
            source_variance=source_variance, fixed_digest=c.digest(fixed), budget=budget,
            target_digest=c.digest({'vectors': vectors}), HF_shard=c.stat(shard))
        input_stats.extend([c.stat(path), c.stat(shard)])
        print(dict(event='actual_fresh_matrix_prepared', layer=layer, vectors=len(vectors), budget=budget), flush=True)
        del weight, entry, rotated, vectors, fixed, transform
        gc.collect()
    source_paths = [*c.HERE.glob('*.py'), index_path, c.HF / 'config.json', c.G_REPORT, source / 'checkpoint_meta.json']
    core = ('litebsq/autoencoder.py', 'litebsq/bsq.py', 'litebsq/parallel_layers.py', 'litebsq/vae_args.py',
        'litebsq/bitpack.py', 'litebsq/protected_channel_quant.py', 'rotation/weight_preprocess.py',
        'rotation/hadamard_utils.py', 'train_utils/cat_data_prep.py', 'train_utils/cat_train_data.py',
        'train_utils/train_args.py', 'train_utils/cat_train_pipeline.py')
    source_paths += [c.ROOT / name for name in core]
    activation = dict(path=str(c.ACTIVATIONS), role='output observation only; never training or gate selection')
    if c.ACTIVATION_MANIFEST.exists():
        c.require(c.sha(c.ACTIVATION_MANIFEST) == 'd0e0e5c52f3e797838f981beaa9749434174177b4fdf6ea8206cc9ce153329e0', 'The fixed existing FineWeb train observation manifest changed.')
        activation.update(manifest=str(c.ACTIVATION_MANIFEST), manifest_sha256=c.sha(c.ACTIVATION_MANIFEST), input=c.stat(c.ACTIVATIONS))
        source_paths.append(c.ACTIVATION_MANIFEST)
        input_stats.append(c.stat(c.ACTIVATIONS))
    record = dict(status='completed', schema='adaptive_vector_rvq_inputs_v1', layers=rows,
        HF_path=str(c.HF), protection_rotation_source=str(source), protection_rotation_checkpoint_id=meta['checkpoint_id'],
        source_policy='Fresh HF values in both arms; G supplies only protected IDs and rotation. HF protection is freshly encoded once and frozen.',
        training=dict(dim=32, bits=48, hidden=128, residual_blocks=0, batch=8192, seed=31, lr=.003,
            optimizer='adamw', betas=[.9, .95], scheduler='linear', warmup=0, BF16=True,
            base_steps=2000, shared_stage1_steps=2000, fixed2_steps=6000, adaptive_steps=[2000, 2000, 2000]),
        activations=activation, source_sha256={str(path.resolve()): c.sha(path) for path in source_paths},
        input_stats=list({item['path']: item for item in input_stats}.values()),
        GPU_initialized=False, elapsed_seconds=time.monotonic() - started)
    c.write(output / 'prepared.json', record)
    c.require(not torch.cuda.is_initialized(), 'Preparation initialized CUDA.')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir.resolve())

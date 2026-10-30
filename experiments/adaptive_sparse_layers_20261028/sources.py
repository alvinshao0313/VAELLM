"""Bind each true layer without changing the original L0-only entry point."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time

import torch

from experiments.adaptive_layer_transfer_20261028 import helpers as h
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.third_stage import stage_digest

HERE = Path(__file__).resolve().parent
ROOT = h.OUTPUT


def free_gpu2():
    def query(field):
        return subprocess.run(['nvidia-smi', '--id=2', field, '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True).stdout.strip()
    uuid = query('--query-gpu=uuid')
    processes = query('--query-compute-apps=pid,used_memory')
    used = int(query('--query-gpu=memory.used'))
    c.require(not processes and used == 0, 'Assigned physical GPU2 is occupied; do not wait or retry.')
    return dict(gpu=2, uuid=uuid, checked_unix=time.time(), compute_processes=[], memory_used_MiB=used)


def same_metadata(left, right):
    return (all(left[key] == right[key] for key in ('n_vectors', 'geometry'))
        and all(a['decoder_spec'] == b['decoder_spec'] and a['normalization'] == b['normalization']
            for a, b in zip(left['stages'], right['stages'], strict=True)))


def bind(layer):
    prepared_path = ROOT / 'prepared/prepared.json'
    prepared = h.check_prepared(prepared_path.parent)
    c.require(c.sha(prepared_path) == '30d7e4ee4e77028bfb6dc9f6db5c7e1cc0c7f4eb6985e1d583e693e03b437470',
        'Use the original completed cross-layer preparation.')
    row = prepared['layers'][str(layer)]
    parent_path = ROOT / f'formal_l{layer}/report.json'
    parent = c.read(parent_path)
    stage_path = Path(parent['stage3_artifact']['path'])
    fixed_path = Path(row['fixed2_path'])
    c.require(parent['status'] == 'completed' and parent['exit_code'] == 0
        and parent['steps'] == 6000 and parent['layer'] == layer
        and parent['prepared_sha256'] == c.sha(prepared_path)
        and parent['target_digest'] == row['target_digest']
        and parent['source_unchanged'] and parent['frozen_banks_unchanged']
        and parent['fresh_stage3_state_and_prefix_exact']
        and parent['fresh_all_fields_and_routes_exact'] and parent['fresh_actual_prefix_errors_exact']
        and c.sha(stage_path) == parent['stage3_artifact']['sha256']
        and c.sha(fixed_path) == row['fixed2_sha256'] == parent['source_sha256'][str(fixed_path)],
        'Use the completed real fixed2 and 6000-step bank for this exact layer.')
    source = codec.load(fixed_path)
    data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
    stage3 = torch.load(stage_path, map_location='cpu', weights_only=True)
    banks = [*source['stages'], stage3]
    c.require(source['mode'] == 'fixed2' and source['n_vectors'] == row['vector_count'] == 1540096
        and source['geometry'] == data['geometry'] == row['geometry']
        and source['geometry']['name'] == f'model.layers.{layer}.mlp.down_proj'
        and source['geometry']['source_key'] == f'model.layers.{layer}.mlp.down_proj.weight'
        and source['geometry']['original_shape'] == [4096, 12288]
        and source['geometry']['compressed_shape'] == [4096, 12032]
        and source['geometry']['transpose'] and source['geometry']['rotation']['block_size'] == 32
        and c.digest(data['fixed']) == c.digest(source['fixed']) == row['fixed_digest']
        and c.digest({'vectors': data['vectors']}) == row['target_digest']
        and data['source_variance'] == row['source_variance']
        and [stage_digest(s) for s in banks] == parent['frozen_bank_digests']
        and [stage_digest(s) for s in source['stages']] == row['frozen_stage_digests']
        and source['fixed']['protected_input_indices'].numel() == 256
        and all(tuple(s['codes'].shape) == (1540096, 6) and s['codes'].dtype == torch.uint8
            and s['decoder_spec']['in_dim'] == 48 and s['decoder_spec']['out_dim'] == 32
            and s['decoder_spec']['hidden_dim'] == 128 and s['decoder_spec']['num_res_blocks'] == 0
            and all(v.dtype == torch.bfloat16 for v in s['decoder_state'].values()) for s in banks),
        'Layer identity, actual target, geometry, normalization or persisted bank changed.')
    paths = {Path(p) for p in prepared['source_sha256']}
    paths.update(c.HERE.glob('*.py'))
    paths.update(HERE.glob('*.py'))
    paths.update(HERE.glob('*.sh'))
    paths.update((prepared_path, parent_path, stage_path, fixed_path, Path(row['control_report']),
        c.ACTIVATIONS, c.ACTIVATION_MANIFEST, Path(h.__file__),
        c.ROOT / 'litebsq/sparse_residual.py', c.ROOT / 'litebsq/vae_linear.py',
        c.ROOT / 'train_utils/cat_train_residual_protection.py'))
    paths.update((c.ROOT / 'experiments/adaptive_sparse_pair_20261028').glob('*.py'))
    return prepared, row, source, data, banks, paths

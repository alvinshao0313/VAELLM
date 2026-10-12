"""Materialize a genuine Q/K target and call the unchanged native CAT payload path."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import random
import time

import numpy as np
from safetensors import safe_open
import torch
from torch import nn

from experiments.full_rht_gate_pair.common import read
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.misc import set_module_by_name
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import decode_named_vae_linear_weights
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from train_utils.cat_train_pipeline import apply_group_vae_payload, train_group_vae_payload
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state
from train_utils.utils import LinearRef, configure_deterministic_mode, set_seed
from train_utils.v6_model_loader import load_v6_model_checkpoint

CATEGORIES = ('q_proj', 'k_proj')
NAMES = tuple(f'model.layers.{layer}.self_attn.{category}'
              for category in CATEGORIES for layer in range(36))
ARMS = {'decoded_block32': ('decoded', 32), 'decoded_full': ('decoded', 0), 'hf_full': ('hf', 0)}
PROTECTION = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')


def parsed(record):
    cat, _, training, vae = parse_cat_runtime_args(record['native_argv'])
    # This is exactly the assignment made by tools/cat_train.py before its pipeline.
    training.seed = int(cat.seed)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(normalized == record['normalized_expected'], 'Native parser differs from CPU preparation.')
    _require(cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 36
             and cat.compression_categories == 'q_proj,k_proj' and cat.after_category_mode == 'none'
             and not cat.channel_refresh_after_category and cat.channel_axis == 'input'
             and cat.channel_rank_metric == 'channel_weight_actmean_abs'
             and training.bf16 and not training.fp16 and cat.deterministic,
             'Fixed native Q/K recipe changed.')
    for cfg in resolved.values():
        _require(cfg.steps == record['steps_per_stage'] and cfg.codebook_bits == cfg.codebook_dim == 32
                 and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
                 and cfg.channel_protect_count == 64 and cfg.recon_loss_type == 'mse',
                 'Fixed native geometry changed.')
    return cat, training, vae, resolved


def rng_record():
    """Small fingerprints at the common, explicitly reset training boundary."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state()
    return {'torch_cpu': tensor_record(cpu), 'torch_cuda': tensor_record(cuda),
            'python': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()}


def materialize_targets(model, base_state, record, device):
    started = time.time()
    proof = read(record['source_geometry_path'])
    _require(proof['status'] == 'verified' and proof['aggregate']['spec_equal_count'] == 72,
             'Require the real source geometry comparison.')
    layers = [(name, model.get_submodule(name)) for name in NAMES]
    for name, layer in layers:
        _require(isinstance(layer, VAELinear) and layer.bias is None and not layer.has_low_rank_residual()
                 and layer.residual_stages == 2 and layer.parallel_parts == 1,
                 'Materialize the genuine unaugmented deployed Q/K: ' + name)
    if record['target_kind'] == 'decoded':
        # Full-category ordering and group8 match the native deployment cache.
        with torch.no_grad():
            decoded = decode_named_vae_linear_weights(layers, dtype=torch.bfloat16,
                group_size=8, compute_device=device, respect_cache_policy=False)
        targets = {item.name: item.decoded_weight.detach().to('cpu').contiguous() for item in decoded}
        del decoded
        for _, layer in layers:
            layer.clear_decoded_weight_cache(); layer.to('cpu')
        gc.collect(); torch.cuda.empty_cache()
    else:
        hf = Path(record['architecture']['base_model_path'])
        index = read(hf / 'model.safetensors.index.json')['weight_map']
        targets = {}
        for name in NAMES:
            key = name + '.weight'
            with safe_open(str(hf / index[key]), framework='pt', device='cpu') as handle:
                targets[name] = handle.get_tensor(key).contiguous()
    _require(set(targets) == set(NAMES), 'Materialization must cover the actual72 targets.')
    plans, identities = {}, {}
    for name in NAMES:
        target = targets[name]; out_features = 4096 if name.endswith('.q_proj') else 1024
        _require(target.dtype == torch.bfloat16 and tuple(target.shape) == (out_features, 4096)
                 and bool(torch.isfinite(target).all()), 'Bad deployed/HF target: ' + name)
        indices = base_state[name + '.protected_input_indices'].clone()
        prior = proof['projections'][name]
        _require(tensor_record(indices) == prior['protection']['protected_input_indices']['f927']
                 == prior['protection']['protected_input_indices']['hf_fresh_block32'],
                 'Use the actual common64 input plan, not a new ranking.')
        q, scales = encode_protected_channel_weight(target[:, indices].T.contiguous(), quant_format='int8')
        expected_kind = 'f927' if record['target_kind'] == 'decoded' else 'hf_fresh_block32'
        _require(tensor_record(q) == prior['protection']['protected_input_qvalues'][expected_kind]
                 and tensor_record(scales) == prior['protection']['protected_input_scales'][expected_kind],
                 'Actual target protection re-encoding differs from its measured source: ' + name)
        plans[name] = indices
        identities[name] = {'target': tensor_record(target), 'protected_input_indices': tensor_record(indices),
                            'protected_input_qvalues': tensor_record(q), 'protected_input_scales': tensor_record(scales)}
    # Every real Linear is allocated before the common RNG reset. No initialization
    # side effect from model loading, decoding or host allocation enters CAT.
    for name in NAMES:
        target = targets[name]
        linear = nn.Linear(target.shape[1], target.shape[0], bias=False, dtype=torch.bfloat16)
        linear.weight = nn.Parameter(target, requires_grad=False)
        set_module_by_name(model, name, linear)
    del layers, targets
    gc.collect(); torch.cuda.empty_cache()
    return plans, {'target_kind': record['target_kind'], 'targets': identities,
        'elapsed_seconds': time.time() - started, 'target_bytes': 1509949440,
        'native_decode_group_size': 8 if record['target_kind'] == 'decoded' else None,
        'original_coordinates': True, 'second_inverse_applied': False,
        'protected_payload_source': 'Native encoding of the materialized target, not a copied INT8 payload.'}


def train(record, device):
    cat, training, vae, resolved = parsed(record)
    _require(ARMS[record['arm']] == (record['target_kind'], record['block_size'])
             and record['steps_per_stage'] in (2, 10000), 'Invalid bounded adapter arm.')
    configure_deterministic_mode(True)
    started = time.time(); torch.cuda.reset_peak_memory_stats(device)
    model, meta, _ = load_v6_model_checkpoint(record['base_dir'], map_location='cpu', strict=True,
                                            expected_kind='final_model')
    _require(meta['checkpoint_id'] == record['base_checkpoint_id'], 'Wrong real f927 source.')
    base_state = torch.load(Path(record['base_dir']) / meta['state_dict_file'], map_location='cpu',
                            weights_only=True, mmap=True)
    _require(len(base_state) == 3423, 'Wrong source persistent inventory.')
    _assert_state_equal(model.state_dict(), base_state, 'Fresh strict f927 before materialization')
    plans, materialized = materialize_targets(model, base_state, record, device)
    runtime = load_v6_cat_runtime_state(record['activation_checkpoint_dir'], required=True, map_location='cpu')
    activation, _, _ = restore_cat_runtime_state(runtime)
    _require(activation['dataset'] == 'alpaca=1' and activation['nsamples'] == 512
             and activation['seqlen'] == 256 and activation['seed'] == 31
             and set(activation['stats_by_linear']) == set(NAMES), 'Wrong retained HF calibration statistics.')
    set_seed(31)
    reset = rng_record(); categories = {}
    for category_idx, category in enumerate(CATEGORIES):
        names = [name for name in NAMES if name.endswith('.' + category)]
        refs = [LinearRef(name, model.get_submodule(name), category, category == 'q_proj') for name in names]
        before_rng = rng_record(); category_start = time.time()
        payload = train_group_vae_payload(model=model, group_refs=refs, group_tag=category + '.L0-35',
            runtime_cfg=resolved[category], vae_args=vae, training_args=training,
            train_device=str(device), convert_device=str(device), do_convert=True,
            batch_size=8192, log_every=1 if record['steps_per_stage'] == 2 else 100,
            eval_every=0, eval_blocks=256, gpu_resident_data=bool(getattr(cat, 'gpu_resident_data', False)),
            activation_runtime=activation, channel_protect_mode='channel', channel_plan=plans,
            channel_scope='layer', channel_rank_metric='channel_weight_actmean_abs',
            channel_axis='input', channel_quant='int8', channel_min_per_layer=int(cat.channel_min_per_layer),
            deterministic=True, shuffle_seed=31 + category_idx * 100000)
        _require(payload is not None and len(payload['all_stage_bits']) == 2
                 and len(payload['all_stage_decoders']) == 2, 'Missing genuine native two-stage payload.')
        apply_group_vae_payload(model=model, group_refs=refs, group_tag=category + '.L0-35',
                                payload=payload, convert_device=str(device))
        categories[category] = {'entry_rng': before_rng, 'exit_rng': rng_record(),
            'elapsed_seconds': time.time() - category_start, 'linears': 36,
            'steps_per_stage': record['steps_per_stage'], 'stages': 2,
            'shuffle_seeds': [31 + category_idx * 100000, 32 + category_idx * 100000]}
        del payload, refs
        gc.collect(); torch.cuda.empty_cache()
    model.eval()
    report = {'materialization': materialized, 'post_materialization_seed31_rng': reset,
        'categories': categories, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'elapsed_seconds': time.time() - started,
        'training_target': record['target_kind'], 'shared_recipe': record['normalized_expected'],
        'source_kind': 'Fresh immutable f927 complement; fresh native CAT VAE initialization, not resume.',
        'no_full8_in_mechanism': record['steps_per_stage'] == 2}
    return model, base_state, report

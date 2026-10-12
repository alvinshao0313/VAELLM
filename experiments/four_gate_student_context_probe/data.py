"""Capture four preselected gate inputs in one frozen f927 forward per document."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import time

import torch
from safetensors import safe_open

from experiments.gate_student_context_probe.data import (
    SOURCE, SOURCE_ID, SPLITS, prepare_data as prepare_l18, records,
)
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


LAYERS = (0, 9, 18, 27)
TRAIN_LAYERS = (0, 9, 27)
FORMAT = 'four_gate_student_context_inputs_v1'


def target_name(index):
    require(index in LAYERS, 'Use only the four layers selected before this experiment.')
    return f'model.layers.{index}.mlp.gate_proj'


def cache_records(cache):
    return {'inputs': {split: tensor_record(value) for split, value in cache['inputs'].items()},
        **{key: tensor_record(cache[key]) for key in ('student_up_weight', 'student_down_weight')}}


def prepare_data(base_root):
    """Reproduce old tokens and read real per-layer states and BF16 HF matrices on CPU."""
    base = Path(base_root).resolve()
    old, batches, old_archive = prepare_l18(base)
    proto = base / 'gate_student_context_probe'
    original = json.loads((proto / 'run/report.json').read_text())
    review = json.loads((proto / 'review.json').read_text())
    require(original['status'] == 'completed' and (proto / 'run.exit').read_text().strip() == '0'
            and sha256(proto / 'run/report.json') == '889c0c6e40ad9ae7f678a78a0962f6f6659aea7d5e1582021474e62212e39158'
            and review['status'] == 'verified' and review['report_sha256'] == sha256(proto / 'run/report.json'),
            'Require the completed real-student L18 control.')
    old_manifest_path = proto / 'run/student_capture/data_manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    old_cache_path = proto / 'run/student_capture/inputs.pt'
    require(old_manifest['status'] == 'captured' and old_manifest['checkpoint_id'] == SOURCE_ID
            and sha256(old_cache_path) == old_manifest['archive']['sha256'], 'Original student cache identity changed.')
    control = original['arms']['original_codes']
    require(sha256(control['state_file']) == control['state_sha256'], 'Retained original-code gate500 endpoint changed.')
    meta = json.loads((SOURCE / 'checkpoint_meta.json').read_text())
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require(meta['checkpoint_id'] == SOURCE_ID and len(state) == old['expected_serialized_tensors'] == 3423,
            'Use the original complete f927 state.')
    specs = {row['name']: row for row in meta['converted_modules']}
    hf_path = Path(meta['base_model_path'])
    index = json.loads((hf_path / 'model.safetensors.index.json').read_text())['weight_map']
    per_layer, archives = {}, {}
    for layer in LAYERS:
        target = target_name(layer)
        selected = {kind: specs[target.replace('gate_proj', kind)] for kind in ('gate_proj', 'up_proj', 'down_proj')}
        for kind, spec in selected.items():
            down = kind == 'down_proj'
            input_size, output_size, count, bits = (12288, 4096, 256, 48) if down else (4096, 12288, 64, 32)
            require((spec['in_features'], spec['out_features'], spec['compressed_in_features'])
                    == (input_size, output_size, input_size - count)
                    and spec['transpose'] == down and spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                    and spec['parallel_stage_decode'] and spec['stage_codebook_dims'] == [32, 32]
                    and spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
                    and spec['protected_residual_stages'] == 0 and not spec['has_bias'] and not spec['has_original_weight']
                    and all(spec[key] is None for key in ('low_rank_a', 'low_rank_b', 'sparse_residual_values', 'sparse_residual_qvalues'))
                    and spec['weight_rotation']['type'] == 'two_sided_hadamard' and spec['weight_rotation']['block_size'] == 32
                    and all((d['in_dim'], d['out_dim'], d['hidden_dim'], d['num_res_blocks'], d['param_dtype'])
                            == (bits, 32, 128, 0, 'bfloat16') for d in spec['stage_decoders']),
                    'Unsupported actual native layer geometry: ' + spec['name'])
            protected = state[spec['name'] + '.protected_input_indices']
            require(protected.dtype == torch.int64 and tuple(protected.shape) == (count,)
                    and protected.unique().numel() == count and int(protected.min()) >= 0 and int(protected.max()) < input_size,
                    'Invalid current source protection indices: ' + spec['name'])
        expected = {key[len(target) + 1:]: tensor_record(value) for key, value in state.items() if key.startswith(target + '.')}
        require(len(expected) == 13, 'Every original gate must retain thirteen persistent tensors.')
        weights, hf_records = {}, {}
        for kind in selected:
            key = target.replace('gate_proj', kind) + '.weight'
            with safe_open(str(hf_path / index[key]), framework='pt', device='cpu') as handle:
                value = handle.get_tensor(key)
            require(value.dtype == torch.bfloat16 and tuple(value.shape)
                    == (selected[kind]['out_features'], selected[kind]['in_features']), 'HF original weight differs: ' + key)
            weights[kind] = value
            hf_records[kind] = {'key': key, 'shard': index[key], **tensor_record(value)}
        archives[layer] = {'targets': {target: {'reference_weight': weights['gate_proj']},
            target.replace('gate_proj', 'down_proj'): {'reference_weight': weights['down_proj']}},
            'up_proj_weight': weights['up_proj']}
        per_layer[str(layer)] = {'target': target, 'expected_gate_state': expected, 'specs': selected, 'hf_weights': hf_records}
    l18 = target_name(18)
    require(torch.equal(archives[18]['targets'][l18]['reference_weight'], old_archive['targets'][l18]['reference_weight'])
            and torch.equal(archives[18]['up_proj_weight'], old_archive['up_proj_weight'])
            and torch.equal(archives[18]['targets'][l18.replace('gate_proj', 'down_proj')]['reference_weight'],
                            old_archive['targets'][l18.replace('gate_proj', 'down_proj')]['reference_weight']),
            'All three original L18 HF matrices must exactly match the old objective.')
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    for path in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit',
                 old_manifest_path, old_cache_path, Path(control['state_file'])):
        inputs[str(path)] = sha256(path)
    record = {'status': 'cpu_ready', 'format': FORMAT, 'source_checkpoint': str(SOURCE), 'checkpoint_id': SOURCE_ID,
        'source_weight_stat': old['source_weight_stat'], 'expected_serialized_tensors': 3423,
        'source_sha256': sources, 'input_sha256': inputs, 'hf_weight_files': old['hf_weight_files'],
        'model_path': str(hf_path), 'model_policy': old['model_policy'], 'layers': list(LAYERS), 'train_layers': list(TRAIN_LAYERS),
        'per_layer': per_layer, 'document_lengths': old['document_lengths'], 'documents': old['documents'],
        'token_batches': old['token_batches'], 'old_cache_path': str(old_cache_path), 'old_cache_tensors': old_manifest['tensors'],
        'original_report': str(proto / 'run/report.json'), 'control_state_path': control['state_file'],
        'control_state_sha256': control['state_sha256'], 'control_endpoint': control['endpoint'],
        'l18_hf_original_three_weights_exact': True,
        'protocol': {'seed': 137, 'max_length': 256, 'microbatch_size': 1, 'split_documents': {s: 32 for s in SPLITS},
            'capture': 'One untouched f927 forward per document, four hooks, stop before L27 gate computation.',
            'fit': 'Independent layer0/9/27 gate-only500; reuse verified L18 gate-only500 without retraining.',
            'population': 'The same already-used train32/held32/replay32, no new blind-test claim.',
            'target': 'Per-layer HF_MLP(xs) on fixed original-student inputs; no teacher forward or rolling recapture.'}}
    check_hashes(sources); check_hashes(inputs); check_stat(record['source_weight_stat'])
    return record, batches, archives


class _AllInputsCaptured(Exception):
    pass


@torch.no_grad()
def capture(record, batches, archives, output, device):
    from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    output, device = Path(output).resolve(), torch.device(device)
    require(not output.exists() and record['checkpoint_id'] == SOURCE_ID and device.type == 'cuda'
            and torch.cuda.device_count() == 1, 'Use a new capture directory and one assigned GPU.')
    require({s: {k: tensor_record(v) for k, v in batch.items()} for s, batch in batches.items()}
            == record['token_batches'], 'Runtime document tokens changed.')
    configure_deterministic_mode(True); set_seed(137); torch.set_num_threads(4)
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True)
    manifest = {**record, 'status': 'capturing', 'started_unix': time.time()}
    write_json(output / 'manifest.json', manifest)
    handles = []; model = None
    try:
        model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], strict=True, map_location='cpu', expected_kind='final_model')
        require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh source strict load failed.')
        model.eval().requires_grad_(False); model.config.use_cache = False
        before = records(model)
        require(len(before) == record['expected_serialized_tensors'], 'Source tensor inventory changed.')
        warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
        policy = record['model_policy']
        require(model.config.model_type == policy['model_type'] and model.config._attn_implementation == policy['attn_implementation']
                and model.config.rope_theta == policy['rope_theta'] and model.config.rope_scaling == policy['rope_scaling']
                and {p.dtype for p in model.parameters()} == {torch.bfloat16}, 'Native model/backend/precision policy changed.')
        rope = {name: tensor_record(value) for name, value in model.named_buffers() if 'inv_freq' in name}
        require(rope and all(row['dtype'] == 'torch.float32' for row in rope.values()), 'Keep native FP32 RoPE buffers.')
        gates = {i: model.get_submodule(target_name(i)) for i in LAYERS}
        for i, gate in gates.items():
            require(isinstance(gate, VAELinear) and records(gate) == record['per_layer'][str(i)]['expected_gate_state']
                    and isinstance(model.get_submodule(target_name(i).rsplit('.', 1)[0]).act_fn, torch.nn.SiLU),
                    'Wrong native per-layer gate or activation.')
        inputs = {i: {s: torch.empty((sum(record['document_lengths'][s]), 4096), dtype=torch.bfloat16)
                       for s in SPLITS} for i in LAYERS}
        seen, current_mask, current_split, offset = set(), None, None, 0

        def make_hook(index):
            def hook(_module, values):
                require(index not in seen, 'A target executed twice in one document.')
                x = values[0]
                require(x.dtype == torch.bfloat16 and x.ndim == 3 and x.shape[:2] == current_mask.shape
                        and x.shape[-1] == 4096, 'Unexpected student gate input geometry.')
                selected = x.detach()[current_mask].contiguous().cpu()
                require(bool(torch.isfinite(selected).all()), 'Nonfinite student input.')
                inputs[index][current_split][offset:offset + len(selected)].copy_(selected)
                seen.add(index)
                if len(seen) == len(LAYERS):
                    require(index == max(LAYERS), 'Unexpected target execution order.')
                    raise _AllInputsCaptured()
            return hook

        for i, gate in gates.items():
            handles.append(gate.register_forward_pre_hook(make_hook(i)))
        for split in SPLITS:
            current_split, offset = split, 0
            batch = batches[split]
            for row, length in enumerate(record['document_lengths'][split]):
                seen = set()
                ids = batch['input_ids'][row:row + 1].to(device)
                mask = batch['attention_mask'][row:row + 1].to(device)
                current_mask = mask.bool()
                require(int(current_mask.sum()) == length, 'Document mask/order changed.')
                try:
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                except _AllInputsCaptured:
                    require(seen == set(LAYERS), 'Capture stopped before every target.')
                else:
                    raise RuntimeError('The model did not execute every selected gate.')
                offset += length
            require(offset == len(inputs[0][split]), 'Incomplete capture split.')
        for handle in handles:
            handle.remove()
        handles = []
        caches = {}
        for i in LAYERS:
            weights = {kind: model.get_submodule(target_name(i).replace('gate_proj', kind))._decode_weight(dtype=torch.bfloat16).detach().cpu()
                       for kind in ('up_proj', 'down_proj')}
            require(tuple(weights['up_proj'].shape) == (12288, 4096) and tuple(weights['down_proj'].shape) == (4096, 12288),
                    'Native student up/down original-coordinate geometry changed.')
            caches[i] = {'checkpoint_id': SOURCE_ID, 'target': target_name(i), 'inputs': inputs[i],
                'document_lengths': record['document_lengths'], 'student_up_weight': weights['up_proj'], 'student_down_weight': weights['down_proj']}
        old_cache = torch.load(record['old_cache_path'], map_location='cpu', weights_only=True, mmap=True)
        require(cache_records(caches[18]) == record['old_cache_tensors'] == cache_records(old_cache),
                'Multi-hook L18 inputs or native up/down differ from the completed control; do not reuse its endpoint.')
        require(records(model) == before and all(not p.requires_grad for p in model.parameters()), 'Frozen capture changed original source state.')
        clear_model_vae_linear_cache(model)
        gates_cpu = {i: deepcopy(gates[i]).to('cpu').eval().requires_grad_(False) for i in LAYERS}
        for i, gate in gates_cpu.items():
            gate.clear_decoded_weight_cache(); gate._build_parallel_stage_decode_plan()
            require(records(gate) == record['per_layer'][str(i)]['expected_gate_state'], 'Detached gate state differs from its actual source.')
        saved = {}
        for i in TRAIN_LAYERS:
            path = output / f'layer_{i}_inputs.pt'
            torch.save(caches[i], path)
            saved[str(i)] = {**file_stat(path), 'sha256': sha256(path)}
        saved['18'] = {'path': record['old_cache_path'], 'sha256': sha256(record['old_cache_path']), 'reused_after_exact_capture': True}
        check_hashes(record['source_sha256']); check_hashes(record['input_sha256']); check_stat(record['source_weight_stat'])
        for item in record['hf_weight_files']:
            check_stat(item)
        manifest.update(status='captured', finished_unix=time.time(), l18_input_and_context_exact=True,
            source_persistent_state_unchanged=True, persistent_tensors_compared=len(before), native_rope_buffers=rope,
            model_policy=policy, warmup=warmup, whole_model_dtype_cast=False,
            source_gate_states={str(i): records(gate) for i, gate in gates_cpu.items()},
            tensors={str(i): cache_records(cache) for i, cache in caches.items()}, archives=saved,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(output / 'manifest.json', manifest)
        return gates_cpu, caches, manifest
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect(); torch.cuda.empty_cache()

"""Replay the fixed 96 documents through the frozen deployed f927 student."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from experiments.linear_output.calibration import capture_inputs
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.middle_exact_decoder_probe.data import prepare_confirmation
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.rotation_factor_probe.capture import require, tensor_record
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
TARGET = 'model.layers.18.mlp.gate_proj'
FORMAT = 'gate_student_context_inputs_v1'
SPLITS = ('train', 'heldout', 'replay32')


def records(module):
    return {name: tensor_record(value) for name, value in module.state_dict().items()}


def prepare_data(base_root):
    """CPU-only token replay and persistent source inventory; do not load a model."""
    base = Path(base_root).resolve()
    middle = base / 'rotation_factor_probe/middle_data/qwen'
    confirmation, replay = prepare_confirmation(base / 'rotation_factor_probe/data/qwen', middle)
    archive, manifest = load_archive(middle)
    require(archive['model_type'] == 'qwen3' and archive['layer_index'] == 18
            and TARGET in archive['targets'] and manifest['hidden_act'] == 'silu', 'Use the preselected Qwen L18 HF reference.')
    meta_path = SOURCE / 'checkpoint_meta.json'
    meta = json.loads(meta_path.read_text())
    require(meta['checkpoint_id'] == SOURCE_ID and meta['checkpoint_kind'] == 'final_model'
            and Path(meta['base_model_path']).resolve() == Path(archive['model_path']).resolve()
            and archive['checkpoint_id'] == SOURCE_ID, 'Original student/HF archive identity differs.')
    config = AutoConfig.from_pretrained(meta['base_model_path'], local_files_only=True, trust_remote_code=False)
    require(config.model_type == 'qwen3' and config.hidden_act == 'silu' and config.num_hidden_layers // 2 == 18,
            'This bounded diagnostic requires native Qwen3 L18 SiLU.')
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], local_files_only=True,
                                             use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    prefix, second_replay, data = build_calibration_batches(tokenizer, seed=137,
        train_documents=64, heldout_documents=32, max_length=256)
    batches = {'train': {key: value[:32] for key, value in prefix.items()},
               'heldout': {key: value[32:] for key, value in prefix.items()}, 'replay32': replay}
    require(data['train']['documents'] == manifest['data']['train']['documents'] + manifest['data']['heldout']['documents']
            and data['heldout']['documents'] == [{key: item[key] for key in data['heldout']['documents'][0]}
                                                for item in confirmation['documents']], 'The fixed 96 document identities differ.')
    require(all(torch.equal(replay[key], value) for key, value in second_replay.items()), 'The fixed replay32 token batch differs.')
    for split in ('train', 'heldout'):
        require({key: tensor_record(value) for key, value in batches[split].items()} == manifest['token_batches'][split],
                'Old32+32 actual token batches changed.')
    documents = {split: manifest['data'][split]['documents'] for split in ('train', 'heldout')}
    documents['replay32'] = confirmation['documents']
    lengths = {split: [int(n) for n in batches[split]['attention_mask'].sum(1).tolist()] for split in SPLITS}
    require(all(len(lengths[s]) == 32 and lengths[s] == [d['valid_tokens'] for d in documents[s]]
                and tuple(batches[s]['input_ids'].shape) == (32, 256) for s in SPLITS), 'Real document/token boundaries changed.')
    state_path = SOURCE / meta['state_dict_file']
    require(state_path.resolve().parent == SOURCE.resolve(), 'The state path escaped its source checkpoint.')
    state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    gate_state = {key[len(TARGET) + 1:]: tensor_record(value) for key, value in state.items() if key.startswith(TARGET + '.')}
    specs = {item['name']: item for item in meta['converted_modules']}
    spec = specs[TARGET]
    require((spec['out_features'], spec['in_features'], spec['compressed_in_features']) == (12288, 4096, 4032)
            and spec['codebook_dim'] == 32 and spec['stage_codebook_dims'] == [32, 32]
            and spec['parallel_parts'] == 1 and spec['residual_stages'] == 2 and not spec['transpose']
            and spec['parallel_stage_decode'] and not spec['has_bias'] and not spec['has_original_weight']
            and spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
            and spec['weight_rotation'] == manifest['protection'][TARGET]['spec']['weight_rotation'],
            'Current student gate geometry differs from the intended D32/K32/S2 input-protected gate.')
    require(spec['weight_rotation']['type'] == 'two_sided_hadamard' and spec['weight_rotation']['block_size'] == 32
            and all(d['in_dim'] == d['out_dim'] == 32 and d['hidden_dim'] == 128 and d['num_res_blocks'] == 0
                    and d['param_dtype'] == 'bfloat16' for d in spec['stage_decoders']), 'Original rotation/decoder topology changed.')
    indices = state[TARGET + '.protected_input_indices']
    require(indices.dtype == torch.int64 and tuple(indices.shape) == (64,) and indices.unique().numel() == 64
            and int(indices.min()) >= 0 and int(indices.max()) < 4096
            and torch.equal(indices, archive['targets'][TARGET]['protected_input_indices']), 'Frozen original protection indices differ.')
    for category in ('gate_proj', 'up_proj', 'down_proj'):
        current = specs[TARGET.replace('gate_proj', category)]
        require(not current['has_bias'] and not current['has_original_weight']
                and current['protected_residual_stages'] == 0
                and all(current[key] is None for key in ('low_rank_a', 'low_rank_b', 'sparse_residual_values',
                                                         'sparse_residual_qvalues')),
                'The three native MLP projections must have no extra activation patch or original-weight branch.')
    sources = dict(confirmation['source_sha256'])
    for relative in ('experiments/gate_student_context_probe/data.py', 'experiments/linear_output/calibration.py',
        'tools/cat_eval.py', 'train_utils/v6_model_loader.py', 'train_utils/checkpoint_v6.py',
        'e2e_common/proxy_trainables.py', 'litebsq/vae_linear.py', 'litebsq/autoencoder.py',
        'litebsq/bitpack.py', 'litebsq/fused_multistage_decoder.py', 'litebsq/packed_bit_linear.py',
        'litebsq/parallel_layers.py', 'litebsq/protected_channel_quant.py', 'litebsq/sparse_residual.py',
        'rotation/model_utils.py', 'rotation/weight_preprocess.py'):
        path = ROOT / relative
        require(str(path) not in sources or sources[str(path)] == sha256(path), 'An existing source pin changed.')
        sources[str(path)] = sha256(path)
    inputs = dict(confirmation['input_sha256'])
    inputs[str(meta_path)] = sha256(meta_path)
    record = {'format': FORMAT, 'status': 'cpu_ready', 'source_checkpoint': str(SOURCE), 'checkpoint_id': SOURCE_ID,
        'source_weight_stat': file_stat(state_path), 'source_sha256': sources, 'input_sha256': inputs,
        'model_path': meta['base_model_path'], 'archive_dir': str(middle), 'target': TARGET,
        'expected_gate_state': gate_state, 'expected_serialized_tensors': len(state), 'target_spec': spec,
        'up_spec': specs[TARGET.replace('gate_proj', 'up_proj')], 'down_spec': specs[TARGET.replace('gate_proj', 'down_proj')],
        'document_lengths': lengths, 'documents': documents,
        'token_batches': {split: {key: tensor_record(value) for key, value in batch.items()} for split, batch in batches.items()},
        'hf_weight_files': confirmation['hf_weight_files'], 'tokenizer': confirmation['tokenizer'],
        'model_policy': {'model_type': 'qwen3', 'attn_implementation': 'flash_attention_2',
            'rope_theta': config.rope_theta, 'rope_scaling': config.rope_scaling, 'parameter_dtype': 'torch.bfloat16',
            'inv_freq_dtype': 'torch.float32', 'hidden_act': config.hidden_act},
        'protocol': {'seed': 137, 'max_length': 256, 'microbatch_size': 1, 'split_documents': {s: 32 for s in SPLITS},
            'fit_split': 'train', 'evaluation_splits': ['heldout', 'replay32'],
            'scope': 'All 96 are already-used local development/replay documents. Student contexts are fresh captures; no claim of unseen evaluation.',
            'target': 'Original HF MLP applied to the same captured student input; no full teacher model forward.'}}
    del state
    check_hashes(sources)
    check_hashes(inputs)
    check_stat(record['source_weight_stat'])
    return record, batches, archive


@torch.no_grad()
def capture(record, batches, archive, output, device):
    """Return one real native student gate and frozen student contexts/up/down."""
    from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    output, device = Path(output).resolve(), torch.device(device)
    require(record['status'] == 'cpu_ready' and record['checkpoint_id'] == SOURCE_ID
            and device.type == 'cuda' and torch.cuda.device_count() == 1, 'Use one assigned GPU and the prepared original student.')
    require(not output.exists() and not output.is_relative_to(SOURCE)
            and not output.is_relative_to(Path(record['archive_dir'])), 'Capture output must be new and outside source data.')
    require(archive['checkpoint_id'] == SOURCE_ID and archive['model_type'] == 'qwen3', 'HF archive identity differs.')
    require({s: {k: tensor_record(v) for k, v in b.items()} for s, b in batches.items()} == record['token_batches'],
            'Runtime tokens differ from preparation.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256']); check_stat(record['source_weight_stat'])
    for item in record['hf_weight_files']:
        check_stat(item)
    configure_deterministic_mode(True)
    set_seed(137)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True)
    manifest = {**record, 'status': 'capturing', 'started_unix': time.time()}
    write_json(output / 'data_manifest.json', manifest)
    model = gate = up = down = None
    try:
        model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], strict=True, map_location='cpu',
                                                      expected_kind='final_model')
        require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
                'Fresh native student strict load failed.')
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        before = records(model)
        require(len(before) == record['expected_serialized_tensors'], 'Loaded persistent inventory differs.')
        gate = model.get_submodule(TARGET)
        up = model.get_submodule(TARGET.replace('gate_proj', 'up_proj'))
        down = model.get_submodule(TARGET.replace('gate_proj', 'down_proj'))
        require(all(isinstance(layer, VAELinear) and layer.bias is None for layer in (gate, up, down))
                and records(gate) == record['expected_gate_state'], 'The actual gate/up/down are not the original compressed projections.')
        warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
        policy = record['model_policy']
        require(model.config.model_type == policy['model_type'] and model.config._attn_implementation == policy['attn_implementation']
                and model.config.rope_theta == policy['rope_theta'] and model.config.rope_scaling == policy['rope_scaling']
                and {p.dtype for p in model.parameters()} == {torch.bfloat16}, 'Native BF16/backend/RoPE policy changed.')
        rope = {name: tensor_record(value) for name, value in model.named_buffers() if 'inv_freq' in name}
        require(rope and all(row['dtype'] == 'torch.float32' for row in rope.values()), 'Original FP32 RoPE buffers changed.')
        require(isinstance(model.get_submodule(TARGET.rsplit('.', 1)[0]).act_fn, torch.nn.SiLU), 'Native activation is not SiLU.')
        captured = {}
        for split in SPLITS:
            captured[split] = capture_inputs(model, TARGET, batches[split], microbatch_size=1).detach().cpu()
            require(captured[split].dtype == torch.bfloat16 and tuple(captured[split].shape)
                    == (sum(record['document_lengths'][split]), 4096), 'Captured token order/geometry/dtype differs.')
        up_weight = up._decode_weight(dtype=torch.bfloat16).detach().cpu()
        down_weight = down._decode_weight(dtype=torch.bfloat16).detach().cpu()
        require(up_weight.dtype == down_weight.dtype == torch.bfloat16 and tuple(up_weight.shape) == (12288, 4096)
                and tuple(down_weight.shape) == (4096, 12288), 'Native full-coordinate up/down decode differs.')
        require(records(model) == before and all(not p.requires_grad for p in model.parameters()),
                'Capture/decode mutated original persistent student state or trainability.')
        clear_model_vae_linear_cache(model)
        gate_cpu = deepcopy(gate).to('cpu').eval().requires_grad_(False)
        gate_cpu.clear_decoded_weight_cache()
        gate_cpu._build_parallel_stage_decode_plan()
        require(records(gate_cpu) == record['expected_gate_state']
                and all(p.dtype == torch.bfloat16 for p in gate_cpu._parallel_stage_decoder.parameters()),
                'Detached native gate persistent state differs from f927.')
        cache = {'format': FORMAT, 'checkpoint_id': SOURCE_ID, 'target': TARGET, 'inputs': captured,
                 'document_lengths': record['document_lengths'], 'student_up_weight': up_weight,
                 'student_down_weight': down_weight}
        manifest.update(runtime={'loader': 'train_utils.v6_model_loader.load_v6_model_checkpoint(strict=True)',
            'warmup': warmup, 'model_policy': policy, 'rope_buffers': rope, 'whole_model_dtype_cast': False,
            'eval': True, 'frozen': True, 'use_cache': False, 'microbatch_size': 1,
            'source_persistent_state_unchanged': True, 'persistent_tensors_compared': len(before),
            'source_state': before, 'gate_state': records(gate_cpu),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)},
            tensors={'inputs': {s: tensor_record(x) for s, x in captured.items()},
                     'student_up_weight': tensor_record(up_weight), 'student_down_weight': tensor_record(down_weight)})
        check_hashes(record['source_sha256']); check_hashes(record['input_sha256']); check_stat(record['source_weight_stat'])
        for item in record['hf_weight_files']:
            check_stat(item)
        torch.save(cache, output / 'inputs.pt')
        manifest.update(status='captured', finished_unix=time.time(),
                        archive={**file_stat(output / 'inputs.pt'), 'sha256': sha256(output / 'inputs.pt')})
        write_json(output / 'data_manifest.json', manifest)
        return gate_cpu, cache, manifest
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        write_json(output / 'data_manifest.json', manifest)
        raise
    finally:
        del model, gate, up, down
        gc.collect()
        torch.cuda.empty_cache()

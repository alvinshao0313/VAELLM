"""Isolated, self-contained mixed native/compact model format, explicitly not v6."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import uuid

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_utils import no_init_weights

from experiments.adaptive_vector_rvq.common import read, require, sha, write
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as native
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.distill_precision import install_precision_runtime, restore_parameter_dtypes
from . import RUNTIME
from .consumer import CompactRVQLinear

FORMAT = 'vaellm_adaptive_native_model_v2'
STATE_FILE = 'pytorch_model.bin'
META_FILE = 'adaptive_model_meta.json'


def compact_from_state(spec, state):
    if spec.get('consumer') == 'fixed2_rice7':
        from experiments.adaptive_class_20261030.sparse import Fixed2RiceLinear
        return Fixed2RiceLinear.from_state(spec, state)
    require('consumer' not in spec, 'Unknown actual compact consumer: ' + str(spec.get('consumer')))
    return CompactRVQLinear.from_state(spec, state)


def runtime_identity():
    paths = [Path(native.__file__).resolve(), RUNTIME / 'rotation/hadamard_utils.py',
             RUNTIME / 'rotation/hadamard_extended.py', RUNTIME / 'rotation/weight_preprocess.py']
    return {str(path): sha(path) for path in paths}


def architecture(config_dir, runtime):
    # Architecture comes only from this directory's config, never pretrained payloads.
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True, trust_remote_code=False)
    require(config.model_type == 'qwen3', 'This first real complete-model prototype covers the verified Qwen3 architecture.')
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
            attn_implementation=runtime['attn_implementation'], trust_remote_code=False)
    model.seqlen = int(runtime['seqlen'])
    generation = Path(config_dir) / 'generation_config.json'
    if generation.is_file():
        model.generation_config = GenerationConfig.from_pretrained(str(config_dir), local_files_only=True)
    require(all(tensor.dtype == torch.float32 for name, tensor in model.named_buffers() if 'inv_freq' in name),
            'Keep original FP32 RoPE buffers; do not cast the complete model.')
    return model


def load_parent(directory, meta):
    require(meta['format'] == native.FORMAT_V6 and not meta.get('shared_protected_residual_decoders')
            and not meta.get('post_norm_head_linear') and not (meta.get('extra_meta') or {}).get('residual_lora'),
            'Use the real G09 rank0 topology without unrelated extended topology.')
    model = architecture(directory, dict(attn_implementation='flash_attention_2', seqlen=2048))
    native._rebuild_converted_modules(model, meta['converted_modules'])
    state = torch.load(Path(directory) / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    restore_parameter_dtypes(model, state)
    result = model.load_state_dict(state, strict=True)
    native.refresh_vae_linear_runtime_after_state_load(model)
    install_precision_runtime(model)
    return model.eval().requires_grad_(False), state, result


def cpu_deployment_state(model):
    """Preserve complete serialized storages/aliases while casting only compact masters."""
    state = model.state_dict()
    compact_decoders = tuple(name + '.decoders.' for name, module in model.named_modules() if isinstance(module, CompactRVQLinear))
    storages, output = {}, {}
    for name, tensor in state.items():
        value = tensor.detach()
        if name.startswith(compact_decoders):
            # Match the already verified standalone codec's BF16 export, not master dtype.
            output[name] = value.to(device='cpu', dtype=torch.bfloat16).contiguous().clone()
            continue
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ('empty', name)
        if key not in storages:
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
            storages[key] = raw.to('cpu').clone()
        output[name] = torch.empty(0, dtype=value.dtype).set_(storages[key].untyped_storage(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
    return output


def tensor_table(state):
    return {name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype), bytes=tensor.numel() * tensor.element_size())
            for name, tensor in state.items()}


def _native_audit_meta(meta):
    return dict(format=native.FORMAT_V6, schema_version=6, checkpoint_kind='final_model', lora_config=None,
                checkpoint_id=meta['checkpoint_id'], converted_modules=meta['native_modules'],
                compressed_targets=[item['name'] for item in meta['native_modules']],
                pending_dense_targets=[item['name'] for item in meta['pending_dense_modules']], skip_targets=[])


def inventories(model):
    categories = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    groups = dict(native=[], compact=[], pending_dense=[])
    for name, module in model.named_modules():
        if not any(name.endswith('.' + category) for category in categories):
            continue
        if isinstance(module, VAELinear):
            groups['native'].append(name)
        elif isinstance(module, CompactRVQLinear):
            groups['compact'].append(name)
        else:
            require(type(module) is torch.nn.Linear, 'Stable projection is neither native, compact nor ordinary dense: ' + name)
            groups['pending_dense'].append(name)
    names = [name for group in groups.values() for name in group]
    require(len(names) == len(set(names)) == int(model.config.num_hidden_layers) * len(categories),
            'The three mutually exclusive inventories must cover every actual model projection.')
    return groups


def _unique_bytes(state):
    seen = set()
    total = 0
    for name, tensor in state.items():
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ('empty', name)
        if identity not in seen:
            total += storage.nbytes()
            seen.add(identity)
    return total


def ledger(meta, state, directory=None):
    require(tensor_table(state) == meta['tensor_table'], 'Full tensor inventory/dtype/shape differs from the manifest.')
    compact_names = {item['name'] for item in meta['compact_modules']}
    native_state = {name: tensor for name, tensor in state.items() if not any(name.startswith(prefix + '.') for prefix in compact_names)}
    base = audit_state_dict(_native_audit_meta(meta), native_state, max_bpw=100., max_lora_rank=0)
    require(not base['contract_violations'] and not base['unclassified_target_keys'], 'Native subset has an invalid or unclassified payload.')
    components = deepcopy(base['components'])
    denominator, logical, packed = base['original_linear_weights'], base['user_linear_bits'], base['packed_linear_bits']
    rows = {}
    compact_storage = set()
    for item in meta['compact_modules']:
        prefix, spec = item['name'], item['spec']
        values = {name[len(prefix) + 1:]: tensor for name, tensor in state.items() if name.startswith(prefix + '.')}
        # Re-resolve from the actual payload, including routes/padding, rather than trusting a claimed bit count.
        layer = compact_from_state(spec, values)
        require(set(values) == set(layer.state_dict()), 'Compact target has omitted or extra persistent fields.')
        target_bits = target_packed = 0
        stage_bits = {f'codes_{index}': int(stage['codes']['logical_shape'][0]) * int(stage['decoder_spec']['in_dim'])
                      for index, stage in enumerate(spec['stages'])}
        target_components = defaultdict(int)
        for name, tensor in values.items():
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ('empty', prefix, name)
            require(identity not in compact_storage and storage.nbytes() == tensor.numel() * tensor.element_size(),
                    'Compact fields must have one compact, unaliased persisted payload each.')
            compact_storage.add(identity)
            physical = storage.nbytes() * 8
            bits = stage_bits.get(name, physical)
            if name in stage_bits:
                component = 'code'
            elif name.startswith('decoders.'):
                component = 'decoder'
            elif name.startswith('sparse_'):
                component = 'rice7_sparse_residual'
            elif name == 'length_frame':
                component = 'route_frame'
            elif name.startswith('weight_rotation.'):
                component = 'rotation'
            elif name.endswith('indices'):
                component = 'protected_indices'
            elif name.endswith('scales'):
                component = 'protected_scales'
            else:
                require(name == 'protected_input_qvalues', 'Unclassified compact deployment field: ' + name)
                component = 'protected_weights'
            entry = components.setdefault(component, dict(budget_bits=0, storage_bits=0, unique_storages=0, dtypes=[]))
            entry['budget_bits'] += bits
            entry['storage_bits'] += physical
            entry['unique_storages'] += 1
            entry['dtypes'] = sorted(set(entry['dtypes']) | {str(tensor.dtype).removeprefix('torch.')})
            target_components[component] += physical // 8
            target_bits += bits
            target_packed += physical
        count = layer.in_features * layer.out_features
        rows[prefix] = dict(original_linear_weights=count, logical_bits=target_bits, packed_bits=target_packed,
                            components_bytes=dict(target_components), vector_dim=spec['vector_dim'],
                            code_row_bytes=[(stage['decoder_spec']['in_dim'] + 7) // 8 for stage in spec['stages']],
                            n_vectors=spec['n_vectors'], length_histogram=torch.bincount(layer.lengths.long(), minlength=len(spec['stages']) + 1).tolist())
        denominator += count
        logical += target_bits
        packed += target_packed
    pending_count = len(meta['pending_dense_modules'])
    result = dict(original_linear_weights=denominator, target_count=len(meta['native_modules']) + len(meta['compact_modules']) + pending_count,
        native_target_count=len(meta['native_modules']), compact_target_count=len(meta['compact_modules']),
        pending_dense_target_count=pending_count,
        user_linear_bits=logical, user_linear_bpw=logical / denominator,
        packed_linear_bits=packed, packed_linear_bpw=packed / denominator, code_padding_bits=packed - logical,
        full_model_tensor_bytes=_unique_bytes(state), outside_linear_bits=base['outside_linear_bits'],
        components=dict(sorted(components.items())), compact_targets=rows,
        budget_passed=not pending_count and logical / denominator <= 2.6, final_budget_applicable=not pending_count, max_bpw=2.6,
        scope='All real projection codes, BF16 decoders/norms, protected q/scales/indices, rotation and complete route frame. Nonprojection tensors and entire file bytes reported separately.')
    require(result['full_model_tensor_bytes'] == packed // 8 + sum(base['outside_linear_bits'].values()) // 8,
            'Complete tensor bytes are not covered by the Linear/outside scopes.')
    if directory is not None:
        directory = Path(directory)
        files = {path.name: path.stat().st_size for path in directory.iterdir() if path.is_file()}
        require(set(files) == set(meta['files']), 'Checkpoint contains unexpected files or misses its manifest inventory.')
        result.update(files_bytes=files, complete_directory_bytes=sum(files.values()),
                      serialization_overhead_bytes=files[STATE_FILE] - result['full_model_tensor_bytes'],
                      metadata_and_config_bytes=sum(size for name, size in files.items() if name != STATE_FILE),
                      complete_directory_bits_per_original_linear_weight=8 * sum(files.values()) / denominator)
    return result


def save(model, directory, *, provenance):
    directory = Path(directory)
    require(not directory.exists(), 'Never overwrite a model/checkpoint or an earlier validation result.')
    native._reject_live_training_adapter_topology(model)
    groups = inventories(model)
    native_specs = native._collect_vae_linear_specs(model)
    compact = [dict(name=name, spec=module.metadata()) for name, module in model.named_modules() if isinstance(module, CompactRVQLinear)]
    pending = [dict(name=name, in_features=model.get_submodule(name).in_features,
                    out_features=model.get_submodule(name).out_features,
                    has_bias=model.get_submodule(name).bias is not None) for name in groups['pending_dense']]
    names = [item['name'] for item in native_specs + compact + pending]
    require(len(set(names)) == len(names) and not any(spec.get('has_original_weight') for spec in native_specs),
            'Every projection needs exactly one active compressed payload and no dense backup.')
    state = cpu_deployment_state(model)
    directory.mkdir(parents=True)
    model.config.to_json_file(directory / 'config.json', use_diff=False)
    files = ['config.json', STATE_FILE, META_FILE]
    if model.generation_config is not None:
        model.generation_config.save_pretrained(directory)
        files.append('generation_config.json')
    meta = dict(format=FORMAT, schema_version=2, checkpoint_kind='mixed_dense_boundary' if pending else 'mixed_final_model', checkpoint_id=str(uuid.uuid4()),
        state_dict_file=STATE_FILE, files=sorted(files), native_modules=native_specs, compact_modules=compact, pending_dense_modules=pending,
        runtime=dict(attn_implementation=model.config._attn_implementation, seqlen=model.seqlen,
                     native_code_sha256=runtime_identity()),
        tensor_table=tensor_table(state), provenance=deepcopy(provenance),
        deployment='BF16 compact decoder outputs/additions; runtime training FP32 masters and optimizer are not saved',
        boundary='Independent mixed native/compact/dense deployment state. Dense intermediates are charged as stored; only fully compressed endpoints are eligible for the final2.6 budget. No quality or transferability result follows from this format.')
    measured = ledger(meta, state)
    require(measured['target_count'] == len(names), 'Incomplete real projection inventory.')
    require(pending or measured['budget_passed'], 'A fully compressed final mixed model exceeds the actual logical2.6 budget.')
    torch.save(state, directory / STATE_FILE)
    meta['state_sha256'] = sha(directory / STATE_FILE)
    meta['config_sha256'] = sha(directory / 'config.json')
    write(directory / META_FILE, meta)
    return meta, ledger(meta, state, directory), state


def load(directory):
    directory = Path(directory)
    meta = read(directory / META_FILE)
    require(meta['format'] == FORMAT and meta['schema_version'] == 2 and meta['state_dict_file'] == STATE_FILE,
            'This loader accepts only the isolated adaptive-native schema.')
    require(meta['runtime']['native_code_sha256'] == runtime_identity(), 'The verified exact native runtime code dependency changed.')
    require(sha(directory / STATE_FILE) == meta['state_sha256'] and sha(directory / 'config.json') == meta['config_sha256'],
            'Checkpoint payload/config identity changed.')
    require(set(path.name for path in directory.iterdir()) == set(meta['files']), 'Unexpected, missing or external payload file.')
    require(not any(path.is_symlink() for path in directory.iterdir()), 'A self-contained checkpoint cannot depend on external file symlinks.')
    state = torch.load(directory / STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    measured = ledger(meta, state, directory)
    model = architecture(directory, meta['runtime'])
    native._rebuild_converted_modules(model, meta['native_modules'])
    for item in meta['compact_modules']:
        name = item['name']
        payload = {key[len(name) + 1:]: tensor for key, tensor in state.items() if key.startswith(name + '.')}
        native.set_module_by_name(model, name, compact_from_state(item['spec'], payload))
    expected = dict(native=[item['name'] for item in meta['native_modules']], compact=[item['name'] for item in meta['compact_modules']],
                    pending_dense=[item['name'] for item in meta['pending_dense_modules']])
    actual = inventories(model)
    require(all(set(actual[key]) == set(expected[key]) for key in expected), 'Fresh projection topology differs from the three inventories.')
    for item in meta['pending_dense_modules']:
        module = model.get_submodule(item['name'])
        require(module.in_features == item['in_features'] and module.out_features == item['out_features']
                and (module.bias is not None) == item['has_bias'], 'Fresh pending dense geometry differs from the actual saved projection.')
    restore_parameter_dtypes(model, state)
    loaded = model.load_state_dict(state, strict=True)
    native.refresh_vae_linear_runtime_after_state_load(model)
    install_precision_runtime(model)
    model.eval().requires_grad_(False)
    return model, meta, loaded, measured

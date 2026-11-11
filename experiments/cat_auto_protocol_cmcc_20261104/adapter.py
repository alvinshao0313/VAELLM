"""Explicit dense Qwen3/Llama adapter, using real HF modules and file headers."""
from hashlib import sha1, sha256
import importlib
import json
from pathlib import Path
import re
import struct

from .contracts import capability, digest, require, validate_inventory

REPO = Path(__file__).resolve().parents[2]
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
DTYPES = {'BF16': ('bfloat16', 16), 'F16': ('float16', 16), 'F32': ('float32', 32), 'F64': ('float64', 64)}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return __import__('hashlib').file_digest(stream, 'sha256').hexdigest()


def public_consumer_identity():
    names = ('litebsq.autoencoder', 'litebsq.vae_linear', 'litebsq.bitpack', 'litebsq.bsq',
        'litebsq.parallel_layers', 'litebsq.vae_args', 'litebsq.misc',
        'rotation.weight_preprocess', 'rotation.hadamard_utils', 'train_utils.checkpoint_v6',
        'train_utils.checkpoint_bpw', 'train_utils.cat_train_pipeline', 'train_utils.channel_protection',
        'train_utils.cat_data_prep', 'train_utils.cat_train_data', 'train_utils.activation_utils',
        'train_utils.cat_category_runtime', 'train_utils.cat_runtime_state_v6', 'train_utils.train_args', 'train_utils.utils')
    paths = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(REPO / name.split('.')[0]),
            'PUBLIC consumer family is not selected; do not mix N snapshot imports: ' + name)
        paths[str(path)] = file_sha(path)
    return dict(family='PUBLIC_v6', version='actual_file_SHA256', source_sha256=paths,
        verified_scope='Public runtime family identity; train/deploy requires separate evidence')


def verify_source(base, manifest_path):
    base = Path(base).resolve()
    path = Path(manifest_path).resolve()
    record = json.loads(path.read_text(encoding='utf-8'))
    require(record['status'] == 'passed' and Path(record['path']).resolve() == base
        and re.fullmatch('[0-9a-f]{40}', record['revision']), 'Bind the actual completed local source verification receipt.')
    files = {}
    for item in record['files']:
        name = item['name']
        require(Path(name).name == name and name not in files, 'Source receipt has duplicate or external paths.')
        actual = base / name
        require(actual.is_file() and actual.stat().st_size == item['bytes'], 'Source byte size changed: ' + name)
        actual_sha = file_sha(actual)
        if 'sha256' in item:
            require(actual_sha == item['sha256'], 'Official source SHA256 changed: ' + name)
        else:
            git = sha1(f"blob {item['bytes']}\0".encode() + actual.read_bytes()).hexdigest()
            require(git == item['git_blob_sha1'], 'Official source Git blob changed: ' + name)
        files[name] = dict(bytes=item['bytes'], sha256=actual_sha)
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    require(required | set(index['weight_map'].values()) <= set(files), 'Source receipt omits required model/tokenizer/shard identity.')
    return dict(repository=record['repository'], revision=record['revision'], path=str(base),
        receipt_path=str(path), receipt_sha256=file_sha(path), files=files,
        verification='Actual local bytes match fixed official Git/LFS receipt; no pretrained tensor materialization')


def _headers(base, index):
    result = {}
    for name in sorted(set(index['weight_map'].values())):
        require(Path(name).name == name, 'Weight index cannot escape the actual model directory.')
        with (base / name).open('rb') as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, 'Truncated safetensors header.')
            length = struct.unpack('<Q', prefix)[0]
            require(0 < length <= 64 * 1024 * 1024, 'Invalid actual safetensors metadata length.')
            header = json.loads(stream.read(length))
        for key, row in header.items():
            if key != '__metadata__':
                result[key] = dict(file=name, **row)
    require(set(index['weight_map']) <= set(result), 'Weight index refers to absent actual tensors.')
    return result


def load_inventory(base, source_manifest):
    import torch
    from transformers import AutoConfig, LlamaForCausalLM, Qwen3ForCausalLM
    from train_utils.config.targets import discover_cat_projection_name_inventory

    public_consumer_identity()
    base = Path(base).resolve()
    source = verify_source(base, source_manifest)
    cfg = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    classes = {'qwen3': Qwen3ForCausalLM, 'llama': LlamaForCausalLM}
    require(cfg.model_type in classes, 'impl_blocked: add an explicit dense model adapter for ' + cfg.model_type)
    cfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = classes[cfg.model_type](cfg)
    inventory = discover_cat_projection_name_inventory(model, compression_categories=CATEGORIES)
    modules = dict(model.named_modules())
    require(set(inventory) == {(layer, category) for layer in range(cfg.num_hidden_layers) for category in CATEGORIES},
        'The explicit adapter must discover every actual target, not guessed suffix shapes.')
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    headers = _headers(base, index)
    rows, parameters = [], set()
    for (layer, category), name in sorted(inventory.items()):
        module = modules[name]
        require(type(module) is torch.nn.Linear and id(module.weight) not in parameters, 'Shared/fused/custom projection needs an explicit adapter.')
        parameters.add(id(module.weight))
        item = headers[name + '.weight']
        shape = list(module.weight.shape)
        require(item['shape'] == shape and item['dtype'] in DTYPES, 'Actual source tensor and real module geometry disagree.')
        dtype, bits = DTYPES[item['dtype']]
        bias = headers.get(name + '.bias')
        require((bias is not None) == (module.bias is not None), 'Actual projection bias topology differs.')
        rows.append(dict(actual_name=name, category_id=category, layer_id=layer, shape=shape,
            dtype=dtype, dtype_bits=bits, numel=module.weight.numel(), bias_numel=0 if bias is None else shape[0],
            source_tensor_file=item['file'], source_tensor_key=name + '.weight'))
    require(all(value.device.type == 'meta' for value in model.state_dict().values()), 'Inventory discovery allocated an unapproved dense model.')
    tokenizer_names = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json')
    model_identity = dict(schema='dense_model_content_identity_v0', repository=source['repository'], revision=source['revision'],
        model_type=cfg.model_type, architecture=type(model).__name__, config_sha256=source['files']['config.json']['sha256'],
        weight_index_sha256=source['files']['model.safetensors.index.json']['sha256'],
        weight_shards_sha256={name: source['files'][name]['sha256'] for name in sorted(set(index['weight_map'].values()))},
        tokenizer_files_sha256={name: source['files'][name]['sha256'] for name in tokenizer_names if name in source['files']})
    result = dict(model_identity=model_identity, source_evidence=source,
        adapter_id=cfg.model_type + '_dense_CAT', adapter_version=1, category_ids=list(CATEGORIES), projections=rows,
        total_target_numel=sum(row['numel'] for row in rows), discovery='Actual meta-device HF modules plus actual safetensors tensor headers',
        weights_loaded_for_inventory=False)
    return validate_inventory(result)


def category_projections(inventory, category_id):
    inventory = validate_inventory(inventory)
    require(category_id in inventory['category_ids'], 'Unknown semantic category.')
    return sorted((row for row in inventory['projections'] if row['category_id'] == category_id), key=lambda row: (row['layer_id'], row['actual_name']))


def execution_capabilities():
    return {
        'whole_category_pooled_core': capability('constructible', 'train_utils.cat_train_pipeline.train_group_vae_payload accepts full actual group_refs; BSQ pools B/M.'),
        'semantic_stage_factory_and_shuffle': capability('constructible', 'fit_adapter privately specializes the pinned PUBLIC function call sites; actual stage set_seed and per-stage shuffle streams are connected.', verification_scope='CPU_actual_tiny_factory; model_GPU_pending'),
        'per_layer_transpose_rotation_same_M': capability('constructible', 'fit_adapter executes actual per-refT/R in the same full category M; real tinyQwen S2 packed save/load fixture passed.', verification_scope='CPU_actual_tiny_Qwen; model_GPU_pending'),
        'controlled_live_fork_resolved_runtime': capability('constructible', 'live.fit_next_live_category consumes its own full native boundary, fits actual dense targets and saves a new UUID/runtime; next-category CPU integration evidence is separate.', verification_scope='actual_code_adapter; model_GPU_pending'),
        'generic_Muon_owner_stage': capability('constructible', 'vae_muon derives four actual R0 ParallelLinear owners and auxiliary LayerNorm/bias leaves; M2/M3 E128/D256 BF16 reference/update/state CPU fixture passed.', verification_scope='CPU_actual_VAE; model_GPU_pending'),
        'automatic_GPU_search_entry': capability('impl_blocked', 'search.py executes only1..4 explicitly frozen candidates with shared deadline/pins/budget; full20 result-driven orchestration and formal model-stage GPU evidence remain required.')}

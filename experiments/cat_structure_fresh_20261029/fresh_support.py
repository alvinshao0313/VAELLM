"""Bindings for a new HF-origin run; no historical checkpoint is an input."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
SCALARS = 262144
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stat(path):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, mtime_ns=value.st_mtime_ns,
                inode=value.st_ino, device=value.st_dev)


def enable_snapshot():
    sys.path.insert(0, str(SNAPSHOT))
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in {'train_utils', 'litebsq', 'rotation', 'e2e_common'}:
            file = getattr(module, '__file__', None)
            require(file is None or Path(file).resolve().is_relative_to(SNAPSHOT),
                    'A project module was already imported outside N: ' + name)


def source_pins():
    paths = [*HERE.glob('*.py'), *SNAPSHOT.rglob('*.py')]
    return {str(path): sha(path) for path in sorted(paths)}


def check_pins(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, 'Bound source changed: ' + path)


def policy(arm):
    require(arm in ('A', 'B'), 'Select A or B.')
    dims = {name: 64 if arm == 'B' and name == 'gate_proj' else 32 for name in CATEGORIES}
    return dict(schema='cat_structure_fresh_two_process_v1', arm=arm, seed=31,
        categories=list(CATEGORIES), layers=36, linear_group_size=36,
        scalar_exposure_per_linear_update=SCALARS,
        category_vector_batches={name: SCALARS // dim for name, dim in dims.items()},
        category_shuffle_seeds={name: 31 + index * 100000 for index, name in enumerate(CATEGORIES)},
        prefix_stop='After native completed after_o_proj category_boundary save returns.',
        suffix_start='Separate process; seed31 again; only this run after_o_proj.',
        entropy='Original CAT M36 pooled BSQ probability loss.',
        scope='Qwen3-8B 36-layer experiment; other model geometries unverified.')


def with_policy(identity, arm):
    require('experiment_runtime' not in identity, 'Unexpected preexisting experiment identity.')
    value = deepcopy(identity)
    value['experiment_runtime'] = policy(arm)
    return value


def projection_names(categories):
    return {f'model.layers.{index}.{"self_attn" if name in CATEGORIES[:4] else "mlp"}.{name}'
            for index in range(36) for name in categories}


def hf_identity(base_model_path):
    """Read config, tokenizer and safetensors headers, never load model tensors."""
    base = Path(base_model_path).resolve(strict=True)
    require(not (base / 'checkpoint_meta.json').exists() and not (base / 'cat_runtime_state.pt').exists(),
            'Initial input must be original HF, not a CAT checkpoint.')
    config = read(base / 'config.json')
    expected = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=4096,
                    intermediate_size=12288, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, torch_dtype='bfloat16')
    require(all(config.get(key) == value for key, value in expected.items()),
            'This isolated entry supports the specified Qwen3-8B geometry only.')
    index = read(base / 'model.safetensors.index.json')
    tensors, shards = {}, []
    for filename in sorted(set(index['weight_map'].values())):
        path = (base / filename).resolve(strict=True)
        require(path.is_relative_to(base), 'HF shard escapes its model directory.')
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = stream.read(size)
        fields = json.loads(header)
        tensor_fields = {key: value for key, value in fields.items() if key != '__metadata__'}
        require(not tensors.keys() & tensor_fields.keys(), 'Duplicate HF tensor names.')
        require(all(index['weight_map'].get(key) == filename for key in tensor_fields), 'HF shard index differs from its header.')
        tensors.update(tensor_fields)
        shards.append(dict(**stat(path), header_sha256=hashlib.sha256(header).hexdigest(), tensor_count=len(tensor_fields)))
    require(set(tensors) == set(index['weight_map']), 'Missing HF index tensors.')
    for name in projection_names(CATEGORIES):
        category = name.rsplit('.', 1)[1]
        shape = {'q_proj': [4096, 4096], 'k_proj': [1024, 4096], 'v_proj': [1024, 4096],
                 'o_proj': [4096, 4096], 'gate_proj': [12288, 4096],
                 'up_proj': [12288, 4096], 'down_proj': [4096, 12288]}[category]
        require(tensors[name + '.weight']['shape'] == shape and tensors[name + '.weight']['dtype'] == 'BF16',
                'HF projection is not the expected original BF16 shape: ' + name)
    files = ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
             'vocab.json', 'merges.txt', 'generation_config.json')
    return dict(base_model_path=str(base), config=config, files_sha256={name: sha(base / name) for name in files},
                shards=shards, tensor_count=len(tensors), tensor_bytes=index['metadata']['total_size'],
                verification='Real HF config/tokenizer hashes and shard headers/stat; no full tensor-content digest or model forward.')


def validate_after_o_meta(meta, base_model_path):
    """Only a finalized full native save can delimit the two real processes."""
    require(meta['checkpoint_kind'] == 'category_boundary'
            and meta['completed_categories'] == list(CATEGORIES[:4])
            and meta['compression_categories'] == list(CATEGORIES)
            and meta['converted_module_count'] == 144
            and set(meta['compressed_targets']) == projection_names(CATEGORIES[:4])
            and set(meta['pending_dense_targets']) == projection_names(CATEGORIES[4:])
            and not meta['skip_targets'] and meta['lora_config'] is None
            and Path(meta['base_model_path']).resolve() == Path(base_model_path).resolve(),
            'Require this full seven-category, finalized native after_o with 144 compressed +108 live dense.')
    require(all(meta['finalized_status'][field] is True for field in
                ('lora_finalized', 'decoder_finalized', 'aux_finalized', 'stable_category_boundary')),
            'after_o has not finished native finalization.')
    extra = meta['extra_meta']
    history = extra['distill_stage_history']
    require(extra['category'] == 'o_proj' and extra['stage'] == 'after_category'
            and [item['category'] for item in history] == list(CATEGORIES[:4])
            and all(item['did_train'] and item['mode'] == 'remaining_lora_prefix_decoder' for item in history)
            and extra['distill_stage']['remaining_lora_target_count'] == 108
            and extra['distill_stage']['decoder_target_count'] == 144,
            'after_o must follow all four actual native recoveries with the original seven-category scope.')


def own_after_o(request):
    """Resolve the suffix input exclusively from this execution's prefix report."""
    output = Path(request['output_dir']).resolve()
    report = read(output / 'prefix.json')
    require(report['status'] == 'prefix_completed' and report['exit_code'] == 0
            and report['request_sha256'] == sha(output / 'request.json')
            and report['seed_calls'] == [31], 'The fresh prefix did not finish successfully.')
    event = report['saved_boundaries'][-1]
    path = prefix_input_path(output, event['path'])
    meta = read(path / 'checkpoint_meta.json')
    validate_after_o_meta(meta, request['base_model_path'])
    require(meta['checkpoint_id'] == event['checkpoint_id']
            and sha(path / 'checkpoint_meta.json') == event['metadata_sha256']
            and sha(path / 'cat_runtime_state.pt') == event['runtime_sha256']
            and stat(path / meta['state_dict_file']) == event['weight_stat'], 'Saved fresh boundary changed.')
    return path, event


def prefix_input_path(output, path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(Path(output).resolve() / 'prefix' / 'native') and path.name == 'after_o_proj',
            'Suffix input must be this execution\'s newly saved after_o.')
    return path


def worker_environment(gpu):
    value = dict(os.environ)
    value.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SNAPSHOT), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONHASHSEED='31',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='offline')
    require(int(value.get('WORLD_SIZE', '1')) == 1, 'Only the native single-GPU process is supported here.')
    return value

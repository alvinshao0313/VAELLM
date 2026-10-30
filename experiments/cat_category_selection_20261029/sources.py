"""Bind the preserved A endpoints and reconstruct their actual shared raw state."""
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from . import ROOT, RUNTIME
from train_utils import checkpoint_v6 as native
from train_utils.distill_precision import install_precision_runtime, restore_parameter_dtypes

DATA = Path('/data/star/VAELLM/cat_category_selection_20261029')
CAT = Path('/data/star/VAELLM/cat_structure_full_20261029')
RAW = Path('/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/A')
NODES = CAT / 'selection_candidates/A'
BASE = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o')
RAW_ID = '94c8c25b-19dd-4c6d-852c-19f5b12c4d6b'
BASE_ID = 'b63bfe79-3889-40c6-bf66-1d002893f874'
CONTROL = CAT / 'A/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/after_gate_proj'
TOPOLOGY = ('compressed_targets', 'pending_dense_targets', 'skip_targets', 'completed_categories',
            'compression_categories', 'target_layers', 'target_modules')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    path = Path(path).resolve()
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def binding(path):
    return dict(**stat(path), sha256=sha(path))


def digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return value.hexdigest()


def equal(actual, expected, label):
    require(set(actual) == set(expected), label + ': persistent tensor keys differ')
    for name, value in actual.items():
        reference = expected[name]
        require(value.dtype == reference.dtype and value.shape == reference.shape and torch.equal(value, reference),
                label + ': dtype/shape/value differs: ' + name)
    return len(actual)


def cpu_only(model=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Caller-activated bitvae with hidden CUDA is required; no GPU is assigned.')
    require(Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(),
            'Use the confirmed bitvae interpreter.')
    if model is not None:
        require(all(value.device.type == 'cpu' for value in model.state_dict().values()), 'CPU-only model required.')
        require(model.config._attn_implementation == 'flash_attention_2', 'Preserve the formal FA2 deployment policy.')
        require(all(value.dtype == torch.float32 for name, value in model.named_buffers() if 'inv_freq' in name),
                'RoPE buffers must remain FP32.')


def architecture(config_path):
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True, trust_remote_code=False)
    require(config.model_type == 'qwen3', 'This bounded exporter serves the assigned Qwen3 control.')
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', trust_remote_code=False)
    model.seqlen = 2048
    cpu_only(model)
    return model


def load_deployment(directory):
    directory = Path(directory)
    model = architecture(directory)
    model, meta, loaded = native.load_v6_full_checkpoint_into_model(model, str(directory),
        expected_kind='category_boundary', map_location='cpu', strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'Strict fresh deployment load failed.')
    model.eval().requires_grad_(False)
    cpu_only(model)
    return model, meta


def verify_archive(directory):
    manifest = read(directory / 'manifest.json')
    for filename, item in manifest['files'].items():
        path = directory / filename
        require(not path.is_symlink() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Protected partial archive changed: ' + str(path))
    return manifest


def source_metadata():
    from train_utils.cat_step_resume_v6 import validate_cat_step_immutable_resume_contract
    cpu_only()
    raw = verify_archive(RAW)
    raw_meta = native.validate_v6_meta(read(RAW / 'raw_checkpoint_meta.json'), expected_kind='round_base')
    require(raw_meta['checkpoint_id'] == raw['source_raw']['checkpoint_id'] == RAW_ID
            and raw['after_o_complement_exact'], 'Use the real preserved raw A gate parent.')
    nodes = {}
    for step in (250, 500):
        directory = NODES / f'gate{step}'
        archive = verify_archive(directory)
        meta = native.load_v6_training_step_meta(str(directory))
        trainer = read(directory / 'trainer_state.json')
        native.validate_training_step_round_base(meta, raw_meta)
        require(archive['checkpoint_id'] == meta['checkpoint_id'] and archive['raw_parent_checkpoint_id'] == RAW_ID
                and trainer['global_step'] == step and trainer['max_steps'] == 500,
                'Use the original completed250/500 checkpoints of one unchanged500 schedule.')
        for field in TOPOLOGY:
            require(meta[field] == raw_meta[field], 'Node/raw topology differs: ' + field)
        contract = meta['immutable_resume_contract']
        require(contract['active_category'] == 'gate_proj' and contract['optimization']['steps'] == 500
                and contract['after_category_mode'] == 'remaining_lora_prefix_decoder'
                and contract['precision']['bf16'] and not contract['precision']['fp16']
                and set(contract['decoder_target_names']) == set(raw_meta['compressed_targets'])
                and set(contract['lora_target_names']) == set(raw_meta['pending_dense_targets'])
                and contract['aux']['norm_train_mode'] == contract['aux']['lm_head_train_mode'] == 'none',
                'Unexpected original whole-prefix/remaining-dense contract.')
        nodes[step] = (meta, archive)
    validate_cat_step_immutable_resume_contract(nodes[250][0]['immutable_resume_contract'],
                                              nodes[500][0]['immutable_resume_contract'])
    return raw_meta, raw, nodes


def build_raw(raw_meta, raw):
    base_meta = native.load_v6_meta(str(BASE))
    require(base_meta['checkpoint_id'] == raw['common_after_o']['checkpoint_id'] == BASE_ID,
            'Use the bound actual common after_o source.')
    original_weight = BASE / base_meta['state_dict_file']
    recorded = raw['common_after_o']['weight_stat']
    actual = stat(original_weight)
    require(all(actual.get(key) == value for key, value in recorded.items() if key in actual),
            'Protected after_o weight stat changed.')
    base = torch.load(original_weight, map_location='cpu', weights_only=True, mmap=True)
    gate = torch.load(RAW / 'gate_state.pt', map_location='cpu', weights_only=True, mmap=True)
    names = tuple(name + '.' for name in raw['gate_names'])
    state = {name: value for name, value in base.items() if not name.startswith(names)}
    state.update(gate)
    require(len(state) == raw['source_raw_tensor_count'] and digest(state) == raw['source_raw_state_digest'],
            'Archive+after_o did not reconstruct the complete original raw2559 tensor state.')
    model = architecture(RAW / 'raw_config.json')
    native._rebuild_converted_modules(model, raw_meta['converted_modules'])
    native.validate_model_target_inventories(model, raw_meta['compressed_targets'], raw_meta['pending_dense_targets'], raw_meta['skip_targets'])
    restore_parameter_dtypes(model, state)
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'Raw2559 strict reconstructed load failed.')
    native.refresh_vae_linear_runtime_after_state_load(model)
    install_precision_runtime(model)
    equal(model.state_dict(), state, 'Reconstructed real raw state')
    model.eval().requires_grad_(False)
    cpu_only(model)
    return model, state


def original_stage():
    from train_utils import cat_after_category_common as canonical
    from train_utils.cat_runtime_state_v6 import validate_cat_runtime_identity
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT / 'experiments/cat_structure_full_20261029'))
        spec = importlib.util.spec_from_file_location('_selection_original_A_recipe', ROOT / 'experiments/cat_structure_full_20261029/recipe.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    runtime = torch.load(RAW / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    row = read(CAT / 'prepared.json')['arms']['A']
    cat, hf, training, vae, cfgs, identity, normalized = module.parse_recipe('A', row['native_argv'], runtime)
    validate_cat_runtime_identity(runtime['runtime_identity'], identity)
    require(not cat.channel_refresh_after_category, 'This read-only boundary reuses raw statistics only with original category refresh disabled.')
    stage = canonical.resolve_cat_after_category_stage(cat, training, category='gate_proj', round_idx=4)
    # Dataset construction performs this same deterministic .12g mix normalization.
    stage.config.data.validate()
    return stage, training


def contract_config_checks(contract, stage, training):
    from transformers import TrainingArguments
    from train_utils.cat_step_resume_v6 import _jsonable
    opt = _jsonable(stage.config.opt)
    opt.pop('logging_steps', None)
    aux = _jsonable(stage.config.aux)
    for key in tuple(aux):
        if key.startswith('residual_lora_'):
            aux.pop(key)
    current = dict(data=_jsonable(stage.config.data), loss=_jsonable(stage.config.loss), optimization=opt,
                   aux=aux, runtime=_jsonable(stage.config.runtime))
    for key, value in current.items():
        require(value == contract[key], 'Original reparsed A stage differs from immutable source contract: ' + key)
    # Canonical _build_training_arguments supplies stage precision and cfg.opt
    # checkpointing; it leaves tf32 at the actual TrainingArguments class default.
    require(contract['precision'] == dict(bf16=stage.bf16, fp16=stage.fp16,
        tf32=TrainingArguments.__dataclass_fields__['tf32'].default,
        gradient_checkpointing=stage.config.opt.gradient_checkpointing,
        gradient_checkpointing_kwargs=stage.config.opt.gradient_checkpointing_kwargs or {}),
        'Original precision differs from immutable source contract.')
    return dict(reparsed_A_stage_fields_exact=list(current), precision_exact=True,
                two_saved_full_immutable_contracts_exact=True,
                dataset_and_teacher_identity='Original saved declarations preserved and byte-bound; no training/teacher replay or exact-resume is claimed.')


def own_pins():
    paths = [*Path(__file__).parent.glob('*.py'), ROOT / 'experiments/cat_structure_full_20261029/recipe.py',
             ROOT / 'experiments/cat_structure_full_20261029/common.py', CAT / 'prepared.json']
    relative = ('train_utils/checkpoint_v6.py', 'train_utils/distill_precision.py', 'train_utils/distill_decoder.py',
                'train_utils/model_level_checkpoint_state.py', 'train_utils/model_level_trainables.py',
                'train_utils/cat_step_resume_v6.py', 'train_utils/cat_after_category_common.py',
                'e2e_common/full_lora.py', 'litebsq/vae_linear.py', 'rotation/hadamard_utils.py',
                'rotation/hadamard_extended.py', 'rotation/weight_preprocess.py')
    paths += [RUNTIME / name for name in relative]
    return {str(path): sha(path) for path in paths}

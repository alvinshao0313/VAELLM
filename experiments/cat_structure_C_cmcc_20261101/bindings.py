"""Fixed original-C recipe and actual migrated native inputs."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cmcc_runtime_20261030.paths import (ARTIFACTS, MODELS, NATIVE_RUNTIME,
    PYTHON, REPO, TOPIC, resolve_legacy_path)

HERE = Path(__file__).resolve().parent
OUTPUT = ARTIFACTS / HERE.name
RESULT = TOPIC / HERE.name
SOURCE = ARTIFACTS / 'qwen_recompression_20261026/C/initial_after_o'
SOURCE_ID = 'b63bfe79-3889-40c6-bf66-1d002893f874'
BLUEPRINT = ARTIFACTS / 'cat_structure_full_20261029/prepared.json'
TERMINAL = TOPIC / 'overnight_goal_20261029/terminal_audit_20261030.json'
HF = MODELS / 'Qwen/Qwen3-8B'
SNAPSHOT = NATIVE_RUNTIME
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
CONTRACT = dict(arm='C', gate_D=32, gate_K=32, stages=2, gate_encoder=256,
    gate_decoder=128, residual_blocks=0, linears_per_category=36,
    native_fit_updates_per_stage=10000, recoveries_per_category=500,
    vector_batch_per_linear=8192, seed=31,
    shuffle_seeds=dict(gate_proj=400031, up_proj=500031, down_proj=600031),
    suffix_rng='Original native continuous RNG; no per-category reseeding.',
    consumer='original native v6; no compact conversion', checkpoint_selection=False)

# Close namespace children before helpers can prepend the repository path.
sys.path.insert(0, str(SNAPSHOT))
for _name in ('train_utils', 'litebsq', 'rotation', 'e2e_common'):
    _package = importlib.import_module(_name)
    _package.__path__ = [str(SNAPSHOT / _name)]
_spec = importlib.util.spec_from_file_location('_original_C_fork_helpers',
    REPO / 'experiments/cat_recompression/fork_core.py')
fork = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fork)
read, write, sha, require, torch = fork.read, fork.write, fork.sha, fork.require, fork.torch


def pins():
    files = set(HERE.glob('*.py')) | set(SNAPSHOT.rglob('*.py'))
    files.update((REPO / 'experiments/cat_recompression/fork_core.py',
        REPO / 'experiments/cmcc_runtime_20261030/paths.py',
        REPO / 'experiments/adaptive_class_20261030/initialization.py',
        REPO / 'experiments/adaptive_vector_rvq/common.py'))
    return {str(path.resolve()): sha(path) for path in sorted(files)}


def check_pins(values):
    for name, digest in values.items():
        require(sha(name) == digest, 'Bound source changed: ' + name)


def rebind(value):
    if isinstance(value, dict):
        return {key: rebind(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rebind(item) for item in value]
    if isinstance(value, tuple):
        return tuple(rebind(item) for item in value)
    if isinstance(value, str) and value.startswith(('/data/star/', '/home/star/models/', '/home/star/VAELLM/')):
        return str(resolve_legacy_path(value))
    return value


def control():
    terminal = read(TERMINAL)
    result = terminal['results']['A']
    require(terminal['status'] == 'verified_existing_complete_results'
        and result['nine_reported'] == 63.52216414161746
        and result['full8_leaves'] == 64 and result['longbench_samples'] == 4750,
        'The original complete A control is missing.')
    acceptance = resolve_legacy_path(result['acceptance'])
    require(sha(acceptance) == result['acceptance_sha256'], 'Original A acceptance changed.')
    return {**result, 'terminal_audit': str(TERMINAL), 'terminal_audit_sha256': sha(TERMINAL),
        'acceptance': str(acceptance), 'acceptance_sha256': sha(acceptance)}


def source_binding():
    previous, meta = read(BLUEPRINT), read(SOURCE / 'checkpoint_meta.json')
    require(previous['status'] == 'cpu_verified' and meta['checkpoint_id'] == SOURCE_ID
        and meta['checkpoint_kind'] == 'category_boundary' and meta['completed_categories'] == list(CATEGORIES[:4])
        and meta['converted_module_count'] == 144 and len(meta['pending_dense_targets']) == 108
        and meta['lora_config'] is None, 'Require the actual stable original after_o parent.')
    require(sha(SOURCE / 'checkpoint_meta.json') == previous['source_metadata_sha256']
        and sha(SOURCE / 'cat_runtime_state.pt') == previous['source_runtime_sha256'],
        'Original after_o metadata/runtime changed.')
    # Historical paths/stats stay historical. Verify actual migrated files here.
    historical = {str(resolve_legacy_path(name)): digest
        for name, digest in previous['snapshot_source_sha256'].items()}
    check_pins(historical)
    return dict(schema='original_C_actual_after_o_source_v1', checkpoint_dir=str(SOURCE),
        checkpoint_id=SOURCE_ID, metadata_sha256=sha(SOURCE / 'checkpoint_meta.json'),
        runtime_sha256=sha(SOURCE / 'cat_runtime_state.pt'),
        checkpoint_weight_stat=fork.stat(SOURCE / meta['state_dict_file']),
        expected_original_state=previous['source_state'], actual_blueprint=str(BLUEPRINT),
        actual_blueprint_sha256=sha(BLUEPRINT), originalA_control=control())


def policy():
    return dict(schema='original_C_M36_E256_v1', **CONTRACT)


def with_policy(identity):
    require('experiment_runtime' not in identity, 'Do not replace an existing runtime policy.')
    return {**deepcopy(identity), 'experiment_runtime': policy()}


def replace_option(argv, name, value):
    require(argv.count(name) == 1, 'Require one actual option: ' + name)
    argv[argv.index(name) + 1] = str(value)


def recipe(output):
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_category_runtime import resolve_category_runtime_configs
    from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
    from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
    previous = read(BLUEPRINT)
    argv = rebind(previous['arms']['A']['native_argv'])
    for name, value in (('--output_dir', output), ('--resume_from_checkpoint', OUTPUT / 'initial_after_o'),
            ('--base_ch', 'default=128,cat:gate_proj=256')):
        replace_option(argv, name, value)
    cat, hf, training, vae = parse_cat_runtime_args(argv)
    training.seed = cat.seed
    cfgs = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    identity = with_policy(build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=CATEGORIES, target_layers='all',
        skip_layers=(), transpose_modules=cat.transpose_modules.split(',')))
    expected = deepcopy(previous['arms']['A']['runtime_identity'])
    expected['resolved_category_runtime']['gate_proj']['base_ch'] = 256
    expected['experiment_runtime'] = policy()
    require(identity == expected, 'C must differ from the actual A recipe only in gate encoder width.')
    for name in CATEGORIES[4:]:
        cfg, recovery = cfgs[name], cat.resolve_after_category_config(name)
        require(cfg.steps == 10000 and cfg.residual_stages == 2 and cfg.codebook_dim == 32
            and recovery.opt.steps == 500 and recovery.opt.batch_size == 4
            and recovery.data.model_max_length == 1024 and recovery.data.dynamic_padding
            and recovery.lora.rank == 12 and recovery.opt.learning_rate == 1e-4
            and recovery.opt.resolved_decoder_lr() == 1e-5,
            'Keep the original M36/10K and 500/B4/1024/FP32 decoder+future rank12 contract.')
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=cfgs)
    normalized['experiment_runtime'] = policy()
    return argv, (cat, hf, training, vae, cfgs), identity, normalized


def event(directory):
    directory = Path(directory).resolve(strict=True)
    meta = read(directory / 'checkpoint_meta.json')
    return dict(path=str(directory), checkpoint_id=meta['checkpoint_id'], kind=meta['checkpoint_kind'],
        completed_categories=meta['completed_categories'], metadata_sha256=sha(directory / 'checkpoint_meta.json'),
        runtime_sha256=sha(directory / 'cat_runtime_state.pt'), weight_stat=fork.stat(directory / meta['state_dict_file']))


def free_gpu(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    uuid, used = {int(parts[0]): (parts[1].strip(), int(parts[2]))
        for line in rows.splitlines() if (parts := line.split(','))}[gpu]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    require(used <= 256 and not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()),
        'Assigned GPU is occupied; stop without waiting or retrying.')
    return dict(gpu=gpu, uuid=uuid, used_MiB=used, checked_unix=time.time())

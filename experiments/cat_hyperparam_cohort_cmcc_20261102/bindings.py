"""Fixed gate-only cohort over the genuine fresh-common native B parent."""
from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, MODELS, NATIVE_RUNTIME, PYTHON, REPO, TOPIC

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
COMMON = ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
HF, SNAPSHOT = MODELS / 'Qwen/Qwen3-8B', NATIVE_RUNTIME
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
ARMS = dict(lr_low=dict(gpu=2, steps=10000, lr=.0015, commitment=.25),
    budget20k=dict(gpu=4, steps=20000, lr=.003, commitment=.25),
    commit_double=dict(gpu=5, steps=10000, lr=.003, commitment=.5))
sys.path.insert(0, str(SNAPSHOT))
for _name in ('train_utils', 'litebsq', 'rotation', 'e2e_common'):
    _package = importlib.import_module(_name)
    _package.__path__ = [str(SNAPSHOT / _name)]
_spec = importlib.util.spec_from_file_location('_gate_hyperparam_fork_core', REPO / 'experiments/cat_recompression/fork_core.py')
fork = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fork)
read, write, sha, require, torch = fork.read, fork.write, fork.sha, fork.require, fork.torch


def pins():
    files = set(HERE.glob('*.py')) | set(SNAPSHOT.rglob('*.py'))
    files.update((REPO / 'experiments/cat_recompression/fork_core.py',
        REPO / 'experiments/cmcc_runtime_20261030/paths.py',
        REPO / 'experiments/adaptive_class_20261030/initialization.py',
        REPO / 'experiments/adaptive_vector_rvq/common.py',
        TOPIC / 'lm_eval_sample_artifacts_20261010/eval_utils_with_samples.py',
        TOPIC / 'lm_eval_sample_artifacts_20261010/sample_artifacts.py'))
    return {str(path.resolve()): sha(path) for path in sorted(files)}


def check_pins(values):
    for name, digest in values.items():
        require(sha(name) == digest, 'Bound source changed: ' + name)


def stable_stat(path):
    return {key: value for key, value in fork.stat(path).items() if key != 'allocated_bytes'}


def event(directory):
    directory = Path(directory).resolve(strict=True)
    meta = read(directory / 'checkpoint_meta.json')
    return dict(path=str(directory), checkpoint_id=meta['checkpoint_id'], kind=meta['checkpoint_kind'],
        metadata_sha256=sha(directory / 'checkpoint_meta.json'), runtime_sha256=sha(directory / 'cat_runtime_state.pt'),
        weight_stat=stable_stat(directory / meta['state_dict_file']))


def sources():
    proof, prefix, raw = (read(COMMON / name) for name in ('boundary_verification.json', 'prefix.json', 'gate_raw2.json'))
    require(proof['status'] == 'verified' and not proof['validation_only']
        and proof['non_gate_after_o_complement_exact'] and proof['gate_recovery_updates'] == 0
        and prefix['status'] == 'prefix_completed' and raw['status'] == 'raw2_completed'
        and (COMMON / 'prefix.exit').read_text().strip() == (COMMON / 'gate_raw2.exit').read_text().strip() == '0',
        'Require the completed formal common chain and genuine gate raw2.')
    parent, control = (Path(row['saved_boundaries'][-1]['path']) for row in (prefix, raw))
    require(event(parent)['checkpoint_id'] == proof['after_o_checkpoint_id'] == '7eeb82b4-3868-4cc5-9fa3-51870ee55894'
        and event(control)['checkpoint_id'] == proof['raw2_checkpoint_id'] == 'f4669891-9028-4665-85a5-71c26ac45c33',
        'Wrong genuine common parent/control UUID.')
    for directory, previous in ((parent, prefix), (control, raw)):
        actual, saved = event(directory), previous['saved_boundaries'][-1]
        require(all(actual[key] == saved[key] for key in actual), 'Common parent/control file identity changed.')
    return dict(after_o=event(parent), control=event(control),
        formal_evidence={str(COMMON / name): sha(COMMON / name) for name in
            ('boundary_verification.json', 'prefix.json', 'gate_raw2.json', 'request.json')})


def replace(argv, option, value):
    require(argv.count(option) == 1, 'Require exactly one option: ' + option)
    argv[argv.index(option) + 1] = str(value)


def recipe(arm, output):
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_category_runtime import resolve_category_runtime_configs
    from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
    from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
    previous, settings = read(COMMON / 'gate_raw2.json'), ARMS[arm]
    argv = list(previous['native_argv'])
    for option, value in (('--output_dir', output), ('--resume_from_checkpoint', OUTPUT / arm / 'initial_after_o'),
            ('--vae_learning_rate', settings['lr']), ('--commitment_loss_weight', settings['commitment'])):
        replace(argv, option, value)
    if arm == 'budget20k':
        replace(argv, '--vae_steps', 'default=10000,cat:gate_proj=20000')
    cat, hf, training, vae = parse_cat_runtime_args(argv)
    training.seed = cat.seed
    cfgs = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=CATEGORIES, target_layers='all',
        skip_layers=(), transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(previous['runtime_identity']['experiment_runtime'])
    expected = deepcopy(previous['runtime_identity'])
    expected['vae_shared']['lr'], expected['vae_shared']['commitment_loss_weight'] = settings['lr'], settings['commitment']
    expected['resolved_category_runtime']['gate_proj']['steps'] = settings['steps']
    require(identity == expected, 'The new arm changed something beyond its single approved factor.')
    gate = cfgs['gate_proj']
    require(gate.codebook_dim == gate.codebook_bits == 64 and gate.residual_stages == 2
        and gate.base_ch == gate.decoder_base_ch == 128 and gate.decoder_type == 'asymmetric'
        and gate.steps == settings['steps'] and identity['vae_shared']['lr_scheduler'] == 'linear'
        and identity['vae_shared']['lr_warmup_steps'] == 0, 'Incorrect fixed gate geometry or fresh linear schedule.')
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=cfgs)
    normalized['experiment_runtime'] = identity['experiment_runtime']
    return argv, (cat, hf, training, vae, cfgs), identity, normalized


def validate_raw2(meta):
    require(meta['checkpoint_kind'] == 'round_base' and meta['completed_categories'] == list(CATEGORIES[:4])
        and meta['compression_categories'] == list(CATEGORIES) and meta['converted_module_count'] == 180
        and len(meta['pending_dense_targets']) == 72 and meta['lora_config'] is None
        and meta['extra_meta']['distill_stage'] is None, 'Require native180 + future72dense with no gate recovery.')
    gates = [item for item in meta['converted_modules'] if item['name'].endswith('.gate_proj')]
    require(len(gates) == 36 and all(item['residual_stages'] == 2 and item['codebook_dim'] == 64
        and item['stage_codebook_dims'] == [64, 64] and all(stage['decoder_type'] == 'asymmetric'
            and stage['in_dim'] == stage['out_dim'] == 64 and stage['hidden_dim'] == 128
            and stage['num_res_blocks'] == 0 for stage in item['stage_decoders']) for item in gates),
        'Incorrect raw2 gate geometry.')


def free_gpu(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    uuid, used = {int(parts[0]): (parts[1].strip(), int(parts[2])) for line in rows.splitlines()
        if (parts := line.split(','))}[gpu]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    require(used <= 256 and not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()),
        'Assigned GPU occupied; stop without waiting or retry.')
    return dict(gpu=gpu, uuid=uuid, used_MiB=used, checked_unix=time.time())

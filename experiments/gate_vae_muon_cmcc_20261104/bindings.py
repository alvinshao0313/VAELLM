"""Gate-only Muon point over the unchanged native after-o calibration boundary."""
from copy import deepcopy
from pathlib import Path

from experiments.cat_geometry_cohort_cmcc_20261102 import bindings as base
from experiments.adaptive_vector_rvq import common as c

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = base.ARTIFACTS / HERE.name, base.TOPIC / HERE.name
REPO, TOPIC, PYTHON, SNAPSHOT, HF, COMMON = base.REPO, base.TOPIC, base.PYTHON, base.SNAPSHOT, base.HF, base.COMMON
read, write, sha, require, torch = base.read, base.write, base.sha, base.require, base.torch
ARMS = dict(muon=dict(gpu=0, steps=10000, dim=64, bits=64, encoder=128, batch=4096))
ORIGINAL_PINS = base.pins
MATURE_OPTIMIZER = REPO / 'experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py'
CONTROL_ROOT = base.ARTIFACTS / 'cat_hyperparam_cohort_cmcc_20261102/control'
CONTROL_FRESH = base.CONTROL_EVIDENCE / 'control_fresh.json'
CATEGORIES = base.CATEGORIES
SEVEN = 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'
CONTRACT = dict(category='gate_proj', source_after_o='7eeb82b4-3868-4cc5-9fa3-51870ee55894',
    control='f4669891-9028-4665-85a5-71c26ac45c33', D=64, K=64, M=36, B=4096, S=2,
    formal_updates_per_stage=10000, short_updates_per_stage=2,
    short_scheduler='linear2 functional short only; formal is fresh linear10000',
    E=128, decoder=128, residual_blocks=0, protected_input_count=64, protected_dtype='INT8', rotation_block=32,
    transpose=False, parallel_parts=1, seed=31, shuffle_seed=400031,
    parameters=1211904, matrix_elements=1179648, auxiliary_elements=32256,
    BF16_parameters_gradients_computation=True, persistent_FP32_parameter_master=False,
    Muon_momentum_FP32=True, NS_steps=5, nesterov=True, momentum=.95,
    auxiliary_AdamW=dict(lr=.003, betas=[.9,.95], eps=1e-8, weight_decay=0),
    kappa='Independent per residual stage, one E+D BF16 Adam displacement match <=1%, max2 doublings/12 bisections',
    control_initialization='No historical stage digest; record genuine factories without asserting exact control initialization',
    budget=dict(short_total=1800, formal_fit=5400, formal_fresh=1800, full8=5400, readout=1800, control_seven=3600),
    automatic_recovery_or_suffix=False, automatic_retry=False)


def recipe(arm, output, *, short=False):
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_category_runtime import resolve_category_runtime_configs
    from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
    from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
    require(arm == 'muon', 'This is one gate optimizer point.')
    previous = read(COMMON / 'gate_raw2.json')
    argv = list(previous['native_argv'])
    for option, value in (('--output_dir', output), ('--resume_from_checkpoint',
            OUTPUT / arm / 'short_initial_after_o' if short else Path(base.sources()['after_o']['path']))):
        base.replace(argv, option, value)
    if short:
        base.replace(argv, '--vae_steps', 'default=10000,cat:gate_proj=2')
        base.replace(argv, '--log_every', '1')
    cat, hf, training, vae = parse_cat_runtime_args(argv)
    training.seed = cat.seed
    cfgs = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=CATEGORIES, target_layers='all', skip_layers=(),
        transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(previous['runtime_identity']['experiment_runtime'])
    expected = deepcopy(previous['runtime_identity'])
    if short:
        expected['resolved_category_runtime']['gate_proj']['steps'] = 2
    require(identity == expected, 'Native gate recipe changed beyond the functional short horizon.')
    cfg = cfgs['gate_proj']
    require(cfg.codebook_dim == cfg.codebook_bits == 64 and cfg.base_ch == cfg.decoder_base_ch == 128
        and cfg.num_res_blocks == cfg.decoder_num_res_blocks == 0 and cfg.residual_stages == 2
        and cfg.decoder_type == 'asymmetric' and cfg.channel_protect_count == 64
        and cfg.weight_rotation_block_size == 32 and vae.lr == .003 and vae.beta1 == .9 and vae.beta2 == .95
        and vae.weight_decay == 0 and cfg.steps == (2 if short else 10000)
        and identity['vae_shared']['lr_scheduler'] == 'linear' and identity['vae_shared']['lr_warmup_steps'] == 0,
        'Keep the complete original native gate VAE/loss/precision/schedule recipe.')
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=cfgs)
    normalized['experiment_runtime'] = identity['experiment_runtime']
    return argv, (cat, hf, training, vae, cfgs), identity, normalized


def pins():
    paths = set(HERE.glob('*.py')) | set((REPO / 'experiments/cat_geometry_cohort_cmcc_20261102').glob('*.py'))
    paths.add(MATURE_OPTIMIZER)
    return {**ORIGINAL_PINS(), **{str(path.resolve()): sha(path) for path in paths}}


def configure():
    base.HERE, base.OUTPUT, base.RESULT, base.ARMS = HERE, OUTPUT, RESULT, ARMS
    base.recipe, base.pins = recipe, pins


def prepared():
    value = read(RESULT / 'prepared.json')
    require(value['status'] == 'cpu_verified' and value['contract'] == CONTRACT, 'Require this gate point CPU source/layout proof.')
    base.check_pins(value['source_sha256'])
    require(base.sources() == value['source_binding'], 'Original gate parent/control changed.')
    return value


def stage_args(parsed):
    args, cfg = deepcopy(parsed[3]), parsed[-1]['gate_proj']
    for key, value in dict(parallel_layers=36, codebook_dim=64, codebook_bits=64, base_ch=128,
            num_res_blocks=0, decoder_base_ch=128, decoder_num_res_blocks=0, decoder_type='asymmetric',
            norm_type='layer', activation_type='swish').items():
        setattr(args, key, value)
    return args


configure()

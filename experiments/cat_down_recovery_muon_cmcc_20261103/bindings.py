"""Fixed raw-down parent, canonical category recipe and independent output paths."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path

from experiments.fresh_fixed2_cmcc_20261030 import sources as original
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC
from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
# Mature math imports historical helpers which prepend another snapshot.
# Close the selected canonical N namespaces before those imports, as in C0.
original.common.s.enable_snapshot()
for _name in ('train_utils','litebsq','rotation','e2e_common'):
    importlib.import_module(_name).__path__=[str(original.common.s.SNAPSHOT/_name)]
from experiments.muon_compact_cmcc_20261030 import optimizer as mature
from experiments.cmcc_runtime_20261030 import sequence as executor

torch,read,write,sha,require = c.torch,c.read,c.write,c.sha,c.require
HERE = Path(__file__).resolve().parent
OUTPUT,RESULT = ARTIFACTS/HERE.name,TOPIC/HERE.name
COMMON = ARTIFACTS/'fresh_common_cmcc_20261030/formal'
C0 = ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0'
RAW = C0/'06_down_proj/round_base'
RAW_ID = 'ed507bcb-9b83-4e4c-9859-daaef44d80d2'
CONTROL_ID = 'cbb15c9f-1c1f-4394-817c-65c226d76ce7'
CONTROL_ACCEPTANCE = TOPIC/'fresh_fixed2_cmcc_20261030/single_20261030_v3/longbench/merged/nine_task_acceptance.json'
REFERENCE_ARGS = C0/'06_down_proj/trainer_state/checkpoint-500/training_args.bin'
SUPPORT = ARTIFACTS/'adaptive_class_cmcc_20261030/evaluation_support'
PREPARED = RESULT/'prepared.json'
GPU = 3
CONTRACT = dict(parent=RAW_ID,control=CONTROL_ID,category='down_proj',all_projection_D=252,
    independent_decoder_stages=504,independent_matrices=1008,matrix_leaves=576,auxiliary_leaves=1152,
    precision='Original actual FP32 D masters/BF16 compute and deployment; frozen codes/protection/rotation/non-D.',
    optimizer='Stage-local Muon NS5/momentum.95/Nesterov/existing .2sqrt(maxshape); auxiliary original AdamW(.9,.999)/eps1e-8/WD0.',
    calibration='First real clipped gradient; clone-only nonzero baseLR1e-5 fresh original AdamW vs baseMuon total unrounded FP32 Frobenius norm ratio; one fixed kappa for all matrices.',
    scheduler='Canonical HF500 cosine/warmup.01. External lr/initial_lr/base_lrs unchanged1e-5; matrix effective_lr=scheduled_lr*kappa internally; aux scheduled_lr.',
    first_step='Actual scheduled LR0: initialize and advance both optimizer states to1, parameters unchanged; step2 original positive LR.',
    save='Original MixedCatTrainer regular250/500 standard optimizer/scheduler/RNG + FP32 state; short save1/return/step2; no exact-resume claim.',
    data_target='Original five-source train, teacher, losses/reduction, B4/max1024, clip1.5, stage_seed37/data_seed31, R0 unchanged.',
    scope='Only this last down-category500 optimizer comparison; not all seven categories, OPD, VAE fitting or a LR grid.')


def pins():
    paths = set(HERE.glob('*.py'))
    paths.update(REPO/'experiments/adaptive_class_evaluation_20261030'/name for name in
        ('bindings.py','loading.py','full8.py','evaluate.py','summarize.py','predict.py','merge.py'))
    paths.update((Path(mature.__file__),Path(recovery.__file__).with_name('training_checkpoint.py'),
        REFERENCE_ARGS,CONTROL_ACCEPTANCE))
    return {**original.source_pins(),**{str(path.resolve()):sha(path) for path in paths}}


def check_pins(values):
    executor.check_pins(values)


def source():
    common = original.bind(COMMON)
    report,meta,acceptance = read(C0/'report.json'),read(RAW/checkpoint.META_FILE),read(CONTROL_ACCEPTANCE)
    require(report['status']=='completed_fresh_C0_fixed2_pipeline' and report['exit_code']==0
        and report['source_binding']==common and report['stages'][2]['round_base_checkpoint_id']==RAW_ID
        and meta['checkpoint_id']==RAW_ID and report['final_checkpoint_id']==CONTROL_ID
        and acceptance['checkpoint_id']==CONTROL_ID and acceptance['compressed_nine_task_score']==63.58416595020284
        and [len(meta[name]) for name in ('native_modules','compact_modules','pending_dense_modules')]==[216,36,0],
        'Use the completed original raw-down C0 and its genuine original AdamW500/full9 control.')
    require(sha(RAW/checkpoint.STATE_FILE)==meta['state_sha256'],'Raw C0 deployed payload changed.')
    return dict(raw_directory=str(RAW),raw_checkpoint_id=RAW_ID,raw_metadata_sha256=sha(RAW/checkpoint.META_FILE),
        raw_state_sha256=meta['state_sha256'],
        raw_weight_stat=dict(bytes=(RAW/checkpoint.STATE_FILE).stat().st_size,mtime_ns=(RAW/checkpoint.STATE_FILE).stat().st_mtime_ns),
        control_checkpoint_id=CONTROL_ID,control_acceptance=str(CONTROL_ACCEPTANCE),control_acceptance_sha256=sha(CONTROL_ACCEPTANCE),
        c0_report=str(C0/'report.json'),c0_report_sha256=sha(C0/'report.json'),common=common)


def recipe(output):
    common = original.bind(COMMON)
    cat,hf,training,vae,cfgs,runtime = original.recipe(output,common)
    stage = recovery.canonical.resolve_cat_after_category_stage(cat,training,category='down_proj',round_idx=6)
    expected = json.loads(json.dumps(read(C0/'report.json')['stages'][2]['original_stage']))
    actual = json.loads(json.dumps(asdict(stage)))
    expected.pop('output_dir');actual.pop('output_dir')
    require(actual==expected and stage.stage_seed==37 and stage.config.data.data_seed==31
        and stage.config.opt.steps==500 and stage.config.opt.batch_size==4
        and stage.config.opt.resolved_decoder_lr()==1e-5 and stage.config.opt.max_grad_norm==1.5
        and stage.config.opt.warmup_ratio==.01 and stage.config.opt.lr_scheduler_type=='cosine',
        'Preserve the entire original category contract; only the explicit optimizer adapter may differ.')
    return cat,hf,training,vae,runtime,stage


def cpu_environment():
    executor.cpu_environment()
    require(not torch.cuda.is_initialized(),'CPU-only preparation must not initialize CUDA.')


def prepared():
    value=read(PREPARED)
    require(value['schema']=='C0_down_category_Muon_prepared_v1' and value['status']=='verified_CPU_actual_layout_and_original_contract'
        and value['contract']==CONTRACT,'Require this fixed last-category optimizer comparison preparation.')
    check_pins(value['source_sha256'])
    require(source()==value['source_binding'],'Original raw/control source binding changed.')
    return value

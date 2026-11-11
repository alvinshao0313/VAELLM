"""Bind the approved current36-down scope against the same-raw all252 control."""
from copy import deepcopy
from pathlib import Path

from experiments.cat_down_vae_muon_closure_cmcc_20261103 import bindings as original

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = original.ARTIFACTS / HERE.name, original.TOPIC / HERE.name
PREPARED = RESULT / 'prepared.json'
DATA_FIXTURE = RESULT / 'actual_data_peak.pt'
RAW, RAW_ID = original.RAW, original.RAW_ID
CONTROL_ID = '983dc190-d6ac-4a51-8225-83389b35011c'
CONTROL_OUTPUT = original.OUTPUT / 'formal'
CONTROL_ACCEPTANCE = original.RESULT / 'evaluation/longbench/merged/nine_task_acceptance.json'
SUPPORT, PYTHON, REPO = original.SUPPORT, original.PYTHON, original.REPO
checkpoint, recovery, executor, c = original.checkpoint, original.recovery, original.executor, original.c
torch, read, write, sha, require = original.torch, original.read, original.write, original.sha, original.require
CONTRACT = deepcopy(original.CONTRACT)
CONTRACT.update(schema='current36_down_original_Adam500_v1', control=CONTROL_ID,
    projection_decoders=36, trainable_leaves=216, trainable_elements=767232,
    precision='Current36 down FP32 decoder masters; BF16 compute/deployment; all non-down decoders frozen in original BF16; LoRA0.',
    frozen='All code/protection/rotation/backbone and all non-down decoder persistent fields exact.',
    sole_variable='Train only all36 native down decoders rather than all252 projection decoders; global clipping inventory and decode execution consequently differ.',
    completion='New-scope real-peak two-update/save/fresh within1GPUh; original500/fresh within8GPUh; same-final-UUID complete nine within10GPUh.',
    source_scope='Original D32 VAE-Muon raw c5ccf174; reuse all252 Adam500 983dc190 as control; no D16/gate/combination.',
    comparison='Fixed500 terminal only, no250 selection; <=64.30526846061926 ends point, micropositive is single-seed candidate only.')


def pins():
    paths = set(HERE.glob('*.py')) | set(HERE.glob('*.sh'))
    return {**original.pins(), **{str(path.resolve()): sha(path) for path in paths}}


check_pins = original.check_pins
cpu_environment = original.cpu_environment


def source():
    binding = original.source()
    report = read(CONTROL_OUTPUT / 'report.json')
    strict = read(CONTROL_OUTPUT / 'strict_reload.json')
    meta = read(CONTROL_OUTPUT / 'final_model' / checkpoint.META_FILE)
    control = read(CONTROL_ACCEPTANCE)
    require(report['status'] == 'completed_VAE_Muon_original_Adam500_and_fresh' and report['exit_code'] == 0
        and report['source_binding'] == binding and report['recovery']['actual_steps'] == 500
        and report['recovery']['frozen_persistent_fields_exact']
        and strict['exit_code'] == 0 and strict['same_input_logits_exact']
        and report['final_checkpoint_id'] == strict['checkpoint_id'] == meta['checkpoint_id'] == control['checkpoint_id'] == CONTROL_ID
        and control['compressed_nine_task_score'] == 64.30526846061926
        and control['downstream_compressed']['leaf_count'] == 64
        and control['downstream_compressed']['sample_count'] == 24742,
        'Require the completed same-raw all252 originalAdam500/full9 control.')
    return dict(raw_binding=binding, raw_checkpoint_id=RAW_ID, control_checkpoint_id=CONTROL_ID,
        control_nine_task_score=control['compressed_nine_task_score'], control_acceptance=str(CONTROL_ACCEPTANCE),
        evidence_sha256={str(path): sha(path) for path in
            (CONTROL_OUTPUT/'report.json', CONTROL_OUTPUT/'strict_reload.json', CONTROL_OUTPUT/'final_model'/checkpoint.META_FILE, CONTROL_ACCEPTANCE)})


def recipe(output):
    return original.recipe(output)


def prepared():
    record = read(PREPARED)
    require(record['status'] == 'verified_CPU_current36_down_and_original_data_contract' and record['contract'] == CONTRACT,
        'Require the approved current36down CPU selection/data proof.')
    check_pins(record['source_sha256'])
    require(source() == record['source_binding'] and sha(DATA_FIXTURE) == record['data']['fixture_sha256'],
        'Original raw/control or true peak data fixture changed.')
    deadline = read(RESULT / 'deadline_cpu.json')
    require(deadline['status'] == 'verified_genuine_plan_absolute_deadline_and_unchanged_protocol'
        and deadline['exit_code'] == 0 and deadline['source_sha256'] == record['source_sha256'],
        'Require the actual10h full21 plan and identical absolute-deadline CPU proof for these exact sources.')
    return record

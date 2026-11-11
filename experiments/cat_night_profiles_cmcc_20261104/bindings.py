"""Exactly eighteen explicit BF16 profile candidates, separate night authorization."""
from copy import deepcopy
from pathlib import Path
import time

from experiments.cat_batch_profiles_cmcc_20261104 import bindings as prior
from experiments.cat_auto_protocol_cmcc_20261104.config import resolve_recipe
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
from experiments.cat_offline_execution_cmcc_20261104.protocol import code_identity, load_baseline, verify_files
from experiments.cat_offline_execution_cmcc_20261104.results import worker_result
from . import search

HERE = Path(__file__).resolve().parent
CARDS = {0:'q_proj', 1:'k_proj', 2:'v_proj', 3:'o_proj', 4:'gate_proj', 5:'down_proj'}
RATES = ('2', '5/2', '3')
BASELINE = prior.REPO/'result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/current/bf16_quality_receipt.json'
SECONDS = dict(technical_validation=900, formal=3600, full8=900)
CANDIDATE_SECONDS, QUEUE_SECONDS, TOTAL_GPU_HOURS = 5400, 16200, 27
read, write, write_new, bound, sha = prior.read, prior.write, prior.write_new, prior.bound, prior.sha


def pins():
    return {**prior.pins(), **code_identity(), **{str(path.resolve()):sha(path) for path in HERE.glob('*.py')}}


def make_manifest(output):
    original, inventory, identity = prior.common(); baseline = load_baseline(BASELINE)
    require(baseline['model_identity'] == identity['model'], 'Night candidates require the actual common BF16 evaluation source.')
    queues = {}
    for gpu, category in CARDS.items():
        rows = []
        for rate in RATES:
            recipe = resolve_recipe(inventory, category, rate, dim=32, stages=2, transpose=False, rotation_level='32',
                protection_fraction='1/64', encoder_width=128, decoder_width=128, optimizer='adamw', multiplier=1)
            rows.append(dict(label=category+'_k'+str(recipe['K']), candidate_id=candidate_id(identity, recipe), recipe=recipe))
        queues[str(gpu)] = rows
    value = dict(schema='cat_night_six_class_rate_profiles_v1', status='prepared_requires_actual_authorization',
        output_root=str(Path(output).resolve()), queues=queues, cards={str(key):value for key,value in CARDS.items()},
        run_identity=identity, parent=original['parent'], inventory=bound(original['inventory_receipt']), data_plan=bound(original['data_plan']),
        baseline=bound(BASELINE), common_protocol_identity=baseline['protocol_identity'], source_sha256=pins(),
        seconds=SECONDS, candidate_seconds=CANDIDATE_SECONDS, queue_seconds=QUEUE_SECONDS, total_GPU_hours=TOTAL_GPU_HOURS,
        technical_steps=2, stages=2, original_schedule_steps=10000, formal_steps=10000,
        role='Explicit rate profiles under the validated common control; not a finalized SC-CAT anchor or generated HPO search',
        failure_rule='Fail only this candidate dependencies; next declared distinct candidate may continue. Changed common source/protocol stops unstarted work.',
        previous_up_cost_reference=bound(prior.CLOSURE), no_retry=True, no_occupancy=True, no_KD=True)
    return dict(value, manifest_identity=digest(value))


def verify_manifest(manifest, output):
    require(manifest == make_manifest(output), 'Fixed recipes/source/data/protocol/cards/budget or unique queue root changed.')
    return manifest


def approval(manifest, manifest_path, authorization_path):
    require(read(manifest_path)==manifest, 'Require this actual prepared manifest file.')
    verify_manifest(manifest, Path(manifest_path).parent)
    auth = read(authorization_path)
    expected = dict(manifest_identity=manifest['manifest_identity'], manifest_sha256=sha(manifest_path),
        output_root=manifest['output_root'], cards=manifest['cards'], total_GPU_hours=27,
        candidate_seconds=5400, queue_seconds=16200, seconds=SECONDS)
    require(auth.get('user_approved') is True and isinstance(auth.get('user_authorization'),dict), 'Actual night user approval is required before any worker/request.')
    user = auth['user_authorization']
    require(isinstance(user.get('reference'),str) and bool(user['reference'].strip())
        and all(user.get(key)==value for key,value in expected.items()), 'Night authorization must bind the exact manifest SHA/root/cards/candidates and27GPUh caps.')
    return user


def phase_window(queue_started, candidate_started, now, phase):
    require(phase in SECONDS and queue_started <= candidate_started <= now, 'Use actual queue/candidate starts and one fixed stage.')
    deadline = min(now+SECONDS[phase], candidate_started+5400, queue_started+16200)
    require(deadline > now, 'Finite night queue/candidate budget exhausted; no extension.')
    return deadline


def technical_proof(manifest, row, directory):
    path=directory/'technical_validation/completed.json'; registry=directory/'technical_validation_request.json'; req=read(registry)
    require(req['night_manifest_identity']==manifest['manifest_identity'] and req['run_identity']==manifest['run_identity']
        and req['candidates'][0]['candidate_id']==row['candidate_id'] and req['phase']=='technical_validation', 'Require this own new night technical registry.')
    result=worker_result(path,row['candidate_id'],require_formal=False,request_reference=bound(registry))
    completed=read(path)
    search.batched.native._validate_model_path_evidence(completed,read(manifest['inventory']['path'])['inventory'],manifest['run_identity'],req['candidates'])
    require(result['recipe']==row['recipe'] and result['phase']=='technical_validation' and not result['formal_eligible'], 'Technical fit cannot become formal or another candidate.')
    return bound(path)


def request(manifest, manifest_path, authorization_path, gpu, row, directory, phase, queue_started, candidate_started):
    user=approval(manifest,manifest_path,authorization_path); now=time.time()
    require(gpu in CARDS and row in manifest['queues'][str(gpu)] and phase in ('technical_validation','formal'), 'Only declared GPU and distinct planned candidate stages may execute.')
    deadline=phase_window(queue_started,candidate_started,now,phase)
    proof=technical_proof(manifest,row,directory) if phase=='formal' else None
    value=dict(schema='cat_auto_explicit_candidates_v0',phase=phase,inventory_receipt=manifest['inventory']['path'],
        inventory_receipt_sha256=manifest['inventory']['sha256'],data_plan=manifest['data_plan']['path'],data_plan_sha256=manifest['data_plan']['sha256'],
        run_identity=manifest['run_identity'],parent=manifest['parent'],candidates=[dict(candidate_id=row['candidate_id'],recipe=row['recipe'],purpose='profile',
            complete_dependency_gpu_hours=SECONDS[phase]/3600,cost_basis=dict(kind='actual_authorized_night_stage_cap',reference=user['reference'],includes='Complete source/teacher/fit/encode/readout/save/fresh costs'))],
        output_dir=str(directory/phase),absolute_deadline_epoch=deadline,authorized_start_epoch=now,authorized_gpu_hours=SECONDS[phase]/3600,
        physical_GPU=gpu,authorization_reference=user['reference'],source_sha256=manifest['source_sha256'],shared_overhead_gpu_hours=0.,
        shared_cost_scope='All overhead included in sole stage cap',technical_steps=2 if phase=='technical_validation' else None,
        required_model_path_evidence=proof,full8=False,KD=False,automatic_retry=False,night_manifest_identity=manifest['manifest_identity'],
        night_authorization=bound(authorization_path),formal_initialization='Fresh original BF16; never technical checkpoint or previous rate weights')
    search.validate_request(value); return value


def integrity(manifest):
    verify_files(manifest['source_sha256'])
    require(sha(manifest['inventory']['path'])==manifest['inventory']['sha256'] and sha(manifest['data_plan']['path'])==manifest['data_plan']['sha256'], 'Shared actual inventory/DataPlan changed.')
    baseline=load_baseline(manifest['baseline']['path'])
    require(sha(manifest['baseline']['path'])==manifest['baseline']['sha256'] and baseline['protocol_identity']==manifest['common_protocol_identity'], 'Shared BF16/common full8 protocol changed.')

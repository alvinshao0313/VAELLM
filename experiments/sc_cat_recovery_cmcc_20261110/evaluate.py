"""Bind new-run exports to the unchanged complete64-leaf/24742 evaluator."""
import argparse
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_offline_execution_cmcc_20261104 import evaluate as native
from experiments.cat_offline_execution_cmcc_20261104.protocol import load_baseline
from experiments.sc_cat_live_cmcc_20261106.evaluate import collect
from train_utils import checkpoint_v6 as v6
from . import io


def execute(request):
    manifest, auth = io.validate_request(request, cpu_controller=True)
    report = io.read(io.bound_path(request['deployment_report']))
    proof = io.read(io.bound_path(report['fresh_proof_reference']))
    require(report['phase']=='formal' and proof==report['fresh_forward_evidence']
        and proof['checkpoint_id']==report['uuid'] and proof['weight_digest']==report['digest']
        and proof['state_exact'] and proof['forward_executed'] and proof['finite_logits']
        and proof['gpu_validation'] and proof['exit_code']==0, 'Complete actual fresh proof required.')
    directory = Path(report['path'])
    meta = v6.load_v6_meta(str(directory))
    baseline = load_baseline(io.bound_path(manifest['baseline']))
    source = meta['extra_meta']['cat_auto_selected_node']
    raw = io.read(io.bound_path(report['source_raw_reference']))
    require(source==report['export_evidence'] and source['phase']=='formal'
        and raw['phase']=='formal' and raw['formal_eligible']
        and raw['run_identity']['model']==baseline['model_identity']
        and raw['round_base_uuid']==source['round_base_id'], 'The evaluated boundary must originate from its own formal raw and node.')
    endpoint = dict(kind='native_v6', path=str(directory), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=io.binding(directory/'checkpoint_meta.json')['sha256'],
        state_sha256=report['digest'], model_identity=baseline['model_identity'], phase='formal', formal_eligible=True)
    start = time.time()
    require(0<request['deadline_epoch']-start<=900, 'Complete eight-task window is at most900seconds.')
    authorization = dict(authorization_reference=auth['authorization_reference'], physical_GPU=request['gpu'],
        authorized_GPU_hours=.25, authorized_start_epoch=start, absolute_deadline_epoch=request['deadline_epoch'])
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    plan = native.make_plan(result=dict(endpoint=endpoint,phase='formal',formal_eligible=True),
        baseline=baseline, baseline_path=io.bound_path(manifest['baseline']), output=output, authorization=authorization)
    plan['live_source'] = dict(deployment_report=request['deployment_report'],
        source_node_id=report['export_evidence']['source_node_id'], source_round_base_id=report['export_evidence']['round_base_id'])
    io.write(output/'plan.json',plan)
    native.run(output/'plan.json')
    return collect(output/'plan.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    execute(io.read(parser.parse_args().request))

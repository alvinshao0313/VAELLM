"""Full-eight evaluation of an actual finalized live node or raw0 boundary."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore, load_complete_eight_score
from experiments.cat_offline_execution_cmcc_20261104 import evaluate as native
from experiments.cat_offline_execution_cmcc_20261104.protocol import collect_native, load_baseline
from train_utils import checkpoint_v6 as v6

from . import live_io as io


def read_score(path):
    value = io.read(io.bound_path(path))
    return FullEightScore(value['task_scores'], value['evaluation_id'], value['evidence'])


def prepare(request):
    report_path = io.bound_path(request['deployment_report'])
    report = io.read(report_path)
    baseline_path = io.bound_path(request['baseline'])
    baseline = load_baseline(baseline_path)
    approval = io.read(io.bound_path(request['authorization']))
    require(approval['confirmed'] and approval['user_approved']
        and approval['authorization_reference'] == io.APPROVAL
        and approval['total_GPU_hours'] == 36 and request['gpu'] in approval['gpus'],
        'The real live batch approval and assigned GPU are required.')
    directory = Path(report['path']).resolve()
    meta = v6.load_v6_meta(str(directory))
    proof = report['fresh_forward_evidence']
    require(report['phase'] == 'formal' and meta['checkpoint_kind'] == 'category_boundary'
        and proof['checkpoint_id'] == report['uuid'] == meta['checkpoint_id']
        and proof['weight_digest'] == report['digest']
        and proof['exit_code'] == 0 and proof['state_exact']
        and proof['forward_executed'] and proof['finite_logits']
        and proof['gpu_validation'] and str(proof['device']).startswith('cuda:'),
        'Evaluate only the actual formal deployment with independent GPU fresh proof.')
    actual_proof = io.read(io.bound_path(report['fresh_proof_reference']))
    require(actual_proof == proof, 'The deployment fresh proof changed.')
    source = meta['extra_meta']['cat_auto_selected_node']
    require(source == report['export_evidence'] and source['phase'] == 'formal',
        'The deployment must preserve its actual source node/raw provenance.')
    raw = io.read(io.bound_path(report['source_raw_reference']))
    require(raw['phase'] == 'formal' and raw['formal_eligible']
        and raw['run_identity']['model'] == baseline['model_identity']
        and raw['round_base_uuid'] == source['round_base_id'],
        'The actual deployment source must be this model and its own formal live raw.')
    endpoint = dict(kind='native_v6', path=str(directory), checkpoint_id=report['uuid'],
        model_identity=baseline['model_identity'],
        metadata_sha256=io.binding(directory / 'checkpoint_meta.json')['sha256'],
        state_sha256=io.binding(directory / meta['state_dict_file'])['sha256'],
        phase='formal', formal_eligible=True)
    require(endpoint['state_sha256'] == report['digest'], 'The bound deployment payload changed.')
    started = time.time()
    remaining = request['deadline_epoch'] - started
    require(0 < remaining <= 900, 'A complete eight-task evaluation has at most15minutes.')
    authorization = dict(authorization_reference=io.APPROVAL, physical_GPU=request['gpu'],
        authorized_GPU_hours=.25, authorized_start_epoch=started,
        absolute_deadline_epoch=request['deadline_epoch'])
    output = Path(request['output']).resolve()
    require(output.is_relative_to(Path(approval['output_root']).resolve()), 'Write only this approved batch.')
    output.mkdir(parents=True, exist_ok=False)
    plan = native.make_plan(result=dict(endpoint=endpoint, phase='formal', formal_eligible=True),
        baseline=baseline, baseline_path=baseline_path, output=output, authorization=authorization)
    plan['live_source'] = dict(deployment_report=io.binding(report_path),
        source_node_id=source['source_node_id'], source_round_base_id=source['round_base_id'],
        qualification='Independent formal deployment verified; full8 and boundary acceptance still pending.')
    io.write(output / 'plan.json', plan)
    return output / 'plan.json'


def collect(plan_path):
    plan = io.read(plan_path)
    baseline, _meta = native.verify_plan(plan)
    output = Path(plan['output_dir'])
    summary = io.read(output / 'full8_summary.json')
    actual = collect_native([Path(row['path']) for row in summary['input_results']],
        endpoint=plan['endpoint'], baseline=baseline)
    require(actual == summary, 'Saved full8 summary differs from its actual result shards.')
    protocol = dict(plan_path=str(plan_path), plan_sha256=io.binding(plan_path)['sha256'],
        checkpoint_meta_sha256=plan['endpoint']['metadata_sha256'], protocol_id=baseline['protocol_identity'],
        leaf_count=64, sample_count=24742, task_metric_keys=baseline['task_metric_keys'],
        expected_stages=plan['expected_stages'])
    score = load_complete_eight_score(summary_path=output / 'full8_summary.json',
        evaluation_path=output / 'evaluation.json', checkpoint_dir=plan['endpoint']['path'],
        source_checkpoint_id=plan['live_source']['source_node_id'], protocol_manifest=protocol)
    io.write(output / 'score.json', dict(task_scores=dict(score.task_scores),
        evaluation_id=score.evaluation_id, evidence=dict(score.evidence),
        source=plan['live_source'], mean_percent=float(score.mean * 100)))
    return score


def execute(request):
    plan = prepare(request)
    native.run(plan)
    return collect(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    execute(io.read(args.request))


if __name__ == '__main__':
    main()

"""Native partial-node fresh checks, full8 and the original fixed train readout."""
import argparse
import logging
import math
from pathlib import Path
import time
from types import SimpleNamespace

from . import bindings as b
from experiments.cat_geometry_cohort_cmcc_20261102 import evaluate as original


def warm(model, count):
    tools = original.module(b.SNAPSHOT / 'tools/cat_eval.py', '_gate_muon_native_prewarm')
    result = tools._prepare_model_for_eval(model, 'cuda', 8, logging.getLogger(__name__), {})
    b.require(result['warmed'] == result['total'] == count and not result['failed'], 'Warm the actual partial native inventory.')
    return result


@b.torch.no_grad()
def deployed_logits(model, tokens):
    result = model(input_ids=tokens.to('cuda'), use_cache=False, logits_to_keep=1).logits
    b.require(result.dtype == b.torch.bfloat16 and b.torch.isfinite(result).all().item(), 'Require finite actual BF16 deployed logits.')
    return result.detach().cpu().contiguous()


def fresh(phase, *, cpu):
    from experiments.cat_geometry_cohort_cmcc_20261102.prepare import verify
    root = b.OUTPUT / 'muon' / phase
    report = b.read(root / 'report.json')
    b.require(report['status'] == 'raw2_completed' and report['exit_code'] == 0 and report['contract'] == b.CONTRACT,
        'Consume only this genuinely completed gate Muon phase.')
    directory = Path(report['raw2']['path'])
    if cpu:
        verify(directory, report['state'], root / 'strict_reload.json', 'muon')
        return
    expected = b.torch.load(root / 'same_input_deployment.pt', map_location='cpu', weights_only=True)
    # The unchanged consumer checks the independent CPU strict/complement proof first.
    b.require(original.checkpoint('muon', phase) == directory, 'Fresh consumer source differs.')
    model, meta, audit = original.load(directory)
    b.require(meta['checkpoint_id'] == expected['checkpoint_id'] and audit['persistent_tensors'] == 2559
        and expected['full_state'] == report['state'] == b.base.fork.state_digest(
            {name:value.detach().cpu() for name,value in model.state_dict().items()}),
        'Independent deployed consumer changed one of2559 actual persistent fields.')
    logits = deployed_logits(model, expected['input_ids'])
    b.require(b.torch.equal(logits, expected['logits']), 'Independent fresh consumer changed same-input BF16 logits.')
    b.write(root / 'deployment_fresh.json', dict(status='verified_native_full_state_and_BF16_logits', exit_code=0,
        checkpoint_id=meta['checkpoint_id'], persistent_fields=2559, same_input_logits_exact=True,
        load_audit=audit, full_state=expected['full_state'], warmup=audit['prewarm'],
        peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved()))


def shard(arm, component, output):
    from transformers import AutoTokenizer
    directory = original.checkpoint(arm)
    model, meta, audit = original.load(directory)
    b.base.validate_raw2(meta, arm)
    b.require(audit['persistent_tensors'] == 2559, 'Native raw2 requires all2559 persistent fields.')
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(b.HF, use_fast=True, trust_remote_code=True, local_files_only=True)
    api = original.module(b.TOPIC / 'lm_eval_sample_artifacts_20261010/eval_utils_with_samples.py', '_gate_muon_sample_evaluation')
    options = SimpleNamespace(tasks='mmlu' if component == 'mmlu' else b.SEVEN, num_fewshot=0,
        batch_size='64' if component == 'mmlu' else '32', lm_limit=None,
        model_path=str(b.HF), eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output.parent / (component + '_logs')), mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    result = api.run_lm_eval(model, tokenizer, options)
    samples = result['sample_artifacts']
    count, leaves = (14042,57) if component == 'mmlu' else (10700,7)
    b.require(samples['total_count'] == count and len(samples['tasks']) == leaves, 'Require the complete official shard.')
    b.write(output, dict(status='completed_full_' + component, arm=arm, checkpoint_id=meta['checkpoint_id'],
        checkpoint_dir=str(directory), checkpoint_loader='v6', strict_load=True, load_audit=audit,
        checkpoint_format=meta['format'], base_model_path=str(b.HF), evals=dict(lm_eval=result),
        boundary='Partial native180+future72dense stage only; no final <=2.6 or nine-task claim.'))


def full8(arm):
    from experiments.qwen3_w25.summarize_results import TASK_METRICS
    from experiments.w25_eval.acceptance import leaf_metrics, _group_agrees
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    root = b.RESULT / 'control' if arm == 'control' else b.OUTPUT / 'muon'
    paths = [b.CONTROL_ROOT / 'mmlu.json' if arm == 'control' else root / 'mmlu.json', root / 'seven.json']
    expected_id = b.CONTRACT['control'] if arm == 'control' else b.read(root / 'formal/report.json')['raw2']['checkpoint_id']
    scores, subjects, versions = {}, {}, {}
    for path, batch in zip(paths, (64,32), strict=True):
        payload = b.read(path)
        b.require(payload['checkpoint_id'] == expected_id and payload['strict_load']
            and payload['checkpoint_loader'] == 'v6' and payload['load_audit']['persistent_tensors'] == 2559,
            'The two native full8 shards must consume the same real endpoint.')
        result = payload['evals']['lm_eval']
        b.require(result['num_fewshot'] == 0 and result['limit'] is None and str(result['batch_size']) == str(batch),
            'Keep complete zero-shot B64/B32 evaluation.')
        current, sub, actual_batch = leaf_metrics(result['raw_results'],result['n_samples'],result['n_shot'],result['configs'])
        _group_agrees(current,result['raw_results'],result['group_results'])
        b.require(actual_batch == batch and not set(current) & set(scores), 'Wrong leaf shard or batch.')
        scores.update(current)
        subjects.update(sub)
        versions.update({task:result['versions'][task] for task in result['n_samples']})
    samples = _sample_artifacts(paths)
    b.require(set(scores) == set(TASK_METRICS) and len(subjects) == 57 and samples['sample_count'] == 24742,
        'Require all64 leaves/24742 raw sample artifacts.')
    b.write(root / 'full8_summary.json', dict(status='completed_native_partial_full8', checkpoint_id=expected_id,
        task_metrics=scores, mean_percent=math.fsum(scores.values())*100/8, mmlu_subject_metrics=subjects,
        leaf_count=64, sample_count=24742, versions=versions, sample_artifacts=samples,
        inputs={str(path):b.sha(path) for path in paths},
        boundary='Gate partial-stage screen with future dense costs; no automatic suffix or final product acceptance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True, choices=('fresh_cpu','fresh_gpu','mmlu','seven','full8','readout'))
    parser.add_argument('--arm', default='muon', choices=('muon','control'))
    parser.add_argument('--phase', default='formal', choices=('short','formal'))
    args = parser.parse_args()
    b.prepared()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    if args.component in ('fresh_cpu','fresh_gpu'):
        fresh(args.phase, cpu=args.component == 'fresh_cpu')
    elif args.component == 'full8':
        full8(args.arm)
    else:
        root = b.RESULT / 'control' if args.arm == 'control' else b.OUTPUT / 'muon'
        output = root / (args.component + '.json')
        b.require(not output.exists(), 'Never overwrite native evaluation evidence.')
        if args.component == 'readout':
            original.readout(args.arm, output, args.phase)
        else:
            shard(args.arm, args.component, output)

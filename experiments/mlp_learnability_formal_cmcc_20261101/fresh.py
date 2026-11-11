"""Independent saved-snapshot strict/frozen/protection/two-card BF16 readback."""
import argparse
import json
from pathlib import Path
import time
import traceback

from . import bindings as b
from .source import execution


@b.torch.no_grad()
def audit(directory,devices):
    began = time.monotonic()
    report_path = directory/'report.json'
    report = b.read(report_path)
    b.require(report['physical_GPUs'] == list(map(int,devices)), 'Use the identical actual training pair for fresh.')
    model,meta,loaded,budget = b.checkpoint.load(directory/'final_model')
    report,expected = execution(directory/'final_model',report_path,meta)
    actual = b.life.state_records(model)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and not actual['extra']
        and actual['tensors'] == expected and b.life.non_mlp_proof(model) == report['source_non_mlp'],
        'The independently loaded actual snapshot or frozen complement differs.')
    groups = b.export.deployment_contract(model,report['arm'],'mixed')
    b.require(budget == report['export']['bit_budget'], 'Actual fresh storage accounting differs.')
    handles,mapping = b.life.apply_map(model)
    b.core.install_precision_runtime(model,b.torch.bfloat16)
    warm = b.core.prewarm_deployment(model)
    protection = b.export.protection_proof(model,report['protection_records'],deployed=True)
    fixture = b.torch.load(directory/'same_input_output.pt',map_location='cpu',weights_only=True)
    logits = b.run_short.probe(model,fixture)
    comparison = b.run_short.comparison(fixture['logits'],logits)
    b.require(comparison['finite'] and comparison['exact'], 'Same two-card BF16 snapshot/fresh logits differ.')
    after = b.life.state_records(model)
    b.require(not after['extra'] and after['tensors'] == expected
        and b.life.non_mlp_proof(model) == report['source_non_mlp'], 'Fresh deployment mutated persistent snapshot state.')
    value = dict(schema='all_MLP_learning_snapshot_fresh_v1',status='verified',arm=report['arm'],
        actual_updates=report['actual_updates'],planned_updates=report['planned_updates'],
        validation_only=report['validation_only'],diagnostic_overbudget=report['arm']=='dense',
        checkpoint_id=meta['checkpoint_id'],source_checkpoint_id=b.C0_ID,
        full_state_exact=True,persistent_tensors_compared=len(expected),non_MLP_source_exact=True,
        protection=protection,same_complete_input_BF16_logits_exact=True,finite=True,
        deployed_to_fresh=comparison,physical_GPUs=list(map(int,devices)),layer_device_map=mapping,
        groups=groups,warmup=warm,bit_budget=budget,training_report=b.binding(report_path),
        export_meta=b.binding(directory/'final_model'/b.checkpoint.META_FILE),
        fixture=report['export']['fixture'],expected_state=report['export']['expected_state'],
        parent_weights_loaded=False,elapsed_seconds=time.monotonic()-began)
    b.write(directory/'fresh_audit.json',value)
    for handle in handles: handle.remove()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--gpus',required=True,type=b.gpus)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    b.require(not (directory/'fresh.exit').exists() and not (directory/'fresh_audit.json').exists(),
        'Never overwrite an independent fresh audit.')
    b.resources(args.gpus)
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    code = 1
    try:
        value = audit(directory,args.gpus)
        code = 0
        print(json.dumps(dict(status=value['status'],checkpoint_id=value['checkpoint_id'],step=value['actual_updates'])),flush=True)
    except BaseException:
        b.write(directory/'fresh_failure.json',dict(error=traceback.format_exc()))
        raise
    finally:
        (directory/'fresh.exit').write_text(str(code)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()

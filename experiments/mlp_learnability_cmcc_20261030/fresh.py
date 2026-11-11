"""Independent exported-state-only strict, protection, budget and BF16 readback."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from . import export,lifecycle as life

torch = b.torch


def comparison(reference,candidate):
    b.require(reference.shape == candidate.shape and reference.dtype == candidate.dtype == torch.bfloat16
        and torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item(),
        'Compare complete finite BF16 logits on the same shape/path.')
    difference = candidate.float()-reference.float()
    norm = float(reference.double().norm())
    return dict(exact=torch.equal(reference,candidate),relative_L2=float(difference.double().norm())/norm if norm else None,
        max_abs=float(difference.abs().max()),changed_elements=int((reference!=candidate).sum()),
        changed_top1_positions=int((reference.argmax(-1)!=candidate.argmax(-1)).sum()),finite=True)


@torch.no_grad()
def audit(directory):
    started = time.monotonic()
    directory = Path(directory).resolve(strict=True)
    report = b.read(directory/'report.json')
    b.require((directory/'run.exit').read_text(encoding='utf-8').strip() == '0'
        and report['schema'] == 'all_MLP_short2_v1' and report['exit_code'] == 0
        and report['status'] == 'training_and_BF16_export_complete_fresh_pending', 'Require an actually completed two-update export.')
    b.require(report['physical_GPUs'] == [6,7] and report['layer_device_map'] == life.LAYER_MAP
        and report['group_indices'] == [2,10] and report['validation_only'] is True
        and report['prototype_only'] == report['source_binding']['prototype_only']
        and report['diagnostic_overbudget'] == (report['arm']=='dense'), 'Unexpected diagnostic identity or resource/protocol binding.')
    for path,expected in report['source_sha256'].items():
        b.require(b.sha(path) == expected, 'Bound implementation changed before independent readback: '+path)
    b.require(b.runtime_bindings() == report['source_binding']['runtime_sha256'], 'Bound source runtime changed before readback.')
    final = directory/'final_model'
    b.require(Path(report['export']['directory']).resolve(strict=True) == final, 'Fresh load must consume this run\'s own single deployment.')
    for key,filename in (('expected_state','expected_state.json'),('fixture','same_input_output.pt')):
        binding = report['export'][key]
        b.require(Path(binding['path']).resolve(strict=True) == directory/filename and b.file_binding(directory/filename) == binding,
                  'Changed or external fresh audit fixture: '+key)
    # The canonical loader reads only this new model's one effective state file.
    model,meta,result,budget = b.checkpoint.load(final)
    provenance = meta['provenance']
    b.require(meta['checkpoint_id'] == report['export']['checkpoint_id'] and budget == report['export']['bit_budget']
        and provenance['schema'] == 'all_MLP_learnability_short2_export_v1'
        and provenance['execution'] == {key:report[key] for key in provenance['execution']}
        and provenance['arm'] == report['arm'] and provenance['actual_updates'] == 2
        and provenance['schedule_steps'] == 500 and provenance['group_indices'] == [2,10]
        and provenance['source_binding'] == report['source_binding']
        and provenance['source_non_mlp'] == report['source_non_mlp']
        and provenance['source_protection'] == report['protection_records']
        and provenance['diagnostic_overbudget'] == (report['arm']=='dense')
        and provenance['prototype_only'] == report['prototype_only'] and provenance['validation_only'] is True,
        'Fresh identity, provenance or actual full storage ledger changed.')
    expected = b.read(directory/'expected_state.json')
    actual = life.state_records(model)
    b.require(not result.missing_keys and not result.unexpected_keys and not actual['extra']
        and actual['tensors'] == expected, 'Independent full persistent state did not reload strictly and exactly.')
    b.require(life.non_mlp_proof(model) == report['source_non_mlp'], 'Fresh attention/non-MLP frozen complement changed.')
    groups = export.deployment_contract(model,report['arm'],report['source_binding']['source_format'])
    if report['arm'] == 'dense':
        b.require(meta['checkpoint_kind'] == 'mixed_dense_boundary' and not budget['budget_passed']
            and not budget['final_budget_applicable'] and budget['pending_dense_target_count'] == 108,
            'Dense diagnostic must retain its actual overbudget boundary.')
    else:
        b.require(meta['checkpoint_kind'] == 'mixed_final_model' and budget['budget_passed'], 'D endpoint lost its real compressed budget contract.')
    handles,device_map = life.apply_map(model)
    b.install_precision_runtime(model,torch.bfloat16)
    warm = b.prewarm_deployment(model)
    protection = export.protection_proof(model,report['protection_records'],deployed=True)
    fixture = torch.load(directory/'same_input_output.pt',map_location='cpu',weights_only=True)
    b.require(fixture['schema'] == 'all_MLP_BF16_same_input_v1'
        and fixture['input_ids'].dtype == fixture['attention_mask'].dtype == torch.int64
        and fixture['input_ids'].ndim == 2 and fixture['input_ids'].shape[0] == 1
        and fixture['input_ids'].shape == fixture['attention_mask'].shape
        and fixture['logits'].shape == (1,1,model.config.vocab_size), 'Unexpected complete input/output fixture.')
    device = model.get_input_embeddings().weight.device
    output = model(input_ids=fixture['input_ids'].to(device),attention_mask=fixture['attention_mask'].to(device),
                   use_cache=False,logits_to_keep=1)
    b.require(output.past_key_values is None, 'Same-input diagnostic must not leave a generation cache.')
    logits = output.logits.detach().cpu()
    same_path = comparison(fixture['logits'],logits)
    b.write(directory/'fresh_comparison.json',dict(status='verified' if same_path['exact'] else 'failed',
        deployed_to_fresh=same_path,live_to_deployed=comparison(fixture['live_logits'],fixture['logits']),
        interpretation='Same BF16 deployment/group8/map requires saved-state equivalence; live FP32-master decode is a distinct numerical path.'))
    b.require(same_path['exact'], 'Same-path fresh BF16 logits differ; measured differences are in fresh_comparison.json.')
    after = life.state_records(model)
    b.require(not after['extra'] and after['tensors'] == expected and life.non_mlp_proof(model) == report['source_non_mlp'],
              'Device-map, deployment prewarm or same-input readout changed a persistent field.')
    audit = dict(schema='all_MLP_short2_fresh_audit_v1',status='verified',arm=report['arm'],
        prototype_only=report['prototype_only'],validation_only=True,diagnostic_overbudget=report['arm']=='dense',
        source_checkpoint_id=report['source_binding']['checkpoint_id'],checkpoint_id=meta['checkpoint_id'],
        full_state_exact=True,strict_missing_keys=list(result.missing_keys),strict_unexpected_keys=list(result.unexpected_keys),
        persistent_tensors_compared=len(expected),non_MLP_source_exact=True,source_non_mlp=report['source_non_mlp'],
        protection=protection,groups=groups,all_deployment_parameters_BF16=True,FP32_RoPE_preserved=True,
        same_complete_input_BF16_logits_exact=True,finite=True,deployed_to_fresh=same_path,
        live_to_deployed=comparison(fixture['live_logits'],fixture['logits']),
        numerical_scope='Fresh compares the identical BF16 deployment path; training FP32-master decode versus BF16 export is measured separately without compensation.',
        input_tokens=int(fixture['input_ids'].shape[-1]),source_case_id=fixture['source_case_id'],
        input_signature=fixture['input_signature'],output_sha256=life.tensor_record(logits),
        physical_GPUs=[6,7],layer_device_map=device_map,warmup=warm,bit_budget=budget,
        training_report=b.file_binding(directory/'report.json'),export_meta=b.file_binding(final/b.checkpoint.META_FILE),
        expected_state=b.file_binding(directory/'expected_state.json'),fixture=b.file_binding(directory/'same_input_output.pt'),
        source_sha256=report['source_sha256'],single_effective_state=str(final/meta['state_dict_file']),
        parent_weights_loaded=False,elapsed_seconds=time.monotonic()-started,
        scope='Full36-layer/all108-MLP two-update lifecycle only; not formal quality, a fresh C0 parent or exact-resume proof.')
    b.write(directory/'fresh_audit.json',audit)
    del handles
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args()
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6,7' and torch.cuda.device_count() == 2
        and Path(sys.executable).resolve() == Path('/home/cmcc/miniconda3/envs/bitvae/bin/python').resolve(),
        'Use caller-activated bitvae and the same assigned physical GPU6/7.')
    args.directory = args.directory.resolve(strict=True)
    b.require(not (args.directory/'fresh_audit.json').exists() and not (args.directory/'fresh.exit').exists(),
              'Never overwrite an earlier independent fresh result.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    try:
        result = audit(args.directory)
    except BaseException as error:
        b.write(args.directory/'fresh_failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        (args.directory/'fresh.exit').write_text('1\n',encoding='utf-8')
        raise
    (args.directory/'fresh.exit').write_text('0\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checkpoint_id=result['checkpoint_id'],
        audit=str(args.directory/'fresh_audit.json'),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()

"""Isolate the CPU-snapshot diagnostic repair and reuse the actual saved short."""
import argparse
import ast
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

from . import bindings as b

ORIGINAL_OUTPUT, ORIGINAL_RESULT = b.OUTPUT, b.RESULT
b.OUTPUT = b.base.OUTPUT = ORIGINAL_OUTPUT / 'cpu_snapshot_repair'
b.RESULT = b.base.RESULT = ORIGINAL_RESULT / 'cpu_snapshot_repair'


def prepare():
    from experiments.cat_geometry_cohort_cmcc_20261102.prepare import verify
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not (b.RESULT / 'prepared.json').exists(),
        'Use a new CPU-only diagnostic repair binding.')
    started = time.time()
    old = b.read(ORIGINAL_RESULT / 'prepared.json')
    failed = b.read(ORIGINAL_OUTPUT / 'muon/short/report.json')
    b.require(failed['status'] == 'failed' and failed['exit_code'] == 1
        and 'State comparison must remain on CPU.' in failed['error'] and failed['seed_calls'] == [31]
        and len(failed['actual_groups']) == 1 and failed['actual_groups'][0]['status'] == 'fitted_and_encoded',
        'Repair only the actual post-fit CPU digest contract error.')
    group = failed['actual_groups'][0]
    b.require(group['actual_updates_per_stage'] == 2 and len(group['actual_optimizer_initializations']) == 2
        and len(group['short_update_checks']) == 2
        and all(len(stage['updates']) == 2 and all(row['finite_gradients'] and row['finite_parameters']
            and row['changed_parameter_elements'] > 0 for row in stage['updates']) for stage in group['short_update_checks']),
        'Both original actual stages must already have completed finite real updates.')
    changed = {str(b.HERE / name) for name in ('fit.py','evaluate.py','run.py')}
    actual_changes = {path for path,digest in old['source_sha256'].items() if b.sha(path) != digest}
    b.require(actual_changes == changed, 'Only CPU snapshots and isolated child dispatch may change.')
    for path in changed:
        before = b.RESULT / 'source_before' / Path(path).name
        b.require(b.sha(before) == old['source_sha256'][path], 'The original failing source snapshot differs.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    b.require(b.base.sources() == old['source_binding'], 'The real gate source/control changed.')
    _, _, identity, _ = b.recipe('muon', b.OUTPUT / 'muon/formal/native')
    b.require(json.loads(json.dumps(identity)) == old['arms']['muon']['formal']['runtime_identity'],
        'The diagnostic repair must preserve the complete formal native identity.')
    record = deepcopy(old)
    record.update(source_sha256=b.pins(), CPU_snapshot_repair=dict(original_prepared=str(ORIGINAL_RESULT / 'prepared.json'),
        original_prepared_sha256=b.sha(ORIGINAL_RESULT / 'prepared.json'), original_failed_short=str(ORIGINAL_OUTPUT / 'muon/short/report.json'),
        original_failed_short_sha256=b.sha(ORIGINAL_OUTPUT / 'muon/short/report.json'),
        changed_old_sources={path:dict(before=old['source_sha256'][path],after=b.sha(path)) for path in sorted(changed)},
        correction='Explicit detach().cpu() snapshots before the existing exact digest; internal child dispatch to this isolated result binding.',
        algorithm_or_data_changed=False, actual_short_training_repeated=False, CUDA_initialized=b.torch.cuda.is_initialized()))
    b.RESULT.mkdir(parents=True,exist_ok=True)
    (b.OUTPUT / 'muon/short').mkdir(parents=True)
    b.write(b.RESULT / 'prepared.json',record)
    # Keep original successful control results as their one authoritative copy.
    (b.RESULT / 'control').symlink_to(ORIGINAL_RESULT / 'control',target_is_directory=True)
    metas = list((ORIGINAL_OUTPUT / 'muon/short/native').rglob('checkpoint_meta.json'))
    b.require(len(metas) == 1, 'Require one actual saved short round_base, without re-fitting.')
    directory, meta = metas[0].parent, b.read(metas[0])
    b.base.validate_raw2(meta,'muon')
    state = b.torch.load(directory / meta['state_dict_file'],map_location='cpu',weights_only=True,mmap=True)
    digest = b.base.fork.state_digest(state)
    b.require(b.base.fork.state_digest({name:value.detach().cpu() for name,value in state.items()}) == digest,
        'Explicit CPU snapshots must preserve every saved dtype/value/field.')
    stages = [dict(stage_index=index,updates=[]) for index in range(2)]
    for line in (ORIGINAL_OUTPUT / 'muon/short_fit.log').read_text(encoding='utf-8').splitlines():
        if line.startswith("{'event': 'VAE_Muon_"):
            event = ast.literal_eval(line)
            stage = stages[event['stage']]
            if event['event'] == 'VAE_Muon_stage_calibration':
                stage['calibration'] = event
            elif event['event'] == 'VAE_Muon_real_update':
                stage['updates'].append(event)
    b.require(all(stage['calibration']['relative_error'] <= .01 and len(stage['updates']) == 2 for stage in stages),
        'Retain the original actual two kappa calibrations and all four update telemetry records.')
    recovered = deepcopy(failed)
    recovered.update(status='raw2_completed',exit_code=0,state=digest,raw2=b.base.event(directory),
        contract=b.CONTRACT,optimizer_stages=stages,
        recovered_short_publication=dict(original_failure=str(ORIGINAL_OUTPUT / 'muon/short/report.json'),
            original_failure_sha256=b.sha(ORIGINAL_OUTPUT / 'muon/short/report.json'),
            actual_saved_endpoint_reused=True,actual_training_repeated=False,
            original_live_logits_lost_before_evidence_save=True,
            new_logit_reference='First independent strict consumer; second independent consumer must match. Not the lost original live output.',
            historical_init_boundary='Original actual grouped optimizer initial hashes retained; canonical flat telemetry was lost on exception. Formal records genuine flat hashes.'))
    b.write(b.OUTPUT / 'muon/short/report.json',recovered)
    del state
    verify(directory,digest,b.OUTPUT / 'muon/short/strict_reload.json','muon')
    gc.collect()
    state = b.torch.load(directory / meta['state_dict_file'],map_location='cpu',weights_only=True,mmap=True)
    control = Path(old['source_binding']['control']['path'])
    control_meta = b.read(control / 'checkpoint_meta.json')
    reference = b.torch.load(control / control_meta['state_dict_file'],map_location='cpu',weights_only=True,mmap=True)
    prefixes = tuple(f'model.layers.{i}.mlp.gate_proj.' for i in range(36))
    protected = {name:value for name,value in state.items() if name.startswith(prefixes)
        and ('protected_' in name or '.weight_rotation.' in name)}
    other = {name:value for name,value in reference.items() if name.startswith(prefixes)
        and ('protected_' in name or '.weight_rotation.' in name)}
    b.require(protected.keys() == other.keys() and len(protected) >= 180, 'Require every original gate q/scale/indices/rotation sign.')
    for name,value in protected.items():
        expected = other[name]
        b.require(value.dtype == expected.dtype and value.shape == expected.shape and b.torch.equal(value,expected),
            'Original gate protection/rotation changed: '+name)
    b.write(b.RESULT / 'CPU_repair_verification.json',dict(status='verified_CPU_snapshot_source_and_saved_short',exit_code=0,
        checkpoint_id=meta['checkpoint_id'],original_training_repeated=False,protected_rotation_fields_exact=len(protected),
        protected_rotation_digest=b.base.fork.state_digest(protected),persistent_fields=digest['tensors'],
        independent_strict_reload=str(b.OUTPUT / 'muon/short/strict_reload.json'),
        source_binding_unchanged=True,formal_identity_unchanged=True,CUDA_initialized=b.torch.cuda.is_initialized(),
        elapsed_seconds=time.time()-started,source_sha256=record['source_sha256']))
    print(dict(status='verified_CPU_snapshot_source_and_saved_short',seconds=time.time()-started,
        protected_rotation_fields=len(protected)),flush=True)


def probe():
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    from .evaluate import warm,deployed_logits
    b.prepared()
    root = b.OUTPUT / 'muon/short'
    report = b.read(root / 'report.json')
    directory = Path(report['raw2']['path'])
    model,meta,loaded = load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys
        and b.base.fork.state_digest(model.state_dict()) == report['state'], 'First independent consumer changed saved fields.')
    warmup = warm(model,180)
    after = {name:value.detach().cpu() for name,value in model.state_dict().items()}
    b.require(b.base.fork.state_digest(after) == report['state'], 'Actual GPU prewarm changed a persistent saved value.')
    runtime = b.torch.load(Path(b.base.sources()['after_o']['path']) / 'cat_runtime_state.pt',map_location='cpu',weights_only=True)
    tokens = runtime['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone()
    logits = deployed_logits(model,tokens)
    b.torch.save(dict(checkpoint_id=meta['checkpoint_id'],full_state=report['state'],input_ids=tokens,
        logits=logits,warmup=warmup,reference_source='First independent strict consumer after the original live expectation was lost.'),
        root / 'same_input_deployment.pt')
    b.write(root / 'prewarm_reference.json',dict(status='independent_short_consumer_prewarm_persistent_exact',exit_code=0,
        checkpoint_id=meta['checkpoint_id'],pid=os.getpid(),persistent_fields=2559,
        all_persistent_values_before_after_GPU_warm_exact=True,warmup=warmup,
        BF16_logits_reference_is_original_live=False,peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=b.torch.cuda.max_memory_reserved()))


def publish():
    b.prepared()
    root = b.OUTPUT / 'muon'
    cpu = b.read(b.RESULT / 'CPU_repair_verification.json')
    strict = b.read(root / 'short/strict_reload.json')
    first = b.read(root / 'short/prewarm_reference.json')
    second = b.read(root / 'short/deployment_fresh.json')
    report = b.read(root / 'short/report.json')
    b.require(cpu['exit_code'] == strict['exit_code'] == first['exit_code'] == second['exit_code'] == 0
        and strict['non_gate_after_o_complement_exact'] and strict['non_gate_tensor_count'] == 2091
        and second['same_input_logits_exact'] and second['persistent_fields'] == 2559
        and first['all_persistent_values_before_after_GPU_warm_exact'] and cpu['protected_rotation_fields_exact'] >= 180,
        'Require the complete repaired saved-short contract, without re-fitting.')
    b.write(root / 'short_run.json',dict(status='completed_functional_short',exit_code=0,
        original_failed_run=str(ORIGINAL_OUTPUT / 'muon/short_run.json'),actual_training_repeated=False,
        checkpoint_id=report['raw2']['checkpoint_id'],CPU_repair_verification=str(b.RESULT / 'CPU_repair_verification.json'),
        fresh_reference=str(root / 'short/prewarm_reference.json'),fresh_consumer=str(root / 'short/deployment_fresh.json'),
        original_live_logits_lost=True,new_independent_saved_consumers_logits_exact=True,
        source_sha256=b.read(b.RESULT / 'prepared.json')['source_sha256'],published_unix=time.time()))
    print(dict(status='completed_functional_short',actual_training_repeated=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True,choices=('prepare','probe','publish','run','fit','evaluate'))
    args,remaining=parser.parse_known_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.target in ('prepare','probe','publish'):
        b.require(not remaining,'Unexpected repair arguments.')
        globals()[args.target]()
        return
    if args.target == 'run':
        from . import run
        run.command=lambda component,*items:[str(b.PYTHON),'-B','-u','-m',__package__+'.repair',
            '--target',component,*map(str,items)]
        p=argparse.ArgumentParser()
        p.add_argument('--phase',required=True,choices=('formal',))
        run.execute(p.parse_args(remaining).phase)
    elif args.target == 'fit':
        from .fit import execute
        p=argparse.ArgumentParser()
        p.add_argument('--phase',required=True,choices=('formal',))
        execute(p.parse_args(remaining).phase)
    else:
        import runpy
        sys.argv=[sys.argv[0],*remaining]
        runpy.run_module(__package__+'.evaluate',run_name='__main__')


if __name__ == '__main__':
    main()

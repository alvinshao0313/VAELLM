"""One explicit BF16 bank union, bounded native export/fresh and existing full8."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha, public_consumer_identity
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, inventory_id, require, validate_inventory, validate_run_identity
from experiments.cat_offline_execution_cmcc_20261104 import evaluate
from experiments.cat_offline_execution_cmcc_20261104.protocol import binding, code_identity, load_baseline, read, verify_files, write
from experiments.cat_profile_coverage_cmcc_20261104 import Snapshot, verify_complete

SCHEMA = 'cat_explicit_joint_export_v1'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def pins():
    paths = list(HERE.glob('*.py')) + [REPO/'experiments/cat_profile_coverage_cmcc_20261104.py',
        REPO/'experiments/cat_night_profiles_cmcc_20261104/run.py',
        REPO/'experiments/cat_up_rate_profiles_cmcc_20261104/run.py']
    return {**code_identity(), **{str(path.resolve()):file_sha(path) for path in paths}}


def limits(value, n):
    require(set(value) == {'export_seconds','full8_seconds','total_seconds','max_logical_bits'}, 'Declare export/full8/total seconds and an explicit whole-working-set logical limit.')
    require(all(isinstance(item,int) and not isinstance(item,bool) and item>0 for item in value.values()), 'Limits must be positive integer seconds/bits.')
    require(value['export_seconds'] + value['full8_seconds'] <= value['total_seconds']
        and value['max_logical_bits'] <= math.floor(Fraction(13,5)*n), 'Stage limits exceed total or logical2.6.')
    return deepcopy(value)


def complete_union(inventory, banks):
    names = {row['actual_name'] for row in inventory['projections']}
    require(set(banks) == set(inventory['category_ids']), 'Name every actual inventory category exactly once.')
    owned = set()
    for category, bank in banks.items():
        expected = {row['actual_name'] for row in inventory['projections'] if row['category_id']==category}
        actual = set(bank['recipe']['group_signature'])
        require(bank['recipe']['category_id']==category and bank['recipe']['S']==2 and actual==expected
            and actual=={row['name'] for row in bank['converted_modules']} and not owned & actual,
            'Explicit banks omit, overlap or change actual S2 category projections.')
        owned.update(actual)
    require(owned==names, 'The explicit bank union is not the full actual inventory.')
    return names


def prepare(spec, *, check_output=True):
    """Small-source verification only; never hash tensor/sample payloads or authorize."""
    snapshot = Snapshot()
    inventory_ref, baseline_ref = spec['inventory_receipt'], spec['baseline_receipt']
    inventory = validate_inventory(snapshot.bound(inventory_ref)['inventory'])
    baseline = snapshot.bound(baseline_ref); identity = validate_run_identity(spec['run_identity'])
    require(identity['model']==inventory['model_identity']==baseline['model_identity']
        and identity['parent_weights_digest']==digest(identity['model']['weight_shards_sha256'])
        and baseline['source_evidence']['inventory_receipt']==inventory_ref
        and baseline['protocol_identity']==digest(baseline['protocol']) and baseline['quality_reference_verified'], 'Require the actual common original BF16/inventory/reference context.')
    snapshot.code(identity['code_identity']['source_sha256'])
    require(identity['consumer_identity']==public_consumer_identity(), 'The actual PUBLIC consumer differs from profile sources.')
    require(set(spec['profiles'])==set(inventory['category_ids']), 'Freeze one explicit completed profile per actual category.')
    context = dict(identity=identity,inventory=inventory,baseline=baseline,baseline_ref=baseline_ref,
        protocol_identity=baseline['protocol_identity'])
    banks, metadata, fees, selection, inputs = {}, {}, {}, set(), set()
    for category, refs in sorted(spec['profiles'].items()):
        canonical = snapshot.bound(refs['canonical']); summary = snapshot.bound(refs['full8']['summary'])
        evaluation = snapshot.bound(refs['full8']['evaluation'])
        require(refs['checkpoint_id']==canonical['endpoint']['checkpoint_id'], 'Explicit profile source UUID differs from its actual canonical endpoint.')
        full8 = Path(refs['full8']['summary']['path']).resolve().parent
        require(full8/'full8_summary.json'==Path(refs['full8']['summary']['path']).resolve()
            and full8/'evaluation.json'==Path(refs['full8']['evaluation']['path']).resolve(), 'Use the actual complete full8 artifact directory.')
        request_ref = canonical['artifact_evidence']['request']; completed_ref = canonical['artifact_evidence']['completed']
        request = snapshot.bound(request_ref); snapshot.bound(completed_ref)
        formal = Path(completed_ref['path']).resolve().parent; path = Path(refs['canonical']['path']).resolve()
        require(path in (formal/'profile_full8_result.json',full8/'profile_full8_result.json'), 'Canonical attachment must be in its actual producer directory.')
        row = dict(category_id=category,candidate_id=canonical['candidate_id'],recipe=canonical['recipe'],
            request=request_ref['path'],formal_dir=str(formal),full8_dir=str(full8),
            origin='up' if path.parent==full8 else 'night',gpu=request['physical_GPU'],source_declared_refs={})
        verified = verify_complete(snapshot,row,context)
        require(canonical['purpose']=='profile' and canonical['recipe']['category_id']==category
            and request['parent']['kind']=='BF16' and Path(request['parent']['path']).resolve()==Path(inventory['source_evidence']['path']).resolve(), 'Only complete own-original-BF16 profile banks may be combined.')
        banks[category] = dict(candidate_id=canonical['candidate_id'],recipe=canonical['recipe'],bank_digest=verified['bank_digest'],
            worker=dict(completed=str(Path(completed_ref['path']).resolve()),request=deepcopy(request_ref)))
        metadata[category] = canonical['bank']
        fees[category] = {key:value for key,value in verified['actual_class_fee'].items() if key not in ('logical_bpw','packed_bpw')}
        selection.add(digest(verified['selection_data_identity'])); inputs.add(verified['full8']['ordered_candidate_input_identity'])
        require(summary['endpoint']==evaluation.get('endpoint',summary['endpoint']) and summary['checkpoint_id']==canonical['endpoint']['checkpoint_id'], 'Profile full8 UUID differs.')
    complete_union(inventory,metadata)
    require(len(selection)==len(inputs)==1 and len({row['candidate_id'] for row in banks.values()})==len(banks), 'Banks use different512/full8 inputs or repeat candidates.')
    requested = limits(spec['limits'],inventory['total_target_numel'])
    require(sum(row['logical_bits'] for row in fees.values())<=requested['max_logical_bits'], 'Actual source class-fee sum exceeds the explicitly requested working-set limit.')
    cards = spec['allowed_physical_GPUs']
    require(isinstance(cards,list) and cards and all(isinstance(card,int) and not isinstance(card,bool) and card>=0 for card in cards)
        and len(cards)==len(set(cards)), 'Declare distinct nonnegative physical GPU IDs; execution must belong to the actual authorization and pass resource checks.')
    vector = [[category,str(Fraction(row['recipe']['nominal_main_rate']))] for category,row in sorted(banks.items())]
    require(spec['rate_vector']==vector, 'The explicit declared rate vector differs from its actual bank recipes.')
    proposal_id = digest(dict(rates=vector,banks={category:row['bank_digest'] for category,row in banks.items()}))
    value = dict(schema=SCHEMA,status='prepared_requires_actual_authorization',proposal_id=proposal_id,rate_vector=vector,
        inventory_receipt=deepcopy(inventory_ref),baseline_receipt=deepcopy(baseline_ref),run_identity=identity,
        profiles=deepcopy(spec['profiles']),banks=banks,source_class_fees=fees,
        protocol_identity=baseline['protocol_identity'],selection_data_identity_digest=next(iter(selection)),
        output_root=str(Path(spec['output_root']).resolve()),allowed_physical_GPUs=cards,limits=requested,
        source_sha256=pins(),small_sources=list(snapshot.bindings.values()),formal_eligible=False,
        payload_bytes_rehashed=False,sample_bytes_reverified=False,
        scope='Explicit bank union preparation only; no nomination, allocation, authorization, model or quality result.')
    require(not check_output or not Path(value['output_root']).exists(), 'Use an independent nonexistent actual execution root.')
    return dict(value,preparation_identity=digest(value))


def verify_preparation(proposal):
    require(proposal['schema']==SCHEMA and proposal['preparation_identity']==digest({key:value for key,value in proposal.items() if key!='preparation_identity'})
        and proposal['status']=='prepared_requires_actual_authorization' and proposal['formal_eligible'] is False, 'Prepared content/version/eligibility changed.')
    verify_files(proposal['source_sha256'])
    snapshot = Snapshot()
    for reference in proposal['small_sources']:
        require(snapshot.ref(reference['path'])==reference, 'Prepared small JSON/code source changed.')
    spec={key:proposal[key] for key in ('inventory_receipt','baseline_receipt','run_identity','profiles','output_root','allowed_physical_GPUs','limits','rate_vector')}
    require(prepare(spec,check_output=False)==proposal, 'Prepared source bindings/recipes/union/fees no longer match actual completed profiles.')
    return proposal


def authorization(proposal, proposal_path, authorization_path, *, now=None):
    """No runtime model I/O or directory claim can precede this actual approval."""
    now = time.time() if now is None else now
    require(read(proposal_path)==proposal, 'Use the actual prepared proposal file.')
    auth = read(authorization_path)
    return authorization_contract(proposal,binding(proposal_path),auth,now=now)


def authorization_contract(proposal, proposal_reference, auth, *, now):
    """Pure binding/window rules; CPU examples never create approved files."""
    require(auth.get('user_approved') is True and isinstance(auth.get('authorization_reference'),str)
        and bool(auth['authorization_reference'].strip()), 'Actual separate user authorization is required; preparation is unapproved.')
    export_phase(auth.get('execution_scope'))
    require(auth['proposal_sha256']==proposal_reference['sha256'] and auth['preparation_identity']==proposal['preparation_identity']
        and auth['proposal_id']==proposal['proposal_id'] and auth['output_root']==proposal['output_root']
        and auth['allowed_physical_GPUs']==proposal['allowed_physical_GPUs'] and auth['limits']==proposal['limits'], 'Authorization differs from the actual proposal SHA/root/working set/resource/budget.')
    gpu=auth['physical_GPU']
    require(isinstance(gpu,int) and not isinstance(gpu,bool) and gpu>=0 and gpu in proposal['allowed_physical_GPUs'], 'The actual assigned GPU is outside this explicit preparation/authorization resource set.')
    require(all(isinstance(auth[key],(float,int)) and not isinstance(auth[key],bool) and math.isfinite(auth[key])
        for key in ('authorized_GPU_hours','authorized_start_epoch','absolute_deadline_epoch')), 'Authorization budget/times must be finite numbers.')
    require(auth['authorized_GPU_hours']>0 and auth['authorized_start_epoch']<=now<auth['absolute_deadline_epoch']
        and auth['absolute_deadline_epoch']-auth['authorized_start_epoch']<=min(proposal['limits']['total_seconds'],auth['authorized_GPU_hours']*3600), 'Require one live finite actual authorization window including all source/export/fresh/evaluation costs.')
    return auth


def execution_request(proposal, proposal_path, authorization_path, auth, now):
    deadline=min(auth['absolute_deadline_epoch'],now+proposal['limits']['total_seconds'])
    return dict(schema=SCHEMA,proposal=binding(proposal_path),authorization=binding(authorization_path),
        execution_scope=auth['execution_scope'],output_root=proposal['output_root'],started_epoch=now,absolute_deadline_epoch=deadline,
        export_deadline_epoch=min(deadline,now+proposal['limits']['export_seconds']))


def export_phase(scope):
    require(scope in ('export_fresh_ledger_only','export_and_full8'), 'Authorization must explicitly distinguish technical export-only from scientific full8 execution.')
    return ('technical_validation',False) if scope=='export_fresh_ledger_only' else ('formal',True)


def install_export(model, *, inventory, identity, banks, output, phase, base_model_path=None, tokenizer=None):
    """The same real native installation/export path is used by the tiny CPU test."""
    from experiments.cat_auto_protocol_cmcc_20261104.fit_adapter import install_category_bank
    from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
    from train_utils import checkpoint_v6 as v6
    require(not Path(output).exists(), 'Never overwrite or retry an existing native joint export.')
    require(phase in ('formal','technical_validation','cpu_fixture'), 'Explicit runtime technical/formal or CPU mechanism phase required.')
    require(phase=='cpu_fixture' or (identity['model'].get('schema')=='dense_model_content_identity_v0'
        and all(bank['formal_eligible'] for bank in banks.values())), 'CPU/technical banks cannot acquire formal export eligibility.')
    names=complete_union(inventory,banks)
    outside=lambda state:{key:value for key,value in state.items() if not any(key.startswith(name+'.') for name in names)}
    original=tensor_digest(outside(model.state_dict()))
    for category in sorted(banks): install_category_bank(model,banks[category],inventory=inventory,expected_parent_identity=identity)
    specs={row['name']:row for row in v6._collect_vae_linear_specs(model)}
    require(specs=={row['name']:row for bank in banks.values() for row in bank['converted_modules']}
        and set(specs)==names, 'Installed native topology is not exactly the explicit complete bank union.')
    state=model.state_dict()
    for bank in banks.values():
        group=bank['recipe']['group_signature']
        require(tensor_digest({key:value for key,value in state.items() if any(key.startswith(name+'.') for name in group)})==bank['tensor_digest'], 'Installed actual bank tensor fields differ.')
    require(tensor_digest(outside(state))==original, 'Joint installation changed original BF16 norm/embed/head or another non-target field.')
    saved=v6.save_v6_full_checkpoint(model,str(output),checkpoint_kind='round_base',compressed_targets=sorted(names),
        pending_dense_targets=[],skip_targets=[],completed_categories=[],compression_categories=inventory['category_ids'],
        target_layers=sorted({row['layer_id'] for row in inventory['projections']}),target_modules=inventory['category_ids'],
        base_model_path=base_model_path,tokenizer=tokenizer,extra_meta=dict(cat_auto_phase=phase,purpose='explicit_joint_export',
            own_parent=identity,bank_digests={category:bank['tensor_digest'] for category,bank in banks.items()},
            formal_search_boundary=False,CAT_recovery_completed=False))
    return saved,dict(actual_complete_bank_union_verified=True,non_target_before_digest=original,non_target_after_digest=tensor_digest(outside(state)),
        inventory_identity=inventory_id(inventory),actual_target_count=len(names),bank_digests={category:bank['tensor_digest'] for category,bank in banks.items()})


def check_ledger(ledger, source_fees, requested, *, formal):
    require(ledger['future_dense_bits']==ledger['future_dense_count']==0 and ledger['final_budget_applicable']
        and set(ledger['per_category'])==set(source_fees) and all(ledger['per_category'][category]==fee for category,fee in source_fees.items()), 'Actual full joint fees do not close to every own bank fee.')
    require(sum(row['logical_bits'] for row in ledger['per_category'].values())==ledger['logical_bits']
        and sum(row['packed_bits'] for row in ledger['per_category'].values())==ledger['packed_bits'], 'Actual category ledger does not close to whole joint.')
    if formal:
        require(ledger['final_budget_passed'] and ledger['logical_bits']<=requested['max_logical_bits'], 'Actual joint exceeds logical2.6 or the explicit whole-working-set limit; no category cap is inferred.')


def export_worker(request_path):
    request=read(request_path); proposal=read(request['proposal']['path'])
    auth=authorization(proposal,request['proposal']['path'],request['authorization']['path'])
    require(request['execution_scope']==auth['execution_scope'], 'Worker execution scope differs from the actual authorization.')
    phase,eligible=export_phase(request['execution_scope'])
    require(binding(request['proposal']['path'])==request['proposal'] and binding(request['authorization']['path'])==request['authorization'], 'Actual execution source binding changed.')
    require(request['schema']==SCHEMA and request['output_root']==proposal['output_root']
        and Path(request_path).resolve()==Path(proposal['output_root'])/'execution_request.json'
        and auth['authorized_start_epoch']<=request['started_epoch']<=time.time()<request['export_deadline_epoch']
        and request['export_deadline_epoch']<=min(request['absolute_deadline_epoch'],request['started_epoch']+proposal['limits']['export_seconds'])
        and request['absolute_deadline_epoch']<=auth['absolute_deadline_epoch'], 'Export worker cannot extend the original complete authorization window.')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')==str(auth['physical_GPU']), 'Worker must inherit this actual assigned GPU and caller bitvae.')
    verify_preparation(proposal)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.cat_offline_execution_cmcc_20261104.results import worker_result
    from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
    from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
    from tools.cat_eval import _prepare_model_for_eval
    import logging
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'Runtime joint validation requires exactly one actually assigned visible GPU.')
    torch.cuda.reset_peak_memory_stats()
    inventory=read(proposal['inventory_receipt']['path'])['inventory'];identity=proposal['run_identity']
    source=Path(inventory['source_evidence']['path']).resolve()
    # This is the first original-BF16 byte audit in this worker, after authorization.
    verify_files({str(source/name):record['sha256'] for name,record in inventory['source_evidence']['files'].items()})
    print(dict(event='original_BF16_bytes_verified',execution_scope=request['execution_scope']),flush=True)
    actual={category:worker_result(row['worker']['completed'],row['candidate_id'],request_reference=row['worker']['request']) for category,row in proposal['banks'].items()}
    require(all(row['purpose']=='profile' and row['phase']=='formal' and row['formal_eligible']
        and row['run_identity']==identity and row['recipe']==proposal['banks'][category]['recipe']
        and row['bank_digest']==proposal['banks'][category]['bank_digest']
        and row['ledger']['per_category'][category]==proposal['source_class_fees'][category] for category,row in actual.items())
        and {digest(row['selection_data_identity']) for row in actual.values()}=={proposal['selection_data_identity_digest']}, 'Runtime actual formal bank/identity/512/fee differs from its prepared source.')
    print(dict(event='actual_formal_bank_sources_verified',categories=sorted(actual)),flush=True)
    model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,use_fast=True,trust_remote_code=False)
    raw=Path(proposal['output_root'])/'raw_model'
    saved,union=install_export(model,inventory=inventory,identity=identity,banks={category:row['bank'] for category,row in actual.items()},
        output=raw,phase=phase,base_model_path=str(source),tokenizer=tokenizer)
    print(dict(event='actual_complete_bank_union_exported',checkpoint_id=saved['checkpoint_id'],**union),flush=True)
    tokens=tokenizer('Verify the saved compressed model.',return_tensors='pt')['input_ids'][:,:16].contiguous()
    def forward(value):
        value.eval();value.config.use_cache=False
        precision=precision_audit(value,attention='sdpa')
        audit=_prepare_model_for_eval(value,'cuda:0',8,logging.getLogger(__name__),{})
        require(audit['total']==audit['warmed']==len(inventory['projections']) and not audit['failed'], 'Every actual joint projection must complete native group8 warmup.')
        audit['precision']=precision
        with torch.no_grad(): result=value(input_ids=tokens.to('cuda:0'),use_cache=False).logits.detach().cpu()
        return result,audit
    expected,live_warmup=forward(model)
    del model;torch.cuda.empty_cache()
    fresh,fresh_evidence=fresh_native(raw,model_factory=lambda directory:architecture(directory,attn_implementation='sdpa'))
    observed,fresh_warmup=forward(fresh)
    numerical=compare_logits(expected,observed)
    del fresh;torch.cuda.empty_cache()
    ledger=measure_checkpoint(raw,inventory);check_ledger(ledger,proposal['source_class_fees'],proposal['limits'],formal=True)
    meta=read(raw/'checkpoint_meta.json')
    require(meta['checkpoint_id']==saved['checkpoint_id']==fresh_evidence['checkpoint_id'] and not meta['pending_dense_targets']
        and not meta['skip_targets'] and meta['completed_categories']==[], 'Joint export UUID/inventory or undeclared recovery completion differs.')
    require(time.time()<request['export_deadline_epoch'], 'Export/fresh/ledger exhausted its original finite stage; no evaluation follows.')
    endpoint=dict(kind='native_v6',path=str(raw),checkpoint_id=meta['checkpoint_id'],metadata_sha256=ledger['source']['metadata_sha256'],
        state_sha256=ledger['source']['state_file_sha256'],model_identity=identity['model'],ledger=ledger,formal_eligible=eligible,phase=phase)
    require(endpoint['state_sha256']==fresh_evidence['weight_digest'], 'Fresh complete-state bytes differ from the measured endpoint.')
    fresh_evidence.update(forward_executed=True,gpu_validation=True,CPU_mechanism_only=False,input_identity=digest(tokens.tolist()),
        same_input_same_BF16_path=numerical,live_warmup=live_warmup,fresh_warmup=fresh_warmup)
    result=dict(schema=SCHEMA,status='exported_actual_joint_awaiting_full8' if eligible else 'completed_technical_joint_export_only',
        execution_scope=request['execution_scope'],proposal_id=proposal['proposal_id'],rate_vector=proposal['rate_vector'],
        banks=proposal['banks'],run_identity=identity,endpoint=endpoint,phase=phase,formal_eligible=eligible,
        union_evidence=union,fresh_evidence=fresh_evidence,source_proposal=request['proposal'],source_authorization=request['authorization'],
        source_execution=binding(request_path),source_sha256=proposal['source_sha256'],complete_full8_evaluated=False,
        actual_bytes=dict(full_model_tensor_bytes=ledger['tensor_bytes'],serialized_state_bytes=(raw/meta['state_dict_file']).stat().st_size,
            model_directory_bytes=sum(path.stat().st_size for path in raw.rglob('*') if path.is_file())),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        quality_status='awaiting_actual_whole_joint_full8' if eligible else 'not_evaluated_technical_only',CAT_recovery_completed=False)
    write(Path(proposal['output_root'])/'joint_result.json',result)
    return result


def compare_logits(expected, observed):
    import torch
    require(expected.dtype==observed.dtype==torch.bfloat16 and expected.shape==observed.shape
        and torch.isfinite(expected).all().item() and torch.isfinite(observed).all().item(), 'Same-path saved/fresh BF16 logits must have matching finite shape/dtype.')
    # Standard torch.testing BF16 tolerances; state/packed contracts above remain exact.
    torch.testing.assert_close(observed,expected,rtol=1.6e-2,atol=1e-5)
    delta=observed.float()-expected.float()
    return dict(max_absolute_error=float(delta.abs().max()),relative_L2=float(delta.norm()/expected.float().norm().clamp_min(1e-30)),
        exact=torch.equal(expected,observed),rtol=1.6e-2,atol=1e-5,passed=True)


def precision_audit(model, *, attention):
    import torch
    rope={name:dict(dtype=str(buffer.dtype),shape=list(buffer.shape)) for name,buffer in model.named_buffers() if 'inv_freq' in name}
    dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()})
    require(model.config._attn_implementation==attention and dtypes==['torch.bfloat16']
        and rope and all(row['dtype']=='torch.float32' for row in rope.values()), 'Live/fresh must retain the same BF16 parameter, attention and FP32RoPE path.')
    return dict(parameter_dtypes=dtypes,rope_buffers=rope,attn_implementation=attention)


def evaluation_plan(result, proposal, *, output, authorization_value):
    require(result['schema']==SCHEMA and result['proposal_id']==proposal['proposal_id'] and result['banks']==proposal['banks']
        and result['run_identity']==proposal['run_identity'] and result['formal_eligible'] and result['phase']=='formal'
        and result['union_evidence']['actual_complete_bank_union_verified'] and result['fresh_evidence']['state_exact']
        and result['fresh_evidence']['gpu_validation'] and result['fresh_evidence']['forward_executed']
        and result['fresh_evidence']['CPU_mechanism_only'] is False and result['fresh_evidence']['same_input_same_BF16_path']['passed'], 'Only this actual full-union GPU/fresh export may enter formal full8; CPU or partial endpoints cannot.')
    plan=evaluate.make_plan(result=result,baseline=load_baseline(proposal['baseline_receipt']['path']),
        baseline_path=proposal['baseline_receipt']['path'],output=output,authorization=authorization_value)
    plan['source_sha256'].update(proposal['source_sha256'])
    plan['joint_source_proposal_id']=proposal['proposal_id']
    return plan


def run(proposal_path, authorization_path):
    proposal_path=Path(proposal_path).resolve();authorization_path=Path(authorization_path).resolve();proposal=read(proposal_path)
    auth=authorization(proposal,proposal_path,authorization_path)
    require(auth['execution_scope']=='export_and_full8', 'The scientific full8 entry cannot consume an export-only technical authorization.')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'The finite controller requires caller bitvae with CUDA hidden.')
    verify_preparation(proposal)
    from experiments.cmcc_runtime_20261030.sequence import check_resources
    from experiments.cat_night_profiles_cmcc_20261104.run import wait_child
    check_resources([str(auth['physical_GPU'])])
    root=Path(proposal['output_root']);root.parent.mkdir(parents=True,exist_ok=True)
    root.mkdir()  # Exclusive one-shot claim, including failed runs.
    request=execution_request(proposal,proposal_path,authorization_path,auth,time.time());write(root/'execution_request.json',request)
    state=dict(schema=SCHEMA,status='running',exit_code=None,proposal=binding(proposal_path),authorization=binding(authorization_path),
        started_epoch=request['started_epoch'],absolute_deadline_epoch=request['absolute_deadline_epoch'],stages=[])
    try:
        environment=evaluate.child_environment(auth['physical_GPU'])
        command=[sys.executable,'-B','-u','-m',__package__+'.producer','--component','export-worker','--request',str(root/'execution_request.json')]
        state['stages'].append(dict(name='export_fresh_ledger',command=command,absolute_deadline_epoch=request['export_deadline_epoch']))
        write(root/'run_started.json',state)
        code=wait_child(command,environment,root,request['export_deadline_epoch'])
        require(code==0, 'Explicit joint export/fresh worker failed; stop without retry or full8.')
        result=read(root/'joint_result.json');now=time.time();deadline=min(request['absolute_deadline_epoch'],now+proposal['limits']['full8_seconds'])
        require(deadline>now, 'Original shared budget is exhausted; no full8 or extension.')
        eval_auth=dict(authorization_reference=auth['authorization_reference'],physical_GPU=auth['physical_GPU'],
            authorized_GPU_hours=(deadline-now)/3600,authorized_start_epoch=now,absolute_deadline_epoch=deadline)
        plan=evaluation_plan(result,proposal,output=root/'full8',authorization_value=eval_auth)
        write(root/'full8/plan.json',plan);state['stages'].append(dict(name='full8',plan=binding(root/'full8/plan.json'),absolute_deadline_epoch=deadline))
        evaluate.run(root/'full8/plan.json')
        summary=read(root/'full8/full8_summary.json')
        frozen_proposal=dict(proposal_id=proposal['proposal_id'],rate_vector=proposal['rate_vector'],banks=proposal['banks'],
            full8=dict(summary=str(root/'full8/full8_summary.json'),evaluation=str(root/'full8/evaluation.json')))
        write(root/'joint_full8_result.json',dict(result,status='completed_explicit_joint_full8',complete_full8_evaluated=True,quality_status='actual_whole_joint_full8_complete',
            full8_receipt=binding(root/'full8/full8_summary.json'),mean_percent=summary['mean_percent'],selection_proposal=frozen_proposal))
        state.update(status='completed_explicit_joint_full8',exit_code=0,checkpoint_id=result['endpoint']['checkpoint_id'])
    except BaseException:
        state.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        finished=time.time()
        state.update(finished_epoch=finished,elapsed_seconds=finished-state['started_epoch'],
            internal_window_seconds=finished-state['started_epoch'],authorization_started_epoch=auth['authorized_start_epoch'],
            conservative_authorization_window_seconds=finished-auth['authorized_start_epoch'],
            conservative_authorization_GPU_hours=(finished-auth['authorized_start_epoch'])/3600,
            cost_scope='Single assigned GPU conservative authorization-start-to-finish window; internal/evaluation windows are included subwindows, never additive.')
        write(root/'run.json',state)
    return state


def validate_gpu(proposal_path, authorization_path):
    """One export-only technical worker; no evaluation or scientific receipt."""
    proposal_path=Path(proposal_path).resolve();authorization_path=Path(authorization_path).resolve();proposal=read(proposal_path)
    auth=authorization(proposal,proposal_path,authorization_path)
    require(auth['execution_scope']=='export_fresh_ledger_only', 'This finite technical entry is export/fresh/ledger-only.')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'The technical controller requires caller bitvae and hidden CUDA.')
    verify_preparation(proposal)
    from experiments.cmcc_runtime_20261030.sequence import check_resources
    from experiments.cat_night_profiles_cmcc_20261104.run import wait_child
    resource=check_resources([str(auth['physical_GPU'])]);now=time.time()
    reserve=auth['cleanup_reserve_seconds']
    require(isinstance(reserve,int) and reserve>0 and now+reserve<auth['absolute_deadline_epoch'], 'Reserve real time for owned technical payload cleanup inside the same authorization window.')
    root=Path(proposal['output_root']);root.parent.mkdir(parents=True,exist_ok=True);root.mkdir()
    request=execution_request(proposal,proposal_path,authorization_path,auth,now)
    request['export_deadline_epoch']=min(request['export_deadline_epoch'],auth['absolute_deadline_epoch']-reserve)
    write(root/'execution_request.json',request)
    state=dict(schema=SCHEMA,status='running_technical_export_only',exit_code=None,execution_scope=auth['execution_scope'],formal_eligible=False,
        quality_status='not_evaluated_technical_only',proposal=binding(proposal_path),authorization=binding(authorization_path),resource_check=resource,
        started_epoch=now,absolute_deadline_epoch=auth['absolute_deadline_epoch'],export_deadline_epoch=request['export_deadline_epoch'])
    write(root/'technical_started.json',state)
    try:
        command=[sys.executable,'-B','-u','-m',__package__+'.producer','--component','export-worker','--request',str(root/'execution_request.json')]
        code=wait_child(command,evaluate.child_environment(auth['physical_GPU']),root,request['export_deadline_epoch'])
        require(code==0, 'Technical export worker failed; stop with no retry or dependent evaluation.')
        result=read(root/'joint_result.json')
        require(result['phase']=='technical_validation' and result['formal_eligible'] is False and result['endpoint']['formal_eligible'] is False
            and result['quality_status']=='not_evaluated_technical_only' and not result['complete_full8_evaluated']
            and 'selection_proposal' not in result, 'Technical export cannot become a scientific selection artifact.')
        state.update(status='completed_technical_export_fresh_ledger_only',exit_code=0,checkpoint_id=result['endpoint']['checkpoint_id'],
            result=binding(root/'joint_result.json'),payload_cleanup='pending_owner_review_and_cleanup_within_authorization')
    except BaseException:
        state.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        finished=time.time();state.update(finished_epoch=finished,internal_window_seconds=finished-now,
            authorization_started_epoch=auth['authorized_start_epoch'],conservative_authorization_window_seconds=finished-auth['authorized_start_epoch'],
            conservative_authorization_GPU_hours=(finished-auth['authorized_start_epoch'])/3600,
            cost_scope='Worker/controller window through finish; final cleanup closure must extend this same authorization-start window, never add overlapping windows.')
        write(root/'technical_run.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',required=True,choices=('prepare','run','validate-gpu','export-worker'))
    parser.add_argument('--input');parser.add_argument('--output');parser.add_argument('--proposal');parser.add_argument('--authorization');parser.add_argument('--request')
    args=parser.parse_args()
    if args.component=='prepare':
        require(args.input and args.output, 'prepare requires explicit spec input and new proposal output.')
        write(args.output,prepare(read(args.input)))
    elif args.component in ('run','validate-gpu'):
        require(args.proposal and args.authorization, 'run requires the actual proposal and separate actual authorization.')
        (run if args.component=='run' else validate_gpu)(args.proposal,args.authorization)
    else:
        require(args.request, 'The bounded internal export worker requires its actual execution request.')
        export_worker(args.request)


if __name__=='__main__':main()


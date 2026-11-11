"""Actual seven-bank tiny mechanisms and unapproved real-profile preparation."""
import argparse
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from . import producer as p


def refused(function):
    try: function()
    except (ValueError,FileNotFoundError,KeyError): return True
    raise AssertionError('Invalid input was accepted.')


def real_spec(output_root):
    from experiments.cat_profile_coverage_cmcc_20261104 import Snapshot, expected_sources, DEFAULT_UP, DEFAULT_EIGHT, DEFAULT_NIGHT
    snapshot=Snapshot();rows,context=expected_sources(snapshot,DEFAULT_UP,DEFAULT_EIGHT,DEFAULT_NIGHT)
    night=snapshot.read(DEFAULT_NIGHT);profiles={}
    for row in rows:
        if row['recipe']['K']!=32: continue
        formal,full8=Path(row['formal_dir']),Path(row['full8_dir'])
        canonical=full8/'profile_full8_result.json' if row['origin']=='up' else formal/'profile_full8_result.json'
        profiles[row['category_id']]=dict(canonical=p.binding(canonical),checkpoint_id=p.read(canonical)['endpoint']['checkpoint_id'],
            full8=dict(summary=p.binding(full8/'full8_summary.json'),evaluation=p.binding(full8/'evaluation.json')))
    n=context['inventory']['total_target_numel']
    return dict(inventory_receipt=night['inventory'],baseline_receipt=night['baseline'],run_identity=context['identity'],profiles=profiles,
        rate_vector=[[category,'2'] for category in sorted(profiles)],output_root=str(output_root),allowed_physical_GPUs=[0,1,2,3,4,5],
        limits=dict(export_seconds=900,full8_seconds=900,total_seconds=1800,max_logical_bits=int(Fraction(13,5)*n)))


def prepare_checks(directory):
    spec=real_spec(directory/'UNEXECUTED_joint_model')
    proposal=p.prepare(spec);p.write(directory/'explicit_spec_UNAPPROVED.json',spec);p.write(directory/'proposal_UNAPPROVED.json',proposal)
    p.verify_preparation(proposal)
    checks={}
    for name,change in (
        ('missing_category',lambda value:value['profiles'].pop(next(iter(value['profiles'])))),
        ('duplicate_foreign_category',lambda value:value['profiles'].update({next(iter(value['profiles'])):deepcopy(list(value['profiles'].values())[1])})),
        ('foreign_parent',lambda value:value['run_identity'].update(parent_uuid='negative_test_foreign_parent')),
        ('foreign_UUID',lambda value:next(iter(value['profiles'].values())).update(checkpoint_id='negative_test_foreign_UUID')),
        ('wrong_rate',lambda value:value['rate_vector'][0].__setitem__(1,'5/2')),
        ('wrong_source_SHA',lambda value:next(iter(value['profiles'].values()))['canonical'].update(sha256='0'*64)),
        ('invalid_GPU_id',lambda value:value.update(allowed_physical_GPUs=[-1])),
        ('overlapping_budget',lambda value:value['limits'].update(total_seconds=1799)),
        ('over_logical_2_6',lambda value:value['limits'].update(max_logical_bits=10**20)),
    ):
        bad=deepcopy(spec);change(bad);checks[name]=refused(lambda:p.prepare(bad))
    unapproved=directory/'not_authorized.json';p.write(unapproved,dict(user_approved=False))
    checks['unapproved_run']=refused(lambda:p.run(directory/'proposal_UNAPPROVED.json',unapproved))
    checks['no_execution_directory_created']=not Path(proposal['output_root']).exists()
    # Pure authorization-binding counterexamples never create a valid future file.
    checks['missing_authorization']=refused(lambda:p.authorization(proposal,directory/'proposal_UNAPPROVED.json',directory/'missing_auth.json'))
    theoretical=dict(user_approved=True,authorization_reference='CPU mathematical past-window test, not actual authorization',
        execution_scope='export_and_full8',
        proposal_sha256=p.file_sha(directory/'proposal_UNAPPROVED.json'),preparation_identity=proposal['preparation_identity'],
        proposal_id=proposal['proposal_id'],output_root=proposal['output_root'],allowed_physical_GPUs=proposal['allowed_physical_GPUs'],
        limits=proposal['limits'],physical_GPU=0,authorized_GPU_hours=.5,authorized_start_epoch=10,absolute_deadline_epoch=1810)
    reference=p.binding(directory/'proposal_UNAPPROVED.json')
    for name,changed,now in (('expired_authorization',{},1811),('budget_extension',dict(absolute_deadline_epoch=1811),15),
        ('foreign_proposal_authorization',dict(proposal_sha256='0'*64),15),('protected_GPU_authorization',dict(physical_GPU=7),15)):
        checks[name]=refused(lambda:p.authorization_contract(proposal,reference,dict(theoretical,**changed),now=now))
    checks['missing_execution_scope']=refused(lambda:p.authorization_contract(proposal,reference,{key:value for key,value in theoretical.items() if key!='execution_scope'},now=15))
    assert p.export_phase('export_fresh_ledger_only')==('technical_validation',False)
    assert p.export_phase('export_and_full8')==('formal',True)
    checks['technical_scope_cannot_enter_formal_full8']=refused(lambda:p.evaluation_plan(dict(schema=p.SCHEMA,proposal_id=proposal['proposal_id'],
        banks=proposal['banks'],run_identity=proposal['run_identity'],formal_eligible=False,phase='technical_validation'),proposal,
        output=directory/'never_technical_full8',authorization_value=None))
    from experiments.cat_offline_execution_cmcc_20261104.selection import freeze_joint
    future=Path(proposal['output_root'])/'full8'
    frozen=freeze_joint(inventory_receipt=proposal['inventory_receipt']['path'],baseline_path=proposal['baseline_receipt']['path'],
        run_identity=proposal['run_identity'],proposals=[dict(proposal_id=proposal['proposal_id'],rate_vector=proposal['rate_vector'],banks=proposal['banks'],
            full8=dict(summary=str(future/'full8_summary.json'),evaluation=str(future/'evaluation.json')))])
    assert frozen['kind']=='joint_set' and len(frozen['proposals'])==1 and not future.exists()
    checks['selection_freeze_joint_schema_only']=True
    return proposal,dict(actual_profile_count=len(proposal['banks']),actual_projection_count=sum(len(row['recipe']['group_signature']) for row in proposal['banks'].values()),
        source9_count=len(proposal['run_identity']['code_identity']['source_sha256']),
        source_class_logical_bits=sum(row['logical_bits'] for row in proposal['source_class_fees'].values()),
        source_class_packed_bits=sum(row['packed_bits'] for row in proposal['source_class_fees'].values()),
        payload_bytes_rehashed=False,sample_bytes_reverified=False,formal_eligible=False,checks=checks,
        proposal=p.binding(directory/'proposal_UNAPPROVED.json'),authorization_generated=False)


def tiny_checks(directory, prepared):
    import torch
    from experiments.cat_auto_protocol_cmcc_20261104.validate_fit_cpu import tiny_fixture
    from experiments.cat_auto_protocol_cmcc_20261104.config import resolve_recipe
    from experiments.cat_auto_protocol_cmcc_20261104.fit_adapter import fit_category, install_category_bank
    from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
    from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
    from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
    from train_utils import checkpoint_v6 as v6
    original,inventory,identity,calibration=tiny_fixture()
    # The old fixture casts its nonpersistent RoPE to BF16. Rebuild with the same
    # canonical BF16 factory as fresh, retaining the exact original state weights.
    config_dir=directory/'source_config';original.config.save_pretrained(config_dir)
    canonical=architecture(config_dir,attn_implementation='eager')
    loaded=canonical.load_state_dict(original.state_dict(),strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert tensor_digest(canonical.state_dict())==tensor_digest(original.state_dict())
    original=canonical.eval()
    assert all(value.dtype==torch.float32 for name,value in original.named_buffers() if 'inv_freq' in name)
    names={row['actual_name'] for row in inventory['projections']};banks={};fees={};source_UUIDs=[]
    for category in inventory['category_ids']:
        recipe=resolve_recipe(inventory,category,'2',dim=8,rotation_level='none',protection_fraction='0')
        fitted=fit_category(deepcopy(original),inventory=inventory,recipe=recipe,run_identity=identity,
            calibration_bundle=calibration,device='cpu',output_dir=directory/category,technical_steps=2)
        assert fitted['fit_report']['technical_steps']==2 and not fitted['fit_report']['formal_eligible']
        banks[category]=fitted['category_bank'];single=directory/category/'raw'
        saved=v6.save_v6_full_checkpoint(fitted['model'],str(single),checkpoint_kind='round_base',compressed_targets=recipe['group_signature'],
            pending_dense_targets=sorted(names-set(recipe['group_signature'])),completed_categories=[],compression_categories=inventory['category_ids'],
            target_layers=sorted({row['layer_id'] for row in inventory['projections']}),target_modules=inventory['category_ids'],extra_meta=dict(cat_auto_phase='cpu_fixture'))
        source_UUIDs.append(saved['checkpoint_id']);fees[category]=measure_checkpoint(single,inventory)['per_category'][category]
        del fitted
    joint=deepcopy(original);saved,union=p.install_export(joint,inventory=inventory,identity=identity,banks=banks,
        output=directory/'joint',phase='cpu_fixture')
    reverse=deepcopy(original)
    for category in reversed(sorted(banks)):install_category_bank(reverse,banks[category],inventory=inventory,expected_parent_identity=identity)
    assert tensor_digest(joint.state_dict())==tensor_digest(reverse.state_dict())
    fresh,evidence=fresh_native(directory/'joint',model_factory=lambda path:architecture(path,attn_implementation='eager'))
    with torch.no_grad():
        expected=joint(input_ids=calibration['input_ids'][0]).logits.detach()
        reversed_logits=reverse(input_ids=calibration['input_ids'][0]).logits.detach()
        observed=fresh(input_ids=calibration['input_ids'][0]).logits.detach()
    p.compare_logits(expected,reversed_logits);numerical=p.compare_logits(expected,observed)
    ledger=measure_checkpoint(directory/'joint',inventory);p.check_ledger(ledger,fees,{},formal=False)
    assert ledger['N']==inventory['total_target_numel'] and ledger['active_compressed_count']==len(names)
    assert saved['checkpoint_id'] not in source_UUIDs and evidence['checkpoint_id']==saved['checkpoint_id'] and evidence['state_exact']
    metadata=p.read(directory/'joint/checkpoint_meta.json')
    assert metadata['completed_categories']==metadata['pending_dense_targets']==metadata['skip_targets']==[]
    bad=deepcopy(banks);bad[next(iter(bad))]['parent_identity']=dict(identity,parent_uuid='foreign')
    rejection=dict(foreign_bank_parent=refused(lambda:p.install_export(deepcopy(original),inventory=inventory,identity=identity,banks=bad,
        output=directory/'bad_parent',phase='cpu_fixture')),
        technical_cannot_export_formal=refused(lambda:p.install_export(deepcopy(original),inventory=inventory,identity=identity,banks=banks,
            output=directory/'bad_formal',phase='formal')),
        tiny_actual_budget_rejected=refused(lambda:p.check_ledger(ledger,fees,dict(max_logical_bits=int(Fraction(13,5)*ledger['N'])),formal=True)),
        CPU_cannot_enter_formal_full8=refused(lambda:p.evaluation_plan(dict(schema=p.SCHEMA,proposal_id=prepared['proposal_id'],banks=prepared['banks'],
            run_identity=prepared['run_identity'],formal_eligible=False,phase='cpu_fixture'),prepared,output=directory/'never_full8',authorization_value=None)))
    return dict(phase='cpu_fixture',formal_eligible=False,actual_tiny_HF_PUBLIC=True,actual_layers=2,actual_categories=len(banks),actual_projection_count=len(names),
        actual_updates_per_stage=2,actual_stages=2,new_UUID=True,whole_state_strict_exact=evidence['state_exact'],
        original_and_fresh_RoPE_FP32=True,
        installation_order_state_exact=True,non_target_complement_exact=union['non_target_before_digest']==union['non_target_after_digest'],
        logits=numerical,ledger=ledger,rejections=rejection,temporary_tensors_removed_on_exit=True,
        limitation='Genuine tiny seven-bank native mechanisms; not8B GPU memory/FA2/full8/quality or final budget success.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--affected-from',help='Reuse the already complete tiny report; only preparation/precision/CLI/source changes are verified.')
    args=parser.parse_args()
    require_env=os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p.require(require_env,'Run CPU validation in caller bitvae with CUDA hidden.')
    import torch
    torch.set_num_threads(2);began=time.monotonic();output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    proposal,real=prepare_checks(output.parent)
    if args.affected_from:
        previous=p.read(args.affected_from)
        assert previous['exit_code']==0 and previous['cuda_initialized'] is False and previous['tiny']['whole_state_strict_exact']
        assert previous['tiny']['installation_order_state_exact'] and previous['tiny']['non_target_complement_exact']
        from experiments.cat_auto_protocol_cmcc_20261104.validate_fit_cpu import tiny_fixture
        from experiments.cat_auto_protocol_cmcc_20261104.live import architecture
        model,tiny_inventory,tiny_identity,_=tiny_fixture()
        assert refused(lambda:p.precision_audit(model,attention='eager'))
        with tempfile.TemporaryDirectory(prefix='cat_joint_precision_cpu_') as temporary:
            model.config.save_pretrained(temporary);canonical=architecture(temporary,attn_implementation='eager')
            canonical.load_state_dict(model.state_dict(),strict=True)
            precision=p.precision_audit(canonical,attention='eager')
            CPU_bank_rejected=refused(lambda:p.install_export(canonical,inventory=tiny_inventory,identity=tiny_identity,
                banks={},output=Path(temporary)/'never_technical_export',phase='technical_validation'))
        tiny=dict(reused_complete_CPU_report=p.binding(args.affected_from),scope='Prior full seven-bank tiny proof reused; affected checks do not repeat fits, installation or fresh.',
            affected_precision_audit=precision,original_cast_RoPE_rejected=True,CPU_bank_cannot_be_model_GPU_technical=CPU_bank_rejected,formal_eligible=False)
    else:
        with tempfile.TemporaryDirectory(prefix='cat_joint_export_cpu_') as temporary:tiny=tiny_checks(Path(temporary),proposal)
    command=[sys.executable,'-B','-m','experiments.cat_joint_export_cmcc_20261104.producer','--help']
    completed=subprocess.run(command,check=True,capture_output=True,text=True)
    assert 'export-worker' in completed.stdout and not torch.cuda.is_initialized()
    p.write(output,dict(exit_code=0,elapsed_seconds=time.monotonic()-began,cuda_initialized=torch.cuda.is_initialized(),
        source_sha256=p.pins(),actual_source_preparation=real,tiny=tiny,CLI_help_exit_code=completed.returncode,
        no_valid_GPU_authorization_created=True,no_8B_payload_or_sample_read=True,no_GPU_execution=True))
    print(dict(exit_code=0,elapsed_seconds=time.monotonic()-began,cuda_initialized=torch.cuda.is_initialized()),flush=True)


if __name__=='__main__':main()

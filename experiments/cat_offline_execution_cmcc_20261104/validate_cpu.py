"""Real reference/worker/tiny-v6 mechanisms; deliberately no evaluator inference."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from .protocol import baseline_receipt, binding, code_identity, read, write
from .results import worker_result, per_projection_fees
from .evaluate import load_bound_native, sample_api, make_plan

ROOT=Path(__file__).resolve().parents[2]
INVENTORY=ROOT/'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104/model_inventory_cpu.json'
PILOT=ROOT/'artifacts/cat_auto_protocol_cmcc_20261104/public_up_pilot_20261104_1455/completed.json'


def rejects(function,*args,**kwargs):
    try:function(*args,**kwargs)
    except (ValueError,FileNotFoundError,KeyError,TypeError):return
    raise AssertionError('A violated actual-artifact contract was accepted.')


def tiny_check(root):
    import torch
    from transformers import BertTokenizerFast
    from train_utils import checkpoint_v6 as v6
    from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
    from experiments.cat_auto_protocol_cmcc_20261104.validate_recovery_cpu import tiny_native_model,CATEGORIES
    torch.manual_seed(731);torch.set_num_threads(2)
    model,prefix,_current,future=tiny_native_model()
    before={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    raw=root/'native'
    v6.save_v6_full_checkpoint(model,str(raw),checkpoint_kind='round_base',compressed_targets=prefix,pending_dense_targets=future,
        completed_categories=['q_proj'],compression_categories=CATEGORIES,target_layers=[0,1],target_modules=CATEGORIES,
        cat_runtime_state=build_cat_runtime_state(activation_runtime=None,global_adaptive_plan=None,runtime_identity=dict(phase='cpu_fixture')),
        extra_meta=dict(cat_auto_phase='cpu_fixture'))
    meta=read(raw/'checkpoint_meta.json')
    endpoint=dict(path=str(raw),checkpoint_id=meta['checkpoint_id'],metadata_sha256=file_sha(raw/'checkpoint_meta.json'),
        state_sha256=file_sha(raw/meta['state_dict_file']),formal_eligible=False,phase='cpu_fixture')
    policy=dict(model_type=model.config.model_type,rope_theta=model.config.rope_theta,rope_scaling=model.config.rope_scaling,attn_implementation='flash_attention_2')
    loaded,audit=load_bound_native(endpoint,policy,device='cpu',cpu_mechanism=True)
    rejects(load_bound_native,dict(endpoint,checkpoint_id='foreign_CPU_counterexample'),policy,device='cpu',cpu_mechanism=True)
    rejects(load_bound_native,endpoint,dict(policy,rope_theta=policy['rope_theta']+1),device='cpu',cpu_mechanism=True)
    assert all(torch.equal(before[name],value.cpu()) and before[name].dtype==value.dtype for name,value in loaded.state_dict().items())
    with torch.inference_mode():
        logits=loaded(input_ids=torch.tensor([[2,5,6,3]]),use_cache=False).logits
        again=loaded(input_ids=torch.tensor([[2,5,6,3]]),use_cache=False).logits
    assert torch.equal(logits,again) and torch.isfinite(logits).all() and not audit['warmup']['failed']
    vocab=root/'vocab.txt';vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntiny\ntrain\n')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False)
    api=sample_api()
    lm=api.lm_eval.models.huggingface.HFLM(pretrained=loaded,tokenizer=tokenizer,backend='causal',trust_remote_code=True,batch_size='64',add_bos_token=False)
    assert lm.max_length==model.config.max_position_embeddings and lm.add_bos_token is False
    return dict(verification_scope='CPU_tiny_native_load_group8_forward_HFLM_constructor_only',checkpoint_id=meta['checkpoint_id'],
        full_state_exact=True,actual_native_modules=len(prefix),warmup=audit['warmup'],rope_FP32_verified=audit['rope_FP32_verified'],
        repeated_forward_exact=True,logits_dtype=str(logits.dtype),actual_HFLM_max_length=lm.max_length,actual_HFLM_add_bos_token=lm.add_bos_token,
        foreign_UUID_rejected=True,changed_RoPE_policy_rejected=True,
        full8_evaluator_called=False,GPU_path_validated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU validation uses caller bitvae/hidden CUDA.')
    import torch
    started=time.time()
    baseline=baseline_receipt(INVENTORY)
    reference_path=args.output.parent/'bf16_quality_receipt.json'
    if reference_path.exists():require(read(reference_path)==baseline,'Existing reference receipt changed.')
    else:write(reference_path,baseline)
    completed=read(PILOT);candidate=completed['candidates'][0]['candidate_id']
    command=read(PILOT.parent/'process.json')['command']
    request_reference=binding(command[command.index('--request')+1])
    actual=worker_result(PILOT,candidate,require_formal=False,request_reference=request_reference)
    assert not actual['formal_eligible'] and actual['phase']=='technical_validation'
    rejects(worker_result,PILOT,candidate,request_reference=request_reference)
    fee=deepcopy(actual['ledger']['per_category'][actual['category_id']]);fee['logical_bits']+=1
    rejects(per_projection_fees,actual['bank'],fee)
    malformed=deepcopy(baseline);malformed['model_identity']={'different_actual_model':True}
    rejects(make_plan,result=actual,baseline=malformed,baseline_path=reference_path,output=args.output.parent/'not_created')
    with tempfile.TemporaryDirectory(prefix='cat_offline_native_CPU_') as directory:
        tiny=tiny_check(Path(directory))
    require(not torch.cuda.is_initialized(),'CPU evidence must never initialize CUDA.')
    result=dict(status='CPU_actual_reference_worker_native_mechanisms_verified',CUDA_initialized=False,elapsed_seconds=time.time()-started,
        source_sha256=code_identity(),baseline=dict(mean_percent=baseline['mean_percent'],leaf_count=baseline['leaf_count'],sample_count=baseline['sample_count'],
            quality_reference_verified=baseline['quality_reference_verified'],ordered_pairing_verified=baseline['ordered_pairing_verified'],protocol_identity=baseline['protocol_identity']),
        actual_technical_worker=dict(candidate_id=candidate,checkpoint_id=actual['endpoint']['checkpoint_id'],rate=actual['rate'],joint_KL=actual['joint_KL'],
            complete_projections=len(actual['per_projection_fees']),actual_class_logical_bits=actual['logical_bits'],actual_class_packed_bits=actual['packed_bits'],
            per_projection_fees_exactly_close=True,formal_eligible=actual['formal_eligible']),
        negative_cases=dict(technical_formal_rejected=True,actual_fee_mismatch_rejected=True,different_model_rejected=True),tiny_native=tiny,
        limitations=['No real-model PUBLIC-v6 full8 GPU execution or positive native complete-score receipt yet.',
            'No S0-S13 persistent executor; historical BF16 ordered prediction identity unavailable and not reconstructed.'])
    write(args.output,result);print({key:result[key] for key in ('status','CUDA_initialized','elapsed_seconds')},flush=True)


if __name__=='__main__':main()

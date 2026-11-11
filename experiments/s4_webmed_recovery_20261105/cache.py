"""Pinned full-HF BF16/FA2 teacher; raw conditional4 once, never pre-mixed."""
import gc
import os
import time
import weakref
from transformers import AutoModelForCausalLM
from . import bindings as b,data


def main():
    b.initialize();resource=b.resources('7');began=time.monotonic()
    dataset,prepared=data.load_dataset()
    b.require(not (b.DATA/'teacher.json').exists() and not (b.DATA/'raw_teacher.pt').exists(),'Never overwrite a teacher cache.')
    print(dict(event='new_teacher_launch',**b.parent.process_event(os.getpid()),resource=resource),flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(b.HF),torch_dtype=b.torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).eval().requires_grad_(False)
    names=[f'model.layers.{layer}.{block}.{category}' for layer in range(36)
        for block,categories in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),('mlp',('gate_proj','up_proj','down_proj')))
        for category in categories]
    teacher=data.original.original.teacher_identity(model.state_dict(),names)
    previous=b.read(data.original.b.DATA/'teacher.json')['teacher_identity']
    b.require(teacher==previous,'Actual full-HF teacher differs from the existing raw teacher identity.')
    life=data.original.b.life
    handles,device_map=life.apply_layer_device_map(model,layer_device_map={index:b.torch.device('cuda:0') for index in range(36)})
    rope=model.model.rotary_emb
    b.require(rope.inv_freq.dtype==rope.original_inv_freq.dtype==b.torch.float32
        and rope.inv_freq.data_ptr()==rope.original_inv_freq.data_ptr(),'Original FP32 RoPE alias required.')
    before=life.state_records(model);targets={}
    with b.torch.no_grad():
        for index,case in enumerate(dataset['training_cases'][:4096],1):
            logits=data.legacy.base.answer_logits(model,case)
            target=data.legacy.base.log_distribution(logits,case).detach().cpu().clone()
            b.require(logits.dtype==b.torch.bfloat16 and target.shape==(1,4) and target.dtype==b.torch.float32
                and b.torch.isfinite(target).all() and b.torch.logsumexp(target,dim=-1).abs().max()<2e-6,'Native conditional4 target failed.')
            targets[case['case_id']]=dict(input_signature=case['input_signature'],target_log_probs=target)
            del logits,target
            if index%32==0:print(dict(event='new_WebMed_raw_teacher',cached=index,seconds=time.monotonic()-began),flush=True)
    b.require(life.state_records(model)==before,'Teacher persistent state changed.')
    peak=b.memory();reference=weakref.ref(model)
    for handle in handles:handle.remove()
    del handles,model,rope;gc.collect();b.torch.cuda.empty_cache();b.torch._C._cuda_clearCublasWorkspaces();b.torch.cuda.empty_cache()
    b.require(reference() is None and b.torch.cuda.memory_allocated()==0,'Release complete teacher before training.')
    payload=dict(dataset=prepared['dataset'],teacher_identity=teacher,targets=targets)
    b.torch.save(payload,b.DATA/'raw_teacher.pt')
    med,_=data.helpers();med.require_same(payload,b.torch.load(b.DATA/'raw_teacher.pt',map_location='cpu',weights_only=True))
    b.write(b.DATA/'teacher.json',dict(status='verified_original_HF_WebMed4096_raw_conditional4',dataset=prepared['dataset'],
        payload=b.binding(b.DATA/'raw_teacher.pt'),teacher_identity=teacher,cases=4096,execution_batch=1,
        mixture_applications=0,teacher_fully_released=True,teacher_persistent_state_exact=True,
        RoPE_FP32_original_alias_preserved=True,device_map=device_map,physical_GPU=7,peak=peak,
        source_sha256=b.pins(),resource=resource,seconds=time.monotonic()-began))
    _,_,evidence=data.load();b.write(b.DATA/'data_contract.json',evidence)
    print(dict(event='new_WebMed_teacher_complete',cases=4096,seconds=time.monotonic()-began),flush=True)


if __name__=='__main__':main()

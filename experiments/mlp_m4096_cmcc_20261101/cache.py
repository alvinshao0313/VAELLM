"""Only new ranks1024..4095 use the identical full-HF BF16 static teacher math."""
import gc
import os
import sys
import time
import weakref

from transformers import AutoModelForCausalLM

from . import bindings as b
from . import data


def main():
    b.require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='1',
        'Use activated bitvae and only the assigned physical GPU1.')
    resource=b.executor.check_resources(['1'])
    b.require(b.torch.cuda.device_count()==1,'Use one BF16 teacher device.')
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    b.torch.set_float32_matmul_precision('highest')
    b.require(not (b.DATA/'teacher.json').exists() and not (b.DATA/'new_M3072.pt').exists(),'Never overwrite teacher targets.')
    dataset,prepared=data.load_dataset()
    began=time.monotonic()
    started_unix=time.time()
    print(dict(event='teacher_launch',pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
        argv=[sys.executable,*sys.argv],resource_check=resource),flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(b.HF),torch_dtype=b.torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).eval().requires_grad_(False)
    names=[f'model.layers.{layer}.{block}.{category}' for layer in range(36)
        for block,categories in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),('mlp',('gate_proj','up_proj','down_proj')))
        for category in categories]
    teacher=data.original.teacher_identity(model.state_dict(),names)
    handles,device_map=b.life.apply_layer_device_map(model,
        layer_device_map={index:b.torch.device('cuda:0') for index in range(36)})
    rope=model.model.rotary_emb
    b.require(rope.inv_freq.dtype==rope.original_inv_freq.dtype==b.torch.float32
        and rope.inv_freq.data_ptr()==rope.original_inv_freq.data_ptr(),'Keep actual full-HF FP32 RoPE.')
    frozen=b.life.state_records(model)
    targets={}
    with b.torch.no_grad():
        for index,case in enumerate(dataset['training_cases'][1024:4096],1):
            b.require(time.monotonic()-began<10800,'The bounded teacher preparation exceeded3h.')
            logits=data.legacy.base.answer_logits(model,case)
            target=data.legacy.base.log_distribution(logits,case).detach().cpu().clone()
            b.require(target.shape==(1,4) and target.dtype==b.torch.float32 and b.torch.isfinite(target).all(),
                'Invalid actual full-HF native target.')
            targets[case['case_id']]=dict(input_signature=case['input_signature'],target_log_probs=target)
            del logits,target
            if index%32==0:
                print(dict(event='new_M3072_teacher',cached=index,elapsed_seconds=time.monotonic()-began),flush=True)
    b.require(b.life.state_records(model)==frozen,'Teacher inference changed a persistent HF tensor.')
    reference=weakref.ref(model)
    for handle in handles:handle.remove()
    del handles
    del model
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()
    b.require(reference() is None and b.torch.cuda.memory_allocated()==0,'Release the single teacher after caching.')
    payload=dict(dataset=prepared['dataset'],teacher_identity=teacher,targets=targets)
    b.torch.save(payload,b.DATA/'new_M3072.pt')
    saved=b.torch.load(b.DATA/'new_M3072.pt',map_location='cpu',weights_only=True)
    b.require(saved.keys()==payload.keys() and saved['teacher_identity']==teacher
        and saved['targets'].keys()==targets.keys() and all(b.torch.equal(item['target_log_probs'],
        saved['targets'][key]['target_log_probs']) for key,item in targets.items()),'Actual teacher save/reload changed.')
    b.write(b.DATA/'teacher.json',dict(status='complete_new_M3072_full_BF16_teacher',dataset=prepared['dataset'],
        pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),argv=[sys.executable,*sys.argv],
        python=sys.executable,started_unix=started_unix,physical_GPU=1,
        actual_device_map=device_map,RoPE_FP32_original_alias_preserved=True,
        teacher_identity=teacher,payload=b.binding(b.DATA/'new_M3072.pt'),
        source_sha256={str(path):b.sha(path) for path in (b.HERE/'cache.py',b.HERE/'data.py',b.HERE/'pool_build.py',
            b.Path(data.legacy.base.__file__))},
        teacher_fully_released=True,actual_new_cases=3072,old_M1024_QA_reused=True,
        resource_check=resource,elapsed_seconds=time.monotonic()-began))
    _dataset,_recovery,evidence=data.load()
    b.write(b.DATA/'data_contract.json',evidence)
    print(dict(status='complete_new_M3072_full_BF16_teacher',elapsed_seconds=time.monotonic()-began,
        actual_exposure=evidence['full_exposure']['total_input_tokens']),flush=True)


if __name__=='__main__':main()

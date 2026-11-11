"""Original-HF B1/FA2/FP32RoPE SciQ conditional4 cache, mixed once."""
import gc
import json
import os
import time
import weakref

from transformers import AutoModelForCausalLM

from . import bindings as b
from experiments.muon_compact_cmcc_20261030 import rollout


def main():
    b.configure()
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '4' and b.torch.cuda.device_count() == 1,
        'Only assigned GPU4 in activated bitvae.')
    resource = b.executor.check_resources(['4'])
    old = b.original.read_prepared(b.ORIGINAL_PREPARED)
    directory = b.OUTPUT/'teacher'
    b.require(not directory.exists(), 'Never overwrite or retry teacher cache.')
    directory.mkdir()
    started = time.monotonic()
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    dataset = b.inputs.read_dataset()
    cases = [case for case in dataset['training_cases'] if case['case_id'].startswith('allenai_sciq/')]
    b.require(len(cases) == 512, 'Require all fixed SciQ512, serial B1.')
    teacher = AutoModelForCausalLM.from_pretrained(str(b.HF),local_files_only=True,trust_remote_code=False,
        torch_dtype=b.torch.bfloat16,attn_implementation='flash_attention_2').eval().requires_grad_(False)
    names = [f'model.layers.{layer}.{block}.{category}' for layer in range(36)
        for block,categories in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),('mlp',('gate_proj','up_proj','down_proj')))
        for category in categories]
    identity = b.original.inputs.teacher_identity(teacher.state_dict(),names)
    b.require(identity == old['record']['teacher_identity'], 'Original full-HF teacher identity differs.')
    rollout.move_teacher(teacher,'cuda:0',b.core.native_policy(teacher))
    before = rollout.teacher_stamp(teacher)
    helper = b.mature.aligned.reference.mixed.helper
    targets = {}
    with b.torch.no_grad():
        for index,case in enumerate(cases):
            b.core.tick(started+1800)
            logits = b.mature.base.answer_logits(teacher,case)
            options = case['native_continuation_ids'][:,0].to(logits.device)
            pure = logits.index_select(-1,options).float().log_softmax(-1).cpu().contiguous()
            b.require(logits.dtype == b.torch.bfloat16 and pure.shape == (1,4)
                and b.torch.isfinite(pure).all(), 'Finite original native B1 last-query distribution required.')
            mixed = helper.target_log_distribution(pure,case)
            b.require(b.torch.allclose(mixed.exp(), .5*pure.exp()+.5*b.torch.nn.functional.one_hot(
                b.torch.tensor([case['gold_index']]),4),rtol=0,atol=1e-7), 'Exactly one teacher/gold mixture required.')
            targets[case['case_id']] = dict(input_signature=case['input_signature'],pure_teacher_log_probs=pure,
                target_log_probs=mixed)
            del logits, options
            if (index+1)%32 == 0:
                print(json.dumps(dict(event='serial_SciQ_HF_cache',cases=index+1,seconds=time.monotonic()-started)),flush=True)
    b.require(rollout.teacher_stamp(teacher) == before, 'Teacher changed while caching.')
    peak = rollout.memory()
    ref = weakref.ref(teacher)
    del teacher
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()
    b.require(ref() is None and b.torch.cuda.memory_allocated() == 0, 'Release complete teacher before training.')
    payload = dict(teacher_identity=identity,targets=targets)
    b.torch.save(payload,directory/'targets.pt')
    saved = b.torch.load(directory/'targets.pt',map_location='cpu',weights_only=True)
    _native,common = b.inputs.helpers()
    common.require_same(saved,payload)
    manifest = dict(status='verified_serial_HF_SciQ512_once_mixed',source_checkpoint_id=b.PARENT_ID,
        dataset=b.binding(b.OUTPUT/'dataset.pt'),payload=b.binding(directory/'targets.pt'),teacher_identity=identity,
        cases=512,teacher_execution_batch=1,teacher_fully_released=True,teacher_unchanged=True,
        mixture_applications=1,mixture_helper=b.binding(helper.__file__),conditional_native4=True,
        actual_HF_input_tokens=sum(len(case['input_ids']) for case in cases),target_positions=512,
        save_read_exact=True,physical_GPU=4,resource_check=resource,peak=peak,
        implementation=b.code_bindings(),elapsed_seconds=time.monotonic()-started)
    b.core.write(directory/'cache.json',manifest)
    print(json.dumps(dict(status=manifest['status'],seconds=manifest['elapsed_seconds'],peak=peak)),flush=True)


if __name__ == '__main__':
    main()

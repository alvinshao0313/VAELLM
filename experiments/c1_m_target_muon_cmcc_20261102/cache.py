"""Cache only the original M1024 full-HF last-query distributions on GPU4."""
import gc
import os
import time
import weakref

from transformers import AutoModelForCausalLM

from . import bindings as b
from . import objective


def main():
    b.configure()
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '4' and b.torch.cuda.device_count() == 1,
        'Use activated bitvae and only the assigned physical GPU4.')
    resource = b.original.executor.check_resources(['4'])
    prepared = b.read_prepared(b.PREPARED)
    directory = b.OUTPUT / 'teacher'
    b.require(not directory.exists(), 'Never overwrite full-M targets.')
    directory.mkdir()
    began = time.monotonic()
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    dataset, recovery, _, _ = b.inputs.load_inputs()
    composed = b.read(b.inputs.HOT / 'teacher_targets.json')
    component = composed['components'][0]
    old_payload = b.torch.load(b.inputs.checked(component['payload_binding']),
        map_location='cpu', weights_only=True, mmap=True)
    pure = old_payload['targets']
    teacher = AutoModelForCausalLM.from_pretrained(str(b.HF), local_files_only=True,
        trust_remote_code=False, torch_dtype=b.torch.bfloat16,
        attn_implementation='flash_attention_2').eval().requires_grad_(False)
    names = [f'model.layers.{layer}.{block}.{category}' for layer in range(36)
        for block, categories in (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
            ('mlp', ('gate_proj', 'up_proj', 'down_proj'))) for category in categories]
    teacher_identity = b.inputs.teacher_identity(teacher.state_dict(), names)
    b.require(teacher_identity == prepared['record']['teacher_identity'], 'Use the unchanged independently loaded full-HF teacher.')
    policy = b.core.native_policy(teacher)
    b.rollout.move_teacher(teacher, 'cuda:0', policy)
    before = b.rollout.teacher_stamp(teacher)
    targets, conditional_checks, pure_checks = {}, [], []
    with b.torch.no_grad():
        for group in dataset['groups']:
            cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
            full, conditional, _ = objective.distributions(teacher, cases)
            for row, case in enumerate(cases):
                value = full[row:row+1].cpu().contiguous()
                b.require(value.shape == (1, 151936) and b.torch.isfinite(value).all()
                    and float((value.exp().sum(-1) - 1).abs().max()) < 2e-6,
                    'Full-HF target must be finite normalized FP32 over all151936 tokens.')
                targets[case['case_id']] = dict(input_signature=case['input_signature'], target_log_probs=value)
                label = b.torch.full_like(conditional[row:row+1], -b.torch.inf)
                label[0, case['gold_index']] = 0.
                remixed = b.torch.logaddexp(conditional[row:row+1] - b.torch.log(b.torch.tensor(2., device=full.device)),
                    label - b.torch.log(b.torch.tensor(2., device=full.device)))
                conditional_checks.append(float((remixed.cpu() - recovery[case['case_id']]['target_log_probs']).abs().max()))
                del label, remixed
                old_pure = pure[case['case_id']]['target_log_probs']
                new_pure = conditional[row:row+1].cpu()
                b.require(old_pure.shape == new_pure.shape == (1, 4), 'Compare the original unmixed F four-option cache.')
                pure_checks.append(dict(case_id=case['case_id'],
                    maximum_probability_difference=float((new_pure.exp() - old_pure.exp()).abs().max()),
                    maximum_log_probability_difference=float((new_pure - old_pure).abs().max()),
                    argmax_changed=new_pure.argmax(-1).item() != old_pure.argmax(-1).item()))
            print(dict(event='full_HF_M_cache_group', group_index=group['group_index'], cached=len(targets),
                elapsed_seconds=time.monotonic() - began), flush=True)
            del full, conditional
    b.require(len(targets) == 1024 and b.rollout.teacher_stamp(teacher) == before, 'Teacher state or original M coverage changed.')
    peak = b.rollout.memory()
    reference = weakref.ref(teacher)
    del teacher
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()
    b.require(reference() is None and b.torch.cuda.memory_allocated() == 0, 'Release complete teacher after caching.')
    payload = dict(teacher_identity=teacher_identity, targets=targets)
    b.torch.save(payload, directory / 'targets.pt')
    saved = b.torch.load(directory / 'targets.pt', map_location='cpu', weights_only=True, mmap=True)
    b.require(saved['teacher_identity'] == teacher_identity and saved['targets'].keys() == targets.keys()
        and all(b.torch.equal(item['target_log_probs'], saved['targets'][key]['target_log_probs'])
            for key, item in targets.items()), 'Full-M cache save/readback changed.')
    b.core.write(directory / 'cache.json', dict(status='verified_full_HF_M1024_last_query',
        source_checkpoint_id=prepared['record']['source_checkpoint_id'], original_prepared=b.binding(b.ORIGINAL_PREPARED),
        teacher_identity=teacher_identity, payload=b.binding(directory / 'targets.pt'), teacher_fully_released=True,
        cases=1024, raw_FP32_probability_bytes=1024 * 151936 * 4,
        old_conditional_mixed_target_max_abs_difference=max(conditional_checks),
        old_pure_F_cache=b.binding(b.inputs.checked(component['payload_binding'])),
        old_pure_F_conditional_comparison=pure_checks,
        old_pure_F_maximum_probability_difference=max(row['maximum_probability_difference'] for row in pure_checks),
        old_pure_F_argmax_changed_cases=sum(row['argmax_changed'] for row in pure_checks),
        conditional_comparison_boundary='Same HF/input/native4; original cache serial vs this M32 BF16 execution may have ordinary rounding differences.',
        physical_GPU=4, resource_check=resource, peak=peak, implementation=b.code_bindings(),
        elapsed_seconds=time.monotonic() - began))
    print(dict(status='verified_full_HF_M1024_last_query', seconds=time.monotonic() - began,
        old_mixed_max_abs_difference=max(conditional_checks)), flush=True)


if __name__ == '__main__':
    main()

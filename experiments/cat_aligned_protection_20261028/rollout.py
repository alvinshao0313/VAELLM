"""Temporary native BF16 generation, followed by exact all252 live-master restoration."""
from contextlib import contextmanager
import gc
import time

from bindings import core, qa, torch, base, LIMITS
from litebsq.vae_linear import prime_named_vae_linear_cache


def memory():
    torch.cuda.synchronize()
    return {key: value() for key, value in {
        'allocated_bytes': torch.cuda.memory_allocated,
        'reserved_bytes': torch.cuda.memory_reserved,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved}.items()}


def leaves(active):
    return {name: {'identity': id(p), 'data_ptr': p.data_ptr(), 'version': p._version,
        'value': core.tensor_record(p), 'requires_grad': p.requires_grad,
        'grad_identity': None if p.grad is None else id(p.grad)} for name, p in active.items()}


def adam(optimizer, active):
    return {'parameter_refs': [[id(p) for p in group['params']] for group in optimizer.param_groups],
        'state': {name: {key: {'identity': id(value), 'data_ptr': value.data_ptr(),
            'value': core.tensor_record(value)} if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state.get(p, {}).items()} for name, p in active.items()}}


def teacher_stamp(model):
    return {'parameters': {name: [id(p), p._version, str(p.dtype), list(p.shape)]
        for name, p in model.named_parameters()}, 'policy': core.native_policy(model)}


def move_teacher(model, device, policy):
    model.to(device=device)
    for child in model.modules():
        value = getattr(child, 'original_inv_freq', None)
        if isinstance(value, torch.Tensor):
            child.original_inv_freq = value.to(device=device)
    core.require(core.native_policy(model) == policy
        and all(value.device == torch.device(device) for value in list(model.parameters())+list(model.buffers()))
        and all(child.original_inv_freq.device == torch.device(device)
            and child.original_inv_freq.dtype == torch.float32
            and torch.equal(child.original_inv_freq, child.inv_freq)
            for child in model.modules() if isinstance(getattr(child, 'original_inv_freq', None), torch.Tensor)),
        'Device-only teacher move must preserve the original FP32 RoPE and its nonregistered alias.')


@contextmanager
def rollout_mode(model, modules, active, optimizer, evidence, expected_elements):
    core.require(len(modules) == 252 and len(active) == 1512
        and sum(p.numel() for p in active.values()) == expected_elements
        and all(p.dtype == torch.float32 and p.requires_grad and p.grad is None for p in active.values())
        and all(value.trainable_decode and not value.cache_decoded_weight for value in modules.values())
        and getattr(model, '_cache', None) is None, 'Rollout must start before accumulation with all252 live masters.')
    before, adam_before = leaves(active), adam(optimizer, active)
    masters = {name: p.data for name, p in active.items()}
    modes = {child: child.training for child in model.modules()}
    checkpoint = {name: value._parallel_stage_decoder.use_checkpoint for name, value in modules.items()}
    cache_setting = model.config.use_cache
    try:
        model.eval(); model.config.use_cache = True
        with torch.no_grad():
            for name, p in active.items():
                p.data = masters[name].to(torch.bfloat16)
            for value in modules.values():
                value.disable_trainable_decode()
            warm = prime_named_vae_linear_cache(list(modules.items()), dtype=torch.bfloat16,
                clear_existing=True, group_size=8)
        core.require(warm == {'total': 252, 'warmed': 252, 'skipped': 0, 'failed': 0}
            and all(value._cached_weight is not None
                and value._cached_weight.dtype == torch.bfloat16
                and not torch.is_inference(value._cached_weight) for value in modules.values()),
            'Require normal no_grad all252 group8 native BF16 caches before inference_mode generation.')
        evidence['prewarm'] = {**warm, 'group_size': 8, **memory()}
        yield
    finally:
        for value in modules.values():
            value.clear_decoded_weight_cache()
        for name, p in active.items():
            p.data = masters[name]
        for value in modules.values():
            value.enable_trainable_decode(parallel_stage_decode=True)
        for child, training in modes.items():
            child.training = training
        for name, value in modules.items():
            value._parallel_stage_decoder.use_checkpoint = checkpoint[name]
        model.config.use_cache = cache_setting
        core.require(all(value._cached_weight is None and value.trainable_decode
            and not value.cache_decoded_weight and value._decoder_compute_dtype == torch.bfloat16
            for value in modules.values()) and getattr(model, '_cache', None) is None,
            'No projection or KV cache may survive into all252 decoder backpropagation.')
        after, adam_after = leaves(active), adam(optimizer, active)
        core.require(after == before and adam_after == adam_before,
            'Generation changed an exact FP32 master, Parameter, gradient or persistent Adam reference/value.')
        evidence.update(master_and_adam_exact=True, parameter_count=len(active),
            optimizer_initialized_leaves=sum(bool(value) for value in adam_before['state'].values()),
            master_before_sha256=core.mapping_digest(before), master_after_sha256=core.mapping_digest(after),
            optimizer_before_sha256=core.mapping_digest(adam_before), optimizer_after_sha256=core.mapping_digest(adam_after),
            frozen_projection_cache_count=0, decoded_and_KV_cache_cleared=True, after_restore=memory())


def replay_case(case, generated, eos_ids):
    tokens = generated['output_token_ids']
    limit = LIMITS.get(case['domain'])
    core.require(limit is not None and case['source']['official_split'] == 'train'
        and 1 <= len(tokens) <= limit, 'Only bound train QA native rollouts may be replayed.')
    positions = [i for i, token in enumerate(tokens) if token in eos_ids]
    core.require(not positions or positions == [len(tokens)-1], 'Original EOS must terminate the raw trajectory.')
    core.require(positions or len(tokens) == limit, 'Do not append EOS or shorten the original generation limit.')
    prompt = case['prompt_input_ids']
    result = dict(case)
    result.update(input_ids=torch.cat((prompt, torch.tensor(tokens[:-1], dtype=torch.long))),
        target_ids=torch.tensor(tokens, dtype=torch.long),
        answer_query_positions=torch.arange(len(prompt)-1, len(prompt)+len(tokens)-1),
        answer_target_positions=torch.arange(len(prompt), len(prompt)+len(tokens)),
        trajectory_eos_mask=torch.tensor([token in eos_ids for token in tokens], dtype=torch.bool))
    return result


def _collect(model, teacher, tokenizer, cases, modules, active, optimizer, record, deadline):
    started = time.monotonic()
    policy = core.native_policy(teacher)
    move_teacher(teacher, 'cpu', policy)
    gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    transition = {'teacher_on_cpu_before_group8': memory()}
    rollouts, rows = {}, []
    with rollout_mode(model, modules, active, optimizer, transition, record['trainable_elements']):
        for case in cases:
            core.tick(deadline)
            generated = qa.generate_one(model, tokenizer, case['prompt_input_ids'].unsqueeze(0),
                {'dataset': case['domain'], 'max_new_tokens': LIMITS[case['domain']]}, diagnostic=True,
                protocol=record['model_config']['protocol'])
            expected_options = qa.generation_options(tokenizer, case['domain'], len(case['prompt_input_ids']),
                LIMITS[case['domain']], protocol=record['model_config']['protocol'])
            core.require(generated['generation_options'] == expected_options
                and generated['forward_trace']['first_calls'][0]['cache_length_before'] == 0
                and generated['forward_trace']['first_calls'][0]['input_shape'] == [1, len(case['prompt_input_ids'])],
                'Native rollout must use the complete original prompt/options and fresh KV.')
            replay = replay_case(case, generated, record['model_config']['protocol']['eos_token_id'])
            replay['source_input_signature'] = replay.pop('input_signature')
            rollouts[case['case_id']] = replay
            rows.append({'case_id': case['case_id'], 'domain': case['domain'],
                'source_input_signature': case['input_signature'],
                'prompt_tokens': len(case['prompt_input_ids']), 'replay_input_tokens': len(replay['input_ids']),
                'generated_tokens': len(replay['target_ids']), 'generation': generated,
                'replay_input': core.tensor_record(replay['input_ids']),
                'answer_query_positions': replay['answer_query_positions'].tolist()})
    transition['rollout_wall_seconds'] = time.monotonic()-started
    move_started = time.monotonic()
    move_teacher(teacher, 'cuda:0', policy)
    transition['teacher_return_seconds'] = time.monotonic()-move_started
    transition['teacher_returned'] = memory()
    return rollouts, rows, transition


def collect(model, teacher, tokenizer, cases, modules, active, optimizer, record, deadline, controller):
    evidence = {}
    with controller.native_generation(active, optimizer, evidence):
        trajectories, rows, transition = _collect(model, teacher, tokenizer, cases, modules, active, optimizer, record, deadline)
    transition['protection_transition'] = evidence
    return trajectories, rows, transition


@torch.no_grad()
def target(teacher, case):
    logits = base.answer_logits(teacher, case)
    result = base.capture.cpu_copy(base.log_distribution(logits, case))
    core.require(result.shape == (len(case['target_ids']), 151936)
        and result.dtype == torch.float32 and torch.isfinite(result).all().item(),
        'Current trajectory teacher target must cover the complete original vocabulary.')
    return result

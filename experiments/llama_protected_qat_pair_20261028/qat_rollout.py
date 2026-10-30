"""Native greedy QA preserving all1344 FP32 D and mixed optimizer states exactly."""
from contextlib import contextmanager, ExitStack
import gc
import time
import qat_common as c
from litebsq.vae_linear import prime_named_vae_linear_cache
from qat_state import optimization_proof

torch = c.torch
runner = c.native.module(c.PUBLIC / 'experiments/longbench_v1/runner.py',
    'experiments.longbench_v1._llama_sparse005_pair_runner')


def generate_one(model, tokenizer, case, policy):
    # Keep the generation policy bound to the explicit public Llama adapter,
    # even when the decoder runtime imports modules from the training snapshot.
    from transformers import GenerationConfig
    ids = case['prompt_input_ids'][None].to(model.get_input_embeddings().weight.device)
    options = c.protocol.generation_options(tokenizer, case['domain'], ids.shape[-1],
        c.LIMITS[case['domain']], protocol=policy)
    config, _ = model._prepare_generation_config(GenerationConfig(**options), use_model_defaults=False)
    c.require(all(getattr(config, key) == value for key, value in options.items()), 'Explicit greedy options changed.')
    trace = runner.ForwardTrace()
    handle = model.register_forward_pre_hook(trace.hook, with_kwargs=True)
    torch.cuda.synchronize(); started = time.monotonic()
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                generation_config=config, use_model_defaults=False)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    c.require(torch.equal(output[0, :ids.shape[-1]], ids[0]), 'Native generation changed original prompt IDs.')
    tokens = output[0, ids.shape[-1]:].detach().cpu().tolist()
    return {'output_token_ids': tokens, 'generated_tokens': len(tokens), 'elapsed_seconds': time.monotonic() - started,
        'generation_options': options, 'effective_generation_options': {key: getattr(config, key) for key in options},
        'use_model_defaults': False, 'forward_trace': trace.record()}


def memory():
    torch.cuda.synchronize()
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def teacher_stamp(model):
    return {'parameters': {name: [id(p), p._version, str(p.dtype), list(p.shape)] for name, p in model.named_parameters()},
        'policy': c.policy(model), 'RoPE': {name: c.tensor_record(value) for name, value in model.named_buffers() if 'inv_freq' in name}}


def move_teacher(model, device, policy):
    model.to(device=device)
    for child in model.modules():
        value = getattr(child, 'original_inv_freq', None)
        if isinstance(value, torch.Tensor):
            child.original_inv_freq = value.to(device=device)
    c.require(c.policy(model) == policy
        and all(value.device == torch.device(device) for value in [*model.parameters(), *model.buffers()])
        and all(child.original_inv_freq.device == torch.device(device) for child in model.modules()
            if isinstance(getattr(child, 'original_inv_freq', None), torch.Tensor)),
        'Teacher offload must change device only, including the nonregistered FP32 RoPE alias.')


@contextmanager
def native_decoders(model, modules, active, optimizer, evidence):
    decoder = {name: p for name, p in active.items() if '._parallel_stage_decoder.' in name}
    c.require(len(modules) == 224 and len(decoder) == 1344 and len(active) in (1344, 1568)
        and all(p.dtype == torch.float32 and p.requires_grad and p.grad is None for p in active.values())
        and all(value.trainable_decode and not value.cache_decoded_weight for value in modules.values())
        and getattr(model, '_cache', None) is None, 'Require all224 live masters before native generation.')
    before = optimization_proof(active, optimizer)
    masters = {name: parameter.data for name, parameter in decoder.items()}
    modes = {child: child.training for child in model.modules()}
    checkpoints = {name: value._parallel_stage_decoder.use_checkpoint for name, value in modules.items()}
    setting = model.config.use_cache
    try:
        model.eval(); model.config.use_cache = True
        with torch.no_grad():
            for name, parameter in decoder.items():
                parameter.data = masters[name].to(torch.bfloat16)
            for value in modules.values():
                value.disable_trainable_decode()
            warm = prime_named_vae_linear_cache(list(modules.items()), dtype=torch.bfloat16,
                clear_existing=True, group_size=8)
        c.require(warm == {'total': 224, 'warmed': 224, 'skipped': 0, 'failed': 0}
            and all(value._cached_weight is not None and value._cached_weight.dtype == torch.bfloat16
                and not torch.is_inference(value._cached_weight) for value in modules.values()),
            'Prewarm all224 native weights with normal no_grad tensors before inference_mode.')
        evidence['prewarm'] = {**warm, 'group_size': 8, **memory()}
        yield
    finally:
        for value in modules.values():
            value.clear_decoded_weight_cache()
        for name, parameter in decoder.items():
            parameter.data = masters[name]
        for value in modules.values():
            value.enable_trainable_decode(parallel_stage_decode=True)
        for child, mode in modes.items():
            child.training = mode
        for name, value in modules.items():
            value._parallel_stage_decoder.use_checkpoint = checkpoints[name]
        model.config.use_cache = setting
        c.require(all(value._cached_weight is None and value.trainable_decode and not value.cache_decoded_weight
            and value._decoder_compute_dtype == torch.bfloat16 for value in modules.values())
            and all(p.requires_grad for p in active.values())
            and getattr(model, '_cache', None) is None and optimization_proof(active, optimizer) == before,
            'Native generation failed exact master/Adam restoration or retained projection/KV caches.')
        evidence.update(master_and_optimizer_exact=True, initialized_leaves=before['initialized_leaves'],
            optimizer_steps=before['steps'], optimizer_kind='Muon_with_aux_Adam' if hasattr(optimizer, 'muon') else 'AdamW', projection_and_KV_caches_cleared=True, after_restore=memory())


def replay_case(case, generated, eos):
    tokens = generated['output_token_ids']
    cap = c.LIMITS[case['domain']]
    c.require(case['source']['official_split'] == 'train' and 1 <= len(tokens) <= cap, 'Bound train QA rollout only.')
    positions = [index for index, token in enumerate(tokens) if token in eos]
    c.require((not positions or positions == [len(tokens) - 1]) and (positions or len(tokens) == cap),
        'Keep the original actual terminal EOS or complete cap; append no EOS.')
    prompt = case['prompt_input_ids']
    result = dict(case)
    result.update(input_ids=torch.cat((prompt, torch.tensor(tokens[:-1], dtype=torch.long))),
        target_ids=torch.tensor(tokens, dtype=torch.long),
        answer_query_positions=torch.arange(len(prompt) - 1, len(prompt) + len(tokens) - 1),
        answer_target_positions=torch.arange(len(prompt), len(prompt) + len(tokens)),
        trajectory_eos_mask=torch.tensor([token in eos for token in tokens], dtype=torch.bool))
    result['source_input_signature'] = result.pop('input_signature')
    return result


def collect(model, teacher, tokenizer, cases, modules, active, optimizer, deadline, controller=None):
    started = time.monotonic()
    policy = c.policy(teacher)
    move_teacher(teacher, 'cpu', policy)
    gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    transition = {'teacher_on_CPU': memory()}
    replay, rows = {}, []
    with ExitStack() as contexts:
        if controller is not None:
            contexts.enter_context(controller.native_generation(active, optimizer, transition.setdefault('q', {})))
        contexts.enter_context(native_decoders(model, modules, active, optimizer, transition.setdefault('decoder', {})))
        for case in cases:
            c.tick(deadline)
            generated = generate_one(model, tokenizer, case, policy)
            expected = c.protocol.generation_options(tokenizer, case['domain'], len(case['prompt_input_ids']),
                c.LIMITS[case['domain']], protocol=policy)
            c.require(generated['generation_options'] == expected
                and generated['forward_trace']['first_calls'][0]['cache_length_before'] == 0
                and generated['forward_trace']['first_calls'][0]['input_shape'] == [1, len(case['prompt_input_ids'])],
                'Use exact original prompt, explicit greedy options and fresh KV.')
            value = replay_case(case, generated, policy['eos_token_id'])
            replay[case['case_id']] = value
            rows.append({'case_id': case['case_id'], 'domain': case['domain'], 'prompt_tokens': len(case['prompt_input_ids']),
                'TF_target_tokens': len(case['target_ids']), 'generated_tokens': len(value['target_ids']),
                'actual_EOS': bool(value['trajectory_eos_mask'][-1]), 'generation': generated,
                'replay_input': c.tensor_record(value['input_ids'])})
    move_teacher(teacher, 'cuda:0', policy)
    transition.update(wall_seconds=time.monotonic() - started, teacher_returned=memory())
    return replay, rows, transition


@torch.no_grad()
def target(teacher, case):
    result = c.objective.log_distribution(c.objective.answer_logits(teacher, case), case).cpu().contiguous()
    c.require(result.shape == (len(case['target_ids']), c.VOCAB) and result.dtype == torch.float32
        and torch.isfinite(result).all().item(), 'Full-F target must cover this exact Llama OPD prefix and vocabulary.')
    return result

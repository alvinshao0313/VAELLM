"""One down or three MLP native group8 chunks, preserving the frozen native caches."""
from contextlib import contextmanager
import gc
import selected_bindings as b
from selected_state import optimization_proof
from litebsq.vae_linear_prewarm import prime_named_vae_linear_cache, _build_vae_linear_prewarm_signature


def release_cuda():
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()


def chunks(model, selected, stage):
    result = []
    first = 16 if stage == 'J32' else 8
    for name, module in selected.items():
        signature = _build_vae_linear_prewarm_signature(name=name, base_layer=module, target_dtype=b.torch.bfloat16)
        peers = [(key, v) for key, v in model.named_modules() if isinstance(v, b.core.VAELinear)
            and _build_vae_linear_prewarm_signature(name=key, base_layer=v, target_dtype=b.torch.bfloat16) == signature]
        position = [key for key, _ in peers].index(name)
        chunk = peers[position//8*8:position//8*8+8]
        category = name.rsplit('.', 1)[-1]
        b.require([key for key, _ in chunk] == [f'model.layers.{i}.mlp.{category}' for i in range(first, first+8)],
            'Actual metadata-selected homogeneous native group8 changed.')
        result.append(chunk)
    b.require(len(result) == len(selected) and len({key for chunk in result for key, _ in chunk}) == 8*len(selected),
        'Selected projections need separate complete native signature groups.')
    return result


class NativeOPD:
    def __init__(self, model, selected, active, optimizer, original_caches, native_chunks, flags):
        self.model, self.selected, self.active, self.optimizer = model, selected, active, optimizer
        self.all = {key: v for key, v in model.named_modules() if isinstance(v, b.core.VAELinear)}
        self.frozen = {key: v for key, v in original_caches.items() if key not in selected}
        self.initial_weights = {key: original_caches[key] for key in selected}
        self.chunks, self.native_flags, self.calls = native_chunks, flags, 0
        b.require(len(active) == 6*len(selected) and len(selected) in (1, 3) and len(self.all) == 252
            and len(self.frozen) == 252-len(selected) and len(native_chunks) == len(selected), 'Wrong selected/frozen native inventory.')

    def frozen_caches(self):
        b.require(all(self.all[key]._cached_weight is value and not self.all[key].trainable_decode
            and self.all[key].cache_decoded_weight for key, value in self.frozen.items()), 'Frozen native cache identity changed.')

    @contextmanager
    def native(self, *, exact_values):
        self.frozen_caches()
        before = optimization_proof(self.active, self.optimizer, exact_values=exact_values)
        masters = {key: p.data for key, p in self.active.items()}
        modes = {child: child.training for child in self.model.modules()}
        old_cache = self.model.config.use_cache
        flags = {key: v._parallel_stage_decoder.use_checkpoint for key, v in self.selected.items()}
        evidence = dict(exact_CPU_value_proof=exact_values, native_group8_chunks=[[key for key, _ in chunk] for chunk in self.chunks],
            frozen_native_cache_count=len(self.frozen), FP32_master_storage_retained_without_clone=True)
        try:
            self.model.eval()
            self.model.config.use_cache = True
            with b.torch.no_grad():
                for key, p in self.active.items():
                    p.data = masters[key].bfloat16()
                for key, v in self.selected.items():
                    v.disable_trainable_decode()
                    v._parallel_stage_decoder.use_checkpoint = self.native_flags[key]
                for chunk in self.chunks:
                    warm = prime_named_vae_linear_cache(chunk, dtype=b.torch.bfloat16, clear_existing=True, group_size=8)
                    b.require(warm == dict(total=8, warmed=8, skipped=0, failed=0), 'Original native group8 warm failed.')
                    for key, v in chunk:
                        if key in self.frozen:
                            b.require(b.torch.equal(v._cached_weight, self.frozen[key]), 'Frozen same-group peer value changed.')
                            v._cached_weight = self.frozen[key]
                if self.calls == 0:
                    b.require(all(b.torch.equal(v._cached_weight, self.initial_weights[key]) for key, v in self.selected.items()),
                        'First selected native view differs from the actual stage parent.')
                    evidence['initial_actual_parent_native_exact'] = True
            self.frozen_caches()
            yield evidence
        finally:
            evidence['native_peak_memory'] = b.rollout.memory()
            for v in self.selected.values():
                v.clear_decoded_weight_cache()
            for key, p in self.active.items():
                p.data = masters[key]
            for key, v in self.selected.items():
                v.enable_trainable_decode(parallel_stage_decode=True)
                v._parallel_stage_decoder.use_checkpoint = flags[key]
            for child, training in modes.items():
                child.training = training
            self.model.config.use_cache = old_cache
            self.frozen_caches()
            b.require(optimization_proof(self.active, self.optimizer, exact_values=exact_values) == before
                and getattr(self.model, '_cache', None) is None
                and all(v._cached_weight is None for v in self.selected.values()), 'Native return changed Parameter/FP32storage/gradient/Adam/cache.')
            evidence.update(FP32_master_and_Adam_preserved=True,
                initialized_Adam_leaves=sum(row['Adam_step'] is not None for row in before['parameters'].values()), all252_native_generation=True)
            self.calls += 1

    def collect(self, teacher, tokenizer, cases, record, deadline, *, exact_values):
        policy = b.core.native_policy(teacher)
        b.rollout.move_teacher(teacher, 'cpu', policy)
        release_cuda()
        trajectories, rows = {}, []
        with self.native(exact_values=exact_values) as transition:
            for case in cases:
                b.core.tick(deadline)
                cap = b.LIMITS[case['domain']]
                generated = b.qa.generate_one(self.model, tokenizer, case['prompt_input_ids'].unsqueeze(0),
                    dict(dataset=case['domain'], max_new_tokens=cap), diagnostic=True, protocol=record['model_config']['protocol'])
                options = b.qa.generation_options(tokenizer, case['domain'], len(case['prompt_input_ids']), cap,
                    protocol=record['model_config']['protocol'])
                b.require(generated['generation_options'] == options and generated['forward_trace']['first_calls'][0]['cache_length_before'] == 0
                    and generated['forward_trace']['first_calls'][0]['input_shape'] == [1, len(case['prompt_input_ids'])],
                    'Preserve original full prompt/cap/EOS/fresh KV.')
                replay = b.rollout.replay_case(case, generated, record['model_config']['protocol']['eos_token_id'])
                replay['source_input_signature'] = replay.pop('input_signature')
                trajectories[case['case_id']] = replay
                rows.append(dict(case_id=case['case_id'], domain=case['domain'], prompt_tokens=len(case['prompt_input_ids']),
                    generated_tokens=len(replay['target_ids']), generation=generated))
        b.rollout.move_teacher(teacher, 'cuda:0', policy)
        return trajectories, rows, transition

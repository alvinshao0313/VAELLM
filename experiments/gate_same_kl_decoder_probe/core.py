"""Native fixed-code decoder updates on the same final top100-plus-tail KL."""
from __future__ import annotations

from copy import deepcopy
import gc
import logging
from pathlib import Path
import time

import torch

from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_student_context_probe.core import BANKS, ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import (
    document_batch, native_policy, probability_metrics, selected_logits, version_inventory,
)
from experiments.joint_reconstruction_probe.core import EPS, LR, schema, stage_norm, storage_bits
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

ARMS = ('original', 'alpaca128', 'alpaca64_fineweb64')
STEPS, HORIZON, TOKENS = 2, 500, 4257
DECODER_PREFIX = '_parallel_stage_decoder.'


def fit(model, gate, batch, teacher, *, deadline, observer, steps=STEPS):
    """One actual full-model graph per document; only the six decoder leaves train."""
    require(steps in (STEPS, HORIZON), 'Only the separate two-update mechanism or full500 is supported.')
    initial = records(gate)
    before_bits, before_schema = storage_bits(gate), schema(gate)
    initial_parameters = {name: value.detach().float().clone()
                          for name, value in gate._parallel_stage_decoder.named_parameters()}
    require(len(initial_parameters) == 6 and len(initial) == 13
            and all(p.dtype == torch.bfloat16 for p in gate.parameters()),
            'Each arm needs the same fresh BF16 decoder, not a fitted endpoint.')
    for parameter in gate._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    gate.enable_trainable_decode(parallel_stage_decode=True)
    gate.eval(); ready(gate)
    parameters = dict(gate._parallel_stage_decoder.named_parameters())
    active = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(set(active) == {TARGET + '.' + DECODER_PREFIX + name for name in parameters}
            and all(value.dtype == torch.float32 for value in active.values()),
            'Only the six actual FP32 native decoder parameters may train.')
    # cache_decoded_weight=False chooses the existing differentiable packed path
    # even under no_grad; record its baseline separately from group8 deployment.
    training_before = score(model, batch, teacher, deadline=deadline)
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    previous = {name: value.detach().clone() for name, value in parameters.items()}
    device = next(gate.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    calls = {'count': 0}
    def observe_decode(module, _args):
        require(module.trainable_decode and not module.cache_decoded_weight
                and module._cached_weight is None and module._decoder_compute_dtype == torch.bfloat16,
                'Decoder optimization must use the live native graph, not a detached weight cache.')
        calls['count'] += 1
    hook = gate.register_forward_pre_hook(observe_decode)
    trajectory = []; started = time.time()
    try:
        for step in range(1, steps + 1):
            require(time.time() < deadline, 'Assigned mechanism deadline expired; no retry or shorter population.')
            torch.cuda.synchronize(device); tick = time.time()
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = LR * (1 - (step - 1) / HORIZON)
            rows = []
            for index in range(32):
                require(time.time() < deadline, 'Assigned mechanism deadline expired.')
                item = document_batch(batch, index, device)
                logits = model(**item, use_cache=False).logits
                selected, labels = selected_logits(logits, item)
                require(torch.equal(labels.cpu(), teacher[index]['labels']), 'Teacher causal labels changed.')
                kl, nll = probability_metrics(selected, teacher[index])
                # Accumulate FP32 master gradients of unnormalized token sums,
                # then divide once; do not average unequal-length documents.
                kl.double().sum().backward()
                rows.append({'index': index, 'causal_tokens': len(labels),
                             'kl_sum': float(kl.detach().double().sum()),
                             'nll_sum': float(nll.detach().double().sum())})
                del item, logits, selected, labels, kl, nll
            require(sum(row['causal_tokens'] for row in rows) == TOKENS
                    and all(p.grad is not None and p.grad.dtype == torch.float32
                            and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                    'Incomplete population or missing/nonfinite FP32 decoder gradient.')
            with torch.no_grad():
                for parameter in parameters.values(): parameter.grad.div_(TOKENS)
            gradients = stage_norm([p.grad for p in parameters.values()])
            require(all(value > 0 for value in gradients), 'Both decoder stages need actual nonzero gradients.')
            optimizer.step()
            deltas = stage_norm([p.detach() - previous[name] for name, p in parameters.items()])
            require(all(value > 0 for value in deltas)
                    and all(bool(torch.isfinite(p).all()) for p in parameters.values()),
                    'Both stages must make finite actual FP32 updates.')
            with torch.no_grad():
                for name, value in parameters.items(): previous[name].copy_(value)
            require(all(p.grad is None for name, p in model.named_parameters() if name not in active),
                    'A frozen model parameter received a gradient.')
            torch.cuda.synchronize(device)
            row = {'step': step, 'documents': rows, 'causal_tokens': TOKENS,
                   'kl_mean': sum(x['kl_sum'] for x in rows) / TOKENS,
                   'nll_mean': sum(x['nll_sum'] for x in rows) / TOKENS,
                   'lr': optimizer.param_groups[0]['lr'], 'stage_gradient_l2': gradients,
                   'stage_update_l2_fp32': deltas, 'seconds': time.time() - tick,
                   'allocated_bytes': torch.cuda.memory_allocated(device),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
            trajectory.append(row); observer(row)
        require(calls['count'] == 32 * steps
                and {str(state[key].dtype) for state in optimizer.state.values()
                     for key in ('exp_avg', 'exp_avg_sq')} == {'torch.float32'},
                'Expected32 native trainable decoder forwards per update and FP32 Adam moments.')
    finally:
        hook.remove()
    training_after = score(model, batch, teacher, deadline=deadline)
    peak = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
            'reserved_bytes': torch.cuda.max_memory_reserved(device)}
    delta_fp32 = stage_norm([p.detach() - initial_parameters[name] for name, p in parameters.items()])
    del optimizer, previous
    for parameter in parameters.values():
        parameter.grad = None
        parameter.data = parameter.data.bfloat16()
    gate.disable_trainable_decode(); gate.eval().requires_grad_(False); ready(gate)
    final = records(gate)
    require(all(final[name] == value for name, value in initial.items() if not name.startswith(DECODER_PREFIX))
            and schema(gate) == before_schema and storage_bits(gate) == before_bits,
            'The two banks/five local tensors or persistent storage changed during fitting.')
    delta_bf16 = stage_norm([p.detach().float() - initial_parameters[name] for name, p in parameters.items()])
    return {'status': 'completed', 'steps': steps, 'schedule_steps': HORIZON,
            'initial_payload': initial, 'final_payload': final, 'training_before': training_before,
            'training_after_fp32_master': training_after, 'trajectory': trajectory,
            'actual_trainable_forward_calls': calls['count'], 'all_six_gradients_finite': True,
            'stage_delta_l2_fp32': delta_fp32, 'stage_delta_l2_bf16': delta_bf16,
            'bf16_changed_decoder_tensors': sum(final[name] != value for name, value in initial.items()
                                               if name.startswith(DECODER_PREFIX)),
            'frozen_local_tensors': 7, 'two_banks_and_five_local_exact': True,
            'persistent_bits': before_bits, 'peak': peak, 'seconds': time.time() - started,
            'settings': {'loss_type': 'kl_top_mass', 'top_k': 100, 'temperature': 1,
                         'lr': LR, 'betas': [.9, .95], 'eps': EPS, 'weight_decay': 0,
                         'horizon': HORIZON, 'warmup': 0, 'documents_per_update': 32,
                         'causal_tokens_per_update': TOKENS, 'microbatch': 1, 'padded_length': 256,
                         'reduction': 'Per-document token-sum backward; divide accumulated FP32 master gradients by4257 once before Adam.',
                         'precision': 'FP32 decoder/Adam; existing native BF16 packed decode, stage sum and model forward.'}}


def probe(data, batch, teacher, arm_states, *, work: Path, deadline, observer,
          steps=STEPS, arm=None, observations=None):
    require((steps == STEPS and arm is None and observations is None)
            or (steps == HORIZON and arm in ARMS and observations is not None
                and tuple(observations) == ('alpaca_held32', 'fineweb_rows500000_500031')),
            'Keep the original three-arm mechanism or a separate500-step single arm with fixed endpoint observations.')
    require(tuple(arm_states) == ARMS and len(teacher) == 32
            and tuple(batch['input_ids'].shape) == (32, 256)
            and int(batch['attention_mask'][:, 1:].sum()) == TOKENS,
            'Use all three fixed-code arms and the exact original train32.')
    source = data['source_data']; device = torch.device('cuda:0')
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True,
                                                  expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Require fresh strict f927.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    initial = records(gate)
    require(initial == source['expected_gate_state']
            == {name: tensor_record(value) for name, value in arm_states['original'].items()},
            'The source and original arm differ.')
    template = deepcopy(gate).cpu()
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    warmup_peak = torch.cuda.max_memory_allocated(device)
    policy = native_policy(model, source['model_policy'])
    frozen = version_inventory(model, exclude_gate=True)
    require(len(frozen) == 3410, 'Unexpected frozen complement.')
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj)
               for i in range(36)]
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True,
                                              group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native categorygroup8 refresh failed.')
        return result
    def first_logits():
        with torch.no_grad():
            return model(**document_batch(batch, 0, device), use_cache=False).logits.detach().cpu()
    original_logits = first_logits(); arms = {}
    selected_arms = ARMS if arm is None else (arm,)
    for arm in selected_arms:
        state = arm_states[arm]
        require(records(gate) == initial and time.time() < deadline, 'Each arm starts from restored f927.')
        expected = {name: tensor_record(value) for name, value in state.items()}
        require(set(expected) == set(initial) and all(expected[name] == value for name, value in initial.items()
                                                     if name not in BANKS), 'Only the predeclared initial codes may differ.')
        loaded = gate.load_state_dict(state, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'Local initial state is incomplete.')
        prime(); require(records(gate) == expected, 'Prewarming changed the arm initial state.')
        before = score(model, batch, teacher, deadline=deadline)
        with torch.enable_grad(), torch.autocast('cuda', enabled=False):
            training = fit(model, gate, batch, teacher, deadline=deadline,
                           observer=lambda row: observer(arm, row), steps=steps)
        require(training['initial_payload'] == expected, 'Fitting did not start with the prepared fresh decoder.')
        path = work / (arm + '_gate_state.pt')
        torch.save({name: value.detach().cpu() for name, value in gate.state_dict().items()}, path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        require({name: tensor_record(value) for name, value in saved.items()} == training['final_payload'],
                'Saved BF16 local state differs.')
        fresh = deepcopy(template).to(device)
        loaded = fresh.load_state_dict(saved, strict=True); fresh.eval().requires_grad_(False); ready(fresh)
        require(not loaded.missing_keys and not loaded.unexpected_keys and records(fresh) == training['final_payload'],
                'Fresh native local strict reload differs.')
        with torch.no_grad():
            deployed = gate._decode_weight(dtype=torch.bfloat16)
            reloaded = fresh._decode_weight(dtype=torch.bfloat16)
            require(torch.equal(deployed, reloaded), 'Fresh local BF16 full-weight decode differs.')
        del fresh, deployed, reloaded, saved
        gc.collect(); torch.cuda.empty_cache()
        prime(); after = score(model, batch, teacher, deadline=deadline)
        require(records(gate) == training['final_payload']
                and version_inventory(model, exclude_gate=True) == frozen
                and native_policy(model, source['model_policy']) == policy,
                'A frozen complement/RoPE value changed.')
        arms[arm] = {'training': training, 'before_deployment': before, 'after_bf16_deployment': after,
                     'saved': {'path': str(path), 'sha256': sha256(path), 'tensor_count': 13,
                               'fresh_strict_local_state_exact': True, 'native_full_weight_reload_exact': True},
                     'frozen_complement_count': 3410, 'frozen_complement_versions_and_pointers_exact': True}
        if observations is not None:
            # These previously observed documents are used only after the fixed
            # final update and BF16 deployment, never to stop or select a step.
            arms[arm]['endpoint_observations'] = {
                name: score(model, item[0], item[1], deadline=deadline)
                for name, item in observations.items()}
            require(records(gate) == training['final_payload']
                    and version_inventory(model, exclude_gate=True) == frozen,
                    'Endpoint observations changed persistent state.')
        gate.load_state_dict(arm_states['original'], strict=True); gate.eval().requires_grad_(False); prime()
        require(records(gate) == initial and torch.equal(first_logits(), original_logits)
                and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                'Original13 state, full logits or frozen gradient flags were not restored.')
        arms[arm]['original_state_and_full_logits_restored_exact'] = True
    return {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'arms': arms, 'warmup': warmup,
            'warmup_peak_allocated_bytes': warmup_peak, 'native_policy': policy,
            'boundary': ('Three independent two-update mechanism runs;500-step schedule only. No500-step endpoint, heldout selection, code update, full8 or framework integration.'
                         if steps == STEPS else
                         'One fresh fixed-code500-update arm; oldheld/FineWeb observations only at the BF16 endpoint. No parameter selection, code update, full8 or framework integration.')}

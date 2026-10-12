"""One fixed Llama projection, native packed endpoints, and whole-model observations."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
import time
import weakref

import torch
from torch.nn import functional as F

from experiments.gate_student_context_evaluation.deploy import native_policy
from experiments.gate_tail_objective_probe.core import (
    document_batch, probability_metrics, selected_logits, teacher_statistics,
)
from experiments.linear_output.calibration import capture_inputs
from experiments.longbench_v1.common import write_json
from experiments.llama_column_feedback_probe.algorithm import ARMS, scalar_ldl, search_columns
from experiments.llama_column_feedback_probe.model import (
    BANKS, SOURCE_ID, TARGET, NativeStageOracle, apply_choices, inventory, read_codes, validate,
)
from experiments.qwen3_w25.block_ldlq import build_noncentered_hessian, prepare_protected_objective
from experiments.qwen3_w25.block_ldlq_model import rotate_hessian
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from rotation.model_utils import get_model
from tools.cat_eval import _prepare_model_for_eval
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint


def records(module):
    return {key: tensor_record(value) for key, value in module.state_dict().items()}


def aggregate(rows):
    causal = sum(row['causal_tokens'] for row in rows)
    sse = sum(row['local_sse'] for row in rows)
    energy = sum(row['local_reference_energy'] for row in rows)
    return {'documents': rows, 'causal_tokens': causal,
            'kl_sum': sum(row['kl_sum'] for row in rows), 'kl_mean': sum(row['kl_sum'] for row in rows) / causal,
            'nll_sum': sum(row['nll_sum'] for row in rows), 'nll_mean': sum(row['nll_sum'] for row in rows) / causal,
            'local_sse': sse, 'local_reference_energy': energy, 'local_nmse': sse / energy,
            'local_document_macro_nmse': sum(row['local_nmse'] for row in rows) / len(rows)}


@torch.no_grad()
def teacher_pass(record, batches, *, device, deadline, observer):
    model = get_model(record['hf_source']['path'], None).eval().requires_grad_(False)
    model.to(device); model.config.use_cache = False
    policy = native_policy(model, record['model_policy'])
    before = inventory(model)
    reference = model.get_submodule(TARGET).weight.detach().cpu().clone()
    require(reference.dtype == torch.bfloat16 and tuple(reference.shape) == (4096, 14336), 'Invalid original HF down matrix.')
    cache = {}; started = time.time()
    for domain, batch in batches.items():
        cache[domain] = []
        for index in range(len(batch['input_ids'])):
            require(time.time() < deadline, 'Teacher pass exceeded the hard deadline.')
            item = document_batch(batch, index, device)
            logits = model(**item, use_cache=False).logits
            selected, labels = selected_logits(logits, item)
            stats = teacher_statistics(selected, labels)
            if index == 0:
                kl, nll = probability_metrics(selected, stats)
                require(torch.equal(kl, torch.zeros_like(kl)) and torch.equal(nll.cpu(), stats['nll']),
                        'Cached native teacher statistics failed self KL/NLL replay.')
                del kl, nll
            cache[domain].append(stats)
            del logits, selected, labels, stats, item
        observer({'stage': 'teacher', 'domain': domain, 'documents': len(cache[domain])})
    require(inventory(model) == before and native_policy(model, record['model_policy']) == policy,
            'Frozen HF teacher persistent state or native FP32 RoPE changed.')
    report = {'seconds': time.time() - started, 'native_policy': policy, 'frozen_state_unchanged': True,
              'reference_weight': tensor_record(reference), 'whole_model_dtype_cast': False,
              'cached_teacher_bytes': sum(t.numel() * t.element_size() for docs in cache.values() for s in docs for t in s.values())}
    ref = weakref.ref(model)
    del model
    gc.collect(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    require(ref() is None, 'Teacher model survived release before student load.')
    report['teacher_model_released'] = True
    return cache, reference, report


@torch.no_grad()
def score(model, layer, reference, batches, teacher, prefix_hashes, *, original, deadline, observer, arm):
    device = next(model.parameters()).device
    results = {}
    for domain, batch in batches.items():
        rows = []
        for index in range(len(batch['input_ids'])):
            require(time.time() < deadline, 'Whole-model observation exceeded hard deadline.')
            item = document_batch(batch, index, device)
            observed = {}
            def hook(_module, args, output):
                x = args[0]
                mask = item['attention_mask'].bool()
                require(tuple(x.shape) == (1, 256, 14336) and output.shape == (1, 256, 4096),
                        'Native down input/output shape differs.')
                digest = tensor_record(x)['sha256']
                key = (domain, index)
                if original:
                    prefix_hashes[key] = digest
                else:
                    require(prefix_hashes[key] == digest, 'Only-down change altered its upstream input.')
                target = F.linear(x, reference)
                error = (output[mask].float() - target[mask].float()).double()
                observed.update(input_sha256=digest, local_sse=float(error.square().sum()),
                                local_reference_energy=float(target[mask].double().square().sum()), calls=1)
            handle = layer.register_forward_hook(hook)
            try:
                logits = model(**item, use_cache=False).logits
            finally:
                handle.remove()
            require(observed.get('calls') == 1, 'Native target Linear was not observed exactly once.')
            selected, labels = selected_logits(logits, item)
            require(torch.equal(labels.cpu(), teacher[domain][index]['labels']), 'Teacher/student causal labels differ.')
            kl, nll = probability_metrics(selected, teacher[domain][index])
            rows.append({'index': index, 'valid_tokens': int(item['attention_mask'].sum()), 'causal_tokens': len(labels),
                         'kl_sum': float(kl.double().sum()), 'nll_sum': float(nll.double().sum()),
                         'local_nmse': observed['local_sse'] / observed['local_reference_energy'],
                         **{key: value for key, value in observed.items() if key != 'calls'}})
            del logits, selected, labels, kl, nll, item
        results[domain] = aggregate(rows)
        observer({'stage': 'score', 'arm': arm, 'domain': domain,
                  **{key: results[domain][key] for key in ('causal_tokens', 'kl_mean', 'nll_mean', 'local_nmse')}})
    return results


def compare(results):
    comparisons = {}
    for left, right in [(arm, 'original') for arm in ARMS] + [('per_column', 'delayed16'), ('per_column', 'off')]:
        domains = {}
        for domain, actual in results[left].items():
            control = results[right][domain]
            require(actual['causal_tokens'] == control['causal_tokens'], 'Comparison populations differ.')
            row = {key + '_delta': actual[key] - control[key] for key in ('kl_mean', 'nll_mean', 'local_nmse')}
            row['documents'] = [{'index': a['index'], 'causal_tokens': a['causal_tokens'],
                'kl_mean_delta': (a['kl_sum'] - b['kl_sum']) / a['causal_tokens'],
                'nll_mean_delta': (a['nll_sum'] - b['nll_sum']) / a['causal_tokens'],
                'local_nmse_delta': a['local_nmse'] - b['local_nmse']}
                for a, b in zip(actual['documents'], control['documents'])]
            domains[domain] = row
        comparisons[left + '_minus_' + right] = domains
    return comparisons


@torch.no_grad()
def measure(record, batches, teacher, reference, *, device, deadline, observer):
    model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], map_location='cpu',
                                                  strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Fresh original e767 strict load is required.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    layer = model.get_submodule(TARGET)
    require(records(layer) == record['expected_target_state'], 'Actual original L16 down bytes differ.')
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    validate(layer)
    policy = native_policy(model, record['model_policy'])
    frozen = inventory(model)
    require(len(frozen) == record['source_serialized_tensors'] - 2, 'Unexpected frozen complement inventory.')
    original_state = records(layer)
    banks = {key: layer.state_dict()[key].detach().clone() for key in BANKS}
    baseline_weight = layer._cached_weight.detach().clone()
    original_codes = read_codes(layer)
    original_rotated = layer._decode_compressed_weight(dtype=torch.bfloat16).detach().clone()
    reference = reference.to(device)
    prefix_hashes = {}
    results = {'original': score(model, layer, reference, batches, teacher, prefix_hashes,
                                original=True, deadline=deadline, observer=observer, arm='original')}
    first = document_batch(batches['alpaca_hold32'], 0, device)
    first_logits = model(**first, use_cache=False).logits.detach().cpu()
    inputs = capture_inputs(model, TARGET, batches['train'], microbatch_size=1)
    require(inputs.dtype == torch.bfloat16 and inputs.shape == (int(batches['train']['attention_mask'].sum()), 14336),
            'Calibration capture must contain every valid training token in original row order.')
    capture_record = tensor_record(inputs)
    hessian = build_noncentered_hessian(inputs, chunk_size=512, device=device)
    del inputs
    protected = layer.protected_input_indices
    mask = torch.ones(14336, dtype=torch.bool, device=device); mask[protected] = False
    compressed = mask.nonzero().flatten()
    protected_weight = layer._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=device).T.double()
    objective = prepare_protected_objective(reference.double(), protected_weight, hessian, compressed, protected,
                                           damping_ratio=record['protocol']['damping_ratio'])
    target = layer.weight_rotation(objective.target)
    rotated_hessian = rotate_hessian(objective.hessian, layer.weight_rotation)
    damping = objective.damping
    lower, diagonal = scalar_ldl(rotated_hessian)
    del objective, hessian, rotated_hessian, protected_weight
    observer({'stage': 'hessian', 'calibration_tokens': capture_record['shape'][0], 'damping': damping})
    oracle = NativeStageOracle(layer)
    choices, reconstructions, oracle_original, proxies = search_columns(target, lower, diagonal, original_codes, oracle,
        deadline=deadline, observer=lambda row: observer({'stage': 'search', **row}))
    require(torch.equal(oracle_original, original_rotated), 'Full original transposed oracle differs from native BF16 decoding.')
    del target, lower, diagonal, oracle, oracle_original, original_rotated
    down_targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.down_proj', model.model.layers[i].mlp.down_proj) for i in range(32)]
    def prime():
        refresh_vae_linear_runtime_after_state_load(layer)
        stats = prime_named_vae_linear_cache(down_targets, dtype=torch.bfloat16, clear_existing=True,
                                            group_size=8, compute_device=device)
        require(stats['warmed'] == 32 and stats['failed'] == 0, 'Native complete down category group8 rebuild failed.')
        return stats
    endpoints, endpoint_records, cache_reports = {}, {}, {}
    try:
        for arm in ARMS:
            apply_choices(layer, original_codes, choices[arm])
            native_rotated = layer._decode_compressed_weight(dtype=torch.bfloat16)
            require(torch.equal(native_rotated, reconstructions[arm]), 'Selected finite candidates do not replay native packed decoder: ' + arm)
            del native_rotated
            cache_reports[arm] = prime()
            current = records(layer)
            require(all(current[key] == value for key, value in original_state.items() if key not in BANKS),
                    'A decoder/protection/rotation tensor changed.')
            actual = [int(torch.count_nonzero(layer.get_stage_part_vq_weight(stage).reshape_as(original_codes[stage])
                                             != original_codes[stage])) for stage in range(2)]
            require(actual == proxies[arm]['stage_xor'], 'Actual packed-bank XOR differs from selected single-bit choices.')
            results[arm] = score(model, layer, reference, batches, teacher, prefix_hashes,
                                 original=False, deadline=deadline, observer=observer, arm=arm)
            require(inventory(model) == frozen and native_policy(model, record['model_policy']) == policy,
                    'A frozen persistent tensor or native policy changed.')
            endpoints[arm] = {key: layer.state_dict()[key].detach().cpu().clone() for key in BANKS}
            endpoint_records[arm] = {key: tensor_record(value) for key, value in endpoints[arm].items()}
        write_json(Path(record['output_dir']) / 'diagnostic.json', {
            'status': 'observations_complete_pending_final_contracts', 'source_checkpoint_id': SOURCE_ID,
            'target': TARGET, 'proxy': proxies, 'results': results, 'comparisons': compare(results),
            'endpoint_state': endpoint_records, 'frozen_tensors': len(frozen),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'native_short_probe_verified': False})
        # The auditor accepts CPU tensors. Frozen-state checks above establish
        # that only these two actual CPU banks differ from the source mmap.
        source_cpu = torch.load(Path(record['source_checkpoint']) / meta['state_dict_file'],
                                map_location='cpu', weights_only=True, mmap=True)
        for arm, endpoint in endpoints.items():
            patched = dict(source_cpu)
            patched.update({TARGET + '.' + key: value for key, value in endpoint.items()})
            budget = audit_state_dict(meta, patched, max_bpw=2.5, max_lora_rank=0)
            require(budget == record['source_bit_budget'], 'A code-only endpoint changed actual storage: ' + arm)
        del source_cpu, patched
    finally:
        for key, value in banks.items():
            layer.state_dict()[key].copy_(value)
        restored_cache = prime()
    require(records(layer) == original_state and torch.equal(layer._cached_weight, baseline_weight)
            and torch.equal(model(**first, use_cache=False).logits.detach().cpu(), first_logits),
            'Restoring original banks failed the persistent/cache/full-logit roundtrip.')
    require(inventory(model) == frozen and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'The full frozen-state or no-gradient contract failed.')
    path = Path(record['output_dir']) / 'endpoints.pt'
    require(not path.exists(), 'Never overwrite a previous endpoint archive.')
    torch.save(endpoints, path)
    replay = torch.load(path, map_location='cpu', weights_only=True)
    require({arm: {key: tensor_record(value) for key, value in state.items()} for arm, state in replay.items()} == endpoint_records,
            'Persisted endpoint packed banks differ from actual measured endpoints.')
    return {'status': 'completed_native_short_probe', 'source_checkpoint_id': SOURCE_ID, 'target': TARGET,
        'calibration_inputs': capture_record, 'damping': damping, 'proxy': proxies, 'results': results,
        'comparisons': compare(results), 'endpoint_state': endpoint_records, 'source_bit_budget': budget,
        'source_target_state': original_state, 'source_full_tensors': record['source_serialized_tensors'],
        'frozen_tensors': len(frozen), 'frozen_complement_versions_exact': True, 'target_noncode_bytes_exact': True,
        'full_original_and_three_endpoint_oracles_equal_native': True, 'persisted_packed_endpoints_replay_exact': True,
        'source_banks_cache_and_full_logits_restored_exact': True, 'prefix_inputs_exact': True,
        'native_policy': policy, 'warmup': warmup, 'endpoint_cache': cache_reports, 'restore_cache': restored_cache,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'eight_task_or_longbench_evaluation_performed': False, 'local_proxy_alone_is_success': False}

"""Frozen native Qwen forwards: conditional MLP error versus tail probabilities."""
from __future__ import annotations

from copy import deepcopy
import gc
import logging
import time
import weakref

import torch
from torch.nn import functional as F

from experiments.gate_student_context_evaluation.deploy import native_policy
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import TARGET, SOURCE_ID
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from rotation.model_utils import get_qwen3
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

ARMS = ('f927', 'original_codes500', 'group_accept500', 'conditional_hf_mlp')
TOP_K = 100


def version_inventory(model, *, exclude_gate=False):
    # Existing strict source/deployment proofs cover bytes. Here there is no optimizer;
    # versions/pointers detect writes or replacement outside the thirteen patched tensors.
    return {k: [v.data_ptr(), v._version, list(v.shape), str(v.dtype), str(v.device)]
            for k, v in model.state_dict().items()
            if not (exclude_gate and k.startswith(TARGET + '.'))}


def teacher_statistics(logits, labels):
    scaled = logits.detach().float()  # T=1, exactly as native kl_top_mass.
    values, indices = scaled.topk(TOP_K, dim=-1, sorted=False)
    return {'top_values': values.cpu(), 'top_indices': indices.cpu(),
            'log_z': torch.logsumexp(scaled, dim=-1, keepdim=True).cpu(),
            'labels': labels.detach().cpu(),
            'nll': F.cross_entropy(scaled, labels, reduction='none').cpu()}


def probability_metrics(logits, stats):
    """Native top100-plus-tail KL algebra with cached full-teacher normalization."""
    device = logits.device
    student = logits.float()
    indices = stats['top_indices'].to(device)
    teacher_top_log = stats['top_values'].to(device) - stats['log_z'].to(device)
    student_top_log = student.gather(-1, indices) - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_top, student_top = teacher_top_log.exp(), student_top_log.exp()
    top_term = (teacher_top * (teacher_top_log - student_top_log)).sum(-1)
    teacher_other = (1.0 - teacher_top.sum(-1)).clamp(0.0, 1.0)
    student_other = (1.0 - student_top.sum(-1)).clamp(0.0, 1.0)
    tiny = torch.finfo(student_other.dtype).tiny
    other_term = torch.where(teacher_other > 0.0,
        teacher_other * (teacher_other.clamp_min(tiny).log() - student_other.clamp_min(tiny).log()),
        torch.zeros_like(teacher_other))
    kl = top_term + other_term
    nll = F.cross_entropy(student, stats['labels'].to(device), reduction='none')
    require(bool(torch.isfinite(kl).all() and torch.isfinite(nll).all()), 'Nonfinite forward scores.')
    return kl, nll


def selected_logits(logits, batch):
    valid = batch['attention_mask'][0, 1:].bool()
    labels = batch['input_ids'][0, 1:][valid]
    require(len(labels) > 0 and logits.shape[:2] == (1, 256), 'Require full fixed256, causal next-token scoring.')
    return logits[0, :-1][valid], labels


def document_batch(batch, index, device):
    return {key: batch[key][index:index + 1].to(device) for key in ('input_ids', 'attention_mask')}


def aggregate(documents):
    tokens = sum(row['causal_tokens'] for row in documents)
    sse = sum(row['local_sse'] for row in documents)
    energy = sum(row['local_reference_energy'] for row in documents)
    return {'documents': documents, 'causal_tokens': tokens,
            'kl_top100_mass_sum': sum(row['kl_top100_mass_sum'] for row in documents),
            'kl_top100_mass_mean': sum(row['kl_top100_mass_sum'] for row in documents) / tokens,
            'nll_sum': sum(row['nll_sum'] for row in documents),
            'nll_mean': sum(row['nll_sum'] for row in documents) / tokens,
            'local_sse': sse, 'local_reference_energy': energy, 'local_nmse': sse / energy,
            'local_document_macro_nmse': sum(row['local_nmse'] for row in documents) / len(documents)}


@torch.no_grad()
def teacher_pass(data, batches, archive, *, device, deadline, observer):
    started = time.time()
    source = data['source_data']
    model = get_qwen3(source['model_path'], None).eval().requires_grad_(False)
    model.to(device)  # Device only: never round native FP32 RoPE with a model dtype cast.
    model.config.use_cache = False
    policy = native_policy(model, source['model_policy'])
    inventory = version_inventory(model)
    model_ref = weakref.ref(model)
    parameter_refs = [weakref.ref(value) for value in model.parameters()]
    layer = model.model.layers[18].mlp
    for name, expected in (('gate_proj', archive['targets'][TARGET]['reference_weight']),
                           ('up_proj', archive['up_proj_weight']),
                           ('down_proj', archive['targets'][TARGET.replace('gate_proj', 'down_proj')]['reference_weight'])):
        require(torch.equal(getattr(layer, name).weight.detach().cpu(), expected), 'Actual native HF MLP differs from its archive: ' + name)
    hf_mlp = deepcopy(layer).cpu().eval().requires_grad_(False)
    cache, rows = {}, {}
    torch.cuda.reset_peak_memory_stats(device)
    for domain, batch in batches.items():
        cache[domain], rows[domain] = [], []
        for index in range(32):
            require(time.time() < deadline, 'Forward diagnostic exceeded the assigned deadline.')
            item = document_batch(batch, index, device)
            logits = model(**item, use_cache=False).logits
            selected, labels = selected_logits(logits, item)
            stats = teacher_statistics(selected, labels)
            cache[domain].append(stats)
            self_kl, self_nll = probability_metrics(selected, stats)
            require(torch.equal(self_kl, torch.zeros_like(self_kl)), 'Cached teacher distribution does not reproduce its own native mass KL.')
            row = {'index': index, 'causal_tokens': len(labels), 'nll_sum': float(stats['nll'].double().sum()),
                   'nll_mean': float(stats['nll'].double().mean()),
                   'statistics': {k: tensor_record(v) for k, v in stats.items()}}
            rows[domain].append(row); observer('teacher', domain, index, row)
            del logits, selected, labels, stats, self_kl, self_nll, item
    require(version_inventory(model) == inventory and native_policy(model, source['model_policy']) == policy,
            'Teacher forward changed persistent state or native FP32 RoPE.')
    report = {'documents': rows, 'native_rope': policy, 'persistent_versions_unchanged': True,
              'actual_hf_three_matrices_equal_archive': True, 'cache_full_vocab_logits': False,
              'cached_tensor_bytes': sum(v.numel() * v.element_size() for docs in cache.values() for d in docs for v in d.values()),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'seconds': time.time() - started}
    del layer, model
    gc.collect(); torch.cuda.empty_cache()
    report.update(teacher_model_weakref_dead=model_ref() is None,
                  teacher_parameter_count=len(parameter_refs),
                  teacher_parameter_refs_alive=sum(value() is not None for value in parameter_refs),
                  conditional_hf_mlp_cpu=all(value.device.type == 'cpu' for value in hf_mlp.parameters()),
                  allocated_after_teacher_release_before_workspace_clear=torch.cuda.memory_allocated(device))
    require(report['teacher_model_weakref_dead'] and report['teacher_parameter_refs_alive'] == 0
            and report['conditional_hf_mlp_cpu'], 'A full-teacher object or parameter survived release.')
    # The one-document native diagnostic identified an 8,519,680-byte cuBLAS
    # workspace after all teacher tensors died; empty_cache alone cannot free it.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    report['allocated_after_teacher_release'] = torch.cuda.memory_allocated(device)
    report['reserved_after_workspace_release'] = torch.cuda.memory_reserved(device)
    require(report['allocated_after_teacher_release'] == 0, 'Release the full teacher before loading the student.')
    return cache, hf_mlp, report


@torch.no_grad()
def student_pass(data, batches, teacher, hf_mlp, endpoints, *, device, deadline, observer):
    source = data['source_data']; started = time.time()
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True,
                                                  expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Require a fresh strict f927 student.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    original = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    require(records(gate) == source['expected_gate_state'], 'Original local gate state differs.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    rope = native_policy(model, source['model_policy'])
    frozen = version_inventory(model, exclude_gate=True)
    require(len(frozen) == 3410, 'The frozen complement is not 3410 tensors.')
    gate_targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj)
                    for i in range(36)]
    baseline_weight = gate._cached_weight.detach().clone()
    hf_mlp.to(device)
    hf_before = version_inventory(hf_mlp)
    layer = model.model.layers[18]
    contexts, reports = {}, {}
    first_logits_record = None
    for arm in ARMS:
        state = original if arm in ('f927', 'conditional_hf_mlp') else endpoints[arm]
        loaded = gate.load_state_dict(state, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'Incomplete local endpoint load.')
        ready(gate)
        # The full evaluator packs each category in layer order, eight at a time.
        # Re-prime this entire category to preserve the exact original batch geometry.
        prime = prime_named_vae_linear_cache(gate_targets, dtype=torch.bfloat16, clear_existing=True,
                                            group_size=8, compute_device=device)
        require(prime['warmed'] == 36 and prime['failed'] == 0, 'Native category cache re-prime failed.')
        require(records(gate) == {k: tensor_record(v) for k, v in state.items()}, 'The loaded endpoint changed during native cache preparation.')
        if arm in ('f927', 'conditional_hf_mlp'):
            require(torch.equal(gate._cached_weight, baseline_weight), 'Restored f927 native group8 gate differs.')
        results = {}
        for domain, batch in batches.items():
            documents = []
            for index in range(32):
                require(time.time() < deadline, 'Forward diagnostic exceeded the assigned deadline.')
                item = document_batch(batch, index, device)
                key = (domain, index); current = {}
                def before_norm(_module, args):
                    current['residual'] = args[0]
                def mlp_output(_module, args, output):
                    x = args[0]
                    if arm == 'f927':
                        contexts[key] = {'input': x.detach().cpu(), 'target': hf_mlp(x).detach().cpu()}
                    else:
                        require(torch.equal(x.detach().cpu(), contexts[key]['input']), 'Changing only L18 gate altered its upstream input.')
                    target = contexts[key]['target'].to(device)
                    emitted = target if arm == 'conditional_hf_mlp' else output
                    valid = item['attention_mask'].bool()
                    error = (emitted[valid].float() - target[valid].float()).double()
                    energy = target[valid].double().square().sum()
                    current.update(local_sse=float(error.square().sum()), local_reference_energy=float(energy),
                                   emitted=emitted, hook_called=1,
                                   oracle_delta_sse=float((target[valid].float() - output[valid].float()).double().square().sum()))
                    if arm == 'conditional_hf_mlp':
                        require(current['local_sse'] == 0, 'The oracle must emit exactly the conditional native HF output.')
                        return emitted
                def after_block(_module, _args, output):
                    require(torch.equal(output[0], current['residual'] + current['emitted']),
                            'The native block must add its residual exactly once after the observed/replaced MLP.')
                    current['residual_once_exact'] = True
                hooks = [layer.post_attention_layernorm.register_forward_pre_hook(before_norm),
                         layer.mlp.register_forward_hook(mlp_output), layer.register_forward_hook(after_block)]
                try:
                    logits = model(**item, use_cache=False).logits
                finally:
                    for handle in hooks: handle.remove()
                require(current['hook_called'] == 1 and current['residual_once_exact'], 'The native L18 hook did not execute.')
                selected, labels = selected_logits(logits, item)
                stats = teacher[domain][index]
                require(torch.equal(labels.cpu(), stats['labels']), 'Teacher/student causal labels differ.')
                kl, nll = probability_metrics(selected, stats)
                row = {'index': index, 'valid_tokens': int(item['attention_mask'].sum()), 'causal_tokens': len(labels),
                       'input_sha256': tensor_record(contexts[key]['input'])['sha256'],
                       'local_sse': current['local_sse'], 'local_reference_energy': current['local_reference_energy'],
                       'local_nmse': current['local_sse'] / current['local_reference_energy'],
                       'kl_top100_mass_sum': float(kl.double().sum()), 'kl_top100_mass_mean': float(kl.double().mean()),
                       'nll_sum': float(nll.double().sum()), 'nll_mean': float(nll.double().mean()),
                       'oracle_replacement_sse': current['oracle_delta_sse'] if arm == 'conditional_hf_mlp' else None}
                documents.append(row); observer(arm, domain, index, row)
                if arm == 'f927' and key == (next(iter(batches)), 0):
                    first_logits_record = tensor_record(logits)
                del logits, selected, labels, kl, nll, current, item
            results[domain] = aggregate(documents)
        require(version_inventory(model, exclude_gate=True) == frozen and native_policy(model, source['model_policy']) == rope,
                'A non-target persistent tensor or native policy changed.')
        reports[arm] = {'domains': results, 'local_state': records(gate), 'frozen_other_tensors': 3410,
                        'prefix_inputs_exact': True, 'native_residual_once_exact': True, 'cache_prewarm': prime}
    gate.load_state_dict(original, strict=True); ready(gate)
    prime_named_vae_linear_cache(gate_targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device=device)
    probe = document_batch(batches[next(iter(batches))], 0, device)
    restored = model(**probe, use_cache=False).logits
    require(tensor_record(restored) == first_logits_record and records(gate) == source['expected_gate_state']
            and version_inventory(model, exclude_gate=True) == frozen and version_inventory(hf_mlp) == hf_before,
            'Removing all interventions and restoring thirteen source tensors failed the native forward replay.')
    require(all(not p.requires_grad for p in model.parameters()), 'This diagnostic must not enable training.')
    return {'arms': reports, 'warmup': warmup, 'native_rope': rope,
            'initial_global_and_category_only_gate_cache_exact': True,
            'source_gate_restore_exact': True, 'source_first_full_logits_restore_exact': True,
            'frozen_complement_versions_and_pointers_exact': True, 'oracle_module_frozen': True,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'seconds': time.time() - started}

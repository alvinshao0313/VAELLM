"""Shared top-mass teacher cache and token-weighted whole-model proposal KL."""
from contextlib import contextmanager
import math
import random
import time

import numpy as np
import torch

from .contracts import digest, require
from .data import INPUT_FIELDS, MIX, ROW_KEY, batch_digest, tensor_digest

LOSS_CONTRACT = dict(loss_type='kl_top_mass', top_k=100, temperature=1.,
    prompt_loss_weight=.3, causal_shift=True,
    aggregation='pooled effective tokens; source weights appear only in record sampling')


@contextmanager
def preserved_eval(*models):
    """Restore every module mode and RNG; do not consume a training random stream."""
    modes = [(module, module.training) for model in models for module in model.modules()]
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        # Parent-first train() invokes native cache invalidation; restore child overrides afterwards.
        for module, mode in modes:
            module.train(mode)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def proposal_batches(samples, tokenizer):
    from train_utils.distill_data import build_distill_data_collator

    require(len(samples) == 512, 'Score the root-frozen common512 records.')
    collate = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
    batches = []
    for start in range(0, len(samples), 4):
        rows = samples[start:start + 4]
        inputs = collate([{key: value for key, value in row.items() if key != ROW_KEY} for row in rows])
        refs = [row[ROW_KEY] for row in rows]
        batches.append(dict(inputs=inputs, records=refs, input_sha256=batch_digest(inputs)))
    return batches


def pack_teacher(logits):
    require(logits.ndim == 3 and logits.shape[1] >= 2 and torch.isfinite(logits).all().item(),
        'Teacher logits must be finite [B,L,V], with a causal next-token position.')
    shifted = logits[:, :-1].detach().float()
    values, indices = shifted.topk(min(100, int(shifted.shape[-1])), dim=-1, sorted=False)
    logp = values - torch.logsumexp(shifted, dim=-1, keepdim=True)
    target = dict(indices=indices.cpu().contiguous(), top_logp=logp.cpu().contiguous(),
        vocab_size=int(shifted.shape[-1]), original_shape=list(logits.shape),
        original_dtype=str(logits.dtype))
    target['tensor_sha256'] = tensor_digest({key: target[key] for key in ('indices', 'top_logp')})
    return target


def cached_token_kl(student_logits, target):
    """Same FP32 full-vocabulary normalizers and outside bin as distill_loss_core."""
    require(list(student_logits.shape) == target['original_shape']
        and student_logits.shape[-1] == target['vocab_size'], 'Teacher cache and student logits differ in shape.')
    require(torch.isfinite(student_logits).all().item(), 'Non-finite student proposal logits.')
    require(tensor_digest({key: target[key] for key in ('indices', 'top_logp')}) == target['tensor_sha256'],
        'Teacher cache tensors changed after they were bound.')
    shifted = student_logits[:, :-1].float()
    indices = target['indices'].to(shifted.device)
    teacher_logp = target['top_logp'].to(shifted.device)
    student_logp = shifted.gather(-1, indices) - torch.logsumexp(shifted, dim=-1, keepdim=True)
    teacher_prob, student_prob = teacher_logp.exp(), student_logp.exp()
    top = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    if indices.shape[-1] == shifted.shape[-1]:
        return top
    teacher_other = (1. - teacher_prob.sum(dim=-1)).clamp(0., 1.)
    student_other = (1. - student_prob.sum(dim=-1)).clamp(0., 1.)
    tiny = torch.finfo(student_other.dtype).tiny
    other = torch.where(teacher_other > 0.,
        teacher_other * (teacher_other.clamp_min(tiny).log() - student_other.clamp_min(tiny).log()),
        torch.zeros_like(teacher_other))
    return top + other


def weighted_rows(token_kl, inputs):
    from train_utils.distill_loss_core import build_prediction_token_masks

    response, prompt = build_prediction_token_masks(labels=inputs['labels'], attention_mask=inputs['attention_mask'])
    weights = (response + .3 * prompt).to(token_kl.device)
    require(tuple(token_kl.shape) == tuple(weights.shape), 'Use the original causal/mask alignment.')
    require(torch.isfinite(token_kl).all().item(), 'Non-finite proposal token KL.')
    return (token_kl.float() * weights).sum(dim=1), weights.sum(dim=1)


def _model_logits(model, inputs):
    from e2e_common.dense_loss import get_output_logits

    device = model.get_input_embeddings().weight.device
    require(not (set(inputs) - set(INPUT_FIELDS)), 'Unsupported proposal model inputs must not be silently discarded.')
    model_inputs = {key: value.to(device) for key, value in inputs.items() if key != 'labels'}
    return get_output_logits(model(**model_inputs, use_cache=False))


def build_teacher_cache(teacher, batches, *, teacher_identity, data_identity):
    require(teacher_identity and isinstance(data_identity, str) and data_identity,
        'Bind teacher weights/tokenizer and the exact root-frozen data manifest.')
    entries, began = [], time.monotonic()
    with preserved_eval(teacher):
        for batch in batches:
            require(batch_digest(batch['inputs']) == batch['input_sha256'], 'Proposal tokens changed before teacher caching.')
            target = pack_teacher(_model_logits(teacher, batch['inputs']))
            entries.append(dict(input_sha256=batch['input_sha256'], target=target,
                record_ids=[row['record_id'] for row in batch['records']]))
    manifest = dict(schema='cat_auto_top_mass_teacher_v0', teacher_identity=teacher_identity,
        data_identity=data_identity, loss_contract=LOSS_CONTRACT,
        batches=[dict(input_sha256=row['input_sha256'], record_ids=row['record_ids'],
            target_sha256=row['target']['tensor_sha256'], shape=row['target']['original_shape'],
            vocab_size=row['target']['vocab_size'], original_dtype=row['target']['original_dtype']) for row in entries])
    return dict(manifest=manifest, cache_identity=digest(manifest), entries=entries,
        elapsed_seconds=time.monotonic() - began,
        tensor_bytes=sum(row['target'][key].numel() * row['target'][key].element_size()
            for row in entries for key in ('indices', 'top_logp')))


def _summary(rows):
    numerator = sum(row['numerator'] for row in rows)
    denominator = sum(row['denominator'] for row in rows)
    require(denominator > 0., 'A complete proposal half must contain effective causal tokens.')
    return dict(KL=numerator / denominator, numerator=numerator,
        effective_token_weight=denominator, records=len(rows))


def aggregate_rows(rows):
    require(len(rows) == 512 and len({row['record_id'] for row in rows}) == 512,
        'Observe all512 distinct frozen proposal records in their fixed order.')
    require(all(row['alias'] in MIX for row in rows), 'Unexpected source in the fixed five-source proposal pool.')
    per_source = {}
    for alias, weight in MIX.items():
        matching = [row for row in rows if row['alias'] == alias]
        per_source[alias] = dict(sampling_weight=weight, measured_record_fraction=len(matching) / 512,
            observed=_summary(matching) if matching else None)
    result = dict(full=_summary(rows), first_half=_summary(rows[:256]),
        last_half=_summary(rows[256:]), per_source=per_source, loss_contract=LOSS_CONTRACT,
        missing_domains_are_diagnostic_only=True, record_order_digest=digest([row['record_id'] for row in rows]))
    require(all(torch.isfinite(torch.tensor(result[key]['KL'])).item()
        for key in ('full', 'first_half', 'last_half')), 'Non-finite common proposal KL.')
    return result


def score_proposal(student, batches, cache, *, data_identity):
    require(cache['manifest']['data_identity'] == data_identity
        and cache['manifest']['loss_contract'] == LOSS_CONTRACT
        and digest(cache['manifest']) == cache['cache_identity'], 'Teacher cache identity/loss protocol mismatch.')
    require(len(batches) == len(cache['entries']) == 128, 'Keep the fixedB4 common512 proposal readout.')
    rows, began = [], time.monotonic()
    with preserved_eval(student):
        for batch, entry, bound in zip(batches, cache['entries'], cache['manifest']['batches']):
            ids = [row['record_id'] for row in batch['records']]
            require(batch_digest(batch['inputs']) == batch['input_sha256'] == entry['input_sha256']
                and ids == entry['record_ids'], 'Teacher/student proposal input, mask or order differs.')
            require(bound == dict(input_sha256=entry['input_sha256'], record_ids=ids,
                target_sha256=entry['target']['tensor_sha256'], shape=entry['target']['original_shape'],
                vocab_size=entry['target']['vocab_size'], original_dtype=entry['target']['original_dtype']),
                'Teacher target entry differs from the immutable cache manifest.')
            token_kl = cached_token_kl(_model_logits(student, batch['inputs']), entry['target'])
            numerators, denominators = weighted_rows(token_kl, batch['inputs'])
            for ref, numerator, denominator in zip(batch['records'], numerators.cpu(), denominators.cpu()):
                rows.append(dict(record_id=ref['record_id'], alias=ref['alias'],
                    numerator=float(numerator), denominator=float(denominator)))
    result = aggregate_rows(rows)
    result.update(data_identity=data_identity, teacher_cache_identity=cache['cache_identity'],
        elapsed_seconds=time.monotonic() - began, rows=rows,
        verification_scope='Actual supplied whole-model forward; runtime eligibility belongs to its model adapter evidence.')
    return result


def score_layer_functions(parent, candidate_modules, batches, *, projection_names,
        parent_identity, data_identity):
    """Stream actual parent inputs through deployed candidates; retain sufficient statistics only."""
    require(parent_identity and data_identity and len(batches) == 128,
        'Bind the parent weights and the full root-frozen512/B4 proposal input.')
    names = tuple(projection_names)
    require(names and len(names) == len(set(names)) and candidate_modules,
        'Use the actual category inventory and distinct deployed candidate IDs.')
    modules = {name: parent.get_submodule(name) for name in names}
    require(all(isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16
        for module in modules.values()), 'Local geometry readout requires the actual dense BF16 parent category.')
    require(all(set(mapping) == set(names) and all(isinstance(module, torch.nn.Module)
        for module in mapping.values()) for mapping in candidate_modules.values()),
        'Every candidate must supply its actual deployment module for every projection.')
    require(not ({id(module) for module in modules.values()} &
        {id(module) for mapping in candidate_modules.values() for module in mapping.values()}),
        'Parent and candidate modules must be independent actual objects.')
    stats = {candidate: {name: [dict(error_squared=0., target_squared=0., valid_tokens=0,
        records=0) for _ in range(2)] for name in names} for candidate in candidate_modules}
    current, handles, record_ids, input_identities = {}, [], [], []

    def hook_for(name):
        def observe(module, arguments, target):
            require(name not in current['seen'], 'A projection ran repeatedly for one proposal batch: ' + name)
            current['seen'].add(name)
            require(arguments and torch.is_tensor(arguments[0]) and torch.is_tensor(target),
                'The deployed local readout requires the actual Linear input and output tensors.')
            x, mask = arguments[0], current['mask']
            require(x.ndim == target.ndim == 3 and x.shape[:2] == target.shape[:2] == mask.shape
                and x.dtype == target.dtype == torch.bfloat16,
                'Do not substitute calibration inputs, flatten away padding, or change BF16 deployment arithmetic.')
            mask = mask.to(target.device).bool().unsqueeze(-1)
            target_squared = float(target.float().square().masked_fill(~mask, 0.).sum(dtype=torch.float64))
            tokens = int(mask.sum())
            require(tokens > 0 and target_squared > 0 and math.isfinite(target_squared) and torch.isfinite(target).all().item(),
                'Non-finite or zero-energy parent output cannot define functional NMSE.')
            for candidate, mapping in candidate_modules.items():
                actual = mapping[name](x)
                require(torch.is_tensor(actual) and actual.shape == target.shape
                    and actual.dtype == target.dtype and actual.device == target.device
                    and torch.isfinite(actual).all().item(), 'The actual deployed candidate output differs in shape/precision or is non-finite.')
                error = (actual.float() - target.float()).square().masked_fill(~mask, 0.).sum(dtype=torch.float64)
                require(torch.isfinite(error).item(), 'Non-finite squared deployment error.')
                row = stats[candidate][name][current['half']]
                row['error_squared'] += float(error)
                row['target_squared'] += target_squared
                row['valid_tokens'] += tokens
                row['records'] += int(x.shape[0])
        return observe

    began = time.monotonic()
    all_candidates = [module for mapping in candidate_modules.values() for module in mapping.values()]
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook_for(name)))
        with preserved_eval(parent, *all_candidates):
            for index, batch in enumerate(batches):
                inputs = batch['inputs']
                require(batch_digest(inputs) == batch['input_sha256'] and len(batch['records']) == 4
                    and inputs['input_ids'].shape[0] == 4, 'The fixed common512/B4 tokens, mask or record order changed.')
                mask = inputs['attention_mask']
                require(torch.all((mask == 0) | (mask == 1)).item(), 'Use the original binary nonpadding input-token mask.')
                current.update(mask=mask, half=index // 64, seen=set())
                # The hook consumes each local activation immediately; no all-layer input bank is cached.
                _model_logits(parent, inputs)
                require(current['seen'] == set(names), 'Not every actual category projection was reached by the parent forward.')
                record_ids.extend(row['record_id'] for row in batch['records'])
                input_identities.append(batch['input_sha256'])
        require(len(record_ids) == len(set(record_ids)) == 512,
            'Local functional readout requires the same512 distinct proposal records.')
    finally:
        for handle in handles:
            handle.remove()
    input_identity = digest(dict(parent_identity=parent_identity, data_identity=data_identity,
        record_ids=record_ids, input_sha256=input_identities))
    results = {}
    for candidate, projections in stats.items():
        require(all(half['records'] == 256 and half['target_squared'] > 0
            for halves in projections.values() for half in halves), 'Both complete proposal halves must reach each projection.')
        results[candidate] = dict(measurement='actual_parent_input_BF16_functional_NMSE',
            selection_halves=[256, 256], parent_input_identity=input_identity,
            data_identity=data_identity, parent_identity=parent_identity, candidate_id=candidate,
            per_projection={name: [half['error_squared'] / half['target_squared'] for half in halves]
                for name, halves in projections.items()},
            parameter_counts={name: int(module.weight.numel()) for name, module in modules.items()},
            sufficient_statistics=projections,
            arithmetic='FP32 squared error/target; FP64 sums; all nonpadding input tokens; no causal shift or response reweighting',
            group_aggregation='Group NMSE=sum(layer error_squared)/sum(layer target_squared); regret weights use actual group/category/full projection parameters',
            elapsed_seconds=time.monotonic() - began,
            verification_scope='Actual supplied parent/candidate forward; full-model runtime eligibility requires separate adapter/pilot evidence.')
    return results

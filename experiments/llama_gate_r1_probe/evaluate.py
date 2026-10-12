"""Native final/fresh audit and fixed development-only complete-model KL/NLL."""
import argparse
import gc
import logging
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.llama_gate_r1_probe.common import (
    DOMAINS, PREFIX, SOURCE, _assert_state_equal, _load_state, budget_check, check, read,
    require, state_inventory, write_json)
from experiments.llama_gate_r1_probe.migrate import strict_load
from experiments.gate_student_context_evaluation.deploy import native_policy
from rotation.model_utils import get_model
from tools.cat_eval import _prepare_model_for_eval
from train_utils.checkpoint_bpw import audit_state_dict


def batch_at(batch, index):
    return {key: batch[key][index:index + 1].to('cuda:0') for key in ('input_ids', 'attention_mask')}


def logits_labels(model, batch, index):
    item = batch_at(batch, index)
    valid = item['attention_mask'][0, 1:].bool()
    logits = model(**item, use_cache=False).logits[0, :-1][valid].float()
    return logits, item['input_ids'][0, 1:][valid]


@torch.no_grad()
def teacher(record, batches):
    # Same top100-plus-tail algebra as the existing gate_tail_objective_probe;
    # cached full normalization avoids retaining full-vocabulary teacher logits.
    model = get_model(record['hf_source']['path'], None).eval().requires_grad_(False).to('cuda:0')
    model.config.use_cache = False
    policy = native_policy(model, record['model_policy'])
    cache = {}
    for domain, batch in batches.items():
        cache[domain] = []
        for index in range(32):
            logits, labels = logits_labels(model, batch, index)
            values, indices = logits.topk(100, dim=-1, sorted=False)
            cache[domain].append({'top_values': values.cpu(), 'top_indices': indices.cpu(),
                'log_z': torch.logsumexp(logits, -1, keepdim=True).cpu(), 'labels': labels.cpu(),
                'nll_sum': float(F.cross_entropy(logits, labels, reduction='none').double().sum())})
    require(native_policy(model, record['model_policy']) == policy, 'Teacher FP32 RoPE changed.')
    del model; gc.collect(); torch.cuda.empty_cache()
    return cache


@torch.no_grad()
def score(model, batches, cache):
    result = {}
    for domain, batch in batches.items():
        rows = []
        for index in range(32):
            logits, labels = logits_labels(model, batch, index)
            reference = cache[domain][index]
            require(torch.equal(labels.cpu(), reference['labels']), 'Teacher/student labels differ.')
            teacher_log = (reference['top_values'] - reference['log_z']).to('cuda:0')
            student_log = logits.gather(-1, reference['top_indices'].to('cuda:0')) - torch.logsumexp(logits, -1, keepdim=True)
            t, s = teacher_log.exp(), student_log.exp()
            t_other, s_other = (1 - t.sum(-1)).clamp(0, 1), (1 - s.sum(-1)).clamp(0, 1)
            tiny = torch.finfo(s_other.dtype).tiny
            kl = (t * (teacher_log - student_log)).sum(-1) + torch.where(t_other > 0,
                t_other * (t_other.clamp_min(tiny).log() - s_other.clamp_min(tiny).log()), torch.zeros_like(t_other))
            nll = F.cross_entropy(logits, labels, reduction='none')
            require(torch.isfinite(kl).all() and torch.isfinite(nll).all(), 'Nonfinite complete-model observation.')
            rows.append({'index': index, 'causal_tokens': len(labels), 'kl_sum': float(kl.double().sum()),
                         'nll_sum': float(nll.double().sum())})
        tokens = sum(x['causal_tokens'] for x in rows)
        result[domain] = {'documents': rows, 'causal_tokens': tokens,
                         **{key + '_mean': sum(x[key + '_sum'] for x in rows) / tokens for key in ('kl', 'nll')}}
    return result


def audit_final(record, arm, model, meta, final):
    root = Path(record['output_dir'])
    initial = SOURCE if arm == 'r0' else root / 'r1_initial'
    source, saved = _load_state(initial), _load_state(final)
    expected = dict(source)
    fp32 = torch.load(root / arm / 'selected_fp32.pt', map_location='cpu', weights_only=True)
    require(set(fp32) == set(record['inventories'][arm]['selected']), 'Wrong actual endpoint parameter inventory.')
    for key, value in fp32.items():
        expected[key] = value.to(source[key].dtype)
    count = _assert_state_equal(saved, expected, 'Native final against source and actual FP32 endpoint')
    _assert_state_equal(model.state_dict(), saved, 'Native fresh reload')
    require(state_inventory(saved) == record['inventories'][arm], 'Final dynamic inventory differs.')
    require(meta['converted_modules'] == read(initial / 'checkpoint_meta.json')['converted_modules'], 'Training changed codec metadata.')
    budget = audit_state_dict(meta, saved, max_bpw=2.5, max_lora_rank=0)
    budget_check(budget, record, arm)
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'source_native_fresh_exact': count,
            'selected_decoder_tensors': len(fp32), 'all_other_tensors_exact': count - len(fp32),
            'bit_budget': budget, 'fixed_packed_banks': record['inventories'][arm]['banks']}


@torch.no_grad()
def evaluate(path, phase):
    record = read(path); check(record)
    root = Path(record['output_dir']); started = time.time()
    batches = torch.load(root / 'holdout.pt', map_location='cpu', weights_only=True)
    require(tuple(batches) == DOMAINS, 'Only the two fixed observation domains are permitted.')
    if phase == 'initial':
        require(not (root / 'teacher.pt').exists(), 'No teacher cache overwrite.')
        cache = teacher(record, batches); torch.save(cache, root / 'teacher.pt')
    else:
        cache = torch.load(root / 'teacher.pt', map_location='cpu', weights_only=True)
    output = {}
    for arm in ('r0', 'r1'):
        initial = SOURCE if arm == 'r0' else root / 'r1_initial'
        final = initial if phase == 'initial' else Path(read(root / arm / 'training.json')['training_result']['saved_model_dir'])
        model, meta = strict_load(final)
        audit = None if phase == 'initial' else audit_final(record, arm, model, meta, final)
        model.config.use_cache = False
        warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
        policy = native_policy(model, record['model_policy'])
        before = {k: (v.data_ptr(), v._version) for k, v in model.state_dict().items()}
        observations = score(model, batches, cache)
        require(before == {k: (v.data_ptr(), v._version) for k, v in model.state_dict().items()}
                and policy == native_policy(model, record['model_policy']), 'Scoring changed persistent state/RoPE.')
        output[arm] = {'checkpoint': str(final), 'checkpoint_id': meta['checkpoint_id'], 'audit': audit,
                       'warmup': warmup, 'domains': observations, 'persistent_state_unchanged': True}
        del model; gc.collect(); torch.cuda.empty_cache()
    paired = {}
    for domain in DOMAINS:
        left, right = output['r0']['domains'][domain], output['r1']['domains'][domain]
        paired[domain] = {'r1_minus_r0': {key: right[key] - left[key] for key in ('kl_mean', 'nll_mean')},
            'documents': [{'index': a['index'], 'causal_tokens': a['causal_tokens'],
                'kl_sum_delta': b['kl_sum'] - a['kl_sum'], 'nll_sum_delta': b['nll_sum'] - a['nll_sum']}
                for a, b in zip(left['documents'], right['documents'])]}
    write_json(root / (phase + '_observations.json'), {'status': 'completed', 'phase': phase,
        'arms': output, 'paired': paired, 'elapsed_seconds': time.time() - started,
        'boundary': 'Development-only64 documents; no full8/LongBench or historically unseen-data claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('initial', 'final'))
    options = parser.parse_args(); evaluate(options.launch_record, options.phase)

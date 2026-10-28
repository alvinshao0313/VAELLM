"""Thin full8/LongBench consumers of the actual native plus Rice7 bundle."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys

import sparse_common as c
import sparse_bundle as bundle

t = c.torch


def check_audit(checkpoint):
    meta = c.read(checkpoint / 'checkpoint_meta.json')
    audit = c.read(checkpoint / 'assembly_report.json')
    c.require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
              and Path(audit['checkpoint_dir']).resolve() == checkpoint
              and audit['base_frozen_tensors_compared'] == 3423 and audit['rice_tensors_compared'] == 864
              and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 4287
              and audit['actual_native_save'] and audit['independent_fresh_strict']
              and audit['bit_budget']['budget_passed'], 'Use the actual complete in-budget strict export.')
    return meta, audit


def check_loaded(proof, identifier, budget):
    c.require(proof['checkpoint_id'] == identifier and proof['persistent_tensors'] == 4287
              and proof['strict_missing'] == proof['strict_unexpected'] == []
              and proof['base_strict_tensors'] == 3423 and proof['rice_strict_tensors'] == 864
              and proof['native_main_banks'] == 504 and proof['actual_MLP_sparse_caches'] == 108,
              'The real full native/Rice consumer inventory changed.')
    c.require(proof['bit_budget'] == budget, 'Actual fresh full-bundle bit accounting changed.')


@t.no_grad()
def load_checkpoint(checkpoint, logger):
    checkpoint = Path(checkpoint).resolve()
    meta, audit = check_audit(checkpoint)
    model, actual, proof = bundle.load_cpu_bundle(checkpoint)
    c.require(actual == meta and proof['bit_budget'] == audit['bit_budget'], 'Fresh local bundle identity/ledger differs.')
    warm, caches = bundle.prepare_model(model, checkpoint, meta, logger)
    proof.update(warmup=warm, actual_MLP_sparse_caches=len(caches), sparse_cache_records=caches,
                 assembly_report=c.binding(checkpoint / 'assembly_report.json'))
    check_loaded(proof, meta['checkpoint_id'], audit['bit_budget'])
    return model, proof


def check_gpu(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta, audit = check_audit(checkpoint)
    provenance = audit['sources']
    prepared = Path(provenance['prepared']['path'])
    report = c.read(prepared.parent / 'gpu_validation.json')
    c.require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
              and report['source_checkpoint_id'] == meta['source_checkpoint_id']
              and report['actual_whole_model_forwards'] == 1 and report['actual_MLP_Flinear_hooks'] == 108
              and report['actual_native_sparse_caches_compared'] == 108
              and report['all_protected_columns_exact'] and report['persistent_state_unchanged'],
              'One independent actual auxiliary32/cache validation is required before formal evaluation.')
    return report


@t.no_grad()
def verify(checkpoint, output):
    c.require(not output.exists(), 'Never overwrite independent GPU verification.')
    checkpoint = checkpoint.resolve()
    meta, audit = check_audit(checkpoint)
    prepared_path = Path(audit['sources']['prepared']['path'])
    prepared = c.check_prepared(prepared_path)
    build = c.read(prepared_path.parent / 'build.json')
    c.require(build['source_checkpoint_id'] == meta['source_checkpoint_id'], 'Actual source/build changed.')
    model, proof = load_checkpoint(checkpoint, logging.getLogger(__name__))
    expected = {name: row['expected_native_sparse_cache'] for name, row in build['modules'].items()}
    c.require(proof['sparse_cache_records'] == expected, 'Independent consumer differs from actual raw-build caches.')
    before = {key: c.tensor_record(value) for key, value in model.state_dict().items()}
    calls, handles = {}, []
    from torch.nn import functional as F
    for name in prepared['names']:
        layer = model.get_submodule(name)
        def check(layer, args, result, key=name):
            c.require(layer.cache_decoded_weight and layer.temporary and not layer.trainable_decode
                      and not layer.always_use_original and t.equal(result, F.linear(args[0], layer._cached_weight, layer.bias)),
                      'Actual forward did not consume the native sparse cache: ' + key)
            calls[key] = calls.get(key, 0) + 1
        handles.append(layer.register_forward_hook(check))
    try:
        cases, evidence = c.aux32()
        c.require(evidence == prepared['auxiliary32'], 'Existing auxiliary32 inputs changed.')
        length = max(len(case['input_ids']) for case in cases)
        pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        ids = t.full((32, length), pad, dtype=t.long)
        mask = t.zeros_like(ids)
        for index, case in enumerate(cases):
            count = len(case['input_ids'])
            ids[index, -count:], mask[index, -count:] = case['input_ids'], 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        result = model(input_ids=ids.to('cuda:0'), attention_mask=mask.to('cuda:0'),
                       position_ids=positions.to('cuda:0'), use_cache=False, logits_to_keep=1)
        c.require(result.past_key_values is None and result.logits.shape == (32, 1, 151936)
                  and result.logits.dtype == t.bfloat16 and t.isfinite(result.logits).all().item()
                  and calls == dict.fromkeys(prepared['names'], 1), 'One actual auxiliary32 forward must cover108 MLPs.')
        choices = t.stack([case['native_continuation_ids'].reshape(4) for case in cases]).to('cuda:0')
        logp = F.log_softmax(result.logits[:, 0].gather(1, choices).float(), dim=-1)
        c.require({key: c.tensor_record(value) for key, value in model.state_dict().items()} == before,
                  'Independent forward changed persistent state.')
        record = dict(status='verified', checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=meta['source_checkpoint_id'],
            actual_whole_model_forwards=1, actual_MLP_Flinear_hooks=108, actual_native_sparse_caches_compared=108,
            all_protected_columns_exact=True, persistent_state_unchanged=True, auxiliary32=evidence,
            auxiliary32_log_probs=logp.cpu().tolist(), native_load_audit=proof, quality_selection=False)
    finally:
        for handle in handles:
            handle.remove()
    del model, result, logp, layer
    gc.collect()
    t.cuda.empty_cache()
    c.write(output, record)
    return record


def load_model(config, logger):
    from experiments.longbench_v1.models import check_config
    from experiments.longbench_v1.common import check_hashes
    check_config(config)
    check_hashes(config['model']['raw_abs005_Rice7']['source_sha256'])
    checkpoint = Path(config['model']['checkpoint']).resolve()
    check_gpu(checkpoint)
    model, proof = load_checkpoint(checkpoint, logger)
    c.require(proof['checkpoint_id'] == config['model']['checkpoint_id'], 'Formal model identity changed.')
    return model, proof


def full8():
    api = c.module(c.RESULT / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_raw_sparse_sample_helpers')
    sample = c.module(api.SAMPLE_EVAL, '_raw_sparse_original_sample_entry')
    args = sample.build_parser().parse_args(sys.argv[2:])
    c.require(args.strict and args.eval_lm_eval and not args.eval_ppl and not args.eval_linear_mse
              and not args.eval_hif4_act and args.adapter_dir is None and args.lm_limit is None
              and args.num_fewshot == 0 and args.tokenizer_use_fast and args.prewarm_group_size == 8
              and args.eval_device == 'cuda:0' and (args.tasks, str(args.lm_batch_size))
              in (('mmlu', '64'), (api.SEVEN, '32')) and not os.path.lexists(args.output_json),
              'Only the original complete zero-shot MMLU64/seven32 protocol is supported.')
    checkpoint = Path(args.checkpoint_dir).resolve()
    check_gpu(checkpoint)
    meta, audit = check_audit(checkpoint)
    logger, timestamp, _ = sample._build_logger(args.eval_log_dir)
    model, proof = load_checkpoint(checkpoint, logger)
    model.config.use_cache = False
    from transformers import AutoTokenizer
    sys.path.insert(0, str(api.SAMPLE_EVAL.parent))
    from eval_utils_with_samples import run_lm_eval
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, trust_remote_code=True, local_files_only=True)
    options = sample._build_lm_eval_args(args, tokenizer_name=meta['base_model_path'], eval_run_ts=timestamp)
    result = run_lm_eval(model, tokenizer, options)
    payload = dict(checkpoint_dir=str(checkpoint), checkpoint_id=meta['checkpoint_id'], base_model_path=meta['base_model_path'],
        checkpoint_loader=c.FORMAT, adapter_dir=None, strict_load=True, eval_hif4_act=False,
        tokenizer_class=type(tokenizer).__name__, tokenizer_is_fast=tokenizer.is_fast, bundle_load_audit=proof,
        evals=dict(cache_warmup=proof['warmup'], lm_eval=result))
    check_loaded(proof, meta['checkpoint_id'], audit['bit_budget'])
    c.write(Path(args.output_json), payload)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'full8':
        full8()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'longbench'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    t.set_num_threads(4)
    t.manual_seed(137)
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.INFO)
    if args.command == 'verify':
        verify(args.checkpoint.resolve(), args.output.resolve())
    else:
        from experiments.longbench_v1 import runner
        from experiments.longbench_v1.protocol import DATASETS
        c.require(c.read(args.config)['mode'] == 'full' and not args.output.exists(), 'One fresh complete full21 run.')
        runner.load_model = load_model
        report = runner.run(args.config, args.output, datasets=list(DATASETS), resume=False, max_runtime_seconds=25200)
        c.require(report['status'] == 'partition_complete' and report['exit_code'] == 0
                  and report['new_predictions'] == 4750 and report['already_present'] == 0,
                  'All21 official tasks/4750 samples must complete; no partial scoring.')


if __name__ == '__main__':
    main()

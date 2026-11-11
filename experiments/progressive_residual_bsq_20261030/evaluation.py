"""Native S4 research-parent evaluation using the unchanged official protocols."""
import argparse
import gc
import logging
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from bindings import BASELINE, HF, MANIFEST, SAMPLE, boundary, module, s

SEVEN = 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'


def load(directory, logger, *, check_largest_group=False):
    import torch
    s.enable_snapshot()
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    from train_utils.checkpoint_bpw import audit_state_dict
    event = boundary(directory)
    model, meta, loaded = load_v6_model_checkpoint(str(directory), base_model_path=str(HF), map_location='cpu', strict=True)
    s.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Strict native fresh load failed.')
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=16, max_lora_rank=0)
    s.require(not budget['contract_violations'] and budget['budget_passed'], 'Research parent storage contract failed.')
    model.eval().requires_grad_(False)
    reference_weights = None
    if check_largest_group:
        from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
        model.to('cuda')
        names = [f'model.layers.{index}.mlp.down_proj' for index in range(24, 32)]
        targets = [NamedVAELinearDecodeTarget(name=name, base_layer=model.get_submodule(name),
            target_dtype=torch.bfloat16, include_low_rank=False) for name in names]
        reference = decode_named_vae_linear_weights(targets, dtype=torch.bfloat16, group_size=8,
            compute_device='cuda', logger=logger, respect_cache_policy=True)
        reference_weights = {item.name: item.decoded_weight.cpu() for item in reference}
        for item in reference:
            item.base_layer.clear_decoded_weight_cache()
        del reference, targets
    from prewarm import prepare
    warmup = prepare(model, logger, reference_weights)
    s.require(warmup['warmed'] == warmup['total'] == meta['converted_module_count'] and not warmup['failed'],
        'Every real native module must complete original group8 prewarm.')
    from experiments.longbench_v1.protocol import model_policy
    policy = model_policy(HF)
    s.require(model.config._attn_implementation == policy['attn_implementation']
        and model.config.rope_scaling == policy['rope_scaling'] and model.config.rope_theta == policy['rope_theta']
        and model.get_input_embeddings().weight.dtype == torch.bfloat16
        and all(value.dtype == torch.float32 for name, value in model.named_buffers() if 'inv_freq' in name),
        'Preserve original BF16/FP32 RoPE/attention policy without a whole-model dtype cast.')
    return model, meta, dict(**event, strict_missing=[], strict_unexpected=[], bit_budget=budget,
        warmup=warmup, model_policy=policy, research_parent=True, final_lowbit_acceptance_evaluated=False)


def functional(model, output, request):
    import torch
    import torch.nn.functional as F
    input_dir = Path(request.get('input_dir', request['output_dir']))
    tokens = torch.load(input_dir / 'functional_inputs.pt', weights_only=True)
    teacher_path = input_dir / 'teacher_logits.pt'
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
    model.config.use_cache = False
    kl_sum, nll_sum, count = 0., 0., 0
    with torch.inference_mode():
        for ids, expected in zip(tokens['input_ids'], teacher['logits'], strict=True):
            logits = model(input_ids=ids.cuda()).logits[:, :-1]
            targets = ids[:, 1:].cuda()
            for start in range(0, logits.shape[1], 32):
                student = logits[:, start:start + 32].float()
                reference = expected[:, start:start + 32].cuda().float()
                logp = F.log_softmax(reference, dim=-1)
                logq = F.log_softmax(student, dim=-1)
                kl_sum += float((logp.exp() * (logp - logq)).sum())
                labels = targets[:, start:start + 32]
                nll_sum += float(F.cross_entropy(student.reshape(-1, student.shape[-1]), labels.reshape(-1), reduction='sum'))
                count += labels.numel()
    result = dict(dataset='vicgalle/alpaca-gpt4', split='train', seed=2031,
        input_ids_sha256=tokens['input_ids_sha256'], teacher_logits_sha256=s.sha(teacher_path),
        effective_next_tokens=count, reduction='sum over vocabulary then valid next-token mean; teacher forward KL',
        forward_kl=kl_sum / count, student_nll=nll_sum / count, student_ppl=math.exp(nll_sum / count),
        teacher_nll=teacher['nll'], teacher_ppl=math.exp(teacher['nll']))
    s.write(output, result)
    return result


def teacher(request):
    import torch
    import torch.nn.functional as F
    s.enable_snapshot()
    from rotation.model_utils import get_qwen3
    directory = Path(request['output_dir'])
    tokens = torch.load(directory / 'functional_inputs.pt', weights_only=True)
    model = get_qwen3(str(HF), None).eval().requires_grad_(False).to('cuda')
    model.config.use_cache = False
    outputs, total, count = [], 0., 0
    with torch.inference_mode():
        for ids in tokens['input_ids']:
            logits = model(input_ids=ids.cuda()).logits[:, :-1]
            total += float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].cuda().reshape(-1), reduction='sum'))
            count += ids[:, 1:].numel()
            outputs.append(logits.cpu())
    torch.save(dict(logits=outputs, nll=total / count, count=count), directory / 'teacher_logits.pt')
    del model
    gc.collect(); torch.cuda.empty_cache()


def downstream(request, directory, output, shard, *, functional_output=None, short=False):
    import torch
    from transformers import AutoTokenizer
    model, meta, audit = load(directory, logging.getLogger('s4_downstream'))
    model.config.use_cache = False
    if short:
        reference_path = Path(request.get('consumer_reference_logits',
            str(Path(request.get('input_dir', request['output_dir'])) / 'short_live_logits.pt')))
        live = torch.load(reference_path, weights_only=True)
        with torch.inference_mode():
            actual = model(input_ids=live['input_ids'].cuda()).logits.cpu()
        s.require(torch.equal(actual, live['logits']) and torch.isfinite(actual).all(), 'Fresh BF16 logits differ from their genuine bound reference.')
        audit['same_input_two_fresh_BF16_logits_exact' if request.get('consumer_reference_logits')
            else 'same_input_live_BF16_logits_exact'] = True
        audit['reference_logits_sha256'] = s.sha(reference_path)
    if functional_output:
        audit['functional'] = functional(model, functional_output, request)
    tokenizer = AutoTokenizer.from_pretrained(str(HF), use_fast=True, trust_remote_code=True, local_files_only=True)
    sys.path.insert(0, str(SAMPLE))
    api = module(SAMPLE / 'eval_utils_with_samples.py', '_progressive_original_hflm_samples')
    args = SimpleNamespace(tasks='mmlu' if shard == 'mmlu' else SEVEN, num_fewshot=0,
        batch_size='64' if shard == 'mmlu' else '32', lm_limit=1 if short else None, model_path=str(HF),
        eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output.parent / (shard + '_samples')),
        mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    result = api.run_lm_eval(model, tokenizer, args)
    payload = dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_loader='v6',
        base_model_path=str(HF), strict_load=True, adapter_dir=None, eval_hif4_act=False,
        tokenizer_class=type(tokenizer).__name__, tokenizer_is_fast=tokenizer.is_fast,
        load_audit=audit, evals=dict(lm_eval=result), validation_only=short)
    if not short:
        from experiments.w25_eval.acceptance import leaf_metrics, _group_agrees
        metrics, _, expected_batch = leaf_metrics(result['raw_results'], result['n_samples'], result['n_shot'], result['configs'])
        _group_agrees(metrics, result['raw_results'], result['group_results'])
        reference = s.read(BASELINE / (shard + '.json'))
        s.require(result['configs'] == reference['configs'] and all(result['versions'][name] == reference['versions'][name]
            for name in result['n_samples']) and str(result['batch_size']) == str(expected_batch),
            'Actual full downstream task versions/configs/batch differ from the verified BF16 baseline.')
        payload['verified_task_metrics'] = metrics
    s.write(output, payload)
    return payload


def partial_probe(request, directory, output):
    import torch
    model, meta, audit = load(directory, logging.getLogger('s4_partial_probe'))
    s.require(meta['completed_categories'] == ['q_proj'] and len(meta['pending_dense_targets']) == 216,
        'Probe the actual partial q-S4 plus future BF16 boundary.')
    pending = {name: dict(dtype=str(model.get_submodule(name).weight.dtype),
        shape=list(model.get_submodule(name).weight.shape)) for name in meta['pending_dense_targets']}
    s.require(all(item['dtype'] == 'torch.bfloat16' for item in pending.values()), 'Future dense weights must stay original BF16.')
    probe = functional(model, output.parent / 'q_partial_functional.json', request)
    s.require(math.isfinite(probe['student_nll']) and math.isfinite(probe['forward_kl']), 'Partial S4 forward is nonfinite.')
    s.write(output, dict(status='verified_actual_partial_S4_future_BF16_forward', checkpoint_id=meta['checkpoint_id'],
        load_audit=audit, pending_dense=pending, functional=probe))


def native_config(request, directory, *, short=False):
    from experiments.longbench_v1 import models
    config = models.make_config(MANIFEST, kind='dense', hf_path=HF, mode='validation' if short else 'full')
    event = boundary(directory)
    meta = s.read(Path(directory) / 'checkpoint_meta.json')
    config['model'].update(kind='compressed', checkpoint=str(directory), checkpoint_id=event['checkpoint_id'],
        metadata_sha256=event['metadata_sha256'], weight_stat=__import__('experiments.longbench_v1.common', fromlist=['file_stat']).file_stat(
            Path(directory) / meta['state_dict_file']))
    config['research_parent'] = dict(stage_count=4, unprotected_main_bpw=4, final_lowbit_acceptance_evaluated=False)
    config['source_sha256'].update(request['source_sha256'])
    return config


def predict(config_path, output, datasets):
    from experiments.longbench_v1 import runner
    runner.load_model = lambda cfg, logger: (lambda result: (result[0], result[2]))(load(cfg['model']['checkpoint'], logger))
    return runner.run(str(config_path), str(output), datasets=datasets, max_runtime_seconds=86400)


def longbench(request, directory, output, *, short=False):
    from experiments.longbench_v1 import runner
    from experiments.longbench_v1.scoring import score_directory
    config = native_config(request, directory, short=short)
    config_path = output.parent / 'longbench_config.json'
    s.write(config_path, config)
    runner.load_model = lambda cfg, logger: (lambda result: (result[0], result[2]))(load(cfg['model']['checkpoint'], logger))
    result = runner.run(str(config_path), str(output)) if short else runner.run(
        str(config_path), str(output), max_runtime_seconds=86400)
    if short:
        s.require(result['status'] == 'diagnostic_complete' and result['new_predictions'] == 3,
            'Require original real long-context, summary and short-generation validation requests.')
        return result
    s.require(result['status'] == 'partition_complete' and result['new_predictions'] == 4750
        and result['already_present'] == 0, 'Require one complete original LongBench21/4750 evaluation.')
    return score_directory(MANIFEST, output, output / 'scores.json')


def reference(request, directory, output):
    import torch
    model, meta, audit = load(directory, logging.getLogger('S4_group8_reference'), check_largest_group=True)
    tokens = torch.load(Path(request.get('input_dir', request['output_dir'])) / 'functional_inputs.pt', weights_only=True)
    ids = tokens['input_ids'][0]
    model.config.use_cache = False
    with torch.inference_mode():
        logits = model(input_ids=ids.cuda()).logits.cpu()
    s.require(torch.isfinite(logits).all(), 'Complete S4 fresh forward produced nonfinite logits.')
    torch.save(dict(input_ids=ids, logits=logits), Path(request['consumer_reference_logits']))
    s.write(output, dict(status='verified_original_group8_and_streaming_S4_full_prewarm',
        checkpoint_id=meta['checkpoint_id'], load_audit=audit,
        independent_fresh_reference_sha256=s.sha(request['consumer_reference_logits']),
        original_fitter_live_logits_available=False,
        boundary='Reference from the actual saved S4 endpoint; not the lost original fitter-live forward.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--component', choices=('teacher', 'probe', 'reference', 'mmlu', 'seven', 'longbench', 'predict'), required=True)
    parser.add_argument('--functional-output', type=Path)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    request = s.read(args.request)
    s.check_pins(request['source_sha256'])
    if args.component == 'teacher':
        teacher(request)
    elif args.component == 'probe':
        partial_probe(request, args.checkpoint, args.output)
    elif args.component == 'reference':
        reference(request, args.checkpoint, args.output)
    elif args.component == 'predict':
        predict(args.config, args.output, args.datasets)
    elif args.component == 'longbench':
        longbench(request, args.checkpoint, args.output, short=args.short)
    else:
        downstream(request, args.checkpoint, args.output, args.component, functional_output=args.functional_output, short=args.short)

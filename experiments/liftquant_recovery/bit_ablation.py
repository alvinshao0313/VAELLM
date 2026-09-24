"""Bounded paired experiment: does learning hard codes improve held-out recovery?"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.liftquant_recovery.ablation_eval import document_split, model_nll, paired_summary
from experiments.liftquant_recovery.ablation_train import run_arm
from experiments.liftquant_recovery.recovery_runtime import (
    audit_topology, block_output, clear_caches, first_inputs, frozen_digest,
    prime_packed_cache, source_fingerprint, targets_by_block, teacher_outputs, tree_to,
)
from rotation.model_utils import get_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@torch.no_grad()
def set_codes(targets, codes):
    for path, module in targets:
        storage = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
        storage.copy_(codes[path].to(storage.device))
        grouped = getattr(module, '_parallel_stage_grouped_vq_packed', None)
        if grouped is not None:
            grouped.copy_(storage)
        module.clear_decoded_weight_cache()


@torch.no_grad()
def block_errors(block, hidden, target, kwargs, batch_size, device):
    block.to(device).eval()
    prime_packed_cache(block)
    gpu_kwargs = tree_to(kwargs, device)
    values = []
    try:
        for start in range(0, len(hidden), batch_size):
            output = block_output(block, hidden[start:start+batch_size].to(device), gpu_kwargs)
            errors = (output.float()-target[start:start+batch_size].to(device).float()).square()
            values.extend(errors.mean(dim=(1, 2)).cpu().tolist())
    finally:
        clear_caches(block)
        block.cpu()
        torch.cuda.empty_cache()
    return values


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke-rows', required=True)
    parser.add_argument('--train-samples', type=int, default=16)
    parser.add_argument('--holdout-samples', type=int, default=16)
    parser.add_argument('--seqlen', type=int, default=128)
    parser.add_argument('--block', type=int, default=9)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--code-lr', type=float, default=2e-5)
    parser.add_argument('--decoder-lr', type=float, default=1.25e-5)
    parser.add_argument('--gpu-memory-gib', type=float, default=12)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--verify-nll-reference', action='store_true')
    args = parser.parse_args()
    if not (0 < args.train_samples <= 16 and 0 < args.holdout_samples <= 16
            and 0 < args.seqlen <= 128 and 0 < args.steps <= 384 and args.block == 9):
        parser.error('This entry is explicitly limited to a small block-9 paired experiment.')
    if args.batch_size != 2 or args.train_samples % 2 or args.holdout_samples % 2:
        parser.error('Use batch=2 and even train/heldout counts for captured block arguments.')
    return args


def run(args):
    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.checkpoint).resolve()
    source_id = source_fingerprint(source)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    ids, data_meta = document_split(args.smoke_rows, tokenizer, args.train_samples,
                                   args.holdout_samples, args.seqlen, args.seed)
    torch.save(ids, output / 'calibration_ids.pt')
    manifest = dict(config=vars(args), source_checkpoint=str(source), source_fingerprint=source_id,
                    data=data_meta, torch_version=torch.__version__, physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')},
                    comparison='decoder-only vs joint; identical decoder optimizer schedule/data/path; fixed final step',
                    endpoint_probe='joint final decoder with original codes; conditional contribution, not an independent training arm',
                    scope='single block, small first-shard document sample; heldout NLL is not downstream benchmark accuracy')
    write_json(output / 'manifest.json', manifest)
    if args.preflight_only:
        write_json(output / 'summary.json', dict(status='PASS', stage='CPU document-split preflight', data=data_meta))
        probe = paired_summary([1.0, 2.0], [0.5, 1.5])
        assert probe["bootstrap_95ci"] == [-0.5, -0.5] and probe["win_count"] == 2
        print(json.dumps(dict(status="PASS", stage="preflight", unique_eligible_documents=data_meta["eligible_documents"], input_sha256=data_meta["input_sha256"])), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for actual paired recovery.')
    device = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gib*2**30/total, 1), device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    student, meta, loaded = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Strict initial model load failed.')
    student.eval().requires_grad_(False)
    all_targets = targets_by_block(student, meta)
    targets = all_targets[args.block]
    allowed_ids = audit_topology(student, meta, [args.block], all_targets)
    allowed_names = {name for name, value in student.state_dict(keep_vars=True).items() if id(value) in allowed_ids}
    if not allowed_names:
        raise ValueError('No mutable state identified.')
    frozen_before = frozen_digest(student, allowed_names)
    block = student.model.layers[args.block]
    initial_state = {name: value.detach().clone().cpu() for name, value in block.state_dict().items()}
    initial_codes = {path: module.get_stage_part_vq_storage(stage_idx=0, part_idx=0).detach().clone().cpu()
                     for path, module in targets}
    manifest.update(source_checkpoint_id=meta['checkpoint_id'], device=torch.cuda.get_device_name(device),
                    mutable_state_names=sorted(allowed_names), frozen_sha256=frozen_before)
    write_json(output / 'manifest.json', manifest)
    print('Initial compressed model loaded; measuring full-model heldout NLL', flush=True)
    baseline_nll = model_nll(student, ids[args.train_samples:], args.batch_size, device)
    if args.verify_nll_reference:
        from experiments.liftquant_recovery.evaluate_pair import streamed_inference
        with streamed_inference(student, device), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student(input_ids=ids[args.train_samples:].to(device), use_cache=False).logits
        labels = ids[args.train_samples:, 1:].to(device)
        reference_nll = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction="none").mean(1).cpu()
        torch.testing.assert_close(torch.tensor(baseline_nll["nll_per_document"]), reference_nll, rtol=1e-6, atol=1e-6)
        manifest["full_model_nll_reference_check"] = {"status": "PASS", "max_abs": (torch.tensor(baseline_nll["nll_per_document"])-reference_nll).abs().max().item()}
        write_json(output / "manifest.json", manifest)
        del logits, labels, reference_nll
        torch.cuda.empty_cache()
    teacher = get_model(meta['base_model_path']).eval().requires_grad_(False)
    hidden, kwargs = first_inputs(teacher, ids, args.batch_size, device)
    for index in range(args.block):
        hidden = teacher_outputs(teacher.model.layers[index], hidden, kwargs, args.batch_size, device)
    target = teacher_outputs(teacher.model.layers[args.block], hidden, kwargs, args.batch_size, device)
    del teacher
    gc.collect()
    baseline_errors = block_errors(block, hidden, target, kwargs, args.batch_size, device)
    results = dict(baseline=dict(block_mse_per_document=baseline_errors, heldout_nll=baseline_nll), arms={})
    write_json(output / 'results.json', results)
    for label, update_codes in (('decoder_only', False), ('joint', True)):
        clear_caches(block)
        block.load_state_dict(initial_state, strict=True)
        set_codes(targets, initial_codes)
        student.eval().requires_grad_(False)
        torch.manual_seed(args.seed)
        print(f'Starting {label}: {args.steps} updates', flush=True)
        record = run_arm(block, targets, hidden, target, kwargs, args, device,
                         update_codes=update_codes, progress_path=output / f'{label}_progress.json')
        actual_errors = block_errors(block, hidden, target, kwargs, args.batch_size, device)
        record['block_mse_per_document'] = actual_errors
        record['heldout_nll'] = model_nll(student, ids[args.train_samples:], args.batch_size, device)
        record['frozen_state_unchanged'] = frozen_digest(student, allowed_names) == frozen_before
        if not record['frozen_state_unchanged']:
            raise ValueError(f'{label}: unexpected frozen-state mutation.')
        results['arms'][label] = record
        write_json(output / 'results.json', results)
        print(f'{label} finished, full-model heldout NLL={record["heldout_nll"]["mean_nll"]:.7g}', flush=True)
    arm_a, arm_b = results["arms"]["decoder_only"], results["arms"]["joint"]
    if arm_a["measurements"][0]["output_sha256"] != arm_b["measurements"][0]["output_sha256"]:
        raise AssertionError("Arms did not start from identical block outputs.")
    if [s["decoder_learning_rates"] for s in arm_a["steps"]] != [s["decoder_learning_rates"] for s in arm_b["steps"]]:
        raise AssertionError("Decoder learning-rate sequences differ between arms.")
    unchanged_steps = arm_b["first_flip_step"] or args.steps
    losses_a = np.asarray([s["train_loss"] for s in arm_a["steps"][:unchanged_steps]])
    losses_b = np.asarray([s["train_loss"] for s in arm_b["steps"][:unchanged_steps]])
    if not np.allclose(losses_a, losses_b, rtol=1e-5, atol=1e-7):
        raise AssertionError("Arms diverged before their first hard-code change.")
    results["paired_invariants"] = dict(initial_outputs_identical=True, decoder_lr_sequences_identical=True,
        pre_flip_steps_compared=unchanged_steps, pre_flip_loss_max_abs=float(np.max(np.abs(losses_a-losses_b))),
        decoder_only_bits_changed=arm_a["bits_changed_from_initial"])
    # Joint decoder held fixed; only its hard code payload is reverted.
    set_codes(targets, initial_codes)
    probe_errors = block_errors(block, hidden, target, kwargs, args.batch_size, device)
    probe_nll = model_nll(student, ids[args.train_samples:], args.batch_size, device)
    results['joint_decoder_original_codes'] = dict(block_mse_per_document=probe_errors, heldout_nll=probe_nll)
    cut = args.train_samples
    base = results['baseline']
    a, b = results['arms']['decoder_only'], results['arms']['joint']
    results['paired_comparisons'] = {}
    for name, reference, candidate in (
        ('joint_minus_decoder_only', a, b), ('joint_minus_initial', base, b),
        ('reverted_codes_minus_joint', b, results['joint_decoder_original_codes']),
    ):
        results['paired_comparisons'][name] = dict(
            heldout_block_mse=paired_summary(reference['block_mse_per_document'][cut:], candidate['block_mse_per_document'][cut:]),
            heldout_full_model_nll=paired_summary(reference['heldout_nll']['nll_per_document'], candidate['heldout_nll']['nll_per_document']))
    block.load_state_dict(initial_state, strict=True)
    set_codes(targets, initial_codes)
    if source_fingerprint(source) != source_id:
        raise ValueError('Source checkpoint metadata changed.')
    if frozen_digest(student, allowed_names) != frozen_before:
        raise ValueError('Frozen-state verification failed after code reversion.')
    write_json(output / 'results.json', results)
    write_json(output / 'summary.json', dict(status='PASS', stage='paired bit contribution diagnostic',
        seconds=time.time()-started, paired_comparisons=results['paired_comparisons'],
        final_experiment_run=False, downstream_benchmark_run=False, model_exported=False,
        max_memory_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        caution='Fixed final step, independent heldout documents, one block and one seed; no universal or downstream claim.'))
    print('Paired contribution diagnostic complete', flush=True)


if __name__ == '__main__':
    run(arguments())

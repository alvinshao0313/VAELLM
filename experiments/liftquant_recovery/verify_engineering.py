"""Bounded old/new numerical regression on the real initialized block-9 checkpoint.

Runs only a small GPU validation, never a formal experiment or model export.
The reference directory must contain the actual pre-change sources.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.liftquant_recovery import all_bits, block_train, recovery_runtime as runtime
from experiments.liftquant_recovery.liftquant_optimizer import verify_reference_sequence
from rotation.model_utils import get_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b, (a, b)


def train_with_final_state_trace(version, block, targets, hidden, truth, kwargs, args):
    """Observe the real optimizer without replacing its implementation or updates."""
    original_builder = version.build_optimizer
    captured, hook_seconds = {}, [0.0]
    def observed_builder(groups, steps):
        optimizer, schedulers = original_builder(groups, steps)
        counter = [0]
        def observe(opt, _args, _kwargs):
            counter[0] += 1
            if counter[0] != steps:
                return
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i, group in enumerate(opt.param_groups):
                for j, parameter in enumerate(group['params']):
                    key = f'{i}:{j}'
                    captured[key] = dict(parameter=digest(parameter), gradient=digest(parameter.grad),
                        state={name: digest(value) if isinstance(value, torch.Tensor) else value
                               for name, value in opt.state[parameter].items()})
            hook_seconds[0] += time.perf_counter()-start
        optimizer.register_step_post_hook(observe)
        return optimizer, schedulers
    version.build_optimizer = observed_builder
    torch.manual_seed(42)
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        record, output = version.train_block(block, targets, hidden, truth, kwargs, args, 'cuda:0')
    finally:
        version.build_optimizer = original_builder
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started-hook_seconds[0]
    assert captured
    return record, output, captured, seconds


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', formal_experiment=False,
                  scope='real block9 old/new equivalence; no downstream quality claim',
                  config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    def save():
        (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    old_bits = load_reference(args.reference/'all_bits.py', 'engineering_reference_bits')
    old_loop = load_reference(args.reference/'block_train.py', 'engineering_reference_loop')
    old_runtime = load_reference(args.reference/'recovery_runtime.py', 'engineering_reference_runtime')
    old_loop.all_bits = old_bits
    report['reference_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reference.glob('*') if p.is_file()}
    report['current_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    report['optimizer_reference'] = verify_reference_sequence()
    from experiments.liftquant_recovery.verify_engineering_kernels import run_checks, run_projection_checks
    report['conditional_backward'] = run_checks(args.reference, args.checkpoint)
    meta = json.loads((args.checkpoint/'checkpoint_meta.json').read_text())
    save()
    ids = torch.load(args.ids, map_location='cpu', weights_only=True)[:4, :64].clone()
    assert ids.shape == (4, 64)
    torch.save(ids, output/'calibration_ids.pt')
    report['input_sha256'] = digest(ids)
    student, meta, loaded = load_v6_model_checkpoint(str(args.checkpoint), map_location='cpu', strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    student.eval().requires_grad_(False)
    targets = runtime.targets_by_block(student, meta)[9]
    block = student.model.layers[9]
    teacher = get_model(meta['base_model_path']).eval().requires_grad_(False)
    old_hidden, old_kwargs = old_runtime.first_inputs(teacher, ids, 2, 'cuda:0')
    hidden, kwargs = runtime.first_inputs(teacher, ids, 2, 'cuda:0')
    assert_tree_equal(old_hidden, hidden)
    assert_tree_equal(old_kwargs, kwargs)
    report['first_inputs_exact'] = True
    del old_hidden, old_kwargs
    for index in range(9):
        hidden = runtime.teacher_outputs(teacher.model.layers[index], hidden, kwargs, 2, 'cuda:0')
    truth = runtime.teacher_outputs(teacher.model.layers[9], hidden, kwargs, 2, 'cuda:0')
    del teacher
    gc.collect()
    initial = {name: tensor.detach().cpu().clone() for name, tensor in block.state_dict().items()}
    module = targets[0][1].cuda()
    report['projection'] = run_projection_checks(module, old_bits)
    module.cpu()
    block.load_state_dict(initial, strict=True)
    runtime.clear_caches(block)
    torch.cuda.empty_cache()
    train_args = SimpleNamespace(nsamples=4, holdout=2, batch_size=2, epochs=33,
                                 code_lr=2e-5, decoder_lr=1.25e-5, audit_every=32)
    block.to('cuda:0')
    gpu_kwargs = runtime.tree_to(kwargs, 'cuda:0')
    split = block_train.measure_split(block, hidden, truth, gpu_kwargs, 2, 2, 'cuda:0')
    previous = dict(mse=old_loop.measure(block, hidden, truth, gpu_kwargs, 2, 'cuda:0'),
        train_mse=old_loop.measure(block, hidden[:2], truth[:2], gpu_kwargs, 2, 'cuda:0'),
        holdout_mse=old_loop.measure(block, hidden[2:], truth[2:], gpu_kwargs, 2, 'cuda:0'))
    assert split == previous, (split, previous)
    report['single_pass_mse_exact'] = split
    runtime.clear_caches(block)
    block.cpu()
    torch.cuda.empty_cache()
    old_record, old_output, old_opt, old_seconds = train_with_final_state_trace(
        old_loop, block, targets, hidden, truth, kwargs, train_args)
    old_final = {name: digest(tensor) for name, tensor in block.state_dict().items()}
    block.load_state_dict(initial, strict=True)
    # A grouped packed cache may outlive load_state_dict; refresh it explicitly.
    for _, module in targets:
        grouped = getattr(module, '_parallel_stage_grouped_vq_packed', None)
        if grouped is not None:
            grouped.copy_(module.get_stage_part_vq_storage(stage_idx=0, part_idx=0))
    runtime.clear_caches(block)
    block.eval().requires_grad_(False)
    new_record, new_output, new_opt, new_seconds = train_with_final_state_trace(
        block_train, block, targets, hidden, truth, kwargs, train_args)
    new_final = {name: digest(tensor) for name, tensor in block.state_dict().items()}
    assert old_opt == new_opt, 'Final FP32 proxy/decoder/gradient/Adam states differ.'
    assert old_final == new_final, 'Final native block state differs.'
    assert_tree_equal(old_output, new_output)
    for old, new in zip(old_record['steps'], new_record['steps']):
        assert old['loss'] == new['loss'] and old['learning_rates'] == new['learning_rates']
        for path in old['audit']:
            for name in ('changed_packed_bytes', 'changed_packed_bits'):
                assert old['audit'][path][name] == new['audit'][path][name]
    for name in ('before_mse','before_train_mse','before_holdout_mse','after_mse','after_train_mse','holdout_mse','hard_to_native_max_abs'):
        assert old_record[name] == new_record[name], (name, old_record[name], new_record[name])
    full_audits = [r['step'] for r in new_record['steps'] if 'proxy_delta_max' in next(iter(r['audit'].values()))]
    assert full_audits == [1, 32, 33], full_audits
    report['real_block'] = dict(status='PASS', steps=33, seqlen=64, train_samples=2, heldout_samples=2,
        exact_loss_lr_flip_trace=True, exact_final_proxy_gradients_adam=True, exact_native_state_output=True,
        full_audit_steps=full_audits, old=old_record, optimized=new_record,
        elapsed_excluding_trace_seconds=dict(reference=old_seconds, optimized=new_seconds),
        timing_caution='One reference-then-optimized run includes cache effects; use interleaved component timing for performance claims.')
    report.update(status='PASS', max_memory_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  max_memory_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  device=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    save()
    print(json.dumps({k:v for k,v in report.items() if k in ('status','max_memory_allocated_gib','scope')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

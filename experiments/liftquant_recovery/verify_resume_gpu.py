"""Real Qwen block 9/10 completed-boundary restart regression (GPU smoke only)."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from experiments.liftquant_recovery.block_train import train_block
from experiments.liftquant_recovery.recovery_resume import (
    boundary_identity, capture_rng, checkpoint_fingerprint, load_boundary,
    mutable_names_by_block, restore_rng, save_boundary, teacher_fingerprint, training_fingerprint,
)
from experiments.liftquant_recovery.recovery_runtime import (
    audit_topology, block_output, clear_caches, first_inputs, frozen_digest,
    prime_packed_cache, targets_by_block, teacher_outputs, tensor_digest, tree_to,
)
from rotation.model_utils import get_model
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ids', required=True, help='Existing real calibration_ids.pt; use first 4 documents x 64 tokens.')
    parser.add_argument('--output', required=True, help='New directory; contains one small boundary and validation.json.')
    return parser.parse_args()


def _block_state(block):
    return {name: value.detach().cpu().clone() for name, value in block.state_dict().items()}


def _assert_state(block, expected):
    actual = block.state_dict()
    if set(actual) != set(expected):
        raise AssertionError('Restart changed native block state names.')
    for name, value in expected.items():
        if actual[name].dtype != value.dtype or not torch.equal(actual[name].cpu(), value):
            raise AssertionError(f'Restart native state differs: {name}')
    return {name: tensor_digest(value) for name, value in expected.items()}


def _assert_rng(actual, expected):
    for key in ('python', 'numpy'):
        if actual[key] != expected[key]:
            raise AssertionError(f'Restart {key} RNG differs.')
    for key in ('torch_cpu', 'torch_cuda'):
        if actual[key] is None or expected[key] is None:
            if actual[key] is not expected[key]:
                raise AssertionError(f'Restart {key} RNG device differs.')
        elif not torch.equal(actual[key], expected[key]):
            raise AssertionError(f'Restart {key} RNG differs.')


def _trajectory(record):
    return [dict(step=item['step'], epoch=item['epoch'], loss=item['loss'],
                 learning_rates=item['learning_rates']) for item in record['steps']]


def run(cli):
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('This regression requires the authorized CUDA smoke environment.')
    device = torch.device('cuda:0')
    cap = 12 * 2**30 / torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(cap, 1.0), device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    ids = torch.load(cli.ids, map_location='cpu', weights_only=True)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] < 4 or ids.shape[1] < 64:
        raise ValueError('Expected at least four real calibration sequences of length 64.')
    if ids.dtype != torch.int64:
        raise ValueError('Expected int64 token IDs.')
    ids = ids[:4, :64].contiguous().clone()
    config = SimpleNamespace(
        checkpoint=str(Path(cli.checkpoint).resolve()), output=str(output), resume=None,
        blocks='9,10', nsamples=4, seqlen=64, batch_size=2, epochs=1, holdout=2,
        seed=42, code_lr=2e-5, decoder_lr=1.25e-5, smoke_rows=None,
        redpajama_arrow_dir=None, gpu_memory_gib=12, audit_every=32,
    )
    print("Hashing the actual initial checkpoint payload.", flush=True)
    source = checkpoint_fingerprint(config.checkpoint)
    code = training_fingerprint(Path(__file__).resolve().parents[2])
    selected = [9, 10]
    model, meta, loaded = load_v6_model_checkpoint(config.checkpoint, map_location='cpu', strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError('Initial native checkpoint did not load strictly.')
    model.eval().requires_grad_(False)
    print("Native model loaded; hashing the actual FP teacher payload.", flush=True)
    teacher_payload = teacher_fingerprint(meta["base_model_path"])
    identity = boundary_identity(config, selected, source, code, ids, teacher_fingerprint=teacher_payload)
    targets = targets_by_block(model, meta)
    if any(index not in targets for index in selected):
        raise ValueError('The supplied checkpoint must contain compressed blocks 9 and 10.')
    allowed_ids = audit_topology(model, meta, selected, targets)
    allowed_names = {name for name, value in model.state_dict(keep_vars=True).items()
                     if id(value) in allowed_ids}
    names_by_block = mutable_names_by_block(allowed_names, selected)
    frozen_before = frozen_digest(model, allowed_names)
    initial = {index: _block_state(model.model.layers[index]) for index in selected}

    teacher = get_model(meta['base_model_path']).eval().requires_grad_(False)
    hidden, kwargs = first_inputs(teacher, ids, config.batch_size, device)
    samples = {}
    for index in range(11):
        target = teacher_outputs(teacher.model.layers[index], hidden, kwargs, config.batch_size, device)
        if index in selected:
            samples[index] = (hidden, target)
        hidden = target
    del teacher, hidden, target
    gc.collect()
    print('Real teacher inputs and targets prepared for blocks 9/10.', flush=True)

    record9, reference9 = train_block(model.model.layers[9], targets[9], *samples[9], kwargs, config, device)
    records = {'9': record9}
    reload_inputs = {9: samples[9][0][:config.batch_size].clone()}
    reload_outputs = {9: reference9}
    boundary_path = output / 'latest_boundary.pt'
    boundary_rng = capture_rng(device)
    boundary_info = save_boundary(boundary_path, model, identity, names_by_block,
                                  [9], records, reload_inputs, reload_outputs, device)
    _assert_rng(capture_rng(device), boundary_rng)
    record10, reference10 = train_block(model.model.layers[10], targets[10], *samples[10], kwargs, config, device)
    continuous = {index: _block_state(model.model.layers[index]) for index in selected}
    continuous_rng = capture_rng(device)
    if frozen_digest(model, allowed_names) != frozen_before:
        raise AssertionError('Continuous training changed frozen native tensors.')
    print('Continuous two-block endpoint captured; restoring the initial native state.', flush=True)

    for index in selected:
        model.model.layers[index].load_state_dict(initial[index], strict=True)
    refresh_vae_linear_runtime_after_state_load(model)
    for index in selected:
        _assert_state(model.model.layers[index], initial[index])
    # Fault injection: corrupt only independent, nonpersistent grouped-code
    # caches. This is not a learned bit flip or a change to native checkpoint state.
    injected_caches = 0
    for path, module in targets[9]:
        primary = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
        grouped = module._parallel_stage_grouped_vq_packed
        if grouped is None or grouped.data_ptr() == primary.data_ptr():
            raise AssertionError(f"{path}: expected an independent grouped packed cache for fault injection.")
        grouped.bitwise_xor_(1)
        if torch.equal(grouped, primary):
            raise AssertionError("Grouped-cache fault injection did not take effect.")
        injected_caches += 1
    _assert_state(model.model.layers[9], initial[9])
    print(f"Injected {injected_caches} stale grouped caches; loading the completed boundary.", flush=True)
    resumed = load_boundary(boundary_path, model, identity, names_by_block)
    refresh_vae_linear_runtime_after_state_load(model)
    for path, module in targets[9]:
        primary = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
        if not torch.equal(module._parallel_stage_grouped_vq_packed, primary):
            raise AssertionError(f"{path}: native refresh left a stale grouped packed cache.")
    if resumed['completed'] != [9] or resumed['records'] != records:
        raise AssertionError('Boundary completed-block inventory or metrics differ.')
    _assert_rng(resumed['rng'], boundary_rng)
    _assert_state(model.model.layers[9], continuous[9])
    _assert_state(model.model.layers[10], initial[10])
    torch.testing.assert_close(samples[9][0][:config.batch_size], resumed['reload_inputs'][9], rtol=0, atol=0)
    block = model.model.layers[9].to(device).eval()
    prime_packed_cache(block)
    with torch.no_grad():
        native = block_output(block, resumed['reload_inputs'][9].to(device), tree_to(kwargs, device)).cpu()
    torch.testing.assert_close(native, resumed['reload_outputs'][9], rtol=0, atol=0)
    clear_caches(block)
    block.cpu()
    torch.cuda.empty_cache()

    # Model loading / teacher-prefix replay may consume RNG. The real input/target
    # tensors above are deterministic and reused; restore at the saved boundary.
    random.random()
    np.random.random(7)
    torch.rand(7)
    torch.rand(7, device=device)
    restore_rng(resumed['rng'], device)
    _assert_rng(capture_rng(device), boundary_rng)
    print("Completed block native output and RNG restored exactly; continuing block 10.", flush=True)
    resumed10, resumed_reference10 = train_block(
        model.model.layers[10], targets[10], *samples[10], kwargs, config, device)
    _assert_rng(capture_rng(device), continuous_rng)
    if _trajectory(resumed10) != _trajectory(record10):
        raise AssertionError('Restart changed loss/LR/step trajectory.')
    torch.testing.assert_close(resumed_reference10, reference10, rtol=0, atol=0)
    endpoint_sha = {str(index): _assert_state(model.model.layers[index], continuous[index])
                    for index in selected}
    frozen_after = frozen_digest(model, allowed_names)
    if frozen_after != frozen_before:
        raise AssertionError('Restart changed frozen native tensors.')
    if checkpoint_fingerprint(config.checkpoint) != source:
        raise AssertionError('Initial checkpoint payload changed during the regression.')
    report = dict(
        status='PASS', stage='real completed-block CUDA restart regression',
        final_experiment_run=False, model_exported=False,
        identity=identity, source_ids=str(Path(cli.ids).resolve()),
        device=torch.cuda.get_device_name(device), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checks=dict(native_state_exact=True, completed_block_native_output_exact=True,
                    resumed_block_output_exact=True, loss_and_learning_rates_exact=True,
                    rng_exact=True, frozen_state_exact=True, source_payload_unchanged=True,
                    injected_stale_grouped_caches_repaired=injected_caches),
        continuous_block10_trajectory=_trajectory(record10),
        resumed_block10_trajectory=_trajectory(resumed10),
        endpoint_state_sha256=endpoint_sha,
        frozen_state_sha256=frozen_after,
        boundary=boundary_info,
        max_memory_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        limitation='One real update per block validates the restart path; it does not establish useful hard-code flips or downstream improvement.',
    )
    (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'checks', 'boundary', 'max_memory_allocated_gib')}), flush=True)


if __name__ == '__main__':
    run(arguments())

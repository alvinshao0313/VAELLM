"""Same native partial checkpoint consumer, full MMLU and train-only deployment errors."""
import argparse
import gc
import importlib.util
import logging
import re
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from . import bindings as b


def module(path, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(Path(path).parent))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


def checkpoint(arm):
    if arm == 'control':
        fresh = b.read(b.RESULT / 'control_fresh.json')
        b.require(fresh['status'] == 'verified_native_full_state_strict'
            and fresh['non_gate_after_o_complement_exact'] and fresh['non_gate_tensor_count'] == 2091,
            'Require the actual control native strict load and shared-parent complement proof.')
        return Path(b.sources()['control']['path'])
    report = b.read(b.OUTPUT / arm / 'formal/report.json')
    b.require(report['status'] == 'raw2_completed' and report['exit_code'] == 0, 'Require completed native raw2.')
    fresh = b.read(b.OUTPUT / arm / 'formal/strict_reload.json')
    b.require(fresh['status'] == 'verified_native_full_state_strict'
        and fresh['checkpoint_id'] == report['raw2']['checkpoint_id'] and fresh['full_state'] == report['state']
        and fresh['non_gate_after_o_complement_exact'] and fresh['non_gate_tensor_count'] == 2091,
        'Require the independent fresh strict readback before downstream consumption.')
    return Path(report['raw2']['path'])


def load(directory):
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    model, meta, loaded = load_v6_model_checkpoint(str(directory), base_model_path=str(b.HF), map_location='cpu', strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['lora_config'] is None,
        'Native consumer must load every persistent field strictly, without adapters.')
    model.eval().requires_grad_(False)
    tools = module(b.SNAPSHOT / 'tools/cat_eval.py', '_gate_hyperparam_native_prewarm')
    warmup = tools._prepare_model_for_eval(model, 'cuda', 8, logging.getLogger('gate_cohort'), {})
    expected = meta['converted_module_count']
    b.require(warmup['warmed'] == warmup['total'] == expected and not warmup['failed']
        and all(value.dtype == b.torch.float32 for name,value in model.named_buffers() if 'inv_freq' in name)
        and model.get_input_embeddings().weight.dtype == b.torch.bfloat16
        and model.config._attn_implementation == 'flash_attention_2', 'Original BF16/FA2/FP32-RoPE/group8 policy failed.')
    return model, meta, dict(checkpoint=b.event(directory), strict_missing=[], strict_unexpected=[],
        persistent_tensors=len(model.state_dict()), prewarm=warmup, future_dense_targets=len(meta['pending_dense_targets']),
        attn_implementation=model.config._attn_implementation, rope_dtype='torch.float32', weights_dtype='torch.bfloat16')


def mmlu(arm, output):
    from transformers import AutoTokenizer
    directory = checkpoint(arm)
    model, meta, audit = load(directory)
    b.validate_raw2(meta)
    b.require(audit['persistent_tensors'] == 2559, 'Native raw2 must contain all2559 persistent fields.')
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(b.HF, use_fast=True, trust_remote_code=True, local_files_only=True)
    api = module(b.TOPIC / 'lm_eval_sample_artifacts_20261010/eval_utils_with_samples.py', '_gate_hyperparam_hflm_samples')
    options = SimpleNamespace(tasks='mmlu', num_fewshot=0, batch_size='64', lm_limit=None,
        model_path=str(b.HF), eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(output.parent / 'mmlu_logs'), mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    result = api.run_lm_eval(model, tokenizer, options)
    artifacts = result['sample_artifacts']
    b.require(artifacts['total_count'] == 14042 and len(artifacts['tasks']) == 57
        and all(name.startswith('mmlu_') for name in artifacts['tasks']), 'Require complete official57-leaf/14042 MMLU.')
    b.write(output, dict(status='completed_full_mmlu', arm=arm, checkpoint_id=meta['checkpoint_id'],
        checkpoint_dir=str(directory), checkpoint_loader='v6', strict_load=True, load_audit=audit,
        checkpoint_format=meta['format'], base_model_path=str(b.HF), evals=dict(lm_eval=result),
        boundary='Mechanism screening at native180+future72dense raw2; not complete nine-task or <=2.6 product acceptance.'))


def readout(arm, output):
    directory, binding = checkpoint(arm), b.sources()
    parent = Path(binding['after_o']['path'])
    runtime = b.torch.load(parent / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    cache = runtime['activation_runtime']['cache']
    inputs = cache['input_ids'][:8]
    b.require(cache['dataset'] == 'alpaca=1' and len(inputs) == 8 and all(value.numel() == 256 for value in inputs),
        'Use fixed first8 genuine Alpaca train calibration inputs without selection.')
    import hashlib
    token_sha = hashlib.sha256(b''.join(value.contiguous().numpy().tobytes() for value in inputs)).hexdigest()
    del runtime
    teacher, _, parent_audit = load(parent)
    activations = {f'model.layers.{index}.mlp.gate_proj': [] for index in range(36)}
    handles = []
    for name in activations:
        def capture(_module, args, name=name):
            activations[name].append(args[0].detach().cpu())
        handles.append(teacher.get_submodule(name).register_forward_pre_hook(capture))
    with b.torch.inference_mode():
        for tokens in inputs:
            teacher(tokens.reshape(1, -1).to('cuda'), use_cache=False)
    for handle in handles:
        handle.remove()
    del teacher
    gc.collect()
    b.torch.cuda.empty_cache()
    model, meta, audit = load(directory)
    b.validate_raw2(meta)
    pmeta = b.read(parent / 'checkpoint_meta.json')
    original = b.torch.load(parent / pmeta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    rows, sums = [], dict(weight_error=0., weight_reference=0., weight_elements=0,
        output_error=0., output_reference=0., output_elements=0)
    with b.torch.inference_mode():
        for name, values in activations.items():
            linear = model.get_submodule(name)
            target = original[name + '.weight'].to('cuda')
            deployed = linear._cached_weight
            b.require(deployed is not None and deployed.dtype == b.torch.bfloat16 and deployed.shape == target.shape,
                'Use the actual BF16 weight cache produced by this native group8 prewarm.')
            delta = deployed.float()-target.float()
            werr, wref = delta.square().sum(dtype=b.torch.float64).item(), target.float().square().sum(dtype=b.torch.float64).item()
            oerr = oref = 0.
            count = 0
            for value in values:
                value = value.to('cuda')
                actual = linear(value)
                reference = b.torch.nn.functional.linear(value, target, linear.bias)
                oerr += (actual.float()-reference.float()).square().sum(dtype=b.torch.float64).item()
                oref += reference.float().square().sum(dtype=b.torch.float64).item()
                count += reference.numel()
            rows.append(dict(name=name, weight_MSE=werr/target.numel(), weight_NMSE=werr/wref,
                output_MSE=oerr/count, output_NMSE=oerr/oref, input_tokens=2048,
                weight_error_sum=werr, weight_reference_sum=wref, output_error_sum=oerr, output_reference_sum=oref))
            sums['weight_error'] += werr
            sums['weight_reference'] += wref
            sums['weight_elements'] += target.numel()
            sums['output_error'] += oerr
            sums['output_reference'] += oref
            sums['output_elements'] += count
            del target, deployed, delta
    log = b.COMMON / 'gate_raw2.log' if arm == 'control' else b.OUTPUT / arm / 'fit.log'
    text = log.read_text(encoding='utf-8')
    observations = [line for line in text.splitlines() if 'gate_proj.L0-35/stage' in line and ('recon=' in line or 'residual rms:' in line)]
    b.require(len(rows) == 36 and observations, 'Incomplete gate deployment or stage training readout.')
    b.write(output, dict(status='completed_deployment_readout', arm=arm, checkpoint_id=meta['checkpoint_id'],
        source_binding=binding, load_audit=audit, parent_load_audit=parent_audit,
        fixed_train_input=dict(dataset='alpaca=1', split='train', source='Existing native parent activation_runtime cache',
            sample_indices=list(range(8)), samples=8, seqlen=256, input_ids_sha256=token_sha,
            activation_source='Actual after_o model dense-gate input for each layer; identical fixed token inputs for all arms.'),
        deployed_native_BF16=dict(weight_MSE=sums['weight_error']/sums['weight_elements'],
            weight_NMSE=sums['weight_error']/sums['weight_reference'], output_MSE=sums['output_error']/sums['output_elements'],
            output_NMSE=sums['output_error']/sums['output_reference'], raw_sums=sums, per_gate=rows),
        stage_training_observations=dict(source_log=str(log), source_log_sha256=b.sha(log), lines=observations,
            boundary='Original per-stage normalized recon print and pre-fusion residual RMS; different units from full deployed BF16 weight/output errors.'),
        boundary='Fixed train-input local mechanism diagnostics. Lower MSE does not establish better MMLU or final product quality.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', *b.ARMS), required=True)
    parser.add_argument('--component', choices=('mmlu', 'readout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    b.require(not args.output.exists(), 'Never overwrite a downstream result.')
    b.check_pins(b.read(b.RESULT / 'prepared.json')['source_sha256'])
    b.torch.set_num_threads(8)
    globals()[args.component](args.arm, args.output)

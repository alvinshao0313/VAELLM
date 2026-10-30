"""G09 initial group0 M32: real live graph, same cached target, no optimizer."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/g09_recovery_20261029'))
import bindings as b

sys.path.insert(0, str(HERE))
import m_math

OUTPUT = Path('/data/star/VAELLM/softmax_numerics_20261029/M_group0_live')


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def implementation():
    paths = [Path(__file__).resolve(), HERE / 'm_math.py', HERE / 'numerics.py',
        Path(b.__file__), Path(b.fit_core.__file__), Path(b.mmlu.__file__), Path(b.core.__file__),
        b.OPD / 'native.py', b.SNAPSHOT / 'litebsq/autoencoder.py', b.SNAPSHOT / 'litebsq/vae_linear.py']
    return {str(path): b.binding(path) for path in paths}


def prepare():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(),
        'M preparation is CPU-only.')
    b.require(not (OUTPUT / 'cpu.json').exists() and not (OUTPUT / 'inputs.pt').exists(),
        'Do not overwrite an existing M preparation.')
    prepared, record = b.read_prepared()
    dataset, recovery, _, _ = b.load_inputs()
    group = record['groups'][0]
    keys = group['case_ids'][:32]
    cases = [recovery[key]['case'] for key in keys]
    target = b.torch.cat([recovery[key]['target_log_probs'] for key in keys])
    b.require(group['group_index'] == 0 and keys == dataset['groups'][0]['case_ids'][:32]
        and all(case['domain'] == 'mmlu' and case['source']['official_split'] == 'train'
            and case['case_id'].startswith('cais_mmlu/auxiliary_train/train/') for case in cases)
        and all(group['weights'][key] == .5/32 for key in keys)
        and target.shape == (32,4) and target.dtype == b.torch.float32 and not target.requires_grad,
        'Keep the exact original train-only group0 M32 mixed target.')
    screened_path = OUTPUT.parent / 'M_group0_cache_screen/report.json'
    screened = b.read(screened_path)
    b.require([row['case_id'] for row in screened['rows']] == keys
        and b.torch.equal(target, b.torch.tensor([row['mixed_logp'] for row in screened['rows']], dtype=b.torch.float32)),
        'Real once-mixed targets differ from the independently screened cache.')
    # Shared HF config determines the same pad as formal M; this performs no model load.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(record['model_config']['model']['hf_path'], local_files_only=True)
    pad = config.pad_token_id if config.pad_token_id is not None else 0
    inputs, options = b.mmlu.batch_inputs(cases, pad)
    # A CPU operator exercise on stored logp coordinates only, never called a student capture.
    exercise, _, _, _ = m_math.audit_logits(target.bfloat16(), target, b.mmlu.weighted_loss)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = dict(schema='M_group0_exact_inputs_v1', cases=cases, case_ids=keys, inputs=inputs,
        options=options, cached_mixed_FP32_logp=target, pad_id=pad,
        source_checkpoint_id=b.SOURCE_ID, formal_prepared=b.binding(b.OUTPUT / 'prepared.json'))
    b.torch.save(payload, OUTPUT / 'inputs.pt')
    record = dict(schema='M_group0_live_CPU_prepared_v1', status='CPU_inputs_and_math_verified_no_model',
        source_checkpoint_id=b.SOURCE_ID, formal_prepared=b.binding(b.OUTPUT / 'prepared.json'),
        source_audit=b.binding(b.SOURCE_AUDIT), original_cache_screen=b.binding(screened_path),
        inputs=b.binding(OUTPUT / 'inputs.pt'), implementation=implementation(),
        documents=32, support=4, case_ids=keys, batch_input_shape=list(inputs['input_ids'].shape),
        actual_input_checks='Original batch_inputs validates left padding, masks, positions, last query, four native continuation IDs.',
        mixture='Original load_inputs returns once-mixed full-F/gold .5/.5 FP32 logp; no new target mixing.',
        weight_per_document=.5/32, temperature=1, CUDA_initialized=b.torch.cuda.is_initialized(),
        CPU_operator_only=dict(input_kind='Stored mixed logp cast to BF16 coordinates, not student logits',
            analytic_vs_autodiff=exercise['FP64_analytic_vs_stable_autodiff']))
    b.core.write(OUTPUT / 'cpu.json', record)
    event('M_group0_CPU_prepared', source=b.SOURCE_ID, input_shape=record['batch_input_shape'])


def check_prepared():
    cpu = b.read(OUTPUT / 'cpu.json')
    b.require(cpu['schema'] == 'M_group0_live_CPU_prepared_v1' and cpu['source_checkpoint_id'] == b.SOURCE_ID,
        'Unexpected M input preparation.')
    for item in [*cpu['implementation'].values(), cpu['formal_prepared'], cpu['source_audit'], cpu['inputs']]:
        b.require(b.binding(item['path']) == item, 'Changed bound M diagnostic input/code: '+item['path'])
    prepared, record = b.read_prepared()
    payload = b.torch.load(cpu['inputs']['path'], map_location='cpu', weights_only=True)
    b.require(payload['schema'] == 'M_group0_exact_inputs_v1' and payload['case_ids'] == record['groups'][0]['case_ids'][:32],
        'Changed formal group0 selection/order.')
    return cpu, prepared, record, payload


def live_capture(cpu, record, payload):
    native = b.module(b.OPD / 'native.py', '_M_group0_same_native_loader')
    model, load_audit = native.load_student(record['model_config'])
    modules = b.selected_modules(model, record)
    initial = b.fit_core.copy_parameters(modules)
    frozen = b.fit_core.frozen_proof(model, set(initial))
    b.require(frozen == record['validated_frozen_state'], 'Actual G09 frozen complement differs.')
    policy = b.core.native_policy(model)
    active = b.fit_core.training_mode(model, modules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    b.require(len(modules) == 252 and len(active) == 1512 and set(active) == set(record['decoder_parameters'])
        and sum(p.numel() for p in active.values()) == record['trainable_elements']
        and model.is_gradient_checkpointing and model.model.gradient_checkpointing
        and all(module._parallel_stage_decoder.use_checkpoint and module.trainable_decode
            and not module.cache_decoded_weight and module._cached_weight is None
            and module._decoder_compute_dtype == b.torch.bfloat16 for module in modules.values()),
        'Require all252/1512 FP32 live decoder leaves and both original checkpoint levels, with no native weight cache.')
    before = {name: b.core.tensor_record(p) for name,p in active.items()}
    identities = {name: (id(p), p._version, p.data_ptr()) for name,p in active.items()}
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    inputs, options = b.mmlu.batch_inputs(payload['cases'], pad)
    b.require(pad == payload['pad_id'] and b.torch.equal(options, payload['options'])
        and all(b.torch.equal(value, payload['inputs'][key]) for key,value in inputs.items()),
        'Real model M padding/mask/positions/continuations differ from original CPU inputs.')
    device = model.get_input_embeddings().weight.device
    b.torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = model(**{key:value.to(device) for key,value in inputs.items()}, use_cache=False, logits_to_keep=1)
    b.require(result.past_key_values is None and result.logits.shape == (32,1,151936)
        and result.logits.dtype == b.torch.bfloat16, 'Actual live M32 BF16 last-query head differs.')
    selected = result.logits[:,0].gather(1, options.to(device))
    b.require(selected.requires_grad and selected.grad_fn is not None, 'The captured logits lost their real live model graph.')
    numeric, tensors, current_boundary, reference_boundary = m_math.audit_logits(
        selected, payload['cached_mixed_FP32_logp'], b.mmlu.weighted_loss)
    for row, case in zip(numeric['rows'], payload['cases'], strict=True):
        row.update(case_id=case['case_id'], input_signature=case['input_signature'], gold_index=case['gold_index'],
            native_continuation_ids=case['native_continuation_ids'].reshape(4).tolist())
    event('M32_live_logits_audited', formula=numeric['formula'], boundary=numeric['BF16_boundary'],
        boundary_exact=numeric['BF16_boundary_exact'], max_student_margin=numeric['student_max_margin'])
    parameter = None
    if not numeric['BF16_boundary_exact']:
        # Non-reentrant full-model and decoder checkpointing supports autograd.grad.
        # Both VJPs use this one graph and their complete BF16 output cotangents.
        names, parameters = list(active), tuple(active.values())
        current_gpu = b.torch.autograd.grad(selected, parameters, grad_outputs=current_boundary,
            retain_graph=True, create_graph=False, allow_unused=False)
        current = tuple(value.detach().cpu() for value in current_gpu)
        del current_gpu
        reference_gpu = b.torch.autograd.grad(selected, parameters, grad_outputs=reference_boundary,
            retain_graph=False, create_graph=False, allow_unused=False)
        reference = tuple(value.detach().cpu() for value in reference_gpu)
        del reference_gpu
        parameter = m_math.parameter_comparison(names, current, reference)
        event('M32_same_graph_two_VJPs_complete', **parameter['aggregate'])
    b.require(all(p.grad is None for p in model.parameters()), 'Read-only VJPs must not accumulate .grad or optimize.')
    after = {name: b.core.tensor_record(p) for name,p in active.items()}
    b.require(before == after and identities == {name:(id(p),p._version,p.data_ptr()) for name,p in active.items()},
        'A live master Parameter value/identity/version changed without an optimizer.')
    del result, selected, current_boundary, reference_boundary
    peak = dict(allocated=b.torch.cuda.max_memory_allocated(), reserved=b.torch.cuda.max_memory_reserved())
    b.fit_core.restore(model, initial)
    b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules), initial)
        and b.fit_core.frozen_proof(model, set(initial)) == frozen
        and b.core.native_policy(model) == policy, 'Actual BF16 source/frozen complement/policy restoration differs.')
    proof = dict(live_master_values_exact=True, Parameter_identity_and_version_unchanged=True,
        live_master_before_sha256=b.core.mapping_digest(before), live_master_after_sha256=b.core.mapping_digest(after),
        all1512_original_BF16_decoders_restored_exact=True, frozen_complement=frozen,
        optimizer_constructed=False, optimizer_steps=0, parameter_grad_accumulated=False,
        no_native_forward_comparison_added=True)
    weak = [weakref.ref(model), *[weakref.ref(p) for p in model.parameters()]]
    return dict(numerics=numeric, parameter_VJP=parameter, state_proof=proof, load_audit=load_audit,
        live_FWBW_peak_bytes=peak, live_capture_seconds=time.monotonic()-started), tensors, weak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve(), 'Use caller-activated bitvae.')
    b.torch.set_num_threads(4)
    if args.prepare_only:
        prepare()
        return
    b.require(args.gpu == 7 and os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
        and b.torch.cuda.device_count() == 1 and Path.cwd() == b.SNAPSHOT,
        'Use the assigned empty GPU7 and established snapshot cwd.')
    b.require(not (OUTPUT / 'report.json').exists(), 'Do not overwrite an existing M diagnostic.')
    cpu, prepared, record, payload = check_prepared()
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    report, tensors, weak = live_capture(cpu, record, payload)
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()
    b.require(all(value() is None for value in weak) and b.torch.cuda.memory_allocated() == 0,
        'Release the full diagnostic model/graph before completing.')
    for item in cpu['implementation'].values():
        b.require(b.binding(item['path']) == item, 'Bound implementation changed during diagnostic.')
    b.torch.save(dict(schema='M_group0_live_logit_capture_v1', **tensors), OUTPUT / 'logits.pt')
    report.update(schema='M_group0_live_numeric_audit_v1', status='complete_read_only_live_M32_diagnostic',
        source_checkpoint_id=b.SOURCE_ID, CPU_preparation=b.binding(OUTPUT / 'cpu.json'),
        formal_prepared=cpu['formal_prepared'], implementation=cpu['implementation'],
        raw_logits=b.binding(OUTPUT / 'logits.pt'), input_payload=cpu['inputs'],
        execution=dict(pid=os.getpid(), argv=sys.argv, python=sys.executable, cwd=str(Path.cwd()),
            assigned_physical_GPU=7, torch_version=b.torch.__version__, elapsed_seconds=time.monotonic()-started,
            GPU_allocated_after_release=0),
        protocol=dict(group_index=0, documents=32, support=4, weight_per_document=.5/32, temperature=1,
            batch_input_shape=cpu['batch_input_shape'], student='G09 actual initial all252/1512 live FP32 master; BF16 compute/head; both non-reentrant checkpoints'),
        limits='Only this source/group0 M32. No QA/R expansion, head-accumulator comparison, Adam update, or nine-task quality claim.')
    b.core.write(OUTPUT / 'report.json', report)
    event(report['status'], output=str(OUTPUT), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()

"""Actual B full-shape math/interface checks; no GPU, quality or C0 claim."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as b
from .free_dense import FreeDenseLinear, deployed_weight, protection_binding
from experiments.adaptive_native_model_20261029.native_prefix import from_native

torch = b.torch
CASES = (('native_gate', 'model.layers.0.mlp.gate_proj', 'native'),
         ('native_down', 'model.layers.35.mlp.down_proj', 'native'),
         ('compact_gate_interface', 'model.layers.0.mlp.gate_proj', 'compact'))


def actual_consumer(model, meta, name, kind):
    original = model.get_submodule(name)
    if kind == 'native':
        b.require(isinstance(original,b.VAELinear), 'The actual B representative must be native.')
        return original
    spec = next(item for item in meta['converted_modules'] if item['name'] == name)
    bundle, lengths, _resolved = from_native(spec, original.state_dict(),
        dict(source_checkpoint_id=meta['checkpoint_id'], scope='CPU compact interface only; no C0 or quality substitution'))
    return b.CompactRVQLinear(bundle,lengths).eval().requires_grad_(False)


def tensor_state(state):
    return {name:value for name,value in state.items() if torch.is_tensor(value)}


def verify_case(module, binding, name):
    protection_before = protection_binding(module)
    original_state = b.digest(module.state_dict())
    if isinstance(module,b.VAELinear):
        b.install_precision_runtime(module,torch.bfloat16)
        b.prime_named_vae_linear_cache([(name,module)],dtype=torch.bfloat16,clear_existing=True,group_size=8)
    dense = FreeDenseLinear.from_compressed(module,binding,name)
    initial = dense.active_weight.detach().clone()
    fixed = {key:value.clone() for key,value in dense.named_buffers()}
    if isinstance(module,b.VAELinear):
        b.prime_named_vae_linear_cache([(name,module)],dtype=torch.bfloat16,clear_existing=True,group_size=8)
    weight = deployed_weight(module,name)
    b.require(torch.equal(dense.decode_weight(),weight), 'Initialization differs from the native/compact deployed W0.')
    torch.manual_seed(137)
    inputs = torch.randn((2,module.in_features),dtype=torch.bfloat16,requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_()
    output = dense(inputs)
    source_output = module(reference_input)
    b.require(torch.equal(output,source_output), 'Initial one-GEMM output differs from the actual deployment consumer.')
    upstream = torch.randn(output.shape,dtype=torch.bfloat16)
    output.backward(upstream)
    source_output.backward(upstream)
    b.require(torch.equal(inputs.grad,reference_input.grad), 'The input autograd boundary changed.')
    expected = upstream.t().matmul(inputs.detach()).index_select(1,dense.unprotected_indices).float()
    b.require(torch.equal(dense.active_weight.grad,expected), 'Active gradient is not the original full-GEMM column complement.')
    b.require(all(p.grad is None for p in module.parameters()) and all(not value.requires_grad for value in dense.buffers()),
              'The original compressed consumer or protected buffers acquired gradients.')
    del output,source_output,inputs,reference_input,expected,weight,upstream
    module.clear_decoded_weight_cache()
    optimizer = torch.optim.Adam([dense.active_weight],lr=1e-5,betas=(.9,.95),eps=1e-8,
                                 weight_decay=0.,foreach=False)
    steps = []
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn((2,module.in_features),dtype=torch.bfloat16)
        loss = dense(x).float().square().mean()
        b.require(torch.isfinite(loss).item(), 'Nonfinite real-shape CPU objective.')
        loss.backward()
        grad = dense.active_weight.grad
        b.require(grad is not None and grad.dtype == torch.float32 and torch.isfinite(grad).all().item()
                  and grad.count_nonzero().item()>0, 'Missing/nonfinite actual free-dense gradient.')
        norm = float(torch.nn.utils.clip_grad_norm_([dense.active_weight],1.5,error_if_nonfinite=True))
        optimizer.param_groups[0]['lr'] = 1e-5*(1-(step-1)/500)
        optimizer.step()
        state = optimizer.state[dense.active_weight]
        b.require(int(state['step']) == step and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
                  and torch.isfinite(dense.active_weight).all().item(), 'FP32 Adam state/update contract failed.')
        b.require(all(torch.equal(dict(dense.named_buffers())[key],value) for key,value in fixed.items()),
                  'A protected column, index set, complement/template or bias changed during optimization.')
        steps.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm,lr=optimizer.param_groups[0]['lr']))
        del x,loss,grad
    b.require(b.digest(module.state_dict()) == original_state and protection_binding(module) == protection_before,
              'CPU mechanism checks changed the original compressed source/protection fields.')
    fp32_changed = int((dense.active_weight.detach()!=initial).sum())
    bf16_changed = int((dense.active_weight.detach().bfloat16()!=initial.bfloat16()).sum())
    b.require(fp32_changed>0 and bf16_changed>0, 'The real-shape updates must change FP32 masters and deployed BF16 values.')
    dense.validate_structure()
    probe = torch.randn((2,module.in_features),dtype=torch.bfloat16)
    with torch.no_grad(): expected_output = dense(probe)
    payload = dict(source=deepcopy_source(dense.source),state={key:(value.detach().cpu().clone() if torch.is_tensor(value) else value)
                                                           for key,value in dense.state_dict().items()},
                   probe=probe,expected_output=expected_output)
    row = dict(shape=[module.out_features,module.in_features],active_shape=list(dense.active_weight.shape),
        active_elements=dense.active_weight.numel(),protected_columns=dense.protected_indices.numel(),
        initial_actual_BF16_W0_exact=True,initial_actual_consumer_output_exact=True,input_gradient_exact=True,
        active_gradient_full_GEMM_complement_exact=True,protected_and_source_state_exact=True,
        one_trainable_parameter=True,trainable_dtype='torch.float32',deployed_dtype='torch.bfloat16',
        fp32_changed_elements=fp32_changed,bf16_changed_elements=bf16_changed,two_updates=steps,
        state_tensor_sha256=b.digest(tensor_state(payload['state'])),source=payload['source'])
    del optimizer,dense,initial,fixed
    gc.collect()
    return row,payload


def deepcopy_source(value):
    from copy import deepcopy
    return deepcopy(value)


def fresh(path):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    model,meta,_measured = b.load_source(saved['source_binding'])
    rows = []
    for label,name,kind in CASES:
        payload = saved['cases'][label]
        module = actual_consumer(model,meta,name,kind)
        if isinstance(module,b.VAELinear):
            b.install_precision_runtime(module,torch.bfloat16)
            b.prime_named_vae_linear_cache([(name,module)],dtype=torch.bfloat16,clear_existing=True,group_size=8)
        dense = FreeDenseLinear.from_compressed(module,saved['source_binding'],name)
        b.require(dense.source == payload['source'], 'Fresh actual compressed-source/protection/W0 binding differs.')
        loaded = dense.load_state_dict(payload['state'],strict=True)
        dense.validate_structure()
        b.require(not loaded.missing_keys and not loaded.unexpected_keys
                  and b.digest(tensor_state(dense.state_dict())) == b.digest(tensor_state(payload['state'])),
                  'Fresh strict module state differs from the saved FP32 master/BF16 protection state.')
        with torch.no_grad(): actual = dense(payload['probe'])
        b.require(torch.equal(actual,payload['expected_output']), 'Fresh same-input BF16 output differs.')
        independent = FreeDenseLinear.from_state(payload['state'],dense.source)
        with torch.no_grad(): independent_output = independent(payload['probe'])
        b.require(torch.equal(independent_output,actual), 'Standalone strict state reload changed deployed BF16 output.')
        bad_source = deepcopy_source(dense.source)
        bad_source['source_checkpoint_id'] = 'wrong-source-must-fail'
        try:
            independent.set_extra_state(bad_source)
        except ValueError:
            rejected = True
        else:
            rejected = False
        b.require(rejected, 'Reload accepted a different original compressed source.')
        rows.append(dict(label=label,fresh_strict_state_exact=True,same_input_BF16_output_exact=True,
                         independently_rederived_source_exact=True,standalone_state_exact=True,wrong_source_rejected=True))
        del module,dense,independent,payload,actual,independent_output
        gc.collect()
    b.write(Path(path).with_suffix('.fresh.json'),dict(status='verified',cases=rows,CUDA_initialized=torch.cuda.is_initialized()))
    print('fresh actual-source CPU strict reload verified',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fresh',type=Path)
    args = parser.parse_args()
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU validation must have no visible CUDA devices.')
    torch.set_num_threads(4)
    if args.fresh:
        fresh(args.fresh)
        return
    b.require(args.output is not None and not args.output.exists(), 'Choose a new CPU result directory.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    binding = b.source(b.SOURCE_B,prototype_only=True)
    print('bound actual native B '+binding['checkpoint_id'],flush=True)
    model,meta,_measured = b.load_source(binding)
    inventory = b.mlp_inventory(model)
    print('strict full B load and actual108 MLP inventory verified',flush=True)
    rows,cases = {},{}
    for label,name,kind in CASES:
        print('checking '+label+' '+name,flush=True)
        module = actual_consumer(model,meta,name,kind)
        row,payload = verify_case(module,binding,name)
        rows[label],cases[label] = row,payload
        del module
        print(label+' real-shape two updates verified',flush=True)
    transient = args.output/'cpu_cases.pt'
    torch.save(dict(source_binding=binding,cases=cases),transient)
    del cases,model
    gc.collect()
    result = subprocess.run([sys.executable,'-m','experiments.mlp_learnability_cmcc_20261030.validate_cpu','--fresh',str(transient)],
                            stdin=subprocess.DEVNULL,check=False)
    b.require(result.returncode == 0, 'Independent actual-source CPU reload failed.')
    reloaded = b.read(transient.with_suffix('.fresh.json'))
    b.require(reloaded['status'] == 'verified' and not reloaded['CUDA_initialized'], 'Fresh process initialized CUDA or did not verify.')
    report = dict(schema='all_MLP_free_dense_CPU_v1',status='CPU_real_shape_mechanics_verified_GPU_pending',
        source_binding=binding,full_source_strict_load=True,inventory=inventory,cases=rows,independent_fresh=reloaded,
        active_elements=sum(row['active_elements'] for row in inventory),
        full_BF16_template_bytes=sum(row['full_BF16_template_bytes'] for row in inventory),
        fp32_parameter_gradient_Adam_mv_bytes=sum(row['active_elements'] for row in inventory)*16,
        source_sha256={str(path):b.sha(path) for path in b.HERE.glob('*.py')},
        CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
        scope='Actual complete native B inventory; representative native gate/down and read-only imported compact gate interface. No GPU, all108 installation, full-model backward, fresh C0, scientific gain or exact-resume claim.',
        GPU_pending=['All108 dense installation from actual source and non-MLP freezing',
            'Real two-GPU layer map0-17/18-35 and autograd crossing',
            'All108 finite gradients/Adam updates and measured per-device peak resources',
            'Full-model two-update objective/long-sequence data, save/fresh contract'])
    b.write(args.output/'report.json',report)
    removed_bytes = transient.stat().st_size+transient.with_suffix('.fresh.json').stat().st_size
    transient.unlink()
    transient.with_suffix('.fresh.json').unlink()
    print('CPU mechanics verified; transient test payload removed '+str(removed_bytes)+' bytes',flush=True)
    print('report '+str(args.output/'report.json'),flush=True)


if __name__ == '__main__':
    main()

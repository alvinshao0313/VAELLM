"""Actual symmetric-up owners, original initialization and component Adam checks."""
import ast
from copy import deepcopy
import io
import hashlib
from pathlib import Path
import sys
import time

from . import bindings as b
from .optimizer import VAEMuon, derive_layout, zeropower
from litebsq.llm_vae import MultiLayerVAE
from transformers import get_scheduler


def owner_checks(parsed, source):
    args = b.stage_args(parsed)
    b.recovery.native.set_seed(31)
    model = MultiLayerVAE(args)
    named, layout, affine = derive_layout(model)
    digest = b.c.digest({f'group0.parameter{index}': value.detach() for index,value in enumerate(model.parameters())})
    b.require(digest == source['original_up_fit']['actual_optimizer_initializations'][0]['initial_parameter_digest'],
        'Real CPU symmetric-up stage0 initialization must match original C0 exactly.')
    generator = b.torch.Generator().manual_seed(71431)
    for spec in layout.values():
        value = b.torch.randn(spec['matrix_shape'],generator=generator)
        original = zeropower(value)
        changed = value.clone();changed[7].add_(.375)
        keep = b.torch.arange(36) != 7
        b.require(b.torch.equal(zeropower(changed)[keep],original[keep]), 'Up owner matrices must remain independent.')
    evidence = {}
    optimizer = VAEMuon(model,args,b.recovery.native.create_optimizer,evidence,diagnostic_steps=(1,2))
    scheduler = get_scheduler('linear',optimizer,num_warmup_steps=0,num_training_steps=10000)
    shadow = [value.detach().clone().requires_grad_() for value in model.parameters()]
    original = b.recovery.native.create_optimizer(shadow,args,args.lr)
    aux_indices = [index for index,(name,value) in enumerate(named.items()) if name in affine]
    for step in range(2):
        for actual, reference in zip(model.parameters(),shadow,strict=True):
            grad = b.torch.randn(actual.shape,generator=generator).to(b.torch.bfloat16)
            actual.grad, reference.grad = grad.clone(), grad.clone()
        for group in original.param_groups:
            group['lr'] = .003 * (1-step/10000)
        optimizer.step();original.step();scheduler.step()
        b.require(all(b.torch.equal(list(model.parameters())[index],shadow[index]) for index in aux_indices),
            'Actual up auxiliary Adam parameters must match original native BF16 AdamW.')
    buffer = io.BytesIO();b.torch.save(optimizer.state_dict(),buffer);buffer.seek(0)
    state = b.torch.load(buffer,map_location='cpu',weights_only=True)
    restored_model = MultiLayerVAE(args);restored_model.load_state_dict(model.state_dict(),strict=True)
    restored = VAEMuon(restored_model,args,b.recovery.native.create_optimizer,{},diagnostic_steps=(1,2))
    restored.load_state_dict(state)
    left = [value for group in optimizer.param_groups for value in group['params']]
    right = [value for group in restored.param_groups for value in group['params']]
    fields = 0
    for actual, loaded in zip(left,right,strict=True):
        for key,value in optimizer.state[actual].items():
            other = restored.state[loaded][key]
            b.require(value.dtype == other.dtype and b.torch.equal(value,other) if isinstance(value,b.torch.Tensor) else value == other,
                'Actual up optimizer state dtype/value changed through save/load.')
            fields += 1
    b.require(restored.param_groups[0]['kappa'] == optimizer.param_groups[0]['kappa'] and restored.actual_step == 2,
        'Up stage kappa/update counter must restore exactly.')
    return dict(stage0_original_initialization_digest=digest, actual_parameters=619776,
        matrix_elements=589824, auxiliary_elements=29952, actual_owner_layout=layout,
        independent144_matrices=True, auxiliary_original_Adam_two_updates_exact=True,
        optimizer_state_roundtrip_fields=fields, optimizer_dtypes_and_values_exact=True,
        CPU_calibration=evidence['calibration'], CPU_updates=evidence['updates'],
        CPU_only=True, CPU_kappa_not_reused_for_GPU=True)


def main():
    b.cpu_environment()
    b.torch.set_num_threads(8)
    b.require(not (b.RESULT / 'prepared.json').exists(), 'Never replace an earlier CPU source binding.')
    began = time.time()
    source = b.sources()
    parent = b.torch.load(b.PARENT / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    targets = {name.removesuffix('.weight'):value for name,value in parent.items() if name.endswith('.mlp.up_proj.weight')}
    prefix = {name:value for name,value in parent.items() if '.mlp.up_proj.' not in name}
    b.require(len(targets) == 36 and all(tuple(value.shape) == (12288,4096) for value in targets.values())
        and b.c.digest(targets) == source['original_up_fit']['source_target_digest'], 'Actual live up BF16 tensors differ from C0.')
    parsed, identity, baseline = b.recipe('muon',b.OUTPUT / 'not_executed')
    b.require(identity == baseline, 'Muon preserves the original complete native runtime identity.')
    owner = owner_checks(parsed,source)
    d16, d16_identity, _ = b.recipe('d16',b.OUTPUT / 'not_executed_d16')
    b.recovery.native.set_seed(31)
    model = MultiLayerVAE(b.stage_args(d16))
    b.require(sum(value.numel() for value in model.parameters()) == 323712
        and all(value.dtype == b.torch.bfloat16 for value in model.parameters()), 'The actual new D16/K16 symmetric factory differs.')
    d16_initialization = b.c.digest({f'group0.parameter{index}':value.detach() for index,value in enumerate(model.parameters())})
    runtime = b.torch.load(Path(source['common']['source_raw2']['path']) / 'cat_runtime_state.pt',map_location='cpu',weights_only=True)
    inputs = runtime['activation_runtime']['cache']['input_ids'][:8]
    b.require(len(inputs) == 8 and all(value.numel() == 256 for value in inputs), 'Freeze the actual original first8 train blocks.')
    input_sha = hashlib.sha256(b''.join(value.contiguous().numpy().tobytes() for value in inputs)).hexdigest()
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    b.write(b.RESULT / 'prepared.json',dict(status='cpu_verified',exit_code=0,source_binding=source,
        contract=b.CONTRACT,frozen_prefix_digest=b.c.digest(prefix),live_up_source_digest=b.c.digest(targets),
        fixed_train_input_sha256=input_sha,
        original_runtime_identity=identity,d16_runtime_identity=d16_identity,Muon_owner_CPU=owner,
        d16_factory=dict(parameters=323712,seed_note='Independent changed-geometry factory; no exact D32 initialization claim',
            initial_digest=d16_initialization), source_sha256=b.pins(),
        CUDA_initialized=False,elapsed_seconds=time.time()-began,command=[sys.executable,*sys.argv]))
    print(dict(status='cpu_verified',seconds=time.time()-began),flush=True)


if __name__ == '__main__':main()

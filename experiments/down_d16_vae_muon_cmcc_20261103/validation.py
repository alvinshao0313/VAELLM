"""Observe only the two genuine new-shape optimizer updates per stage."""
from contextlib import contextmanager

from . import bindings as b
from litebsq.bitpack import pack_bool_tensor_to_uint8,unpack_uint8_tensor_to_bool


@contextmanager
def updates(native, *, enabled):
    rows,original = [],native.create_optimizer

    def create(*args,**kwargs):
        optimizer = original(*args,**kwargs)
        row = dict(stage=len(rows),updates=[])
        rows.append(row)
        step = optimizer.step

        def observe(_optimizer,*step_args,**step_kwargs):
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            gradients = [p.grad for p in parameters if p.grad is not None]
            b.require(gradients and all(b.torch.isfinite(value).all().item() for value in gradients),
                'New down shape has missing or nonfinite gradients.')
            norm = sum(value.detach().double().square().sum().item() for value in gradients)
            b.require(norm > 0,'New down shape has zero total gradient.')
            before = [p.detach().clone() for p in parameters]
            result = step(*step_args,**step_kwargs)
            changed = sum(b.torch.count_nonzero(p.detach() != old).item() for p,old in zip(parameters,before,strict=True))
            b.require(changed > 0 and all(b.torch.isfinite(p).all().item() for p in parameters),
                'New down shape did not update finite parameters.')
            row['updates'].append(dict(step=len(row['updates'])+1,gradient_squared_norm=norm,
                changed_parameter_elements=changed,finite_gradients=True,finite_parameters=True))
            return result

        optimizer.step = observe.__get__(optimizer,type(optimizer))
        return optimizer

    if enabled:
        native.create_optimizer=create
    try:
        yield rows
    finally:
        native.create_optimizer=original


@b.torch.no_grad()
def codes(model):
    rows=[]
    for layer in range(36):
        name=f'model.layers.{layer}.mlp.down_proj'
        module=model.get_submodule(name)
        b.require(module.codebook_dim==16 and module.residual_stages==2 and module.parallel_parts==1,
            'Actual Muon fit must retain the original D16 two-stage code topology.')
        for stage in range(2):
            storage=module.get_stage_part_vq_storage(stage).detach().cpu()
            logical=tuple(module.get_stage_part_vq_spec(stage)['logical_shape'])
            b.require(logical==(3080192,1,24) and tuple(storage.shape)==(3080192,1,3),
                'The actual D16/K24 packed code geometry changed.')
            decoded=unpack_uint8_tensor_to_bool(storage,logical_shape=logical)
            b.require(b.torch.equal(storage,pack_bool_tensor_to_uint8(decoded,logical_shape=logical)),
                'Actual D16 packed payload failed exact native bit readback/repack.')
            rows.append(dict(name=name,stage=stage,logical_shape=list(logical),packed_shape=list(storage.shape),
                packed_digest=b.c.digest({'code':storage}),all_bits_exact_roundtrip=True))
    return dict(actual_stages=72,all_packed_bits_roundtrip_exact=True,rows=rows)

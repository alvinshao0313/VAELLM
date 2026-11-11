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
        b.require(module.residual_stages==2 and module.parallel_parts==1 and module.codebook_dim==8,
            'Real new payload must retain logical D8 and exactly two stages.')
        for stage in range(2):
            storage=module.get_stage_part_vq_storage(stage).detach().cpu()
            spec=module.get_stage_part_vq_spec(stage)
            logical=tuple(spec['logical_shape'])
            b.require(logical==(6160384,1,12) and tuple(storage.shape)==(6160384,1,2)
                and not (storage[...,1]&240).any(),'Actual K12 payload has a wrong layout or nonzero high4 padding.')
            unpacked=unpack_uint8_tensor_to_bool(storage,logical_shape=logical)
            repacked=pack_bool_tensor_to_uint8(unpacked,logical_shape=logical)
            b.require(b.torch.equal(storage,repacked),'Actual K12 packed payload failed exact all-bit readback/repack.')
            rows.append(dict(name=name,stage=stage,logical_shape=list(logical),packed_shape=list(storage.shape),
                high4_zero=True,all_bits_exact_roundtrip=True,packed_digest=b.c.digest({'code':storage})))
    return dict(actual_stages=72,all_packed_bits_roundtrip_exact=True,high4_padding_zero=True,rows=rows)


@b.torch.no_grad()
def decoder_numerics(model):
    effective=b.EFFECTIVE
    decoders,inputs,names=[],[],[]
    for layer in range(8):
        name=f'model.layers.{layer}.mlp.down_proj'
        module=model.get_submodule(name)
        if getattr(module,'_parallel_stage_decoder',None) is None:
            module.pack_parallel_stage_decoder_(trainable=False)
        decoder=module._parallel_stage_decoder.to('cuda')
        # Native grouped VQ is stage-major unpacked bool cast to BF16. Use the
        # same first257 real code rows without materializing the full dense VQ.
        packed=b.torch.cat([module.get_stage_part_vq_storage(stage)[:257].detach().cpu() for stage in range(2)],dim=1)
        value=unpack_uint8_tensor_to_bool(packed,logical_shape=(257,2,12)).to(device='cuda',dtype=b.torch.bfloat16)
        decoders.append(decoder);inputs.append(value);names.append(name)
    batch=effective.fused_decode_batched_same_shape_decoders(decoders,inputs)
    rows=[]
    def error(actual,reference):
        delta=actual.float()-reference.float()
        rms=reference.float().square().mean().sqrt().item()
        relative=delta.square().mean().sqrt().item()/max(rms,1e-12)
        absolute=delta.abs().max().item()
        b.require(tuple(actual.shape)==(257,8) and actual.dtype==b.torch.bfloat16
            and b.torch.isfinite(actual).all().item() and relative<=.005 and absolute<=.04*rms+.002,
            'D8 fused/reference difference exceeds the declared BF16 numerical bound.')
        return dict(relative_RMSE=relative,max_abs=absolute,reference_RMS=rms,
            bound_relative_RMSE=.005,bound_max_abs=.04*rms+.002)
    for name,decoder,value,grouped in zip(names,decoders,inputs,batch,strict=True):
        weights=tuple(w if index in (0,1) else w.to(value.dtype)
            for index,w in enumerate(effective.extract_packed_symmetric_stage_weights(decoder)))
        reference=effective.reference_multistage_symmetric_decode(value,*weights)
        single=effective.fused_decode_packed_symmetric_decoder(decoder,value)
        rows.append(dict(name=name,single_vs_native_reference=error(single,reference),
            group8_vs_native_reference=error(grouped,reference),single_vs_group8=error(single,grouped)))
    return dict(status='actual_D8_single_and_group8_masked_fused_vs_native_decomposed_verified',
        real_layers=8,real_code_rows_per_layer=257,shape=[257,2,12],logical_OUT=8,calculation_tile_OUT=16,
        reference='Unmodified original reference_multistage_symmetric_decode definition copied byte-for-byte with the source.',
        effective_consumer=b.consumer.contract(),rows=rows,
        boundary='BF16 kernel equivalence sample only. Full model group8 warm/fresh remains separately checked; no quality claim.')

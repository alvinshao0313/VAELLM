"""CPU source/geometry/padding and exact two-launcher-diff proof; no CUDA work."""
import ast
from copy import deepcopy
from dataclasses import asdict
import difflib
import json
import sys
import time

from . import bindings as b
from litebsq.bitpack import pack_bool_tensor_to_uint8,unpack_uint8_tensor_to_bool


def main():
    b.cpu_environment()
    b.torch.set_num_threads(4)
    b.require(not b.PREPARED.exists() and not b.OUTPUT.exists(),'Never overwrite this neighbor source proof or output.')
    started=time.monotonic()
    source,control=b.sources(),b.control()
    parsed,identity,baseline=b.recipe(b.OUTPUT/'formal')
    activation,_plan,restored=b.base.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
    b.require(restored==baseline and activation['cache'].nsamples==512 and activation['cache'].seqlen==256,
        'Preserve actual original512x256 legal train calibration and runtime.')
    state=b.torch.load(b.PARENT/b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    targets={name.removesuffix('.weight'):value for name,value in state.items() if '.mlp.down_proj.' in name}
    old=b.read(b.base.RESULT/'prepared.json')
    b.require(old['source_binding']==source,'Reuse frozen-prefix evidence only for this identical actual after-up source binding.')
    digest=b.c.digest(targets)
    b.require(len(targets)==36 and digest==old['live_down_source_digest']==source['original_down_fit']['source_target_digest'],
        'D8 must fit the same36 actual live dense down tensors, never D16/32 down state or recovery state.')
    layouts={name:dict(original_shape=list(value.shape),transposed_unprotected_shape=[value.shape[1]-256,value.shape[0]],
        elements=(value.shape[1]-256)*value.shape[0],vectors_D8=(value.shape[1]-256)*value.shape[0]//8)
        for name,value in targets.items()}
    b.require(all(row['original_shape']==[4096,12288] and row['transposed_unprotected_shape']==[12032,4096]
        and row['elements']%8==0 and row['vectors_D8']==6160384 for row in layouts.values()),
        'Every actual transposed unprotected down matrix must flatten exactly into D8 vectors.')
    # Exhaust all real K12 codewords through the unchanged native packer.
    numbers=b.torch.arange(4096,dtype=b.torch.int64).view(-1,1)
    bits=numbers.bitwise_and(1<<b.torch.arange(12)).ne(0).unsqueeze(1)
    packed=pack_bool_tensor_to_uint8(bits,logical_shape=bits.shape)
    b.require(tuple(packed.shape)==(4096,1,2) and not (packed[...,1]&240).any()
        and b.torch.equal(unpack_uint8_tensor_to_bool(packed,logical_shape=bits.shape),bits),
        'K12 must retain12 logical bits in2 bytes with exact readback and zero high4 padding bits.')
    before=b.consumer.ORIGINAL.read_text(encoding='utf-8')
    after=b.consumer.EFFECTIVE.read_text(encoding='utf-8')
    old_line='BLOCK_OUT = min(_next_pow2(OUT), 64)'
    new_line='BLOCK_OUT = max(16, min(_next_pow2(OUT), 64))'
    b.require(before.count(old_line)==2 and after==before.replace(old_line,new_line),
        'Only two launcher tile expressions may differ; retain logical OUT, masks, strides and exact kernel body.')
    b.require(all(max(16,min(b.EFFECTIVE._next_pow2(dim),64))==min(b.EFFECTIVE._next_pow2(dim),64)
        for dim in range(16,65)),'Every supported D>=16 launch tile must remain unchanged.')
    delta_decoder_bytes=36*2*2*((128*12+128+2*128+8*128+8)-(128*24+128+2*128+16*128+16))
    delta_code_bytes=36*2*(6160384*2-(6160384//2)*3)
    prior=control['budget']
    logical=prior['user_linear_bits']+8*delta_decoder_bytes
    physical=prior['packed_linear_bits']+8*(delta_code_bytes+delta_decoder_bytes)
    tensor=prior['full_model_tensor_bytes']+delta_code_bytes+delta_decoder_bytes
    predicted=dict(evidence='Analytic expectation from actual D16 geometry/ledger; not measured D8 storage.',
        user_linear_bits=logical,packed_linear_bits=physical,original_linear_weights=prior['original_linear_weights'],
        user_linear_bpw=logical/prior['original_linear_weights'],packed_linear_bpw=physical/prior['original_linear_weights'],
        full_model_tensor_bytes=tensor,delta_decoder_bytes=delta_decoder_bytes,delta_packed_code_bytes=delta_code_bytes,
        relative_tensor_change_vs_D16=tensor/prior['full_model_tensor_bytes']-1)
    b.require(abs(predicted['user_linear_bpw']-2.497463723887568)<1e-14
        and abs(predicted['packed_linear_bpw']-2.7662731668223506)<1e-14 and tensor==4891672128,
        'The explicit actual-geometry analytic budget disagrees with the approved expectation.')
    b.require(b.consumer.install() is b.EFFECTIVE and sys.modules['litebsq.vae_linear'].fused_decode_packed_symmetric_decoder
        is b.EFFECTIVE.fused_decode_packed_symmetric_decoder,'Actual single/group module bindings must resolve the effective file.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    record=dict(status='CPU_verified_D8_parent_geometry_and_two_tile_extension',schema='same_parent_D8_prepared_v1',
        command=[sys.executable,*sys.argv],created_unix=time.time(),source_binding=source,D16_control=control,contract=b.CONTRACT,
        live_down_source_digest=digest,frozen_prefix_digest=old['frozen_prefix_digest'],
        reused_frozen_prefix_proof=str(b.base.RESULT/'prepared.json'),reused_frozen_prefix_proof_sha256=b.sha(b.base.RESULT/'prepared.json'),
        actual_layouts=layouts,category_configs={name:asdict(cfg) for name,cfg in parsed[-2].items()},
        runtime_identity=identity,original_runtime_identity=baseline,predicted_budget=predicted,
        packed_K12_proof=dict(exhaustive_codewords=4096,logical_bits=12,physical_bytes_per_codeword=2,
            high4_padding_bits_zero=True,all_bits_exact_roundtrip=True),
        effective_consumer=b.consumer.contract(),kernel_unified_diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile=str(b.consumer.ORIGINAL),tofile=str(b.consumer.EFFECTIVE))),
        supported_D16_to64_tile_expression_unchanged=True,consumer_single_and_group_imports_bound=True,
        source_sha256=b.pins(),CUDA_initialized=b.torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
        boundary='CPU new-source/config/geometry/bit/diff proof only. Await explicit Astra core release, then actual new-shape short/peak/fresh before formal. No control weights imported.')
    b.require(not record['CUDA_initialized'],'CPU proof initialized CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.write(b.PREPARED,record)
    print(json.dumps(dict(status=record['status'],elapsed_seconds=record['elapsed_seconds'],predicted_budget=predicted,
        actual_downs=36,exhaustive_K12_codewords=4096,changed_kernel_lines=2,CUDA_initialized=False)),flush=True)


if __name__=='__main__':
    main()

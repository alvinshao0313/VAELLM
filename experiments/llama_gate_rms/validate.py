"""Exercise the actual trained RMS deployment path; no downstream quality claim."""
import argparse
import logging
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.llama_gate_rms.common import check, read, _require
from experiments.llama_gate_rms.audit import audit
from experiments.longbench_v1.common import sha256, write_json
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from train_utils.v6_model_loader import load_v6_model_checkpoint
from tools.cat_eval import _prepare_model_for_eval


def validate(path):
    record = read(path); output = Path(record['output_dir']); started = time.time()
    _require(not record['full_evaluation'] and record['steps_per_stage'] == 2, 'Use only the real RMS diagnostic.')
    proof = audit(record, Path(record['donor_checkpoint_dir'])); write_json(output / 'audit.json', proof)
    model, meta, loaded = load_v6_model_checkpoint(str(output / 'final_model'), map_location='cpu',
        strict=True, expected_kind='final_model')
    _require(meta['checkpoint_id'] == proof['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
             'Independent native RMS load failed.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    _require(warmup['total'] == warmup['warmed'] == record['architecture']['projection_count']
        and warmup['failed'] == warmup['skipped'] == 0 and warmup['prewarm_dtype'] == 'torch.bfloat16',
        'Native all-projection group8 BF16 prewarm failed.')
    names = sorted(n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj'))
    for name in names:
        decoder = model.get_submodule(name)._parallel_stage_decoder
        _require(decoder.norm_out.norm_type == 'rms' and not packed_symmetric_decoder_supports_fuse(decoder)
            and len(decoder.state_dict()) == record['native_cpu_construction']['decoder_leaves_per_projection'],
            'Actual RMS modules did not retain their native generic packed path.')
    # Compare stage outputs before summation, with identical trained BF16 weights,
    # RMS and activation. Only the packed first-linear operand differs.
    module = model.get_submodule('model.layers.0.mlp.gate_proj'); decoder = module._parallel_stage_decoder
    packed = module._get_parallel_stage_grouped_vq_packed(device=torch.device('cuda:0'))[:4096]
    logical = (len(packed), decoder.num_models, decoder.in_dim)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = module._decode_packed_u8_with_decoder(decoder, packed, logical_shape=logical,
            activation_dtype=torch.bfloat16, sparse_bit_grouped=True)
        expected = decoder(unpack_uint8_tensor_to_bool(packed, logical_shape=logical).to(torch.bfloat16))
    _require(actual is not None and actual.dtype == expected.dtype == torch.bfloat16
        and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'Native RMS CUDA packed decode failed.')
    epsilon = 2 * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, expected, rtol=epsilon, atol=epsilon)
    parity = {'passed': True, 'layer': 0, 'shape': list(actual.shape), 'dtype': str(actual.dtype),
        'stages_compared_separately': decoder.num_models, 'rtol': epsilon, 'atol': epsilon,
        'max_abs': float((actual.float() - expected.float()).abs().max()),
        'boundary': 'Two BF16 rounding units in combined relative/absolute tolerance; not CUDA bitwise equivalence.'}
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    ids = tokenizer('A short test sentence.', return_tensors='pt', add_special_tokens=False)['input_ids'][:, :4]
    _require(ids.shape == (1, 4), 'Require four actual tokenizer tokens.')
    with torch.inference_mode(): logits = model(input_ids=ids.to('cuda:0'), use_cache=False).logits
    rope = {key: str(value.dtype) for key, value in model.named_buffers() if 'inv_freq' in key}
    _require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()) and rope
        and set(rope.values()) == {'torch.float32'}, 'Native BF16 integration forward/FP32 RoPE failed.')
    proof.update(status='verified_native_rms_short', native_group8_prewarm=warmup, kernel_parity=parity,
        integration_forward={'input_shape': list(ids.shape), 'output_shape': list(logits.shape),
            'dtype': str(logits.dtype), 'finite': True, 'rope': rope},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        audit_sha256=sha256(output / 'audit.json'), elapsed_seconds=time.time() - started,
        scope='Real2 updates per stage, native source union/save/load/group8 and four-token integration only; no full8/LongBench quality or convergence proof.')
    check(record); write_json(output / 'verification.json', proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record', type=Path, required=True)
    validate(parser.parse_args().launch_record)

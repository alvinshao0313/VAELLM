"""Check VAELLM development on a new server without downloading model weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace


def environment(repo):
    if Path(os.environ.get('CONDA_PREFIX', '')).name != 'bitvae' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '')):
        raise RuntimeError('先在调用者终端激活 bitvae；当前解释器：' + sys.executable)
    for name in ('litebsq/autoencoder.py', 'train_utils/checkpoint_v6.py', 'tools/cat_train.py'):
        if not (repo / name).is_file():
            raise FileNotFoundError('缺少项目源码：' + str(repo / name))
    sys.path.insert(0, str(repo))
    modules = ('torch', 'transformers', 'datasets', 'accelerate', 'peft', 'deepspeed', 'lm_eval',
               'train_utils.cat_train_pipeline', 'compressed_e2e_fintuning.runtime_v6',
               'train_utils.cat_checkpoint_distill')
    for name in modules:
        importlib.import_module(name)
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'repo': str(repo), 'imports': list(modules),
            'versions': {name: version(name) for name in ('torch', 'transformers', 'datasets', 'accelerate',
                         'peft', 'deepspeed', 'lm-eval', 'flash-attn', 'fast-hadamard-transform', 'triton')}}


def vae_train(device):
    import torch
    from litebsq.autoencoder import MultiLayerVAE
    from litebsq.vae_args import apply_autoencoder_arch_defaults

    args = SimpleNamespace(codebook_bits=8, codebook_dim=16, base_ch=32, num_res_blocks=0,
                           decoder_type='symmetric', norm_type='layer', activation_type='swish',
                           quantizer_type='BSQ', recon_loss_type='mse', parallel_layers=2,
                           vae_weight_dtype='bf16' if device == 'cuda:0' else 'fp32',
                           vae_autocast_dtype='bf16' if device == 'cuda:0' else 'fp32',
                           use_checkpoint=False)
    apply_autoencoder_arch_defaults(args)
    torch.manual_seed(31)
    model = MultiLayerVAE(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(64, 2, 16, device=device,
                    dtype=torch.bfloat16 if device == 'cuda:0' else torch.float32)
    before = [p.detach().clone() for p in model.parameters()]
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, values = model(x, is_train=True)
        loss = values['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('VAE loss非有限值')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('VAE梯度缺失或非有限值')
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = sum(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    if not changed:
        raise RuntimeError('VAE参数没有更新')
    return {'device': device, 'steps': 3, 'losses': losses, 'changed_parameters': changed}


def checkpoint_roundtrip(work):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from litebsq.autoencoder import Decoder
    from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
    from litebsq.vae_linear import VAELinear
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint

    bits = torch.randint(0, 2, (256, 1, 9), dtype=torch.bool)
    packed = pack_bool_tensor_to_uint8(bits, logical_shape=bits.shape)
    if not torch.equal(bits, unpack_uint8_tensor_to_bool(packed, logical_shape=bits.shape)):
        raise RuntimeError('非8整数倍码位的打包往返不一致')
    results = []
    for name, config_type, model_type in [('qwen', Qwen2Config, Qwen2ForCausalLM), ('llama', LlamaConfig, LlamaForCausalLM)]:
        config = config_type(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                             attention_dropout=0.0, sliding_window=None)
        config._attn_implementation = 'eager'
        model = model_type(config).eval()
        decoders = [Decoder(in_dim=9, out_dim=16, hidden_dim=32, num_res_blocks=0,
                            norm_type='layer', decoder_type='symmetric') for _ in range(2)]
        model.model.layers[0].self_attn.q_proj = VAELinear(
            in_features=64, out_features=64, bias=None, original_weight=None,
            stage_vq_weights=[bits.clone(), ~bits], stage_decoders=decoders,
            codebook_dim=16, transpose=False)
        tokens = torch.arange(16).unsqueeze(0)
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits
        path = work / name
        saved = save_v6_full_checkpoint(model, str(path), checkpoint_kind='final_model',
                                       compressed_targets=['model.layers.0.self_attn.q_proj'])
        restored, meta, loaded = load_v6_full_checkpoint_into_model(
            model_type(config), str(path), expected_kind='final_model', strict=True)
        restored.eval()
        if loaded.missing_keys or loaded.unexpected_keys or meta['checkpoint_id'] != saved['checkpoint_id']:
            raise RuntimeError('v6身份或strict加载失败')
        actual_state, expected_state = restored.state_dict(), model.state_dict()
        if actual_state.keys() != expected_state.keys():
            raise RuntimeError('v6持久张量键不一致')
        for key, value in expected_state.items():
            if actual_state[key].dtype != value.dtype or not torch.equal(actual_state[key], value):
                raise RuntimeError('v6持久张量不一致：' + key)
        with torch.no_grad():
            actual = restored(tokens, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        results.append({'model': name, 'checkpoint_id': meta['checkpoint_id'], 'exact_state_tensors': len(actual_state),
                        'logits_max_abs': float((actual - expected).abs().max())})
    return {'models': results, 'packed_bits_exact': True, 'original_weights_downloaded': False}


def cuda_kernels():
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from rotation.hadamard_utils import get_hadK, matmul_hadU, matmul_hadU_cuda

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用：检查新机器nvidia-smi、驱动及当前PyTorch的CUDA构建')
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('当前可见GPU0不支持本项目所需BF16')
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    product = a @ a.T
    product.float().square().mean().backward()
    if not torch.isfinite(a.grad).all() or not torch.count_nonzero(a.grad):
        raise RuntimeError('CUDA BF16矩阵乘反向异常')
    x = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    had, factor = get_hadK(128)
    transformed = matmul_hadU_cuda(x, had, factor)
    reference = matmul_hadU(x.detach().cpu().float()).to('cuda')
    torch.testing.assert_close(transformed.float(), reference, rtol=0.02, atol=0.02)
    transformed.float().square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or not torch.count_nonzero(x.grad):
        raise RuntimeError('Hadamard CUDA扩展反向异常')
    q, k, v = [torch.randn(1, 128, 4, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    reference = F.scaled_dot_product_attention(*(t.detach().float().transpose(1, 2) for t in (q, k, v)),
                                               is_causal=True).transpose(1, 2)
    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.03)
    output.float().square().mean().backward()
    if not all(t.grad is not None and torch.isfinite(t.grad).all() and torch.count_nonzero(t.grad) for t in (q, k, v)):
        raise RuntimeError('FlashAttention反向异常')
    torch.cuda.synchronize()
    return {'device': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda,
            'visible_devices': torch.cuda.device_count(), 'tested_device': 0,
            'bf16_matmul_backward': True, 'hadamard_forward_backward': True,
            'flash_attention_forward_backward': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def packed_decoder():
    import torch
    from torch.nn import functional as F
    from litebsq.autoencoder import Decoder
    from litebsq.bitpack import pack_bool_tensor_to_uint8
    from litebsq.fused_multistage_decoder import fused_decode_packed_symmetric_decoder
    from litebsq.packed_bit_linear import packed_u8_parallel_linear

    decoder = Decoder(in_dim=32, out_dim=32, hidden_dim=128, num_res_blocks=0,
                      norm_type='layer', decoder_type='symmetric', num_models=2).cuda()
    bits = torch.randint(0, 2, (64, 2, 32), device='cuda', dtype=torch.bool)
    packed = pack_bool_tensor_to_uint8(bits, logical_shape=bits.shape)
    parameters = tuple(decoder.parameters())
    gradient = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = packed_u8_parallel_linear(packed, decoder.linear_in, logical_in_dim=32, activation_dtype=torch.bfloat16)
        reference = decoder.linear_out(F.silu(decoder.norm_out(hidden))).sum(dim=1).to(torch.bfloat16)
        expected_grads = torch.autograd.grad(reference, parameters, gradient)
        actual = fused_decode_packed_symmetric_decoder(decoder, bits.to(torch.bfloat16))
        actual_grads = torch.autograd.grad(actual, parameters, gradient)
    epsilon = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, reference, atol=2 * epsilon, rtol=2 * epsilon)
    errors = {}
    for (name, parameter), expected, observed in zip(decoder.named_parameters(), expected_grads, actual_grads):
        if observed.dtype != parameter.dtype or not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
            raise RuntimeError('Packed decoder梯度dtype或数值异常：' + name)
        relative = float((observed.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
        limit = 1e-5 if name == 'linear_in.conv.weight' else 2 * epsilon
        if relative > limit:
            raise RuntimeError(f'Packed decoder梯度误差过大：{name} {relative} > {limit}')
        errors[name] = relative
    torch.cuda.synchronize()
    return {'triton_packed_and_fused_forward_backward': True, 'gradient_relative_errors': errors}


def nccl_worker(rank, count, rendezvous):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=count, timeout=timedelta(seconds=60))
    try:
        for _ in range(3):
            value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
            dist.all_reduce(value)
            if value.item() != count * (count + 1) / 2:
                raise RuntimeError(f'NCCL rank {rank} all_reduce结果错误')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def nccl_check(work):
    import torch
    count = torch.cuda.device_count()
    if count < 2:
        raise RuntimeError('--nccl需要至少两张可见GPU；请用CUDA_VISIBLE_DEVICES显式选择空闲卡')
    torch.multiprocessing.spawn(nccl_worker, args=(count, (work / 'nccl_store').as_uri()), nprocs=count, join=True)
    return {'nccl_devices': count, 'all_reduce_iterations': 3}


def handoff_check(package, output):
    results = []
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    for name, command in [('verify', ['verify']), ('qwen', ['check-load', '--model', 'qwen']), ('llama', ['check-load', '--model', 'llama'])]:
        log = output / f'handoff_{name}.log'
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, str(package / 'handoff.py'), *command], env=env,
                           cwd=package, stdout=stream, stderr=subprocess.STDOUT, check=True)
        results.append({'check': name, 'log': str(log), 'status': 'passed'})
    return {'checks': results, 'gpu_model_forward_tested': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='必须是尚不存在的输出目录')
    parser.add_argument('--cpu-only', action='store_true', help='只检查CPU开发路径，不表示GPU验收通过')
    parser.add_argument('--nccl', action='store_true', help='对全部可见GPU做NCCL检查，调用前选择空闲GPU')
    parser.add_argument('--handoff', type=Path, help='可选：解压后的研究接续包，逐文件校验和两成品CPU重载')
    args = parser.parse_args()
    if args.cpu_only and args.nccl:
        parser.error('--cpu-only与--nccl互斥')
    output, repo = args.output.resolve(), args.repo.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    report = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'checks': [],
              'scope': '开发环境与小规模真实模块自检，不代表8B正式训练、完整评测或吞吐验收。'}
    current = 'environment'
    try:
        with tempfile.TemporaryDirectory(prefix='probe_', dir=output) as temporary:
            work = Path(temporary)
            checks = [('environment', lambda: environment(repo)), ('vae_cpu_train', lambda: vae_train('cpu')),
                      ('v6_qwen_llama_roundtrip', lambda: checkpoint_roundtrip(work))]
            if not args.cpu_only:
                checks += [('cuda_extensions', cuda_kernels), ('vae_cuda_train', lambda: vae_train('cuda:0')),
                           ('triton_packed_decoder', packed_decoder)]
            if args.nccl:
                checks.append(('nccl', lambda: nccl_check(work)))
            if args.handoff:
                checks.append(('handoff_models_cpu', lambda: handoff_check(args.handoff.resolve(), output)))
            for current, function in checks:
                print('CHECK ' + current, flush=True)
                started = time.perf_counter()
                details = function()
                report['checks'].append({'name': current, 'status': 'passed', 'seconds': time.perf_counter() - started, 'details': details})
                (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report['status'] = 'cpu_passed_gpu_not_tested' if args.cpu_only else 'passed'
    except Exception:
        report['status'] = 'failed'
        report['failed_check'] = current
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr)
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}, ensure_ascii=False), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())

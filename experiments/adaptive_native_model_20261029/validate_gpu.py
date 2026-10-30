"""Two original M32 updates on the real full compressed prototype and fresh deployment."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from experiments.adaptive_native_bridge_20261029.inputs import load_groups
from experiments.adaptive_native_bridge_20261029.consumer import CompactRVQLinear as BridgeLinear
from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_native_model_20261029.validate_cpu import PREFIX, INPUT, INPUT_SHA, equal_state, event
from experiments.adaptive_vector_rvq import common as c
from tools.cat_eval import _prepare_model_for_eval


def frozen_state(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith(PREFIX + '.decoders.')}


def prepare(model):
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(warm['total'] == 251 and warm['warmed'] == 251 and not warm['failed'], 'Warm all actual remaining native projections through group8.')
    c.require(model.config._attn_implementation == 'flash_attention_2'
              and all(value.dtype == torch.float32 for name, value in model.named_buffers() if 'inv_freq' in name),
              'Preserve native FA2 and FP32 RoPE runtime.')
    return warm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, cpu_dir = args.output.resolve(), args.cpu_result.resolve()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '7' and torch.cuda.device_count() == 1
              and not output.exists(), 'Use only coordinated GPU7 and a new independent validation output.')
    cpu = c.read(cpu_dir / 'report.json')
    c.require(cpu['status'] == 'CPU_verified_GPU_pending' and cpu['fresh']['no_external_parent_weight_read']
              and c.sha(INPUT) == INPUT_SHA, 'Complete real CPU validation is required before GPU use.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    report = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time(),
                  source_CPU_report_sha256=c.sha(cpu_dir / 'report.json'), source_checkpoint_id=cpu['saved_checkpoint_id'],
                  boundary='251 G09 native projections + historical compact L18; format/training validation only, no same-parent quality claim')
    c.write(output / 'report.json', report)
    try:
        backend, groups, data = load_groups()
        model, meta, loaded, budget = checkpoint.load(cpu_dir / 'model')
        c.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Use the strict complete saved model.')
        report.update(initial_bit_budget=budget, data=data, warmup=prepare(model))
        layer = model.get_submodule(PREFIX)
        bridge = BridgeLinear(INPUT, 'dynamic3', 'cuda:0')
        with torch.no_grad():
            initial_weight = layer.decode_weight().detach()
            c.require(torch.equal(initial_weight, bridge.decode_weight()), 'The generic actual-spec consumer changed the verified bridge BF16 decode.')
        del bridge
        original_frozen = c.digest(frozen_state(model))
        layer.enable_decoder_training()
        initial_parameters = {name: value.detach().cpu().clone() for name, value in layer.named_parameters()}
        c.require(len(initial_parameters) == 18 and all(value.dtype == torch.float32 for value in layer.parameters()),
                  'Only the actual 18 compact decoder FP32 masters may train.')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
        rows = []
        for step, (cases, target) in enumerate(groups, 1):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = 1e-6 * (1 - (step - 1) / 500)
            loss, document_loss = backend.mmlu.weighted_loss(backend.mmlu.log_distribution(model, cases), target)
            c.require(torch.isfinite(loss).item(), 'Nonfinite true full compressed M32 loss.')
            loss.backward()
            c.require(all(parameter.grad is not None and parameter.grad.dtype == torch.float32
                          and torch.isfinite(parameter.grad).all().item() for parameter in layer.parameters()),
                      'Every actual compact decoder leaf must receive a finite FP32 gradient.')
            c.require(all(parameter.grad is None for name, parameter in model.named_parameters()
                          if not name.startswith(PREFIX + '.decoders.')), 'A frozen native/nonprojection parameter received a gradient.')
            norm = float(torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.5, error_if_nonfinite=True))
            c.require(norm > 0, 'The true end-model objective did not reach the compact decoder.')
            optimizer.step()
            c.require(all(int(value['step']) == step for value in optimizer.state.values()), 'Adam state must advance on both consecutive actual updates.')
            builds = layer.cache_builds
            with torch.no_grad():
                observed = backend.mmlu.log_distribution(model, groups[0][0]).cpu()
            c.require(layer.cache_builds == builds + 1, 'Changed compact masters must invalidate the inference cache.')
            row = dict(step=step, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'],
                       gradient_norm_before_clip=norm, cache_builds=layer.cache_builds,
                       nonzero_gradient_elements=sum(int(torch.count_nonzero(p.grad)) for p in layer.parameters()),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            event('full_compressed_M32_update', **row)
            del loss, document_loss, observed
        c.require(c.digest(frozen_state(model)) == original_frozen, 'Original 251 native/all nonprojection or compact frozen fields changed.')
        changes = {name: int(torch.count_nonzero(value.detach().cpu() - initial_parameters[name])) for name, value in layer.named_parameters()}
        c.require(any(changes.values()), 'Two updates did not change the actual FP32 decoder masters.')
        model.eval()
        c.require(layer._decoded_cache is None, 'Train/eval transition must release the compact decoded cache.')
        with torch.no_grad():
            expected_logp = backend.mmlu.log_distribution(model, groups[0][0]).cpu()
            deployed_weight = layer.decode_weight().detach()
            weight_digest = c.digest({'weight': deployed_weight})
            weight_changes = int(torch.count_nonzero(deployed_weight != initial_weight))
        # The complete deploy payload must preserve all old precision and cast only the compact masters.
        saved, saved_budget, state = checkpoint.save(model, output / 'model', provenance=dict(
            parent_checkpoint_id=meta['checkpoint_id'], original_G09_checkpoint_id=cpu['source_checkpoint_id'],
            source_CPU_report_sha256=report['source_CPU_report_sha256'], historical_target_mismatch=True,
            updates=2, data=data, use='Isolated whole-compressed-model training/serializer validation'))
        for index, decoder in enumerate(layer.decoders):
            expected = {name: value.detach().cpu().bfloat16() for name, value in decoder.state_dict().items()}
            actual = {name: state[f'{PREFIX}.decoders.{index}.{name}'] for name in expected}
            equal_state(actual, expected, 'Actual compact master-to-BF16 deployment')
        c.require(saved_budget['user_linear_bits'] == budget['user_linear_bits']
                  and saved_budget['packed_linear_bits'] == budget['packed_linear_bits'], 'Decoder-only deployment changed the actual storage budget.')
        report.update(training=rows, changed_master_elements=changes, changed_BF16_weight_elements=weight_changes,
                      frozen_fields_exact=True, initial_weight_matches_verified_bridge_exact=True,
                      compact_BF16_deployment_matches_master_cast_exact=True, final_bit_budget=saved_budget,
                      serialized_deployment_digest=c.digest(state), saved_checkpoint_id=saved['checkpoint_id'],
                      expected_weight_digest=weight_digest, frozen_digest=original_frozen)
        del optimizer, model, layer, initial_weight, deployed_weight, state
        gc.collect()
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        fresh, fresh_meta, fresh_loaded, fresh_budget = checkpoint.load(output / 'model')
        c.require(c.digest(fresh.state_dict()) == report['serialized_deployment_digest']
                  and not fresh_loaded.missing_keys and not fresh_loaded.unexpected_keys,
                  'Fresh complete deployment tensor state differs from actual serialized payload.')
        report['fresh_warmup'] = prepare(fresh)
        fresh_layer = fresh.get_submodule(PREFIX)
        with torch.no_grad():
            fresh_logp = backend.mmlu.log_distribution(fresh, groups[0][0]).cpu()
            fresh_weight_digest = c.digest({'weight': fresh_layer.decode_weight()})
        c.require(torch.equal(fresh_logp, expected_logp) and fresh_weight_digest == weight_digest,
                  'Saved BF16 full compressed deployment changed its actual weight or full M32 distribution.')
        c.require(c.digest(frozen_state(fresh)) == original_frozen, 'Fresh complete deployment changed any frozen field.')
        report.update(status='full_compressed_M2_saved_fresh_verified', exit_code=0,
                      fresh_strict_deployment_tensors_exact=True, fresh_BF16_weight_exact=True,
                      fresh_full_M32_log_distribution_exact=True, fresh_compact_master_parameters_saved=False,
                      fresh_checkpoint_id=fresh_meta['checkpoint_id'], fresh_bit_budget=fresh_budget,
                      full_M32_max_abs_difference=float((fresh_logp - expected_logp).abs().max()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      source_hashes={str(path): c.sha(path) for path in Path(__file__).parent.glob('*.py')})
        del fresh, fresh_layer
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    event(report['status'], seconds=report['elapsed_seconds'], peak_allocated_bytes=report['peak_allocated_bytes'])


if __name__ == '__main__':
    main()

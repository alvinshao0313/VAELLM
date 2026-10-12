"""Measure existing four/sixteen-group decoder500 endpoints with the old scorer."""
import argparse
import gc
import logging
from pathlib import Path
import time

import torch

from experiments.gate_kl_coverage16_diagnosis.common import (
    ARM, POPULATIONS, SOURCE_ID, TARGET, read, check, bound,
)
from experiments.gate_kl_calibration_probe.run import teacher_chunks, teacher_populations
from experiments.gate_kl_calibration_probe.core import score_population
from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_tail_objective_probe.core import teacher_pass, native_policy, version_inventory, document_batch
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.utils import configure_deterministic_mode, set_seed
from train_utils.v6_model_loader import load_v6_model_checkpoint


def comparison(now, before):
    output = {}
    for name in POPULATIONS:
        a, b = now[name], before[name]
        require(a['causal_tokens'] == b['causal_tokens'], 'Compared population token counts differ.')
        output[name] = {key + '_delta': a[key] - b[key]
                        for key in ('kl_sum', 'kl_mean', 'nll_sum', 'nll_mean')}
    return output


@torch.no_grad()
def run(path):
    p = read(path); bound(p)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require the one assigned GPU.')
    output = Path(p['output_dir']) / 'kl'; require(not output.exists(), 'Never overwrite a KL measurement.')
    output.mkdir(); started = time.time(); deadline = started + p['kl_budget_seconds']
    torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
    batches = torch.load(p['batch_path'], map_location='cpu', weights_only=True)
    require({name: {key: tensor_record(value) for key, value in batch.items()} for name, batch in batches.items()}
            == p['token_batches'], 'Fixed native token batches changed.')
    archive = torch.load(p['archive_path'], map_location='cpu', weights_only=True, mmap=True)
    teacher_by_chunk, hf_mlp, teacher_report = teacher_pass(p['data'], teacher_chunks(batches), archive,
        device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
    del hf_mlp, archive
    old_teacher = torch.load(p['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
    validate_teacher_cache(old_teacher, batches, read(p['old_teacher_report']))
    teacher = teacher_populations(teacher_by_chunk, old_teacher, batches)
    gc.collect(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    require(torch.cuda.memory_allocated() == 0, 'Release teacher GPU state before loading the frozen student.')
    model, meta, loaded = load_v6_model_checkpoint(p['source_checkpoint'], map_location='cpu', strict=True,
                                                expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Use actual fresh f927 for both fixed endpoints.')
    model.eval().requires_grad_(False); model.config.use_cache = False; gate = model.get_submodule(TARGET)
    require(records(gate) == p['original_state'], 'The source gate differs from the original thirteen tensors.')
    torch.cuda.reset_peak_memory_stats(0)
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    require(warmup['total'] == warmup['warmed'] == 252 and warmup['failed'] == 0, 'Incomplete native source group8 prewarm.')
    policy = native_policy(model, p['data']['source_data']['model_policy'])
    frozen = version_inventory(model, exclude_gate=True)
    require(len(frozen) == 3410, 'Frozen source complement inventory differs.')
    original = {k: v.detach().clone() for k, v in gate.state_dict().items()}
    weight0 = gate._cached_weight.detach().clone()
    def first_logits():
        return model(**document_batch(batches[ARM], 0, torch.device('cuda:0')), use_cache=False).logits.detach().cpu()
    logits0 = first_logits()
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True,
                                              group_size=8, compute_device=torch.device('cuda:0'))
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native category group8 refresh failed.')
        return result
    results = {}
    try:
        for arm in ('four_decoder500', 'sixteen_decoder500'):
            info = p['kl_endpoints'][arm]
            endpoint = torch.load(info['path'], map_location='cpu', weights_only=True, mmap=True)
            require({k: tensor_record(v) for k, v in endpoint.items()} == info['state'], 'A reviewed decoder500 endpoint changed.')
            incompatible = gate.load_state_dict(endpoint, strict=True)
            require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'Endpoint must load strictly.')
            cache = prime(); require(records(gate) == info['state'], 'Actual patched endpoint differs.')
            populations = {}
            for name in POPULATIONS:
                populations[name] = score_population(model, batches[name], teacher[name], deadline=deadline)
                print({'arm': arm, 'population': name, 'kl_mean': populations[name]['kl_mean']}, flush=True)
            require(records(gate) == info['state'] and version_inventory(model, exclude_gate=True) == frozen
                    and native_policy(model, p['data']['source_data']['model_policy']) == policy,
                    'Scoring changed persistent endpoint/complement or native policy.')
            results[arm] = {'checkpoint_id': info['checkpoint_id'], 'state': info['state'],
                'populations': populations, 'native_group8_cache': cache,
                'versus_same_codes_before_decoder_fit': comparison(populations, info['before_fit_observations'])}
            del endpoint
    finally:
        gate.load_state_dict(original, strict=True); prime()
        require(records(gate) == p['original_state'] and torch.equal(gate._cached_weight, weight0)
                and torch.equal(first_logits(), logits0), 'Original thirteen tensors/cache/full logits did not restore.')
    require(all(not value.requires_grad and value.grad is None for value in model.parameters())
            and version_inventory(model, exclude_gate=True) == frozen, 'No parameter may train during diagnosis.')
    report = {'status': 'completed', 'source_checkpoint_id': SOURCE_ID,
        'preparation_sha256': p['preparation_sha256'], 'teacher': teacher_report, 'warmup': warmup,
        'arms': results, 'sixteen500_minus_four500': comparison(results['sixteen_decoder500']['populations'],
                                                              results['four_decoder500']['populations']),
        'original13_weight_full_logits_restored_exact': True, 'frozen_complement_count': len(frozen),
        'frozen_complement_versions_and_pointers_exact': True, 'native_policy': policy,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'elapsed_seconds': time.time() - started,
        'boundary': 'Previously fixed development populations and unchanged top100+tail-mass KL scorer. No optimizer, gradient, proposal selection or quality gate. Pre-fit comparisons reuse the measured identical-source/data observations; normal cross-run floating differences are not forced exact.'}
    check(p); write_json(output / 'report.json', report)
    print({'report_sha256': sha256(output / 'report.json'), 'status': 'completed'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--preparation', type=Path, required=True)
    run(parser.parse_args().preparation)

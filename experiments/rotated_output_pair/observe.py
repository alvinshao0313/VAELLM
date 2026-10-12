"""Native local v6 persistence and fixed development observations; no endpoint selection."""
import gc
import logging
import time

import torch
from torch import nn

from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_tail_objective_probe.core import document_batch, native_policy, version_inventory
from experiments.linear_output.artifacts import linear_host, load_linear
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.cat_train_pipeline import apply_group_vae_payload
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.utils import LinearRef
from train_utils.v6_model_loader import load_v6_model_checkpoint

from .data import ARMS, SOURCE, SOURCE_ID, TARGET


@torch.no_grad()
def persist(payload, data, destination):
    original = nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16)
    original.weight.copy_(data['weight'])
    host = linear_host(TARGET, original)
    apply_group_vae_payload(model=host, group_refs=[LinearRef(TARGET, host.get_submodule(TARGET), 'gate_proj', False)],
                           group_tag=TARGET, payload=payload, convert_device='cuda:0')
    gate = host.get_submodule(TARGET).to('cuda:0').eval().requires_grad_(False)
    ready(gate)
    before = gate._decode_weight(dtype=torch.bfloat16).cpu()
    require(bool(torch.isfinite(before).all()), 'Native endpoint decode is nonfinite.')
    state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    require(len(state) == 13, 'Keep native S2 gate13 state without encoder or LoRA.')
    save_v6_full_checkpoint(host.cpu(), str(destination), checkpoint_kind='final_model', compressed_targets=[TARGET],
        base_model_path=data['meta']['base_model_path'], save_config=False,
        extra_meta={'standalone_linear_only': True, 'scope': 'isolated rotated complete-output paired diagnostic'})
    restored = load_linear(TARGET, original, destination).to('cuda:0').eval().requires_grad_(False)
    require(all(torch.equal(v.cpu(), state[k]) for k, v in restored.state_dict().items()), 'Native13 save/reload differs.')
    ready(restored)
    after = restored._decode_weight(dtype=torch.bfloat16).cpu()
    require(torch.equal(before, after), 'Same native standalone kernel changed after strict reload.')
    result = {'state': {k: tensor_record(v) for k, v in state.items()}, 'native13_roundtrip_exact': True,
              'same_kernel_bf16_decode_roundtrip_exact': True, 'weight': tensor_record(before)}
    del host, gate, restored, before, after, original
    gc.collect(); torch.cuda.empty_cache()
    return state, result


@torch.no_grad()
def evaluate(data, endpoints, policy, *, deadline):
    started = time.time()
    model, meta, loaded = load_v6_model_checkpoint(str(SOURCE), map_location='cpu', strict=True, expected_kind='final_model')
    require(meta == data['meta'] and meta['checkpoint_id'] == SOURCE_ID
            and not loaded.missing_keys and not loaded.unexpected_keys, 'Strict f927 source load differs.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    original = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    require(len(model.state_dict()) == 3423 and len(original) == 13, 'Native source inventory differs.')
    for state in endpoints.values():
        require(set(state) == set(original) and all(v.shape == original[k].shape and v.dtype == original[k].dtype
                for k, v in state.items()), 'Endpoint geometry/dtypes must match source gate13.')
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    rope = native_policy(model, policy); frozen = version_inventory(model, exclude_gate=True)
    require(len(frozen) == 3410, 'Unexpected frozen complement.')
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]

    def install(state):
        gate.load_state_dict(state, strict=True); ready(gate)
        return prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device='cuda:0')

    first = model(**document_batch(data['batches']['train'], 0, 'cuda:0'), use_cache=False).logits.cpu()
    reports = {}
    hf = data['weight'].float().to('cuda:0')
    for arm in ('f927', *ARMS):
        require(time.time() < deadline, 'Deadline exceeded before fixed endpoint observations.')
        state = original if arm == 'f927' else endpoints[arm]
        cache = install(state)
        require(records(gate) == {k: tensor_record(v) for k, v in state.items()}, 'Installed state differs.')
        actual = gate._cached_weight.detach().clone()
        local = {}
        for domain, x in data['capture']['inputs'].items():
            # Same fixed student-prefix input, all input columns including protected error.
            error_sum, energy = 0., 0.
            for chunk in x.split(256):
                x32 = chunk.to('cuda:0').float()
                error_sum += float((x32 @ (actual.float() - hf).T).double().square().sum())
                energy += float((x32 @ hf.T).double().square().sum())
            local[domain] = {'valid_tokens': len(x), 'sse': error_sum, 'reference_energy': energy, 'nmse': error_sum / energy}
        full = {domain: score(model, batch, data['teachers'][domain], deadline=deadline)
                for domain, batch in data['batches'].items()}
        install(state)
        require(torch.equal(actual, gate._cached_weight), 'Repeated group8 decode changed same endpoint.')
        reports[arm] = {'local_fixed_prefix': local, 'full_model': full, 'cache': cache,
                        'group8_reload_exact': True, 'group8_weight': tensor_record(actual)}
        print(arm + ' fixed development observations complete', flush=True)
    install(original)
    last = model(**document_batch(data['batches']['train'], 0, 'cuda:0'), use_cache=False).logits.cpu()
    require(torch.equal(first, last) and records(gate) == {k: tensor_record(v) for k, v in original.items()},
            'Original13 state and full logits restore failed.')
    require(version_inventory(model, exclude_gate=True) == frozen and native_policy(model, policy) == rope,
            'Frozen complement or native RoPE changed.')
    # Only shape/dtype changes can affect this fixed native storage ledger; no full checkpoint copy is written.
    model.cpu()  # Native storage auditing requires CPU strided tensors.
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.5, max_lora_rank=0)
    return {'arms': reports, 'warmup': warmup, 'native_policy': rope, 'source_restored_exact': True,
            'frozen_complement_count': len(frozen), 'frozen_versions_and_pointers_exact': True,
            'budget': budget, 'elapsed_seconds': time.time() - started}

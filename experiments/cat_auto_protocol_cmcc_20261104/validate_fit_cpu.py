"""Real tiny HF/PUBLIC mechanism fixtures; never a formal model-shape pilot."""
import argparse
from copy import deepcopy
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from train_utils import checkpoint_v6 as v6
from train_utils.utils import set_seed
from litebsq.autoencoder import MultiLayerVAE

from .adapter import CATEGORIES, public_consumer_identity
from .adapter import file_sha
from .config import resolve_recipe
from .contracts import digest, inventory_id
from .fit_adapter import fit_category, fit_code_identity, install_category_bank, observe_bsq, _validate_actual_identity, _formal_evidence
from .geometry import architecture_args, component_plan
from .identity import semantic_seed
from .ledger import measure_native
from .vae_muon import VAEMuon, derive_layout, tensor_digest


def tiny_fixture(layers=2):
    set_seed(132)
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8, max_position_embeddings=64, attention_dropout=0.,
        tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    rows = []
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and name.rsplit('.', 1)[-1] in CATEGORIES:
            rows.append(dict(actual_name=name, category_id=name.rsplit('.', 1)[-1], layer_id=int(name.split('.')[2]),
                shape=list(module.weight.shape), dtype='bfloat16', dtype_bits=16, numel=module.weight.numel(), bias_numel=0))
    inventory = dict(model_identity=dict(schema='actual_tiny_Qwen3_CPU_mechanism_fixture', config=digest(cfg.to_json_string()),
        initial_weights=tensor_digest(model.state_dict())), adapter_id='qwen3_dense_CAT', adapter_version=1,
        category_ids=list(CATEGORIES), projections=rows, total_target_numel=sum(row['numel'] for row in rows))
    identity = dict(model=inventory['model_identity'], parent_uuid='CPU_fixture_parent', parent_weights_digest=tensor_digest(model.state_dict()),
        data_identity=dict(phase='cpu_fixture', record_ids=['synthetic_tokens_0', 'synthetic_tokens_1']),
        code_identity=fit_code_identity(), consumer_identity=public_consumer_identity(), search_seed=31)
    calibration = dict(input_ids=[torch.arange(16).reshape(1, 16), torch.arange(16, 32).reshape(1, 16)],
        manifest=dict(phase='cpu_fixture'), data_identity=identity['data_identity'], model_path='actual_tiny_fixture', dataset_spec='alpaca=1.0')
    return model, inventory, identity, calibration


def rng_checks(model_identity):
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    results = {}
    try:
        for order in (('up_proj', 'gate_proj'), ('gate_proj', 'up_proj')):
            for category in order:
                seed = semantic_seed(model_identity, category, 'residual_stage_1', 'vae_initialization', 31)
                set_seed(seed)
                value = (random.random(), float(np.random.rand()), torch.rand(5).tolist())
                if category in results:
                    assert value == results[category]
                results[category] = value
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
    assert torch.equal(saved[2], torch.get_rng_state())
    return dict(actual_native_set_seed_uint32=True, category_reorder_invariant=True, restored_host_rng=True)


def identity_rejection_checks(fixture):
    _, inventory, identity, calibration = fixture
    for field in ('consumer_identity', 'code_identity'):
        bad = deepcopy(identity); bad[field] = dict(source_sha256={'changed': 'not_actual'})
        try:
            _validate_actual_identity(bad)
        except ValueError:
            continue
        raise AssertionError('Mismatched actual source identity was accepted.')
    try:
        _formal_evidence(calibration, inventory, identity, resolve_recipe(inventory, 'up_proj', '2.5', dim=8, rotation_level='none'), None, 'cuda:0')
    except ValueError:
        return dict(actual_code_and_consumer_mismatch_rejected=True, CPU_inventory_data_cannot_upgrade_on_CUDA_label=True)
    raise AssertionError('CPU fixture inventory was upgraded by a CUDA label.')


def muon_checks():
    output = []
    for models in (2, 3):
        model, inventory, identity, _ = tiny_fixture(models)
        recipe = resolve_recipe(inventory, 'up_proj', '2.5', dim=8, rotation_level='none', protection_fraction='0',
            encoder_width=128, decoder_width=256, optimizer='vae_muon', multiplier=.5)
        args = architecture_args(recipe)
        args.parallel_layers, args.vae_weight_dtype = models, 'bf16'
        args.new_quant, args.quantizer_type = True, 'BSQ'
        for key, value in recipe['loss'].items():
            setattr(args, key, value)
        args.quantizer_type = 'BSQ'
        vae = MultiLayerVAE(args)
        named, layout, affine = derive_layout(vae)
        assert len(layout) == 4 and len(affine) == 8 and all(row['num_models'] == models for row in layout.values())
        evidence = {}
        optimizer = VAEMuon(vae, multiplier=.5, evidence=evidence)
        x = torch.randn(32, models, 8).to(torch.bfloat16)
        # Observer uses the real graph; on/off must leave the eventual update exact.
        pristine = deepcopy(vae)
        pristine_optimizer = VAEMuon(pristine, multiplier=.5, evidence={})
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            held = {}
            handle = vae.model.encoder.register_forward_hook(lambda _module, _inputs, value: held.update(h=value))
            _, loss = vae(x, is_train=True)
            handle.remove()
            prior_rng = torch.get_rng_state()
            diagnostic = observe_bsq(vae, loss, held['h'])
            assert torch.equal(prior_rng, torch.get_rng_state()) and all(parameter.grad is None for parameter in vae.parameters())
            loss['loss'].backward(); optimizer.step()
            pristine_optimizer.zero_grad(set_to_none=True)
            _, direct = pristine(x, is_train=True)
            direct['loss'].backward(); pristine_optimizer.step()
            assert tensor_digest(dict(vae.named_parameters())) == tensor_digest(dict(pristine.named_parameters()))
            assert all(torch.equal(a.grad, b.grad) for a, b in zip(vae.parameters(), pristine.parameters(), strict=True))
        saved = deepcopy(optimizer.state_dict())
        restored = VAEMuon(vae, multiplier=.5, evidence={})
        restored.load_state_dict(saved)
        assert restored.actual_step == 2 and all(restored.state[named[name]]['momentum_buffer'].dtype == torch.float32 for name in layout)
        assert all(torch.equal(restored.state[named[name]]['momentum_buffer'], optimizer.state[named[name]]['momentum_buffer']) for name in layout)
        output.append(dict(M=models, encoder_width=128, decoder_width=256, kappa=evidence['calibration'],
            actual_updates=2, optimizer_state_roundtrip=True, observer_on_off_grad_update_RNG_exact=True, diagnostic=diagnostic))
    return output


def fit_checks(root):
    model, inventory, identity, calibration = tiny_fixture()
    original = deepcopy(model)
    banks, reports = [], []
    for category in ('gate_proj', 'up_proj'):
        recipe = resolve_recipe(inventory, category, '2.5', dim=8, rotation_level='none', protection_fraction='0')
        recipe['transpose'] = {name: index == 1 for index, name in enumerate(recipe['group_signature'])}
        recipe['rotation']['per_layer'] = {name: 'none' if index == 0 else 'full' for index, name in enumerate(recipe['group_signature'])}
        own = deepcopy(original)
        result = fit_category(own, inventory=inventory, recipe=recipe, run_identity=identity, calibration_bundle=calibration,
            device='cpu', output_dir=root / category, technical_steps=2)
        assert not result['fit_report']['formal_eligible'] and result['fit_report']['payload_stages'] == 2
        assert len(result['fit_report']['stages']) == 2 and all(row['actual_steps'] == 2 for row in result['fit_report']['stages'])
        assert len({row['init_seed'] for row in result['fit_report']['stages']}) == 2
        banks.append(result['category_bank']); reports.append(result['fit_report'])
    joint = deepcopy(original)
    for bank in banks:
        install_category_bank(joint, bank, inventory=inventory, expected_parent_identity=identity)
    # Fresh actual native module rebuild consumes the same persisted packed banks.
    fresh = deepcopy(original)
    for bank in reversed(banks):
        install_category_bank(fresh, bank, inventory=inventory, expected_parent_identity=identity)
    assert tensor_digest(joint.state_dict()) == tensor_digest(fresh.state_dict())
    with torch.no_grad():
        a, b = joint(input_ids=calibration['input_ids'][0]).logits, fresh(input_ids=calibration['input_ids'][0]).logits
    assert torch.equal(a, b)
    names = [name for bank in banks for name in bank['recipe']['group_signature']]
    all_names = [row['actual_name'] for row in inventory['projections']]
    runtime = __import__('train_utils.cat_runtime_state_v6', fromlist=['build_cat_runtime_state']).build_cat_runtime_state(
        activation_runtime=None, global_adaptive_plan=None, runtime_identity=dict(phase='cpu_fixture'))
    saved = v6.save_v6_full_checkpoint(joint, str(root / 'joint_round_base'), checkpoint_kind='round_base',
        compressed_targets=names, pending_dense_targets=[name for name in all_names if name not in names],
        completed_categories=[], compression_categories=list(CATEGORIES), target_layers=[0, 1], target_modules=list(CATEGORIES),
        cat_runtime_state=runtime, extra_meta=dict(phase='cpu_fixture'))
    metadata = v6.load_v6_meta(str(root / 'joint_round_base'))
    ledger = measure_native(metadata, joint.state_dict(), inventory, directory=root / 'joint_round_base')
    assert ledger['N'] == inventory['total_target_numel'] and ledger['future_dense_count'] == 10 and ledger['future_dense_bits'] > 0
    assert ledger['source']['source_checkpoint_kind'] == 'round_base' and metadata['checkpoint_kind'] == 'round_base'
    assert not ledger['final_budget_applicable']
    for bank in banks:
        predicted = component_plan(inventory, bank['recipe'])
        actual = ledger['per_category'][bank['category_id']]
        assert predicted['logical_bits'] == actual['logical_bits'] and predicted['packed_bits'] == actual['packed_bits']
    return dict(real_tiny_Qwen3=True, formal_eligible=False, actual_M=2, actual_stages=2, actual_updates_per_stage=2,
        native_pooled_BSQ=True, layer_T_R_map_in_one_M=True, independent_persisted_two_bank_installation=True,
        installation_order_fresh_logits_state_exact=True, full_partial_ledger=ledger, reports=reports,
        inventory_identity=inventory_id(inventory), temporary_weights_removed_on_exit=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--lr-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(2)
    began = time.monotonic()
    report = dict(phase='cpu_fixture', formal_eligible=False, scope='Actual tiny HF/PUBLIC mechanisms only; not Qwen8B GPU resource or quality evidence')
    with tempfile.TemporaryDirectory(prefix='cat_auto_fit_cpu_') as temporary:
        fixture = tiny_fixture()
        if args.lr_only:
            model, inventory, identity, calibration = fixture
            recipe = resolve_recipe(inventory, 'up_proj', '2', dim=32, rotation_level='32', optimizer='adamw')
            result = fit_category(model, inventory=inventory, recipe=recipe, run_identity=identity, calibration_bundle=calibration,
                device='cpu', output_dir=Path(temporary) / 'lr_only', technical_steps=2)
            updates = [stage['updates'] for stage in result['fit_report']['stages']]
            assert all([row['actual_learning_rates'] for row in stage] == [[.003], [.0029997]] for stage in updates)
            report.update(scope='Affected actual tiny PUBLIC S2 two-step LR recording only; prior fit/consumer evidence reused',
                two_stage_actual_LRs=[[row['actual_learning_rates'] for row in stage] for stage in updates],
                fit_report=result['fit_report'])
            import ast
            for name in ('fit_adapter.py', 'search.py'):
                ast.parse(Path(__file__).with_name(name).read_text())
        else:
            report['semantic_rng'] = rng_checks(fixture[1]['model_identity'])
            report['identity_rejection'] = identity_rejection_checks(fixture)
            report['fit'] = fit_checks(Path(temporary))
            report['Muon'] = muon_checks()
    report.update(exit_code=0, elapsed_seconds=time.monotonic() - began, cuda_initialized=torch.cuda.is_initialized())
    report['source_sha256'] = {str(Path(__file__).with_name(name)): file_sha(Path(__file__).with_name(name)) for name in
        ('validate_fit_cpu.py', 'fit_adapter.py', 'vae_muon.py', 'contracts.py', 'identity.py', 'config.py', 'geometry.py', 'ledger.py', 'search.py')}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(dict(exit_code=0, elapsed_seconds=report['elapsed_seconds'], cuda_initialized=report['cuda_initialized']), flush=True)


if __name__ == '__main__':
    main()

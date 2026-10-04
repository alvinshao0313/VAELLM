"""Actual native head selection/update/save/restore; no 8B/GPU resource claim."""
from argparse import Namespace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, TrainerControl

from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, collect_e2e_mutable_state
from experiments.qwen3_w25.assemble_categories import CATEGORIES, _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _validate_export_storage
from experiments.qwen3_w25.local_bits import _code_banks
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.head_state import HEAD_KEY, audit_head_state, restore_head_state, validate_configuration
from experiments.w25_eval.model_inventory import model_inventory
from train_utils import checkpoint_v6 as v6
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.config.configs import AuxTrainableConfig, DistillDataConfig, DistillLossConfig, DistillOptimizationConfig, DistillRuntimeConfig, LoRAConfig
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.train_args import HFArguments, TrainingArguments
from test_qwen3_w25_e2e_aux_export import _model, TARGETS


def configuration(base='/home/star/models/Qwen/Qwen3-8B'):
    architecture = model_inventory(base)
    layers = list(range(architecture['num_hidden_layers']))
    cfg = Namespace(train_mode='none', data=DistillDataConfig(seed=0), loss=DistillLossConfig(),
        opt=DistillOptimizationConfig(steps=2000, distill_fp32_components=('lm_head',)),
        aux=AuxTrainableConfig(lm_head_train_mode='full', lm_head_lr=3e-5),
        runtime=DistillRuntimeConfig(parallel_mode='dp'), lora=LoRAConfig(),
        target_layers=layers, target_modules='all', bit_active_ratio=.05,
        bit_optimizer='adam', bit_lr='4e-5', bit_round_steps='128', bit_weight_decay=0.,
        bit_proxy_coordinates='decoder_sensitivity')
    args = TrainingArguments(output_dir='unused-cpu-head-fixture', use_cpu=True, report_to=[],
        max_steps=2000, seed=0, bf16=True, fp16=False, tf32=False, eval_strategy='no',
        warmup_ratio=.01, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps', save_steps=400, save_total_limit=1)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=args, tokenizer=tokenizer,
        input_checkpoint_id='cpu-head-base', resolved_target_layers=tuple(layers),
        resolved_target_modules=CATEGORIES, teacher_identity=None)
    contract['lora'] = None
    snapshot = json.loads(json.dumps(_build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=HFArguments(), training_args=args, run_output_dir=args.output_dir)))
    meta = {'train_mode': 'none', 'lora_config': None, 'norm_train_mode': 'none', 'lm_head_train_mode': 'full',
        'round_base_checkpoint_id': 'cpu-head-base', 'target_layers': layers,
        'target_modules': list(CATEGORIES), 'immutable_resume_contract': contract}
    return meta, {'global_step': 400, 'max_steps': 2000}, snapshot, architecture, args


@pytest.mark.parametrize('base,count', [('/home/star/models/Qwen/Qwen3-8B', 252),
    ('/home/star/models/meta-llama/Llama-3.1-8B-Instruct', 224)])
def test_real_architecture_and_contract_reject_decoder_or_schedule_changes(base, count):
    meta, state, snapshot, architecture, args = configuration(base)
    assert architecture['projection_count'] == count and not architecture['weights_loaded_for_inventory']
    assert validate_configuration(meta, state, snapshot, architecture) == meta['immutable_resume_contract']
    for section, key, value in [('aux', 'norm_train_mode', 'final'), ('opt', 'distill_fp32_components', ['decoder', 'lm_head'])]:
        changed = deepcopy(snapshot)
        changed['canonical_config'][section][key] = value
        with pytest.raises(ValueError):
            validate_configuration(meta, state, changed, architecture)
    with pytest.raises(ValueError):
        validate_configuration({**meta, 'train_mode': 'decoder'}, state, snapshot, architecture)
    with pytest.raises(ValueError):
        validate_configuration(meta, {**state, 'max_steps': 400}, snapshot, architecture)
    callback = Budget400Callback()
    callback.on_train_begin(args, SimpleNamespace(global_step=0), TrainerControl())
    assert not callback.on_step_end(args, SimpleNamespace(global_step=399), TrainerControl()).should_training_stop
    control = callback.on_step_end(args, SimpleNamespace(global_step=400), TrainerControl())
    assert control.should_save and control.should_training_stop


@pytest.fixture(scope='module')
def native_export(tmp_path_factory):
    root = tmp_path_factory.mktemp('head_only_native')
    model = _model()
    original = deepcopy(model.state_dict())
    base = root / 'base'
    saved = v6.save_v6_full_checkpoint(model, str(base), checkpoint_kind='final_model',
        compressed_targets=TARGETS, base_model_path='cpu-head-only-fixture')
    selected = tuple((name, model.get_submodule(name)) for name in TARGETS)
    inventory = {name: sorted(name + '._parallel_stage_decoder.' + key for key in module._parallel_stage_decoder.state_dict())
                 for name, module in selected}
    banks = {name: [(name + '.' + key, shape) for key, shape in _code_banks(module)] for name, module in selected}
    selection = build_model_level_trainable_selection(model,
        aux=AuxTrainableConfig(lm_head_train_mode='full', lm_head_lr=3e-5), compressed_modules=selected,
        rank=8, alpha=16, dropout=0, train_decoder=False, train_lora=False, fp32_components=('lm_head',), freeze=True)
    args = SimpleNamespace(bf16=True, fp16=False)
    configure_distill_precision(selection, components=('lm_head',), training_args=args)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {HEAD_KEY}
    # Use an actual tiny Qwen forward; no mocked decoder or alternate head graph.
    optimizer = torch.optim.AdamW(selection.lm_head_parameters.values(), lr=3e-5, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float().square().mean()
        loss.backward()
        assert model.lm_head.weight.grad.dtype == torch.float32
        assert all(p.grad is None for name, p in model.named_parameters() if name != HEAD_KEY)
        optimizer.step()
    mutable, classes, manifest = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    assert classes == {HEAD_KEY: 'lm_head'} and len(mutable) == 1
    mutable = {key: value.detach().clone() for key, value in mutable.items()}
    step = root / 'step'
    v6.save_v6_training_step_payload(str(step), round_base_ref=str(base),
        round_base_checkpoint_id=saved['checkpoint_id'], mutable_state=mutable, mutable_state_manifest=manifest,
        compressed_targets=TARGETS, train_mode='none', lm_head_train_mode='full',
        base_model_path='cpu-head-only-fixture', immutable_resume_contract={'cpu_serialization_test': True})
    restored, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(base), strict=True)
    selected = tuple((name, restored.get_submodule(name)) for name in TARGETS)
    _, _, snapshot, _, _ = configuration()
    report = restore_head_state(restored, selected, checkpoint=step, snapshot=snapshot)
    assert report == {'tensors_compared': 1, 'component_counts': {'lm_head': 1}, 'optimizer_restored': False}
    _assert_state_equal({HEAD_KEY: restored.lm_head.weight}, mutable, 'Actual native FP32 head restore')
    prepare_model_export(restored, args)
    final = root / 'final'
    restored.eval().requires_grad_(False)
    v6.save_v6_full_checkpoint(restored, str(final), checkpoint_kind='final_model', compressed_targets=TARGETS,
        base_model_path='cpu-head-only-fixture', train_mode='none', lm_head_train_mode='full')
    fresh, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(final), strict=True)
    actual = torch.load(final / 'pytorch_model.bin', weights_only=True, map_location='cpu')
    _assert_state_equal(actual, fresh.state_dict(), 'Actual fresh native strict reload')
    report = audit_head_state(original, actual, inventory, banks, mutable[HEAD_KEY])
    before, after = (audit_checkpoint(path, max_bpw=2.5, max_lora_rank=0) for path in (base, final))
    _validate_export_storage(before, after, report)
    assert before['user_linear_bpw'] == after['user_linear_bpw'] and before['components'] == after['components']
    return original, actual, inventory, banks, mutable[HEAD_KEY], report


def test_actual_head_update_frozen_storage_and_cast(native_export):
    original, actual, inventory, banks, head, report = native_export
    assert report['changed_head_fp32_elements'] > 0
    assert report['frozen_tensors_compared'] == len(original) - 1
    assert report['decoder_tensor_count'] == 42 and report['code_bank_count'] == 14
    assert report['changed_decoder_elements'] == report['changed_code_bits'] == 0
    assert report['head_matches_selected_fp32_cast_exactly'] and not report['head_fusion_performed']
    rounded = original[HEAD_KEY].float()
    rounded.view(-1)[0] = torch.nextafter(rounded.view(-1)[0], torch.tensor(float('inf')))
    assert torch.equal(rounded.bfloat16(), original[HEAD_KEY])
    assert audit_head_state(original, original, inventory, banks, rounded)['changed_head_bf16_elements'] == 0


def test_reject_frozen_change_wrong_head_cast_or_no_update(native_export):
    original, actual, inventory, banks, head, _ = native_export
    for key in [next(iter(inventory.values()))[0], next(iter(banks.values()))[0][0], 'model.norm.weight']:
        changed = dict(actual)
        changed[key] = actual[key].clone()
        changed[key].view(-1)[0] += 1
        with pytest.raises(ValueError):
            audit_head_state(original, changed, inventory, banks, head)
    with pytest.raises(ValueError):
        audit_head_state(original, {**actual, HEAD_KEY: actual[HEAD_KEY] + 1}, inventory, banks, head)
    with pytest.raises(ValueError):
        audit_head_state(original, original, inventory, banks, original[HEAD_KEY].float())

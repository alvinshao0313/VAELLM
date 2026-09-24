"""Training inventories and v6 persistence for real small Qwen3 residual adapters."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from e2e_common.full_lora import finalize_model_level_lora
from train_utils.checkpoint_v6 import save_v6_full_checkpoint, load_v6_full_checkpoint_into_model
from train_utils.config.cli import parse_e2e_cli, parse_cat_cli
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_optimizer import build_model_level_param_groups, ModelLevelOptimizerLRConfig
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state, restore_model_level_mutable_state
from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state


def model():
    torch.manual_seed(710)
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=8, tie_word_embeddings=False)
    cfg.use_cache = False
    return Qwen3ForCausalLM(cfg)


def selection(root, *, dense=False, mode="additive"):
    return build_model_level_trainable_selection(
        root, aux=AuxTrainableConfig(residual_lora_rank=2,
                                    residual_lora_alpha=4, residual_lora_lr=3e-3, residual_lora_mode=mode),
        dense_target_modules=("model.layers.0.self_attn.q_proj",) if dense else (),
        rank=2, alpha=4, dropout=0, train_lora=dense,
    )


def optimizer(sel):
    groups = build_model_level_param_groups(sel, model=sel.peft_model,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0,
                                             residual_lora_lr=3e-3))
    assert next(g for g in groups if g['group_name'] == 'residual_lora')['lr'] == 3e-3
    return torch.optim.AdamW(groups)


def step(sel, opt):
    sel.peft_model.train()
    opt.zero_grad(set_to_none=True)
    loss = sel.peft_model(input_ids=torch.tensor([[1, 2, 3, 4]]),
                          labels=torch.tensor([[1, 2, 3, 4]])).loss
    loss.backward()
    opt.step()
    return loss.detach()


@pytest.mark.parametrize('parser,base', [
    (parse_e2e_cli, ['--student_checkpoint_dir', 'unused', '--dataset_mix', 'alpaca=1']),
    (parse_cat_cli, ['--model_path', 'unused', '--compression_categories', 'q_proj']),
])
def test_cli_residual_settings_and_rank_limit(parser, base, capsys):
    assert parser(base).aux.residual_lora_mode == 'none'
    assert not hasattr(parser(base).aux, 'residual_lora_enabled')
    for mode in ['none', 'additive', 'replace']:
        assert parser(base + ['--residual_lora_mode', mode]).aux.residual_lora_mode == mode
    for old_flag in [['--residual_lora_enabled', 'true'], ['--residual_lora_enabled=false']]:
        with pytest.raises(SystemExit) as exc:
            parser(base + old_flag)
        assert exc.value.code == 2
        assert 'unrecognized arguments' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser(base + ['--residual_lora_mode', 'invalid'])
    cfg = parser(base + ['--residual_lora_mode', 'additive', '--residual_lora_rank', '8',
                         '--residual_lora_alpha', '12', '--residual_lora_lr', '0.0003',
                         '--distill_fp32_components', 'lora,residual_lora'])
    assert cfg.aux.residual_lora_mode == 'additive'
    assert cfg.aux.residual_lora_alpha == 12
    assert cfg.aux.residual_lora_lr == 3e-4
    for rank in ['0', '9']:
        with pytest.raises(SystemExit) as exc:
            parser(base + ['--residual_lora_rank', rank])
        assert exc.value.code == 2
        assert 'residual_lora_rank' in capsys.readouterr().err


def test_residual_only_and_disabled_default():
    root = model()
    before = set(root.state_dict())
    sel = selection(root, mode="none")
    assert not sel.residual_lora_parameters
    assert set(root.state_dict()) == before
    sel = selection(root)
    assert sel.residual_lora_parameters
    assert not sel.lora_parameters
    assert {id(p) for p in sel.residual_lora_parameters.values()} == {
        id(p) for p in root.parameters() if p.requires_grad}
    old = {k: p.clone() for k, p in sel.residual_lora_parameters.items()}
    assert torch.isfinite(step(sel, optimizer(sel)))
    assert any(not torch.equal(old[k], p) for k, p in sel.residual_lora_parameters.items())


@pytest.mark.parametrize('collect,restore', [
    (collect_model_level_mutable_state, restore_model_level_mutable_state),
    (collect_e2e_mutable_state, restore_e2e_mutable_state),
])
@pytest.mark.parametrize("mode", ["additive", "replace"])
def test_step_resume_next_update_is_identical(collect, restore, mode):
    import copy
    first = selection(model(), mode=mode)
    opt = optimizer(first)
    step(first, opt)
    state, classes, manifest = collect(first.peft_model, selection=first, selected_vae_modules=[])
    assert set(classes.values()) == {'residual_lora'}
    saved_state = {k: v.detach().clone() for k, v in state.items()}
    saved_opt = copy.deepcopy(opt.state_dict())
    step(first, opt)
    second = selection(model(), mode=mode)
    restore(second.peft_model, selection=second, selected_vae_modules=[],
            checkpoint_state=saved_state, checkpoint_manifest=manifest)
    second_opt = optimizer(second)
    second_opt.load_state_dict(saved_opt)
    step(second, second_opt)
    for key, tensor in first.peft_model.state_dict().items():
        torch.testing.assert_close(tensor, second.peft_model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', ['additive', 'replace'])
def test_peft_merge_export_reload_and_next_category(tmp_path, dtype, mode):
    sel = selection(model().to(dtype=dtype), dense=True, mode=mode)
    args = SimpleNamespace(bf16=dtype == torch.bfloat16, fp16=False)
    configure_distill_precision(sel, components=('lora', 'residual_lora'), training_args=args)
    assert all(p.dtype == torch.float32 for p in sel.residual_lora_parameters.values())
    step(sel, optimizer(sel))
    root = finalize_model_level_lora(sel.peft_model, compressed_proxy_names=[])
    prepare_model_export(root, args)
    root.eval()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = root(input_ids=inputs).logits
    saved = save_v6_full_checkpoint(root, str(tmp_path / 'full'), checkpoint_kind='final_model',
                                   compressed_targets=[])
    loaded, meta, _ = load_v6_full_checkpoint_into_model(model(), saved['output_dir'])
    assert meta['extra_meta']['residual_lora']['mode'] == mode
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=inputs).logits, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='conflicts'):
        incompatible = selection(model(), mode='replace' if mode == 'additive' else 'additive')
        load_v6_full_checkpoint_into_model(incompatible.peft_model, saved['output_dir'])
    # Disabled training retains the learned inference path; a later category re-enables in place.
    frozen = selection(loaded, mode="none")
    assert not frozen.residual_lora_parameters
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=inputs).logits, expected, rtol=0, atol=0)
    again = selection(loaded, mode=mode)
    before = {k: p.detach().clone() for k, p in again.residual_lora_parameters.items()}
    ids = {id(p) for p in again.residual_lora_parameters.values()}
    repeat = selection(again.peft_model, dense=True, mode=mode)
    assert ids == {id(p) for p in repeat.residual_lora_parameters.values()}
    for key, tensor in repeat.residual_lora_parameters.items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)


def test_e2e_trainer_one_cpu_step_with_compressed_decoder_lora(tmp_path):
    from datasets import Dataset
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from litebsq.autoencoder import Decoder
    from litebsq.vae_linear import VAELinear
    from train_utils.model_level_optimizer import attach_model_level_optimizer_contract
    from train_utils.train_args import TrainingArguments

    root = model()
    decoder = Decoder(in_dim=8, out_dim=8, hidden_dim=16, num_res_blocks=0,
                      norm_type='layer', decoder_type='symmetric')
    compressed = VAELinear(in_features=16, out_features=16, bias=None, original_weight=None,
        vq_weight=torch.randint(0, 2, (32, 1, 8), dtype=torch.bool),
        decoder=decoder, codebook_dim=8, transpose=False)
    root.model.layers[0].self_attn.q_proj = compressed
    sel = build_model_level_trainable_selection(root,
        aux=AuxTrainableConfig(residual_lora_mode="additive", residual_lora_rank=2),
        compressed_modules=[('model.layers.0.self_attn.q_proj', compressed)],
        rank=2, alpha=4, dropout=0, train_decoder=True, train_lora=True)
    snapshots = {group: {k: p.detach().clone() for k, p in getattr(sel, group).items()}
                 for group in ['lora_parameters', 'decoder_parameters', 'residual_lora_parameters']}
    frozen = root.model.embed_tokens.weight.detach().clone()
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=1,
        learning_rate=1e-3, save_strategy='no', eval_strategy='no', logging_strategy='no',
        report_to=[], disable_tqdm=True, remove_unused_columns=False, use_cpu=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    ds = Dataset.from_dict({'input_ids': [[1,2,3,4]], 'labels': [[1,2,3,4]],
                           'attention_mask': [[1,1,1,1]]})
    trainer = VAEDecoderE2ETrainer(model=sel.peft_model, args=args, train_dataset=ds, loss_type='sft')
    attach_model_level_optimizer_contract(trainer, selection=sel,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0))
    trainer.train()
    assert trainer.state.global_step == 1
    for group, before in snapshots.items():
        assert before
        assert any(not torch.equal(tensor, getattr(sel, group)[name]) for name, tensor in before.items()), group
    torch.testing.assert_close(root.model.embed_tokens.weight, frozen, rtol=0, atol=0)


def test_e2e_residual_mode_alone_controls_aux_only_training():
    base = ['--student_checkpoint_dir', 'unused', '--dataset_mix', 'alpaca=1', '--train_mode', 'none']
    for mode in ['additive', 'replace']:
        cfg = parse_e2e_cli(base + ['--residual_lora_mode', mode])
        assert cfg.aux.residual_lora_mode == mode
    with pytest.raises(SystemExit) as exc:
        parse_e2e_cli(base + ['--residual_lora_mode', 'none'])
    assert exc.value.code == 2

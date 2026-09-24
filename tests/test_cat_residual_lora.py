"""CPU integration coverage for residual-only recovery of the final CAT category."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_after_category_common import (
    ResolvedCatAfterCategoryStage,
    run_canonical_remaining_lora,
)
from train_utils.config.configs import (
    AfterCategoryResolvedConfig,
    AuxTrainableConfig,
    DistillDataConfig,
    DistillLossConfig,
    DistillOptimizationConfig,
    DistillRuntimeConfig,
    LoRAConfig,
)
from train_utils.distill_data import DistillDatasetBundle
from train_utils.model_level_trainables import build_model_level_trainable_selection


def _tiny_compressed_qwen():
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=9,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0,
        )
    )
    decoder = Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).float().extract_single(0)
    projection = VAELinear(
        in_features=16,
        out_features=16,
        bias=None,
        original_weight=None,
        vq_weight=torch.randint(0, 2, (64, 1, 9), dtype=torch.int64).bool(),
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )
    projection.pack_parallel_stage_decoder_(trainable=False)
    model.model.layers[0].mlp.down_proj = projection
    return model


@pytest.mark.parametrize("mode", ["none", "additive", "replace"])
def test_final_cat_category_trains_only_residual_lora(tmp_path, mode):
    torch.manual_seed(53)
    model = _tiny_compressed_qwen()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntiny\ntrain\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    cfg = AfterCategoryResolvedConfig(
        data=DistillDataConfig(model_max_length=8, dynamic_padding=False, group_by_length=False),
        loss=DistillLossConfig(loss_type="sft", prompt_loss_weight=1.0),
        opt=DistillOptimizationConfig(
            steps=1,
            batch_size=2,
            learning_rate=1e-5,
            gradient_checkpointing=False,
            warmup_ratio=0,
        ),
        lora=LoRAConfig(rank=2, alpha=4, dropout=0),
        aux=AuxTrainableConfig(
            residual_lora_mode=mode,
            residual_lora_rank=2,
            residual_lora_alpha=4,
            residual_lora_lr=1e-2,
        ),
        runtime=DistillRuntimeConfig(),
    )
    stage = ResolvedCatAfterCategoryStage(
        mode="remaining_lora",
        config=cfg,
        train_device="cpu",
        base_seed=53,
        stage_seed=53,
        output_dir=str(tmp_path),
        deterministic=False,
        fp16=False,
        bf16=False,
        reset_completed=False,
        save_strategy="no",
    )
    token_ids = [[5, 6, 7, 3], [7, 8, 5, 3]]
    dataset = Dataset.from_dict(
        {"input_ids": token_ids, "attention_mask": [[1] * 4] * 2, "labels": token_ids}
    )
    bundle = DistillDatasetBundle(
        train_dataset=dataset,
        eval_dataset=None,
        dataset_mix_spec=None,
        source_stats=[],
        is_iterable=False,
        cache_key=("tiny-cat-residual",),
        group_by_length=False,
    )
    cache_key = (
        cfg.data.dataset_mix,
        cfg.data.dataset_task,
        cfg.data.model_max_length,
        cfg.data.seed,
        cfg.data.data_seed,
        id(tokenizer),
    )
    vae_args = SimpleNamespace(
        _cached_lora_tokenizer=tokenizer,
        _cached_canonical_after_category_datasets={cache_key: bundle},
    )
    model.eval()
    probe = torch.tensor(token_ids)
    with torch.no_grad():
        original_logits = model(input_ids=probe).logits.clone()
    residual_names = set()
    if mode != "none":
        # Measure training from the installed topology in both modes.
        selection = build_model_level_trainable_selection(
            model,
            aux=cfg.aux,
            rank=2,
            alpha=4,
            dropout=0,
            train_lora=False,
        )
        residual_ids = {id(param) for param in selection.residual_lora_parameters.values()}
        residual_names = {name for name, param in model.named_parameters() if id(param) in residual_ids}
        assert residual_names
    model.eval()
    with torch.no_grad():
        before_logits = model(input_ids=probe).logits.clone()
    if mode in {"none", "additive"}:
        torch.testing.assert_close(before_logits, original_logits, rtol=0, atol=0)
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    result = run_canonical_remaining_lora(
        model=model,
        category="down_proj",
        compression_categories=("down_proj",),
        skip_layers=frozenset(),
        newly_compressed_target_count=1,
        stage=stage,
        vae_args=vae_args,
        logger=logging.getLogger(__name__),
    )
    assert result.did_train is (mode != "none")
    assert result.remaining_lora_target_count == 0
    assert result.decoder_target_count == 0
    after_state = result.model.state_dict()
    for name, before in before_state.items():
        if name not in residual_names:
            torch.testing.assert_close(after_state[name], before, rtol=0, atol=0)
    result.model.eval()
    with torch.no_grad():
        after_logits = result.model(input_ids=probe).logits
    if mode != "none":
        assert set(after_state) == set(before_state)
        assert any(not torch.equal(before_state[name], after_state[name]) for name in residual_names)
        assert not torch.equal(before_logits, after_logits)
        assert not any(parameter.requires_grad for parameter in result.model.parameters())
    else:
        assert set(after_state) == set(before_state)
        torch.testing.assert_close(after_logits, before_logits, rtol=0, atol=0)

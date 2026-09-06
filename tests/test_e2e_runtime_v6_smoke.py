from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import BertTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import load_v6_meta, save_v6_full_checkpoint


def _tiny_tokenizer(output_dir: Path):
    vocab_file = output_dir / "vocab.txt"
    vocab_file.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntiny\ntrain\nVAELLM\nfinalization\nparity\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.save_pretrained(str(output_dir))
    return tokenizer


def _tiny_vae_linear() -> VAELinear:
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
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
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )


def _build_local_round_base(tmp_path: Path) -> Path:
    base_dir = tmp_path / "Qwen2-tiny-base"
    base_dir.mkdir()
    config = Qwen2Config(
        vocab_size=12,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(str(base_dir), safe_serialization=False)
    tokenizer = _tiny_tokenizer(base_dir)
    model.model.layers[0].self_attn.q_proj = _tiny_vae_linear()

    round_base = tmp_path / "round_base"
    save_v6_full_checkpoint(
        model,
        str(round_base),
        checkpoint_kind="round_base",
        compressed_targets=("model.layers.0.self_attn.q_proj",),
        train_mode="none",
        base_model_path=str(base_dir),
        tokenizer=tokenizer,
        save_config=True,
    )
    return round_base


def test_runtime_v6_tiny_decoder_smoke_runs_to_atomic_final_model(tmp_path: Path, monkeypatch):
    round_base = _build_local_round_base(tmp_path)
    train_file = tmp_path / "train.jsonl"
    with train_file.open("w", encoding="utf-8") as handle:
        for text in ("hello world tiny train", "tiny train hello world"):
            handle.write(json.dumps({"text": text}) + "\n")

    argv = [
        "--student_checkpoint_dir", str(round_base),
        "--run_root_dir", str(tmp_path / "runs"),
        "--train_mode", "decoder",
        "--target_layers", "0",
        "--target_modules", "q_proj",
        "--train_file", str(train_file),
        "--dataset_task", "lm",
        "--loss_type", "sft",
        "--steps", "1",
        "--batch_size", "1",
        "--learning_rate", "1e-3",
        "--decoder_lr", "1e-3",
        "--weight_decay", "0.0",
        "--gradient_checkpointing", "false",
        "--model_max_length", "8",
        "--dynamic_padding", "true",
        "--parallel_mode", "layer_mp",
        "--layer_device_map", "cpu",
        "--offload_mode", "none",
        "--skip_ppl_eval", "true",
        "--save_tokenizer", "true",
        "--save_strategy", "no",
        "--logging_strategy", "no",
        "--report_to", "none",
        "--disable_tqdm", "true",
    ]
    cfg, hf_args, training_args = parse_args(argv)
    # Keep this synthetic CPU layer_mp fixture out of Trainer's single-process
    # DataParallel auto-wrap on hosts where multiple GPUs are visible.
    training_args._n_gpu = 1
    monkeypatch.setattr(
        "compressed_e2e_fintuning.runtime_v6_pipeline.default_dataloader_num_workers",
        lambda: 0,
    )

    result = run(cfg, hf_args, training_args)

    assert int(result["global_step"]) == 1
    final_dir = Path(str(result["saved_model_dir"]))
    assert final_dir.is_dir()
    meta = load_v6_meta(str(final_dir))
    assert meta["checkpoint_kind"] == "final_model"
    assert meta["train_mode"] == "decoder"
    assert meta["compressed_targets"] == ["model.layers.0.self_attn.q_proj"]
    assert meta["finalized_status"]["decoder_finalized"] is True
    assert meta["finalized_status"]["inference_forward_parity"] is True
    assert meta["runtime_audit"]["runtime"] == "compressed_e2e_fintuning.runtime_v6"
    assert meta["runtime_audit"]["structural_finalization_forward_parity"]["max_abs"] >= 0.0
    assert meta["runtime_audit"]["runtime_cleanup_forward_parity"]["max_abs"] >= 0.0
    assert meta["runtime_audit"]["finalization_forward_parity"]["max_abs"] >= 0.0

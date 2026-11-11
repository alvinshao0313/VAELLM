"""Actual tiny teacher KL through the formal builder; CPU mechanism evidence only."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time

from datasets import Dataset
import torch
from transformers import BertTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from e2e_common.data import DatasetMixSourcePreset
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_runtime_state

from . import data as d
from .contracts import digest, require
from .live import export_selected_node, source_pins
from .native_builder import RecoveryExecutionPolicy, make_native_recovery_trainer
from .nodes import RecoverySettings
from .replay import SelectedCategory


CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def tiny_root(directory: Path, tokenizer):
    torch.manual_seed(881)
    config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=2048, attention_dropout=.1, tie_word_embeddings=False,
        bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    config._attn_implementation = "sdpa"
    teacher = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    teacher_dir = directory / "tiny_dense_teacher"
    teacher.save_pretrained(teacher_dir)
    tokenizer.save_pretrained(teacher_dir)
    model = deepcopy(teacher).eval().requires_grad_(False)
    projections, prefix, future = [], [], []
    for index in range(2):
        for category in CATEGORIES:
            name = f"model.layers.{index}.{'self_attn' if category in CATEGORIES[:4] else 'mlp'}.{category}"
            old = model.get_submodule(name)
            projections.append(dict(actual_name=name, category_id=category, layer_id=index,
                shape=list(old.weight.shape), dtype="bfloat16", numel=old.weight.numel()))
            if category in ("q_proj", "k_proj"):
                decoder = Decoder(in_dim=9, out_dim=16, hidden_dim=8, num_res_blocks=0,
                    norm_type="layer", decoder_type="linear", use_checkpoint=False,
                    num_models=1).to(dtype=torch.bfloat16).extract_single(0)
                layer = VAELinear(in_features=16, out_features=16, bias=None, original_weight=None,
                    vq_weight=torch.randint(2, (16, 1, 9), dtype=torch.bool), decoder=decoder,
                    codebook_dim=16, transpose=False)
                layer.pack_parallel_stage_decoder_(trainable=False)
                parent, field = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), field, layer)
                prefix.append(name)
            else:
                future.append(name)
    identity = dict(schema="cpu_fixture_dense_content", model_type="qwen3",
        config_sha256=d.file_sha256(teacher_dir / "config.json"),
        weights_sha256=d.file_sha256(teacher_dir / "model.safetensors"))
    inventory = dict(model_identity=identity, source_evidence=dict(path=str(teacher_dir)),
        adapter_id="qwen3_dense_CAT_CPU_fixture", adapter_version=1, category_ids=list(CATEGORIES),
        projections=projections, total_target_numel=sum(row["numel"] for row in projections))
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None,
        runtime_identity=dict(phase="cpu_fixture", cat_shared=dict(channel_refresh_after_category=False)))
    raw_dir = directory / "round_base"
    raw = v6.save_v6_full_checkpoint(model, str(raw_dir), checkpoint_kind="round_base",
        compressed_targets=prefix, pending_dense_targets=future, completed_categories=["q_proj"],
        compression_categories=CATEGORIES, target_layers=[0, 1], target_modules=CATEGORIES,
        cat_runtime_state=runtime, base_model_path=str(teacher_dir),
        extra_meta=dict(cat_auto_phase="cpu_fixture", distill_stage_history=[dict(category="q_proj", did_train=False)]))
    del teacher, model
    return inventory, raw_dir, raw


def fixture_sources(directory: Path):
    sources = []
    for alias in d.MIX:
        rows = [dict(instruction="hello world", input="train " * 24 + ("tiny " * (index % 3)),
            output="answer world", fixture_row=index, fixture_source=alias) for index in range(6000)]
        path = directory / f"{alias}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        preset = DatasetMixSourcePreset(alias=alias, path=str(path), config=None,
            train_split="train", eval_split=None, text_field="text", text_format="alpaca", supports_sft=True)
        sources.append(d.bind_source(Dataset.from_list(rows), preset, revision="explicit_CPU_fixture_v0",
            license_record=dict(scope="locally authored CPU fixture"),
            provenance_record=dict(phase="cpu_fixture", generated_by=__file__, split="train"),
            artifact_digest=d.file_sha256(path)))
    return sources


def run(directory: Path):
    vocab = directory / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "tiny", "train", "answer"]
        + [f"unused{index}" for index in range(10, 128)]) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    inventory, raw_dir, raw = tiny_root(directory, tokenizer)
    sources = fixture_sources(directory)
    prepared = d.freeze_data_plan(sources, sources[0], tokenizer,
        model_identity=inventory["model_identity"], category_ids=["k_proj"], search_seed=31)
    policy = RecoveryExecutionPolicy(train_device="cpu", attention_implementation="sdpa",
        teacher_model_offload="none", teacher_output_offload="none", teacher_output_pin_memory=False)
    recovery = make_native_recovery_trainer(round_base_dir=raw_dir, inventory=inventory,
        category_id="k_proj", settings=RecoverySettings(), prepared_data=prepared,
        sources=sources, tokenizer=tokenizer, output_dir=directory / "KD_run",
        execution_policy=policy, phase="cpu_fixture", technical_stop_after=2)
    try:
        trainer = recovery.trainer
        require(trainer.loss_type == "kl_top_mass" and trainer.teacher_runtime.is_loaded,
            "Exercise the actual independently loaded teacher KL, not a substituted SFT loss.")
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
        result = recovery.train()
        changed = [name for name, parameter in trainer.model.named_parameters() if name in before and not torch.equal(parameter, before[name])]
        require(changed and all(torch.isfinite(parameter).all().item() for parameter in trainer.model.parameters()), "Actual KD must update finite native masters.")
        node_dir = directory / "KD_run/trainer_state/checkpoint-2"
        node = v6.load_v6_training_step_meta(str(node_dir))
        selected = SelectedCategory("k_proj", "CPU_fixture_recipe", "CPU_fixture_candidate", RecoverySettings(), 2,
            "CPU_fixture_KD_trajectory", node["checkpoint_id"], raw["checkpoint_id"],
            recovery.provenance["semantic_seeds"], phase="cpu_fixture")
        boundary = export_selected_node(node_directory=node_dir, destination=directory / "KD_boundary",
            selected=selected, current_targets=[row["actual_name"] for row in inventory["projections"] if row["category_id"] == "k_proj"],
            phase="cpu_fixture")
        require(boundary["fresh_evidence"]["state_exact"] and not boundary["formal_eligible"], "CPU actual-KD export must remain a CPU fixture.")
        return dict(phase="cpu_fixture", actual_teacher="Independent real locally saved tiny dense Qwen3 via original DistillTeacherRuntime",
            teacher_loss="original kl_top_mass/top100/T1/prompt.3", actual_steps=result["actual_steps"],
            original_schedule_max=result["schedule_max"], actual_batch=4, actual_workers=16,
            observed_batches=result["actual_data"]["actual_updates"], native_decoders=len(recovery.selection.decoder_targets),
            future_dense_lora_owners=len(recovery.selection.dense_targets), changed_master_leaves=len(changed),
            actual_optimizer=dict(type=type(trainer.optimizer).__name__, betas=list(trainer.optimizer.defaults["betas"]),
                eps=trainer.optimizer.defaults["eps"], groups=[dict(name=group["group_name"], initial_lr=group["initial_lr"], weight_decay=group["weight_decay"]) for group in trainer.optimizer.param_groups]),
            max_grad_norm=trainer.args.max_grad_norm, original_cosine_warmup=trainer.args.warmup_ratio,
            root_data_identity=prepared["data_identity"], semantic_seeds=recovery.provenance["semantic_seeds"],
            observed_input_identity=result["actual_data"]["token_order_digest"],
            actual_save1_return_to_step2=len(result["save_audits"]) == 2,
            own_round_base_node_export_fresh_exact=True, formal_eligible=False,
            limitation="Actual tiny CPU builder/KD/data/save/export connection only; no model-scale GPU/runtime/quality/BF16 replay claim.")
    finally:
        recovery.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "Hide CUDA for this CPU-only builder check.")
    output = Path(args.output).resolve()
    require(not output.exists(), "Do not overwrite earlier CPU evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="native_builder_CPU_", dir=output.parent) as directory:
        evidence = run(Path(directory))
    report = dict(status="actual_native_KD_builder_CPU_verified", evidence=evidence,
        source_pins=source_pins(), elapsed_seconds=time.monotonic() - began,
        CUDA_initialized=torch.cuda.is_initialized(), temporary_models_removed=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()

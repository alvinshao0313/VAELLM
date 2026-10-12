"""Prepare the fixed isolated probe; never reserve or launch a GPU."""
import argparse
import json
from pathlib import Path
import sys

from experiments.longbench_v1.common import sha256, source_pins, write_json
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.long_context_recovery.data import read_manifest


def training_arguments(manifest, output, alias):
    return ["--long_context_manifest", str(Path(manifest).resolve()),
        "--student_checkpoint_dir", str(SOURCE), "--run_root_dir", str(Path(output).resolve()),
        "--train_mode", "decoder", "--target_layers", "all", "--target_modules", "v_proj,o_proj",
        "--distill_fp32_components", "decoder", "--norm_train_mode", "none", "--lm_head_train_mode", "none",
        "--residual_lora_mode", "none", "--dataset_mix", alias + "=1", "--dataset_task", "sft",
        "--model_max_length", "8192", "--dynamic_padding", "true", "--group_by_length", "false",
        "--seed", "0", "--data_seed", "0", "--loss_type", "kl_top_mass", "--top_k", "100",
        "--temperature", "1", "--prompt_loss_weight", "0.3", "--hidden_loss_weight", "0",
        "--pre_mlp_hidden_loss_weight", "0", "--selective_student_topk", "false",
        "--batch_size", "1", "--gradient_accumulation_steps", "1", "--steps", "2000",
        "--decoder_lr", "1e-5", "--learning_rate", "1e-5", "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.01", "--weight_decay", "0", "--max_grad_norm", "1.5", "--logging_steps", "1",
        "--gradient_checkpointing", "true", "--gradient_checkpointing_kwargs", '{"use_reentrant": false}',
        "--vae_decoder_checkpoint", "true", "--parallel_mode", "dp", "--teacher_output_offload", "cpu",
        "--teacher_model_offload", "none", "--teacher_output_pin_memory", "true",
        "--teacher_output_chunk_tokens", "8", "--offload_mode", "none", "--bf16", "true",
        "--eval_after_save", "false", "--eval_tasks", "", "--skip_ppl_eval", "true",
        "--save_strategy", "steps", "--save_steps", "2", "--save_total_limit", "2",
        "--save_tokenizer", "true", "--report_to", "none", "--full_determinism", "false"]


def prepare(manifest_path, output):
    from experiments.long_context_recovery.train import parse_configuration
    from experiments.w25_eval.subset_export import CODE_FILES
    root = Path(__file__).resolve().parents[2]
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Prepare a new independent diagnostic directory.")
    manifest = read_manifest(manifest_path)
    cpu_path = manifest_path.parent / "resource_cpu_validation.json"
    cpu = json.loads(cpu_path.read_text())
    require(cpu["status"] == "passed" and cpu["manifest_sha256"] == sha256(manifest_path)
            and len(cpu["selected_records"]) == 4
            and all(row["input_tokens"] == 8192 and row["non_special_assistant_tokens"] > 0
                    for row in cpu["selected_records"]),
            "The native CPU tokenizer/data proof must cover all four real long assistant examples.")
    argv = training_arguments(manifest_path, output, manifest["alias"])
    cfg, _hf, args, names, architecture, inventory = parse_configuration(argv)
    require(manifest["purpose"] == "resource_diagnostic", "Only resource-selected original-format records are permitted.")
    import torch
    require(not torch.cuda.is_initialized(), "Preparation must remain CPU-only.")
    sources = source_pins()
    sources.update({str(root / name): sha256(root / name) for name in CODE_FILES})
    for directory in (root / "compressed_e2e_fintuning", Path(__file__).parent):
        sources.update({str(path): sha256(path) for path in directory.rglob("*.py")})
    meta = json.loads((SOURCE / "checkpoint_meta.json").read_text())
    model_path = Path(meta["base_model_path"])
    evidence_paths = [manifest_path, cpu_path, Path(manifest["source"]["path"]), Path(manifest["output"]["path"]),
                      SOURCE / "checkpoint_meta.json", SOURCE / meta["state_dict_file"],
                      model_path / "config.json", model_path / "tokenizer_config.json",
                      model_path / "tokenizer.json", model_path / "generation_config.json"]
    from experiments.longbench_v1.common import file_stat
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    record = {"schema": "long_context_ov72_resource_v1", "status": "prepared_not_scheduled",
        "gpu": None, "queued": False, "automatic_retry": False,
        "source_checkpoint": str(SOURCE), "source_checkpoint_id": SOURCE_ID,
        "output_dir": str(output), "manifest": str(manifest_path), "training_argv": argv,
        "argv": [sys.executable, "-u", "-m", "experiments.long_context_recovery.train", *argv],
        "source_sha256": sources, "input_evidence_sha256": {str(path): sha256(path) for path in evidence_paths},
        "hf_weight_stats": [file_stat(model_path / name) for name in sorted(set(index["weight_map"].values()))],
        "selected_targets": list(names), "expected_inventory": inventory,
        "data_registration": cfg.data.data_registration,
        "environment": {"CUDA_VISIBLE_DEVICES": None, "OMP_NUM_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(root),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "resource_budget_seconds": 1180, "termination_grace_seconds": 20,
        "training_timeout_seconds": 600, "export_timeout_seconds": 180,
        "audit_timeout_seconds": 240, "verify_timeout_seconds": 120,
        "resource_note": "Await root GPU/time binding; no queue/retry. Four real8192-token batches; no quality claim.",
        "schedule": {"max_steps": args.max_steps, "warmup_steps": args.get_warmup_steps(args.max_steps),
                     "save_steps": args.save_steps, "save_total_limit": args.save_total_limit, "stop_step": 4},
        "architecture": architecture, "cuda_initialized": torch.cuda.is_initialized()}
    output.mkdir(parents=True)
    write_json(output / "launch.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = prepare(args.manifest, args.output)
    print(json.dumps({key: value[key] for key in ("status", "output_dir", "gpu", "expected_inventory", "schedule", "cuda_initialized")}, indent=2))

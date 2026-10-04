"""Fixed recipes for rebuilding the measured Qwen3-8B W2.5 success chain.

All values are kept in source: running the fresh pipeline does not require old
result directories. Commands match the historical export-callback entrypoints;
the fresh pipeline substitutes its endpoint-only wrapper, retaining the native
training arguments and exporting saved checkpoints after the independent jobs finish.
"""
from __future__ import annotations

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
TASK_MIX = "vaellm_eval_task=1"
MIXED_DATA = (
    "edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,"
    "edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.500"
)
DATA_FILES = (
    "data/edgerazor_qwen3/ii_7M_instruct.jsonl",
    "data/edgerazor_qwen3/ii_gen_1.4M_instruct.jsonl",
    "data/edgerazor_qwen3/tulu_0.6M_instruct.jsonl",
    "data/edgerazor_qwen3/am_1.4M_instruct.jsonl",
    "data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl",
)


def cat_command(*, hf_model, output_dir) -> list[str]:
    """Use the original CAT recipe, overriding only its model and output paths."""
    return ["bash", str(REPO / "scripts/cat_w25_remaining.sh"),
            "--model_path", str(hf_model), "--output_dir", str(output_dir)]


def training_command(recipe: str, *, source_dir, hf_model, output_dir, export_gpu) -> list[str]:
    """Build one measured E2E recipe or the fixed256 optimizer reference.

    ``formal_reference`` is configuration for the fixed256 recovery stage. It
    must not be run as another training branch: that stage consumes its parsed
    optimizer/runtime settings and supplies its own fixed batches and target.
    """
    if recipe not in {"ce10", "ce50", "decoder", "last_joint", "formal_reference"}:
        raise ValueError(f"Unknown Qwen3 W2.5 reproduction recipe: {recipe}")
    subset = recipe == "last_joint"
    pure_kl = recipe in {"decoder", "formal_reference"}
    task_only = recipe in {"ce50", "last_joint"}
    bit_mode = subset or recipe == "formal_reference"
    entry = ("experiments.qwen3_w25.train_subset_with_step_exports" if subset
             else "experiments.qwen3_w25.train_with_step_exports")
    command = [
        sys.executable, "-m", entry,
        "--export_cuda_visible_devices", str(export_gpu),
        "--student_checkpoint_dir", str(source_dir),
        "--teacher_model_path", str(hf_model),
        "--run_root_dir", str(output_dir),
        "--train_mode", "decoder_sparse_bit" if bit_mode else "decoder",
        "--distill_fp32_components", "decoder",
        "--seed", "0", "--data_seed", "0",
        "--dataset_mix", TASK_MIX if task_only else MIXED_DATA,
        "--dataset_task", "sft", "--dynamic_padding", "true",
        "--model_max_length", "1024", "--group_by_length", "true",
        "--target_layers", "0" if subset else "0-35",
        "--target_modules", "down_proj" if subset else "all",
        "--loss_type", "kl_top_mass" if pure_kl else "kd_top_mass",
        "--top_k", "100", "--temperature", "1",
        "--prompt_loss_weight", "0" if task_only else "0.3",
        "--hidden_loss_weight", "0", "--pre_mlp_hidden_loss_weight", "0",
        "--selective_student_topk", "false",
        "--steps", "2000", "--batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--decoder_lr", "1e-5", "--learning_rate", "1e-5",
        "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.01",
        "--weight_decay", "0", "--max_grad_norm", "1.5",
        "--logging_steps", "10", "--gradient_checkpointing", "true",
        "--gradient_checkpointing_kwargs", '{"use_reentrant": false}',
        "--residual_lora_mode", "none", "--norm_train_mode", "none",
        "--lm_head_train_mode", "none", "--parallel_mode", "dp",
        "--teacher_output_offload", "cpu" if pure_kl else "none",
        "--teacher_model_offload", "none", "--teacher_output_pin_memory", "true",
        "--teacher_output_chunk_tokens", "8", "--vae_decoder_checkpoint", "true",
        "--offload_mode", "none", "--eval_after_save", "false",
        "--eval_hif4_act", "false", "--eval_tasks", "",
        "--eval_num_fewshot", "0", "--eval_batch_size", "auto",
        "--eval_device", "cuda", "--eval_prewarm_group_size", "8",
        "--skip_ppl_eval", "true", "--save_tokenizer", "true",
        "--full_determinism", "false", "--bf16", "true",
        "--eval_strategy", "no", "--save_strategy", "steps",
        "--save_steps", "400", "--save_total_limit", "1",
    ]
    if not pure_kl:
        command.extend(["--alpha", "0.9" if recipe == "ce10" else "0.5"])
    if subset:
        command.extend([
            "--bit_active_ratio", "1", "--bit_optimizer", "adam",
            "--bit_lr", "4e-5", "--bit_round_steps", "512",
            "--bit_weight_decay", "0", "--bit_proxy_coordinates", "decoder_sensitivity",
        ])
    elif pure_kl:
        # These settings are inactive for decoder-only; preserve its actual CLI.
        command.extend([
            "--bit_active_ratio", "0.05", "--bit_proxy_coordinates", "decoder_sensitivity",
            "--bit_optimizer", "adam", "--bit_lr", "4e-5",
            "--bit_weight_decay", "0", "--bit_round_steps", "128",
        ])
    return command


def format_protocol() -> dict:
    """The measured ChatML CE50 joint256 arm, including its 2000-step schedule."""
    return {
        "fresh_source_for_every_arm": True,
        "loss_type": "kd_top_mass", "alpha": 0.5,
        "top_k": 100, "temperature": 1, "prompt_loss_weight": 0.0,
        "batch_size": 4, "gradient_accumulation_steps": 1,
        "optimizer_updates": 256, "training_sequences": 1024, "heldout_sequences": 32,
        "decoder_lr": 1e-5, "decoder_clip": 1.5, "weight_decay": 0,
        "decoder_scheduler": "cosine2000/warmup20 (not a new256-step schedule)",
        "trainable_decoder_tensors": 6, "trainable_decoder_elements": 21312,
        "code_active_ratio": 1, "active_score_elements": 147849216,
        "bit_optimizer": "adam", "bit_lr": 4e-5, "bit_round_steps": 512,
        "bit_stable_steps": 10,
        "round_resets": "existing production stable10/round rule; not forced-fixed rounds",
        "trainable_norm_head_or_lora": False,
        "frozen_tensors_after_promotion": 3415,
        "all_decoder_storage": "FP32", "compute_dtype": "BF16",
        "teacher_loader": "existing runtime_v6 _load_teacher dtype=None reference(B)",
        "teacher_logits_offload": "cpu", "teacher_chunk_tokens_argument": 8,
        "frozen_native_cache_modules": 251,
        "heldout_steps": [0, 128, 256], "heldout_formats": ["chatml", "bare_space"],
        "archive_cached_prompt_weight": 0.3, "archive_cached_denominator_used": False,
    }

"""Fixed scientific and resource contract for the four-step OV72 probe."""
from pathlib import Path


SOURCE_ID = "f927ffda-dd5b-4d00-b63a-5e839a182d70"
SOURCE = Path("/home/star/VAELLM/result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model")


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_config(cfg):
    require(Path(cfg.student_checkpoint_dir).resolve() == SOURCE,
            "The probe starts from the fixed f927 full-model source.")
    require(cfg.train_mode == "decoder" and tuple(cfg.opt.distill_fp32_components) == ("decoder",)
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none",
            "Only FP32 decoder leaves may train; codes, norm and head are frozen, without LoRA.")
    require(cfg.target_layers == "all" and set(cfg.target_modules) == {"v_proj", "o_proj"},
            "The resource probe selects all OV72 projections.")
    require(cfg.loss.loss_type == "kl_top_mass" and cfg.loss.top_k == 100
            and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == .3
            and cfg.loss.hidden_loss_weight == cfg.loss.pre_mlp_hidden_loss_weight == 0
            and not cfg.loss.selective_student_topk,
            "Keep pure KL-mass K100/T1/prompt0.3, without CE, hidden losses or selective-head changes.")
    require(cfg.data.dataset_task == "sft" and cfg.data.model_max_length == 8192
            and cfg.data.dynamic_padding and not cfg.data.train_file
            and cfg.data.dataset_mix in ("longalign=1", "longalpaca=1"),
            "Use one registered native long SFT preset at max8192, with original truncation/masks.")
    require(cfg.opt.steps == 2000 and cfg.opt.batch_size == cfg.opt.gradient_accumulation_steps == 1
            and cfg.opt.warmup_ratio == .01 and cfg.opt.lr_scheduler_type == "cosine"
            and cfg.opt.decoder_lr == cfg.opt.learning_rate == 1e-5
            and cfg.opt.gradient_checkpointing
            and cfg.opt.gradient_checkpointing_kwargs == {"use_reentrant": False},
            "Keep B1/accum1 and the native 2000-step cosine schedule with warmup20 and both checkpoint paths.")
    require(cfg.runtime.parallel_mode == "dp" and cfg.runtime.offload_mode == "none"
            and cfg.runtime.teacher_model_offload == "none" and cfg.runtime.teacher_output_offload == "cpu"
            and cfg.runtime.vae_decoder_checkpoint and not cfg.runtime.distill_hif4_act
            and not cfg.runtime.evaluation.eval_after_save
            and not str(cfg.runtime.evaluation.eval_tasks or "").strip()
            and cfg.runtime.evaluation.skip_ppl_eval,
            "Use one DP GPU, resident teacher, CPU targets and separate final export; no in-training evaluation.")
    require(cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            "Start fresh; the isolated observation callback stops at optimizer step4.")


def validate_training_args(args):
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
            and args.save_steps == 2 and args.save_total_limit == 2
            and getattr(args.save_strategy, "value", args.save_strategy) == "steps"
            and args.bf16 and not args.fp16,
            "Preserve the scientific schedule, BF16 compute and both native saves at steps2/4.")

"""Frozen-state, full-model KL gradient observations; no optimization or export."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch

SOURCE_ID = "552282f2-868a-4105-90df-46f3c6f6a818"
TARGETS = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj")
SELECTION_SEED = 137


def score_epsilon_statistics(gradients, *, scale, lr=4e-5, eps=1e-8):
    """Fresh Adam first-step scalar reference, before score clamping."""
    magnitude = torch.as_tensor(gradients).detach().cpu().double().abs().reshape(-1)
    if not len(magnitude) or not torch.isfinite(magnitude).all() or min(scale, lr, eps) <= 0:
        raise ValueError("Need finite observations and positive scale, learning rate, and epsilon.")
    q = torch.tensor([0., .01, .1, .25, .5, .75, .9, .99, 1.], dtype=torch.float64)
    reference = lr * magnitude / (magnitude + eps)
    return {"observations": len(magnitude), "eps": eps, "lr": lr, "score_scale": scale,
            "initial_threshold_distance": scale / 2, "quantile_probabilities": q.tolist(),
            "absolute_score_gradient_quantiles": torch.quantile(magnitude, q).tolist(),
            "fraction_below_eps": float((magnitude < eps).double().mean()),
            "fraction_equal_eps": float((magnitude == eps).double().mean()),
            "fraction_above_eps": float((magnitude > eps).double().mean()),
            "fresh_first_step_unclamped_magnitude_quantiles": torch.quantile(reference, q).tolist(),
            "interpretation": "lr*abs(g)/(abs(g)+eps) in exact scalar arithmetic for fresh first Adam step only; excludes score clamp and is not a 128-step trajectory"}


def gradient_statistics(gradients, weights):
    """Descriptive FP64 statistics of already-computed FP32 batch gradients."""
    g = torch.as_tensor(gradients).detach().cpu().double()
    w = torch.as_tensor(weights).detach().cpu().double()
    if g.ndim != 2 or w.shape != (len(g),) or not len(g) or not g.shape[1]:
        raise ValueError("Need nonempty [batches, coordinates] and one weight per batch.")
    if not torch.isfinite(g).all() or not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError("Finite gradients and positive weights are required.")
    w = w / w.sum()
    mean = (g * w[:, None]).sum(0)
    rms = (g.square() * w[:, None]).sum(0).sqrt()
    norm = g.norm(dim=1)
    denom = norm[:, None] * norm[None, :]
    valid = denom > 0
    cosine = torch.where(valid, (g @ g.T) / denom.clamp_min(torch.finfo(g.dtype).tiny), 0)
    positive, negative = (g > 0).sum(0), (g < 0).sum(0)
    nonzero = positive + negative
    majority = torch.where(nonzero > 0, torch.maximum(positive, negative) / nonzero.clamp_min(1), 0)
    noise = ((g - mean).square().sum(1) * w).sum().sqrt()
    mean_norm, weighted_norm = float(mean.norm()), float((norm * w).sum())
    return {"batches": len(g), "coordinates": g.shape[1], "normalized_weights": w.tolist(),
            "cosine": cosine.tolist(), "cosine_valid": valid.tolist(), "gradient_norms": norm.tolist(),
            "mean_gradient_norm": mean_norm, "weighted_mean_gradient_norm": weighted_norm,
            "cancellation_ratio": mean_norm / weighted_norm if weighted_norm else None,
            "vector_snr": mean_norm / float(noise) if noise > 0 else None,
            "zero_noise": bool(noise == 0), "nonzero_counts": nonzero.tolist(),
            "positive_counts": positive.tolist(), "negative_counts": negative.tolist(),
            "unweighted_nonzero_sign_majority_fraction": majority.tolist(),
            "coordinate_mean_over_rms": torch.where(rms > 0, mean / rms.clamp_min(torch.finfo(g.dtype).tiny), 0).tolist()}


def batch_manifest(batch, batch_index, *, sample_offset, prompt_weight=0.3):
    from train_utils.distill_loss_core import build_prediction_token_masks
    values = {k: batch[k].detach().cpu() for k in ("input_ids", "attention_mask", "labels")}
    response, prompt = build_prediction_token_masks(labels=values["labels"], attention_mask=values["attention_mask"])
    denominator = float((response + prompt_weight * prompt).sum().clamp_min(1))
    records = []
    for i in range(len(values["input_ids"])):
        valid = values["attention_mask"][i].bool()
        digest = hashlib.sha256()
        for key in ("input_ids", "labels"):
            digest.update(values[key][i][valid].contiguous().numpy().tobytes())
        records.append({"iterator_sample_ordinal": sample_offset + i, "input_and_label_sha256": digest.hexdigest(),
                        "input_tokens": int(valid.sum()), "causal_response_tokens": int(response[i].sum()),
                        "causal_prompt_tokens": int(prompt[i].sum())})
    return {"batch_index": batch_index, "shape": list(values["input_ids"].shape), "samples": records,
            "causal_response_tokens": int(response.sum()), "causal_prompt_tokens": int(prompt.sum()),
            "loss_denominator_fp32": denominator, "prompt_loss_weight": prompt_weight,
            "index_scope": "consumed iterator order; raw source row indices are not exposed by the canonical builder"}


def read_formal_config(formal_launch, output):
    from compressed_e2e_fintuning.args import parse_args
    launch = json.loads(Path(formal_launch).read_text())
    argv = list(launch["actual_argv"])
    if argv[1:3] != ["-m", "experiments.qwen3_w25.train_with_step_exports"]:
        raise ValueError("Expected the known completed bits05 formal entry.")
    argv = argv[3:]
    i = argv.index("--export_cuda_visible_devices")
    del argv[i:i + 2]
    for flag in ("--run_root_dir", "--output_dir"):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(output)
    cfg, hf_args, training_args = parse_args(argv)
    if (cfg.loss.loss_type != "kl_top_mass" or cfg.loss.top_k != 100 or cfg.loss.temperature != 1
            or cfg.loss.prompt_loss_weight != 0.3 or cfg.loss.hidden_loss_weight or cfg.loss.pre_mlp_hidden_loss_weight
            or cfg.loss.selective_student_topk or cfg.opt.batch_size != 4 or cfg.opt.gradient_accumulation_steps != 1
            or cfg.data.model_max_length != 1024 or cfg.data.seed != 0 or cfg.data.data_seed != 0
            or cfg.opt.steps != 2000 or cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none"
            or cfg.runtime.teacher_output_offload != "cpu" or cfg.runtime.teacher_model_offload != "none"
            or cfg.train_mode != "decoder_sparse_bit" or cfg.opt.distill_fp32_components != ("decoder",)
            or cfg.data.dataset_task != "sft" or not cfg.data.dynamic_padding
            or cfg.data.dataset_mix_weights != (.341, .067, .028, .064, .5)
            or cfg.bit_optimizer != "adam" or float(cfg.bit_lr) != 4e-5 or cfg.bit_weight_decay != 0
            or int(cfg.bit_round_steps) != 128 or cfg.bit_proxy_coordinates != "decoder_sensitivity"
            or not cfg.runtime.vae_decoder_checkpoint or cfg.runtime.distill_hif4_act
            or not args_checkpointing_matches(training_args)
            or cfg.aux.norm_train_mode != "none" or cfg.aux.lm_head_train_mode != "none"
            or cfg.aux.residual_lora_mode != "none" or not training_args.bf16 or training_args.fp16):
        raise ValueError("Formal configuration differs from the approved diagnostic reference.")
    return cfg, hf_args, training_args


def args_checkpointing_matches(args):
    return bool(args.gradient_checkpointing) and args.gradient_checkpointing_kwargs == {"use_reentrant": False}


def _coordinate_plan(module, module_name):
    from sparse_bit_tuning.sampler import AffineSamplerState
    parameters = list(sorted(module._parallel_stage_decoder.named_parameters()))
    total = sum(p.numel() for _, p in parameters)
    state = AffineSamplerState.create(canonical_key=module_name + "|decoder_gradient", training_seed=SELECTION_SEED,
                                     n_bits=total, n_active=256)
    indices = state.active_indices()
    mapping = []
    for index in indices:
        cursor = 0
        for name, parameter in parameters:
            if index < cursor + parameter.numel():
                mapping.append({"parameter": name, "flat_index_in_parameter": index - cursor, "flat_index": index})
                break
            cursor += parameter.numel()
    return parameters, indices, {"total_coordinates": total, "observed_coordinates": 256,
                                 "sampler": state.to_metadata(), "coordinates": mapping}


def run_kl_gradient_diagnostic(formal_launch, output_dir):
    from transformers.trainer_pt_utils import AcceleratorConfig
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _load_teacher, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from e2e_common.lazy_datasets import default_dataloader_num_workers
    from experiments.qwen3_w25.local_bits import _digest
    from litebsq.vae_linear import VAELinear
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    from train_utils.distill_data import build_distill_dataset, build_distill_data_collator
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    import logging

    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Use a fresh diagnostic output directory.")
    cfg, hf_args, args = read_formal_config(formal_launch, output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one authorized CUDA device before starting.")
    source = Path(cfg.student_checkpoint_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must not be inside the protected source.")
    output.mkdir(parents=True)
    device = torch.device("cuda:0")
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_dir": str(source),
        "formal_launch": str(Path(formal_launch).resolve()),
        "formal_launch_sha256": hashlib.sha256(Path(formal_launch).read_bytes()).hexdigest(),
        "purpose": "fixed-state cross-batch gradients under formal full-model KL; no optimization, finite flip, or export",
        "configuration": {"distinct_batches": 8, "repeat_first_batch": 1, "batch_size": 4, "max_length": 1024,
            "canonical_max_steps": 2000, "selection_seed": SELECTION_SEED, "model_seed": 0, "data_seed": 0,
            "observed_bits_per_stage": 128, "observed_decoder_coordinates_per_target": 256,
            "decoder_trainable_modules": 252, "targets": list(TARGETS), "gradient_clipping": False},
        "limitations": ["The sparse registration coverage differs from formal 5%; hard forward state never changes.",
            "Equal-batch statistics and loss-denominator-weighted statistics are distinct.",
            "Weighted gradient aggregation is descriptive; not a claim of bitwise equivalence to one larger batch.",
            "STE gradients do not establish finite-flip benefit or decoder compensation during optimization.",
            "The repeated batch is a numerical repeat observation, not a certified error bound.",
            "Small fixed coordinate samples and eight batches do not establish whole-model or downstream behavior."],
        "batches": [], "gradient_rows": {}, "coordinate_plans": {}}
    repo = Path(__file__).resolve().parents[2]
    pin_names = ["experiments/qwen3_w25/kl_gradient_diagnostic.py", "experiments/qwen3_w25/diagnose_kl_gradients.py"]
    formal = json.loads(Path(formal_launch).read_text())
    pins = dict(formal["source_sha256"])
    for name in pin_names:
        pins[str(repo / name)] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    for name, expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Formal source pin changed: {name}")
    report["code_sha256"] = pins
    path = output / "diagnostic.json"
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    started, trainer, manager = time.perf_counter(), None, None
    try:
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(0)
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
        if meta["checkpoint_id"] != SOURCE_ID:
            raise ValueError("Unexpected source checkpoint identity.")
        modules = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, VAELinear)]
        if len(modules) != 252:
            raise ValueError("Expected all 252 converted modules.")
        _enable_internal_decode_runtime(modules, decoder_enabled=True, sparse_enabled=True, vae_decoder_checkpoint=True)
        selection = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=modules,
            rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="decoder_sparse_bit", freeze=True, fp32_components=cfg.opt.distill_fp32_components)
        model = selection.peft_model or model
        configure_distill_precision(selection, components=cfg.opt.distill_fp32_components, training_args=args)
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(args.gradient_checkpointing_kwargs))
        model.to(device)
        selected = [(name, model.get_submodule(name)) for name in TARGETS]
        n_gate = math.prod(selected[0][1].get_stage_part_vq_spec(0, 0)["logical_shape"])
        manager = SparseBitTuningManager(root_model=model, targets=selected,
            target_devices={name: device for name in TARGETS}, training_seed=SELECTION_SEED, streaming=False,
            config=SparseBitTuningConfig(enabled=True, active_ratio=128/n_gate, optimizer="adam", bit_lr=4e-5,
                round_steps=128, proxy_coordinates="decoder_sensitivity"), calibration_dtype=torch.bfloat16)
        manager.initialize_scores()
        report["sparse_registration"] = {"active_ratio": 128/n_gate,
            "formal_optimizer": cfg.bit_optimizer, "optimizer": manager.bit_optimizer.optimizer_name,
            "bit_lr": manager.bit_optimizer.lr, "weight_decay": manager.bit_optimizer.weight_decay,
            "round_steps_reference": 128, "adam_eps": 1e-8, "optimizer_steps": 0,
            "total_source_bits_in_targets": sum(s.n_bits for s in manager.bank_specs),
            "registered_score_coordinates": sum(s.n_active for s in manager.bank_specs), "observed_coordinates": 512,
            "banks": [{"key": s.canonical_key, "n_bits": s.n_bits, "registered": s.n_active, "observed": 128,
                       "logical_indices": manager.sampler_states[s.canonical_key].active_indices()[:128],
                       "scale": manager.score_module.coordinate_scales[s.canonical_key],
                       "initial_threshold_distance": manager.score_module.coordinate_scales[s.canonical_key]/2} for s in manager.bank_specs]}
        if any(s.n_active < 128 for s in manager.bank_specs):
            raise RuntimeError("Insufficient predetermined sparse coordinates.")
        plans = {}
        for name, layer in selected:
            parameters, indices, plan = _coordinate_plan(layer, name)
            plans[name] = (parameters, indices)
            report["coordinate_plans"][name] = plan
        tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), hf_args)
        _sync_model_padding_config(model, tokenizer)
        bundle = build_distill_dataset(cfg.data, tokenizer)
        if not bundle.is_iterable:
            raise ValueError("Expected the same canonical mixed iterable training dataset.")
        args.output_dir = str(output / "unused_trainer_output")
        args.dataloader_num_workers = default_dataloader_num_workers()
        args.dataloader_pin_memory = True
        args.group_by_length = False
        args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
        args.remove_unused_columns = False
        args.report_to = []
        collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
        teacher, teacher_identity = _load_teacher(cfg, hf_args, args, model, base_model_path=str(meta["base_model_path"]),
                                                dp_device=device, log=logging.getLogger(__name__))
        trainer = VAEDecoderE2ETrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=bundle.train_dataset,
            eval_dataset=None, data_collator=collator, loss_config=cfg.loss, teacher_model=teacher,
            teacher_output_offload="cpu", teacher_model_offload="none", teacher_output_pin_memory=True,
            teacher_output_chunk_tokens=8, sparse_bit_manager=manager, callbacks=[])
        model_for_step = trainer.accelerator.prepare_model(model)
        loader = trainer.get_train_dataloader()
        iterator = iter(loader)
        before = _digest(model.state_dict(), set())
        report["data_source_stats"] = bundle.source_stats
        report["teacher_identity"] = teacher_identity
        report["runtime"] = {"loss": vars(cfg.loss), "data": vars(cfg.data),
            "dataloader_num_workers": args.dataloader_num_workers, "decoder_parameters": len(selection.decoder_parameters),
            "decoder_elements": sum(p.numel() for p in selection.decoder_parameters.values()),
            "full_determinism": args.full_determinism, "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "accelerator_mixed_precision": trainer.accelerator.mixed_precision}
        if len(selection.decoder_parameters) != 1512 or any(not p.requires_grad or p.dtype != torch.float32 for p in selection.decoder_parameters.values()):
            raise RuntimeError("Full formal FP32 decoder gradient inventory was not preserved.")
        report["status"] = "running"
        save()
        gradients, first_batch = {}, None
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(9):
            batch = next(iterator) if index < 8 else {key: value.clone() for key, value in first_batch.items()}
            if index == 0:
                first_batch = {key: value.detach().cpu().clone() for key, value in batch.items()}
            record = batch_manifest(batch, index, sample_offset=index*4 if index < 8 else 0)
            record["repeat_of_batch"] = 0 if index == 8 else None
            model.zero_grad(set_to_none=True)
            loss = trainer.training_step(model_for_step, batch)
            if trainer.optimizer is not None or trainer.lr_scheduler is not None:
                raise RuntimeError("A diagnostic unexpectedly constructed a main optimizer or scheduler.")
            if any(p.grad is None for p in selection.decoder_parameters.values()):
                raise RuntimeError("A formal decoder gradient is missing.")
            frozen = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            if frozen:
                raise RuntimeError(f"Frozen parameters received gradients: {frozen[:4]}")
            manager.bit_optimizer.validate_gradients()
            for name, _layer in selected:
                for stage in range(2):
                    spec = manager.bank_spec(name, stage, 0)
                    raw = manager.score_module.score_chunks[spec.chunk_id].grad[spec.score_start:spec.score_start+128].detach().cpu().float()
                    scale = manager.score_module.coordinate_scales[spec.canonical_key]
                    gradients.setdefault(f"{name}|stage{stage}|score", []).append(raw.clone())
                    gradients.setdefault(f"{name}|stage{stage}|bit", []).append(raw * scale)
                parameters, indices = plans[name]
                vector = torch.cat([p.grad.detach().reshape(-1).cpu().float() for _, p in parameters])[indices]
                gradients.setdefault(f"{name}|decoder", []).append(vector)
            if any(not torch.isfinite(rows[-1]).all() for rows in gradients.values()):
                raise FloatingPointError("A recorded gradient is nonfinite.")
            record["loss"] = float(loss)
            report["batches"].append(record)
            report["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"batch": index, "loss": float(loss), "denominator": record["loss_denominator_fp32"],
                              "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
            save()
        for name in TARGETS:
            gradients[f"{name}|bit_both_stages"] = [torch.cat([gradients[f"{name}|stage{s}|bit"][i] for s in range(2)]) for i in range(9)]
        denominators = [row["loss_denominator_fp32"] for row in report["batches"][:8]]
        for key, rows in gradients.items():
            matrix = torch.stack(rows)
            report["gradient_rows"][key] = matrix.tolist()
            first, repeat = matrix[0].double(), matrix[8].double()
            report.setdefault("statistics", {})[key] = {
                "equal_batch": gradient_statistics(matrix[:8], [1]*8),
                "loss_denominator_weighted": gradient_statistics(matrix[:8], denominators),
                "repeat_first": {"max_abs": float((first-repeat).abs().max()),
                    "relative_l2": float((first-repeat).norm()/first.norm()) if first.norm() else None,
                    "nonzero_sign_disagreements": int(((first*repeat)<0).sum()),
                    "zero_count_first": int((first==0).sum()), "zero_count_repeat": int((repeat==0).sum())}}
        for name in TARGETS:
            for stage in range(2):
                key = f"{name}|stage{stage}|score"
                spec = manager.bank_spec(name, stage, 0)
                scale = manager.score_module.coordinate_scales[spec.canonical_key]
                rows = gradients[key]
                report.setdefault("score_epsilon", {})[key] = {
                    "distinct_batches_pooled": score_epsilon_statistics(torch.stack(rows[:8]), scale=scale),
                    "per_batch": [score_epsilon_statistics(row, scale=scale) for row in rows],
                    "pooling": "8 batches x 128 predetermined coordinates; repeated ninth batch excluded from pooled statistics"}
        report["persistent_state_unchanged"] = _digest(model.state_dict(), set()) == before
        if not report["persistent_state_unchanged"]:
            raise RuntimeError("Frozen-state observation changed model or score values.")
        report["optimizer_steps"] = 0
        report["main_optimizer_constructed"] = False
        report["bit_optimizer_moments_allocated"] = any(s.exp_avg is not None or s.exp_avg_sq is not None for s in manager.bit_optimizer._state.values())
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if trainer is not None:
            trainer._release_active_teacher_targets()
        if manager is not None:
            manager.detach_runtime()
        report["elapsed_seconds"] = time.perf_counter() - started
        save()

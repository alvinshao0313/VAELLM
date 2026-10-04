"""Isolated paired task-format recovery using the existing L0-down optimizers.

The new archive supplies both formats of the same raw training/heldout records.
Only the six native decoder leaves and optionally two complete bit banks train.
The old formal configuration is reused for optimizer/loss/runtime settings only;
neither its task50 archive nor its prior pure-bit trajectory is reused.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import read_formal_config
from experiments.qwen3_w25.kl_joint_diagnostic import CODE_NAMES, snapshot
from experiments.qwen3_w25.kl_round_diagnostic import TARGET, _sample_state
from experiments.qwen3_w25.teacher_loading_diagnostic import model_inventory, sha256

FORMATS = ("chatml", "bare_space")


def configure_loss_recipe(loss_config, protocol):
    """Select either measured pure KL or the existing response-only CE50 loss."""
    loss_type = protocol["loss_type"]
    prompt_weight = float(protocol["prompt_loss_weight"])
    if (protocol["top_k"] != 100 or protocol["temperature"] != 1
            or loss_config.top_k != 100 or loss_config.temperature != 1
            or loss_config.hidden_loss_weight or loss_config.pre_mlp_hidden_loss_weight
            or loss_config.selective_student_topk):
        raise ValueError("Only the existing nonselective massK100/T1 recipes are supported.")
    if loss_type == "kl_top_mass" and prompt_weight == .3:
        name, alpha = "kl_prompt03", float(loss_config.alpha)
    elif loss_type == "kd_top_mass" and prompt_weight == 0 and protocol.get("alpha") == .5:
        name, alpha = "ce50_prompt0", .5
    else:
        raise ValueError("Expected kl_top_mass/prompt.3 or kd_top_mass/alpha.5/prompt0.")
    loss_config.loss_type = loss_type
    loss_config.alpha = alpha
    loss_config.prompt_loss_weight = prompt_weight
    loss_config.validate()
    return {"name": name, "loss_type": loss_type, "alpha": alpha,
            "alpha_applies": loss_type == "kd_top_mass", "top_k": 100, "temperature": 1.,
            "prompt_loss_weight": prompt_weight,
            "reported_loss": "mass-KL" if name == "kl_prompt03" else "0.5 response CE + 0.5 response mass-KL"}


def loss_denominators(batch, prompt_loss_weight):
    """Use the production causal masks, never the archive's cached .3 weight."""
    from train_utils.distill_loss_core import build_prediction_token_masks

    weight = float(prompt_loss_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Expected a finite nonnegative prompt loss weight.")
    response, prompt = build_prediction_token_masks(labels=batch["labels"], attention_mask=batch["attention_mask"])
    return {"causal_response_tokens": int(response.sum()), "causal_prompt_tokens": int(prompt.sum()),
            "loss_denominator_fp32": float((response + weight * prompt).sum().clamp_min(1)),
            "response_denominator_fp32": float(response.sum().clamp_min(1)), "prompt_loss_weight": weight}


def check_inputs(record):
    if (record["training_format"] not in FORMATS or record["mode"] not in ("decoder", "joint")
            or record["steps"] != 256):
        raise ValueError("Expected a fresh-source B4/256-step paired-format arm.")
    source_id = record["source_checkpoint_id"]
    meta = json.loads((Path(record["source_checkpoint"]) / "checkpoint_meta.json").read_text())
    if not isinstance(source_id, str) or not source_id.strip() or meta["checkpoint_id"] != source_id:
        raise ValueError("Source checkpoint identity differs from the prepared format-recovery record.")
    evidence = {**record["source_sha256"], **record["input_metadata_sha256"], **record["input_evidence_sha256"],
                record["data_archive"]["path"]: record["data_archive"]["sha256"],
                record["formal_reference"]["path"]: record["formal_reference"]["sha256"]}
    for path, digest in evidence.items():
        if sha256(path) != digest:
            raise ValueError(f"Pinned source/data/evidence changed: {path}")
    for item in record["input_stats"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
            raise ValueError(f"Protected input stat changed: {item['path']}")


def answer_ce(logits, batch, manifest):
    """Score only declared content tokens, with exactly one causal shift."""
    if logits.shape[:2] != batch["input_ids"].shape or len(manifest["samples"]) != len(logits):
        raise ValueError("Answer CE batch/manifest geometry differs.")
    result = []
    for index, sample in enumerate(manifest["samples"]):
        positions = sample["answer_positions"]
        if not positions or positions != sorted(set(positions)):
            raise ValueError("Need nonempty ordered, unique answer target positions.")
        pos = torch.tensor(positions, dtype=torch.long, device=logits.device)
        if int(pos.min()) < 1 or int(pos.max()) >= logits.shape[1]:
            raise ValueError("Answer target outside causal input positions.")
        ids = batch["input_ids"][index].to(logits.device)[pos]
        labels = batch["labels"][index].to(logits.device)[pos]
        mask = batch["attention_mask"][index].to(logits.device)[pos]
        if not torch.equal(ids, labels) or not bool(mask.bool().all()):
            raise ValueError("Answer content is not fully included in the real response labels.")
        selected = logits[index, pos-1].float()
        ce = torch.logsumexp(selected, -1) - selected.gather(1, ids[:, None]).squeeze(1)
        if not bool(torch.isfinite(ce).all()):
            raise FloatingPointError("Nonfinite answer-content CE.")
        values = ce.detach().cpu().tolist()
        total = math.fsum(values)
        result.append({"input_and_label_sha256": sample["input_and_label_sha256"],
                       "iterator_sample_ordinal": sample["iterator_sample_ordinal"],
                       "answer_tokens": len(values), "answer_ce_sum": total,
                       "answer_ce_mean": total/len(values), "first_answer_token_ce": values[0]})
    return result


def summarize_heldout(rows):
    if len(rows) != 8 or any(not math.isfinite(row["loss"]) or row["denominator"] <= 0 for row in rows):
        raise ValueError("Expected eight finite heldout B4 batches with real positive denominators.")
    result = {"batches": rows, "equal_batch_loss": statistics.mean(row["loss"] for row in rows),
              "loss_denominator_weighted": math.fsum(row["loss"]*row["denominator"] for row in rows)/math.fsum(row["denominator"] for row in rows)}
    for key in ("student_answer_ce", "teacher_answer_ce"):
        sequences = [sample for row in rows for sample in row[key]]
        if len(sequences) != 32:
            raise ValueError("Heldout answer observations lost a sequence.")
        result[key] = {"sequences": 32, "answer_tokens": sum(s["answer_tokens"] for s in sequences),
                       "token_mean": math.fsum(s["answer_ce_sum"] for s in sequences)/sum(s["answer_tokens"] for s in sequences),
                       "sequence_mean": statistics.mean(s["answer_ce_mean"] for s in sequences),
                       "sequence_mean_nll": statistics.mean(s["answer_ce_sum"] for s in sequences),
                       "sequence_mean_first_token": statistics.mean(s["first_answer_token_ce"] for s in sequences)}
    return result


def heldout_both_formats(trainer, model, archive):
    """Keep the configured trainer loss; observe CE on that same forward only."""
    from e2e_common.dense_loss import get_output_logits

    result = {}
    was_training = model.training
    model.eval()
    try:
        for name in FORMATS:
            rows = []
            data = archive["formats"][name]
            for batch, manifest in zip(data["batches"][256:], data["manifests"][256:]):
                denominators = loss_denominators(batch, trainer.loss_config.prompt_loss_weight)
                prepared = trainer._prepare_inputs(batch)
                teacher_values = []
                def observe_teacher(module, arguments, outputs):
                    del module, arguments
                    teacher_values.append(answer_ce(get_output_logits(outputs), prepared, manifest))
                handle = trainer.teacher_model.register_forward_hook(observe_teacher)
                try:
                    with torch.no_grad(), trainer.compute_loss_context_manager():
                        loss, outputs = trainer.compute_loss(model, prepared, return_outputs=True)
                        student_values = answer_ce(get_output_logits(outputs), prepared, manifest)
                    if len(teacher_values) != 1:
                        raise RuntimeError("Expected exactly one real teacher forward per heldout batch.")
                    rows.append({"archive_batch_index": manifest["batch_index"], "loss": float(loss),
                                 "denominator": denominators["loss_denominator_fp32"], **denominators,
                                 "student_answer_ce": student_values, "teacher_answer_ce": teacher_values[0]})
                    del outputs, loss
                finally:
                    handle.remove()
                    trainer._release_active_teacher_targets()
            result[name] = summarize_heldout(rows)
    finally:
        model.train(was_training)
    return result


def run_format_recovery(launch_record):
    from transformers.trainer_pt_utils import AcceleratorConfig
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _load_teacher, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from experiments.liftquant_recovery.recovery_runtime import prime_packed_cache
    from experiments.qwen3_w25.format_training_data import validate_format_training_archive
    from experiments.qwen3_w25.format_answer_labels import derive_training_labels
    from experiments.qwen3_w25.local_bits import _changed_bits, _digest
    from litebsq.vae_linear import VAELinear
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    launch_path = Path(launch_record).resolve()
    record = json.loads(launch_path.read_text())
    check_inputs(record)
    source_id = record["source_checkpoint_id"]
    output = Path(record["output_dir"]).resolve()
    if output.exists():
        raise ValueError("Use a fresh observation directory.")
    cfg, hf_args, args = read_formal_config(record["formal_reference"]["path"], output)
    loss_recipe = configure_loss_recipe(cfg.loss, record["protocol"])
    if (cfg.opt.resolved_decoder_lr() != 1e-5 or args.max_grad_norm != 1.5 or args.weight_decay != 0
            or args.get_warmup_steps(2000) != 20 or args.full_determinism or args.tf32
            or str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)) != "cosine"):
        raise ValueError("The reused formal optimizer/loss/precision settings changed.")
    source = Path(cfg.student_checkpoint_dir).resolve()
    if source != Path(record["source_checkpoint"]).resolve() or output.is_relative_to(source):
        raise ValueError("Fresh source or isolated output differs.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned GPU must be visible.")
    for key, value in record["environment"].items():
        if value is not None and os.environ.get(key) != value:
            raise ValueError(f"Launch environment differs: {key}")
    archive = torch.load(record["data_archive"]["path"], map_location="cpu", weights_only=True)
    archive_info = validate_format_training_archive(archive)
    selected_format, joint = record["training_format"], record["mode"] == "joint"
    train_data, training_labels = derive_training_labels(
        archive["formats"][selected_format],
        training_label_region=record.get("training_label_region", "response"),
        training_format=selected_format, loss_recipe=loss_recipe)
    if record.get("training_label_region", "response") == "answer" and training_labels != record["training_labels"]:
        raise ValueError("Derived answer-label provenance differs from the prepared configuration.")
    output.mkdir(parents=True)
    started, trainer, manager = time.monotonic(), None, None
    device = torch.device("cuda:0")
    report = {"status": "preparing", "launch_record": str(launch_path), "launch_sha256_at_start": sha256(launch_path),
              "source_checkpoint_id": source_id, "source_dir": str(source), "steps": 256, "joint": joint,
              "training_format": selected_format, "target": TARGET, "worker_pid": os.getpid(),
              "loss_recipe": loss_recipe,
              "training_labels": training_labels,
              "worker_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
              "source_sha256": record["source_sha256"], "input_metadata_sha256": record["input_metadata_sha256"],
              "input_stats": record["input_stats"], "data_archive": {**record["data_archive"], **archive_info},
              "configuration": {"batch_size": 4, "max_length": 1024, "seed": 0, "loss": asdict(cfg.loss),
                  "data_reference": "new paired task-only archive; formal task50 dataset is not consumed",
                  "decoder_lr": 1e-5, "decoder_weight_decay": 0., "decoder_clip": 1.5,
                  "canonical_scheduler_steps": 2000, "warmup_steps": 20, "bit_lr": 4e-5 if joint else None,
                  "bit_optimizer": "adam" if joint else None, "active_ratio": 1. if joint else 0.,
                  "round_steps": 512 if joint else None, "stable_steps": 10 if joint else None,
                  "all_decoder_storage": "FP32", "compute_dtype": "BF16", "tf32": False},
              "loss_denominator_source": "recomputed from actual causal masks and configured prompt weight; archive .3 cache is not used",
              "limitations": ["Local L0-down recovery, not whole-model format training or a downstream quality result.",
                  "bare_space changes wrapping and adds a separator; equal sequence/update budgets do not equal token exposures.",
                  "Loss values across input formats or loss recipes are different objectives; compare matching heldout baselines and actual full8 exports.",
                  "Answer CE includes a separator when fused with an answer token, and excludes independent separator tokens.",
                  "teacher_output_chunk_tokens=8 is retained as an existing argument; the nonselective mass loss actually transfers full logits."],
              "steps_observed": [], "heldout": {}, "round_reset_events": [], "persistent_commit_events": [],
              "first_flip_step": None}
    report["limitations"].append("Training label-region changes supervised tokens and the token-mean denominator; heldout retains original response labels and separate answer CE.")
    def save():
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    save()
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        configure_e2e_determinism(False)
        set_e2e_seed(0)
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
        if meta["checkpoint_id"] != source_id:
            raise ValueError("Unexpected fresh source identity.")
        modules = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, VAELinear)]
        if len(modules) != 252:
            raise ValueError("Expected all252 VAELinear modules.")
        _enable_internal_decode_runtime(modules, decoder_enabled=True, sparse_enabled=True, vae_decoder_checkpoint=True)
        promotion = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=modules,
            rank=0, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="decoder_sparse_bit", freeze=True, fp32_components=("decoder",))
        configure_distill_precision(promotion, components=("decoder",), training_args=args)
        model.requires_grad_(False)
        for _, layer in modules:
            layer.disable_trainable_decode()
        target = model.get_submodule(TARGET)
        selected = [(TARGET, target)]
        selection = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=selected,
            rank=0, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="decoder_sparse_bit" if joint else "trainable_decoder", freeze=True,
            fp32_components=("decoder",))
        configure_distill_precision(selection, components=("decoder",), training_args=args)
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        decoder_names = {TARGET+"."+name for name in target.state_dict() if name.startswith("_parallel_stage_decoder.")}
        code_names = {TARGET+"."+name for name in CODE_NAMES}
        if len(parameters) != 6 or set(parameters) != decoder_names or any(p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError("Only six real FP32 packed decoder leaves may train continuously.")
        initial_decoder = snapshot({name: value for name, value in target.state_dict().items() if name.startswith("_parallel_stage_decoder.")})
        original_codes = snapshot({name: target.state_dict()[name] for name in CODE_NAMES})
        excluded = decoder_names | (code_names if joint else set())
        frozen_before = _digest(model.state_dict(), excluded)
        report["initial_promoted_state_sha256"] = _digest(model.state_dict(), set())
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(args.gradient_checkpointing_kwargs))
        model.to(device)
        target._build_parallel_stage_decode_plan()
        for name, layer in modules:
            if name != TARGET:
                prime_packed_cache(layer)
        cached = [(layer, layer._cached_weight.data_ptr()) for name, layer in modules if name != TARGET]
        if any(layer.trainable_decode or not layer.cache_decoded_weight or layer._cached_weight.dtype != torch.bfloat16 for layer, _ in cached):
            raise RuntimeError("The251 frozen modules do not use native BF16 caches.")
        if joint:
            manager = SparseBitTuningManager(root_model=model, targets=selected, target_devices={TARGET: device},
                training_seed=0, streaming=False, calibration_dtype=torch.bfloat16,
                config=SparseBitTuningConfig(enabled=True, active_ratio=1., optimizer="adam", bit_lr=4e-5,
                                            weight_decay=0., round_steps=512, proxy_coordinates="decoder_sensitivity"))
        tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), hf_args)
        _sync_model_padding_config(model, tokenizer)
        args.output_dir = str(output / "unused_trainer_output")
        args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
        args.remove_unused_columns, args.report_to = False, []
        teacher, identity = _load_teacher(cfg, hf_args, args, model, base_model_path=str(meta["base_model_path"]),
                                         dp_device=device, log=logging.getLogger(__name__))
        teacher_inventory = model_inventory(teacher)
        if (teacher_inventory["requires_grad_parameters"] or teacher_inventory["training"]
                or set(teacher_inventory["parameter_dtype_counts"]) != {"torch.bfloat16"}
                or any(item["inv_freq"]["dtype"] != "torch.float32" for item in teacher_inventory["rope_attributes"].values())):
            raise RuntimeError("Teacher does not preserve the original B-loader BF16 parameters/FP32 RoPE.")
        trainer = VAEDecoderE2ETrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=None,
            eval_dataset=None, loss_config=cfg.loss, teacher_model=teacher, teacher_output_offload="cpu",
            teacher_model_offload="none", teacher_output_pin_memory=True, teacher_output_chunk_tokens=8,
            sparse_bit_manager=manager, callbacks=[])
        attach_model_level_optimizer_contract(trainer, selection=selection,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-5, weight_decay=0., decoder_lr=1e-5))
        trainer.create_optimizer()
        trainer.create_scheduler(num_training_steps=2000)
        model_for_step = trainer.accelerator.prepare_model(model)
        main_optimizer = trainer._sparse_bit_main_optimizer if joint else trainer.optimizer
        if {id(p) for group in main_optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters.values()}:
            raise RuntimeError("Continuous optimizer inventory differs from the six decoder leaves.")
        expected = {id(p) for p in parameters.values()} | (manager.score_module.bit_parameter_ids() if joint else set())
        if {id(p) for p in model.parameters() if p.requires_grad} != expected:
            raise RuntimeError("Unexpected trainable component.")
        phase = {"name": "setup", "step": 0}
        if joint:
            fixed = {s.canonical_key: [manager.sampler_states[s.canonical_key].logical_index(i) for i in range(128)] for s in manager.bank_specs}
            report["bit_registration"] = {"active_scores": sum(s.n_active for s in manager.bank_specs),
                                           "initial_sample": _sample_state(manager, fixed)}
            if report["bit_registration"]["active_scores"] != 147849216 or manager.bit_round_steps != 512 or manager.stable_steps != 10:
                raise ValueError("Expected both complete banks and the existing R512/stable10 schedule.")
            original_commit = manager.commit_bank_to_persistent
            def observe_commit(spec):
                original_commit(spec)
                report["persistent_commit_events"].append({"step": phase["step"], "phase": phase["name"],
                                                            "bank": spec.canonical_key, "round": manager.global_bit_round})
            manager.commit_bank_to_persistent = observe_commit
            finish = manager._finish_nonstreaming_round
            def observe_reset():
                event = {"step": phase["step"], "before": _sample_state(manager, fixed)}
                finish()
                event["after"] = _sample_state(manager, fixed)
                report["round_reset_events"].append(event)
            manager._finish_nonstreaming_round = observe_reset
        frozen_count = len(model.state_dict()) - len(excluded) - (len(manager.score_module.score_chunks) if joint else 0)
        if frozen_count != (3415 if joint else 3417):
            raise RuntimeError("Unexpected frozen-state inventory.")
        report["runtime"] = {"teacher_identity": identity, "teacher_B_inventory": teacher_inventory,
            "teacher_path": "runtime_v6._load_teacher -> dtype=None reference loader; no explicit floating-buffer cast",
            "mixed_precision": trainer.accelerator.mixed_precision, "promoted_decoder_tensors": len(promotion.decoder_parameters),
            "trainable_decoder_tensors": len(parameters), "trainable_decoder_elements": sum(p.numel() for p in parameters.values()),
            "frozen_native_cache_modules": len(cached), "frozen_tensor_count": frozen_count,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "full_determinism": args.full_determinism}
        report["heldout"]["0"] = heldout_both_formats(trainer, model_for_step, archive)
        report["setup_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["setup_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["setup_seconds"], report["status"] = time.monotonic()-started, "running"
        save()
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(256):
            phase.update(name="optimizer_step", step=index+1)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            loss = trainer.training_step(model_for_step, train_data["batches"][index])
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters.values()):
                raise FloatingPointError("Nonfinite loss/decoder gradient or disconnected decoder.")
            if joint:
                manager.bit_optimizer.validate_gradients()
                if any(not bool(torch.isfinite(p.grad).all()) for p in manager.score_module.score_chunks):
                    raise FloatingPointError("Nonfinite score gradient.")
            norm = trainer.accelerator.clip_grad_norm_(parameters.values(), 1.5)
            lrs = [group["lr"] for group in main_optimizer.param_groups]
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.state.global_step = index+1
            torch.cuda.synchronize(device)
            manifest = train_data["manifests"][index]
            row = {"step": index+1, "batch_manifest_index": index, "loss": float(loss), "seconds": time.monotonic()-tick,
                   "decoder_gradient_norm_before_clipping": float(norm), "decoder_learning_rates_used": lrs,
                   **loss_denominators(train_data["batches"][index], cfg.loss.prompt_loss_weight),
                   "answer_tokens": sum(len(sample["answer_positions"]) for sample in manifest["samples"]),
                   "input_tokens": sum(sample["input_tokens"] for sample in manifest["samples"])}
            if joint:
                row["telemetry"] = dict(trainer._last_sparse_bit_telemetry)
                if report["first_flip_step"] is None and row["telemetry"]["bit/step_flip_count"] > 0:
                    report["first_flip_step"] = index+1
                if index+1 in (1, 2, 8, 127, 128, 129, 256):
                    row["score_sample_after"] = _sample_state(manager, fixed)
            report["steps_observed"].append(row)
            print(json.dumps(row), flush=True)
            save()
            if index+1 in (128, 256):
                phase.update(name="heldout_commit", step=index+1)
                if joint:
                    manager.final_commit()
                report["heldout"][str(index+1)] = heldout_both_formats(trainer, model_for_step, archive)
                save()
        phase.update(name="terminal_commit", step=256)
        if joint:
            packed = manager.checkpoint_packed_snapshot()
            manager.final_commit()
            for spec in manager.bank_specs:
                if not torch.equal(target.get_stage_part_vq_storage(spec.stage_idx, spec.part_idx).detach().cpu(), packed[spec.canonical_key]):
                    raise RuntimeError("Final persistent codes differ from live grouped codes.")
            manager.detach_runtime()
        final_decoder = snapshot({name: value for name, value in target.state_dict().items() if name.startswith("_parallel_stage_decoder.")})
        final_codes = snapshot({name: target.state_dict()[name] for name in CODE_NAMES})
        report["decoder_changed_elements"] = {name: int((final_decoder[name] != value).sum()) for name, value in initial_decoder.items()}
        if not any(report["decoder_changed_elements"].values()):
            raise RuntimeError("No genuine decoder update.")
        report["decoder_moment_dtypes"] = {name: {key: str(main_optimizer.state[p][key].dtype) for key in ("exp_avg", "exp_avg_sq")}
                                            for name, p in parameters.items()}
        if any(dtype != "torch.float32" for state in report["decoder_moment_dtypes"].values() for dtype in state.values()):
            raise RuntimeError("Decoder Adam moments are not FP32.")
        if joint:
            report["cumulative_flip_events_all_steps"] = sum(row["telemetry"]["bit/step_flip_count"] for row in report["steps_observed"])
        report["first_persistent_commit"] = report["persistent_commit_events"][0] if report["persistent_commit_events"] else None
        report["net_xor_by_stage"] = [_changed_bits(original_codes[name], final_codes[name], tuple(target.get_stage_part_vq_spec(stage, 0)["logical_shape"]))
                                     for stage, name in enumerate(CODE_NAMES)]
        if not joint and any(report["net_xor_by_stage"]):
            raise RuntimeError("Decoder-only arm modified persistent codes.")
        frozen_after = _digest(model.state_dict(), excluded)
        report["frozen_state"] = {"before": frozen_before, "after": frozen_after, "unchanged": frozen_before == frozen_after,
                                   "excluded_keys": sorted(excluded), "reference": "after all252 FP32 decoder promotion; original persistent codes"}
        report["frozen_cache_pointers_unchanged"] = all(layer._cached_weight is not None and layer._cached_weight.data_ptr() == pointer for layer, pointer in cached)
        if frozen_before != frozen_after or not report["frozen_cache_pointers_unchanged"]:
            raise RuntimeError("Frozen persistent state or native cache changed.")
        if teacher_inventory != model_inventory(teacher):
            raise RuntimeError("Frozen teacher inventory/buffers changed.")
        payload = {"format": "qwen3_kl_joint_bridge_v1", "source_checkpoint_id": source_id, "joint": joint,
                   "steps": 256, "target": TARGET, "training_format": selected_format,
                   "loss_recipe": loss_recipe,
                   "training_labels": training_labels,
                   "data_archive_sha256": record["data_archive"]["sha256"], "decoder_fp32": final_decoder, "codes": final_codes}
        payload_path = output / "selected_state.pt"
        torch.save(payload, payload_path)
        loaded = torch.load(payload_path, map_location="cpu", weights_only=True)
        if any(not torch.equal(value, loaded[part][name]) for part in ("decoder_fp32", "codes") for name, value in payload[part].items()):
            raise RuntimeError("Saved endpoint payload reload differs.")
        report["selected_payload"] = {"path": str(payload_path), "sha256": sha256(payload_path), "reloaded_exact": True, "is_full_checkpoint": False}
        report["peak_cuda_allocated_bytes"], report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
        report["overall_peak_cuda_allocated_bytes"] = max(report["setup_peak_cuda_allocated_bytes"], report["peak_cuda_allocated_bytes"])
        report["overall_peak_cuda_reserved_bytes"] = max(report["setup_peak_cuda_reserved_bytes"], report["peak_cuda_reserved_bytes"])
        report["timing"] = {"steady_median_seconds": statistics.median(row["seconds"] for row in report["steps_observed"][1:]),
                             "steady_max_seconds": max(row["seconds"] for row in report["steps_observed"][1:])}
        check_inputs(record)
        report.update(source_pins_data_unchanged=True, status="complete")
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            trainer._release_active_teacher_targets()
        if manager is not None:
            manager.detach_runtime()
        report["elapsed_seconds"] = time.monotonic()-started
        save()
    return report

"""Observe genuine native training without changing its data, loss or updates."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from experiments.long_context_recovery.contract import require, validate_training_args
from experiments.w25_eval.subset_state import persistent_inventory


def batch_record(inputs, *, special_ids):
    ids, labels, attention = (inputs[key] for key in ("input_ids", "labels", "attention_mask"))
    require(ids.ndim == 2 and labels.shape == attention.shape == ids.shape,
            "Native SFT input/label/mask shapes disagree.")
    valid = attention[:, 1:].bool()
    response = labels[:, 1:].ne(-100) & valid
    prompt = labels[:, 1:].eq(-100) & valid
    ordinary = response.clone()
    for token in special_ids:
        ordinary &= ids[:, 1:].ne(token)
    return {"shape": list(ids.shape), "effective_tokens": int(attention.sum()),
            "response_tokens_shifted": int(response.sum()), "prompt_tokens_shifted": int(prompt.sum()),
            "response_non_special_tokens_shifted": int(ordinary.sum()),
            "input_sha256": hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def model_policy(model):
    buffers = {name: str(value.dtype) for name, value in model.named_buffers() if "inv_freq" in name}
    require(bool(buffers) and set(buffers.values()) == {"torch.float32"},
            "Long-context teacher/student must retain their original FP32 RoPE inverse frequencies.")
    require(model.config.use_cache is False and model.config.model_type == "qwen3"
            and model.config.rope_scaling is None
            and model.config._attn_implementation == "flash_attention_2",
            "Keep original Qwen3 FA2/unscaled RoPE with training KV disabled.")
    return {"inv_freq_dtypes": buffers, "use_cache": model.config.use_cache,
            "rope_scaling": model.config.rope_scaling, "attn_implementation": model.config._attn_implementation}


class LongContextObservations(TrainerCallback):
    def __init__(self, names, architecture, output):
        self.names, self.architecture = tuple(names), architecture
        self.output = Path(output)
        self.trainer = None
        self.handles = []
        self.step = None
        self.active = {}
        self.before = {}
        self.layer_calls = Counter()
        self.report = {"diagnostic_not_quality": True, "batches": [], "steps": [], "saves": [],
                       "teacher_forwards": [], "student_forwards": []}

    def flush(self):
        from experiments.longbench_v1.common import write_json
        write_json(self.output / "observations.json", self.report)

    def bind_trainer(self, trainer):
        require(self.trainer is None, "Only one genuine native trainer is observed.")
        self.trainer = trainer
        require(trainer.teacher_model is not None, "The probe requires the original BF16 teacher.")
        self.report["teacher_policy"] = model_policy(trainer.teacher_model)

    def observe_batch(self, inputs):
        record = batch_record(inputs, special_ids=self.trainer.tokenizer.all_special_ids)
        require(record["shape"] == [1, 8192] and record["effective_tokens"] == 8192,
                "Resource validation requires a real B1 x8192 sequence; padding is not coverage.")
        require(record["response_non_special_tokens_shifted"] > 0,
                "The real truncated sequence must retain supervised assistant content.")
        record["step"] = self.step
        self.report["batches"].append(record)
        self.flush()

    def _observe_forward(self, kind, model, _args, _kwargs, outputs):
        require(model.config.use_cache is False and getattr(outputs, "past_key_values", None) is None,
                "A native training forward unexpectedly retained KV cache.")
        policy = model_policy(model)
        logits = outputs.logits
        require(tuple(logits.shape) == (1, 8192, model.config.vocab_size),
                "The native full-vocabulary forward changed batch, sequence or vocabulary geometry.")
        self.report[kind + "_forwards"].append({"step": self.step, "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype), "logits_bytes": logits.numel() * logits.element_size(),
            "use_cache": policy["use_cache"], "inv_freq_dtypes": policy["inv_freq_dtypes"],
            "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()})

    def on_train_begin(self, args, state, control, **kwargs):
        validate_training_args(args)
        require(state.global_step == 0 and torch.cuda.device_count() == 1, "Use one assigned GPU and a fresh start.")
        model = kwargs["model"]
        self.report["student_policy"] = model_policy(model)
        require(model.is_gradient_checkpointing, "HF gradient checkpointing did not activate in the real trainer.")
        inventory, banks = persistent_inventory(model, self.architecture)
        expected = {key for name in self.names for key in inventory[name]}
        self.active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        require(set(self.active) == expected and len(expected) == 432
                and all(value.dtype == torch.float32 for value in self.active.values())
                and not hasattr(model, "sparse_bit_tuning"), "Only the actual OV72 FP32 decoder leaves may train.")
        optimizer = kwargs["optimizer"]
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        require({id(p) for group in optimizer.param_groups for p in group["params"]}
                == {id(p) for p in self.active.values()}, "The optimizer includes unexpected parameters.")
        modules = dict(model.named_modules())
        require(all(modules[name]._parallel_stage_decoder.use_checkpoint for name in self.names),
                "Selected native decoder activation checkpointing is disabled.")
        self.report["inventory"] = {"selected_projections": len(self.names), "decoder_tensors": len(expected),
            "decoder_elements": sum(value.numel() for value in self.active.values()),
            "all_code_banks": sum(map(len, banks.values())), "active_score_elements": 0,
            "hf_gradient_checkpointing": True, "decoder_gradient_checkpointing": True}
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda _m, _a, index=index: self.layer_calls.update([index])))
        self.handles.append(model.register_forward_hook(
            lambda m, a, kw, out: self._observe_forward("student", m, a, kw, out), with_kwargs=True))
        self.handles.append(self.trainer.teacher_model.register_forward_hook(
            lambda m, a, kw, out: self._observe_forward("teacher", m, a, kw, out), with_kwargs=True))
        self.flush()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = int(state.global_step) + 1
        self.layer_calls.clear()
        self.before = {name: value.detach().cpu().clone() for name, value in self.active.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.current = {"step": self.step, "start_allocated_bytes": torch.cuda.memory_allocated(),
                        "start_reserved_bytes": torch.cuda.memory_reserved()}
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grads = [value.grad for value in self.active.values()]
        require(all(g is not None and g.dtype == torch.float32 for g in grads),
                "A selected FP32 decoder is missing its real gradient.")
        finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
        require(finite and math.isfinite(norm) and norm > 0, "Nonfinite or all-zero decoder gradients.")
        self.current.update(gradient_norm_after_native_clipping=norm, gradient_tensors=len(grads), gradients_finite=finite)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        optimizer = kwargs["optimizer"]
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        moments = sorted({str(value.dtype) for item in optimizer.state.values() for key, value in item.items()
                          if key in ("exp_avg", "exp_avg_sq")})
        changed = sum(int(torch.count_nonzero(value.detach().cpu() != self.before[name]))
                      for name, value in self.active.items())
        require(moments == ["torch.float32"], "Decoder optimizer moments are not FP32.")
        require(len(self.layer_calls) == 36 and min(self.layer_calls.values()) >= 2,
                "The actual backward did not cover every checkpointed Transformer layer.")
        self.current.update(changed_decoder_elements=changed, main_moment_dtypes=moments,
            layer_forward_calls=dict(self.layer_calls), elapsed_seconds=time.perf_counter() - self.started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            end_allocated_bytes=torch.cuda.memory_allocated(), end_reserved_bytes=torch.cuda.memory_reserved(),
            learning_rates_after_scheduler=[group["lr"] for group in optimizer.param_groups])
        self.report["steps"].append(self.current)
        self.before.clear()
        if state.global_step == 4:
            control.should_save = True
            control.should_training_stop = True
        self.flush()
        return control

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        meta = json.loads((path / "checkpoint_meta.json").read_text())
        require(state.global_step in (2, 4) and meta["train_mode"] == "decoder"
                and not (path / "sparse_bit_tuning").exists(), "Unexpected native save boundary or bit state.")
        self.report["saves"].append({"step": state.global_step, "checkpoint_dir": str(path),
                                     "checkpoint_id": meta["checkpoint_id"]})
        self.flush()
        return control

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def on_train_end(self, args, state, control, **kwargs):
        self.close()
        require([item["step"] for item in self.report["steps"]] == [1, 2, 3, 4]
                and [item["step"] for item in self.report["saves"]] == [2, 4]
                and len(self.report["batches"]) == 4
                and len(self.report["teacher_forwards"]) == len(self.report["student_forwards"]) == 4,
                "Four actual batches/updates, native saves and teacher/student forwards were not all observed.")
        require(sum(item["changed_decoder_elements"] for item in self.report["steps"]) > 0,
                "The native warmup schedule never changed any selected decoder.")
        self.flush()
        return control


@contextmanager
def observe_native_trainer(observation):
    """Install observers only in this interpreter; call the original trainer unchanged."""
    from compressed_e2e_fintuning import runtime_v6_pipeline as pipeline
    original = pipeline.VAEDecoderE2ETrainer

    class ObservedNativeTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observation.bind_trainer(self)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            observation.observe_batch(inputs)
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    pipeline.VAEDecoderE2ETrainer = ObservedNativeTrainer
    try:
        yield
    finally:
        pipeline.VAEDecoderE2ETrainer = original
        observation.close()

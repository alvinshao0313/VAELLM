"""Fixed-code decoder fitting to the actual BF16 gate/up/down output."""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.full_weight_bits_probe.core import popcounts
from experiments.joint_reconstruction_probe.core import (EPS, LR, SEED, decode, error_stats,
    require, restore, schema, stage_norm, storage_bits)
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import aggregate, squared_error
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import TwoSidedHadamard, rotation_spec


TOTAL_STEPS = 500
TARGET = "model.layers.18.mlp.gate_proj"


def records(module):
    return {name: tensor_record(value) for name, value in module.state_dict().items()}


class GateObjective(nn.Module):
    """Frozen original-model signals and a differentiable protected restoration."""
    def __init__(self, archive, target_name, device):
        super().__init__()
        require(target_name == TARGET and archive["layer_index"] == 18,
                "This isolated comparison is the preselected Qwen L18 gate only.")
        bundle = archive["targets"][target_name]
        self.lengths = list(archive["document_lengths"]["train"])
        x, gate = bundle["train_inputs"], bundle["reference_weight"]
        up = archive["up_proj_weight"]
        down = archive["targets"][target_name.replace("gate_proj", "down_proj")]["reference_weight"]
        indices = bundle["protected_input_indices"]
        require(len(self.lengths) == 32 and all(type(n) is int and n > 0 for n in self.lengths)
                and sum(self.lengths) == len(x), "Use all valid tokens of the old 32 training documents.")
        require(gate.shape == up.shape == (12288, 4096) and down.shape == (4096, 12288)
                and x.ndim == 2 and x.shape[1] == 4096
                and all(t.dtype == torch.bfloat16 for t in (x, gate, up, down)),
                "Original HF weights and captured activations must retain their native BF16 geometry.")
        require(indices.dtype == torch.int64 and tuple(indices.shape) == (64,)
                and indices.unique().numel() == 64 and int(indices.min()) >= 0 and int(indices.max()) < 4096,
                "Keep the original 64 unique INT8 input-protection columns.")
        keep = torch.ones(4096, dtype=torch.bool)
        keep[indices] = False
        self.register_buffer("keep_indices", keep.nonzero().flatten().to(device))
        self.register_buffer("protected_indices", indices.detach().clone().to(device))
        self.register_buffer("protected_qvalues", bundle["protected_input_qvalues"].detach().clone())
        self.register_buffer("protected_scales", bundle["protected_input_scales"].detach().clone())
        self.rotation = TwoSidedHadamard(12288, 4032,
            rotation_spec("two_sided", 32, SEED, target_name)).to(device)
        self.register_buffer("inputs", x.detach().to(device))
        self.register_buffer("original_gate", gate.detach().to(device))
        self.register_buffer("original_up", up.detach().to(device))
        self.register_buffer("original_down", down.detach().to(device))
        protected = decode_protected_channel_weight(self.protected_qvalues, self.protected_scales,
            quant_format="int8", dtype=torch.bfloat16, device=device).T
        template = torch.zeros_like(self.original_gate)
        template[:, self.protected_indices] = protected
        self.register_buffer("protected_template", template)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            up_output = F.linear(self.inputs, self.original_up)
            reference = F.linear(F.silu(F.linear(self.inputs, self.original_gate)) * up_output,
                                 self.original_down)
        self.register_buffer("up_output", up_output)
        self.register_buffer("reference", reference)
        self.normalizer = float(reference.double().square().sum()) / reference.numel()
        require(self.normalizer > 0 and math.isfinite(self.normalizer)
                and not list(self.parameters())
                and not any(value.requires_grad for value in self.buffers()),
                "HF targets, up/down, rotation, protection and inputs must be fixed finite signals.")

    def restore_weight(self, rotated):
        require(rotated.dtype == torch.bfloat16 and tuple(rotated.shape) == (12288, 4032),
                "The decoder must produce the native BF16 unprotected matrix.")
        # The production inverse computes internally in FP32 and returns BF16.
        # index_copy propagates only the unprotected columns back to the decoder.
        unrotated = self.rotation(rotated, inverse=True)
        return self.protected_template.index_copy(1, self.keep_indices, unrotated)

    def prediction(self, weight):
        return F.linear(F.silu(F.linear(self.inputs, weight)) * self.up_output, self.original_down)

    def forward(self, rotated):
        prediction = self.prediction(self.restore_weight(rotated))
        raw_mse = F.mse_loss(prediction.float(), self.reference.float())
        return raw_mse / self.normalizer, raw_mse

    @torch.no_grad()
    def measure(self, rotated):
        with torch.autocast(device_type=self.inputs.device.type, enabled=False):
            prediction = self.prediction(self.restore_weight(rotated))
            raw_mse = F.mse_loss(prediction.float(), self.reference.float())
            loss = raw_mse / self.normalizer
        error, signal = squared_error(prediction, self.reference)
        return {"raw_mse": float(raw_mse), "nmse_loss": float(loss),
                "nmse_fp64_sums": error / signal, "squared_error": error,
                "squared_reference": signal, "tokens": len(self.inputs)}

    @torch.no_grad()
    def verify_step0(self, rotated, bundle):
        device = self.inputs.device
        with torch.autocast(device_type=device.type, enabled=False):
            weight = self.restore_weight(rotated)
            native = restore(bundle, TARGET, rotated, device).to(device)
            require(torch.equal(weight, native), "Differentiable restoration differs from the original BF16 restore.")
            cached = self.prediction(weight)
            direct = F.linear(F.silu(F.linear(self.inputs, native))
                              * F.linear(self.inputs, self.original_up), self.original_down)
            cached_loss = F.mse_loss(cached.float(), self.reference.float()) / self.normalizer
            direct_loss = F.mse_loss(direct.float(), self.reference.float()) / self.normalizer
            require(bool(torch.isfinite(cached_loss)) and bool(torch.isfinite(direct_loss)),
                    "Step0 comparison has a nonfinite objective.")
            require(torch.equal(cached, direct) and torch.equal(cached_loss, direct_loss),
                    "Caching the fixed up output changed the same-shape BF16 prediction or loss.")
            # Different GEMM row counts can produce normal BF16 rounding differences.
            # Keep their measured size; do not demand full-batch/document exactness.
            documents, dense_parts, candidate_parts, offset = [], [], [], 0
            for index, length in enumerate(self.lengths):
                x = self.inputs[offset:offset + length]
                offset += length
                u = F.linear(x, self.original_up)
                dense = F.linear(F.silu(F.linear(x, self.original_gate)) * u, self.original_down)
                candidate = F.linear(F.silu(F.linear(x, weight)) * u, self.original_down)
                error, signal = squared_error(candidate, dense)
                require(signal > 0 and math.isfinite(error), "Invalid per-document reference energy.")
                documents.append({"index": index, "tokens": length, "squared_error": error,
                                  "squared_reference": signal, "nmse": error / signal})
                dense_parts.append(dense)
                candidate_parts.append(candidate)
            document_dense, document_candidate = torch.cat(dense_parts), torch.cat(candidate_parts)
        document_metrics = aggregate(documents)
        whole_metrics = self.measure(rotated)
        return {"native_restore_exact": True,
            "cached_vs_direct_prediction": error_stats(cached, direct),
            "cached_vs_direct_prediction_exact": torch.equal(cached, direct),
            "cached_nmse_loss": float(cached_loss), "direct_nmse_loss": float(direct_loss),
            "cached_minus_direct_loss": float(cached_loss - direct_loss),
            "whole_batch_dense_vs_document_dense": error_stats(self.reference, document_dense),
            "whole_batch_candidate_vs_document_candidate": error_stats(cached, document_candidate),
            "whole_batch_metrics": whole_metrics, "per_document_metrics": document_metrics,
            "whole_minus_document_nmse": whole_metrics["nmse_fp64_sums"] - document_metrics["nmse"],
            "interpretation": "Same input tokens and BF16 function; whole-batch/document GEMM differences are recorded, not silently treated as exact."}


def train(layer, archive, target_name, *, steps, deadline, observer):
    require(steps in (2, TOTAL_STEPS), "Only two mechanism updates or the fixed 500-step comparison are allowed.")
    device = next(layer.parameters()).device
    require(device.type == "cuda" and not layer.transpose and layer.residual_stages == 2
            and layer._parallel_stage_layout == [(0, 0), (1, 0)]
            and getattr(layer, "_sparse_bit_binding", None) is None,
            "Require the native CUDA two-stage gate with fixed codes and no bit-update runtime.")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    original = {name: value.detach().cpu().clone() for name, value in layer.state_dict().items()}
    before_schema, before_bits = schema(layer), storage_bits(layer)
    initial = {name: value.detach().clone() for name, value in layer._parallel_stage_decoder.named_parameters()}
    require(initial and all(value.dtype == torch.bfloat16 for value in initial.values()),
            "Both arms must begin from the common native2k BF16 decoder, not a fitted endpoint.")
    objective = GateObjective(archive, target_name, device).eval()
    fixed_before = records(objective)
    native = decode(layer, deployment=True)
    report = {"before_deployment": objective.measure(native)}
    del native
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    layer._decoder_compute_dtype = torch.bfloat16
    layer.training = True
    layer._parallel_stage_decoder.eval()
    layer._build_parallel_stage_decode_plan()
    parameters = dict(layer._parallel_stage_decoder.named_parameters())
    identities = {name: id(value) for name, value in parameters.items()}
    require(all(p.requires_grad and p.dtype == torch.float32 for p in parameters.values()),
            "Both packed decoder stages require trainable FP32 masters.")
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        before = layer._decode_weight(dtype=torch.bfloat16)
    report["step0_verification"] = objective.verify_step0(before, archive["targets"][target_name])
    report["before_train_decode"] = objective.measure(before)
    del before
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    telemetry, gradient_max = [], [0.0, 0.0]
    for step in range(1, steps + 1):
        if time.time() >= deadline:
            raise TimeoutError("The assigned output-fitting deadline expired; no automatic retry.")
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = LR * (1.0 - (step - 1) / TOTAL_STEPS)
        with torch.autocast("cuda", enabled=False):
            rotated = layer._decode_weight(dtype=torch.bfloat16)
            loss, raw_mse = objective(rotated)
        require(bool(torch.isfinite(loss)), f"Nonfinite full-training MLP loss at step {step}.")
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32
                    and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                "A decoder gradient is missing, nonfinite or outside FP32.")
        norms = stage_norm([p.grad for p in parameters.values()])
        require(all(value > 0 for value in norms), "Each decoder stage needs a real gradient through restored MLP output.")
        gradient_max = [max(a, b) for a, b in zip(gradient_max, norms, strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        item = {"step": step, "raw_mse": float(raw_mse), "nmse_loss": float(loss),
            "decoder_lr": optimizer.param_groups[0]["lr"], "stage_decoder_gradient_l2": norms,
            "step_seconds": time.perf_counter() - step_started}
        telemetry.append(item)
        if step <= 2 or step % 50 == 0:
            observer(item)
        del rotated, loss, raw_mse
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        live = layer._decode_weight(dtype=torch.bfloat16)
    report["after_live_train_decode"] = objective.measure(live)
    moments = sorted({str(state[key].dtype) for state in optimizer.state.values()
                      for key in ("exp_avg", "exp_avg_sq")})
    require(moments == ["torch.float32"], "Adam moments must remain FP32.")
    layer._build_parallel_stage_decode_plan()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        rebuilt = layer._decode_weight(dtype=torch.bfloat16)
    require(torch.equal(live, rebuilt), "Rebuilding the fixed packed plan changed the live decoder output.")
    del live, rebuilt, optimizer
    fp32_delta = stage_norm([p.detach() - initial[name].float() for name, p in parameters.items()])
    for name, parameter in parameters.items():
        parameter.grad = None
        parameter.data = parameter.data.to(initial[name].dtype)
    layer.eval().disable_trainable_decode()
    layer._build_parallel_stage_decode_plan()
    deployed = decode(layer, deployment=True)
    report["after_deployment"] = objective.measure(deployed)
    bf16_delta = stage_norm([p.detach().float() - initial[name].float() for name, p in parameters.items()])
    final = layer.state_dict()
    bank_names = ("vq_weight", "vq_weight_s1")
    changed = [popcounts(original[name].to(device), final[name]) for name in bank_names]
    require(changed == [0, 0] and schema(layer) == before_schema and storage_bits(layer) == before_bits,
            "Fixed codes, persistent topology or storage size changed.")
    require(all(torch.equal(value, final[name].cpu()) for name, value in original.items()
                if not name.startswith("_parallel_stage_decoder.")), "A nondecoder persistent tensor changed.")
    require(identities == {name: id(value) for name, value in layer._parallel_stage_decoder.named_parameters()}
            and all(value > 0 for value in fp32_delta + bf16_delta),
            "Both registered stages must retain real FP32 and BF16 updates.")
    require(records(objective) == fixed_before, "Original HF signals, inputs, rotation or protection changed.")
    report.update(status="completed", steps=steps, schedule_steps=TOTAL_STEPS, joint=False,
        trajectory=telemetry, stage_final_net_xor=changed, stage_decoder_gradient_l2_max=gradient_max,
        stage_decoder_delta_l2_fp32=fp32_delta, stage_decoder_delta_l2_bf16=bf16_delta,
        decoder_moment_dtypes=moments, frozen_state_unchanged=True, persistent_schema_unchanged=True,
        frozen_objective_buffers_unchanged=True, objective_buffer_inventory=fixed_before,
        final_commit_live_decode_exact=True, persistent_bits_before=before_bits, persistent_bits_after=storage_bits(layer),
        initial_payload={name: tensor_record(value) for name, value in original.items()}, final_payload=records(layer),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        elapsed_seconds=time.perf_counter() - started,
        settings={"target": "Original HF BF16 gate/SiLU/up/down output on all old train32 valid tokens",
            "loss": "Full-batch output MSE / fixed original output mean-square", "normalizer": objective.normalizer,
            "train_tokens": len(objective.inputs), "train_document_lengths": objective.lengths,
            "decoder_lr": LR, "decoder_betas": [.9, .95], "decoder_eps": EPS, "decoder_weight_decay": 0,
            "decoder_schedule": "500-step linear, warmup0, including the two-update mechanism probe",
            "decoder_master": "float32", "compute_and_stage_sum": "bfloat16",
            "restoration": "Native block32 inverse: internal FP32, returned BF16; original INT8 columns fixed",
            "sampling": "Identical complete train32 tokens every update; no text minibatch sampling",
            "heldout_use": "None in fitting or step selection; caller evaluates fixed terminal endpoints only",
            "mechanism_endpoint_is_formal_initialization": False,
            "scope": "Single Qwen L18 diagnostic; historical full36 local-output gains did not establish downstream gains."})
    return deployed.cpu(), report

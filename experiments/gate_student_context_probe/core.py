"""Finite weight-neighbor search and fixed-code fitting in the real student MLP."""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.exact_bit_neighbor_probe.core import CHUNK, block_sse, flip_grouped, packed_decode
from experiments.full_weight_bits_probe.core import popcounts
from experiments.joint_reconstruction_probe.core import (EPS, LR, error_stats, require,
    schema, stage_norm, storage_bits)
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import aggregate, squared_error


NAME = "model.layers.18.mlp.gate_proj"
STEPS = 500
BANKS = ("vq_weight", "vq_weight_s1")


def records(module):
    return {name: tensor_record(value) for name, value in module.state_dict().items()}


def ready(layer):
    require(not layer.transpose and layer.residual_stages == 2
            and layer._parallel_stage_layout == [(0, 0), (1, 0)]
            and layer.in_features == 4096 and layer.out_features == 12288
            and layer.compressed_in_features == 4032 and layer.weight_rotation is not None
            and getattr(layer, "_sparse_bit_binding", None) is None,
            "Require the actual two-stage f927 L18 gate including rotation and protection.")
    layer._decoder_compute_dtype = torch.bfloat16
    layer.clear_decoded_weight_cache()
    layer._build_parallel_stage_decode_plan()


@torch.no_grad()
def search_once(layer, archive, *, deadline, observer):
    """Current code plus all 64 single-bit neighbors, scored only in weight space."""
    ready(layer)
    require(not layer.training and all(not p.requires_grad for p in layer.parameters()),
            "Search must not train a decoder or attach score parameters.")
    device = next(layer.parameters()).device
    started = time.perf_counter()
    original, before_schema = records(layer), schema(layer)
    before_bits = storage_bits(layer)
    original_banks = [layer.get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)]
    hf = archive["targets"][NAME]["reference_weight"].to(device).float()
    keep = torch.ones(4096, dtype=torch.bool, device=device)
    keep[layer.protected_input_indices] = False
    target = layer.weight_rotation(hf[:, keep].contiguous(), inverse=False)
    require(target.dtype == torch.float32 and target.shape == (12288, 4032),
            "The original HF weight target must use the current rotation and protection coordinates.")
    target_record = tensor_record(target)
    grouped = layer._get_parallel_stage_grouped_vq_packed(device=device)
    blocks = target.reshape(-1, 32)
    require(tuple(grouped.shape) == (len(blocks), 2, 4), "Unexpected native grouped bank order.")
    before = layer._decode_compressed_weight(dtype=torch.bfloat16)
    baseline_metrics = error_stats(before, target)
    del before, hf
    selected = grouped.clone()
    choices = torch.zeros(len(blocks), dtype=torch.uint8, device=device)
    trajectory = []
    for start in range(0, len(blocks), CHUNK):
        require(time.time() < deadline, "The assigned search deadline expired; no retry.")
        end = min(start + CHUNK, len(blocks))
        packed, ref = grouped[start:end], blocks[start:end]
        baseline = packed_decode(layer, packed)
        initial_sse = block_sse(baseline, ref)
        best_sse, best_output = initial_sse.clone(), baseline.clone()
        choice = torch.zeros(end - start, dtype=torch.uint8, device=device)
        for candidate in range(64):
            value = packed_decode(layer, flip_grouped(packed, candidate))
            sse = block_sse(value, ref)
            require(bool(torch.isfinite(sse).all()), "Nonfinite finite-neighbor score.")
            improved = sse < best_sse
            choice = torch.where(improved, candidate + 1, choice)
            best_sse = torch.where(improved, sse, best_sse)
            best_output = torch.where(improved[:, None], value, best_output)
        rows = torch.arange(end - start, device=device)[choice > 0]
        bit = choice[choice > 0].long() - 1
        selected[start:end][rows, bit // 32, (bit % 32) // 8] ^= (1 << (bit % 8)).to(torch.uint8)
        require(torch.equal(packed_decode(layer, selected[start:end]), best_output)
                and bool((best_sse <= initial_sse).all()), "Committed choices do not match the scored native candidates.")
        choices[start:end] = choice
        row = {"start": start, "end": end, "before_sse": float(initial_sse.sum()),
               "after_sse": float(best_sse.sum()), "changed_blocks": int((choice > 0).sum())}
        trajectory.append(row)
        if start == 0 or end == len(blocks) or len(trajectory) % 10 == 0:
            observer(row)
    for stage in range(2):
        layer.get_stage_part_vq_storage(stage, 0).copy_(selected[:, stage:stage + 1])
    ready(layer)
    after = layer._decode_compressed_weight(dtype=torch.bfloat16)
    xor = [popcounts(old, layer.get_stage_part_vq_storage(i, 0)) for i, old in enumerate(original_banks)]
    final = records(layer)
    require(sum(xor) == int((choices > 0).sum())
            and all(final[k] == v for k, v in original.items() if k not in BANKS)
            and schema(layer) == before_schema and storage_bits(layer) == before_bits,
            "Search changed more than one bit per selected vector, or changed frozen persistent state.")
    report = {"status": "completed", "passes": 1, "choices_per_vector": 65,
        "target": target_record, "target_semantics": "Original HF gate after current protected columns are removed and current native block32 rotation is applied in FP32",
        "initial_payload": original, "final_payload": final, "frozen_noncode_exact": True,
        "stage_xor": xor, "choice_histogram": torch.bincount(choices.long(), minlength=65).cpu().tolist(),
        "before_full_native_rotated": baseline_metrics, "after_full_native_rotated": error_stats(after, target),
        "chunks": trajectory, "elapsed_seconds": time.perf_counter() - started,
        "selection_uses_activations_or_dev": False, "persistent_bits": before_bits,
        "boundary": "Chunk scoring is exact on its native path; full-matrix BF16 differences are measured separately. Weight improvement does not establish MLP improvement."}
    return choices.cpu(), report


class Objective(nn.Module):
    def __init__(self, archive, cache, device):
        super().__init__()
        self.lengths = cache["document_lengths"]["train"]
        x = cache["inputs"]["train"].to(device)
        require(len(self.lengths) == 32 and sum(self.lengths) == len(x) and x.dtype == torch.bfloat16,
                "Fit exactly the original 32 documents' valid student-prefix tokens.")
        self.register_buffer("inputs", x)
        self.register_buffer("student_up", cache["student_up_weight"].to(device))
        self.register_buffer("student_down", cache["student_down_weight"].to(device))
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            gate = archive["targets"][NAME]["reference_weight"].to(device)
            up = archive["up_proj_weight"].to(device)
            down = archive["targets"][NAME.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
            self.register_buffer("up_output", F.linear(x, self.student_up))
            self.register_buffer("reference", F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down))
        self.normalizer = float(self.reference.double().square().sum()) / self.reference.numel()
        require(self.normalizer > 0 and math.isfinite(self.normalizer) and not list(self.parameters()),
                "The conditional HF output target and student context must be frozen and finite.")

    def prediction(self, full_weight):
        require(full_weight.dtype == torch.bfloat16 and full_weight.shape == (12288, 4096),
                "Use the actual full restored VAELinear weight; do not inverse-rotate it again.")
        return F.linear(F.silu(F.linear(self.inputs, full_weight)) * self.up_output, self.student_down)

    def forward(self, full_weight):
        raw = F.mse_loss(self.prediction(full_weight).float(), self.reference.float())
        return raw / self.normalizer, raw


@torch.no_grad()
def measure(layer, archive, cache, device):
    """Endpoint-only document metrics; heldout and replay never enter fitting."""
    ready(layer)
    weight = layer._decode_weight(dtype=torch.bfloat16)
    gate = archive["targets"][NAME]["reference_weight"].to(device)
    up = archive["up_proj_weight"].to(device)
    down = archive["targets"][NAME.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
    student_up, student_down = (cache[k].to(device) for k in ("student_up_weight", "student_down_weight"))
    result = {"original_coordinate_weight": error_stats(weight, gate)}
    with torch.autocast(device_type=device.type, enabled=False):
        for split, values in cache["inputs"].items():
            documents, offset = [], 0
            for index, length in enumerate(cache["document_lengths"][split]):
                x = values[offset:offset + length].to(device)
                reference = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
                candidate = F.linear(F.silu(F.linear(x, weight)) * F.linear(x, student_up), student_down)
                error, signal = squared_error(candidate, reference)
                require(signal > 0 and math.isfinite(error), "Invalid conditional endpoint energy.")
                documents.append({"index": index, "tokens": length, "squared_error": error,
                                  "squared_reference": signal, "nmse": error / signal})
                offset += length
            require(offset == len(values) and len(documents) == 32, "Incomplete endpoint population.")
            result[split] = aggregate(documents)
    return result


def train(layer, archive, cache, *, steps, deadline, observer):
    require(steps in (2, STEPS), "Only two mechanism updates or the fixed 500-step budget are supported.")
    ready(layer)
    device = next(layer.parameters()).device
    require(device.type == "cuda", "Real student-context fitting requires the assigned CUDA device.")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    original, before_schema, before_bits = records(layer), schema(layer), storage_bits(layer)
    bank_copies = [layer.get_stage_part_vq_storage(i, 0).clone() for i in range(2)]
    initial = {name: p.detach().clone() for name, p in layer._parallel_stage_decoder.named_parameters()}
    require(initial and all(v.dtype == torch.bfloat16 for v in initial.values()), "Both arms require fresh f927 BF16 decoder values.")
    objective = Objective(archive, cache, device).eval()
    fixed = records(objective)
    for p in layer._parallel_stage_decoder.parameters():
        p.data = p.data.float()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    layer.training = True
    layer._parallel_stage_decoder.eval()
    ready(layer)
    params = dict(layer._parallel_stage_decoder.named_parameters())
    require(all(p.requires_grad and p.dtype == torch.float32 for p in params.values()), "Decoder masters must be trainable FP32.")
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        weight = layer._decode_weight(dtype=torch.bfloat16)
        prediction = objective.prediction(weight)
        direct = F.linear(F.silu(layer(objective.inputs)) * F.linear(objective.inputs, objective.student_up), objective.student_down)
        require(torch.equal(prediction, direct), "Cached restoration/context differs from the same-shape native VAELinear forward.")
        before_loss, before_raw = objective(weight)
        before = {"nmse_loss": float(before_loss), "raw_mse": float(before_raw)}
        # Per-document GEMMs are an endpoint metric, not a different training population.
        per_document = measure(layer, archive, {"inputs": {"train": cache["inputs"]["train"]},
            "document_lengths": {"train": cache["document_lengths"]["train"]},
            "student_up_weight": cache["student_up_weight"], "student_down_weight": cache["student_down_weight"]}, device)
    del weight, prediction, direct, before_loss, before_raw
    optimizer = torch.optim.AdamW(params.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    trajectory, maximum = [], [0., 0.]
    for step in range(1, steps + 1):
        require(time.time() < deadline, "The assigned fitting deadline expired; no retry.")
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = LR * (1 - (step - 1) / STEPS)
        with torch.autocast("cuda", enabled=False):
            weight = layer._decode_weight(dtype=torch.bfloat16)
            loss, raw = objective(weight)
        require(bool(torch.isfinite(loss)), "Nonfinite conditional output loss.")
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params.values()), "Invalid FP32 decoder gradient.")
        norms = stage_norm([p.grad for p in params.values()])
        require(all(v > 0 for v in norms), "Both decoder stages need nonzero actual output gradients.")
        maximum = [max(x, y) for x, y in zip(maximum, norms, strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        row = {"step": step, "nmse_loss": float(loss), "raw_mse": float(raw),
            "lr": optimizer.param_groups[0]["lr"], "stage_gradient_l2": norms, "seconds": time.perf_counter() - tick}
        trajectory.append(row)
        if step <= 2 or step % 50 == 0:
            observer(row)
        del weight, loss, raw
    require({str(s[k].dtype) for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq")} == {"torch.float32"}, "Adam moments must be FP32.")
    with torch.no_grad():
        live = layer._decode_weight(dtype=torch.bfloat16)
        ready(layer)
        require(torch.equal(live, layer._decode_weight(dtype=torch.bfloat16)), "Rebuilding the packed plan changed the live decode.")
    del optimizer, live
    delta_fp32 = stage_norm([p.detach() - initial[k].float() for k, p in params.items()])
    for k, p in params.items():
        p.grad = None
        p.data = p.data.to(initial[k].dtype)
    layer.eval().disable_trainable_decode()
    layer.requires_grad_(False)
    ready(layer)
    delta_bf16 = stage_norm([p.detach().float() - initial[k].float() for k, p in params.items()])
    with torch.no_grad():
        deployed = layer._decode_weight(dtype=torch.bfloat16)
        loss, raw = objective(deployed)
    final = records(layer)
    require(all(final[k] == v for k, v in original.items() if not k.startswith("_parallel_stage_decoder."))
            and all(popcounts(v, layer.get_stage_part_vq_storage(i, 0)) == 0 for i, v in enumerate(bank_copies))
            and schema(layer) == before_schema and storage_bits(layer) == before_bits and records(objective) == fixed,
            "Frozen codes/context/protection/rotation or persistent storage changed.")
    require(all(v > 0 for v in delta_fp32 + delta_bf16), "Both stage updates must survive BF16 deployment.")
    report = {"status": "completed", "steps": steps, "schedule_steps": STEPS,
        "before": before, "after_deployment": {"nmse_loss": float(loss), "raw_mse": float(raw)},
        "step0_native_full_forward_exact": True, "step0_per_document": per_document,
        "step0_whole_minus_document_nmse": before["nmse_loss"] - per_document["train"]["nmse"],
        "trajectory": trajectory, "stage_gradient_l2_max": maximum,
        "stage_delta_l2_fp32": delta_fp32, "stage_delta_l2_bf16": delta_bf16,
        "initial_payload": original, "final_payload": final, "frozen_state_exact": True,
        "frozen_objective_buffers_exact": True, "persistent_bits": before_bits,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "elapsed_seconds": time.perf_counter() - started,
        "settings": {"lr": LR, "betas": [.9, .95], "eps": EPS, "weight_decay": 0, "horizon": STEPS,
            "warmup": 0, "normalizer": objective.normalizer, "train_tokens": len(objective.inputs),
            "target": "HF_MLP(frozen f927 student input), not independent full-teacher-forward output",
            "student_context": "Actual fixed f927 compressed up/down; full native gate decode includes rotation and INT8",
            "population": "All original train32 valid tokens on every update; no heldout/replay in gradients",
            "precision": "FP32 decoder and Adam; native BF16 decode/stage sum/MLP; FP32 MSE",
            "boundary": "One gate only; past full36 local improvements did not predict downstream quality."}}
    return report

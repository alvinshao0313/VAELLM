"""Real q-boundary remaining-LoRA check; a fixture, not a new CAT initial run.

The complete Qwen3-8B student keeps all seven projection categories. Only its
36 q projections come from the documented seed-31 pure-CAT block32 source.
The canonical runner owns data, teacher KD, optimizer, merging and BF16 export.
This state/merge check uses packed training decode and does not validate the
whole-decoder inference kernel. Run only with an explicit GPU allocation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from experiments.qwen3_w25.assemble_categories import CATEGORIES, EXPECTED, _require
from litebsq.vae_linear import VAELinear


REPO = Path(__file__).resolve().parents[2]
MODEL = Path("/home/star/models/Qwen/Qwen3-8B")
Q_SOURCE = REPO / "result/qwen3_8b_w25_20260930/reconstruction_block32_o3_down3/home_star_models_Qwen_Qwen3-8B_20261001_112401/final_model"
Q_SOURCE_ID = "ab0280cf-3637-47e3-b771-074b74532a54"
RECIPE = REPO / "scripts/cat_w25_remaining.sh"
MIX = "edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.500"
CODE_FILES = (
    "experiments/qwen3_w25/validate_remaining_lora.py", "scripts/cat_w25_remaining.sh",
    "train_utils/cat_after_category_common.py", "train_utils/cat_after_category_distill.py",
    "train_utils/cat_checkpoint_v6.py", "train_utils/checkpoint_v6.py",
    "train_utils/model_level_trainables.py", "train_utils/model_level_optimizer.py",
    "train_utils/distill_precision.py", "train_utils/distill_teacher.py", "train_utils/distill_data.py",
    "e2e_common/full_lora.py", "litebsq/vae_linear.py", "litebsq/fused_multistage_decoder.py",
)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def recipe_arguments(output):
    source = RECIPE.read_text(encoding="utf-8")
    marker = "python tools/cat_train.py"
    _require(source.count(marker) == 1, "Expected the one canonical CAT command in the recipe.")
    argv = shlex.split(source.split(marker, 1)[1].replace("\\\n", ""))
    _require(argv[-1] == "$@", "Unexpected recipe command tail.")
    return argv[:-1] + ["--output_dir", str(output), "--steps", "default=2",
                       "--logging_steps", "default=1", "--save_strategy", "no"]


def _check_stage(stage, cat_args):
    from e2e_common.data import normalize_dataset_mix_spec

    cfg = stage.config
    _require(tuple(cat_args.compression_categories.split(",")) == tuple(CATEGORIES), "Keep all seven categories in canonical order.")
    _require(stage.mode == "remaining_lora_prefix_decoder" and stage.stage_seed == 31,
             "Expected the first q remaining-prefix stage at seed 31.")
    _require(cfg.opt.steps == 2 and cfg.opt.batch_size == 4 and cfg.opt.gradient_accumulation_steps == 1,
             "This diagnostic requires exactly two B4 optimizer steps.")
    _require(cfg.data.model_max_length == 1024 and cfg.data.dynamic_padding and cfg.data.group_by_length,
             "Keep the original dynamic-padding/max1024 data protocol.")
    _require(cfg.data.dataset_mix == normalize_dataset_mix_spec(MIX)[2] and cfg.data.dataset_task == "sft",
             "Keep the real five-source SFT mixture.")
    _require((cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout) == (12, 24, 0.1), "Expected rank12/alpha24/dropout0.1.")
    _require(set(cfg.opt.distill_fp32_components) == {"lora", "decoder"} and stage.bf16 and not stage.fp16,
             "Expected FP32 LoRA/decoder parameters with BF16 computation.")
    _require(cfg.loss.loss_type == "kl_top_mass" and cfg.loss.top_k == 100
             and cfg.loss.prompt_loss_weight == 0.3, "Keep the original real teacher KD objective.")
    _require(cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none",
             "Auxiliary trainables are outside this check.")


class _TrainingObserver:
    """Read-only hooks on the real student and the real AdamW; no replacements."""

    def __init__(self, model, q_names, dense_names):
        self.model, self.q_names, self.dense_names = model, q_names, dense_names
        self.batches, self.teacher_batches, self.steps = [], [], []
        self.trainable_ids = None

    def student_forward(self, _module, _args, kwargs):
        if not self.model.training or not torch.is_grad_enabled():
            return
        ids = kwargs.get("input_ids")
        _require(torch.is_tensor(ids) and ids.ndim == 2 and ids.shape[0] == 4 and 0 < ids.shape[1] <= 1024,
                 "Actual student batch must retain B4 and the max1024 cap.")
        self.batches.append(list(ids.shape))
        if self.trainable_ids is not None:
            return
        lora_params, decoder_params = [], []
        for name in self.dense_names:
            layer = self.model.get_submodule(name)
            _require(set(layer.lora_A) == set(layer.lora_B) == {"default"} and layer.r["default"] == 12,
                     f"Expected one real rank12 remaining adapter: {name}.")
            lora_params.extend((layer.lora_A["default"].weight, layer.lora_B["default"].weight))
        for name in self.q_names:
            module = self.model.get_submodule(name)
            _require(isinstance(module, VAELinear) and module.trainable_decode, f"Missing trainable q decoder: {name}.")
            decoder_params.extend(module._parallel_stage_decoder.parameters())
        _require(all(p.requires_grad and p.dtype == torch.float32 for p in lora_params + decoder_params),
                 "Both real trainable components must retain FP32 storage.")
        self.trainable_ids = {id(p) for p in lora_params + decoder_params}
        _require(self.trainable_ids == {id(p) for p in self.model.parameters() if p.requires_grad},
                 "Unexpected trainable parameters outside remaining LoRA and q decoders.")
        self.components = {"lora": lora_params, "decoder": decoder_params}
        self.initial_components = {key: [p.detach().cpu().clone() for p in values]
                                   for key, values in self.components.items()}

    def teacher_forward(self, _module, _args, kwargs):
        ids = kwargs.get("input_ids")
        _require(torch.is_tensor(ids), "Real teacher did not receive input_ids.")
        self.teacher_batches.append(list(ids.shape))

    def optimizer_step(self, optimizer, _args, _kwargs):
        if not isinstance(optimizer, torch.optim.AdamW):
            return
        params = [p for group in optimizer.param_groups for p in group["params"]]
        _require(self.trainable_ids is not None and {id(p) for p in params} == self.trainable_ids,
                 "Actual AdamW parameter inventory differs from the observed real trainables.")
        record = {"step": len(self.steps) + 1, "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        for component, values in self.components.items():
            _require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in values),
                     f"Missing/non-finite {component} gradient at actual AdamW step.")
            record[component + "_grad_max"] = max(float(p.grad.detach().abs().max()) for p in values)
            _require(record[component + "_grad_max"] > 0, f"No {component} gradient path.")
            record[component + "_changed_fp32_elements"] = sum(
                int(torch.count_nonzero(p.detach().cpu() != before))
                for p, before in zip(values, self.initial_components[component]))
        self.steps.append(record)
        _require(len(self.steps) <= 2, "Canonical runner exceeded the two-step diagnostic budget.")


def run(output):
    from e2e_common.data import DATASET_MIX_SOURCE_PRESETS
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from e2e_common.full_lora import iter_named_peft_lora_layers
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from train_utils.cat_after_category_common import get_or_build_cat_projection_name_inventory, resolve_cat_after_category_stage
    from train_utils.cat_after_category_distill import run_after_category_distill
    from train_utils.cat_checkpoint_v6 import save_cat_v6_full_checkpoint
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_train_runtime import load_model_for_cat_train
    from train_utils.checkpoint_v6 import _atomic_write_json
    from train_utils.distill_teacher import DistillTeacherRuntime
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    output = Path(output).resolve()
    _require(not output.exists(), "Use a new independent validation directory.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and int(os.environ.get("WORLD_SIZE", "1")) == 1, "Allocate exactly one visible CUDA GPU/process.")
    output.mkdir(parents=True)
    log = logging.getLogger(__name__)
    report = {"status": "failed", "fixture": "original full Qwen3-8B plus seed31 pure-CAT block32 q projections",
              "scope": "real two-step q-category boundary; not fresh CAT compression, final 2.5-bpw model or quality evaluation",
              "fusion_forward_parity_checked": False,
              "fusion_forward_parity_boundary": "canonical runner combines dense merge with decoder FP32-to-BF16 export; this check audits state and strict reload",
              "code_sha256": {name: _sha(REPO / name) for name in CODE_FILES}}
    started, handles = time.perf_counter(), []
    try:
        argv = recipe_arguments(output)
        cat_args, hf_args, training_args, vae_args = parse_cat_runtime_args(argv)
        _require(Path(vae_args.model_path).resolve() == MODEL, "Unexpected original model.")
        stage = resolve_cat_after_category_stage(cat_args, training_args, category="q_proj", round_idx=0)
        _check_stage(stage, cat_args)
        report["effective_cat_arguments"] = argv
        report["dataset_sources"] = {}
        for term in MIX.split(","):
            alias, weight = term.split("=")
            path = REPO / DATASET_MIX_SOURCE_PRESETS[alias].path
            _require(path.is_file(), f"Missing real local mixture source: {path}.")
            report["dataset_sources"][alias] = {"path": str(path), "bytes": path.stat().st_size, "weight": float(weight)}
        configure_e2e_determinism(True)
        set_e2e_seed(31)
        donor, donor_meta, _ = load_v6_model_checkpoint(str(Q_SOURCE), strict=True, map_location="cpu", expected_kind="final_model")
        _require(donor_meta["checkpoint_id"] == Q_SOURCE_ID and set(donor_meta["compressed_targets"]) == set(EXPECTED),
                 "Unexpected complete block32 q donor.")
        q_names = tuple(name for name in EXPECTED if name.endswith(".q_proj"))
        dense_names = tuple(name for name in EXPECTED if name not in q_names)
        q_modules = {name: deepcopy(donor.get_submodule(name)) for name in q_names}
        del donor
        gc.collect()
        model = load_model_for_cat_train(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args)
        for name, module in q_modules.items():
            model.set_submodule(name, module)
        del q_modules
        inventory = get_or_build_cat_projection_name_inventory(model, vae_args=vae_args, compression_categories=CATEGORIES)
        _require(set(inventory.values()) == set(EXPECTED) and len(q_names) == 36 and len(dense_names) == 216,
                 "Fixture lost the full seven-category inventory.")
        _require(all(type(model.get_submodule(name)) is nn.Linear for name in dense_names), "Remaining projections must be actual dense Linear modules.")
        initial = {key: _tensor_record(value) for key, value in model.state_dict().items()}
        decoder_keys = {name + "._parallel_stage_decoder." + key for name in q_names
                        for key in model.get_submodule(name)._parallel_stage_decoder.state_dict()}
        dense_keys = {name + ".weight" for name in dense_names}
        report.update(q_source_dir=str(Q_SOURCE), q_source_checkpoint_id=Q_SOURCE_ID,
                      q_source_state_sha256=_sha(Q_SOURCE / "pytorch_model.bin"),
                      compression_categories=list(CATEGORIES), compressed_q_count=36, remaining_dense_count=216)
        teacher = DistillTeacherRuntime(model_path=str(MODEL), access_token=hf_args.access_token,
            forward_device="cuda:0", dtype=torch.bfloat16, model_offload="none", logger=log)
        observer = _TrainingObserver(model, q_names, dense_names)
        teacher_model = teacher.get_or_load()
        handles = [model.register_forward_pre_hook(observer.student_forward, with_kwargs=True),
                   teacher_model.register_forward_pre_hook(observer.teacher_forward, with_kwargs=True),
                   register_optimizer_step_post_hook(observer.optimizer_step)]
        torch.cuda.reset_peak_memory_stats()
        result = run_after_category_distill(model=model, category="q_proj", cat_args=cat_args, vae_args=vae_args,
            training_args=training_args, logger=log, lora_round_idx=0,
            transpose_modules=tuple(cat_args.transpose_modules.split(",")), only_decoder_projections=True,
            compression_categories=CATEGORIES, teacher_runtime=teacher, newly_compressed_target_count=36,
            current_category_target_names=q_names, online_cat=True)
        for handle in handles:
            handle.remove()
        handles.clear()
        model = result.model
        _require(result.did_train and len(observer.steps) == len(observer.batches) == 2 and len(observer.teacher_batches) >= 2,
                 "Expected two actual student/teacher KD batches and two AdamW steps.")
        _require(all(observer.steps[-1][key + "_changed_fp32_elements"] > 0 for key in ("lora", "decoder")),
                 "Both real FP32 components must update within the two-step budget; the first warmup step may have LR zero.")
        _require(result.distill_meta["remaining_lora_target_count"] == 216 and result.distill_meta["decoder_target_count"] == 36,
                 "Canonical runner reported an incomplete recovery inventory.")
        report.update(optimizer_steps=observer.steps, student_batch_shapes=observer.batches,
                      teacher_batch_shapes=observer.teacher_batches, trainable_parameter_count=len(observer.trainable_ids),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      distill_meta=result.distill_meta)
        decode_stats = VAELinear.get_fuse_stats()
        report["training_decode_stats"] = decode_stats
        # On these parallel q decoders, packed-u8 also increments generic hit.
        # Equal positive counters with no miss exclude a wholefused call.
        _require(decode_stats["hit"] == decode_stats["packed_u8_linear_hit"] > 0 and decode_stats["miss"] == 0,
                 "Expected every parallel q decode hit to use packed uint8, without misses or wholefused calls.")
        _require(model is observer.model and not list(iter_named_peft_lora_layers(model)),
                 "PEFT wrapper/layers survived canonical finalization.")
        _require(all(type(model.get_submodule(name)) is nn.Linear for name in dense_names), "Dense merge did not restore ordinary Linear modules.")
        _require(all(not model.get_submodule(name).trainable_decode for name in q_names)
                 and not any(p.requires_grad for p in model.parameters()), "Trainable runtime survived finalization.")
        final = {key: _tensor_record(value) for key, value in model.state_dict().items()}
        _require(initial.keys() == final.keys() and not any("lora" in key.lower() or "low_rank" in key.lower() for key in final),
                 "Final state inventory contains external LoRA or changed topology.")
        changed = {key for key in final if final[key] != initial[key]}
        _require(changed <= decoder_keys | dense_keys, "Frozen q codes/INT8/rotation or other model tensors changed.")
        _require(bool(changed & decoder_keys) and bool(changed & dense_keys), "Both decoder and merged dense updates must survive BF16 export.")
        for key, value in model.state_dict().items():
            _require(initial[key]["shape"] == final[key]["shape"] and initial[key]["dtype"] == final[key]["dtype"], f"Final shape/dtype changed: {key}.")
            if key in decoder_keys | dense_keys:
                _require(value.dtype == torch.bfloat16 and torch.isfinite(value).all(), f"Invalid BF16 result: {key}.")
        teacher._model = None
        del teacher_model, teacher, observer, result
        gc.collect()
        torch.cuda.empty_cache()
        saved = save_cat_v6_full_checkpoint(model, str(output / "category_boundary"), checkpoint_kind="category_boundary",
            category="q_proj", completed_categories=("q_proj",), compression_categories=CATEGORIES,
            cat_args=cat_args, vae_args=vae_args, training_args=training_args,
            tokenizer=vae_args._cached_lora_tokenizer, base_model_path=str(MODEL), distill_stage_meta=report["distill_meta"], round_idx=0)
        del model
        gc.collect()
        fresh, meta, _ = load_v6_model_checkpoint(str(output / "category_boundary"), strict=True, map_location="cpu", expected_kind="category_boundary")
        _require(meta["checkpoint_id"] == saved["checkpoint_id"] and meta["lora_config"] is None,
                 "Fresh category boundary identity/LoRA metadata differs.")
        _require(set(meta["compressed_targets"]) == set(q_names) and set(meta["pending_dense_targets"]) == set(dense_names)
                 and not meta["skip_targets"] and tuple(meta["compression_categories"]) == tuple(CATEGORIES),
                 "Saved q boundary lost the complete canonical inventory.")
        _require({key: _tensor_record(value) for key, value in fresh.state_dict().items()} == final,
                 "Fresh strict reload differs from the complete canonical BF16 final state.")
        report.update(status="verified", checkpoint_id=meta["checkpoint_id"], strict_reloaded_tensors=len(final),
                      changed_decoder_tensors=len(changed & decoder_keys), changed_dense_weights=len(changed & dense_keys),
                      frozen_tensors_compared=len(final) - len(decoder_keys | dense_keys),
                      fresh_strict_reload_state_exact=True, no_external_lora=True)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for handle in handles:
            handle.remove()
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_write_json(report, str(output / "report.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(run(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

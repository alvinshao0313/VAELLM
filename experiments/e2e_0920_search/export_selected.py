#!/usr/bin/env python3
"""Export a selected E2E LoRA step without resuming or changing its schedule.

Export runs on one GPU. Invoke --evaluate-only in a fresh process (optionally
torchrun with four ranks) to strictly reload and evaluate the saved artifact.
--validate-only inspects metadata without importing torch or loading weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
TASKS = ("boolq", "rte", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "mmlu")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _evaluation_config(contract: dict) -> dict:
    evaluation = contract["runtime"]["evaluation"]
    tasks = tuple(str(evaluation["eval_tasks"]).split(","))
    if tasks != TASKS or evaluation["eval_num_fewshot"] != 0 or evaluation["eval_limit"] is not None:
        raise ValueError("Selected export requires the unchanged full eight-task zero-shot protocol.")
    if evaluation["eval_hif4_act"]:
        raise ValueError("This search exports without HiFloat4 activation evaluation.")
    return evaluation


def _validate_contract(meta: dict) -> dict:
    contract = meta["immutable_resume_contract"]
    aux = contract["aux"]
    if meta["train_mode"] != "lora" or contract["train_mode"] != "lora":
        raise ValueError("This exporter supports train_mode=lora only; decoder/bit training is unsupported.")
    for key, expected in (("norm_train_mode", "all"), ("lm_head_train_mode", "linear")):
        if meta[key] != expected or aux[key] != expected:
            raise ValueError(f"This exporter requires {key}={expected}.")
    if aux.get("residual_lora_mode", "none") != "none":
        raise ValueError("Residual LoRA is outside this search export topology.")
    runtime = contract["runtime"]
    if runtime["parallel_mode"] != "dp" or runtime["offload_mode"] != "none" or runtime["distill_hif4_act"]:
        raise ValueError("This exporter requires the DP/no-offload/no-HiFloat4 search path.")
    precision = contract["precision"]
    if not precision["bf16"] or precision["fp16"]:
        raise ValueError("This exporter requires the search's BF16 compute/export precision.")
    lora = contract["lora"]
    ranks = [int(lora["rank"]), *(int(value) for value in lora["rank_pattern"].values())]
    if any(not 1 <= rank <= 8 for rank in ranks):
        raise ValueError(f"Search export requires every actual LoRA rank <= 8: {ranks}")
    if meta["target_layers"] != contract["target_layers"] or meta["target_modules"] != contract["target_modules"]:
        raise ValueError("Checkpoint topology and saved training contract disagree.")
    _evaluation_config(contract)
    return contract


def inspect_step(checkpoint: Path) -> tuple[dict, dict, dict]:
    meta = read_json(checkpoint / "checkpoint_meta.json")
    if meta.get("format") != "vaellm_model_checkpoint_v6" or meta.get("checkpoint_kind") != "training_step":
        raise ValueError("--checkpoint must be a committed v6 training_step checkpoint.")
    contract = _validate_contract(meta)
    if meta["lora_config"] != contract["lora"]:
        raise ValueError("Checkpoint LoRA topology differs from the saved training contract.")
    if (checkpoint / "sparse_bit_tuning").exists():
        raise ValueError("Sparse Bit state is unsupported by this LoRA-only exporter.")
    if not (checkpoint / "training_model_state.pt").is_file():
        raise FileNotFoundError(checkpoint / "training_model_state.pt")
    state = read_json(checkpoint / "trainer_state.json")
    if not 0 < int(state["global_step"]) <= int(state["max_steps"]):
        raise ValueError("Selected checkpoint has an invalid optimizer step.")
    if int(state["max_steps"]) != int(contract["optimization"]["steps"]):
        raise ValueError("TrainerState and original schedule disagree; do not rewrite either to export.")
    if checkpoint.parent.name != "trainer_state":
        raise ValueError("Select the checkpoint inside its original run/trainer_state directory.")
    snapshot = read_json(checkpoint.parent.parent / "normalized_e2e_runtime_args.json")
    cfg = snapshot["canonical_config"]
    for key in ("train_mode", "target_layers", "data", "loss", "runtime"):
        if cfg[key] != contract[key]:
            raise ValueError(f"Run configuration does not match selected checkpoint: {key}")
    for key, value in contract["optimization"].items():
        if cfg["opt"].get(key) != value:
            raise ValueError(f"Run optimization setting does not match selected checkpoint: {key}")
    for key, value in contract["aux"].items():
        if cfg["aux"].get(key) != value:
            raise ValueError(f"Run auxiliary setting does not match selected checkpoint: {key}")
    for key in ("rank", "alpha", "dropout"):
        if cfg["lora"][key] != contract["lora"][key]:
            raise ValueError(f"Run LoRA setting does not match selected checkpoint: {key}")
    return meta, state, snapshot


def _compressed_base_digest(selected) -> str:
    """Include every packed/decoder tensor, excluding the trained LoRA payload."""
    import torch

    digest = hashlib.sha256()
    for name, module in selected:
        for key, tensor in sorted(module.state_dict().items()):
            if key in {"low_rank_a", "low_rank_b"}:
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}.{key}:{tuple(value.shape)}:{value.dtype}\n".encode())
            digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _eval_args(contract: dict) -> SimpleNamespace:
    from compressed_e2e_fintuning.runtime_v6 import _build_eval_args

    cfg = SimpleNamespace(runtime=SimpleNamespace(evaluation=SimpleNamespace(**_evaluation_config(contract))))
    return _build_eval_args(cfg)


def _setup_seed(snapshot: dict) -> None:
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed

    configure_e2e_determinism(bool(snapshot["training_args"]["full_determinism"]))
    set_e2e_seed(int(snapshot["canonical_config"]["data"]["seed"]))


def export_checkpoint(checkpoint: Path, output_dir: Path, *, device: str = "cuda:0") -> dict:
    meta, state, snapshot = inspect_step(checkpoint)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Export is single-process; use torchrun only with --evaluate-only.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite export output: {output_dir}")
    import torch
    from compressed_e2e_fintuning.runtime_v6 import (
        _assert_final_runtime_clean, _build_v6_tokenizer, _collect_existing_full_low_rank,
        _sync_model_padding_config,
    )
    from compressed_e2e_fintuning.runtime_v6_pipeline import (
        _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe,
    )
    from compressed_e2e_fintuning.v6_runtime_state import restore_e2e_mutable_state
    from e2e_common.full_lora import collect_exact_peft_lora_config, finalize_model_level_lora
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from rotation.model_utils import get_layers
    from train_utils.checkpoint_v6 import (
        load_v6_training_model_state, load_v6_training_step_meta,
        resolve_training_step_round_base_ref, save_v6_full_checkpoint,
    )
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.distill_precision import configure_distill_precision, prepare_model_export
    from train_utils.model_level_trainables import build_model_level_trainable_selection, finalize_lm_head_linear_if_needed
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Selected-model export requires an available CUDA GPU for its real parity probes.")
    torch.cuda.set_device(torch.device(device))
    log = logging.getLogger("export_selected")
    _setup_seed(snapshot)
    meta = load_v6_training_step_meta(str(checkpoint))
    base_dir, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    contract = meta["immutable_resume_contract"]
    model, loaded_meta, _ = load_v6_model_checkpoint(base_dir, strict=True)
    if loaded_meta["checkpoint_id"] != meta["round_base_checkpoint_id"]:
        raise ValueError("Selected step resolves to a different round base.")
    if (base_meta.get("extra_meta") or {}).get("residual_lora") is not None:
        raise ValueError("Round base contains unsupported residual LoRA topology.")
    selected = collect_e2e_compressed_targets(
        model, target_layers=tuple(meta["target_layers"]), target_modules=tuple(meta["target_modules"]),
        num_layers=len(list(get_layers(model))),
    )
    frozen_digest = _compressed_base_digest(selected)
    lora = meta["lora_config"]
    fp32_components = contract["optimization"].get("distill_fp32_components", ())
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(**contract["aux"]), compressed_modules=selected,
        dense_target_modules=(), rank=int(lora["rank"]), alpha=float(lora["alpha"]),
        dropout=float(lora["dropout"]), rank_explicit=True, fp32_components=fp32_components,
        initial_low_rank_payloads=_collect_existing_full_low_rank(selected),
        train_decoder=False, train_lora=True, freeze=True,
    )
    model = selection.peft_model or model
    training_args = SimpleNamespace(bf16=True, fp16=False)
    configure_distill_precision(selection, components=fp32_components, training_args=training_args, logger=log)
    actual_lora = collect_exact_peft_lora_config(
        model, default_rank=int(lora["rank"]), alpha=float(lora["alpha"]), dropout=float(lora["dropout"]),
    )
    if actual_lora != lora:
        raise ValueError("Rebuilt adapter topology does not match the selected checkpoint.")
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location="cpu")
    restore_e2e_mutable_state(
        model, selection=selection, selected_vae_modules=selected,
        checkpoint_state=mutable, checkpoint_manifest=manifest,
    )
    del mutable
    if _compressed_base_digest(selected) != frozen_digest:
        raise RuntimeError("Restoring this LoRA step changed frozen compressed weights or decoder state.")
    tokenizer = _build_v6_tokenizer(str(base_meta["base_model_path"]), SimpleNamespace(**snapshot["hf_args"]))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    model.eval().to(device)
    probe_inputs = _build_finalization_probe_inputs(tokenizer)
    before, dtype = _run_finalization_probe(model, probe_inputs)
    model = finalize_model_level_lora(model, compressed_proxy_names=[name for name, _ in selected])
    core, core_dtype = _run_finalization_probe(model, probe_inputs)
    if core_dtype != dtype:
        raise RuntimeError("LoRA finalization changed output precision.")
    audit = {
        "runtime": "experiments.e2e_0920_search.export_selected",
        "source_checkpoint": str(checkpoint), "source_checkpoint_id": meta["checkpoint_id"],
        "source_global_step": int(state["global_step"]), "source_max_steps": int(state["max_steps"]),
        "frozen_compressed_base_sha256": frozen_digest,
        "core_structural_finalization_forward_parity": _assert_finalization_probe_close(
            before, core, output_dtype=dtype, label="Core structural finalization parity",
        ),
    }
    if not finalize_lm_head_linear_if_needed(model, lm_head_train_mode="linear"):
        raise RuntimeError("Expected the trained post-norm head linear to be fused.")
    after, after_dtype = _run_finalization_probe(model, probe_inputs)
    if after_dtype != dtype:
        raise RuntimeError("LM-head fusion changed output precision.")
    fusion_tolerance = dict(output_dtype=dtype, ulp_multiplier=32.0, rtol_override=1e-3,
                            atol_override=0.25, relative_l2_limit=5e-3)
    audit["lm_head_fusion_forward_parity"] = _assert_finalization_probe_close(
        core, after, label="LM-head fusion forward parity", **fusion_tolerance,
    )
    audit["finalization_forward_parity"] = _assert_finalization_probe_close(
        before, after, label="End-to-end finalization parity", **fusion_tolerance,
    )
    clear_model_vae_linear_cache(model)
    _assert_final_runtime_clean(model)
    audit["export_dtype"] = str(prepare_model_export(model, training_args))
    exported, exported_dtype = _run_finalization_probe(model, probe_inputs)
    difference = exported - after
    audit["export_precision_probe"] = {
        "output_dtype": str(exported_dtype), "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / after.norm().clamp_min(1e-12)),
    }
    clear_model_vae_linear_cache(model)
    model.requires_grad_(False).to("cpu")
    if _compressed_base_digest(selected) != frozen_digest:
        raise RuntimeError("Finalization/export changed frozen compressed weights or decoder state.")
    result = save_v6_full_checkpoint(
        model, str(output_dir), checkpoint_kind="final_model",
        **{key: tuple(base_meta.get(key) or ()) for key in (
            "compressed_targets", "pending_dense_targets", "skip_targets", "legacy_original_only_sources",
            "completed_categories", "compression_categories",
        )},
        train_mode="lora", norm_train_mode="all", lm_head_train_mode="linear", lora_config=None,
        resolved_learning_rates=meta["resolved_learning_rates"], target_layers=meta["target_layers"],
        target_modules=meta["target_modules"], immutable_resume_contract=contract,
        finalized_status={"sparse_bit_committed": False, "decoder_finalized": False, "lora_finalized": True,
                          "aux_finalized": True, "runtime_clean": True, "inference_forward_parity": True},
        runtime_audit=audit, base_model_path=str(base_meta["base_model_path"]),
        tokenizer=tokenizer, save_config=True,
    )
    record = {"checkpoint_id": result["checkpoint_id"], "output_dir": str(output_dir), **audit}
    (output_dir / "export_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Exported selected step %d to %s; fresh-process evaluation is still required.", state["global_step"], output_dir)
    return record


def evaluate_checkpoint(checkpoint: Path, output_dir: Path) -> dict | None:
    """Strict full-checkpoint reload; same task partition and scoring as training."""
    import torch
    from compressed_e2e_fintuning.mid_eval import run_e2e_lm_eval
    from compressed_e2e_fintuning.runtime_v6 import (
        _assert_final_runtime_clean, _build_v6_tokenizer, _sync_model_padding_config,
    )
    from train_utils.checkpoint_v6 import load_v6_meta
    from train_utils.distill_precision import install_precision_runtime
    from train_utils.distributed_guard import distributed_guarded_all, distributed_guarded_main
    from train_utils.lora_utils import ensure_distill_process_group_initialized, is_distill_main_process
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Fresh selected-model evaluation requires CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) not in (1, 4):
        raise ValueError("Use one or four ranks for the search's eight-task evaluation.")
    ensure_distill_process_group_initialized()

    def create_output():
        # Only rank zero checks/creates this directory, after all ranks join.
        # Existing results and concurrent invocations must fail on every rank.
        output_dir.mkdir(parents=True, exist_ok=False)

    meta = distributed_guarded_all(lambda: load_v6_meta(str(checkpoint)))
    if meta["checkpoint_kind"] != "final_model":
        raise ValueError("Fresh evaluation requires an independently loadable final_model.")
    contract = _validate_contract(meta)
    if not (meta.get("finalized_status") or {}).get("runtime_clean"):
        raise ValueError("Saved final model does not declare clean finalized topology.")
    distributed_guarded_main(create_output, barrier=True)
    model, loaded_meta, _ = distributed_guarded_all(
        lambda: load_v6_model_checkpoint(str(checkpoint), strict=True, expected_kind="final_model"),
    )
    if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
        raise RuntimeError("Selected full checkpoint changed during reload.")
    _assert_final_runtime_clean(model)
    tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), SimpleNamespace())
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    model.eval().requires_grad_(False)
    install_precision_runtime(model, torch.bfloat16)
    result = run_e2e_lm_eval(
        model=model, tokenizer=tokenizer, args=_eval_args(contract),
        base_model_path=str(meta["base_model_path"]), output_dir=str(output_dir),
        log=logging.getLogger("export_selected"), eval_tag="reloaded", move_to_device=True,
        cache_decoded_weight=False,
    )
    def write_summary():
        if result is None:
            raise RuntimeError("Main rank did not receive the full eight-task evaluation result.")
        summary = _evaluation_summary(checkpoint, meta, result)
        (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    summary = distributed_guarded_main(write_summary, barrier=True)
    return summary if is_distill_main_process() else None


def _evaluation_summary(checkpoint: Path, meta: dict, result: dict) -> dict:
    raw_metrics = result["result"]["task_metrics"]
    if set(raw_metrics) != set(TASKS):
        raise ValueError("Evaluation did not return exactly the eight search task metrics.")
    metrics = {task: float(raw_metrics[task]) for task in TASKS}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        raise ValueError(f"Invalid full-evaluation metrics: {metrics}")
    return {
        "checkpoint_dir": str(checkpoint), "checkpoint_id": meta["checkpoint_id"],
        "strict_reload": True, "task_metrics": metrics,
        "task_metric_keys": {task: result["result"]["task_metric_keys"][task] for task in TASKS},
        "mean_percent": sum(metrics.values()) / len(TASKS) * 100,
        "raw_metrics_path": result["json_path"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Single-process export GPU; evaluation uses each local rank.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    checkpoint, output_dir = args.checkpoint.resolve(), args.output_dir.resolve()
    if not args.evaluate_only and output_dir.exists():
        raise FileExistsError(f"Output must be a new independent directory: {output_dir}")
    if args.validate_only:
        meta, state, _ = inspect_step(checkpoint)
        print(json.dumps({"checkpoint_id": meta["checkpoint_id"], "global_step": state["global_step"],
                          "max_steps": state["max_steps"], "metadata_valid": True}, ensure_ascii=False))
        return
    sys.path.insert(0, str(REPO))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.evaluate_only:
            result = evaluate_checkpoint(checkpoint, output_dir)
        else:
            result = export_checkpoint(checkpoint, output_dir, device=args.device)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

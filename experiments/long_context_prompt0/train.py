"""The existing OV72 long-SFT recipe with response-only KL weighting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import write_json
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.long_context_recovery.data import registered_config, registered_preset
from experiments.long_context_recovery.observations import observe_native_trainer
from experiments.long_context_recovery.train_400 import Exposure
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "result/qwen3_8b_mmlu65_20261004/best1200_ov72_longalpaca8k_s400"


def read(path):
    return json.loads(Path(path).read_text())


def training_arguments(output):
    argv = list(read(CONTROL / "launch.json")["training_argv"])
    for flag, value in (("--run_root_dir", str(Path(output).resolve())), ("--prompt_loss_weight", "0")):
        require(argv.count(flag) == 1, f"Ambiguous control option: {flag}")
        argv[argv.index(flag) + 1] = value
    return argv


def parse_configuration(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--long_context_manifest", required=True)
    registration, remaining = local.parse_known_args(argv)
    cfg, hf, args = parse_args(remaining)
    require(argv == training_arguments(cfg.run_root_dir),
            "Only the output path and prompt_loss_weight=.3 to 0 may differ from the actual control argv.")
    cfg.data = registered_config(cfg.data, registration.long_context_manifest)
    control = read(CONTROL / "launch.json")
    require(cfg.data.data_registration == control["data_registration"]
            and cfg.loss.prompt_loss_weight == 0 and cfg.loss.loss_type == "kl_top_mass"
            and Path(cfg.student_checkpoint_dir).resolve() == SOURCE,
            "Keep the identical original SFT stream/source and use native response-only pure KL.")
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
            and args.save_steps == 400 and args.save_total_limit == 1
            and args.gradient_accumulation_steps == args.per_device_train_batch_size == 1,
            "The existing 2000-step schedule, warmup20, B1/accum1 and save400 must remain unchanged.")
    meta = read(SOURCE / "checkpoint_meta.json")
    require(meta["checkpoint_id"] == SOURCE_ID, "The fresh f927 input identity changed.")
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    inventory = expected_inventory(SOURCE, architecture, names)
    require(list(names) == control["selected_targets"] and inventory == control["expected_inventory"],
            "The actual OV72/432 decoder/2991 frozen tensor inventory changed.")
    return cfg, hf, args, names, architecture, inventory


def train(argv=None):
    cfg, hf, args, names, _architecture, inventory = parse_configuration(argv)
    output = Path(cfg.run_root_dir)
    require(output.is_dir() and not (output / "training.json").exists(), "Use a fresh prepared output.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Bind one assigned GPU.")
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    exposure = Exposure(output / "exposure.json")
    report = {"status": "running", "expected_inventory": inventory, "selected_targets": list(names),
              "data_registration": cfg.data.data_registration, "prompt_loss_weight": 0}
    try:
        with registered_preset(cfg.data), observe_native_trainer(exposure):
            result = run(cfg, hf, args, extra_callbacks=(Budget400Callback(),))
        require(result["global_step"] == 400 and result["round_base_checkpoint_id"] == SOURCE_ID,
                "The native run must complete 400 real updates from f927.")
        run_dir = Path(result["run_output_dir"])
        snapshot = read(run_dir / "normalized_e2e_runtime_args.json")
        require(snapshot["canonical_config"]["data"]["data_registration"] == cfg.data.data_registration
                and snapshot["canonical_config"]["loss"]["prompt_loss_weight"] == 0,
                "The actual native snapshot lost the data identity or response-only objective.")
        for relative in ("trainer_state/checkpoint-400/checkpoint_meta.json", "final_model/checkpoint_meta.json"):
            contract = read(run_dir / relative)["immutable_resume_contract"]
            require(contract["data"]["data_registration"] == cfg.data.data_registration
                    and contract["loss"]["prompt_loss_weight"] == 0,
                    "Saved native data or loss identity changed.")
        sources = read(run_dir / "run_meta.json")["data_sources"]
        require(len(sources) == 1 and sources[0]["alias"] == "longalpaca"
                and sources[0]["path"] == cfg.data.data_registration["path"]
                and sources[0]["raw_rows"] == 12000 and sources[0]["is_iterable"],
                "Native training did not use the unchanged full LongAlpaca stream.")
        require(len(exposure.batches) == 400
                and exposure.batches == read(CONTROL / "exposure.json")["batches"],
                "Actual 400 batches differ from the control in input/label hashes, order or token exposure.")
        report.update(status="trained", training_result=result, actual_control_exposure_exact=True)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        exposure.close()
        write_json(output / "training.json", report)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()

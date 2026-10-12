"""Run four observed native long-SFT updates on the unchanged 2000-step schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import write_json
from experiments.long_context_recovery.contract import SOURCE_ID, require, validate_config, validate_training_args
from experiments.long_context_recovery.data import registered_config, registered_preset
from experiments.long_context_recovery.observations import LongContextObservations, observe_native_trainer
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory


def parse_configuration(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--long_context_manifest", required=True)
    local, remaining = parser.parse_known_args(argv)
    cfg, hf, args = parse_args(remaining)
    cfg.data = registered_config(cfg.data, local.long_context_manifest)
    validate_config(cfg)
    validate_training_args(args)
    require(cfg.data.data_registration["purpose"] == "resource_diagnostic",
            "This four-step entry consumes only the explicitly geometry-selected resource diagnostic.")
    meta = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    require(meta["checkpoint_id"] == SOURCE_ID, "The initial source checkpoint UUID changed.")
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    inventory = expected_inventory(cfg.student_checkpoint_dir, architecture, names)
    require(inventory["selected_projections"] == 72 and inventory["selected_decoder_tensors"] == 432
            and inventory["selected_decoder_elements"] == 1387008 and inventory["frozen_tensors"] == 2991
            and inventory["all_code_banks"] == 504, "The actual f927 OV72 inventory changed.")
    return cfg, hf, args, names, architecture, inventory


def check_saved_data(run_dir, registration):
    run_dir = Path(run_dir)
    snapshot = json.loads((run_dir / "normalized_e2e_runtime_args.json").read_text())
    require(snapshot["canonical_config"]["data"]["data_registration"] == registration,
            "Native normalized configuration lost the registered source identity.")
    for relative in ("trainer_state/checkpoint-2/checkpoint_meta.json",
                     "trainer_state/checkpoint-4/checkpoint_meta.json", "final_model/checkpoint_meta.json"):
        meta = json.loads((run_dir / relative).read_text())
        require(meta["immutable_resume_contract"]["data"]["data_registration"] == registration,
                f"Native checkpoint changed its data registration: {relative}")
    sources = json.loads((run_dir / "run_meta.json").read_text())["data_sources"]
    require(len(sources) == 1 and sources[0]["alias"] == registration["alias"]
            and sources[0]["path"] == registration["path"]
            and sources[0]["raw_rows"] == registration["rows"] and sources[0]["is_iterable"],
            "The actual native training dataset differs from its registered original-format rows.")


def train(argv=None):
    cfg, hf, args, names, architecture, inventory = parse_configuration(argv)
    output = Path(cfg.run_root_dir)
    require(output.is_dir() and not (output / "training.json").exists()
            and not (output / "observations.json").exists(), "Use the prepared fresh diagnostic output directory.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "The root must bind exactly one CUDA GPU before the actual short run.")
    observer = LongContextObservations(names, architecture, output)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    report = {"status": "running", "diagnostic_not_quality": True, "expected_inventory": inventory,
              "data_registration": cfg.data.data_registration, "selected_targets": list(names)}
    try:
        with registered_preset(cfg.data), observe_native_trainer(observer):
            result = run(cfg, hf, args, extra_callbacks=(observer,))
        require(result["global_step"] == 4 and result["round_base_checkpoint_id"] == SOURCE_ID,
                "The native diagnostic did not stop after four real updates from f927.")
        check_saved_data(result["run_output_dir"], cfg.data.data_registration)
        report.update(status="trained", training_result=result, observations=observer.report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      observations=observer.report)
        raise
    finally:
        observer.close()
        write_json(output / "training.json", report)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()

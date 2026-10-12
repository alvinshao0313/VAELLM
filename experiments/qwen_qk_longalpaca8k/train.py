"""Run the unchanged native trainer with the matched QK decoder subset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.long_context_recovery.contract import SOURCE_ID, require
from experiments.long_context_recovery.data import registered_preset
from experiments.long_context_recovery.observations import LongContextObservations, observe_native_trainer
from experiments.long_context_recovery.train_400 import Exposure
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.qwen_qk_longalpaca8k.config import CONTROLS, configuration, read


def train(record_path):
    record = read(record_path)
    mode, output = record["mode"], Path(record["output_dir"])
    require(not (output / "training.json").exists() and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "Use one assigned GPU and a fresh prepared output.")
    check_hashes(record["source_sha256"])
    check_hashes(record["input_evidence_sha256"])
    cfg, hf, args, names, architecture, inventory, _ = configuration(mode, output, record["training_argv"])
    require(list(names) == record["selected_targets"] and inventory == record["expected_inventory"],
            "Actual training scope differs from CPU preparation.")
    if mode == "formal":
        proof = Path(record["resource_validation"]["path"]) / "verification.json"
        require(sha256(proof) == record["resource_validation"]["verification_sha256"]
                and read(proof)["status"] == "verified", "The actual QK8K resource proof is required.")
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    observer = (LongContextObservations(names, architecture, output) if mode == "resource"
                else Exposure(output / "exposure.json"))
    callbacks = (observer,) if mode == "resource" else (Budget400Callback(),)
    report = {"status": "running", "mode": mode, "selected_targets": list(names),
              "expected_inventory": inventory, "data_registration": cfg.data.data_registration}
    try:
        with registered_preset(cfg.data), observe_native_trainer(observer):
            result = run(cfg, hf, args, extra_callbacks=callbacks)
        stop = 4 if mode == "resource" else 400
        require(result["global_step"] == stop and result["round_base_checkpoint_id"] == SOURCE_ID,
                "Native training did not complete the selected prefix from fresh f927.")
        native = Path(result["run_output_dir"])
        registration = cfg.data.data_registration
        snapshot = read(native / "normalized_e2e_runtime_args.json")
        require(snapshot["canonical_config"]["data"]["data_registration"] == registration,
                "The actual native snapshot lost the original data identity.")
        saves = (2, 4) if mode == "resource" else (400,)
        for relative in [*(f"trainer_state/checkpoint-{step}/checkpoint_meta.json" for step in saves),
                         "final_model/checkpoint_meta.json"]:
            contract = read(native / relative)["immutable_resume_contract"]
            require(contract["data"]["data_registration"] == registration
                    and set(contract["target_modules"]) == {"q_proj", "k_proj"},
                    "Saved native data or selected decoder scope changed.")
        sources = read(native / "run_meta.json")["data_sources"]
        require(len(sources) == 1 and sources[0]["alias"] == "longalpaca"
                and sources[0]["path"] == registration["path"]
                and sources[0]["raw_rows"] == registration["rows"] and sources[0]["is_iterable"],
                "The original native LongAlpaca stream was not used.")
        if mode == "resource":
            report["observations"] = observer.report
        else:
            require(len(observer.batches) == 400
                    and observer.batches == read(CONTROLS["formal"] / "exposure.json")["batches"],
                    "All400 actual input/label hashes, token counts and ordering must match the OV control.")
            report["actual_control_exposure_exact"] = True
        report.update(status="trained", training_result=result)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        if mode == "resource":
            report["observations"] = observer.report
        raise
    finally:
        observer.close()
        write_json(output / "training.json", report)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    train(parser.parse_args().launch_record)

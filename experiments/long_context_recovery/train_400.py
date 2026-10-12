"""Native OV72 decoder recovery on the unchanged full LongAlpaca SFT stream."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import write_json
from experiments.long_context_recovery.contract import SOURCE_ID, require, validate_config
from experiments.long_context_recovery.data import registered_config, registered_preset
from experiments.long_context_recovery.observations import batch_record, observe_native_trainer
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
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
    require(cfg.data.data_registration["purpose"] == "full_dataset"
            and cfg.data.data_registration["alias"] == "longalpaca"
            and cfg.data.data_registration["rows"] == 12000
            and cfg.data.data_registration["selection"] is None,
            "Use all 12000 unchanged LongAlpaca rows through the existing native stream.")
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
            and args.save_steps == 400 and args.save_total_limit == 1
            and getattr(args.save_strategy, "value", args.save_strategy) == "steps"
            and args.bf16 and not args.fp16,
            "Keep the 2000-step BF16 schedule, warmup20, save400 and Budget400Callback.")
    meta = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    require(meta["checkpoint_id"] == SOURCE_ID, "The fresh source must be the fixed f927 checkpoint.")
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    inventory = expected_inventory(cfg.student_checkpoint_dir, architecture, names)
    require(inventory["selected_projections"] == 72 and inventory["selected_decoder_tensors"] == 432
            and inventory["selected_decoder_elements"] == 1387008
            and inventory["frozen_tensors"] == 2991 and inventory["all_code_banks"] == 504,
            "Actual source inventory differs from the verified OV72 path.")
    return cfg, hf, args, names, architecture, inventory


class Exposure:
    """Record actual native inputs; never select, pad or rewrite examples."""
    def __init__(self, path):
        self.path, self.trainer, self.batches = Path(path), None, []

    def bind_trainer(self, trainer):
        self.trainer = trainer

    def observe_batch(self, inputs):
        record = batch_record(inputs, special_ids=self.trainer.tokenizer.all_special_ids)
        record["optimizer_step"] = int(self.trainer.state.global_step) + 1
        self.batches.append(record)

    def close(self):
        write_json(self.path, {"sampling": "unchanged_native_stream", "batches": self.batches})


def train(argv=None):
    cfg, hf, args, names, _architecture, inventory = parse_configuration(argv)
    output = Path(cfg.run_root_dir)
    require(output.is_dir() and not (output / "training.json").exists(), "Use a fresh prepared output.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Bind one assigned GPU.")
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    exposure = Exposure(output / "exposure.json")
    report = {"status": "running", "expected_inventory": inventory, "selected_targets": list(names),
              "data_registration": cfg.data.data_registration}
    try:
        with registered_preset(cfg.data), observe_native_trainer(exposure):
            result = run(cfg, hf, args, extra_callbacks=(Budget400Callback(),))
        require(result["global_step"] == 400 and result["round_base_checkpoint_id"] == SOURCE_ID,
                "The native run must complete 400 real updates from f927.")
        run_dir = Path(result["run_output_dir"])
        registration = cfg.data.data_registration
        snapshot = json.loads((run_dir / "normalized_e2e_runtime_args.json").read_text())
        require(snapshot["canonical_config"]["data"]["data_registration"] == registration,
                "Native normalized configuration lost the data identity.")
        for relative in ("trainer_state/checkpoint-400/checkpoint_meta.json", "final_model/checkpoint_meta.json"):
            meta = json.loads((run_dir / relative).read_text())
            require(meta["immutable_resume_contract"]["data"]["data_registration"] == registration,
                    "Saved checkpoint data identity changed.")
        sources = json.loads((run_dir / "run_meta.json").read_text())["data_sources"]
        require(len(sources) == 1 and sources[0]["alias"] == "longalpaca"
                and sources[0]["path"] == registration["path"] and sources[0]["raw_rows"] == 12000
                and sources[0]["is_iterable"], "Native training did not use the full original dataset.")
        require(len(exposure.batches) == 400, "Missing native input exposure records.")
        report.update(status="trained", training_result=result)
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

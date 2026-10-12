"""Validate this probe's real parser, canonical tokenizer and native data identity."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from transformers import TrainerControl, TrainerState

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, validate_e2e_immutable_resume_contract
from e2e_common import data as data_module
from experiments.mmlu_sft_probe.data import ALIAS, registered_preset, sha256
from experiments.mmlu_sft_probe.train import parse_configuration
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.distill_data import _LazySinglePresetIterable, build_distill_data_collator, build_distill_dataset, encode_canonical_record


def verify(launch_path: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=''.")
    launch = json.loads(launch_path.read_text())
    cfg, hf, training = parse_configuration(launch["argv"][3:])
    reference = json.loads(Path(launch["comparison"]["reference_launch"]).read_text())
    ref, ref_hf, ref_training = parse_args(reference["argv"][3:])
    actual_data = asdict(cfg.data)
    registration = actual_data.pop("data_registration")
    assert actual_data == asdict(ref.data) and asdict(cfg.loss) == asdict(ref.loss)
    assert asdict(cfg.opt) == asdict(ref.opt) and cfg.train_mode == ref.train_mode == "decoder"
    assert cfg.student_checkpoint_dir == ref.student_checkpoint_dir
    assert cfg.runtime.teacher_output_offload == ref.runtime.teacher_output_offload
    assert training.max_steps == 2000 and training.save_steps == 400
    assert training.get_warmup_steps(2000) == 20 and training.per_device_train_batch_size == 4
    assert training.gradient_accumulation_steps == 1 and cfg.opt.decoder_lr == 1e-5
    callback = Budget400Callback()
    callback.on_train_begin(training, TrainerState(), TrainerControl())
    control = callback.on_step_end(training, TrainerState(global_step=400), TrainerControl())
    assert control.should_save and control.should_training_stop
    metadata = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    architecture = model_inventory(metadata["base_model_path"])
    validate_recipe(metadata, architecture)
    assert selected_names(cfg.target_layers, cfg.target_modules, architecture) == ("model.layers.0.mlp.down_proj",)
    tokenizer = data_module.build_tokenizer(metadata["base_model_path"])
    old = data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS]
    with registered_preset(cfg.data):
        bundle = build_distill_dataset(cfg.data, tokenizer)
        assert isinstance(bundle.train_dataset, _LazySinglePresetIterable)
        assert bundle.is_iterable and not bundle.group_by_length and len(bundle.source_stats) == 1
        assert bundle.source_stats[0]["path"] == registration["path"]
        assert bundle.source_stats[0]["raw_rows"] == 99842
        raw = bundle.train_dataset.raw_dataset
        iterator = iter(bundle.train_dataset)
        samples, rows, skipped = [], [], 0
        # A fixed small prefix checks actual source/labels, with the unchanged
        # encoder's invalid/truncated-response skip behavior, not a new filter.
        for index in range(32):
            record = dict(raw[index])
            sample = encode_canonical_record(record, tokenizer, text_format="edgerazor_messages",
                                            text_field="messages", task="sft", model_max_length=1024)
            if sample is None:
                skipped += 1
                continue
            actual = next(iterator)
            assert all(torch.equal(actual[key], sample[key]) for key in sample)
            response = sample["labels"] != -100
            assert bool(response.any()) and bool((~response).any())
            assert torch.equal(sample["labels"][response], sample["input_ids"][response])
            assert sample["input_ids"][-1].item() == tokenizer.eos_token_id and response[-1].item()
            answer = record["messages"][1]["content"].strip()
            supervised = tokenizer.decode(sample["labels"][response].tolist(), skip_special_tokens=False)
            assert answer in supervised and "<think>" in supervised and "</think>" in supervised
            rows.append({"permuted_index": index, "length": len(sample["input_ids"]),
                         "prompt_tokens": int((~response).sum()), "response_tokens": int(response.sum()),
                         "supervised_ids": sample["labels"][response].tolist(), "supervised_text": supervised})
            samples.append(sample)
            if len(samples) == 8:
                break
        assert len(samples) == 8
        collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
        batches = [collator(samples[start:start + 4]) for start in (0, 4)]
        for batch in batches:
            assert batch["input_ids"].shape[0] == 4 and batch["input_ids"].shape[1] <= 1024
            assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS] is old
    payload = _build_e2e_runtime_parameter_payload(cfg=cfg, hf_args=hf, training_args=training,
                                                   run_output_dir=str(launch_path.parent))
    contract = build_e2e_immutable_resume_contract(
        cfg=cfg, training_args=training, tokenizer=tokenizer, input_checkpoint_id=metadata["checkpoint_id"],
        resolved_target_layers=[0], resolved_target_modules=["down_proj"], teacher_identity=None)
    assert payload["canonical_config"]["data"]["data_registration"] == contract["data"]["data_registration"] == registration
    validate_e2e_immutable_resume_contract(contract, deepcopy(contract))
    changed = deepcopy(contract)
    changed["data"]["data_registration"]["sha256"] = "0" * 64
    try:
        validate_e2e_immutable_resume_contract(contract, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Native immutable contract ignored changed data identity.")
    assert not torch.cuda.is_initialized()
    return {"status": "passed", "actual_parser": True, "cuda_initialized": False,
            "source_manifest_sha256": sha256(registration["manifest_path"]),
            "registered_data": registration, "normalized_payload": payload,
            "dataset_class": type(bundle.train_dataset).__name__, "source_stats": bundle.source_stats,
            "actual_group_by_length": bundle.group_by_length, "raw_rows_checked": len(rows) + skipped,
            "skipped_by_existing_encoder": skipped, "checked_examples": rows,
            "batch_shapes": [list(batch["input_ids"].shape) for batch in batches],
            "registry_restored": True, "native_contract_carries_identity": True,
            "changed_identity_rejected": True,
            "scope": "Real tokenizer/data/parser and serialization only; no model load, training or GPU. Native 16-worker sharding is unchanged, not replayed here."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.launch)
    path = args.launch.parent / "cpu_validation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "dataset_class", "batch_shapes", "cuda_initialized")}))


if __name__ == "__main__":
    main()

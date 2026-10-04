"""CPU contracts for the expanded native decoder-only resource diagnostic."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.qwen3_w25.validate_e2e_subset_export import check_inputs
from experiments.w25_eval.validate_decoder_subset_training import configuration


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result/qwen3_8b_mmlu65_20261004"


@pytest.fixture(params=(128, 256), scope="module")
def prepared(request):
    path = RESULT / f"best1200_mlp_h{request.param}_decoder_validate_s4/launch.json"
    if not path.is_file():
        pytest.skip("Prepared scope diagnostic is not installed.")
    record = json.loads(path.read_text())
    if not (Path(record["source_checkpoint"]) / "pytorch_model.bin").is_file():
        pytest.skip("Pinned full-width initial checkpoint is not installed.")
    return request.param, record


def test_real_full_mlp_configuration_inventory_and_fresh_source(prepared):
    width, record = prepared
    check_inputs(record)
    cfg, _, args, names, architecture = configuration(record)
    assert not torch.cuda.is_initialized()
    assert record["gpu"] is record["export_gpu"] is None and not record["queued"]
    assert cfg.train_mode == "decoder" and cfg.target_layers == "all"
    assert len(names) == 3 * architecture["num_hidden_layers"] == 108
    assert record["expected_inventory"] == {
        "deployed_projections": 252, "serialized_tensors": 3423,
        "all_decoder_tensors": 1512, "all_code_banks": 504,
        "selected_projections": 108, "selected_decoder_tensors": 648,
        "selected_code_banks": 0, "active_score_elements": 0,
        "frozen_tensors": 2775, "selected_projection_code_banks": 216,
        "selected_decoder_elements": {128: 2006784, 256: 4006656}[width],
    }
    assert args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
    assert args.save_steps == 2 and args.per_device_train_batch_size == 4
    assert cfg.data.model_max_length == 1024 and cfg.loss.loss_type == "kl_top_mass"
    assert cfg.loss.prompt_loss_weight == .3 and cfg.opt.decoder_lr == 3e-6
    meta = json.loads((Path(record["source_checkpoint"]) / "checkpoint_meta.json").read_text())
    assert {stage["hidden_dim"] for spec in meta["converted_modules"]
            if spec["name"] in names for stage in spec["stage_decoders"]} == {width}


def test_stale_single_projection_counts_are_rejected(prepared):
    _, record = prepared
    bad = deepcopy(record)
    bad["expected_inventory"]["selected_decoder_tensors"] = 6
    bad["expected_inventory"]["frozen_tensors"] = 3417
    with pytest.raises(ValueError, match="inventory differs"):
        configuration(bad)

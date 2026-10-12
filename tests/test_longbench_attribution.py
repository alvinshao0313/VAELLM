"""CPU selection and genuine VAELinear-to-HF tensor restoration contracts."""
import json

import pytest
import torch
from safetensors.torch import save_file

from experiments.longbench_attribution.protocol import TASKS, select_rows
from experiments.longbench_attribution.restore import frozen_inventory, restore_targets
from test_qwen3_w25_e2e_aux_export import _model, TARGETS


def test_selection_uses_only_input_geometry_and_stable_ties():
    rows = [{"dataset": task, "row_index": i, "input_tokens": 100 + i // 2,
             "answers": [str(199 - i)]} for task in TASKS for i in reversed(range(200))]
    chosen = select_rows(rows)
    assert [row["row_index"] for row in chosen] == [i * 199 // 7 for i in range(8)] * 2
    for row in rows:
        row["answers"] = ["changed label has no role in sampling"]
        row["prediction"] = "arbitrary"
    assert [row["row_index"] for row in select_rows(rows)] == [row["row_index"] for row in chosen]
    with pytest.raises(ValueError, match="200-row"):
        select_rows(rows[:-1])


def test_real_vae_restoration_preserves_other_tensors_and_uses_exact_bf16(tmp_path):
    model = _model().eval().requires_grad_(False)
    selected = tuple(name for name in TARGETS if name.endswith(("q_proj", "k_proj")))
    original = frozen_inventory(model, selected)
    values = {name + ".weight": torch.randn(model.get_submodule(name).out_features,
        model.get_submodule(name).in_features, dtype=torch.bfloat16) for name in selected}
    save_file(values, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in values}}))
    report = restore_targets(model, tmp_path, selected)
    assert report["selected_count"] == 2 and report["non_target_state_and_identity_exact"]
    assert frozen_inventory(model, selected) == original
    for name in selected:
        module = model.get_submodule(name)
        assert type(module) is torch.nn.Linear and not module.weight.requires_grad and not module.training
        assert torch.equal(module.weight, values[name + ".weight"])
        assert module.bias is None
    with pytest.raises(ValueError, match="original compressed"):
        restore_targets(model, tmp_path, selected)
    with torch.no_grad():
        model.model.norm.weight[0].add_(1)
    assert frozen_inventory(model, selected) != original
    assert not torch.cuda.is_initialized()

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_after_category_common import (
    ResolvedCatAfterCategoryStage,
    run_canonical_remaining_lora_prefix_decoder,
)
from train_utils.cat_after_category_distill import run_after_category_distill
from train_utils.cat_checkpoint_v6 import save_cat_v6_full_checkpoint
from train_utils.cat_data_prep import LinearSplitMeta
from train_utils.cat_train_pipeline import _build_vae_linear_from_stage_payload, apply_group_vae_payload
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
from train_utils.config.configs import (
    AfterCategoryResolvedConfig,
    AuxTrainableConfig,
    DistillDataConfig,
    DistillLossConfig,
    DistillOptimizationConfig,
    DistillRuntimeConfig,
    LoRAConfig,
)
from train_utils.distill_decoder import (
    NamedMainDecoderTarget,
    enable_main_decoder_targets,
    finalize_main_decoder_targets,
)
from train_utils.utils import LinearRef


NAME = "model.layers.0.down_proj"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        block = nn.Module()
        block.down_proj = nn.Linear(4, 4, bias=False)
        self.model.layers = nn.ModuleList([block])

    def forward(self, x):
        return self.model.layers[0].down_proj(x)


def _inputs(*, stages=2, parts=1, decoder_type="symmetric", dims=None):
    torch.manual_seed(731)
    dims = dims or [4] * stages
    split_meta = LinearSplitMeta(
        linear_name=NAME, transpose=False, sort_mode="none",
        parallel_rows=1, parallel_cols=parts,
        restore_row_indices=None, restore_col_indices=None, part_restore_col_indices=None,
        compressed_in_features=4, compressed_out_features=4,
        protected_input_indices=None, protected_input_weight=None,
        protected_input_qvalues=None, protected_input_scales=None,
        protected_output_indices=None, protected_output_weight=None,
        protected_output_qvalues=None, protected_output_scales=None,
    )
    bits, decoders = [], []
    for dim in dims:
        # Use the real conversion representation: ordinary Linear after extraction.
        grouped = Decoder(
            in_dim=64, out_dim=dim, hidden_dim=8, num_res_blocks=1,
            norm_type="layer", decoder_type=decoder_type, num_models=max(2, parts),
        )
        stage_decoders = [grouped.extract_single(i) for i in range(parts)]
        for decoder in stage_decoders:
            decoder._fuse_q_scale()
            first_linear = decoder.linear if decoder_type == "linear" else decoder.linear_in
            assert isinstance(first_linear, nn.Linear)
        stage_bits = [torch.randint(0, 2, (16 // parts // dim, 1, 64), dtype=torch.bool)
                      for _ in range(parts)]
        bits.append(stage_bits[0] if parts == 1 else stage_bits)
        decoders.append(stage_decoders[0] if parts == 1 else stage_decoders)
    return dict(
        old_module=nn.Linear(4, 4, bias=False), transpose=False,
        split_meta=split_meta, stage_split_metas=[split_meta] * stages,
        stage_part_bits_payload=bits, stage_part_decoders_payload=decoders,
        stage_codebook_dims=dims, parallel_rows=1, parallel_cols=parts,
        parallel_parts=parts, bias=None,
    )


def _serial(inputs):
    return VAELinear(
        in_features=4, out_features=4, bias=None, original_weight=None,
        vq_weight=None, decoder=None,
        stage_vq_weights=inputs["stage_part_bits_payload"],
        stage_decoders=inputs["stage_part_decoders_payload"],
        codebook_dim=inputs["stage_codebook_dims"][0],
        stage_codebook_dims=inputs["stage_codebook_dims"],
        transpose=False, parallel_parts=inputs["parallel_parts"],
        parallel_rows=1, parallel_cols=inputs["parallel_parts"],
    )


def _save(model, output, kind):
    return save_cat_v6_full_checkpoint(
        model, str(output), checkpoint_kind=kind,
        category=None if kind == "final_model" else "down_proj",
        completed_categories=() if kind == "round_base" else ("down_proj",),
        compression_categories=("down_proj",),
        cat_args=SimpleNamespace(after_category_mode="none", target_layers="all", skip_layers=""),
        vae_args=SimpleNamespace(model_path="tiny"), training_args=SimpleNamespace(),
        base_model_path="tiny",
    )


def _update_decoder(layer):
    target = NamedMainDecoderTarget(NAME, layer)
    before_ids = tuple(id(p) for p in layer._parallel_stage_decoder.parameters())
    parameters = enable_main_decoder_targets([target])
    assert tuple(id(p) for p in parameters) == before_ids
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    before = [p.detach().clone() for p in parameters]
    loss = layer(torch.eye(4, device=parameters[0].device)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step()
    assert any(not torch.equal(old, p) for old, p in zip(before, parameters))
    finalize_main_decoder_targets([target])
    assert layer.parallel_stage_decode and not layer.trainable_decode


@pytest.mark.parametrize("kind", ["round_base", "category_boundary", "final_model"])
@pytest.mark.parametrize("decoder_type", ["linear", "symmetric"])
@pytest.mark.parametrize("stages,parts", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_cat_conversion_packed_roundtrip(tmp_path, kind, decoder_type, stages, parts):
    inputs = _inputs(stages=stages, parts=parts, decoder_type=decoder_type)
    serial = _serial(inputs)
    expected_weight = serial._decode_weight(dtype=torch.float32).detach()
    x = torch.randn(3, 4)
    expected = serial(x).detach()
    layer = _build_vae_linear_from_stage_payload(**inputs)
    assert layer.parallel_stage_decode and not layer.trainable_decode
    assert layer._parallel_stage_decoder.num_models == stages * parts
    assert not any(p.requires_grad for p in layer._parallel_stage_decoder.parameters())
    torch.testing.assert_close(layer._decode_weight(dtype=torch.float32), expected_weight)
    torch.testing.assert_close(layer(x), expected)
    model = _Model()
    model.model.layers[0].down_proj = layer
    before_ids = tuple(id(p) for p in layer.parameters())
    state = copy.deepcopy(model.state_dict())
    decoder_keys = [key for key in state if "decoder" in key]
    assert decoder_keys and all("._parallel_stage_decoder." in key for key in decoder_keys)
    assert any((".linear.weight" if stages * parts == 1 else ".conv.weight") in key
               for key in decoder_keys)
    output = tmp_path / kind
    saved = _save(model, output, kind)
    assert saved["meta_payload"]["converted_modules"][0]["parallel_stage_decode"] is True
    assert tuple(id(p) for p in layer.parameters()) == before_ids
    loaded, _, result = load_v6_full_checkpoint_into_model(_Model(), str(output), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert state.keys() == loaded.state_dict().keys()
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, state[key]), key
    torch.testing.assert_close(loaded(x), expected)
    _update_decoder(loaded.model.layers[0].down_proj)


@pytest.mark.parametrize("recovery", ["none", "zero", "trained"])
def test_cat_packing_does_not_depend_on_recovery(tmp_path, recovery):
    model = _Model()
    layer = _build_vae_linear_from_stage_payload(**_inputs())
    model.model.layers[0].down_proj = layer
    packed = layer._parallel_stage_decoder
    assert layer.parallel_stage_decode
    if recovery == "none":
        result = run_after_category_distill(
            model=model, category="down_proj", cat_args=SimpleNamespace(after_category_mode="none"),
            vae_args=SimpleNamespace(), training_args=SimpleNamespace(),
            logger=logging.getLogger(__name__), lora_round_idx=0,
            transpose_modules=(), only_decoder_projections=True,
            compression_categories=("down_proj",), online_cat=True,
        )
        assert not result.did_train
    elif recovery == "zero":
        config = AfterCategoryResolvedConfig(
            data=DistillDataConfig(), loss=DistillLossConfig(),
            opt=DistillOptimizationConfig(steps=0), lora=LoRAConfig(),
            aux=AuxTrainableConfig(), runtime=DistillRuntimeConfig(),
        )
        stage = ResolvedCatAfterCategoryStage(
            mode="remaining_lora_prefix_decoder", config=config, train_device="cpu",
            base_seed=42, stage_seed=42, output_dir=str(tmp_path),
            deterministic=True, fp16=False, bf16=False, reset_completed=False,
        )
        result = run_canonical_remaining_lora_prefix_decoder(
            model=model, category="down_proj", compression_categories=("down_proj",),
            newly_compressed_target_count=1, stage=stage,
            vae_args=SimpleNamespace(), logger=logging.getLogger(__name__), skip_layers="",
        )
        assert not result.did_train and result.decoder_target_count == 1
    else:
        _update_decoder(layer)
    assert layer._parallel_stage_decoder is packed and layer.parallel_stage_decode
    _save(model, tmp_path / "final_model", "final_model")
    loaded, _, _ = load_v6_full_checkpoint_into_model(_Model(), str(tmp_path / "final_model"))
    torch.testing.assert_close(loaded(torch.eye(4)), model(torch.eye(4)))


@pytest.mark.parametrize("mismatch", ["dims", "architecture", "vq_shape"])
def test_cat_conversion_rejects_incompatible_stages_with_module_name(mismatch):
    inputs = _inputs(dims=[4, 8] if mismatch == "dims" else None)
    if mismatch == "architecture":
        inputs["stage_part_decoders_payload"][1].activation_type = "relu"
    elif mismatch == "vq_shape":
        inputs["stage_part_bits_payload"][1] = torch.zeros(4, 1, 32, dtype=torch.bool)
    expected = {"dims": "identical stage codebook dims", "architecture": "config mismatch",
                "vq_shape": "identical per-stage packed VQ shapes"}[mismatch]
    with pytest.raises(ValueError, match=rf"{NAME}.*{expected}"):
        _build_vae_linear_from_stage_payload(**inputs)


@pytest.mark.parametrize("compatible", [True, False])
def test_cat_payload_application_packs_before_installing(compatible):
    inputs = _inputs(dims=[4, 4] if compatible else [4, 8])
    model = _Model()
    original = model.model.layers[0].down_proj
    meta = inputs["split_meta"]
    payload = dict(
        format="vaellm_group_vae_payload", version=1,
        target_common_split_metas=[meta], parts_per_linear=1, row_parts=1, col_parts=1,
        residual_stages=2, all_stage_bits=inputs["stage_part_bits_payload"],
        all_stage_decoders=[[decoder] for decoder in inputs["stage_part_decoders_payload"]],
        all_stage_codebook_dims=inputs["stage_codebook_dims"],
        all_stage_split_metas=[[meta], [meta]],
    )
    kwargs = dict(
        model=model, group_refs=[LinearRef(NAME, original, "down_proj", False)],
        group_tag="down_proj", payload=payload, convert_device="cpu",
    )
    if compatible:
        apply_group_vae_payload(**kwargs)
        layer = model.model.layers[0].down_proj
        assert layer is not original and layer.parallel_stage_decode
        assert layer._parallel_stage_decoder.num_models == 2
    else:
        with pytest.raises(ValueError, match=NAME):
            apply_group_vae_payload(**kwargs)
        assert model.model.layers[0].down_proj is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("decoder_type", ["linear", "symmetric"])
def test_cat_packed_cuda_decode_and_update(stages, decoder_type):
    inputs = _inputs(stages=stages, decoder_type=decoder_type)
    serial = _serial(copy.deepcopy(inputs)).to("cuda")
    layer = _build_vae_linear_from_stage_payload(**inputs).to("cuda")
    x = torch.eye(4, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(layer(x), serial(x))
    _update_decoder(layer)


@pytest.mark.parametrize("kind", ["round_base", "category_boundary", "final_model"])
@pytest.mark.parametrize("invalid", ["serial", "flag", "duplicate", "alias", "count"])
def test_cat_save_rejects_invalid_topology_without_mutation(tmp_path, kind, invalid):
    model = _Model()
    inputs = _inputs()
    layer = _serial(inputs) if invalid == "serial" else _build_vae_linear_from_stage_payload(**inputs)
    model.model.layers[0].down_proj = layer
    if invalid == "flag":
        layer.parallel_stage_decode = False
    elif invalid == "duplicate":
        layer.decoder = inputs["stage_part_decoders_payload"][0]
    elif invalid == "alias":
        layer.decoder = layer._parallel_stage_decoder
    elif invalid == "count":
        layer._parallel_stage_decoder.num_models = 3
    parameters = tuple(id(p) for p in layer.parameters())
    state = copy.deepcopy(layer.state_dict())
    with pytest.raises(ValueError, match=NAME):
        _save(model, tmp_path / kind, kind)
    assert not (tmp_path / kind).exists()
    assert tuple(id(p) for p in layer.parameters()) == parameters
    assert state.keys() == layer.state_dict().keys()
    for key, value in layer.state_dict().items():
        assert torch.equal(value, state[key]), key

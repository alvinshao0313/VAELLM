import copy
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from e2e_common.checkpoint_io import (
    _collect_single_vae_linear_spec,
    _remap_legacy_decoder_keys_if_needed,
    load_e2e_checkpoint_into_model,
)
from litebsq.autoencoder import Decoder
from litebsq.bitpack import (
    build_bitpack_u8_spec,
    pack_bool_tensor_to_uint8,
    unpack_uint8_tensor_to_bool,
)
from litebsq.vae_linear import VAELinear
from tools.convert_cat_checkpoint_to_bitpack import convert_checkpoint
from e2e_common.post_norm_head import LMHeadWithPostNormLinear, ensure_post_norm_head_linear
from train_utils.model_checkpoint_io import (
    _build_distributed_run_output_dir,
    _collect_vae_linear_specs,
    _decoder_to_spec,
    _refresh_vae_linear_runtime_after_state_load,
    load_checkpoint_into_model,
    save_model_checkpoint,
)


_DISABLED_SORT_RESTORE_FIELDS = (
    "restore_row_indices",
    "restore_col_indices",
    "part_restore_col_indices",
    "stage_restore_row_indices",
    "stage_restore_col_indices",
    "stage_part_restore_col_indices",
)


class _DummyDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.linear_in = nn.Module()
        self.decoder.linear_in.linear = nn.Linear(4, 4, bias=True)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyLmHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(3, 5, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


def _make_decoder(latent_dim: int, codebook_dim: int) -> Decoder:
    decoder = Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    decoder = decoder.to(dtype=torch.float32)
    with torch.no_grad():
        for idx, param in enumerate(decoder.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values + float(idx + 1))
    return decoder


def _build_single_stage_vae_linear() -> tuple[VAELinear, torch.Tensor, Decoder]:
    latent_dim = 9
    codebook_dim = 4
    bits = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    decoder = _make_decoder(latent_dim=latent_dim, codebook_dim=codebook_dim)
    layer = VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=copy.deepcopy(decoder),
        codebook_dim=codebook_dim,
        transpose=False,
    )
    return layer, bits, decoder


def _build_two_stage_parallel_vae_linear() -> VAELinear:
    latent_dim = 9
    codebook_dim = 4
    part0 = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
        ],
        dtype=torch.bool,
    )
    part1 = torch.tensor(
        [
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    stage0 = [part0, part1]
    stage1 = [~part0, ~part1]
    stage_decoders = [
        [_make_decoder(latent_dim, codebook_dim), _make_decoder(latent_dim, codebook_dim)],
        [_make_decoder(latent_dim, codebook_dim), _make_decoder(latent_dim, codebook_dim)],
    ]
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[stage0, stage1],
        stage_decoders=stage_decoders,
        codebook_dim=codebook_dim,
        stage_codebook_dims=[codebook_dim, codebook_dim],
        transpose=False,
        parallel_parts=2,
        parallel_rows=1,
        parallel_cols=2,
    )


class LegacyCheckpointRemapTest(unittest.TestCase):
    def test_remap_legacy_decoder_linear_keys(self):
        model = _DummyDecoderModel()
        legacy_state = {
            "decoder.linear_in.weight": torch.randn(4, 4),
            "decoder.linear_in.bias": torch.randn(4),
        }
        remapped = _remap_legacy_decoder_keys_if_needed(model, legacy_state)
        self.assertNotIn("decoder.linear_in.weight", remapped)
        self.assertNotIn("decoder.linear_in.bias", remapped)
        self.assertIn("decoder.linear_in.linear.weight", remapped)
        self.assertIn("decoder.linear_in.linear.bias", remapped)

    def test_keep_new_decoder_linear_keys(self):
        model = _DummyDecoderModel()
        new_style_state = {
            "decoder.linear_in.linear.weight": torch.randn(4, 4),
            "decoder.linear_in.linear.bias": torch.randn(4),
        }
        remapped = _remap_legacy_decoder_keys_if_needed(model, new_style_state)
        self.assertIn("decoder.linear_in.linear.weight", remapped)
        self.assertIn("decoder.linear_in.linear.bias", remapped)
        self.assertEqual(set(remapped.keys()), set(new_style_state.keys()))


class DistributedRunOutputDirTest(unittest.TestCase):
    def test_nonzero_rank_uses_broadcast_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_run_dir = os.path.join(tmpdir, "shared_run")
            os.makedirs(shared_run_dir)

            def _broadcast(payload, src):
                self.assertEqual(src, 0)
                payload[0] = shared_run_dir

            with mock.patch("train_utils.model_checkpoint_io.torch.distributed.is_available", return_value=True), mock.patch(
                "train_utils.model_checkpoint_io.torch.distributed.is_initialized", return_value=True
            ), mock.patch(
                "train_utils.model_checkpoint_io.torch.distributed.get_world_size", return_value=2
            ), mock.patch(
                "train_utils.model_checkpoint_io.torch.distributed.get_rank", return_value=1
            ), mock.patch(
                "train_utils.model_checkpoint_io.torch.distributed.broadcast_object_list", side_effect=_broadcast
            ), mock.patch(
                "train_utils.model_checkpoint_io._build_run_output_dir", side_effect=AssertionError("rank1 must not create run dir")
            ):
                got = _build_distributed_run_output_dir(tmpdir, "model")

        self.assertEqual(got, shared_run_dir)


class PostNormHeadCheckpointTest(unittest.TestCase):
    def test_post_norm_head_wrapper_round_trip(self):
        model = _TinyLmHeadModel()
        ensure_post_norm_head_linear(model)
        with torch.no_grad():
            model.lm_head.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).view(5, 3) / 10.0)
            model.lm_head.post_norm_linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.2, 0.0],
                        [0.0, 0.7, 0.1],
                        [0.3, 0.0, 1.2],
                    ],
                    dtype=torch.float32,
                )
            )
        inputs = torch.randn(2, 4, 3)
        expected_logits = model(inputs).detach().clone()
        expected_base = model.lm_head.lm_head.weight.detach().clone()
        expected_post = model.lm_head.post_norm_linear.weight.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            with open(os.path.join(tmpdir, "checkpoint_meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertIs(meta["post_norm_head_linear"], True)

            restored_model = _TinyLmHeadModel()
            restored_model, _meta, _ = load_checkpoint_into_model(restored_model, tmpdir)

        self.assertIsInstance(restored_model.lm_head, LMHeadWithPostNormLinear)
        self.assertTrue(torch.equal(restored_model.lm_head.lm_head.weight, expected_base))
        self.assertTrue(torch.equal(restored_model.lm_head.post_norm_linear.weight, expected_post))
        self.assertTrue(torch.allclose(restored_model(inputs), expected_logits))

    def test_save_model_checkpoint_does_not_mutate_live_post_norm_head(self):
        model = _TinyLmHeadModel()
        ensure_post_norm_head_linear(model)
        lm_head_id = id(model.lm_head)
        base_weight = model.lm_head.lm_head.weight.detach().clone()
        post_weight = model.lm_head.post_norm_linear.weight.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)

        self.assertIsInstance(model.lm_head, LMHeadWithPostNormLinear)
        self.assertEqual(id(model.lm_head), lm_head_id)
        self.assertTrue(torch.equal(model.lm_head.lm_head.weight, base_weight))
        self.assertTrue(torch.equal(model.lm_head.post_norm_linear.weight, post_weight))


class BitpackUtilityTest(unittest.TestCase):
    def test_roundtrip_keeps_bits_for_multiple_sizes(self):
        for latent_dim in (8, 32, 33):
            bits = torch.arange(latent_dim * 2, dtype=torch.int64).view(2, latent_dim).remainder(3).eq(0)
            packed = pack_bool_tensor_to_uint8(bits, logical_shape=bits.shape)
            unpacked = unpack_uint8_tensor_to_bool(packed, logical_shape=bits.shape)
            self.assertTrue(torch.equal(bits, unpacked))


class VAELinearBitpackTest(unittest.TestCase):
    def test_decode_matches_between_logical_bool_and_packed_storage(self):
        latent_dim = 9
        codebook_dim = 4
        part0 = torch.tensor(
            [
                [[True, False, True, False, True, False, True, False, True]],
                [[False, True, False, True, False, True, False, True, False]],
            ],
            dtype=torch.bool,
        )
        part1 = torch.tensor(
            [
                [[True, True, False, False, True, True, False, False, True]],
                [[False, False, True, True, False, False, True, True, False]],
            ],
            dtype=torch.bool,
        )
        stage0 = [part0, part1]
        stage1 = [~part0, ~part1]
        decoder_stage0 = [_make_decoder(latent_dim, codebook_dim), _make_decoder(latent_dim, codebook_dim)]
        decoder_stage1 = [_make_decoder(latent_dim, codebook_dim), _make_decoder(latent_dim, codebook_dim)]

        logical_layer = VAELinear(
            in_features=4,
            out_features=4,
            bias=None,
            original_weight=None,
            vq_weight=None,
            decoder=None,
            stage_vq_weights=[stage0, stage1],
            stage_decoders=[[copy.deepcopy(decoder_stage0[0]), copy.deepcopy(decoder_stage0[1])], [copy.deepcopy(decoder_stage1[0]), copy.deepcopy(decoder_stage1[1])]],
            codebook_dim=codebook_dim,
            stage_codebook_dims=[codebook_dim, codebook_dim],
            transpose=False,
            parallel_parts=2,
            parallel_rows=1,
            parallel_cols=2,
        )
        packed_layer = VAELinear(
            in_features=4,
            out_features=4,
            bias=None,
            original_weight=None,
            vq_weight=None,
            decoder=None,
            stage_vq_weights=[
                [pack_bool_tensor_to_uint8(part0, logical_shape=part0.shape), pack_bool_tensor_to_uint8(part1, logical_shape=part1.shape)],
                [pack_bool_tensor_to_uint8(~part0, logical_shape=(~part0).shape), pack_bool_tensor_to_uint8(~part1, logical_shape=(~part1).shape)],
            ],
            stage_vq_storage_specs=[
                [build_bitpack_u8_spec(logical_shape=part0.shape), build_bitpack_u8_spec(logical_shape=part1.shape)],
                [build_bitpack_u8_spec(logical_shape=(~part0).shape), build_bitpack_u8_spec(logical_shape=(~part1).shape)],
            ],
            stage_decoders=[[copy.deepcopy(decoder_stage0[0]), copy.deepcopy(decoder_stage0[1])], [copy.deepcopy(decoder_stage1[0]), copy.deepcopy(decoder_stage1[1])]],
            codebook_dim=codebook_dim,
            stage_codebook_dims=[codebook_dim, codebook_dim],
            transpose=False,
            parallel_parts=2,
            parallel_rows=1,
            parallel_cols=2,
        )
        logical_decoded = logical_layer._decode_weight(dtype=torch.float32)
        packed_decoded = packed_layer._decode_weight(dtype=torch.float32)
        self.assertTrue(torch.allclose(logical_decoded, packed_decoded))


class CheckpointBitpackIOTest(unittest.TestCase):
    def test_new_checkpoint_specs_omit_disabled_sort_restore_fields(self):
        layer, _, _ = _build_single_stage_vae_linear()
        model = _DummyModel()
        model.proj = layer

        cat_spec = _collect_vae_linear_specs(model)[0]
        e2e_spec = _collect_single_vae_linear_spec("proj", layer)

        for field_name in _DISABLED_SORT_RESTORE_FIELDS:
            self.assertNotIn(field_name, cat_spec)
            self.assertNotIn(field_name, e2e_spec)

    def test_save_and_load_checkpoint_uses_packed_uint8(self):
        layer, _, _ = _build_single_stage_vae_linear()
        model = _DummyModel()
        model.proj = layer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            state_dict = torch.load(os.path.join(tmpdir, "pytorch_model.bin"), map_location="cpu")
            self.assertEqual(state_dict["proj.vq_weight"].dtype, torch.uint8)

            restored_model = _DummyModel()
            restored_model, meta, _ = load_checkpoint_into_model(restored_model, tmpdir)
            self.assertEqual(meta["version"], 5)
            self.assertIsInstance(restored_model.proj, VAELinear)
            self.assertTrue(torch.allclose(model.proj._decode_weight(dtype=torch.float32), restored_model.proj._decode_weight(dtype=torch.float32)))

    def test_mainline_vae_checkpoint_does_not_depend_on_original_weight(self):
        layer, _, _ = _build_single_stage_vae_linear()
        model = _DummyModel()
        model.proj = layer
        inputs = torch.randn(3, 4)
        expected = model(inputs).detach()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            state_dict = torch.load(os.path.join(tmpdir, "pytorch_model.bin"), map_location="cpu")
            self.assertFalse(any(key.endswith(".original_weight") for key in state_dict))
            with open(os.path.join(tmpdir, "checkpoint_meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["converted_module_count"], 1)
            spec = meta["converted_modules"][0]
            self.assertIs(spec["has_original_weight"], False)
            self.assertIs(spec["always_use_original"], False)
            self.assertIs(spec["protect_original_weight"], False)

            restored_model = _DummyModel()
            restored_model, _meta, _ = load_checkpoint_into_model(restored_model, tmpdir)

        self.assertIsInstance(restored_model.proj, VAELinear)
        self.assertIsNone(restored_model.proj.original_weight)
        self.assertIs(restored_model.proj.always_use_original, False)
        self.assertIs(restored_model.proj.protect_original_weight, False)
        self.assertTrue(torch.allclose(restored_model(inputs), expected))

    def test_load_old_checkpoint_rejects_with_conversion_message(self):
        layer, bits, decoder = _build_single_stage_vae_linear()
        model = _DummyModel()
        model.proj = layer
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_state = model.state_dict()
            legacy_state["proj.vq_weight"] = bits.clone()
            torch.save(legacy_state, os.path.join(tmpdir, "pytorch_model.bin"))
            legacy_meta = {
                "format": "vaellm_state_dict_with_meta",
                "version": 4,
                "base_model_path": "dummy/base",
                "state_dict_file": "pytorch_model.bin",
                "converted_module_count": 1,
                "converted_modules": [
                    {
                        "name": "proj",
                        "in_features": 4,
                        "out_features": 4,
                        "compressed_in_features": 4,
                        "compressed_out_features": 4,
                        "codebook_dim": 4,
                        "transpose": False,
                        "parallel_parts": 1,
                        "parallel_rows": 1,
                        "parallel_cols": 1,
                        "residual_stages": 1,
                        "stage_codebook_dims": [4],
                        "has_bias": False,
                        "has_original_weight": False,
                        "always_use_original": False,
                        "protect_original_weight": False,
                        "vq_weights": [{"shape": list(bits.shape), "dtype": "bool"}],
                        "decoders": [_decoder_to_spec(decoder)],
                        "stage_vq_weights": None,
                        "stage_decoders": None,
                    }
                ],
            }
            with open(os.path.join(tmpdir, "checkpoint_meta.json"), "w", encoding="utf-8") as handle:
                json.dump(legacy_meta, handle, ensure_ascii=False, indent=2)
            with self.assertRaisesRegex(ValueError, "convert_cat_checkpoint_to_bitpack.py"):
                load_checkpoint_into_model(_DummyModel(), tmpdir)

    def test_conversion_script_converts_legacy_checkpoint(self):
        layer, bits, decoder = _build_single_stage_vae_linear()
        model = _DummyModel()
        model.proj = layer
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = os.path.join(tmpdir, "legacy")
            os.makedirs(legacy_dir, exist_ok=True)
            legacy_state = model.state_dict()
            legacy_state["proj.vq_weight"] = bits.clone()
            torch.save(legacy_state, os.path.join(legacy_dir, "pytorch_model.bin"))
            with open(os.path.join(legacy_dir, "note.txt"), "w", encoding="utf-8") as handle:
                handle.write("keep me")
            legacy_meta = {
                "format": "vaellm_state_dict_with_meta",
                "version": 4,
                "base_model_path": "dummy/base",
                "state_dict_file": "pytorch_model.bin",
                "converted_module_count": 1,
                "converted_modules": [
                    {
                        "name": "proj",
                        "in_features": 4,
                        "out_features": 4,
                        "compressed_in_features": 4,
                        "compressed_out_features": 4,
                        "codebook_dim": 4,
                        "transpose": False,
                        "parallel_parts": 1,
                        "parallel_rows": 1,
                        "parallel_cols": 1,
                        "residual_stages": 1,
                        "stage_codebook_dims": [4],
                        "has_bias": False,
                        "has_original_weight": False,
                        "always_use_original": False,
                        "protect_original_weight": False,
                        "vq_weights": [{"shape": list(bits.shape), "dtype": "bool"}],
                        "decoders": [_decoder_to_spec(decoder)],
                        "stage_vq_weights": None,
                        "stage_decoders": None,
                    }
                ],
            }
            with open(os.path.join(legacy_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as handle:
                json.dump(legacy_meta, handle, ensure_ascii=False, indent=2)

            packed_dir = os.path.join(tmpdir, "packed")
            convert_checkpoint(legacy_dir, packed_dir)

            packed_meta_path = os.path.join(packed_dir, "checkpoint_meta.json")
            packed_state_path = os.path.join(packed_dir, "pytorch_model.bin")
            with open(packed_meta_path, "r", encoding="utf-8") as handle:
                packed_meta = json.load(handle)
            packed_state = torch.load(packed_state_path, map_location="cpu")
            self.assertEqual(packed_meta["version"], 5)
            self.assertEqual(
                packed_meta["converted_modules"][0]["vq_weights"][0]["storage_format"],
                "bitpack_u8",
            )
            self.assertEqual(packed_state["proj.vq_weight"].dtype, torch.uint8)
            self.assertTrue(os.path.exists(os.path.join(packed_dir, "note.txt")))

            restored_model, _, _ = load_checkpoint_into_model(_DummyModel(), packed_dir)
            self.assertTrue(torch.allclose(model.proj._decode_weight(dtype=torch.float32), restored_model.proj._decode_weight(dtype=torch.float32)))

    def test_e2e_loader_rebuilds_parallel_grouped_vq_after_state_load(self):
        source_model = _DummyModel()
        source_model.proj = _build_two_stage_parallel_vae_linear()
        source_weight = source_model.proj._decode_weight(dtype=torch.float32).detach().clone()
        x = torch.tensor(
            [
                [1.0, -2.0, 0.5, 3.0],
                [-1.0, 0.25, 2.0, -0.5],
            ],
            dtype=torch.float32,
        )
        source_output = source_model.proj(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(source_model, tmpdir, save_config=False)

            restored_model = _DummyModel()
            restored_model, meta, _ = load_e2e_checkpoint_into_model(
                restored_model,
                tmpdir,
                map_location="cpu",
                strict=True,
                materialize_proxy_decoded_linears=False,
            )

            restored = restored_model.proj
            self.assertIsInstance(restored, VAELinear)
            self.assertTrue(bool(meta["converted_modules"][0]["parallel_stage_decode"]))
            self.assertIsNotNone(restored._parallel_stage_decoder)

            layout = list(restored._parallel_stage_layout)
            expected_grouped_vq = restored._build_parallel_stage_grouped_vq_weight(layout)
            actual_grouped_vq = restored._parallel_stage_grouped_vq_weight
            self.assertTrue(torch.equal(actual_grouped_vq, expected_grouped_vq))

            restored_weight = restored._decode_weight(dtype=torch.float32)
            restored_output = restored(x)
            self.assertTrue(torch.allclose(restored_weight, source_weight, rtol=0.0, atol=1e-6))
            self.assertTrue(torch.allclose(restored_output, source_output, rtol=0.0, atol=1e-5))


class RuntimeRefreshContractTest(unittest.TestCase):
    def test_refresh_rebuilds_main_and_protected_parallel_plans_before_clearing_cache(self):
        layer = _build_two_stage_parallel_vae_linear()
        model = _DummyModel()
        model.proj = layer

        main_calls = []
        protected_calls = []
        clear_calls = []

        layer._parallel_stage_decoder = nn.Identity()
        layer._protected_residual_parallel_decoder = nn.Identity()
        layer._build_parallel_stage_decode_plan = mock.Mock(
            side_effect=lambda: main_calls.append("main")
        )
        layer._build_protected_residual_parallel_decode_plan = mock.Mock(
            side_effect=lambda: protected_calls.append("protected")
        )
        layer.clear_decoded_weight_cache = mock.Mock(
            side_effect=lambda: clear_calls.append("clear")
        )

        _refresh_vae_linear_runtime_after_state_load(model)

        self.assertEqual(main_calls, ["main"])
        self.assertEqual(protected_calls, ["protected"])
        self.assertEqual(clear_calls, ["clear"])


if __name__ == "__main__":
    unittest.main()

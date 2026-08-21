from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.down_layer_sensitivity.core import (
    DownLayerRef,
    assert_cached_weights_on_device,
    assert_down_original_devices,
    assert_down_restore_set,
    compute_down_weight_metrics,
    discover_down_layers,
    hoist_cached_weights_to_device,
    hoist_down_original_weights,
    pin_down_original_weights_to_cpu,
    prewarm_compressed_weights,
    reset_all_vae_to_compressed,
    set_down_restore_set,
    stage_cached_weight_to_cpu,
    unload_non_down_original_weights,
)
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear

_NUM_LAYERS = 36


def _make_decoder(*, codebook_dim: int = 4, latent_dim: int = 9) -> Decoder:
    return Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)


def _make_vq_bits(*, compressed_out: int, compressed_in: int, codebook_dim: int = 4, latent_dim: int = 9):
    expected = int(compressed_out) * int(compressed_in)
    n_blocks = expected // int(codebook_dim)
    rows = []
    for block_idx in range(n_blocks):
        pattern = [((block_idx + bit_idx) % 2) == 0 for bit_idx in range(latent_dim)]
        rows.append([pattern])
    return torch.tensor(rows, dtype=torch.bool)


def _build_vae_linear(
    *,
    in_features: int = 4,
    out_features: int = 4,
    original_weight: torch.Tensor | None = None,
    always_use_original: bool = False,
    protect_original_weight: bool = False,
) -> VAELinear:
    codebook_dim = 4
    bits = _make_vq_bits(compressed_out=out_features, compressed_in=in_features, codebook_dim=codebook_dim)
    if original_weight is None:
        original_weight = torch.randn(out_features, in_features)
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=None,
        original_weight=original_weight,
        vq_weight=bits,
        decoder=copy.deepcopy(_make_decoder(codebook_dim=codebook_dim)),
        codebook_dim=codebook_dim,
        transpose=False,
        compressed_in_features=in_features,
        compressed_out_features=out_features,
        always_use_original=always_use_original,
        protect_original_weight=protect_original_weight,
    )


class _SyntheticDownModel(nn.Module):
    def __init__(
        self,
        *,
        num_hidden_layers: int = _NUM_LAYERS,
        down_modules: dict[int, nn.Module] | None = None,
        include_down: set[int] | None = None,
    ):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        down_modules = dict(down_modules or {})
        include_down = set(range(num_hidden_layers)) if include_down is None else set(include_down)
        for layer_idx in range(num_hidden_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            if layer_idx in include_down:
                layer.mlp.down_proj = down_modules.get(
                    layer_idx,
                    _build_vae_linear(
                        original_weight=torch.randn(4, 4),
                    ),
                )
            self.model.layers.append(layer)


def _down_name(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.mlp.down_proj"


def _make_down_refs(layer_indexes: list[int]) -> list[DownLayerRef]:
    refs = []
    for layer_idx in layer_indexes:
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        refs.append(
            DownLayerRef(
                layer_idx=layer_idx,
                name=_down_name(layer_idx),
                module=module,
            )
        )
    return refs


class TestDiscoverDownLayers:
    def test_valid_contiguous_down_refs_are_sorted_by_layer_index(self):
        model = _SyntheticDownModel()
        refs = discover_down_layers(model)
        assert len(refs) == _NUM_LAYERS
        assert [ref.layer_idx for ref in refs] == list(range(_NUM_LAYERS))
        assert refs == sorted(refs, key=lambda ref: ref.layer_idx)
        for ref in refs:
            assert ref.name == _down_name(ref.layer_idx)
            assert isinstance(ref.module, VAELinear)

    def test_missing_layer_raises_value_error(self):
        model = _SyntheticDownModel(include_down=set(range(_NUM_LAYERS - 1)))
        with pytest.raises(ValueError, match="Expected contiguous down layer indexes"):
            discover_down_layers(model)

    def test_duplicate_non_contiguous_layer_index_raises_value_error(self):
        model = _SyntheticDownModel(include_down=set(range(1, _NUM_LAYERS)))
        with pytest.raises(ValueError, match="Expected contiguous down layer indexes"):
            discover_down_layers(model)

    def test_matched_down_module_not_vae_linear_raises_type_error(self):
        down_modules = {0: nn.Linear(4, 4)}
        for layer_idx in range(1, _NUM_LAYERS):
            down_modules[layer_idx] = _build_vae_linear(original_weight=torch.randn(4, 4))
        model = _SyntheticDownModel(down_modules=down_modules)
        with pytest.raises(TypeError, match="Expected VAELinear"):
            discover_down_layers(model)

    def test_down_original_weight_none_raises_value_error(self):
        codebook_dim = 4
        bits = _make_vq_bits(compressed_out=4, compressed_in=4, codebook_dim=codebook_dim)
        down_modules = {
            0: VAELinear(
                in_features=4,
                out_features=4,
                bias=None,
                original_weight=None,
                vq_weight=bits,
                decoder=copy.deepcopy(_make_decoder(codebook_dim=codebook_dim)),
                codebook_dim=codebook_dim,
                transpose=False,
                compressed_in_features=4,
                compressed_out_features=4,
            ),
        }
        for layer_idx in range(1, _NUM_LAYERS):
            down_modules[layer_idx] = _build_vae_linear(original_weight=torch.randn(4, 4))
        model = _SyntheticDownModel(down_modules=down_modules)
        with pytest.raises(ValueError, match="missing original_weight"):
            discover_down_layers(model)

    def test_always_use_original_true_raises_value_error(self):
        down_modules = {
            0: _build_vae_linear(
                original_weight=torch.randn(4, 4),
                always_use_original=True,
            ),
        }
        for layer_idx in range(1, _NUM_LAYERS):
            down_modules[layer_idx] = _build_vae_linear(original_weight=torch.randn(4, 4))
        model = _SyntheticDownModel(down_modules=down_modules)
        with pytest.raises(ValueError, match="always_use_original=True"):
            discover_down_layers(model)


class TestRestoreSetState:
    def test_no_cross_job_state_leakage(self):
        down_layers = _make_down_refs([0, 1, 2, 3, 7])
        model = nn.Module()
        for ref in down_layers:
            setattr(model, f"down_{ref.layer_idx}", ref.module)

        reset_all_vae_to_compressed(model)
        set_down_restore_set(down_layers, {3})
        assert_down_restore_set(down_layers, {3})

        reset_all_vae_to_compressed(model)
        set_down_restore_set(down_layers, {7})
        assert_down_restore_set(down_layers, {7})

        reset_all_vae_to_compressed(model)
        set_down_restore_set(down_layers, set())
        assert_down_restore_set(down_layers, set())

        reset_all_vae_to_compressed(model)
        set_down_restore_set(down_layers, {0, 1, 2})
        assert_down_restore_set(down_layers, {0, 1, 2})

    def test_unknown_restore_layer_raises_value_error(self):
        down_layers = _make_down_refs([0, 1])
        with pytest.raises(ValueError, match="Unknown down layer indices: \\[99\\]"):
            set_down_restore_set(down_layers, {99})


class TestUnloadNonDownOriginalWeights:
    def test_unloading_does_not_change_compressed_forward(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        module.set_temporary(True)
        x = torch.randn(2, 4)
        y_before = module(x)
        module.unload_original_linear()
        y_after = module(x)
        assert torch.allclose(y_before, y_after)

    def test_unload_counts_and_down_original_kept(self):
        down_module = _build_vae_linear(original_weight=torch.randn(4, 4))
        non_down_module = _build_vae_linear(original_weight=torch.randn(4, 4))
        protected_module = _build_vae_linear(
            original_weight=torch.randn(4, 4),
            protect_original_weight=True,
        )
        already_unloaded = _build_vae_linear(original_weight=torch.randn(4, 4))
        already_unloaded.unload_original_linear()

        model = nn.Module()
        model.down = down_module
        model.other = non_down_module
        model.protected = protected_module
        model.already = already_unloaded

        stats = unload_non_down_original_weights(model, {"down"})
        assert stats == {
            "total_vae": 4,
            "down_original_kept": 1,
            "non_down_original_unloaded": 1,
            "non_down_already_unloaded": 1,
            "non_down_protected_original_kept": 1,
        }
        assert non_down_module.original_weight is None
        assert non_down_module.temporary is True
        assert protected_module.original_weight is not None
        assert protected_module.temporary is True

    def test_unload_retained_original_without_protection_raises_runtime_error(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        model = nn.Module()
        model.other = module

        def _fail_unload():
            return False

        module.unload_original_linear = _fail_unload
        with pytest.raises(RuntimeError, match="retained original_weight without protect_original_weight"):
            unload_non_down_original_weights(model, set())


class TestStagedPrewarmHelpers:
    def test_stage_cached_weight_to_cpu(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        weight = torch.randn(4, 4)
        stage_cached_weight_to_cpu(module, weight)
        assert module._cached_weight is not None
        assert module._cached_weight.device.type == "cpu"
        assert torch.allclose(module._cached_weight, weight.cpu())

    def test_hoist_and_assert_cached_weights_on_device(self):
        module_a = _build_vae_linear(original_weight=torch.randn(4, 4))
        module_b = _build_vae_linear(original_weight=torch.randn(4, 4))
        stage_cached_weight_to_cpu(module_a, torch.randn(4, 4))
        stage_cached_weight_to_cpu(module_b, torch.randn(4, 4))

        model = nn.Module()
        model.a = module_a
        model.b = module_b

        moved = hoist_cached_weights_to_device(model, torch.device("cpu"))
        assert moved == 2
        assert_cached_weights_on_device([module_a, module_b], torch.device("cpu"))

    def test_assert_cached_weights_missing_raises(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        with pytest.raises(RuntimeError, match="Missing prewarmed decoded-weight cache"):
            assert_cached_weights_on_device([module], torch.device("cpu"))


class TestPrewarmCompressedWeights:
    def test_batched_prewarm_leaves_caches_on_target_device(self):
        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = _build_vae_linear(original_weight=torch.randn(4, 4))
                self.l1 = _build_vae_linear(original_weight=torch.randn(4, 4))
                self.l2 = _build_vae_linear(original_weight=torch.randn(4, 4))

        model = _TinyModel()
        device = torch.device("cpu")
        stats = prewarm_compressed_weights(model, device, group_size=2, down_layers=[])
        assert stats["total"] == 3
        assert stats["warmed"] == 3
        assert stats["skipped"] == 0
        assert stats["failed"] == 0
        for module in (model.l0, model.l1, model.l2):
            assert module._cached_weight is not None
            assert module._cached_weight.device == device
            assert module.temporary is True


class TestDownOriginalLazyDevices:
    def _three_down_refs(self):
        modules = {
            0: _build_vae_linear(original_weight=torch.randn(4, 4)),
            1: _build_vae_linear(original_weight=torch.randn(4, 4)),
            2: _build_vae_linear(original_weight=torch.randn(4, 4)),
        }
        refs = [
            DownLayerRef(layer_idx=idx, name=f"model.layers.{idx}.mlp.down_proj", module=modules[idx])
            for idx in (0, 1, 2)
        ]
        return refs

    def test_pin_and_hoist_restore_subset(self):
        refs = self._three_down_refs()
        device = torch.device("cpu")
        assert pin_down_original_weights_to_cpu(refs) == 0
        assert_down_original_devices(refs, set(), device)

        hoist_down_original_weights(refs, {1}, device)
        assert_down_original_devices(refs, {1}, device)

        pin_down_original_weights_to_cpu(refs)
        assert_down_original_devices(refs, set(), device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hoist_restore_subset_moves_only_selected_to_cuda(self):
        refs = self._three_down_refs()
        cuda = torch.device("cuda:0")
        hoist_down_original_weights(refs, {0, 1, 2}, cuda)
        assert all(ref.module.original_weight.device == cuda for ref in refs)

        moved = pin_down_original_weights_to_cpu(refs)
        assert moved == 3
        assert_down_original_devices(refs, set(), cuda)

        moved = hoist_down_original_weights(refs, {1}, cuda)
        assert moved == 1
        assert refs[0].module.original_weight.device.type == "cpu"
        assert refs[1].module.original_weight.device == cuda
        assert refs[2].module.original_weight.device.type == "cpu"
        assert_down_original_devices(refs, {1}, cuda)

        pin_down_original_weights_to_cpu(refs)
        assert_down_original_devices(refs, set(), cuda)

    def test_hoist_unknown_layer_raises(self):
        refs = self._three_down_refs()
        with pytest.raises(ValueError, match="Unknown down layer indices"):
            hoist_down_original_weights(refs, {99}, torch.device("cpu"))

    def test_assert_original_device_mismatch_raises(self):
        refs = self._three_down_refs()
        # Expect layer 0 on a non-cpu device while it remains on cpu.
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for cross-device assert mismatch test")
        with pytest.raises(RuntimeError, match="original_weight on"):
            assert_down_original_devices(refs, {0}, torch.device("cuda:0"))


class TestComputeDownWeightMetrics:
    def test_uses_prewarmed_cache_without_redecode(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        cached = torch.randn(4, 4)
        module._cached_weight = cached.detach()
        refs = [DownLayerRef(layer_idx=0, name="model.layers.0.mlp.down_proj", module=module)]

        metrics = compute_down_weight_metrics(refs)
        assert len(metrics) == 1
        row = metrics[0]
        assert row["layer_idx"] == 0
        assert row["name"] == "model.layers.0.mlp.down_proj"
        assert row["numel"] == 16
        assert row["nmse"] >= 0.0
        assert row["relative_fro_error"] == pytest.approx(row["nmse"] ** 0.5)

    def test_metrics_ok_with_original_on_cpu(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        module._cached_weight = torch.randn(4, 4)
        refs = [DownLayerRef(layer_idx=0, name="model.layers.0.mlp.down_proj", module=module)]
        pin_down_original_weights_to_cpu(refs)
        metrics = compute_down_weight_metrics(refs)
        assert metrics[0]["numel"] == 16
        assert module.original_weight.device.type == "cpu"

    def test_missing_cache_raises_runtime_error(self):
        module = _build_vae_linear(original_weight=torch.randn(4, 4))
        refs = [DownLayerRef(layer_idx=0, name="model.layers.0.mlp.down_proj", module=module)]
        with pytest.raises(RuntimeError, match="Missing prewarmed decoded-weight cache"):
            compute_down_weight_metrics(refs)

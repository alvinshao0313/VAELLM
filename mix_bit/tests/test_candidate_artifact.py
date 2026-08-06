from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from mix_bit.candidate_artifact import save_candidate_artifact_from_model
from mix_bit.schema import CandidateMode


def _make_decoder(*, in_dim: int, out_dim: int) -> Decoder:
    return Decoder(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=16,
        num_res_blocks=0,
        decoder_type="linear",
        norm_type="group",
        activation_type="swish",
    )


def _make_vae_linear(
    *,
    in_features: int = 8,
    out_features: int = 8,
    codebook_bits: int = 16,
    codebook_dim: int = 4,
    residual_stages: int = 2,
    with_bias: bool = False,
    with_original: bool = True,
) -> VAELinear:
    n_blocks = (in_features * out_features) // codebook_dim
    logical = (n_blocks, 1, codebook_bits)
    stages = []
    decoders = []
    for _ in range(residual_stages):
        stages.append(torch.randint(0, 2, logical, dtype=torch.bool))
        decoders.append(_make_decoder(in_dim=codebook_bits, out_dim=codebook_dim))
    bias = nn.Parameter(torch.zeros(out_features)) if with_bias else None
    original = nn.Parameter(torch.randn(out_features, in_features)) if with_original else None
    if residual_stages == 1:
        return VAELinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            original_weight=original,
            vq_weight=stages[0],
            decoder=decoders[0],
            codebook_dim=codebook_dim,
            transpose=False,
            parallel_parts=1,
        )
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        original_weight=original,
        stage_vq_weights=stages,
        stage_decoders=decoders,
        codebook_dim=codebook_dim,
        stage_codebook_dims=[codebook_dim] * residual_stages,
        transpose=False,
        parallel_parts=1,
    )


class _Host(nn.Module):
    def __init__(self, modules: dict[str, VAELinear]):
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleList([nn.ModuleDict({})])})
        # Flatten as named modules under model.layers.0.*
        layer = nn.Module()
        for name, module in modules.items():
            setattr(layer, name, module)
        self.model = nn.Module()
        self.layers = nn.ModuleList([layer])
        # Ensure names are model.layers.0.<suffix> style via wrapper
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.embed_tokens = nn.Embedding(16, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.dense_backbone = nn.Linear(8, 8, bias=False)


def _trial_spec(tmp_path: Path, module_names: list[str], mode: CandidateMode | None = None) -> Path:
    if mode is None:
        mode = CandidateMode(name="b16d4s2", nominal_bit=8.0, codebook_bits=16, codebook_dim=4, residual_stages=2)
    payload = {
        "model_id": "toy",
        "run_id": "toy_run",
        "category_name": "q_proj",
        "target_module_suffix": "q_proj",
        "transpose_module_suffixes": ["q_proj"],
        "expected_module_names": module_names,
        "resolved_linear_group_size": len(module_names),
        "model_inventory_fingerprint": "f" * 64,
        "run_config_sha256": "r" * 64,
        "candidate_space_sha256": "c" * 64,
        "training_recipe_sha256": "t" * 64,
        "model_profile_sha256": "p" * 64,
        "mode": {
            "name": mode.name,
            "nominal_bit": mode.nominal_bit,
            "codebook_bits": mode.codebook_bits,
            "codebook_dim": mode.codebook_dim,
            "residual_stages": mode.residual_stages,
        },
        "trial_root": str(tmp_path / "trial"),
        "cat_train_output_parent": str(tmp_path / "trial" / "runs"),
        "command_args": [],
    }
    path = tmp_path / "trial_spec.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_direct_export_keeps_only_exact_target_module_prefixes(tmp_path: Path):
    q0 = _make_vae_linear()
    q1 = _make_vae_linear()
    host = _Host({"q_proj": q0, "k_proj": q1})
    # rename path: model.layers.0.q_proj
    names = ["model.layers.0.q_proj"]
    # rebuild host with only expected naming via ModuleDict nesting
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    host.model.layers[0].k_proj = q1
    host.embed_tokens = nn.Embedding(16, 8)
    host.norm = nn.LayerNorm(8)
    host.lm_head = nn.Linear(8, 16, bias=False)
    host.dense_backbone = nn.Linear(8, 8, bias=False)

    spec = _trial_spec(tmp_path, names)
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(spec),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    state = torch.load(out / "module_state.pt", map_location="cpu", weights_only=False)
    assert all(k.startswith("model.layers.0.q_proj.") for k in state)
    assert not any(k.startswith("model.layers.0.k_proj.") for k in state)


def test_direct_export_contains_no_embedding_norm_lm_head_or_dense_backbone_keys(tmp_path: Path):
    q0 = _make_vae_linear()
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    host.embed_tokens = nn.Embedding(16, 8)
    host.norm = nn.LayerNorm(8)
    host.lm_head = nn.Linear(8, 16, bias=False)
    host.dense_backbone = nn.Linear(8, 8, bias=False)
    names = ["model.layers.0.q_proj"]
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(_trial_spec(tmp_path, names)),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    state = torch.load(out / "module_state.pt", map_location="cpu", weights_only=False)
    banned = ("embed_tokens", "norm", "lm_head", "dense_backbone")
    assert not any(any(b in k for b in banned) for k in state)


def test_direct_export_rejects_has_original_weight(tmp_path: Path):
    q0 = _make_vae_linear(with_original=True)
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    # protect original so unload fails / remains
    q0.protect_original_weight = True
    names = ["model.layers.0.q_proj"]
    with pytest.raises(ValueError, match="original_weight"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, names)),
            output_dir=str(tmp_path / "artifact"),
            source_run_dir=str(tmp_path / "run"),
        )


def test_direct_export_vq_payload_is_uint8_bitpacked(tmp_path: Path):
    q0 = _make_vae_linear()
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"])),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    state = torch.load(out / "module_state.pt", map_location="cpu", weights_only=False)
    vq_keys = [k for k in state if "vq_weight" in k.split(".")[-1] or ".vq_weight" in k]
    assert vq_keys
    for key in vq_keys:
        assert state[key].dtype == torch.uint8


def test_validate_vq_buffers_rejects_wrong_shaped_uint8():
    from mix_bit.candidate_artifact import _validate_vq_buffers

    module = _make_vae_linear()
    local = module.state_dict()
    vq_key = next(k for k in local if k.split(".")[-1].startswith("vq_weight"))
    wrong = torch.zeros((3, 3, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="does not match any bitpack_u8 storage spec"):
        _validate_vq_buffers(module, {vq_key: wrong})


def test_decoded_weight_cache_is_never_persisted(tmp_path: Path):
    q0 = _make_vae_linear()
    # populate cache
    _ = q0(torch.randn(2, 8))
    assert q0._cached_weight is not None
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"])),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    state = torch.load(out / "module_state.pt", map_location="cpu", weights_only=False)
    assert not any("cached" in k for k in state)


def test_candidate_artifact_rebuilds_every_module_strictly(tmp_path: Path):
    q0 = _make_vae_linear(with_bias=True)
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"])),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    assert (out / "completed.json").is_file()
    assert (out / "candidate_meta.json").is_file()
    assert (out / "module_state.pt").is_file()
    meta = json.loads((out / "candidate_meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "vaellm_candidate_modules_v1"
    completed = json.loads((out / "completed.json").read_text(encoding="utf-8"))
    assert completed["module_state_sha256"]
    assert completed["candidate_meta_sha256"]


def test_candidate_meta_carries_all_resolved_hashes(tmp_path: Path):
    q0 = _make_vae_linear(with_bias=True)
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = q0
    out = tmp_path / "artifact"
    save_candidate_artifact_from_model(
        model=host,
        trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"])),
        output_dir=str(out),
        source_run_dir=str(tmp_path / "run"),
    )
    meta = json.loads((out / "candidate_meta.json").read_text(encoding="utf-8"))
    assert meta["run_config_sha256"] == "r" * 64
    assert meta["candidate_space_sha256"] == "c" * 64
    assert meta["training_recipe_sha256"] == "t" * 64
    assert meta["model_profile_sha256"] == "p" * 64
    assert meta["model_inventory_fingerprint"] == "f" * 64


def _build_host_with(module: VAELinear) -> nn.Module:
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = module
    host.embed_tokens = nn.Embedding(16, 8)
    host.norm = nn.LayerNorm(8)
    host.lm_head = nn.Linear(8, 16, bias=False)
    host.dense_backbone = nn.Linear(8, 8, bias=False)
    return host


def _s2_mode() -> CandidateMode:
    return CandidateMode(
        name="b16d4s2",
        nominal_bit=8.0,
        codebook_bits=16,
        codebook_dim=4,
        residual_stages=2,
    )


def test_export_rejects_trial_s2_when_actual_module_is_s1(tmp_path: Path):
    # Trial claims residual_stages=2, but the actual module is a single-stage s1 module.
    module = _make_vae_linear(
        codebook_bits=16, codebook_dim=4, residual_stages=1, with_original=False
    )
    host = _build_host_with(module)
    with pytest.raises(ValueError, match="residual_stages"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"], _s2_mode())),
            output_dir=str(tmp_path / "artifact"),
            source_run_dir=str(tmp_path / "run"),
        )


def test_export_rejects_trial_mode_when_actual_codebook_dim_differs(tmp_path: Path):
    # Trial claims codebook_dim=4, but the actual module uses codebook_dim=8.
    module = _make_vae_linear(
        codebook_bits=16, codebook_dim=8, residual_stages=2, with_original=False
    )
    host = _build_host_with(module)
    with pytest.raises(ValueError, match="codebook_dim"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"], _s2_mode())),
            output_dir=str(tmp_path / "artifact"),
            source_run_dir=str(tmp_path / "run"),
        )


def test_export_rejects_trial_mode_when_actual_vq_logical_bits_differ(tmp_path: Path):
    # Trial claims codebook_bits=16, but the actual module packs 8 logical bits per codebook.
    module = _make_vae_linear(
        codebook_bits=8, codebook_dim=4, residual_stages=2, with_original=False
    )
    host = _build_host_with(module)
    with pytest.raises(ValueError, match="logical_shape"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"], _s2_mode())),
            output_dir=str(tmp_path / "artifact"),
            source_run_dir=str(tmp_path / "run"),
        )


def test_export_rejects_trial_mode_when_decoder_in_dim_differs(tmp_path: Path):
    # Trial claims codebook_bits=16, but the decoder in_dim is 8 (mismatched decoder).
    module = _make_vae_linear(
        codebook_bits=16, codebook_dim=4, residual_stages=2, with_original=False
    )
    # Sabotage every stage decoder in_dim so it no longer matches codebook_bits; the
    # packer requires all stage decoders to agree, so we mutate all of them together.
    for stage_idx in range(2):
        module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=0).in_dim = 8
    host = _build_host_with(module)
    with pytest.raises(ValueError, match="in_dim"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"], _s2_mode())),
            output_dir=str(tmp_path / "artifact"),
            source_run_dir=str(tmp_path / "run"),
        )


def test_export_does_not_write_completed_on_contract_failure(tmp_path: Path):
    # Pre-existing completed.json must be removed when the contract fails, so a stale
    # seemingly-complete file never survives a failed re-export.
    out = tmp_path / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    (out / "completed.json").write_text(
        json.dumps({"module_state_sha256": "0" * 64, "candidate_meta_sha256": "1" * 64}),
        encoding="utf-8",
    )
    module = _make_vae_linear(
        codebook_bits=16, codebook_dim=4, residual_stages=1, with_original=False
    )
    host = _build_host_with(module)
    with pytest.raises(ValueError, match="residual_stages"):
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=str(_trial_spec(tmp_path, ["model.layers.0.q_proj"], _s2_mode())),
            output_dir=str(out),
            source_run_dir=str(tmp_path / "run"),
        )
    assert not (out / "completed.json").is_file()

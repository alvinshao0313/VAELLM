from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from mix_bit.candidate_artifact import save_candidate_artifact_from_model
from mix_bit.candidate_pool import generate_candidate_trials, write_trial_spec
from mix_bit.checkpoint_pool import CandidatePoolIndex, build_candidate_pool_index
from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import ModelInventory, inventory_from_targets
from mix_bit.schema import (
    CalibrationConfig,
    CandidateMode,
    CandidateSpaceConfig,
    CandidateTrainingSpec,
    CategorySpec,
    MixBitRunConfig,
    ModelProfile,
    ResolvedRunConfig,
    TrainingRecipeConfig,
)


HIDDEN = 8
VOCAB = 16
BASELINE_MODE = "b4d4s2"


class _TinyTokenizer:
    """Tiny save/reload tokenizer fixture for assembler tests (no HF network)."""

    def __init__(
        self,
        *,
        vocab_size: int = 32,
        pad_token_id: int | None = 0,
        eos_token_id: int = 0,
        bos_token_id: int | None = None,
        unk_token_id: int | None = None,
        name_or_path: str = "tiny",
        chat_template: str | None = None,
        vocab_seed: int = 0,
        model_max_length: int = 1024,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.unk_token_id = unk_token_id
        self.name_or_path = name_or_path
        self.padding_side = "left"
        self.truncation_side = "right"
        self.model_max_length = model_max_length
        self.chat_template = chat_template
        self.init_kwargs = {"name_or_path": name_or_path, "vocab_size": vocab_size}
        self.mix_bit_pad_token_normalized_from_eos = False
        self._vocab_seed = vocab_seed
        self._vocab = {f"tok_{i + vocab_seed}": i for i in range(vocab_size)}
        self.special_tokens_map: dict = {}

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {}

    def save_pretrained(self, output_dir):
        import os

        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "chat_template": self.chat_template,
            "vocab_seed": self._vocab_seed,
            "model_max_length": self.model_max_length,
        }
        (Path(output_dir) / "tiny_tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )


def _patch_tiny_tokenizer(monkeypatch: pytest.MonkeyPatch):
    def _fake_from_pretrained(path, *, local_files_only=False, trust_remote_code=False, **_kwargs):
        marker = Path(path) / "tiny_tokenizer.json"
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            data["name_or_path"] = str(path)
            return _TinyTokenizer(**data)
        if local_files_only:
            raise OSError(
                f"No tokenizer files found in {path} (local_files_only=True)"
            )
        return _TinyTokenizer(name_or_path=str(path))

    monkeypatch.setattr("mix_bit.model_adapter.AutoTokenizer.from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr("mix_bit.assembler.AutoTokenizer.from_pretrained", _fake_from_pretrained)


def _make_decoder(cdim: int) -> Decoder:
    return Decoder(
        in_dim=cdim,
        out_dim=cdim,
        hidden_dim=16,
        num_res_blocks=0,
        decoder_type="linear",
        norm_type="group",
        activation_type="swish",
    )


def _make_vae_linear(
    *,
    in_features: int = HIDDEN,
    out_features: int = HIDDEN,
    cdim: int = 4,
    residual_stages: int = 2,
    with_bias: bool = False,
    transpose: bool = False,
) -> VAELinear:
    n_blocks = (in_features * out_features) // cdim
    logical = (n_blocks, 1, cdim)
    stages = []
    decoders = []
    for _ in range(residual_stages):
        stages.append(torch.randint(0, 2, logical, dtype=torch.bool))
        decoders.append(_make_decoder(cdim))
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=None if not with_bias else nn.Parameter(torch.zeros(out_features)),
        original_weight=None,
        stage_vq_weights=stages,
        stage_decoders=decoders,
        codebook_dim=cdim,
        stage_codebook_dims=[cdim] * residual_stages,
        transpose=transpose,
        parallel_parts=1,
    )


class _ToyLM(nn.Module):
    """Tiny decoder-style model with embeddings / norms / lm_head + target Linears."""

    def __init__(self, n_layers: int = 2, hidden: int = HIDDEN, vocab: int = VOCAB):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.q_proj = nn.Linear(hidden, hidden, bias=False)
            layer.k_proj = nn.Linear(hidden, hidden, bias=False)
            layers.append(layer)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            model_type = "toy"
            _name_or_path = "toy-model"
            vocab_size = vocab

            def save_pretrained(self, path: str) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text(
                    json.dumps({"model_type": "toy", "vocab_size": vocab}),
                    encoding="utf-8",
                )

        self.config = _Cfg()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = x + layer.q_proj(x) + layer.k_proj(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def _toy_profile() -> ModelProfile:
    return ModelProfile(
        model_id="toy",
        model_path="toy-model",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size="all", allow_tail_group=True),
        layer_index_patterns=(r"(?:^|\.)model\.layers\.(\d+)\.",),
        categories=(
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
        ),
        regression_expectations={},
    )


def _toy_modes() -> tuple[CandidateMode, ...]:
    # codebook_bits == codebook_dim == 4 so nominal_bit == residual_stages.
    return (
        CandidateMode(
            name=BASELINE_MODE,
            nominal_bit=2.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=2,
        ),
        CandidateMode(
            name="b4d4s1",
            nominal_bit=1.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=1,
        ),
    )


def _make_resolved(tmp_path: Path, profile: ModelProfile, modes: tuple[CandidateMode, ...]) -> ResolvedRunConfig:
    recipe = TrainingRecipeConfig(
        recipe_id="toy_recipe",
        values={
            "seed": 31,
            "deterministic": True,
            "steps_per_category": 10,
            "batch_size": 8,
            "base_ch": 8,
            "num_res_blocks": 0,
            "decoder_base_ch": 8,
            "decoder_num_res_blocks": 0,
            "norm_type": "layer",
            "activation_type": "swish",
            "decoder_type": "linear",
            "recon_loss_type": "mse",
            "quantizer_type": "BSQ",
            "gamma0": 1.0,
            "gamma": 1.0,
            "zeta": 1.0,
            "inv_temperature": 100.0,
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "lr_scheduler": "linear",
            "lr_warmup_steps": 0,
            "l1_weight": 1.0,
            "lfq_weight": 1.0,
            "commitment_loss_weight": 0.25,
            "entropy_loss_weight": 0.01,
            "normalize_weight": True,
            "vae_decoder_checkpoint": True,
            "new_quant": True,
            "log_every": 1,
            "eval_every": 0,
            "eval_blocks": 8,
            "outlier_protect_mode": "channel",
            "outlier_protect_count": 0,
            "outlier_protect_min_per_layer": 0,
            "distill_after_category": "none",
            "eval_ppl": False,
            "eval_tasks": "",
            "rot_llm": False,
            "fp16": False,
            "bf16": True,
        },
    )
    config = MixBitRunConfig(
        run_id="toy_run",
        model_profile=profile,
        candidate_space=CandidateSpaceConfig(
            candidate_space_id="toy_space",
            baseline_mode=BASELINE_MODE,
            target_average_bit=2.0,
            modes=modes,
        ),
        training_recipe=recipe,
        calibration=CalibrationConfig(
            source_jsonl=str(tmp_path / "calib.jsonl"),
            input_format="text",
            max_samples=1,
            max_length=8,
            seed=0,
            label_mode="all_nonpad",
        ),
    )
    root = tmp_path / "result" / "mix_bit" / profile.model_id
    return ResolvedRunConfig(
        config=config,
        run_config_path=str(tmp_path / "run.json"),
        run_config_sha256="r" * 64,
        model_profile_path=str(tmp_path / "profile.json"),
        model_profile_sha256="p" * 64,
        candidate_space_path=str(tmp_path / "space.json"),
        candidate_space_sha256="c" * 64,
        training_recipe_path=str(tmp_path / "recipe.json"),
        training_recipe_sha256="t" * 64,
        canonical_model_root=str(root),
        canonical_run_root=str(root / "runs" / config.run_id),
    )


def _inventory_for(profile: ModelProfile, model: nn.Module) -> ModelInventory:
    adapter = get_model_adapter("generic_decoder")
    targets = adapter.discover_target_linears(model, profile)
    return inventory_from_targets(
        profile=profile,
        model=model,
        targets=targets,
        model_profile_sha256="p" * 64,
    )


def _export_pool(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
) -> CandidatePoolIndex:
    trials = generate_candidate_trials(resolved, inventory)
    targets_by_cat: dict[str, list[str]] = {}
    for target in inventory.targets:
        targets_by_cat.setdefault(target.category, []).append(target.module_name)

    for trial in trials:
        host = nn.Module()
        host.model = nn.Module()
        # Build nested structure model.layers.<i>.<suffix>
        max_block = max(t.block_index for t in inventory.targets)
        layers = []
        for block_idx in range(max_block + 1):
            layer = nn.Module()
            for target in inventory.targets:
                if target.block_index != block_idx:
                    continue
                if target.category != trial.category_name:
                    continue
                setattr(
                    layer,
                    target.module_suffix,
                    _make_vae_linear(
                        cdim=trial.mode.codebook_dim,
                        residual_stages=trial.mode.residual_stages,
                        transpose=bool(target.transpose),
                    ),
                )
            layers.append(layer)
        host.model.layers = nn.ModuleList(layers)
        host.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        host.norm = nn.LayerNorm(HIDDEN)
        host.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        trial_spec_path = write_trial_spec(trial, command=["python", "-c", "pass"])
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=trial_spec_path,
            output_dir=str(Path(trial.trial_root) / "artifact"),
            source_run_dir=str(Path(trial.trial_root) / "runs" / "fake"),
        )

    return build_candidate_pool_index(resolved, inventory)


@pytest.fixture()
def assembled_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mix_bit.model_inventory import write_model_inventory

    profile = _toy_profile()
    modes = _toy_modes()
    resolved = _make_resolved(tmp_path, profile, modes)
    template = _ToyLM(n_layers=2)
    inventory = _inventory_for(profile, template)
    inventory_path = tmp_path / "inventory.json"
    write_model_inventory(inventory, inventory_path)
    pool_index = _export_pool(resolved, inventory)

    def _load_model(self, _profile, *, access_token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr(
        "mix_bit.model_adapter.GenericDecoderAdapter.load_model",
        _load_model,
    )
    _patch_tiny_tokenizer(monkeypatch)
    return {
        "tmp_path": tmp_path,
        "profile": profile,
        "resolved": resolved,
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "pool_index": pool_index,
        "template": template,
    }


def test_uniform_assignment_selects_baseline_for_all_modules(assembled_world):
    from mix_bit.assembler import build_uniform_assignments

    pool_index = assembled_world["pool_index"]
    inventory = assembled_world["inventory"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    assert set(assignments) == {t.module_name for t in inventory.targets}
    assert all(mode == BASELINE_MODE for mode in assignments.values())
    assert len(assignments) == len(inventory.targets)


def test_overlay_contains_assignments_and_source_hashes_but_no_tensors(assembled_world):
    from mix_bit.assembler import (
        build_uniform_assignments,
        write_uniform_baseline_overlay,
    )

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    out_dir = Path(resolved.canonical_run_root) / "baseline" / BASELINE_MODE
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    overlay_path = write_uniform_baseline_overlay(
        output_dir=str(out_dir),
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        mode_name=BASELINE_MODE,
    )
    payload = json.loads(Path(overlay_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "uniform_baseline_overlay"
    assert payload["mode"] == BASELINE_MODE
    assert payload["target_average_bit"] == 2.0
    assert payload["achieved_average_bit"] == 2.0
    assert payload["run_config_sha256"] == resolved.run_config_sha256
    assert payload["model_inventory_fingerprint"] == inventory.fingerprint_sha256
    assert payload["candidate_manifest_sha256"]
    assert "modules" in payload
    assert len(payload["modules"]) == len(inventory.targets)
    for entry in payload["modules"]:
        assert entry["mode_name"] == BASELINE_MODE
        assert entry["compact_state_sha256"]
        assert entry["module_spec_sha256"]
        assert "tensor" not in json.dumps(entry).lower()
    blob = Path(overlay_path).read_bytes()
    assert b"original_weight" not in blob
    assert b"vq_weight" not in blob
    assert "overlay_sha256" in payload


def test_overlay_directory_contains_no_full_model_state_embedding_or_lm_head_file(assembled_world):
    from mix_bit.assembler import (
        build_uniform_assignments,
        write_uniform_baseline_overlay,
    )

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    out_dir = Path(resolved.canonical_run_root) / "baseline" / BASELINE_MODE
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    write_uniform_baseline_overlay(
        output_dir=str(out_dir),
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        mode_name=BASELINE_MODE,
    )
    names = {p.name for p in out_dir.rglob("*") if p.is_file()}
    assert "baseline_overlay.json" in names
    banned = {
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "embedding.pt",
        "lm_head.pt",
        "module_state.pt",
    }
    assert names.isdisjoint(banned)
    for path in out_dir.rglob("*"):
        if path.is_file():
            text = path.read_bytes()
            assert b"embed_tokens" not in text or path.name.endswith(".json")
            if path.suffix in {".bin", ".pt", ".safetensors"}:
                pytest.fail(f"unexpected tensor file under baseline dir: {path}")


def test_builder_groups_modules_by_compact_artifact(assembled_world, monkeypatch):
    from mix_bit import assembler
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)

    seen_groups: list[list[str]] = []
    original = assembler._group_candidates_by_compact_artifact

    def _spy(cands):
        groups = original(cands)
        seen_groups.extend([[c.module_name for c in group] for group in groups.values()])
        return groups

    monkeypatch.setattr(assembler, "_group_candidates_by_compact_artifact", _spy)
    build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    assert seen_groups
    # Each group should share one category under uniform baseline (one artifact per category).
    flat = [name for group in seen_groups for name in group]
    assert sorted(flat) == sorted(assignments)


def test_builder_opens_each_compact_state_once(assembled_world, monkeypatch):
    from mix_bit import assembler
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments
    from mix_bit.checkpoint_pool import load_compact_state_mmap

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)

    open_counts: dict[str, int] = {}
    original = load_compact_state_mmap

    def _counting(source):
        path = source.compact_state_path
        open_counts[path] = open_counts.get(path, 0) + 1
        return original(source)

    monkeypatch.setattr(assembler, "load_compact_state_mmap", _counting)
    build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    assert open_counts
    assert all(count == 1 for count in open_counts.values())
    # Uniform baseline: one compact artifact per category.
    assert len(open_counts) == len(resolved.config.model_profile.categories)


def test_builder_loads_only_selected_module_prefixes(assembled_world, monkeypatch):
    from mix_bit import assembler
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments
    from mix_bit.module_swap import build_candidate_module

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    # Mixed assignment: baseline for all except one module uses the other mode.
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    other_mode = "b4d4s1"
    victim = inventory.targets[0].module_name
    assignments[victim] = other_mode
    victim_category = inventory.targets[0].category
    same_category_others = [
        t.module_name
        for t in inventory.targets
        if t.category == victim_category and t.module_name != victim
    ]
    assert same_category_others, "fixture must have another module in the victim category"

    built_names: list[str] = []
    install_events: list[tuple[frozenset[str], frozenset[str]]] = []
    original_install = assembler._install_selected_modules_from_compact_state
    original_build = build_candidate_module

    def _spy_install(model, compact_state, candidates, *, device):
        selected = frozenset(c.module_name for c in candidates)
        present = frozenset(
            name
            for name in assignments
            if any(key.startswith(f"{name}.") for key in compact_state)
        )
        install_events.append((selected, present))
        return original_install(model, compact_state, candidates, device=device)

    def _spy_build(candidate, compact_state, *, device):
        assert compact_state
        assert all(key.startswith(f"{candidate.module_name}.") for key in compact_state)
        built_names.append(candidate.module_name)
        return original_build(candidate, compact_state, device=device)

    monkeypatch.setattr(assembler, "_install_selected_modules_from_compact_state", _spy_install)
    monkeypatch.setattr(assembler, "build_candidate_module", _spy_build)

    model = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    assert sorted(built_names) == sorted(assignments)
    assert built_names.count(victim) == 1

    # Victim's other-mode artifact still contains sibling category prefixes; only victim
    # is selected/installed from that open.
    mixed_events = [
        (selected, present)
        for selected, present in install_events
        if victim in selected and any(name in present for name in same_category_others)
    ]
    assert mixed_events, "expected an artifact open that contains non-selected sibling prefixes"
    for selected, present in mixed_events:
        assert victim in selected
        assert selected.isdisjoint(same_category_others)
        assert any(name in present for name in same_category_others)

    for other in same_category_others:
        assert isinstance(model.get_submodule(other), VAELinear)
    assert isinstance(model.get_submodule(victim), VAELinear)


def test_audit_vocab_uses_config_or_input_embeddings_not_top_level_only():
    from mix_bit.assembler import _resolve_audit_vocab_size

    class _Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, 8)
            self.config = type("Cfg", (), {})()

        def get_input_embeddings(self):
            return self.model.embed_tokens

    nested = _Nested()
    assert _resolve_audit_vocab_size(nested) == 32

    class _ConfigOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"vocab_size": 64})()

    assert _resolve_audit_vocab_size(_ConfigOnly()) == 64

    with pytest.raises(ValueError, match="vocab size"):
        _resolve_audit_vocab_size(nn.Linear(4, 4))


def test_built_model_contains_exact_converted_module_set(assembled_world):
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    model = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    vae_names = {name for name, mod in model.named_modules() if isinstance(mod, VAELinear)}
    expected = {t.module_name for t in inventory.targets}
    assert vae_names == expected
    # Unchanged backbone modules remain dense.
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.norm, nn.LayerNorm)


def test_built_model_has_no_original_weight_payload(assembled_world):
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    model = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            assert module.original_weight is None, name
            assert "original_weight" not in module.state_dict()


def test_two_independent_in_memory_builds_match_logits(assembled_world):
    from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    model_a = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    model_b = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (2, 5))
    with torch.no_grad():
        logits_a = model_a(input_ids).logits
        logits_b = model_b(input_ids).logits
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-4, atol=1e-4)


def test_full_checkpoint_save_is_called_only_by_explicit_final_save_path(
    assembled_world, monkeypatch
):
    from mix_bit import assembler
    from mix_bit.assembler import (
        build_uniform_assignments,
        prepare_uniform_baseline_overlay,
        save_full_checkpoint_from_assignments,
    )

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    tmp_path = assembled_world["tmp_path"]

    save_calls: list[str] = []
    real_save = assembler.save_model_checkpoint

    def _spy_save(model, output_dir, **kwargs):
        save_calls.append(str(output_dir))
        return real_save(model, output_dir, **kwargs)

    monkeypatch.setattr(assembler, "save_model_checkpoint", _spy_save)

    prepare_uniform_baseline_overlay(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        device="cpu",
        skip_audit=True,
    )
    assert save_calls == []

    out_dir = tmp_path / "final_full_ckpt"
    # Reload equivalence needs get_model; point it at toy template.
    template = assembled_world["template"]

    def _get_model(_path, _token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr("train_utils.model_checkpoint_io.get_model", _get_model)
    monkeypatch.setattr("rotation.model_utils.get_model", _get_model)

    save_full_checkpoint_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=build_uniform_assignments(pool_index, BASELINE_MODE),
        output_dir=str(out_dir),
        device="cpu",
        mix_bit_provenance={"kind": "test_final_save"},
    )
    assert save_calls == [str(out_dir)]
    assert (out_dir / "pytorch_model.bin").is_file() or any(out_dir.glob("*.bin")) or (
        out_dir / "model.safetensors"
    ).is_file() or (out_dir / "checkpoint_meta.json").is_file()


OTHER_MODE = "b4d4s1"


def _mixed_assignments(inventory: ModelInventory, pool_index: CandidatePoolIndex) -> dict[str, str]:
    from mix_bit.assembler import build_uniform_assignments

    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    # Flip first module of each category to the lower-bit mode so mixed sources span artifacts.
    seen_cats: set[str] = set()
    for target in inventory.targets:
        if target.category in seen_cats:
            continue
        assignments[target.module_name] = OTHER_MODE
        seen_cats.add(target.category)
    return assignments


def _write_allocation_json(
    *,
    path: Path,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: dict[str, str],
    is_globally_optimal: bool = True,
    allow_suboptimal: bool = False,
    kl_mode: str = "teacher_topk",
    metric_name: str = "forward_kl_teacher_topk_renorm",
    teacher_topk: int | None = 256,
    run_config_sha256: str | None = None,
    model_inventory_sha256: str | None = None,
    candidate_manifest_sha256: str | None = None,
    candidate_space_sha256: str | None = None,
    cost_table_sha256: str = "a" * 64,
    cost_table_meta_sha256: str = "b" * 64,
    corrupt_objective: bool = False,
) -> dict[str, Any]:
    from mix_bit.schema import sha256_file
    from mix_bit.solver import bit_to_units

    order = {t.module_name: idx for idx, t in enumerate(inventory.targets)}
    entries = []
    objective = 0.0
    used_units = 0
    total_params = 0
    weighted_bits = 0.0
    for target in sorted(inventory.targets, key=lambda t: order[t.module_name]):
        mode = assignments[target.module_name]
        cand = pool_index.candidates[(target.module_name, mode)]
        # Synthetic but finite costs: baseline cost 0, other mode cost 0.01 * block_index+1
        cost = 0.0 if mode == BASELINE_MODE else 0.01 * (target.block_index + 1)
        objective += cost
        used_units += int(target.param_count) * bit_to_units(cand.nominal_bit)
        total_params += int(target.param_count)
        weighted_bits += int(target.param_count) * float(cand.nominal_bit)
        entries.append(
            {
                "module_name": target.module_name,
                "category": target.category,
                "module_suffix": target.module_suffix,
                "block_index": target.block_index,
                "in_features": target.in_features,
                "out_features": target.out_features,
                "has_bias": target.has_bias,
                "param_count": target.param_count,
                "mode": mode,
                "nominal_bit": cand.nominal_bit,
                "mean_delta_kl": cost,
                "compact_state_sha256": cand.source.compact_state_sha256,
                "per_sample_sha256": "c" * 64,
            }
        )

    target_bit = float(resolved.config.candidate_space.target_average_bit)
    budget_units = bit_to_units(target_bit) * total_params
    achieved = weighted_bits / float(total_params)
    if corrupt_objective:
        objective = objective + 123.0

    payload = {
        "kind": "mix_bit_allocation",
        "model_id": inventory.model_id,
        "run_id": resolved.config.run_id,
        "solver_name": "scipy.optimize.milp",
        "solver_status": 0,
        "solver_message": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
        "scipy_version": "1.0.0",
        "is_globally_optimal": is_globally_optimal,
        "allow_suboptimal": allow_suboptimal,
        "time_limit_sec": None,
        "objective_scale": 1.0,
        "objective_delta_kl": objective,
        "baseline_mode": BASELINE_MODE,
        "baseline_objective_delta_kl": 0.0,
        "baseline_kl_mean": 1.5,
        "predicted_mixed_model_kl": 1.5 + objective,
        "kl_mode": kl_mode,
        "metric_name": metric_name,
        "teacher_topk": teacher_topk,
        "target_average_bit": target_bit,
        "bit_unit_denominator": 2,
        "used_bit_units": used_units,
        "budget_bit_units": budget_units,
        "budget_slack_bit_units": budget_units - used_units,
        "achieved_average_bit": achieved,
        "total_target_parameters": total_params,
        "run_config_sha256": run_config_sha256 if run_config_sha256 is not None else resolved.run_config_sha256,
        "model_inventory_sha256": (
            model_inventory_sha256
            if model_inventory_sha256 is not None
            else inventory.fingerprint_sha256
        ),
        "candidate_manifest_sha256": (
            candidate_manifest_sha256
            if candidate_manifest_sha256 is not None
            else sha256_file(pool_index.manifest_path)
        ),
        "candidate_space_sha256": (
            candidate_space_sha256
            if candidate_space_sha256 is not None
            else resolved.candidate_space_sha256
        ),
        "cost_table_sha256": cost_table_sha256,
        "cost_table_meta_sha256": cost_table_meta_sha256,
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _allocation_path(resolved: ResolvedRunConfig) -> Path:
    return Path(resolved.canonical_run_root) / "allocation" / "topk_k256" / "optimal_2bit.json"


def _patch_reload_model(monkeypatch: pytest.MonkeyPatch, template: nn.Module) -> None:
    def _get_model(_path, _token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr("train_utils.model_checkpoint_io.get_model", _get_model)
    monkeypatch.setattr("rotation.model_utils.get_model", _get_model)


def test_mixed_assembler_uses_selected_compact_source_for_each_module(
    assembled_world, monkeypatch
):
    from mix_bit import assembler
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.module_swap import build_candidate_module

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    seen: dict[str, str] = {}
    original = build_candidate_module

    def _spy(candidate, compact_state, *, device):
        seen[candidate.module_name] = candidate.source.compact_state_sha256
        return original(candidate, compact_state, device=device)

    monkeypatch.setattr(assembler, "build_candidate_module", _spy)
    assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    for name, mode in assignments.items():
        expected = pool_index.candidates[(name, mode)].source.compact_state_sha256
        assert seen[name] == expected


def test_two_modules_from_same_compact_artifact_open_source_once(
    assembled_world, monkeypatch
):
    from mix_bit import assembler
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.checkpoint_pool import load_compact_state_mmap

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    # Uniform baseline => each category artifact opens once for multiple modules.
    from mix_bit.assembler import build_uniform_assignments

    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    open_counts: dict[str, int] = {}
    original = load_compact_state_mmap

    def _counting(source):
        open_counts[source.compact_state_path] = open_counts.get(source.compact_state_path, 0) + 1
        return original(source)

    monkeypatch.setattr(assembler, "load_compact_state_mmap", _counting)
    assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    assert open_counts
    assert all(count == 1 for count in open_counts.values())
    # At least one artifact must supply more than one selected module.
    groups = assembler._group_candidates_by_compact_artifact(
        [pool_index.candidates[(n, m)] for n, m in assignments.items()]
    )
    assert any(len(group) >= 2 for group in groups.values())


def test_unselected_module_prefix_is_not_loaded(assembled_world, monkeypatch):
    from mix_bit import assembler
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.module_swap import build_candidate_module

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    built: list[str] = []
    original = build_candidate_module

    def _spy(candidate, compact_state, *, device):
        assert all(key.startswith(f"{candidate.module_name}.") for key in compact_state)
        built.append(candidate.module_name)
        return original(candidate, compact_state, device=device)

    monkeypatch.setattr(assembler, "build_candidate_module", _spy)
    assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    assert sorted(built) == sorted(assignments)


def test_allocation_hash_or_inventory_mismatch_is_rejected(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        model_inventory_sha256="0" * 64,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    with pytest.raises(ValueError, match="inventory"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )

    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        run_config_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="run_config"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_checkpoint_extra_meta_contains_full_provenance(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint, derive_mixed_model_dir
    from mix_bit.schema import sha256_file
    from train_utils.model_checkpoint_io import META_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    inv_path = assembled_world["inventory_path"]
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=inv_path,
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(result["output_dir"])
    assert out_dir == derive_mixed_model_dir(alloc_path)
    meta = json.loads((out_dir / META_FILENAME).read_text(encoding="utf-8"))
    mix = meta["extra_meta"]["mix_bit"]
    assert mix["kind"] == "optimal_mixed_bit"
    assert mix["model_id"] == inventory.model_id
    assert mix["run_id"] == resolved.config.run_id
    assert mix["run_config_sha256"] == resolved.run_config_sha256
    assert mix["model_profile_sha256"] == resolved.model_profile_sha256
    assert mix["candidate_space_sha256"] == resolved.candidate_space_sha256
    assert mix["training_recipe_sha256"] == resolved.training_recipe_sha256
    assert mix["model_inventory_fingerprint"] == inventory.fingerprint_sha256
    assert mix["model_inventory_sha256"] == sha256_file(inv_path)
    assert mix["candidate_manifest_sha256"] == sha256_file(pool_index.manifest_path)
    assert mix["cost_table_sha256"]
    assert mix["cost_table_meta_sha256"]
    assert mix["allocation_sha256"] == sha256_file(alloc_path)
    assert mix["baseline_mode"] == BASELINE_MODE
    assert mix["target_average_bit"] == 2.0
    assert "achieved_average_bit" in mix
    assert "used_bit_units" in mix and "budget_bit_units" in mix
    assert "objective_delta_kl" in mix
    assert "predicted_mixed_model_kl" in mix
    assert "is_globally_optimal" in mix
    assert mix["kl_mode"] == "teacher_topk"
    assert mix["metric_name"] == "forward_kl_teacher_topk_renorm"
    assert mix["teacher_topk"] == 256
    assert [e["module_name"] for e in mix["assignments"]] == [
        t.module_name for t in inventory.targets
    ]
    assert mix["compact_artifacts"]
    for art in mix["compact_artifacts"]:
        assert art["compact_state_path"]
        assert art["compact_state_sha256"]


def test_final_checkpoint_contains_unchanged_embedding_norm_and_lm_head_once(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from train_utils.model_checkpoint_io import STATE_DICT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    state = torch.load(
        Path(result["output_dir"]) / STATE_DICT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    for key in ("embed_tokens.weight", "norm.weight", "norm.bias", "lm_head.weight"):
        assert key in state
        assert sum(1 for k in state if k == key) == 1
    # Backbone dense tensors unchanged vs template (loaded once from original model).
    template = assembled_world["template"]
    torch.testing.assert_close(state["embed_tokens.weight"], template.embed_tokens.weight.detach().cpu())
    torch.testing.assert_close(state["lm_head.weight"], template.lm_head.weight.detach().cpu())
    torch.testing.assert_close(state["norm.weight"], template.norm.weight.detach().cpu())


def test_final_checkpoint_target_linears_store_uint8_packed_codes_not_dense_weights(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from train_utils.model_checkpoint_io import STATE_DICT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    state = torch.load(
        Path(result["output_dir"]) / STATE_DICT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    for target in inventory.targets:
        dense_key = f"{target.module_name}.weight"
        assert dense_key not in state
        vq_keys = [
            k
            for k in state
            if k.startswith(f"{target.module_name}.")
            and k.split(".")[-1].startswith("vq_weight")
        ]
        assert vq_keys, target.module_name
        for key in vq_keys:
            assert state[key].dtype == torch.uint8, key


def test_saved_checkpoint_has_no_original_weight_payload(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from train_utils.model_checkpoint_io import META_FILENAME, STATE_DICT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out = Path(result["output_dir"])
    state = torch.load(out / STATE_DICT_FILENAME, map_location="cpu", weights_only=False)
    assert not any("original_weight" in k for k in state)
    meta_text = (out / META_FILENAME).read_text(encoding="utf-8")
    assert '"has_original_weight": true' not in meta_text
    assert "original_weight" not in meta_text.lower() or '"has_original_weight": false' in meta_text


def test_no_complete_model_checkpoint_exists_before_final_assembly(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import (
        assemble_optimal_mixed_checkpoint,
        prepare_uniform_baseline_overlay,
    )
    from train_utils.model_checkpoint_io import META_FILENAME, STATE_DICT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    prepare_uniform_baseline_overlay(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        device="cpu",
        skip_audit=True,
    )
    run_root = Path(resolved.canonical_run_root)
    banned = {STATE_DICT_FILENAME, META_FILENAME, "model.safetensors"}
    before = [
        p for p in run_root.rglob("*") if p.is_file() and p.name in banned
    ]
    assert before == []

    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out = Path(result["output_dir"])
    assert (out / META_FILENAME).is_file()
    assert (out / STATE_DICT_FILENAME).is_file()
    # Only under mixed_model/.../final_model
    others = [
        p
        for p in run_root.rglob(STATE_DICT_FILENAME)
        if p.is_file() and p.parent.resolve() != out.resolve()
    ]
    assert others == []


def test_reloaded_mixed_checkpoint_has_same_assignments(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.model_adapter import get_model_adapter
    from train_utils.model_checkpoint_io import META_FILENAME, load_checkpoint_into_model

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out = Path(result["output_dir"])
    meta = json.loads((out / META_FILENAME).read_text(encoding="utf-8"))
    saved_assign = {
        e["module_name"]: e["mode_name"] for e in meta["extra_meta"]["mix_bit"]["assignments"]
    }
    assert saved_assign == assignments

    adapter = get_model_adapter(resolved.config.model_profile.adapter)
    reloaded = adapter.load_model(resolved.config.model_profile)
    load_checkpoint_into_model(model=reloaded, model_dir=str(out), map_location="cpu", strict=True)
    for name in assignments:
        module = reloaded.get_submodule(name)
        assert isinstance(module, VAELinear)
        assert module.original_weight is None


def test_existing_output_without_mix_bit_provenance_requires_overwrite(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint, derive_mixed_model_dir
    from train_utils.model_checkpoint_io import META_FILENAME, STATE_DICT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    out_dir = derive_mixed_model_dir(alloc_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Incomplete / corrupt prior output: state+meta present but mix_bit missing.
    (out_dir / STATE_DICT_FILENAME).write_bytes(b"not-a-real-state")
    (out_dir / META_FILENAME).write_text(
        json.dumps({"format": "vaellm_state_dict_with_meta", "extra_meta": {}}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )

    # Corrupt mix_bit object (wrong type) also requires overwrite.
    (out_dir / META_FILENAME).write_text(
        json.dumps(
            {"format": "vaellm_state_dict_with_meta", "extra_meta": {"mix_bit": "bad"}},
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )

    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
        overwrite=True,
    )
    assert result["skipped_identical"] is False
    meta = json.loads((out_dir / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["extra_meta"]["mix_bit"]["kind"] == "optimal_mixed_bit"


def test_allocation_corrupt_objective_or_missing_compact_hash_is_rejected(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _patch_reload_model(monkeypatch, assembled_world["template"])

    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        corrupt_objective=True,
    )
    with pytest.raises(ValueError, match="objective_delta_kl"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )

    payload = _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    payload["entries"][0]["compact_state_sha256"] = ""
    alloc_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compact_state_sha256"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_final_checkpoint_writes_state_fingerprint_manifest_and_returns_path(
    assembled_world, monkeypatch
):
    import inspect

    from mix_bit.assembler import (
        assemble_optimal_mixed_checkpoint,
        save_full_checkpoint_from_assignments,
    )
    from mix_bit.state_fingerprint import STATE_FINGERPRINT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out = Path(result["output_dir"])
    fingerprint_path = out / STATE_FINGERPRINT_FILENAME
    assert fingerprint_path.is_file()
    assert result["state_fingerprint"] == str(fingerprint_path.resolve())
    manifest = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "mix_bit_state_fingerprint_v1"
    assert manifest["key_count"] == len(manifest["entries"])

    # Fingerprint key count must equal the reload model state key count.
    from mix_bit.model_adapter import get_model_adapter
    from train_utils.model_checkpoint_io import load_checkpoint_into_model

    reloaded = get_model_adapter(resolved.config.model_profile.adapter).load_model(
        resolved.config.model_profile
    )
    load_checkpoint_into_model(model=reloaded, model_dir=str(out), map_location="cpu", strict=True)
    assert manifest["key_count"] == len(reloaded.state_dict())

    # save_full_checkpoint_from_assignments must no longer clone the full state dict.
    source = inspect.getsource(save_full_checkpoint_from_assignments)
    assert "reference_state" not in source
    assert ".cpu().clone()" not in source
    assert "assert_close" not in source


def test_skip_identical_requires_state_fingerprint_file(assembled_world, monkeypatch):
    from mix_bit.assembler import (
        assemble_optimal_mixed_checkpoint,
        derive_mixed_model_dir,
    )
    from mix_bit.state_fingerprint import STATE_FINGERPRINT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    assert first["skipped_identical"] is False
    out_dir = Path(first["output_dir"])
    fingerprint_path = out_dir / STATE_FINGERPRINT_FILENAME
    assert fingerprint_path.is_file()

    # Remove the fingerprint manifest: skip must be rejected with overwrite hint.
    fingerprint_path.unlink()

    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_skip_identical_rejects_tampered_fingerprint_manifest(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.state_fingerprint import STATE_FINGERPRINT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(first["output_dir"])
    fingerprint_path = out_dir / STATE_FINGERPRINT_FILENAME

    # Tamper: replace with a manifest whose kind is wrong.
    fingerprint_path.write_text(
        json.dumps({"kind": "not_the_kind", "key_count": 0, "entries": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )

    # Tamper: keep a valid kind but corrupt the sha256 of an entry so the strict
    # reload fingerprint comparison fails.
    from mix_bit.model_adapter import get_model_adapter
    from mix_bit.state_fingerprint import fingerprint_model_state
    from train_utils.model_checkpoint_io import load_checkpoint_into_model

    reloaded = get_model_adapter(resolved.config.model_profile.adapter).load_model(
        resolved.config.model_profile
    )
    load_checkpoint_into_model(model=reloaded, model_dir=str(out_dir), map_location="cpu", strict=True)
    valid_manifest = fingerprint_model_state(reloaded)
    first_key = next(iter(valid_manifest["entries"]))
    valid_manifest["entries"][first_key]["sha256"] = "0" * 64
    fingerprint_path.write_text(
        json.dumps(valid_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_skip_identical_returns_true_when_all_files_valid_and_reload_hash_matches(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.state_fingerprint import STATE_FINGERPRINT_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(first["output_dir"])
    assert (out_dir / STATE_FINGERPRINT_FILENAME).is_file()

    second = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    assert second["skipped_identical"] is True
    assert second["state_fingerprint"] == str((out_dir / STATE_FINGERPRINT_FILENAME).resolve())
    assert second["assignment_count"] == len(assignments)


# --- Tokenizer fingerprint v2 save/reload tests (Task 9 Step 3) ---

def test_final_checkpoint_saves_and_reloads_tokenizer_with_matching_fingerprint(
    assembled_world, monkeypatch
):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from mix_bit.calibration import compute_tokenizer_config_sha256
    from mix_bit.model_adapter import AutoTokenizer, normalize_tokenizer_for_mix_bit
    from train_utils.model_checkpoint_io import META_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    _patch_tiny_tokenizer(monkeypatch)

    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(result["output_dir"])
    assert (out_dir / "tiny_tokenizer.json").is_file()
    # Local-only reload must succeed and fingerprint must match meta.
    reloaded = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=False)
    normalize_tokenizer_for_mix_bit(reloaded, source_label=str(out_dir))
    reload_sha = compute_tokenizer_config_sha256(reloaded)
    meta = json.loads((out_dir / META_FILENAME).read_text(encoding="utf-8"))
    mix = meta["extra_meta"]["mix_bit"]
    assert mix["tokenizer_fingerprint_version"] == 2
    assert mix["tokenizer_fingerprint_sha256"] == reload_sha
    assert mix["source_tokenizer_reported_name_or_path"] == "toy-model"
    assert result["tokenizer_fingerprint_sha256"] == reload_sha
    assert result["tokenizer_reported_name_or_path"] == str(out_dir.resolve())


def test_skip_identical_rejects_missing_tokenizer_files(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    _patch_tiny_tokenizer(monkeypatch)

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(first["output_dir"])
    (out_dir / "tiny_tokenizer.json").unlink()
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_skip_identical_rejects_legacy_tokenizer_fingerprint_version(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint
    from train_utils.model_checkpoint_io import META_FILENAME

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    _patch_tiny_tokenizer(monkeypatch)

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(first["output_dir"])
    meta_path = out_dir / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["extra_meta"]["mix_bit"]["tokenizer_fingerprint_version"] = 1
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


def test_skip_identical_rejects_tampered_tokenizer_content(assembled_world, monkeypatch):
    from mix_bit.assembler import assemble_optimal_mixed_checkpoint

    resolved = assembled_world["resolved"]
    inventory = assembled_world["inventory"]
    pool_index = assembled_world["pool_index"]
    assignments = _mixed_assignments(inventory, pool_index)
    alloc_path = _allocation_path(resolved)
    _write_allocation_json(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
    )
    _patch_reload_model(monkeypatch, assembled_world["template"])
    _patch_tiny_tokenizer(monkeypatch)

    first = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=assembled_world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out_dir = Path(first["output_dir"])
    marker = out_dir / "tiny_tokenizer.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    data["vocab_seed"] = 99
    marker.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=assembled_world["inventory_path"],
            pool_index=pool_index,
            allocation_path=str(alloc_path),
            device="cpu",
        )


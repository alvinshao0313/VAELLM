from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import SparseBitTuningConfig, active_count, resolve_round_steps, resolve_stable_steps
from .module import BankSpec, SparseBitTuningModule
from .optimizer import BitOptimizerManager
from .packed_ops import PackedBitRuntimeMeta, initialize_scores_from_packed, project_scores_to_packed
from .sampler import AffineSamplerState


@dataclass(frozen=True)
class SparseBitStepTelemetry:
    global_bit_round: int
    bit_round_step: int
    step_flip_count: int
    cumulative_flip_count: int
    stable_counter: int
    stable_steps: int
    had_flip: bool
    round_ended: bool


class SparseBitModuleBinding:
    """Thin non-Module binding attached to one VAELinear.

    The binding never advances global round state by itself.  It only resolves the
    current score/meta for the exact packed residency used by the current decode and
    delegates delayed streaming transitions back to the manager.
    """

    def __init__(self, manager: "SparseBitTuningManager", module_path: str, module: nn.Module) -> None:
        self.manager = manager
        self.module_path = str(module_path)
        self.module = module

    @property
    def bank_specs(self) -> Tuple[BankSpec, ...]:
        return self.manager.module_bank_specs(self.module_path)

    def _score_view(self, spec: BankSpec) -> torch.Tensor:
        return self.manager.score_module.score_view(spec)

    def _serial_meta(self, spec: BankSpec, *, device: torch.device) -> PackedBitRuntimeMeta:
        state = self.manager.sampler_states[spec.canonical_key]
        return PackedBitRuntimeMeta.build(
            states=(state,),
            model_indices=(0,),
            score_offsets=(0,),
            logical_in_dim=int(spec.latent_dim),
            device=torch.device(device),
        )

    def _grouped_context(self, *, device: torch.device) -> tuple[torch.Tensor, PackedBitRuntimeMeta]:
        specs = tuple(sorted(self.bank_specs, key=lambda s: int(s.score_start)))
        if not specs:
            raise RuntimeError(f"{self.module_path}: no Sparse Bit banks.")
        chunk_ids = {int(spec.chunk_id) for spec in specs}
        if len(chunk_ids) != 1:
            raise RuntimeError(f"{self.module_path}: banks span multiple score chunks.")
        latent_dims = {int(spec.latent_dim) for spec in specs}
        if len(latent_dims) != 1:
            raise RuntimeError(
                f"{self.module_path}: grouped Sparse Bit decode requires identical latent dims, got {sorted(latent_dims)}."
            )
        model_indices_plan = getattr(self.module, "_parallel_stage_model_indices", None)
        if not isinstance(model_indices_plan, torch.Tensor):
            raise RuntimeError(f"{self.module_path}: parallel stage model-index plan is missing.")
        model_indices = tuple(
            int(model_indices_plan[int(spec.stage_idx), int(spec.part_idx)].item()) for spec in specs
        )
        span_start = min(int(spec.score_start) for spec in specs)
        span_end = max(int(spec.score_end) for spec in specs)
        score = self.manager.score_module.score_chunks[int(specs[0].chunk_id)][span_start:span_end]
        offsets = tuple(int(spec.score_start) - span_start for spec in specs)
        states = tuple(self.manager.sampler_states[spec.canonical_key] for spec in specs)
        meta = PackedBitRuntimeMeta.build(
            states=states,
            model_indices=model_indices,
            score_offsets=offsets,
            logical_in_dim=next(iter(latent_dims)),
            device=torch.device(device),
        )
        return score, meta

    def prepare_forward(
        self,
        packed: torch.Tensor,
        *,
        grouped: bool,
        stage_idx: Optional[int],
        part_idx: Optional[int],
        training: bool,
        grad_enabled: bool,
    ) -> tuple[torch.Tensor, PackedBitRuntimeMeta]:
        def resolve_current() -> tuple[torch.Tensor, PackedBitRuntimeMeta]:
            if grouped:
                return self._grouped_context(device=packed.device)
            if stage_idx is None or part_idx is None:
                raise RuntimeError(f"{self.module_path}: serial Sparse Bit decode requires stage_idx/part_idx.")
            spec = self.manager.bank_spec(self.module_path, int(stage_idx), int(part_idx))
            return self._score_view(spec), self._serial_meta(spec, device=packed.device)

        if bool(self.manager.streaming) and self.manager.module_has_pending_transition(self.module_path):
            # The newly prefetched packed residency still comes from the old CPU baseline.
            # First materialize the just-finished old round into THIS residency.  Otherwise
            # old-round flips that are absent from the next active subset would disappear.
            old_score, old_meta = resolve_current()
            project_scores_to_packed(packed, old_score, old_meta)
            if bool(training) and bool(grad_enabled):
                self.manager.complete_streaming_transition(self.module_path)
                score, meta = resolve_current()
                # New scores were initialized from the committed baseline.  SET is still
                # required because this residency was created before the CPU commit.
                project_scores_to_packed(packed, score, meta)
                return score, meta
            # eval/no_grad intentionally evaluates old-round post-step state and leaves
            # lifecycle pending until the next real training forward.
            return old_score, old_meta

        score, meta = resolve_current()
        if bool(self.manager.streaming):
            # Streaming packed residency may have been freshly copied from a CPU baseline.
            # SET is idempotent, so repeated checkpoint recompute/eval calls are correct.
            project_scores_to_packed(packed, score, meta)
        return score, meta

    @torch.no_grad()
    def initialize_scores(self) -> None:
        # Streaming must not create a long-lived grouped GPU residency during init.
        if not bool(self.manager.streaming) and bool(getattr(self.module, "parallel_stage_decode", False)):
            grouped = getattr(self.module, "_parallel_stage_grouped_vq_packed", None)
            if isinstance(grouped, torch.Tensor):
                target = self.manager.module_device(self.module_path)
                packed = self.module._get_parallel_stage_grouped_vq_packed(device=target)
                score, meta = self._grouped_context(device=packed.device)
                initialize_scores_from_packed(packed, score, meta)
                return
        for spec in self.bank_specs:
            self.manager.initialize_bank_score_from_persistent(spec)

    @torch.no_grad()
    def project_nonstreaming_current(self) -> int:
        if bool(self.manager.streaming):
            raise RuntimeError("project_nonstreaming_current called in streaming mode.")
        if bool(getattr(self.module, "parallel_stage_decode", False)):
            grouped = getattr(self.module, "_parallel_stage_grouped_vq_packed", None)
            if isinstance(grouped, torch.Tensor):
                target = self.manager.module_device(self.module_path)
                packed = self.module._get_parallel_stage_grouped_vq_packed(device=target)
                score, meta = self._grouped_context(device=packed.device)
                counter = project_scores_to_packed(packed, score, meta)
                return int(counter.item())
        total = 0
        for spec in self.bank_specs:
            storage = self.module.get_stage_part_vq_storage(
                stage_idx=int(spec.stage_idx), part_idx=int(spec.part_idx)
            )
            if storage.device != spec.device:
                raise RuntimeError(
                    f"{spec.canonical_key}: non-streaming packed storage is on {storage.device}, "
                    f"expected score device {spec.device}."
                )
            meta = self._serial_meta(spec, device=storage.device)
            counter = project_scores_to_packed(storage, self._score_view(spec), meta)
            total += int(counter.item())
        return int(total)


class SparseBitTuningManager:
    def __init__(
        self,
        *,
        root_model: nn.Module,
        targets: Sequence[tuple[str, nn.Module]],
        target_devices: Mapping[str, torch.device],
        training_seed: int,
        config: SparseBitTuningConfig,
        streaming: bool,
    ) -> None:
        self.root_model = root_model
        self.config = config.normalized()
        if not bool(self.config.enabled):
            raise ValueError("SparseBitTuningManager requires config.enabled=true.")
        self.training_seed = int(training_seed)
        self.streaming = bool(streaming)
        self._modules: Dict[str, nn.Module] = {}
        raw_specs: list[BankSpec] = []
        for module_path, module in targets:
            path = str(module_path)
            if path in self._modules:
                raise ValueError(f"duplicate Sparse Bit target module path: {path}")
            self._modules[path] = module
            if path not in target_devices:
                raise ValueError(f"missing target device for Sparse Bit module {path!r}.")
            device = torch.device(target_devices[path])
            residual_stages = int(getattr(module, "residual_stages", 0))
            parallel_parts = int(getattr(module, "parallel_parts", 0))
            if residual_stages < 1 or parallel_parts < 1:
                raise ValueError(
                    f"{path}: invalid residual_stages/parallel_parts={residual_stages}/{parallel_parts}."
                )
            for stage_idx in range(residual_stages):
                for part_idx in range(parallel_parts):
                    storage = module.get_stage_part_vq_storage(stage_idx=stage_idx, part_idx=part_idx)
                    spec = module.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx)
                    logical_shape = tuple(int(v) for v in spec["logical_shape"])
                    if len(logical_shape) != 3 or int(logical_shape[1]) != 1:
                        raise ValueError(
                            f"{path}|stage={stage_idx}|part={part_idx}: unsupported logical_shape={logical_shape}."
                        )
                    if storage.dtype != torch.uint8:
                        raise ValueError(
                            f"{path}|stage={stage_idx}|part={part_idx}: Sparse Bit requires normal uint8 packed payload, "
                            f"got {storage.dtype}."
                        )
                    B, _one, latent_dim = logical_shape
                    expected_shape = (B, 1, (latent_dim + 7) // 8)
                    if tuple(int(v) for v in storage.shape) != expected_shape:
                        raise ValueError(
                            f"{path}|stage={stage_idx}|part={part_idx}: packed shape {tuple(storage.shape)} "
                            f"!= expected {expected_shape}."
                        )
                    n_bits = int(B) * int(latent_dim)
                    key = f"{path}|stage={stage_idx}|part={part_idx}"
                    raw_specs.append(
                        BankSpec(
                            canonical_key=key,
                            module_path=path,
                            stage_idx=int(stage_idx),
                            part_idx=int(part_idx),
                            logical_shape=logical_shape,
                            n_bits=n_bits,
                            n_active=active_count(n_bits, self.config.active_ratio),
                            device=device,
                        )
                    )
        if not raw_specs:
            raise ValueError("Sparse Bit target selection produced zero normal packed banks.")
        if hasattr(root_model, "sparse_bit_tuning"):
            raise RuntimeError("root model already has sparse_bit_tuning attribute.")
        self.score_module = SparseBitTuningModule(raw_specs)
        root_model.add_module("sparse_bit_tuning", self.score_module)
        self._bank_specs: Tuple[BankSpec, ...] = self.score_module.bank_specs
        self._bank_by_identity: Dict[tuple[str, int, int], BankSpec] = {
            (spec.module_path, int(spec.stage_idx), int(spec.part_idx)): spec for spec in self._bank_specs
        }
        self.sampler_states: Dict[str, AffineSamplerState] = {
            spec.canonical_key: AffineSamplerState.create(
                canonical_key=spec.canonical_key,
                training_seed=self.training_seed,
                n_bits=int(spec.n_bits),
                n_active=int(spec.n_active),
            )
            for spec in self._bank_specs
        }
        self.pending_next_states: Dict[str, AffineSamplerState] = {}
        self.bindings: Dict[str, SparseBitModuleBinding] = {}
        for path, module in self._modules.items():
            binding = SparseBitModuleBinding(self, path, module)
            if getattr(module, "_sparse_bit_binding", None) is not None:
                raise RuntimeError(f"{path}: existing _sparse_bit_binding detected.")
            module._sparse_bit_binding = binding
            self.bindings[path] = binding
        self.bit_optimizer = BitOptimizerManager(self.score_module, self.config)
        self.bit_round_steps: Optional[int] = None
        self.stable_steps: Optional[int] = None
        self.global_bit_round = 0
        self.bit_round_step = 0
        self.stable_counter = 0
        self.cumulative_flip_count = 0
        self.had_flip = False
        self._initialized_scores = False

    @property
    def bank_specs(self) -> Tuple[BankSpec, ...]:
        return self._bank_specs

    def module_bank_specs(self, module_path: str) -> Tuple[BankSpec, ...]:
        return tuple(spec for spec in self._bank_specs if spec.module_path == str(module_path))

    def bank_spec(self, module_path: str, stage_idx: int, part_idx: int) -> BankSpec:
        try:
            return self._bank_by_identity[(str(module_path), int(stage_idx), int(part_idx))]
        except KeyError as exc:
            raise KeyError(
                f"unknown Sparse Bit bank: {module_path}|stage={stage_idx}|part={part_idx}"
            ) from exc

    def module_device(self, module_path: str) -> torch.device:
        specs = self.module_bank_specs(module_path)
        devices = {torch.device(spec.device) for spec in specs}
        if len(devices) != 1:
            raise RuntimeError(f"{module_path}: banks span devices {sorted(str(x) for x in devices)}.")
        return next(iter(devices))

    def configure_schedule(self, *, total_optimizer_steps: int) -> None:
        self.bit_round_steps = resolve_round_steps(
            self.config.round_steps,
            total_optimizer_steps=int(total_optimizer_steps),
            active_ratio=float(self.config.active_ratio),
        )
        self.stable_steps = resolve_stable_steps(int(self.bit_round_steps))

    @torch.no_grad()
    def initialize_scores(self) -> None:
        if self._initialized_scores:
            return
        for binding in self.bindings.values():
            binding.initialize_scores()
        self.score_module.mark_initialized()
        self._initialized_scores = True

    def _serial_meta(self, spec: BankSpec, *, device: torch.device) -> PackedBitRuntimeMeta:
        return PackedBitRuntimeMeta.build(
            states=(self.sampler_states[spec.canonical_key],),
            model_indices=(0,),
            score_offsets=(0,),
            logical_in_dim=int(spec.latent_dim),
            device=torch.device(device),
        )

    @torch.no_grad()
    def initialize_bank_score_from_persistent(self, spec: BankSpec) -> None:
        module = self._modules[spec.module_path]
        storage = module.get_stage_part_vq_storage(
            stage_idx=int(spec.stage_idx), part_idx=int(spec.part_idx)
        )
        score = self.score_module.score_view(spec)
        if storage.device == score.device and storage.is_contiguous():
            packed = storage
        else:
            packed = storage.detach().to(device=score.device, dtype=torch.uint8).contiguous()
        meta = self._serial_meta(spec, device=score.device)
        initialize_scores_from_packed(packed, score, meta)

    @torch.no_grad()
    def commit_bank_to_persistent(self, spec: BankSpec) -> None:
        module = self._modules[spec.module_path]
        storage = module.get_stage_part_vq_storage(
            stage_idx=int(spec.stage_idx), part_idx=int(spec.part_idx)
        )
        score = self.score_module.score_view(spec)
        if storage.device == score.device and storage.is_contiguous():
            packed = storage
            copy_back = False
        else:
            packed = storage.detach().to(device=score.device, dtype=torch.uint8).contiguous()
            copy_back = True
        meta = self._serial_meta(spec, device=score.device)
        project_scores_to_packed(packed, score, meta)
        if copy_back:
            storage.copy_(packed.to(device=storage.device, dtype=torch.uint8))

    @torch.no_grad()
    def commit_module_to_persistent(self, module_path: str) -> None:
        for spec in self.module_bank_specs(module_path):
            self.commit_bank_to_persistent(spec)

    def module_has_pending_transition(self, module_path: str) -> bool:
        return any(
            spec.canonical_key in self.pending_next_states for spec in self.module_bank_specs(module_path)
        )

    @torch.no_grad()
    def complete_streaming_transition(self, module_path: str) -> None:
        if not bool(self.streaming):
            raise RuntimeError("complete_streaming_transition called outside streaming mode.")
        specs = self.module_bank_specs(module_path)
        if not any(spec.canonical_key in self.pending_next_states for spec in specs):
            return
        if not all(spec.canonical_key in self.pending_next_states for spec in specs):
            raise RuntimeError(f"{module_path}: partial pending Sparse Bit module transition.")
        # Current score still represents the old round. Commit it before replacing sampler state.
        self.commit_module_to_persistent(module_path)
        for spec in specs:
            self.sampler_states[spec.canonical_key] = self.pending_next_states.pop(spec.canonical_key)
        self.bit_optimizer.reset_bank_state(specs)
        for spec in specs:
            self.initialize_bank_score_from_persistent(spec)

    @torch.no_grad()
    def _finish_nonstreaming_round(self) -> None:
        for module_path in self._modules:
            self.commit_module_to_persistent(module_path)
        self.sampler_states = {
            key: state.advance() for key, state in self.sampler_states.items()
        }
        self.bit_optimizer.reset_round_state()
        for binding in self.bindings.values():
            binding.initialize_scores()

    @torch.no_grad()
    def _mark_streaming_round_pending(self) -> None:
        if self.pending_next_states:
            raise RuntimeError("Sparse Bit streaming round ended while a previous transition is still pending.")
        self.pending_next_states = {
            key: state.advance() for key, state in self.sampler_states.items()
        }

    @staticmethod
    def _sum_scalar_counters(counters: Iterable[torch.Tensor]) -> int:
        return int(sum(int(counter.item()) for counter in counters))

    @torch.no_grad()
    def optimizer_step(self) -> SparseBitStepTelemetry:
        if self.bit_round_steps is None or self.stable_steps is None:
            raise RuntimeError("Sparse Bit schedule must be configured before optimizer_step().")
        if not self._initialized_scores:
            raise RuntimeError("Sparse Bit scores must be initialized before optimizer_step().")
        if self.pending_next_states:
            pending_modules = sorted(
                {spec.module_path for spec in self._bank_specs if spec.canonical_key in self.pending_next_states}
            )
            raise RuntimeError(
                "Sparse Bit optimizer step reached with pending streaming transitions; target modules did not all execute "
                f"in the preceding training forward: {pending_modules}"
            )
        next_step = int(self.bit_round_step) + 1
        score_flip_counters = self.bit_optimizer.step_scores(optimizer_step_in_round=next_step)
        if bool(self.streaming):
            step_flips = self._sum_scalar_counters(score_flip_counters)
        else:
            step_flips = sum(binding.project_nonstreaming_current() for binding in self.bindings.values())

        self.bit_round_step = next_step
        self.cumulative_flip_count += int(step_flips)
        if int(step_flips) > 0:
            self.had_flip = True
            self.stable_counter = 0
        elif bool(self.had_flip):
            self.stable_counter += 1

        ended = bool(
            int(self.bit_round_step) >= int(self.bit_round_steps)
            or (bool(self.had_flip) and int(self.stable_counter) >= int(self.stable_steps))
        )
        old_round = int(self.global_bit_round)
        telemetry = SparseBitStepTelemetry(
            global_bit_round=old_round,
            bit_round_step=int(self.bit_round_step),
            step_flip_count=int(step_flips),
            cumulative_flip_count=int(self.cumulative_flip_count),
            stable_counter=int(self.stable_counter),
            stable_steps=int(self.stable_steps),
            had_flip=bool(self.had_flip),
            round_ended=ended,
        )
        if ended:
            if bool(self.streaming):
                self._mark_streaming_round_pending()
            else:
                self._finish_nonstreaming_round()
            self.global_bit_round += 1
            self.bit_round_step = 0
            self.stable_counter = 0
            self.cumulative_flip_count = 0
            self.had_flip = False
        return telemetry

    @torch.no_grad()
    def final_commit(self) -> None:
        # If a streaming round just ended, current score still represents the just-finished
        # old round; do not initialize an unused next subset before final save.
        for module_path in self._modules:
            self.commit_module_to_persistent(module_path)
        self.pending_next_states.clear()

    @torch.no_grad()
    def checkpoint_packed_snapshot(self) -> Dict[str, torch.Tensor]:
        """Materialize current logical hard bits without mutating the live round/baseline."""
        if not self._initialized_scores:
            raise RuntimeError("Sparse Bit scores must be initialized before checkpoint snapshot.")
        snapshot: Dict[str, torch.Tensor] = {}
        for spec in self._bank_specs:
            module = self._modules[spec.module_path]
            storage = module.get_stage_part_vq_storage(
                stage_idx=int(spec.stage_idx), part_idx=int(spec.part_idx)
            )
            score = self.score_module.score_view(spec)
            packed = storage.detach().to(device=score.device, dtype=torch.uint8).contiguous().clone()
            meta = self._serial_meta(spec, device=packed.device)
            project_scores_to_packed(packed, score, meta)
            snapshot[spec.canonical_key] = packed.to(device="cpu", dtype=torch.uint8).contiguous()
        return snapshot

    @torch.no_grad()
    def restore_checkpoint_packed(self, packed_banks: Mapping[str, torch.Tensor]) -> None:
        expected = {spec.canonical_key for spec in self._bank_specs}
        provided = {str(key) for key in packed_banks}
        if provided != expected:
            raise ValueError(
                "Sparse Bit packed sidecar bank set mismatch: "
                f"missing={sorted(expected - provided)} extra={sorted(provided - expected)}"
            )
        touched_modules = set()
        for spec in self._bank_specs:
            source = packed_banks[spec.canonical_key]
            if not isinstance(source, torch.Tensor) or source.dtype != torch.uint8:
                raise TypeError(
                    f"{spec.canonical_key}: checkpoint packed payload must be uint8 Tensor, "
                    f"got {type(source)}/{getattr(source, 'dtype', None)}."
                )
            module = self._modules[spec.module_path]
            target = module.get_stage_part_vq_storage(
                stage_idx=int(spec.stage_idx), part_idx=int(spec.part_idx)
            )
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"{spec.canonical_key}: checkpoint packed shape {tuple(source.shape)} "
                    f"!= target {tuple(target.shape)}."
                )
            target.copy_(source.to(device=target.device, dtype=torch.uint8))
            touched_modules.add(spec.module_path)
        for module_path in sorted(touched_modules):
            module = self._modules[module_path]
            if bool(getattr(module, "parallel_stage_decode", False)) and getattr(
                module, "_parallel_stage_decoder", None
            ) is not None:
                module._build_parallel_stage_decode_plan()
            else:
                module.clear_decoded_weight_cache()

    def restore_coverage_metadata(self, metadata: dict) -> None:
        if str(metadata.get("format")) != "sparse_bit_tuning_coverage" or int(
            metadata.get("version", -1)
        ) != 1:
            raise ValueError(
                f"unsupported Sparse Bit coverage format/version: "
                f"{metadata.get('format')!r}/{metadata.get('version')!r}."
            )
        if int(metadata.get("training_seed", -1)) != int(self.training_seed):
            raise ValueError(
                f"Sparse Bit training seed mismatch: checkpoint={metadata.get('training_seed')} "
                f"current={self.training_seed}."
            )
        if float(metadata.get("bit_active_ratio", -1.0)) != float(self.config.active_ratio):
            raise ValueError(
                f"Sparse Bit active ratio mismatch: checkpoint={metadata.get('bit_active_ratio')} "
                f"current={self.config.active_ratio}."
            )
        expected_targets = sorted(self._modules)
        checkpoint_targets = sorted(str(x) for x in metadata.get("target_modules", []))
        if checkpoint_targets != expected_targets:
            raise ValueError(
                f"Sparse Bit target set mismatch: checkpoint={checkpoint_targets} current={expected_targets}."
            )
        raw_banks = metadata.get("banks")
        if not isinstance(raw_banks, list):
            raise TypeError("Sparse Bit coverage metadata banks must be a list.")
        by_key = {str(item.get("canonical_key")): item for item in raw_banks if isinstance(item, dict)}
        expected_keys = {spec.canonical_key for spec in self._bank_specs}
        if set(by_key) != expected_keys:
            raise ValueError(
                "Sparse Bit coverage bank set mismatch: "
                f"missing={sorted(expected_keys - set(by_key))} extra={sorted(set(by_key) - expected_keys)}"
            )
        restored: Dict[str, AffineSamplerState] = {}
        for spec in self._bank_specs:
            state = AffineSamplerState.from_metadata(by_key[spec.canonical_key])
            if int(state.n_bits) != int(spec.n_bits) or int(state.n_active) != int(spec.n_active):
                raise ValueError(
                    f"{spec.canonical_key}: checkpoint N_bits/N_active={state.n_bits}/{state.n_active} "
                    f"!= current={spec.n_bits}/{spec.n_active}."
                )
            restored[spec.canonical_key] = state
        self.sampler_states = restored
        self.pending_next_states.clear()
        self.global_bit_round = int(metadata.get("global_bit_round", 0))
        self.bit_round_step = 0
        self.stable_counter = 0
        self.cumulative_flip_count = 0
        self.had_flip = False
        self.bit_optimizer.reset_round_state()
        self._initialized_scores = False
        self.score_module._initialized = False

    def detach_runtime(self) -> None:
        for path, module in self._modules.items():
            if getattr(module, "_sparse_bit_binding", None) is self.bindings.get(path):
                delattr(module, "_sparse_bit_binding")
        if getattr(self.root_model, "sparse_bit_tuning", None) is self.score_module:
            delattr(self.root_model, "sparse_bit_tuning")
        self.bit_optimizer.clear_state()

    def coverage_metadata(self) -> dict:
        states = self.pending_next_states if self.pending_next_states else self.sampler_states
        return {
            "format": "sparse_bit_tuning_coverage",
            "version": 1,
            "global_bit_round": int(self.global_bit_round),
            "training_seed": int(self.training_seed),
            "bit_active_ratio": float(self.config.active_ratio),
            "target_modules": sorted(self._modules),
            "banks": [states[spec.canonical_key].to_metadata() for spec in self._bank_specs],
        }

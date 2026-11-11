"""Original native CAT Trainer, with retained 125-step nodes and state audits."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from transformers import TrainerCallback

from train_utils.cat_after_category_common import CanonicalCatSFTTrainer
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state

from .nodes import RecoverySettings
from .identity import semantic_seed


def recovery_seeds(model_identity, category_id: str, search_seed: int) -> dict:
    return {purpose: semantic_seed(model_identity, category_id, "recovery", purpose, search_seed)
        for purpose in ("lora_init", "dropout", "data")}


def _tensor_hash(values) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def training_snapshot(trainer) -> dict:
    """Same preservation boundary as MixedCatTrainer, using native component state."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    values, _components, _manifest = collect_model_level_mutable_state(model,
        selection=trainer.model_level_trainable_selection, selected_vae_modules=trainer._v6_selected_vae_modules)
    optimizer = trainer.optimizer
    if optimizer is None or not optimizer.state:
        raise ValueError("A node save must follow a real optimizer step with nonempty state.")
    opt = optimizer.state_dict()
    tensors, scalars = {}, {}
    for owner, fields in opt["state"].items():
        for field, value in fields.items():
            (tensors if isinstance(value, torch.Tensor) else scalars)[f"{owner}/{field}"] = value
    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        rng.update({f"cuda{index}": value for index, value in enumerate(torch.cuda.get_rng_state_all())})
    return dict(mutable=_tensor_hash(values), optimizer_tensors=_tensor_hash(tensors),
        optimizer_scalars=scalars, optimizer_groups=deepcopy(opt["param_groups"]),
        optimizer_identity=(id(optimizer), tuple((id(parameter), id(fields), tuple((key, id(value))
            for key, value in fields.items())) for parameter, fields in optimizer.state.items())),
        scheduler=deepcopy(trainer.lr_scheduler.state_dict()), rng=_tensor_hash(rng),
        host_rng=hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))).hexdigest(),
        modes=tuple((name, module.training) for name, module in model.named_modules()))


def configure_node_retention(args, settings: RecoverySettings) -> None:
    if args.max_steps != settings.schedule_max:
        raise ValueError("Build the original T-step scheduler; node retention must not shorten its domain.")
    if args.save_only_model or args.ignore_data_skip:
        raise ValueError("Keep native optimizer/scheduler/RNG and the original data-skip contract.")
    args.save_strategy = "steps"
    args.save_steps = 125
    args.save_total_limit = None
    args.load_best_model_at_end = False
    if str(getattr(args.eval_strategy, "value", args.eval_strategy)).lower() != "no":
        raise ValueError("Read-only full-eight evaluation runs on independent exports; in-Trainer evaluation is unsupported.")


class RetainRecoveryNodes(TrainerCallback):
    def __init__(self, settings: RecoverySettings, *, technical_stop_after: int | None = None):
        self.settings = settings
        self.technical_stop_after = technical_stop_after
        self.save_steps = 1 if technical_stop_after is not None else 125
        self.node_steps = (1, 2) if technical_stop_after is not None else settings.node_steps

    def on_train_begin(self, args, state, control, **kwargs):
        if args.max_steps != self.settings.schedule_max or args.save_steps != self.save_steps or args.save_total_limit is not None:
            raise ValueError("Retain every 125-step node of the unchanged 500/1000 trajectory until selection.")
        if args.load_best_model_at_end:
            raise ValueError("Complete the continuous trajectory before exporting and selecting nodes.")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.node_steps:
            control.should_save = True
        if self.technical_stop_after is not None and state.global_step == self.technical_stop_after:
            control.should_training_stop = True
        return control


class NativeNodeTrainer(CanonicalCatSFTTrainer):
    """Reuse native KD and v6 payloads; never finalize or evaluate the live master."""
    def __init__(self, *args, recovery_settings: RecoverySettings, semantic_seeds: dict,
                 input_observer=None, technical_stop_after: int | None = None,
                 bound_iterable_loader: bool = False, **kwargs):
        from .live import source_pins
        source_pins()
        if set(semantic_seeds) != {"lora_init", "dropout", "data"} or any(not isinstance(value, int) or not 0 <= value < 2**32 for value in semantic_seeds.values()):
            raise ValueError("Use the three shared semantic recovery streams, independent of raw/T/LR/scope and run order.")
        if "args" not in kwargs:
            raise ValueError("Pass TrainingArguments explicitly by keyword before Trainer initializes its random streams.")
        kwargs["args"].seed = semantic_seeds["dropout"]
        kwargs["args"].data_seed = semantic_seeds["data"]
        super().__init__(*args, **kwargs)
        self.recovery_settings = recovery_settings
        self.semantic_seeds = dict(semantic_seeds)
        self.input_observer = input_observer
        self.bound_iterable_loader = bool(bound_iterable_loader)
        if technical_stop_after not in (None, 2):
            raise ValueError("The isolated save1-return-to-step2 pilot has exactly two updates.")
        self.technical_stop_after = technical_stop_after
        configure_node_retention(self.args, recovery_settings)
        if technical_stop_after is not None:
            self.args.save_steps = 1
        self.add_callback(RetainRecoveryNodes(recovery_settings, technical_stop_after=technical_stop_after))
        self.node_save_audits: list[dict] = []

    def get_train_dataloader(self):
        if not self.bound_iterable_loader:
            return super().get_train_dataloader()
        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            raise TypeError("The root-frozen native KD loader requires its actual canonical IterableDataset.")
        if self.args.per_device_train_batch_size != 4 or self.args.dataloader_num_workers != 16 or self.args.group_by_length or self.args.gradient_accumulation_steps != 1:
            raise ValueError("Keep original B4/16workers/Iterable/GAS1 data consumption.")
        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise NotImplementedError("The frozen data trace currently covers one recovery process; multi-rank data partitioning needs its own actual trace.")
        loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=4,
            num_workers=16, collate_fn=self.data_collator, drop_last=False,
            pin_memory=self.args.dataloader_pin_memory,
            generator=torch.Generator().manual_seed(self.semantic_seeds["data"]))
        return self.accelerator.prepare(loader)

    def configure_v6_step_checkpoint(self, *, context: dict, selected_vae_modules) -> None:
        context = deepcopy(context)
        context.setdefault("runtime_audit", {})["cat_auto_recovery"] = dict(
            semantic_seeds=self.semantic_seeds, schedule_max=self.recovery_settings.schedule_max,
            selected_scope=self.recovery_settings.scope, native_source=str(Path(__file__).resolve()),
            technical_stop_after=self.technical_stop_after, formal_eligible=self.technical_stop_after is None)
        super().configure_v6_step_checkpoint(context=context, selected_vae_modules=selected_vae_modules)

    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.input_observer is not None:
            self.input_observer(inputs, int(self.state.global_step))
        return super().compute_loss(model, inputs, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        if not self._v6_step_checkpoint_enabled():
            raise ValueError("Configure the original native v6 round_base/immutable contract before training.")
        allowed = (1, 2) if self.technical_stop_after is not None else self.recovery_settings.node_steps
        if self.state.global_step not in allowed:
            raise ValueError("Only the confirmed 125-step node grid is saved by this search Trainer.")
        before = training_snapshot(self)
        result = super()._save_checkpoint(model, trial)
        after = training_snapshot(self)
        if before != after:
            raise RuntimeError("Saving changed mutable masters, optimizer objects/values, scheduler, mode or training RNG.")
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        required = ("training_model_state.pt", "checkpoint_meta.json", "optimizer.pt", "scheduler.pt", "trainer_state.json")
        if not all((directory / name).is_file() for name in required) or not list(directory.glob("rng_state*.pth")):
            raise RuntimeError("The original native node omitted required model/optimizer/scheduler/RNG/trainer files.")
        saved = torch.load(directory / "training_model_state.pt", map_location="cpu", weights_only=True)
        if _tensor_hash(saved) != before["mutable"]:
            raise RuntimeError("Serialized all-mutable state differs from the actual save point.")
        self.node_save_audits.append(dict(step=int(self.state.global_step), schedule_max=self.args.max_steps,
            directory=str(directory), live_state_unchanged=True, all_mutable_exact=True,
            model_format="vaellm_checkpoint_v6", exact_resume_verified=False,
            technical_stop_after=self.technical_stop_after, formal_eligible=False))
        return result


def retained_node_directories(trainer_directory: str | Path, settings: RecoverySettings) -> tuple[Path, ...]:
    directories = tuple(Path(trainer_directory) / f"checkpoint-{step}" for step in settings.node_steps)
    for directory in directories:
        if not (directory / "checkpoint_meta.json").is_file() or not (directory / "training_model_state.pt").is_file():
            raise FileNotFoundError(f"The complete node grid must remain present until selection: {directory}")
    return directories

"""Mixed training-step payload; HF still owns Adam/scheduler/RNG/state saving."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from train_utils.cat_after_category_common import CanonicalCatSFTTrainer
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state
from experiments.adaptive_vector_rvq import common as c
from . import checkpoint


def mutable_state(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    selected = trainer.model_level_trainable_selection
    values, components, _native_manifest = collect_model_level_mutable_state(model,
        selection=selected, selected_vae_modules=trainer.selected_native)
    buffers = {id(value): (name, value) for name, value in model.named_buffers()}
    persistent = set(model.state_dict())
    for _name, module in trainer.selected_compact:
        for decoder in module.decoders:
            for value in decoder.buffers():
                name, actual = buffers[id(value)]
                if name in persistent:
                    values[name], components[name] = actual, 'decoder'
    return values, components


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    tensors = {}
    scalars = {}
    for parameter, fields in state['state'].items():
        for field, value in fields.items():
            key = f'{parameter}/{field}'
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                scalars[key] = value
    return dict(tensors=c.digest(tensors), scalars=scalars, groups=deepcopy(state['param_groups']))


def training_snapshot(trainer):
    values, _components = mutable_state(trainer)
    optimizer = trainer.optimizer
    c.require(optimizer is not None and optimizer.state, 'A real mixed step save requires nonempty Adam state.')
    identity = (id(optimizer), tuple((id(parameter), id(fields), tuple((key, id(value)) for key, value in fields.items()))
                                  for parameter, fields in optimizer.state.items()))
    rng = dict(cpu=torch.get_rng_state(), **{f'cuda{index}': value for index, value in enumerate(torch.cuda.get_rng_state_all())})
    return dict(masters=c.digest(values), optimizer=optimizer_digest(optimizer), object_identity=identity,
                scheduler=deepcopy(trainer.lr_scheduler.state_dict()), tensor_rng=c.digest(rng),
                host_rng=hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))).hexdigest())


class MixedCatTrainer(CanonicalCatSFTTrainer):
    def __init__(self, *args, mixed_round_base, mixed_round_meta, selected_native, selected_compact, regular_save, **kwargs):
        super().__init__(*args, **kwargs)
        c.require(not self._v6_step_checkpoint_enabled() and not self.args.save_only_model,
                  'Mixed state uses its own full-D/LoRA payload and the original HF optimizer/RNG saving.')
        self.mixed_round_base = Path(mixed_round_base).resolve()
        self.mixed_round_meta = mixed_round_meta
        self.selected_native, self.selected_compact = selected_native, selected_compact
        self.regular_save = regular_save
        self.save_audits = []

    def save_model(self, output_dir=None, _internal_call=False):
        target = Path(output_dir or self.args.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        values, components = mutable_state(self)
        saved = {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}
        c.require(all(value.dtype == torch.float32 for name, value in saved.items() if name in dict(self.model.named_parameters())),
                  'Do not cast actual FP32 D/LoRA masters into a deployment checkpoint during training.')
        torch.save(saved, target / 'training_model_state.pt')
        torch.save(self.args, target / 'training_args.bin')
        self.processing_class.save_pretrained(target)
        manifest = dict(format='vaellm_mixed_training_step_v1', actual_global_step=int(self.state.global_step),
            round_base_ref=os.path.relpath(self.mixed_round_base, target),
            round_base_checkpoint_id=self.mixed_round_meta['checkpoint_id'],
            round_base_meta_sha256=c.sha(self.mixed_round_base / checkpoint.META_FILE),
            round_base_format=checkpoint.FORMAT, runtime=checkpoint.runtime_identity(),
            tensor_table=checkpoint.tensor_table(saved), component_by_actual_parameter_name=components,
            master_tensor_digest=c.digest(saved), saved_float_dtypes=sorted({str(value.dtype) for value in saved.values() if value.is_floating_point()}),
            regular_save_contract=self.regular_save, actual_trigger_steps=self.args.save_steps,
            actual_schedule_max_steps=self.args.max_steps, exact_resume_implemented=False,
            precision='Persist actual native/compact FP32 D and dense PEFT LoRA; no v6 or adapter-only declaration')
        c.write(target / 'mixed_training_meta.json', manifest)

    def _save_checkpoint(self, model, trial):
        before = training_snapshot(self)
        result = super()._save_checkpoint(model, trial)
        after = training_snapshot(self)
        c.require(before == after, 'Checkpoint saving changed actual masters, Adam objects/values, scheduler or RNG before returning to training.')
        directory = Path(self._get_output_dir(trial=trial)) / f'checkpoint-{self.state.global_step}'
        required = ('training_model_state.pt', 'mixed_training_meta.json', 'training_args.bin', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
        c.require(all((directory / name).is_file() for name in required), 'Mixed regular save omitted required actual training state.')
        payload = torch.load(directory / 'training_model_state.pt', map_location='cpu', weights_only=True)
        saved_optimizer = torch.load(directory / 'optimizer.pt', map_location='cpu', weights_only=True)
        saved_scheduler = torch.load(directory / 'scheduler.pt', map_location='cpu', weights_only=True)
        c.require(c.digest(payload) == before['masters'] and saved_optimizer['state']
                  and saved_scheduler == before['scheduler'], 'The actual serialized training state differs from the live save point.')
        live_state = self.optimizer.state_dict()
        c.require(set(saved_optimizer['state']) == set(live_state['state'])
                  and saved_optimizer['param_groups'] == live_state['param_groups'], 'Saved Adam inventory/groups changed.')
        for key, fields in saved_optimizer['state'].items():
            for field, value in fields.items():
                actual = live_state['state'][key][field]
                c.require(torch.equal(value, actual.detach().cpu()) if isinstance(value, torch.Tensor) else value == actual,
                          'Saved Adam tensor/scalar value differs from the live state.')
        audit = dict(actual_step=int(self.state.global_step), directory=str(directory), master_and_Adam_objects_values_unchanged=True,
            scheduler_and_RNG_unchanged=True, persisted_FP32_masters_exact=True, optimizer_leaves=len(saved_optimizer['state']),
            original_HF_optimizer_scheduler_RNG_trainer_state_saved=True, resume_tested=False,
            files={name: (directory / name).stat().st_size for name in required})
        self.save_audits.append(audit)
        c.write(Path(self.args.output_dir) / 'mixed_save_audits.json', self.save_audits)
        print(json.dumps(dict(event='mixed_checkpoint_saved_return_to_training', **audit)), flush=True)
        return result

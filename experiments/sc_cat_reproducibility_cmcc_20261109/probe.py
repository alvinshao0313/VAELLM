"""Compare real first-category KD prefixes with only FA2 backward determinism varied."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback
import types

import numpy as np
import torch
from transformers import modeling_flash_attention_utils as flash

from experiments.cat_auto_protocol_cmcc_20261104.native_builder import (
    RecoveryExecutionPolicy, make_native_recovery_trainer,
)
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_auto_protocol_cmcc_20261104.recovery import _tensor_hash
from experiments.sc_cat_live_cmcc_20261106.live_io import load_inputs


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rng():
    return dict(torch=_tensor_hash({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}),
        host=hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))).hexdigest())


def parameters(model, *, gradients=False):
    return {name: (value.grad if gradients else value).detach().cpu().contiguous().clone()
        for name, value in model.named_parameters()
        if value.requires_grad and (not gradients or value.grad is not None)}


def describe(values):
    groups = {}
    for label in ('lora_A', 'lora_B', 'decoder'):
        subset = {name: value for name, value in values.items() if label in name}
        groups[label] = dict(sha256=_tensor_hash(subset), tensors=len(subset),
            elements=sum(value.numel() for value in subset.values()),
            l2=sum(float(value.double().square().sum()) for value in subset.values()) ** .5)
    return dict(sha256=_tensor_hash(values), groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(status='running', phase='technical_validation',
        original_result=dict(path=str(Path(args.original_result).resolve()), sha256=sha(args.original_result)),
        implementation=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        started_epoch=started, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        flash_environment=os.environ['FLASH_ATTENTION_DETERMINISTIC'],
        flash_deterministic_global=flash.deterministic_g,
        flash_implementation=dict(path=flash.__file__, sha256=sha(flash.__file__)),
        torch_deterministic=torch.are_deterministic_algorithms_enabled(),
        steps=[], flash_calls={})
    run = None
    try:
        torch.set_num_threads(4)
        original = read(args.original_result)
        # Retain the original algorithm; fail instead of silently reading a newer consumer.
        pins = original['actual_run']['provenance']['source_pins']
        for path, expected in pins.items():
            if sha(path) != expected:
                raise RuntimeError('Original recovery consumer changed: ' + path)
        for name in ('flash_attn_func', 'flash_attn_varlen_func'):
            function = getattr(flash, name)

            @functools.wraps(function)
            def observed(*values, _function=function, _name=name, **kwargs):
                actual = kwargs.get('deterministic')
                if actual is not flash.deterministic_g:
                    raise RuntimeError('Actual FA2 call did not use the requested backward policy.')
                key = _name + ':' + str(actual)
                report['flash_calls'][key] = report['flash_calls'].get(key, 0) + 1
                return _function(*values, **kwargs)

            setattr(flash, name, observed)
        inventory, prepared, sources, tokenizer = load_inputs(original, with_sources=True)
        report['data_identity'] = prepared['data_identity']
        report['settings'] = original['settings']
        run = make_native_recovery_trainer(round_base_dir=original['live_raw']['round_base_dir'],
            inventory=inventory, category_id='k_proj', settings=RecoverySettings(**original['settings']),
            prepared_data=prepared, sources=sources, tokenizer=tokenizer, output_dir=output / 'train',
            execution_policy=RecoveryExecutionPolicy(**original['actual_run']['provenance']['execution_policy']),
            phase='technical_validation', technical_stop_after=2)
        report['semantic_seeds'] = run.provenance['semantic_seeds']
        initial = parameters(run.trainer.model)
        report['initial'] = describe(initial)
        torch.save(initial, output / 'initial.pt')
        del initial
        training_step = run.trainer.training_step
        compute_loss = run.trainer.compute_loss

        def loss_observer(self, *values, **kwargs):
            result = compute_loss(*values, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            report['steps'][-1]['loss'] = float(loss.detach())
            return result

        def step_observer(self, *values, **kwargs):
            step = int(self.state.global_step)
            row = dict(update_index=step, rng_before=rng())
            report['steps'].append(row)
            result = training_step(*values, **kwargs)
            row['rng_after'] = rng()
            grads = parameters(self.model, gradients=True)
            row['gradient'] = describe(grads)
            if step == 0:
                torch.save(grads, output / 'first_gradients.pt')
            return result

        run.trainer.compute_loss = types.MethodType(loss_observer, run.trainer)
        run.trainer.training_step = types.MethodType(step_observer, run.trainer)
        result = run.train()
        report['actual_run'] = result
        report['final'] = describe(parameters(run.trainer.model))
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if not report['flash_calls']:
            raise RuntimeError('No genuine FA2 call was observed.')
        report.update(status='completed', exit_code=0)
    except Exception:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - started
        (output / 'probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if run is not None:
            run.close()
    print(json.dumps({key: report[key] for key in ('status', 'elapsed_seconds', 'flash_calls')}), flush=True)


if __name__ == '__main__':
    main()

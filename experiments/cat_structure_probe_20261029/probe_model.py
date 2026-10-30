"""One same-parent, three-gate conditional readout on the existing64 train cases."""
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_structure_model_probe_common', HERE/'common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
sys.path.insert(0, str(c.SNAPSHOT))

import torch
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_v6 import _rebuild_converted_modules, refresh_vae_linear_runtime_after_state_load
from tools.cat_eval import _prepare_model_for_eval
import experiments
experiments.__path__.append(str(c.ROOT/'experiments'))
from experiments.adaptive_native_bridge_20261029.inputs import load_groups


def restore_endpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    c.require(value['format'] == 'cat_biaxial_packed_endpoint_v1' and not value['scaled'],
              'Use the actual native unscaled structure endpoint.')
    spec = {**value['native_spec'], 'name': 'linear'}
    host = torch.nn.Module()
    host.add_module('linear', torch.nn.Linear(spec['in_features'], spec['out_features'], bias=False, dtype=torch.bfloat16))
    _rebuild_converted_modules(host, [spec])
    state = {key[len('codec.'):]: tensor for key, tensor in value['state_dict'].items()}
    c.require(all(key.startswith('codec.') for key in value['state_dict']), 'Do not drop an extra learned scale or other field.')
    host.linear.load_state_dict(state, strict=True)
    layer = host.linear.to('cuda:0').eval().requires_grad_(False)
    layer._decoder_compute_dtype = torch.bfloat16
    refresh_vae_linear_runtime_after_state_load(layer)
    c.require(layer.prime_decoded_weight_cache(torch.bfloat16), 'Prime each actual new native gate once.')
    return layer


@torch.no_grad()
def main():
    destination = c.OUTPUT/'model_probe.json'
    c.require(not destination.exists() and os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit(),
              'Use a new result and one explicitly assigned GPU.')
    reports = {arm: c.read(c.OUTPUT/f'formal_{arm}'/'report.json') for arm in c.ARMS}
    for arm, report in reports.items():
        c.require(report['status'] == 'completed_native_probe' and report['exit_code'] == 0
                  and report['steps_per_stage'] == 10000 and report['source_checkpoint_id'] == c.SOURCE_ID,
                  'All four real finished architecture endpoints must exist before loading the model.')
        for item in report['endpoints'].values():
            c.require(c.sha(item['path']) == item['sha256'], 'A source endpoint changed.')
    backend, groups, evidence = load_groups()
    torch.set_num_threads(4)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model, meta, loaded = load_v6_model_checkpoint(str(c.SOURCE), map_location='cpu', strict=True,
                                                 expected_kind='category_boundary')
    c.require(meta['checkpoint_id'] == c.SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
              'Use the actual compressed-attention parent.')
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    names = [f'model.layers.{layer}.mlp.gate_proj' for layer in c.LAYERS]
    originals = {name: model.get_submodule(name) for name in names}
    c.require(all(type(layer) is torch.nn.Linear for layer in originals.values()), 'The three source gates must remain dense.')

    def score():
        predictions, probabilities, losses = [], [], []
        for cases, target in groups:
            logp = backend.mmlu.log_distribution(model, cases)
            loss, _ = backend.mmlu.weighted_loss(logp, target)
            losses.append(float(loss))
            probabilities.append(logp.cpu())
            for case, values in zip(cases, logp.cpu(), strict=True):
                predictions.append(dict(case_id=case['case_id'], gold_index=case['gold_index'],
                    predicted_index=int(values.argmax()), four_option_log_probs=values.tolist()))
        return torch.cat(probabilities), dict(mixed_objective=sum(losses)/len(losses),
            correct=sum(row['gold_index'] == row['predicted_index'] for row in predictions),
            total=len(predictions), samples=predictions)

    parent, baseline = score()
    result = dict(status='running', source_checkpoint_id=c.SOURCE_ID, data=evidence, baseline=baseline,
        warmup=warm, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), arms={},
        protocol='Same fixed64 auxiliary_train cases, native4-option M32; .5F+.5gold target once; mean of the two original weighted losses.',
        boundary='Three-gate conditional substitution, other projections include dense weights. Not a complete low-bit model, blind validation, MMLU test score or nine-task acceptance.')
    for arm, report in reports.items():
        try:
            for name, item in report['endpoints'].items():
                parent_module = model.get_submodule(name.rsplit('.', 1)[0])
                parent_module.gate_proj = restore_endpoint(item['path'])
            logp, measured = score()
            measured['parent_four_option_KL'] = float(torch.nn.functional.kl_div(logp.double(), parent.double(),
                log_target=True, reduction='batchmean'))
            measured['endpoint_sha256'] = {name: value['sha256'] for name, value in report['endpoints'].items()}
            result['arms'][arm] = measured
            print(dict(arm=arm, correct=measured['correct'], total=measured['total'],
                       mixed_objective=measured['mixed_objective'], parent_KL=measured['parent_four_option_KL']), flush=True)
        finally:
            for name, original in originals.items():
                model.get_submodule(name.rsplit('.', 1)[0]).gate_proj = original
    c.require(all(model.get_submodule(name) is layer for name, layer in originals.items()), 'Restore the exact original module objects.')
    result.update(status='completed_conditional_train_readout', source_modules_restored=True,
        seconds=time.monotonic()-started, implementation_sha256=c.sha(Path(__file__)),
        native_loader=str(Path(load_v6_model_checkpoint.__code__.co_filename).resolve()))
    c.write(destination, result)


if __name__ == '__main__':
    main()

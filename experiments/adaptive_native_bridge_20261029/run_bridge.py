"""Two real M32 updates of one compact RVQ layer inside the complete HF model."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from experiments.adaptive_native_bridge_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_bridge_20261029.inputs import load_groups
from experiments.adaptive_vector_rvq import common as c
from experiments.adaptive_sparse_pair_20261028 import sparse_file
from experiments.adaptive_vector_rvq.third_stage_export import inventory as dynamic_inventory
from rotation.model_utils import get_qwen3

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/star/VAELLM/adaptive_sparse_layers_20261028/layer_18')
FILES = {'fixed2_sparse': SOURCE / 'sparse_abs005.avrqsp',
         'dynamic3': Path('/data/star/VAELLM/adaptive_grouped_fit_20261029/grouped/endpoint_18/weighted_dynamic3.avrvqz')}
PREFIX = 'model.layers.18.mlp.down_proj'


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def frozen_model_digest(model):
    return c.digest({name: tensor for name, tensor in model.state_dict().items() if not name.startswith(PREFIX + '.')})


def load_hf():
    model = get_qwen3(str(c.HF), None).eval().requires_grad_(False).to('cuda:0')
    c.require(model.get_input_embeddings().weight.dtype == torch.bfloat16
              and model.config._attn_implementation == 'flash_attention_2', 'Keep original HF BF16/FA2 loading.')
    c.require(all(tensor.dtype == torch.float32 for name, tensor in model.named_buffers() if 'inv_freq' in name),
              'Do not cast original FP32 RoPE buffers.')
    return model


def persistent_inventory(module):
    if module.sparse is None:
        return dynamic_inventory(module.template)
    return sparse_file.inventory(module.template, module.sparse)


def run_arm(kind, output, backend, groups):
    started = time.monotonic()
    model = load_hf()
    # The frozen F cache was built by restoring all projections from this same HF.
    teacher = backend.teacher_recipe()
    names = list(backend.hot.student_projection_names(str(c.HF), 'all'))
    old = backend.load_state(teacher['model']['checkpoint'])
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal
    actual_backbone = {name: value.detach().cpu() for name, value in
                       backend.outside_projections(model.state_dict(), names).items()}
    count = _assert_state_equal(actual_backbone,
                               backend.outside_projections(old, names), 'Original HF/full-F backbone')
    c.require(count == 147 and Path(teacher['model']['hf_path']).resolve() == c.HF.resolve(),
              'The once-mixed targets must belong to the same original HF teacher.')
    del old, actual_backbone
    layer = CompactRVQLinear(FILES[kind], kind, 'cuda:0')
    c.require(layer.template['geometry']['name'] == PREFIX, 'Use the actual L18 artifact identity.')
    with torch.no_grad():
        initial_weight = layer.decode_weight().detach()
        reference = layer.reference_weight('cuda:0')
        c.require(torch.equal(initial_weight, reference), 'Trainable consumer changed the persisted BF16 decode.')
        del reference
    model.model.layers[18].mlp.down_proj = layer
    before = frozen_model_digest(model)
    frozen = layer.frozen_digest()
    initial_parameters = {name: value.detach().cpu().clone() for name, value in layer.named_parameters()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    rows = []
    for step, (cases, target) in enumerate(groups, 1):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = 1e-6 * (1 - (step - 1)/500)
        loss, per_document_loss = backend.mmlu.weighted_loss(backend.mmlu.log_distribution(model, cases), target)
        c.require(torch.isfinite(loss).item(), 'Nonfinite actual M32 bridge loss.')
        loss.backward()
        c.require(all(value.grad is not None and value.grad.dtype == torch.float32
                      and torch.isfinite(value.grad).all().item() for value in layer.parameters()),
                  'Every actual decoder leaf needs a finite FP32 gradient.')
        c.require(all(value.grad is None for name, value in model.named_parameters()
                      if not name.startswith(PREFIX + '.decoders.')), 'Frozen HF parameters received gradients.')
        norm = float(torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.5, error_if_nonfinite=True))
        c.require(norm > 0, 'The full-model objective did not reach the compact decoders.')
        optimizer.step()
        builds = layer.cache_builds
        with torch.no_grad():
            observed = backend.mmlu.log_distribution(model, groups[0][0]).cpu()
        c.require(layer.cache_builds == builds + 1, 'Changed masters must invalidate the inference cache.')
        row = dict(step=step, loss=float(loss.detach()), gradient_norm_before_clip=norm,
            lr=optimizer.param_groups[0]['lr'], cache_builds=layer.cache_builds,
            nonzero_gradient_elements=sum(int(torch.count_nonzero(p.grad)) for p in layer.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        rows.append(row)
        event('full_model_M_update', arm=kind, **row)
        del loss, per_document_loss
    c.require(layer.frozen_digest() == frozen and frozen_model_digest(model) == before,
              'Compact fixed fields or the other HF model tensors changed.')
    changes = {name: int(torch.count_nonzero(value.detach().cpu() - initial_parameters[name]))
               for name, value in layer.named_parameters()}
    c.require(any(changes.values()), 'The optimizer did not update any actual master parameter.')
    model.eval()
    c.require(layer._decoded_cache is None, 'Changing train/eval mode must release cached weights.')
    with torch.no_grad():
        deployed_logp = backend.mmlu.log_distribution(model, groups[0][0]).cpu()
        current_weight = layer.decode_weight()
        changed_weights = int(torch.count_nonzero(current_weight != initial_weight))
        current_digest = c.digest({'weight': current_weight})
    output.mkdir()
    filename = output / ('layer.avrqsp' if kind == 'fixed2_sparse' else 'layer.avrvqz')
    size = layer.save_compact(filename)
    c.require(size == FILES[kind].stat().st_size, 'Decoder-only updates must preserve the actual storage budget.')
    expected_decoder = [{name: value.detach().cpu().bfloat16().contiguous().clone() for name, value in decoder.state_dict().items()}
                        for decoder in layer.decoders]
    original_inventory = persistent_inventory(layer)
    fixed_inventory = {name: value for name, value in original_inventory.items() if '/decoder/' not in name}
    fixed_digest = c.digest(fixed_inventory)
    del optimizer, current_weight, initial_weight, layer, model
    gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    c.require(torch.cuda.memory_allocated() == 0, 'Release the first model before independent fresh loading.')
    fresh = load_hf()
    restored = CompactRVQLinear(filename, kind, 'cuda:0')
    restored.requires_grad_(False)
    fresh.model.layers[18].mlp.down_proj = restored
    fresh.eval()
    reloaded_inventory = persistent_inventory(restored)
    c.require(c.digest({name: value for name, value in reloaded_inventory.items() if '/decoder/' not in name}) == fixed_digest,
              'Saved compact file changed a frozen persistent field.')
    for stage, expected in zip(restored.template['stages'], expected_decoder, strict=True):
        _assert_state_equal(stage['decoder_state'], expected, 'Saved current BF16 decoder')
    with torch.no_grad():
        fresh_weight = restored.decode_weight()
        c.require(c.digest({'weight': fresh_weight}) == current_digest, 'Fresh compact deployment weight differs.')
        actual = backend.mmlu.log_distribution(fresh, groups[0][0]).cpu()
    c.require(torch.equal(actual, deployed_logp), 'Fresh full-model same-input output changed after compact reload.')
    result = dict(status='verified_bridge_only', arm=kind, source=c.stat(FILES[kind]), source_sha256=c.sha(FILES[kind]),
        layer_name=PREFIX, initial_same_file_BF16_weight_exact=True, full_HF_model_frozen_digest=before,
        compact_frozen_buffers_digest=frozen, teacher_nonprojection_tensors_compared=count,
        trainable_decoder_leaves=len(changes), master_changed_elements=changes,
        actual_deployment_weight_changed_elements=changed_weights, trajectory=rows,
        compact_file=dict(path=str(filename), bytes=size, sha256=c.sha(filename)),
        compact_linear_bpw=size*8/(4096*12288), fresh_compact_weight_exact=True, fresh_full_model_output_exact=True,
        boundary='Single real L18 bridge with two M32 updates; the remaining HF projections are dense. '
                 'Not a <=2.6bpw complete model, formal M/QA/R recovery, or downstream quality result.',
        elapsed_seconds=time.monotonic()-started)
    del fresh_weight, restored, fresh
    gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    c.require(torch.cuda.memory_allocated() == 0, 'Release fresh model at the arm boundary.')
    c.write(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
              and Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(),
              'Use caller-activated bitvae and the single assigned GPU.')
    c.require(not args.output.exists(), 'Never overwrite an existing bridge run.')
    torch.set_num_threads(4)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    backend, groups, evidence = load_groups()
    paths = [*HERE.glob('*.py'), *FILES.values()]
    pins = {str(path): c.sha(path) for path in paths}
    args.output.mkdir(parents=True)
    record = dict(status='running', command=[sys.executable, *sys.argv], pid=os.getpid(),
        started_unix=time.time(), gpu=args.gpu, source_sha256=pins, data=evidence, arms={})
    c.write(args.output / 'report.json', record)
    try:
        for kind in FILES:
            record['arms'][kind] = run_arm(kind, args.output / kind, backend, groups)
            c.write(args.output / 'report.json', record)
        c.require(all(c.sha(path) == digest for path, digest in pins.items()), 'Bound implementation/artifacts changed.')
        record.update(status='verified_bridge_only', exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time())
        c.write(args.output / 'report.json', record)


if __name__ == '__main__':
    main()

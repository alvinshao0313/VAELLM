"""Actual tiny-Qwen BF16 mechanism checks for the streaming proposal readout."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from .contracts import require
from .data import batch_digest, file_sha256
from .proposal_readout import score_layer_functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists()
        and not torch.cuda.is_initialized(), 'Use an isolated CPU caller and a fresh output.')
    began = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(913)
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=137, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attention_dropout=0.)
    config._attn_implementation = 'eager'
    parent = Qwen3ForCausalLM(config).to(torch.bfloat16).train()
    names = tuple(f'model.layers.{layer}.self_attn.q_proj' for layer in range(2))
    exact = {name: copy.deepcopy(parent.get_submodule(name)) for name in names}
    zero = {name: copy.deepcopy(parent.get_submodule(name)) for name in names}
    with torch.no_grad():
        for module in zero.values():
            module.weight.zero_()
    exact[names[0]].eval()
    all_modules = [module for model in (parent, *exact.values(), *zero.values()) for module in model.modules()]
    modes = [module.training for module in all_modules]
    rng = torch.get_rng_state()
    before = {key: value.clone() for key, value in parent.state_dict().items()}
    batches = []
    for index in range(128):
        length = 5 + index % 3
        ids = torch.arange(4 * length).reshape(4, length) % 130 + 1
        mask = torch.ones_like(ids)
        mask[1, -2:] = 0
        labels = ids.clone()
        labels[:, :2] = -100
        labels[mask == 0] = -100
        inputs = dict(input_ids=ids, labels=labels, attention_mask=mask)
        batches.append(dict(inputs=inputs, input_sha256=batch_digest(inputs),
            records=[dict(record_id=str(index * 4 + row)) for row in range(4)]))
    expected_tokens = [sum(int(batch['inputs']['attention_mask'].sum())
        for batch in batches[start:start + 64]) for start in (0, 64)]
    actual = score_layer_functions(parent, dict(exact=exact, zero=zero), batches,
        projection_names=names, parent_identity='actual_tiny_cpu_Qwen_BF16_fixture',
        data_identity='CPU_512_record_mechanism_not_formal_dataPlan')
    for name in names:
        require(actual['exact']['per_projection'][name] == [0., 0.]
            and actual['zero']['per_projection'][name] == [1., 1.],
            'Actual identical/zero deployment modules did not give exact0/1 NMSE.')
        require([half['valid_tokens'] for half in actual['zero']['sufficient_statistics'][name]] == expected_tokens,
            'Padding or the fixed256/256 boundary entered the functional objective incorrectly.')
        require(not parent.get_submodule(name)._forward_hooks, 'Readout leaked a parent hook.')
    require(modes == [module.training for module in all_modules]
        and torch.equal(rng, torch.get_rng_state()), 'Readout changed actual modes or training RNG.')
    require(all(torch.equal(value, parent.state_dict()[key]) for key, value in before.items()),
        'The readout changed parent persistent state.')
    try:
        score_layer_functions(parent, dict(incomplete={names[0]: exact[names[0]]}), batches,
            projection_names=names, parent_identity='tiny', data_identity='CPU')
    except ValueError:
        pass
    else:
        raise AssertionError('An incomplete actual category was accepted.')
    require(not torch.cuda.is_initialized(), 'CPU local-readout validation initialized CUDA.')
    result = dict(status='completed_cpu_streaming_actual_BF16_readout', exit_code=0,
        elapsed_seconds=time.monotonic() - began, cuda_initialized=False,
        actual_tiny_Qwen_layers=2, proposal_records=512, batch_size=4,
        nonpadding_tokens_per_half=expected_tokens,
        identical_deployment_NMSE_exact_zero=True, zero_deployment_NMSE_exact_one=True,
        independent_half_statistics=True, modes_rng_parent_state_exact=True,
        incomplete_category_rejected=True, hooks_removed=True,
        source_sha256={str(path): file_sha256(path) for path in
            (Path(__file__), Path(__file__).with_name('proposal_readout.py'))},
        boundary='Actual tiny-Qwen CPU mechanism, not full Qwen8B input/candidate/GPU or formal proposal evidence.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Narrow checks for raw-row namespaces and positional input forwarding."""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch

from . import data as d
from . import proposal_readout as p
from .contracts import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists() and not torch.cuda.is_initialized(),
        'Run the CPU mechanism check with CUDA_VISIBLE_DEVICES empty before model/accelerator imports.')
    torch.set_num_threads(4)
    began = time.monotonic()
    old = json.loads(Path(args.prior).read_text(encoding='utf-8'))
    identity = old['actual_source']['actual_source']
    from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, _load_preset_raw_datasets
    preset = DATASET_MIX_SOURCE_PRESETS['vaellm_eval_task']
    raw, _unused_eval = _load_preset_raw_datasets(preset)
    source = d.bind_source(raw, preset, revision=identity['revision'], license_record=identity['license'],
        provenance_record=identity['provenance'], artifact_digest=identity['artifact_sha256'])
    alias_identity = copy.deepcopy(source.identity)
    alias_identity.update(uri='relocated_description_only', license={'description': 'Different metadata'},
        provenance={'description': 'Calibration alias'}, fingerprint='description_only')
    alias = d.BoundSource(raw, replace(preset, alias='cpu_calibration_alias'), alias_identity)
    original_ref = source.row_reference(0, dict(raw[0]))
    alias_ref = alias.row_reference(0, dict(raw[0]))
    require(original_ref['record_id'] == alias_ref['record_id']
        and original_ref['source_id'] != alias_ref['source_id'], 'Same artifact/raw row changed ID under an alias or description.')
    require(source.row_reference(1, dict(raw[1]))['record_id'] != original_ref['record_id'], 'Different original rows collided.')

    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(749)
    config = Qwen3Config(vocab_size=137, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
        position_ids=torch.tensor([[0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10]]), cache_position=torch.arange(6))
    with torch.no_grad():
        actual = p._model_logits(model, inputs)
        expected = model(input_ids=ids, attention_mask=inputs['attention_mask'], position_ids=inputs['position_ids'],
            cache_position=inputs['cache_position'], use_cache=False).logits
    require(torch.equal(actual, expected), 'The actual Qwen CPU forward dropped positional inputs.')
    altered = {**inputs, 'position_ids': inputs['position_ids'] + 1}
    require(d.batch_digest(inputs) != d.batch_digest(altered), 'Position IDs did not enter the exact input identity.')
    altered_cache = {**inputs, 'cache_position': inputs['cache_position'] + 1}
    require(d.batch_digest(inputs) != d.batch_digest(altered_cache), 'Cache positions did not enter the exact input identity.')
    try:
        d.batch_digest({**inputs, 'unknown_position_input': torch.arange(6)})
    except ValueError:
        pass
    else:
        raise AssertionError('An unsupported positional field was silently discarded.')
    require(not torch.cuda.is_initialized(), 'The data-contract checks initialized CUDA.')
    here = Path(__file__).parent
    result = dict(status='completed_cpu_data_contracts', exit_code=0,
        elapsed_seconds=time.monotonic() - began, cuda_initialized=False,
        same_artifact_row_same_id_under_descriptions=True, distinct_raw_rows_distinct=True,
        actual_tiny_Qwen_positional_forward_exact=True, positional_identity_bound=True,
        unsupported_inputs_rejected=True, previous_data_receipt_sha256=d.file_sha256(args.prior),
        source_sha256={str(here / name): d.file_sha256(here / name)
            for name in ('data.py', 'proposal_readout.py', 'validate_data_cpu.py', 'validate_data_contract_cpu.py')},
        boundary='Namespace and positional CPU contracts only. Reuse prior token/loss/worker evidence; no full-model GPU proof or formal five-source dataPlan.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'exit_code', 'elapsed_seconds', 'cuda_initialized')}), flush=True)


if __name__ == '__main__':
    main()

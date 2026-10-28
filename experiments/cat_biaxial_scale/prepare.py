"""Bind original Qwen weights and train-only diagnostic inputs for the CAT pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from safetensors import safe_open
import torch
from transformers import AutoTokenizer

from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.distill_data import build_distill_data_collator, encode_canonical_record


LAYERS = (0, 18, 35)
CATEGORIES = ('gate_proj', 'up_proj', 'down_proj')
TARGETS = tuple(f'model.layers.{layer}.mlp.{category}' for layer in LAYERS for category in CATEGORIES)
ROW_START, DOCUMENTS, LENGTH = 700000, 128, 256


def prepare(model_path, protection_source, data_root, output_dir):
    model_path, protection_source, data_root, output = map(
        lambda path: Path(path).resolve(), (model_path, protection_source, data_root, output_dir))
    require(not output.exists(), 'Preparation output must be new.')
    metadata = json.loads((protection_source / 'checkpoint_meta.json').read_text())
    require(metadata['checkpoint_id'] == 'ab0280cf-3637-47e3-b771-074b74532a54'
            and metadata['after_category_mode'] == 'none'
            and Path(metadata['base_model_path']).resolve() == model_path,
            'Use the verified pure CAT checkpoint for protection indices only.')
    state_file = protection_source / metadata['state_dict_file']
    state = torch.load(state_file, map_location='cpu', weights_only=True, mmap=True)
    weight_map = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    targets = {}
    for name in TARGETS:
        with safe_open(str(model_path / weight_map[name + '.weight']), framework='pt', device='cpu') as stream:
            weight = stream.get_tensor(name + '.weight').clone()
        expected_shape = (4096, 12288) if name.endswith('down_proj') else (12288, 4096)
        count = 256 if name.endswith('down_proj') else 64
        indices = state[name + '.protected_input_indices'].clone()
        require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == expected_shape,
                f'Unexpected original weight: {name}.')
        require(indices.dtype == torch.int64 and tuple(indices.shape) == (count,)
                and torch.equal(indices, indices.unique(sorted=True))
                and 0 <= indices.min() and indices.max() < weight.shape[1],
                f'Invalid original protection indices: {name}.')
        targets[name] = dict(reference_weight=weight, protected_input_indices=indices)
    del state
    source = json.loads((data_root / 'manifest.json').read_text())
    data_path = Path(source['jsonl_path']).resolve()
    require(source['repo'] == 'HuggingFaceFW/fineweb-edu' and source['split'] == 'train'
            and source['revision'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
            and data_path.stat().st_size == source['jsonl_bytes'], 'FineWeb source differs.')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    samples, documents = [], []
    with data_path.open('rb') as stream:
        for row_index, line in enumerate(stream):
            if row_index < ROW_START:
                continue
            if row_index >= ROW_START + DOCUMENTS:
                break
            row = json.loads(line)
            sample = encode_canonical_record(row, tokenizer, text_format='text', text_field='text',
                                             task='lm', model_max_length=LENGTH)
            require(sample is not None and sample['input_ids'].numel() >= 2,
                    'A fixed diagnostic document is invalid; no filtering or replacement.')
            samples.append(sample)
            documents.append(dict(row_index=row_index,
                jsonl_line_sha256=hashlib.sha256(line).hexdigest(), valid_tokens=int(sample['attention_mask'].sum())))
    require(len(samples) == DOCUMENTS, 'Fixed diagnostic rows are missing.')
    batch = build_distill_data_collator(tokenizer, model_max_length=LENGTH, dynamic_padding=False)(samples)
    batch = {key: batch[key] for key in ('input_ids', 'attention_mask')}
    output.mkdir(parents=True)
    archive = dict(format='cat_biaxial_inputs_v1', model_path=str(model_path), targets=targets)
    torch.save(archive, output / 'inputs.pt')
    torch.save(batch, output / 'tokens.pt')
    report = dict(status='prepared', model_path=str(model_path), protection_source=str(protection_source),
        protection_checkpoint_id=metadata['checkpoint_id'], protection_state_file=file_stat(state_file),
        protection_use='Indices only; every candidate re-encodes the original BF16 columns to INT8.',
        original_weight_files=[file_stat(model_path / name) for name in sorted({weight_map[name + '.weight'] for name in TARGETS})],
        targets={name: {key: tensor_record(value) for key, value in bundle.items()} for name, bundle in targets.items()},
        data_source={key: source[key] for key in ('repo', 'revision', 'config', 'split', 'license', 'license_version',
            'original_shard', 'jsonl_sha256')}, data_file=file_stat(data_path), documents=documents,
        protocol=dict(row_start=ROW_START, documents=DOCUMENTS, max_length=LENGTH,
            use='Fixed readout only. Weight fitting and scale initialization use no text.',
            input_context='Original BF16 Qwen model; local linear and MLP diagnostics, not full-model evaluation.',
            selection='Contiguous physical train rows predetermined before candidate results; no filtering.'),
        valid_tokens=sum(row['valid_tokens'] for row in documents),
        archives={name: dict(**file_stat(output / name), sha256=sha256(output / name)) for name in ('inputs.pt', 'tokens.pt')},
        source_sha256={str(Path(__file__).resolve()): sha256(Path(__file__))}, finished_at_unix=time.time())
    write_json(output / 'manifest.json', report)
    print(json.dumps(dict(status=report['status'], targets=len(targets), valid_tokens=report['valid_tokens'])), flush=True)


@torch.no_grad()
def capture(prepared_dir):
    from rotation.model_utils import get_qwen3

    root = Path(prepared_dir).resolve()
    report = json.loads((root / 'manifest.json').read_text())
    require(report['status'] == 'prepared' and not (root / 'activations.pt').exists(), 'Capture must be fresh.')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Assign exactly one CUDA GPU.')
    batch = torch.load(root / 'tokens.pt', weights_only=True)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    teacher = get_qwen3(report['model_path'], None).eval().requires_grad_(False).to('cuda:0')
    teacher.config.use_cache = False
    require(all(parameter.dtype == torch.bfloat16 for parameter in teacher.parameters()), 'Require original BF16 parameters.')
    names = [f'model.layers.{layer}.mlp.{category}' for layer in LAYERS for category in ('gate_proj', 'down_proj')]
    captured = {name: [] for name in names}
    active = None

    def hook(name):
        def record_input(module, args):
            value = args[0].detach()
            require(value.dtype == torch.bfloat16, 'Captured input must be BF16.')
            captured[name].append(value[active].cpu().contiguous())
        return record_input

    handles = [teacher.get_submodule(name).register_forward_pre_hook(hook(name)) for name in names]
    started = time.time()
    try:
        for start in range(0, DOCUMENTS, 4):
            current = {key: value[start:start + 4].to('cuda:0') for key, value in batch.items()}
            active = current['attention_mask'].bool()
            teacher.model(**current, use_cache=False)
            if (start + 4) % 32 == 0:
                print(json.dumps(dict(captured_documents=start + 4)), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    captured = {name: torch.cat(values) for name, values in captured.items()}
    require(all(value.shape[0] == report['valid_tokens'] for value in captured.values()), 'Valid token count differs.')
    torch.save(dict(format='cat_biaxial_activations_v1', inputs=captured,
                    document_lengths=[row['valid_tokens'] for row in report['documents']]), root / 'activations.pt')
    report.update(status='captured', activations={name: tensor_record(value) for name, value in captured.items()},
        capture=dict(seconds=time.time() - started, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            backend=teacher.config._attn_implementation, tf32=False, batch_size=4,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gate_up_shared_input=True),
        activations_archive=dict(**file_stat(root / 'activations.pt'), sha256=sha256(root / 'activations.pt')))
    write_json(root / 'manifest.json', report)
    print(json.dumps(dict(status=report['status'], seconds=report['capture']['seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--model-path', required=True)
    prep.add_argument('--protection-source', required=True)
    prep.add_argument('--data-root', required=True)
    prep.add_argument('--output-dir', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--prepared-dir', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    (prepare if command == 'prepare' else capture)(**args)


if __name__ == '__main__':
    main()

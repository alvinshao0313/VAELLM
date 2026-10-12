"""CPU preparation only: identical native-encoded documents for LoRA and DoRA."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from e2e_common.data import _normalize_edgerazor_messages
from experiments.llama_order_control.prepare import normalized
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.llama_dora_probe.state import require, tensor_sha
from train_utils.cat_after_category_common import ResolvedCatAfterCategoryStage
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.distill_data import build_distill_dataset, build_distill_data_collator, encode_canonical_record
from train_utils.runtime_snapshot import to_runtime_jsonable

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'result/llama31_8b_w25_20261004/remaining_prefix_block32_s500'
SOURCE = CONTROL / 'home_star_models_meta-llama_Llama-3.1-8B-Instruct_20261004_104649/after_gate_proj'
SOURCE_ID = '89044995-d6b4-4bc3-a832-440f9144e423'


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def configuration(output):
    original = read(CONTROL / 'launch.json')
    cat, _, training, _ = parse_cat_runtime_args(original['argv'][2:])
    cfg = cat.resolve_after_category_config('gate_proj')
    require(cat.after_category_mode == 'remaining_lora_prefix_decoder'
            and cfg.lora.rank == 12 and cfg.lora.alpha == 24 and cfg.lora.dropout == .1
            and cfg.opt.steps == 500 and cfg.opt.batch_size == 4 and cfg.opt.gradient_accumulation_steps == 1
            and cfg.opt.learning_rate == 1e-4 and cfg.opt.decoder_lr == 1e-5
            and cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.alpha == .5
            and cfg.data.model_max_length == 1024 and cfg.data.seed == cfg.data.data_seed == 31
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
            and tuple(cfg.opt.distill_fp32_components) == ('lora', 'decoder')
            and training.bf16 and not training.fp16, 'Original gate recipe changed.')
    require(not cfg.runtime.distill_hif4_act and cfg.runtime.offload_mode == 'none'
            and cfg.runtime.parallel_mode == 'dp', 'Only the actual native no-HiF4 single-device path is covered.')
    cfg.opt = replace(cfg.opt, steps=64)
    return ResolvedCatAfterCategoryStage(mode=cat.after_category_mode, config=cfg,
        train_device='cuda:0', base_seed=31, stage_seed=31, output_dir=str(output),
        deterministic=cat.deterministic, fp16=False, bf16=True, reset_completed=False, save_strategy='no')


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['evidence_sha256'])
    for stat in record['input_stats']:
        check_stat(stat)
    require(sha256(record['samples_path']) == record['samples_sha256'], 'Fixed samples changed.')


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use fresh output and CUDA-hidden bitvae.')
    control = read(CONTROL / 'launch.json'); meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and meta['checkpoint_kind'] == 'category_boundary'
            and meta['lora_config'] is None and len(meta['compressed_targets']) == 160
            and len(meta['pending_dense_targets']) == 64 and not meta['skip_targets'], 'Wrong after-gate source.')
    require(all(name.endswith(('.up_proj', '.down_proj')) for name in meta['pending_dense_targets'])
            and all(spec['parallel_stage_decode'] for spec in meta['converted_modules']),
            'Require the real packed-prefix and remaining dense geometry.')
    check_hashes(control['source_sha256']); check_hashes(control['input_evidence_sha256'])
    require(normalized(control['argv'][2:]) == read(CONTROL / 'normalized_prepared_args.json'),
            'Original native parser no longer matches the completed control.')
    stage = configuration(output); cfg = stage.config
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bundle = build_distill_dataset(cfg.data, tokenizer)
    stream = bundle.train_dataset
    require(bundle.is_iterable and hasattr(stream, 'presets'), 'Require the existing canonical mixed SFT stream.')
    samples, documents, seen_docs, seen_prompts, seen_tokens = [], [], set(), set(), set()
    source_cursors = [0] * len(stream.presets)
    for ordinal, (source_idx, raw) in enumerate(stream.raw_dataset.iter_worker_with_source(worker_id=0, num_workers=1)):
        preset = stream.presets[source_idx]
        permutation = stream.raw_dataset.permutations[source_idx]
        source_row = int(permutation[source_cursors[source_idx] % len(permutation)])
        source_cursors[source_idx] += 1
        require(preset.text_format == 'edgerazor_messages', 'This probe only uses the actual native message mixture.')
        messages = _normalize_edgerazor_messages(dict(raw))
        if messages is None:
            continue
        sample = encode_canonical_record(dict(raw), tokenizer, text_format=preset.text_format,
            text_field=preset.text_field, task=cfg.data.dataset_task, model_max_length=1024)
        if sample is None:
            continue
        doc, prompt, token = json_sha(messages), json_sha([m for m in messages if m['role'] != 'assistant']), tensor_sha(sample['input_ids'])
        if doc in seen_docs or prompt in seen_prompts or token in seen_tokens:
            continue
        seen_docs.add(doc); seen_prompts.add(prompt); seen_tokens.add(token)
        samples.append(sample)
        documents.append({'source': preset.alias, 'source_row_index': source_row,
            'native_stream_ordinal': ordinal, 'document_sha256': doc,
            'prompt_group_sha256': prompt, 'input_ids_sha256': token, 'tokens': sample['input_ids'].numel(),
            'response_tokens': int((sample['labels'][1:] != -100).sum())})
        if len(samples) == 320:
            break
    require(len(samples) == 320, 'Insufficient distinct valid canonical documents.')
    collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
    batch_records = []
    for begin in range(0, 320, 4):
        batch = collator([dict(x) for x in samples[begin:begin + 4]])
        batch_records.append({'begin_document': begin, 'shape': list(batch['input_ids'].shape),
            'tensor_sha256': {name: tensor_sha(value) for name, value in batch.items()}})
    evidence = {**control['input_evidence_sha256'], **control['model_metadata_sha256']}
    for path in (CONTROL / 'launch.json', CONTROL / 'normalized_prepared_args.json', SOURCE / 'checkpoint_meta.json'):
        evidence[str(path)] = sha256(path)
    sources = dict(control['source_sha256'])
    sources.update({str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')})
    for name in ('experiments/llama_order_control/prepare.py', 'experiments/longbench_v1/common.py',
                 'tests/test_llama_dora_probe.py'):
        sources[str(ROOT / name)] = sha256(ROOT / name)
    for directory in ('train_utils', 'e2e_common', 'litebsq', 'rotation'):
        sources.update({str(p): sha256(p) for p in (ROOT / directory).rglob('*.py')})
    import peft
    require(peft.__version__ == '0.10.0', 'This isolated prototype targets the installed PEFT0.10.')
    for relative in ('tuners/lora/layer.py', 'tuners/lora/model.py', 'peft_model.py'):
        path = Path(peft.__file__).parent / relative; sources[str(path)] = sha256(path)
    record = {'schema': 'llama_after_gate_dora_probe_v1', 'status': 'prepared', 'gpu': None,
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID, 'base_model_path': meta['base_model_path'],
        'python_executable': sys.executable, 'output_dir': str(output), 'arms': ['lora', 'dora'],
        'configuration': to_runtime_jsonable(cfg), 'dense_names': sorted(meta['pending_dense_targets']),
        'decoder_names': sorted(meta['compressed_targets']), 'source_sha256': sources, 'evidence_sha256': evidence,
        'input_stats': [*control['input_data_stat'], *control['source_weight_stat'], file_stat(SOURCE / meta['state_dict_file'])],
        'documents': {'train': documents[:256], 'heldout': documents[256:]},
        'batch_records': {'train': batch_records[:64], 'heldout': batch_records[64:]},
        'data_source_counts': dict(Counter(d['source'] for d in documents)),
        'data_boundary': 'Canonical weighted source stream, seed31, worker0/1; first256 distinct valid documents train, next64 held. Document, prompt-group and token hashes are disjoint. No downstream/LongBench evaluation files read.',
        'schedule_boundary': 'Same64-step cosine, warmup_ratio.01, B4/GAS1, explicit fixed sequential order for both arms; not the original500 or historical3000-step schedule.',
        'source_boundary': 'After-gate already includes the original500-step recovery. This tests additional recovery, not reproduction of the historical recipe.',
        'authorization': 'User2026-10-03 permits isolated new-method short probes; production integration requires later approval.',
        'exact_resume_supported': False, 'goal_acceptance': False, 'cuda_initialized': False}
    check_hashes(sources); check_hashes(evidence)
    for stat in record['input_stats']:
        check_stat(stat)
    output.mkdir(parents=True)
    torch.save({'train': samples[:256], 'heldout': samples[256:]}, output / 'samples.pt')
    record.update(samples_path=str(output / 'samples.pt'), samples_sha256=sha256(output / 'samples.pt'))
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    write_json(output / 'preparation.json', record)
    print(json.dumps({'status': 'prepared', 'samples': 320, 'training_batches': 64,
        'heldout_batches': 16, 'samples_bytes': (output / 'samples.pt').stat().st_size,
        'preparation_sha256': sha256(output / 'preparation.json'), 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)

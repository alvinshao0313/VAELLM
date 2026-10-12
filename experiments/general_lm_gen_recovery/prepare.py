"""CPU binding for full-corpus, lazy native Gen LM recovery; no scheduling."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer
from e2e_common.data import DATASET_MIX_SOURCE_PRESETS
from experiments.general_lm_gen_recovery.train import (
    CONTROL, DATASET_MIX, R2, ROOT, SOURCE_ID, check_inputs, configuration, read, training_arguments,
)
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.distill_data import build_distill_data_collator, build_distill_dataset
from train_utils.runtime_snapshot import to_runtime_jsonable

ENCODING_PROOF = R2 / 'llama_gen64_cpu.json'
ENCODING_SHA = '3b8138533fa7bada59349fcc7724c0d3d2eb4ce4a09c22ab253f11a3bade3ae0'
DEFAULT_OUTPUT = R2 / 'general_lm_llama_gen_s2000'


def native_stat(item):
    return {'path': item['path'], 'size': item['bytes'], 'mtime_ns': item['mtime_ns'], 'inode': item['inode']}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def data_interface(cfg, hf, proof):
    """Read the existing full raw cache and encode one B4; never map/tokenize the corpus."""
    started = time.monotonic()
    tokenizer = _build_v6_tokenizer(cfg.teacher_model_path, hf)
    identity = proof['tokenizer']
    require(tokenizer.is_fast and tokenizer.padding_side == identity['padding_side'] == 'right'
            and tokenizer.eos_token_id == tokenizer.pad_token_id == identity['eos'] == identity['pad']
            and tokenizer.bos_token_id == identity['bos']
            and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == identity['chat_template_sha256'],
            'Native Llama tokenizer/chat identity differs from the real64-document proof.')
    bundle = build_distill_dataset(cfg.data, tokenizer)
    dataset = bundle.train_dataset; view = dataset.raw_dataset; raw = view.raw_dataset
    require(type(dataset).__name__ == '_CanonicalMapDataset' and not bundle.is_iterable and not bundle.group_by_length
            and len(dataset) == len(raw) == proof['raw_rows'] == 1456927
            and raw._fingerprint == proof['dataset_fingerprint']
            and dataset.preset.text_field == 'messages' and dataset.preset.text_format == 'edgerazor_messages',
            'Use the complete native single-alias map dataset, not a diagnostic subset or SFT iterable.')
    cache_stats = [file_stat(item['filename']) for item in raw.cache_files]
    require(cache_stats == [native_stat(item) for item in proof['dataset_cache_files']], 'Raw Arrow cache identity changed.')
    samples = [dataset[index] for index in range(4)]
    for index, sample in enumerate(samples):
        expected = proof['rows'][index]
        require(int(view.permutation[index]) == expected['physical_row']
                and tensor_sha(sample['input_ids']) == expected['tasks']['lm']['input_ids_sha256']
                and torch.equal(sample['labels'], sample['input_ids'])
                and int(sample['input_ids'][-1]) == tokenizer.eos_token_id,
                'Native Gen encoding differs from the already measured LM documents.')
    batch = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)(samples)
    expected_batch = proof['batches']['lm'][0]
    require(list(batch['input_ids'].shape) == expected_batch['shape']
            and tensor_sha(batch['input_ids']) == expected_batch['batch_ids_sha256']
            and tensor_sha(batch['labels']) == expected_batch['batch_labels_sha256']
            and torch.all(batch['labels'][batch['attention_mask'] == 0] == -100), 'Native B4 padding/label contract changed.')
    return {'dataset_class': type(dataset).__name__, 'is_iterable': bundle.is_iterable,
        'group_by_length': bundle.group_by_length, 'total_raw_rows': len(dataset),
        'source_stats': bundle.source_stats, 'raw_fingerprint': raw._fingerprint,
        'existing_arrow_file_count': len(cache_stats), 'existing_arrow_bytes': sum(item['size'] for item in cache_stats),
        'full_corpus_tokenization_required': False, 'full_corpus_token_cache_created': False,
        'diagnostic_only_encoded_documents': 4, 'diagnostic_B4_shape': list(batch['input_ids'].shape),
        'diagnostic_causal_tokens': int(batch['attention_mask'][:, 1:].sum()),
        'diagnostic_input_ids_sha256': tensor_sha(batch['input_ids']),
        'diagnostic_labels_sha256': tensor_sha(batch['labels']), 'elapsed_seconds': time.monotonic() - started,
        'boundary': 'This B4 is the native raw-permutation diagnostic prefix. Formal Trainer SeedableRandomSampler permutes the entire map dataset again; this is not its first training batch.'}


def build_record(output):
    """Construct a CPU record in memory; only prepare() writes an unbound launch."""
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized() and Path.cwd() == ROOT,
            'Use the public repository with CUDA hidden for CPU binding.')
    control = read(CONTROL); proof = read(ENCODING_PROOF)
    require(control['status'] == 'prepared_unbound' and control['source_checkpoint_id'] == SOURCE_ID
            and sha256(ENCODING_PROOF) == ENCODING_SHA and proof['status'] == 'verified_cpu_encoding'
            and proof['cuda_initialized'] is False and proof['control_sha256'] == sha256(CONTROL)
            and proof['source_checkpoint_id'] == SOURCE_ID, 'Bind the real immutable FineWeb recipe and Gen64 proof.')
    check_inputs(control); check_hashes(proof['source_sha256'])
    preset = DATASET_MIX_SOURCE_PRESETS['edgerazor_ii_gen']
    require(str((ROOT / preset.path).resolve()) == proof['dataset_file_stat']['path']
            and preset.text_field == 'messages' and preset.text_format == 'edgerazor_messages'
            and preset.supports_lm, 'Native alias semantics changed.')
    stats = [native_stat(proof['dataset_file_stat']), *[native_stat(item) for item in proof['dataset_cache_files']]]
    for item in stats:
        check_stat(item)
    record = deepcopy(control)
    record.update(schema='llama_gen_native_lm_fresh2000_v1', status='prepared_unbound', gpu=None, export_gpu=None,
        output_dir=str(output), python_executable=sys.executable, control_preparation=str(CONTROL),
        input_data_stat=stats, training_argv=training_arguments(control, output),
        argv=[sys.executable, '-u', '-m', 'experiments.general_lm_gen_recovery.train', '--launch_record', str(output / 'launch.json')],
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled'}, cuda_initialized=False,
        data={'dataset_mix': DATASET_MIX, 'dataset_task': 'lm', 'raw_source_file': preset.path,
            'raw_rows': proof['raw_rows'], 'raw_fingerprint': proof['dataset_fingerprint'],
            'native_text_field': preset.text_field, 'native_text_format': preset.text_format,
            'encoding_proof': str(ENCODING_PROOF), 'encoding_proof_sha256': ENCODING_SHA,
            'all_nonpadding_causal_labels': True, 'model_max_length': 1024, 'seed': 0, 'data_seed': 0,
            'group_by_length': False, 'training_subset_limit': None, 'language_filter': None,
            'sampler': 'Unchanged native map Trainer SeedableRandomSampler after the native alias raw permutation.',
            'comparison_boundary': 'Dataset distribution and native chat BOS/headers/content change together versus plain-text FineWeb; not a Chinese-only intervention. No LongAlign is loaded.'})
    record['initialization'].update(data_replay='Fresh Gen map sampler epoch0/seed0; no optimizer or data-cursor resume.',
        comparison_boundary='Same e767 history (192 FineWeb1600 projections +32 fresh D16down) and independent2000 recipe as the active FineWeb control; data and its native chat rendering differ.')
    record['input_evidence_sha256'].update({str(CONTROL): sha256(CONTROL), str(ENCODING_PROOF): ENCODING_SHA})
    record['source_sha256'].update(proof['source_sha256'])
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text()); record['source_sha256'][str(path.resolve())] = sha256(path)
    cfg, hf, args, names, architecture = configuration(record)
    interface = data_interface(cfg, hf, proof)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'all_projection_count': len(names),
        'model_type': architecture['model_type'], 'data_interface': interface}
    record['validation_reuse']['gen_native_lm_encoding'] = str(ENCODING_PROOF)
    record['estimate'].update(throughput_boundary='Gen lazy native-chat encoding and dynamic token lengths differ; inherited10–15h is an estimate pending actual throughput.',
        data_preparation='Existing4.92GB raw Arrow cache is reused; no full tokenization or new corpus copy. Runtime encodes only requested samples.')
    record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
        for name, argv, allowed in stage_commands(checkpoint=output / 'NATIVE_RUN/trainer_state/checkpoint-400',
            source_dir=record['source_checkpoint'], output_dir=output / 'step_0400')]
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU binding initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    return record


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a new independent output; never overwrite a preparation.')
    free = shutil.disk_usage(output.parent).free
    require(free >= 30_000_000_000, 'Reserve30GB for five independent full exports.')
    record = build_record(output); record['estimate']['disk_free_at_prepare'] = free
    output.mkdir(); write_json(output / 'launch.json', record)
    summary = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False,
        'launch_sha256': sha256(output / 'launch.json'), 'source_checkpoint_id': SOURCE_ID,
        'source_pin_count': len(record['source_sha256']), 'input_pin_count': len(record['input_evidence_sha256']),
        'data_stat_count': len(record['input_data_stat']), 'source_bpw': record['source_bit_budget']['user_linear_bpw'],
        'expected_inventory': record['expected_inventory'], 'data_interface': record['cpu_configuration']['data_interface'],
        'scope': 'Complete native Gen map LM, independent fresh2000 optimizer/schedule; no GPU binding, queue or short endpoint reuse.'}
    write_json(output / 'preparation.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    prepare(parser.parse_args().output)

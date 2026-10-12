"""Prepare complete fixed-grid INT4 protection sidecars without loading the model."""
from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path
import sys
import time

import torch

from experiments.long_context_recovery.contract import require
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.longbench_attribution.restore import tensor_digest
from experiments.protected_int4_probe.codec import decode_output_scale, encode_output_scale
from experiments.protected_int4_probe.measure import ROOT, SOURCES, errors, tensor_record
from experiments.protected_int4_quality.common import check as check_plain, summarize as summarize_plain
from experiments.protected_int4_scale_search.search import RATIOS, search_scales
from experiments.protected_int4_scale_search.measure import aggregate
from experiments.protected_int4_scale_quality.protection import ARM, SCHEMA, read, check, payload_from
from litebsq.protected_channel_quant import decode_protected_channel_weight

RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'


def prepare(label, output):
    output = Path(output).resolve(); started = time.time()
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not output.exists(), 'Use a fresh CPU-only output directory.')
    torch.set_num_threads(4)
    plain_dir = RESULT / 'protected_int4_quality' / label / 'output_scale_int4'
    base = read(plain_dir / 'launch.prepared.json'); check_plain(base)
    completed = read(plain_dir / 'report.json'); review = read(plain_dir / 'independent_review.json')
    require(completed['status'] == 'diagnostic_complete' and completed['exit_code'] == 0
            and review['status'] == 'verified' and review['source_checkpoint_id'] == base['source_checkpoint_id'],
            'Require the completed independently reviewed same-source plain control.')
    check_hashes(review['evidence_sha256'])
    plain_scores = summarize_plain(plain_dir, base)
    require(all(completed[k] == v for k, v in plain_scores.items()), 'Plain raw full-eight results changed.')
    proof_path = RESULT / 'protected_int4_scale_search_cpu_r1/report.json'; proof = read(proof_path)
    require(proof['status'] == 'cpu_scale_search_complete' and proof['protocol']['ratios'] == list(RATIOS)
            and proof['packed_serialized_reload_exact'] and proof['packed_bf16_decode_no_row_worse'],
            'Require the actual representative fixed-grid proof.')
    check_hashes(proof['source_sha256']); check_hashes(proof['input_sha256'])
    source, identity, _layers, state_count, count = SOURCES[label]
    meta = read(source / 'checkpoint_meta.json'); state_path = source / meta['state_dict_file']
    require(meta['checkpoint_id'] == identity == base['source_checkpoint_id']
            and len(meta['converted_modules']) == count and state_path.resolve().parent == source,
            'Source identity/inventory changed.')
    state_stat = file_stat(state_path)
    state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    baseline = torch.load(base['payload_path'], map_location='cpu', weights_only=True)
    require(len(state) == state_count and baseline['source_checkpoint_id'] == identity
            and baseline['arm'] == 'output_scale_int4' and set(baseline['projections']) == set(base['selected_targets']),
            'Actual source or original plain sidecar inventory differs.')
    payloads, manifest, rows = {}, {}, {}
    source_bits = candidate_bits = expanded_bits = 0
    for spec in meta['converted_modules']:
        name = spec['name']; old = base['payload_manifest'][name]
        qvalues, scales, indices = (state[name + '.' + k] for k in
                                   ('protected_input_qvalues', 'protected_input_scales', 'protected_input_indices'))
        require(spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
                and spec['protected_input_weight'] is None and indices.numel() in (64, 256)
                and indices.numel() % spec['codebook_dim'] == 0, 'Original protection geometry differs.')
        for tensor, key in ((qvalues, 'native_qvalues_sha256'), (scales, 'native_scales_sha256'), (indices, 'indices_sha256')):
            require(tensor_digest(tensor) == old[key], 'Actual source differs from primary plain control: ' + name + '/' + key)
        original_row = baseline['projections'][name]; original = payload_from(original_row)
        require(torch.equal(indices, original_row['indices']) and original.bits == 4 and original.signs is None,
                'Original plain payload/indices differ.')
        reference = decode_protected_channel_weight(qvalues, scales, quant_format='int8', dtype=torch.bfloat16,
                                                    device=torch.device('cpu')).t().contiguous()
        regenerated = encode_output_scale(reference, bits=4)
        require(all(torch.equal(original.tensors()[k], v) for k, v in regenerated.tensors().items()),
                'First candidate must be the exact original plain payload.')
        before = errors(decode_output_scale(original), reference)
        require(before == old['versus_native_int8_bf16'], 'Original plain BF16 error differs.')
        chosen, choices, old_rows, new_rows = search_scales(reference, original, row_chunk=512)
        decoded = decode_output_scale(chosen).t().contiguous(); after = errors(decoded.t(), reference)
        require(chosen.storage() == original.storage() == old['candidate_payload_storage']
                and chosen.storage()['padding_bits'] == 0 and after['sse'] <= before['sse']
                and bool((new_rows <= old_rows).all()), 'Storage or no-worse BF16 row contract failed.')
        payloads[name] = {'shape': chosen.shape, 'bits': 4, **chosen.tensors(), 'indices': indices.clone()}
        native_bits = sum(t.untyped_storage().nbytes() * 8 for t in (qvalues, scales))
        require(native_bits == old['source_payload_bits_without_indices'], 'Native source storage differs.')
        source_bits += native_bits; candidate_bits += chosen.storage()['total_bits']
        expanded_bits += decoded.untyped_storage().nbytes() * 8
        manifest[name] = {**old, 'decoded_input_by_output_sha256': tensor_digest(decoded),
            'versus_native_int8_bf16': after, 'candidate_payload_storage': chosen.storage(),
            'candidate_payload_tensors': {k: tensor_record(v) for k, v in payloads[name].items() if torch.is_tensor(v)}}
        row_bits = chosen.storage()['total_bits'] + indices.untyped_storage().nbytes() * 8
        rows[name] = {'category': name.rsplit('.', 1)[1], 'baseline': before, 'selected': after,
            'selection_counts': torch.bincount(choices, minlength=17).tolist(),
            'strictly_improved_rows': int((new_rows < old_rows).sum()), 'rows_with_worse_sse': int((new_rows > old_rows).sum()),
            'baseline_storage_bits': row_bits, 'selected_storage_bits': row_bits}
    require(len(payloads) == count and file_stat(state_path) == state_stat, 'Incomplete coverage or source changed.')
    del state, baseline
    output.mkdir(parents=True); payload_path = output / 'protected_payloads.pt'
    artifact = {'schema': 'protected_int4_payload_v1', 'source_checkpoint_id': identity, 'arm': ARM, 'projections': payloads}
    torch.save(artifact, payload_path); restored = torch.load(payload_path, map_location='cpu', weights_only=True)
    require(restored.keys() == artifact.keys() and restored['schema'] == artifact['schema']
            and restored['source_checkpoint_id'] == identity and restored['arm'] == ARM
            and restored['projections'].keys() == payloads.keys(), 'Persisted sidecar identity differs.')
    for name, row in restored['projections'].items():
        require(row.keys() == payloads[name].keys() and all(torch.equal(row[k], v) if torch.is_tensor(v) else row[k] == v
                for k, v in payloads[name].items()), 'Persisted sidecar state differs.')
        require(tensor_digest(decode_output_scale(payload_from(row)).t().contiguous())
                == manifest[name]['decoded_input_by_output_sha256'], 'Persisted BF16 replay differs.')
    account = copy.deepcopy(base['accounting']); total = account['source_user_linear_bits']
    require(source_bits == account['replaced_native_qvalues_scales_bits']
            and candidate_bits == account['candidate_packed_values_scales_signs_bits']
            and total - source_bits + candidate_bits == account['candidate_projected_linear_bits']
            and total - source_bits + expanded_bits == account['expanded_simulation_linear_bits'],
            'Actual full-source accounting differs from same-source plain control.')
    account['sidecar_container_bytes'] = payload_path.stat().st_size
    record = copy.deepcopy(base)
    record.update(schema=SCHEMA, arm=ARM, status='prepared_unbound', gpu=None, output_dir=str(output),
        payload_path=str(payload_path), payload_manifest=manifest, accounting=account,
        argv=[sys.executable, '-u', '-m', 'experiments.protected_int4_scale_quality.run', '--launch-record', str(output / 'launch.json')],
        scale_search={'ratios': list(RATIOS), 'row_chunk': 512, 'candidate_zero': 'exact_same_source_plain_payload',
            'objective': 'per_output_BF16_decode_weight_SSE', 'tie_rule': 'keep_earlier_candidate'},
        plain_control={'source_checkpoint_id': identity, 'arm': 'output_scale_int4', 'mean_percent': plain_scores['mean_percent'],
            'task_metrics': plain_scores['task_metrics'], 'accounting': base['accounting'],
            'raw_result_paths': {shard: str(plain_dir / ('lm_eval_results_' + shard + '.json')) for shard in ('mmlu', 'seven')}},
        cpu_search={'projections': rows, 'aggregate': aggregate(list(rows.values())),
            'categories': {category: aggregate([r for r in rows.values() if r['category'] == category])
                           for category in sorted({r['category'] for r in rows.values()})},
            'sidecar_saved_fresh_reload_exact': True, 'all_output_rows_no_worse': True})
    for directory in (Path(__file__).parent, ROOT / 'experiments/protected_int4_scale_search'):
        for path in directory.glob('*.py'):
            ast.parse(path.read_text()); record['source_sha256'][str(path.resolve())] = sha256(path)
    for path in (proof_path, payload_path, plain_dir / 'launch.prepared.json', plain_dir / 'report.json',
                 plain_dir / 'independent_review.json', *map(Path, record['plain_control']['raw_result_paths'].values())):
        record['input_sha256'][str(path)] = sha256(path)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['preparation_elapsed_seconds'] = time.time() - started
    write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'model': label, 'projections': count,
        'source_pins': len(record['source_sha256']), 'inputs': len(record['input_sha256']),
        'aggregate': record['cpu_search']['aggregate'], 'accounting': account,
        'elapsed_seconds': record['preparation_elapsed_seconds'], 'prepared_sha256': sha256(output / 'launch.prepared.json'),
        'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(SOURCES), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); prepare(args.model, args.output_dir)

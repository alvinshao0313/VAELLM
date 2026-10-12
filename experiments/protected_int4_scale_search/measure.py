"""CPU-only representative scale search; no activation or downstream objective."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import torch

from experiments.protected_int4_probe.codec import Payload, decode_output_scale, encode_output_scale, require
from experiments.protected_int4_probe.measure import ROOT, SOURCES, digest, errors, stat_record, tensor_record
from experiments.protected_int4_scale_search.search import RATIOS, search_scales
from experiments.longbench_attribution.restore import tensor_digest
from litebsq.protected_channel_quant import decode_protected_channel_weight

RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PROOF = RESULT / 'protected_int4_probe_cpu_r1'


def read(path):
    return json.loads(Path(path).read_text())


def aggregate(rows):
    energy = sum(x['baseline']['reference_energy'] for x in rows)
    before = sum(x['baseline']['sse'] for x in rows)
    after = sum(x['selected']['sse'] for x in rows)
    counts = [sum(x['selection_counts'][i] for x in rows) for i in range(len(RATIOS))]
    return {'matrices': len(rows), 'rows': sum(counts), 'reference_energy': energy,
            'baseline_sse': before, 'selected_sse': after, 'baseline_pooled_nmse': before / energy,
            'selected_pooled_nmse': after / energy, 'sse_reduction_percent': 100 * (1 - after / before),
            'selection_counts': counts, 'selection_fractions': [n / sum(counts) for n in counts],
            'strictly_improved_rows': sum(x['strictly_improved_rows'] for x in rows),
            'baseline_storage_bits': sum(x['baseline_storage_bits'] for x in rows),
            'selected_storage_bits': sum(x['selected_storage_bits'] for x in rows)}


def measure(output):
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Run from the repository root with CUDA hidden.')
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh output directory.')
    torch.set_num_threads(4)
    started = time.time()
    proof, review = read(PROOF / 'report.json'), read(PROOF / 'review.json')
    artifact_path = PROOF / 'selected_payloads.pt'
    require(proof['status'] == 'cpu_protected_codec_comparison_complete' and review['status'] == 'verified'
            and digest(PROOF / 'report.json') == review['report_sha256']
            and digest(artifact_path) == proof['payload_artifact']['sha256'] == review['selected_payloads_sha256'],
            'Original representative proof or payload changed.')
    source_paths = [*sorted(Path(__file__).parent.glob('*.py')),
                    ROOT / 'experiments/protected_int4_probe/codec.py',
                    ROOT / 'experiments/protected_int4_probe/measure.py',
                    ROOT / 'experiments/longbench_attribution/restore.py',
                    ROOT / 'litebsq/protected_channel_quant.py']
    source_hashes = {str(p.resolve()): digest(p) for p in source_paths}
    for path in ('experiments/protected_int4_probe/codec.py', 'litebsq/protected_channel_quant.py'):
        require(digest(ROOT / path) == proof['source_sha256'][str(ROOT / path)], 'Original codec changed.')
    inputs = {str(p): digest(p) for p in (PROOF / 'report.json', PROOF / 'review.json', artifact_path)}
    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
    reports, saved, source_stats = {}, {}, {}
    for label, (directory, identity, layers, state_count, projection_count) in SOURCES.items():
        prior = proof['models'][label]
        meta_path = directory / 'checkpoint_meta.json'
        meta = read(meta_path)
        require(prior['checkpoint_id'] == meta['checkpoint_id'] == identity
                and prior['checkpoint_dir'] == str(directory)
                and len(meta['converted_modules']) == projection_count
                and digest(meta_path) == proof['checkpoint_meta_sha256'][str(meta_path)],
                'Source identity or original metadata changed.')
        inputs[str(meta_path)] = digest(meta_path)
        quality_path = RESULT / 'protected_int4_quality' / label / 'output_scale_int4' / 'launch.prepared.json'
        quality = read(quality_path); inputs[str(quality_path)] = digest(quality_path)
        require(quality['source_checkpoint_id'] == identity and quality['source_checkpoint'] == str(directory),
                'Existing full-eight plain simulation has a different source.')
        state_path = directory / meta['state_dict_file']
        require(state_path.resolve().parent == directory, 'State path escaped the fixed source.')
        source_stats[label] = stat_record(state_path)
        state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
        require(len(state) == state_count and len(artifact[label]) == 21, 'Source/representative inventory differs.')
        specs = {x['name']: x for x in meta['converted_modules']}
        rows, saved[label] = {}, {}
        for name, entry in artifact[label].items():
            native = entry['native']; old = prior['projections'][name]
            actual = {}
            for key, suffix in (('values', 'qvalues'), ('scales', 'scales'), ('indices', 'indices')):
                value = state[name + '.protected_input_' + suffix]
                require(torch.equal(value, native[key]) and value.dtype == native[key].dtype
                        and tensor_record(value) == old['arms']['native_input_scale_int8']['storage']['components'][key],
                        'Selected original payload differs from actual source: ' + name + '/' + key)
                quality_key = {'values': 'native_qvalues_sha256', 'scales': 'native_scales_sha256',
                               'indices': 'indices_sha256'}[key]
                require(tensor_digest(value) == quality['payload_manifest'][name][quality_key],
                        'Actual source differs from evaluated plain simulation: ' + name + '/' + key)
                actual[key] = tensor_record(value)
            spec = specs[name]
            require(spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
                    and native['indices'].numel() % spec['codebook_dim'] == 0, 'Protection geometry changed.')
            reference = decode_protected_channel_weight(native['values'], native['scales'], quant_format='int8',
                dtype=torch.bfloat16, device=torch.device('cpu')).t().contiguous()
            row = entry['output_scale_int4']
            original = Payload(row['values'], row['scales'], tuple(row['shape']), row['bits'])
            regenerated = encode_output_scale(reference, bits=4)
            require(all(torch.equal(original.tensors()[k], v) for k, v in regenerated.tensors().items()),
                    'Original first candidate is not the exact historical plain payload.')
            baseline = errors(decode_output_scale(original), reference)
            require(baseline == old['arms']['output_scale_int4']['versus_native_bf16'], 'Original metrics changed.')
            chosen, choices, old_rows, new_rows = search_scales(reference, original, row_chunk=512)
            selected = errors(decode_output_scale(chosen), reference)
            require(chosen.storage() == original.storage() and chosen.storage()['padding_bits'] == 0
                    and selected['sse'] <= baseline['sse'] and bool((new_rows <= old_rows).all()),
                    'Packed replay, per-row SSE or exact same-storage contract failed.')
            indices = native['indices'].clone()
            require(torch.equal(indices, native['indices']), 'Protection indices changed.')
            bits = original.storage()['total_bits'] + actual['indices']['storage_bits']
            counts = torch.bincount(choices, minlength=len(RATIOS)).tolist()
            rows[name] = {'layer': old['layer'], 'category': old['category'], 'shape': list(reference.shape),
                'source_protection': actual, 'reference_bf16': tensor_record(reference),
                'baseline': baseline, 'selected': selected,
                'sse_reduction_percent': 100 * (1 - selected['sse'] / baseline['sse']),
                'selection_counts': counts, 'selection_fractions': [n / len(choices) for n in counts],
                'strictly_improved_rows': int((new_rows < old_rows).sum()),
                'rows_with_worse_sse': int((new_rows > old_rows).sum()),
                'baseline_storage_bits': bits, 'selected_storage_bits': bits, 'indices_exact': True,
                'baseline_payload': {k: tensor_record(v) for k, v in original.tensors().items()},
                'selected_payload': {k: tensor_record(v) for k, v in chosen.tensors().items()},
                'decoded_bf16_sha256': tensor_record(decode_output_scale(chosen))['sha256']}
            saved[label][name] = {'shape': chosen.shape, 'bits': 4, **chosen.tensors(), 'indices': indices}
            print(json.dumps({'model': label, 'target': name, 'sse_reduction_percent': rows[name]['sse_reduction_percent']}), flush=True)
        require(stat_record(state_path) == source_stats[label], 'Actual source changed while reading.')
        del state
        reports[label] = {'checkpoint_id': identity, 'checkpoint_dir': str(directory), 'layers': list(layers),
            'actual_source_matches_historical_payload_and_full8_plain_manifest': True, 'projections': rows,
            'aggregate': aggregate(list(rows.values())),
            'categories': {category: aggregate([x for x in rows.values() if x['category'] == category])
                           for category in sorted({x['category'] for x in rows.values()})}}
    output.mkdir(parents=True)
    payload_path = output / 'selected_payloads.pt'; torch.save(saved, payload_path)
    reloaded = torch.load(payload_path, map_location='cpu', weights_only=True)
    require(reloaded.keys() == saved.keys(), 'Reloaded model inventory differs.')
    for label, entries in saved.items():
        require(reloaded[label].keys() == entries.keys(), 'Reloaded projection inventory differs.')
        for name, original in entries.items():
            row = reloaded[label][name]
            require(row.keys() == original.keys() and all(torch.equal(row[k], v) if torch.is_tensor(v)
                    else row[k] == v for k, v in original.items()), 'Serialized payload bytes/schema changed.')
            replay = Payload(row['values'], row['scales'], tuple(row['shape']), row['bits'])
            require(tensor_record(decode_output_scale(replay))['sha256']
                    == reports[label]['projections'][name]['decoded_bf16_sha256'], 'Reloaded BF16 replay differs.')
    for path, expected in {**source_hashes, **inputs}.items():
        require(digest(path) == expected, 'Source/evidence changed during the experiment: ' + path)
    require(not torch.cuda.is_initialized(), 'CPU experiment initialized CUDA.')
    report = {'status': 'cpu_scale_search_complete', 'models': reports,
        'protocol': {'ratios': list(RATIOS), 'candidate_count': 17, 'candidate_zero': 'Exact historical plain packed payload',
            'selection': 'Per-output FP64 SSE of actual packed -> FP32 multiply -> BF16 decode; strict improvement only',
            'scale_rule': 'BF16(ratio * row_absmax / 7), zero row scale1; signed INT4[-8,7], no rotation or zero point',
            'reference': 'Actual source native INT8 deployment materialized as BF16, not original HF weights',
            'row_chunk': 512, 'cpu_threads': 4, 'activation_objective': False,
            'ratio_choices_stored_in_inference_payload': False, 'indices_and_storage_unchanged': True},
        'source_sha256': source_hashes, 'input_sha256': inputs, 'source_weight_stats': source_stats,
        'payload_artifact': {'path': str(payload_path), 'sha256': digest(payload_path), 'bytes': payload_path.stat().st_size},
        'packed_serialized_reload_exact': True, 'packed_bf16_decode_no_row_worse': True,
        'cuda_initialized': False, 'elapsed_seconds': time.time() - started,
        'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'downstream_quality': 'not_evaluated', 'model_exported': False, 'formal_integration': 'not_performed',
        'boundary': 'Only42 prespecified matrices. Lower weight SSE does not establish activation loss or full-eight accuracy improvement.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'],
        'report_sha256': digest(output / 'report.json'), 'payload_sha256': digest(payload_path),
        'aggregates': {k: v['aggregate'] for k, v in reports.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    measure(parser.parse_args().output_dir)

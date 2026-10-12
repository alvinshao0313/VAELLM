"""CPU encoding and native evaluation contract for one isolated INT4 protection arm."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import torch

from experiments.protected_int4_quality.common import ROOT, PROOF, SOURCES, ARMS, read, check
from experiments.protected_int4_quality.protection import payload_from
from experiments.protected_int4_probe.codec import encode_output_scale, decode_output_scale
from experiments.protected_int4_probe.measure import errors
from experiments.longbench_attribution.restore import tensor_digest
from experiments.longbench_v1.common import source_pins, file_stat, check_hashes, sha256, write_json
from experiments.longbench_v1.protocol import model_policy
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from litebsq.protected_channel_quant import decode_protected_channel_weight


def prepare(label, arm, output):
    output = Path(output).resolve(); started = time.time()
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not output.exists(), 'Use a fresh CPU-only output.')
    torch.set_num_threads(4)
    proof = read(PROOF); check_hashes(proof['source_sha256'])
    require(proof['status'] == 'cpu_protected_codec_comparison_complete'
            and proof['mechanism']['status'] == 'mechanism_verified', 'Require the actual codec/rotation proof.')
    source, identity, _layers, state_count, projection_count = SOURCES[label]
    meta = read(source / 'checkpoint_meta.json'); require(meta['checkpoint_id'] == identity, 'Source identity changed.')
    architecture = model_inventory(meta['base_model_path'])
    control_dir = source.parent
    summary_path = control_dir / ('acceptance.json' if label == 'qwen3' else 'full8_summary.json')
    control_paths = [control_dir / (shard + '.json') for shard in ('mmlu', 'seven')]
    control = collect_results(control_paths, architecture, checkpoint=(source, meta)); summary = read(summary_path)
    budget = summary['bit_budget']
    require(budget['checkpoint_id'] == identity and len(meta['converted_modules']) == projection_count,
            'Source measured budget or projection inventory differs.')
    state_path = source / meta['state_dict_file']; state_stat = file_stat(state_path)
    state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    require(len(state) == state_count, 'Source persistent state inventory differs.')
    payloads, manifest = {}, {}; source_payload_bits = candidate_payload_bits = expanded_bits = 0
    for spec in meta['converted_modules']:
        name = spec['name']; prefix = name + '.'
        require(spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
                and spec['protected_input_weight'] is None, 'Only actual native INT8 input protection is supported.')
        qvalues, scales, indices = (state[prefix + key] for key in
                                   ('protected_input_qvalues', 'protected_input_scales', 'protected_input_indices'))
        require(qvalues.dtype == torch.uint8 and scales.dtype == torch.bfloat16 and indices.dtype == torch.int64
                and qvalues.shape == (indices.numel(), spec['out_features']) and scales.shape == indices.shape
                and indices.numel() in (64, 256) and indices.numel() % spec['codebook_dim'] == 0,
                'Actual protected payload geometry changed.')
        reference = decode_protected_channel_weight(qvalues, scales, quant_format='int8', dtype=torch.bfloat16,
                                                    device=torch.device('cpu')).t().contiguous()
        payload = encode_output_scale(reference, bits=4, use_rotation=arm == ARMS[1], seed=31)
        decoded = decode_output_scale(payload, dtype=torch.bfloat16).t().contiguous()
        payloads[name] = {'shape': payload.shape, 'bits': 4, **payload.tensors(), 'indices': indices.clone()}
        storage = payload.storage(); require(storage['padding_bits'] == 0, 'No padding expected for64/256 protection.')
        native_bits = sum(t.untyped_storage().nbytes() * 8 for t in (qvalues, scales))
        source_payload_bits += native_bits; candidate_payload_bits += storage['total_bits']
        expanded_bits += decoded.untyped_storage().nbytes() * 8
        manifest[name] = {'shape_out_by_input': list(reference.shape), 'native_qvalues_sha256': tensor_digest(qvalues),
            'native_scales_sha256': tensor_digest(scales), 'indices_sha256': tensor_digest(indices),
            'decoded_input_by_output_sha256': tensor_digest(decoded), 'candidate_payload_storage': storage,
            'source_payload_bits_without_indices': native_bits, 'versus_native_int8_bf16': errors(decoded.t(), reference)}
    require(len(payloads) == projection_count and file_stat(state_path) == state_stat, 'Incomplete coverage/source changed.')
    del state
    output.mkdir(parents=True)
    artifact = {'schema': 'protected_int4_payload_v1', 'source_checkpoint_id': identity, 'arm': arm, 'projections': payloads}
    payload_path = output / 'protected_payloads.pt'; torch.save(artifact, payload_path)
    restored = torch.load(payload_path, map_location='cpu', weights_only=True)
    require(restored.keys() == artifact.keys() and set(restored['projections']) == set(payloads), 'Sidecar reload topology differs.')
    for name, row in restored['projections'].items():
        for key, value in payloads[name].items():
            require(torch.equal(row[key], value) if torch.is_tensor(value) else row[key] == value, 'Sidecar round-trip differs.')
        decoded = decode_output_scale(payload_from(row)).t().contiguous()
        require(tensor_digest(decoded) == manifest[name]['decoded_input_by_output_sha256'], 'Persisted payload replay differs.')
    del artifact, payloads, restored
    candidate_bits = budget['user_linear_bits'] - source_payload_bits + candidate_payload_bits
    simulation_bits = budget['user_linear_bits'] - source_payload_bits + expanded_bits
    accounting = {'source_user_linear_bits': budget['user_linear_bits'], 'original_linear_weights': budget['original_linear_weights'],
        'replaced_native_qvalues_scales_bits': source_payload_bits, 'candidate_packed_values_scales_signs_bits': candidate_payload_bits,
        'unchanged_indices_already_in_source_budget': True, 'candidate_projected_linear_bits': candidate_bits,
        'candidate_projected_linear_bpw': candidate_bits / budget['original_linear_weights'],
        'expanded_simulation_linear_bits': simulation_bits, 'expanded_simulation_linear_bpw': simulation_bits / budget['original_linear_weights'],
        'sidecar_container_bytes': payload_path.stat().st_size,
        'boundary': 'Candidate accounting uses actual packed sidecar tensor storage plus unchanged source components, counting indices once. The evaluated in-memory model contains expanded BF16 protection. No v6 INT4 checkpoint exists; container bytes are separate.'}
    source_hashes = source_pins()
    for directory in (Path(__file__).parent, ROOT / 'experiments/protected_int4_probe'):
        for path in directory.glob('*.py'):
            ast.parse(path.read_text()); source_hashes[str(path.resolve())] = sha256(path)
    for name in ('experiments/longbench_attribution/restore.py', 'experiments/llama_category_full8/common.py',
                 'experiments/long_context_recovery/contract.py',
                 'experiments/w25_eval/acceptance.py', 'experiments/qwen3_w25/summarize_results.py'):
        source_hashes[str(ROOT / name)] = sha256(ROOT / name)
    hf = Path(meta['base_model_path']); index = read(hf / 'model.safetensors.index.json')['weight_map']
    inputs = {str(p): sha256(p) for p in (PROOF, source / 'checkpoint_meta.json', source / 'config.json', summary_path,
        *control_paths, payload_path, hf / 'config.json', hf / 'tokenizer.json', hf / 'tokenizer_config.json',
        hf / 'generation_config.json', hf / 'model.safetensors.index.json')}
    record = {'schema': 'protected_int4_expanded_bf16_simulation_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'model_label': label, 'arm': arm, 'source_checkpoint': str(source),
        'source_checkpoint_id': identity, 'hf_path': str(hf), 'architecture': architecture, 'model_policy': model_policy(hf),
        'expected_serialized_tensors': state_count, 'selected_targets': sorted(manifest), 'payload_path': str(payload_path),
        'payload_manifest': manifest, 'accounting': accounting, 'source_sha256': source_hashes, 'input_sha256': inputs,
        'protected_stats': [state_stat, *[file_stat(hf / name) for name in sorted(set(index.values()))]],
        'control_result_paths': dict(zip(('mmlu', 'seven'), map(str, control_paths), strict=True)),
        'control_mean_percent': control['mean_percent'], 'control_task_metrics': control['task_metrics'],
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'lm_eval', 'datasets', 'flash-attn')},
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4',
            'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT)},
        'timeout_seconds': 900, 'termination_grace_seconds': 20,
        'protocol': {'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64, 'seven_batch_size': 32,
            'leaf_count': 64, 'sample_count': 24742, 'prewarm_group_size': 8, 'reference': 'Existing INT8-deployed BF16 protected values, not original HF.',
            'forward': 'Native VAELinear none-protection materialization and native bare-likelihood evaluator; no forward replacement or output correction.',
            'integration_validation': 'Check all native cached protection columns against persisted sidecar replay; then one finite native4-token model forward in the same arm before complete8.',
            'model_exported': False, 'goal_acceptance': False, 'longbench_status': 'not_evaluated'}, 'cuda_initialized': False}
    record['argv'] = [sys.executable, '-u', '-m', 'experiments.protected_int4_quality.run', '--launch-record', str(output / 'launch.json')]
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'model': label, 'arm': arm, 'source_pins': len(source_hashes),
        'projections': len(manifest), 'accounting': accounting, 'elapsed_seconds': time.time() - started,
        'prepared_sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(SOURCES), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); prepare(args.model, args.arm, args.output_dir)

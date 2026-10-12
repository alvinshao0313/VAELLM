"""Existing Qwen L18 data and native geometry for one bounded paired diagnostic."""
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer

from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
from experiments.gate_tail_objective_probe.data import _fineweb_batch, DOMAINS
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.rotation_factor_probe.capture import require, tensor_record
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import recipe
from litebsq.vae_args import apply_autoencoder_arch_defaults
from train_utils.cat_data_prep import LinearPrepRef, prepare_group_linear_entries, materialize_prepared_group_data

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = PUBLIC
OUTPUT = R2 / 'qwen_l18_rotated_output_pair'
SOURCE = PUBLIC / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
TARGET = 'model.layers.18.mlp.gate_proj'
ARMS = ('weight_mse', 'linear_output_mse')
STEPS, SEED, CHUNK = 256, 31, 65536


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def assets():
    capture = R2 / 'gate_student_context_probe/run/student_capture'
    manifest = read(capture / 'data_manifest.json')
    require(manifest['status'] == 'captured' and manifest['checkpoint_id'] == SOURCE_ID
            and sha256(capture / 'inputs.pt') == manifest['archive']['sha256'], 'Student capture differs.')
    cached = torch.load(capture / 'inputs.pt', map_location='cpu', weights_only=True, mmap=True)
    require(cached['checkpoint_id'] == SOURCE_ID and cached['target'] == TARGET
            and cached['document_lengths'] == manifest['document_lengths'], 'Capture identity differs.')
    for split, x in cached['inputs'].items():
        require(tensor_record(x) == manifest['tensors']['inputs'][split], 'Captured student input differs.')
    archive, hf_manifest = load_archive(R2 / 'rotation_factor_probe/middle_data/qwen')
    meta = read(SOURCE / 'checkpoint_meta.json')
    spec = next(s for s in meta['converted_modules'] if s['name'] == TARGET)
    require(meta['checkpoint_id'] == archive['checkpoint_id'] == SOURCE_ID and spec == manifest['target_spec']
            and not spec['transpose'] and spec['residual_stages'] == 2 and spec['codebook_dim'] == 32
            and spec['compressed_in_features'] == 4032 and spec['weight_rotation']['block_size'] == 32,
            'Require the existing full L18 gate S2/input64/block32 geometry.')
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    prefix, replay, alpaca = build_calibration_batches(tokenizer, seed=137, train_documents=64,
                                                  heldout_documents=32, max_length=256)
    native = {'train': {k: v[:32] for k, v in prefix.items()},
              'heldout': {k: v[32:] for k, v in prefix.items()}, 'replay32': replay}
    require({s: {k: tensor_record(v) for k, v in b.items()} for s, b in native.items()}
            == manifest['token_batches'], 'Historical native Alpaca tokens differ.')
    tail = R2 / 'gate_tail_objective_probe_r1'
    tail_record, tail_report = read(tail / 'launch.prepared.json'), read(tail / 'run/report.json')
    fineweb_path = Path(tail_record['data']['input_data_stat'][0]['path'])
    fineweb, _, _ = _fineweb_batch(tokenizer, fineweb_path)
    batches = {'train': native['train'], DOMAINS[0]: native['heldout'], DOMAINS[1]: fineweb}
    require({s: {k: tensor_record(v) for k, v in batches[s].items()} for s in DOMAINS}
            == tail_record['data']['token_batches'], 'Development documents differ.')
    teachers = {}
    teacher_sources = [(R2 / 'gate_same_kl_decoder_probe_s2', 'teacher_train'), (tail, 'teacher')]
    inputs = [capture / n for n in ('data_manifest.json', 'inputs.pt')]
    inputs += [R2 / 'rotation_factor_probe/middle_data/qwen' / n for n in ('data_manifest.json', 'inputs.pt')]
    inputs += [SOURCE / n for n in ('checkpoint_meta.json', 'config.json')]
    for folder, field in teacher_sources:
        report = read(folder / 'run/report.json')
        path = folder / 'run/teacher_top100.pt'
        require(report['status'] == 'completed' and sha256(path) == report['teacher_cache_sha256'], 'Teacher cache differs.')
        cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(cache, batches, report[field])
        require(not set(cache).intersection(teachers), 'Duplicate teacher domain.')
        teachers.update(cache)
        inputs += [path, folder / 'run/report.json', folder / 'review.json']
    inputs += [tail / 'launch.prepared.json', Path(meta['base_model_path']) / 'tokenizer.json',
               Path(meta['base_model_path']) / 'tokenizer_config.json']
    weight = archive['targets'][TARGET]['reference_weight']
    indices = archive['targets'][TARGET]['protected_input_indices']
    require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == (12288, 4096)
            and tuple(indices.shape) == (64,) and sum(manifest['document_lengths']['train']) == 4289,
            'Actual matrix or token geometry differs.')
    stats = [file_stat(SOURCE / meta['state_dict_file']), file_stat(fineweb_path), file_stat(alpaca['source_arrow']), *manifest['hf_weight_files']]
    return dict(weight=weight, indices=indices, spec=spec, meta=meta, capture=cached, manifest=manifest,
                batches=batches, teachers=teachers, inputs=inputs, stats=stats)


def native_inputs(data, *, steps=STEPS):
    weight = data['weight']
    entries = prepare_group_linear_entries(group_refs=[LinearPrepRef(TARGET, weight, 4096, 12288, False)],
        activation_weight_by_linear=None, channel_protect_count=64, channel_axis='input', channel_quant='int8',
        recon_loss_type='mse', channel_plan={TARGET: data['indices']}, apply_outlier_channel_removal=True)
    material = materialize_prepared_group_data(prepared_entries=entries, intra_parallel=(1, 1), codebook_dim=32,
        batch_size=CHUNK, normalize_weight=False, recon_loss_type='mse', train_device='cpu')
    require(tuple(material.stacked_data.shape) == (1548288, 1, 32), 'Native vector mapping differs.')
    argv, runtime, vae_args, _ = recipe(data['meta']['base_model_path'], 'gate_proj', steps=steps, seed=SEED)
    vae_args.parallel_layers = 1
    for name in ('residual_stages', 'codebook_bits', 'codebook_dim', 'base_ch', 'num_res_blocks',
                 'decoder_base_ch', 'decoder_num_res_blocks', 'norm_type', 'activation_type', 'decoder_type', 'recon_loss_type'):
        setattr(vae_args, name, getattr(runtime, name))
    apply_autoencoder_arch_defaults(vae_args)
    require(vae_args.vae_weight_dtype == 'bf16' and vae_args.vae_autocast_dtype == 'bf16'
            and runtime.residual_stages == 2, 'Retain actual BF16 native VAE configuration.')
    return material, vae_args, argv

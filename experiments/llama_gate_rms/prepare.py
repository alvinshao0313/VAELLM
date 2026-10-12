"""Prepare RMS CAT validation and a non-ready formal recipe; never bind a GPU."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

from experiments.llama_gate_rms.common import ROOT, ORIGINAL, RESULT, BASE, BASE_ID, CONTROL, CONTROL_ID, OUTPUT, check, read, _require, recipe, rms_spec, verify_short
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import file_stat, sha256, write_json
from litebsq.autoencoder import MultiLayerVAE, pack_decoders
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.vae_args import apply_autoencoder_arch_defaults
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import _build_decoder_from_spec, _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta


def construction(meta, control_audit, vae, cfg):
    gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    reference = gates[0]; new = rms_spec(reference)
    _require(len(gates) == 32 and all({**s, 'name': reference['name'],
        'weight_rotation': {**s['weight_rotation'], 'seed': reference['weight_rotation']['seed']}} == reference for s in gates)
        and reference['transpose'] and reference['residual_stages'] == cfg.residual_stages
        and all(d['hidden_dim'] == 128 and d['num_res_blocks'] == 0 and d['activation_type'] == 'swish'
                and d['decoder_type'] == 'symmetric' for d in reference['stage_decoders']), 'Wrong matched gate geometry.')
    with torch.random.fork_rng(devices=[]):
        old = pack_decoders([_build_decoder_from_spec(s) for s in reference['stage_decoders']])
        decoder = pack_decoders([_build_decoder_from_spec(s) for s in new['stage_decoders']])
        args = SimpleNamespace(**{**vars(vae), **vars(cfg), 'parallel_layers': len(gates)})
        apply_autoencoder_arch_defaults(args); model = MultiLayerVAE(args)
        _require(model.model.arch_spec.norm_type == 'rms' and model.model.encoder.norm_out.norm_type == 'rms'
            and model.model.decoder.norm_out.norm_type == 'rms' and model.model.decoder.num_models == len(gates)
            and packed_symmetric_decoder_supports_fuse(old) and not packed_symmetric_decoder_supports_fuse(decoder),
            'Actual encoder/decoder norm or execution eligibility differs.')
        bits = lambda module: sum(v.numel() * v.element_size() * 8 for v in module.state_dict().values())
        delta_bits = (bits(decoder) - bits(old)) * len(gates)
        holder = torch.nn.Module(); holder.add_module('gate_proj', torch.nn.Linear(4096, 14336, bias=False, dtype=torch.bfloat16))
        expected = deepcopy(new); expected['name'] = 'gate_proj'; _rebuild_converted_modules(holder, [expected])
        state = holder.state_dict()
        physical = sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}.values()) * 8
        _require(_collect_vae_linear_specs(holder)[0] == expected
            and physical == control_audit['bit_budget']['per_target'][reference['name']]['storage_bits'] + delta_bits // len(gates),
            'Native v6 RMS geometry/storage differs.')
        shapes = {k[len('gate_proj.'):]: [list(v.shape), str(v.dtype)] for k, v in state.items()
                  if k.startswith('gate_proj._parallel_stage_decoder.')}
        result = {'arch_spec': asdict(model.model.arch_spec), 'decoder_tensor_shapes': shapes,
            'old_decoder_leaves_per_projection': len(old.state_dict()), 'decoder_leaves_per_projection': len(decoder.state_dict()),
            'old_decoder_bits_per_projection': bits(old), 'decoder_bits_per_projection': bits(decoder),
            'decoder_delta_bits': delta_bits, 'physical_bits_per_gate': physical, 'serialized_tensors_per_gate': len(state),
            'grouped_encoder_parameters': sum(p.numel() for p in model.model.encoder.parameters()),
            'grouped_decoder_parameters': sum(p.numel() for p in model.model.decoder.parameters()),
            'scope': 'Actual CPU grouped VAE and full-size v6 module geometry; no learned weights or CUDA proof.'}
    inv = {'selected_projections': len(gates), 'stages_per_projection': reference['residual_stages'],
        'selected_stages': sum(len(s['stage_decoders']) for s in gates),
        'code_banks': sum(len(list(_code_specs(s))) for s in meta['converted_modules']),
        'serialized_tensors': control_audit['source_union_save_fresh_tensors'] + len(gates) * (len(decoder.state_dict()) - len(old.state_dict()))}
    budget = deepcopy(control_audit['bit_budget'])
    for key in ('user_linear_bits', 'packed_linear_bits'): budget[key] += delta_bits
    for key in ('budget_bits', 'storage_bits'): budget['components']['decoder'][key] += delta_bits
    budget['components']['decoder']['unique_storages'] += len(gates) * (len(decoder.state_dict()) - len(old.state_dict()))
    _require(inv['serialized_tensors'] == 2947 and inv['code_banks'] == 448 and delta_bits == -131072,
             'Unexpected metadata-derived RMS inventory/budget.')
    return result, inv, budget


def prepare(complete):
    _require(ROOT != ORIGINAL and Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only S preparation required.')
    formal = OUTPUT / 'rms'; short = OUTPUT / 'validate_s2'
    if complete:
        pending = formal / 'launch.pending.json'; record = read(pending); check(record)
        _require(record['status'] == 'awaiting_short_proof' and not (formal / 'launch.prepared.json').exists(), 'Fresh proof completion required.')
        proof = verify_short(record, short)
        for name in ('launch.prepared.json', 'launch.json', 'verification.json', 'audit.json', 'train.exit', 'assembly.exit', 'audit.exit', 'dispatcher.exit', 'outer.exit'):
            record['input_sha256'][str(short / name)] = sha256(short / name)
        record['input_sha256'][str(pending)] = sha256(pending)
        record.update(status='prepared_unbound', validation_proof={'path': str(short / 'verification.json'),
            'sha256': sha256(short / 'verification.json'), 'checkpoint_id': proof['checkpoint_id'], 'short_weights_consumed': False})
        check(record); write_json(formal / 'launch.prepared.json', record)
        print({'prepared': str(formal / 'launch.prepared.json'), 'sha256': sha256(formal / 'launch.prepared.json')}, flush=True)
        return
    _require(not OUTPUT.exists(), 'Do not overwrite a previous preparation.')
    template, ca = read(CONTROL / 'launch.json'), read(CONTROL / 'audit.json')
    review = read(CONTROL / 'independent_review.json'); meta = load_v6_meta(str(CONTROL / 'final_model'))
    _require(template['status'] == 'completed' and template['exit_code'] == 0
        and all((CONTROL / (s + '.exit')).read_text().strip() == '0' for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
        and ca['status'] == 'verified' and review['status'] == 'independent_cpu_verified'
        and ca['checkpoint_id'] == meta['checkpoint_id'] == template['checkpoint_id'] == CONTROL_ID
        and ca['base_checkpoint_id'] == BASE_ID and load_v6_meta(str(BASE))['checkpoint_id'] == BASE_ID,
        'The exact completed98a8/e767 control chain is required.')
    current, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    _require(current == review['actual_full8'], 'Control raw result differs from its independent review.')
    formal_argv, formal_norm, vae, cfg = recipe(template, formal, 10000)
    native, inv, budget = construction(meta, ca, vae, cfg)
    sources, pending, seen = {}, [*Path(__file__).parent.glob('*.py'), ROOT / 'tools/cat_train.py',
        ROOT / 'tools/cat_eval.py', ROOT / 'experiments/w25_eval/assemble_categories.py'], set()
    while pending:
        path = pending.pop().resolve()
        if path in seen: continue
        seen.add(path); ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in modules:
                local = ROOT / (name.replace('.', '/') + '.py')
                if local.is_file(): pending.append(local)
    common_sources = set(sources) & set(template['source_sha256'])
    _require(all(sources[p] == template['source_sha256'][p] for p in common_sources), 'Shared executed control source changed.')
    for path in Path(__file__).parent.glob('*.py'):
        _require(sha256(path) == sha256(ORIGINAL / path.relative_to(ROOT)), 'Public/S new source mismatch.')
    evidence = {p: v for p, v in template['input_sha256'].items() if '/result/' not in p}
    files = [*[CONTROL / f for f in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'independent_review.json')],
        CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json',
        BASE / 'checkpoint_meta.json', BASE / 'config.json', BASE / 'assembly_report.json', BASE.parent / 'review.json',
        *map(Path, template['control_result_paths']), RESULT / 'llama_gate_small_rate_pair/validate_k34_s2/verification.json',
        RESULT / 'llama_gate_encoder_width/h256/audit.json', RESULT / 'llama_gate_encoder_width/h256/independent_review.json']
    evidence.update({str(p): sha256(p) for p in files})
    stats = [s for s in template['protected_stats'] if '/result/' not in s['path']]
    stats.append(file_stat(BASE / load_v6_meta(str(BASE))['state_dict_file']))
    common = {key: deepcopy(template[key]) for key in ('architecture', 'environment', 'protocol', 'control_result_paths')}
    common.update(schema='llama_gate_rms_v1', gpu=None, queued=False, automatic_retry=False, cwd=str(ROOT),
        python_executable=sys.executable, base_dir=str(BASE), base_checkpoint_id=BASE_ID,
        historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        source_sha256=sources, input_sha256=evidence, protected_stats=stats,
        native_cpu_construction=native, expected_inventory=inv, expected_budget_components=budget['components'],
        expected_user_linear_bits=budget['user_linear_bits'], expected_packed_linear_bits=budget['packed_linear_bits'],
        expected_padding_bits=budget['code_padding_bits'], original_linear_weights=budget['original_linear_weights'],
        control_source_intersection_verified=len(common_sources),
        comparison_boundary='Primary fresh RMS gate minus98a8 LayerNorm with exact e767 other192/backbone. Norm changes both encoder and decoder, centering/bias/eps and execution path; not decoder-only. Original native LR is linear. Short2 changes its horizon and proves functionality/resources only; formal starts fresh and never consumes short weights.',
        resource_estimate={'disk_provision_bytes': 34000000000, 'short_seconds': [250, 600], 'formal_seconds': [1300, 2500],
            'training_peak_gib_estimate': [45, 65], 'filesystem_free_at_prepare': shutil.disk_usage(RESULT).free,
            'layer_control_seconds': template['finished_unix'] - template['started_unix'],
            'boundary': 'RMS GPU packed execution unmeasured. Norm-independent K34/rotation proofs reused; no repeat pack replay. Formal and short payloads need not coexist.'},
        experience={'prior': 'All50 retained actual Llama CAT normalized recipes use LayerNorm+Swish; encoder/decoder width negatives do not test RMS.',
            'source': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md')},
        cuda_initialized=False)
    records = []
    for directory, steps, ready in ((short, 2, True), (formal, 10000, False)):
        argv, normalized, _, _ = recipe(template, directory, steps)
        record = deepcopy(common); record.update(output_dir=str(directory), steps_per_stage=steps,
            status='prepared_unbound' if ready else 'awaiting_short_proof', full_evaluation=not ready,
            training_argv=argv, normalized_expected=normalized, timeout_seconds=1200 if ready else 4500, termination_grace_seconds=20,
            launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_rms.run', '--launch-record', str(directory / 'launch.json')])
        check(record); records.append((directory, 'launch.prepared.json' if ready else 'launch.pending.json', record))
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    for directory, filename, record in records:
        directory.mkdir(parents=True); write_json(directory / filename, record)
        print({'path': str(directory / filename), 'sha256': sha256(directory / filename), 'source': len(sources),
            'input': len(evidence), 'stat': len(stats), 'inventory': inv,
            'logical_bpw': budget['user_linear_bits'] / budget['original_linear_weights'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--complete-proof', action='store_true')
    prepare(parser.parse_args().complete_proof)

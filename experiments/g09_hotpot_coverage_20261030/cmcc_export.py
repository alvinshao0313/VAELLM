"""Continue the completed H192 endpoint on cmcc; never repeat its training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cmcc_runtime_20261030.paths import resolve_legacy_path

S = ROOT / 'result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01'
OLD = Path('/data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_20261030')
OUT = Path('/data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030')
PREPARED = OUT / 'export_prepared.json'
HF = Path('/home/cmcc/models/Qwen/Qwen3-8B').resolve()
SOURCE_ID = 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346'
AUDIT = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/G/09_joint1200/native_audit.json'
FIELDS = ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights',
          'components', 'outside_linear_bits', 'code_padding_bits')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def native():
    # Import the complete native package before project helpers prepend ROOT.
    # Package __path__ then remains pinned even when a later helper edits sys.path.
    sys.path[:0] = [str(S), str(ROOT)]
    names = ('rotation.hadamard_utils', 'rotation.hadamard_extended', 'rotation.weight_preprocess',
             'rotation.model_utils', 'train_utils.checkpoint_v6', 'train_utils.checkpoint_bpw',
             'train_utils.v6_model_loader', 'litebsq.vae_linear', 'litebsq.vae_linear_prewarm')
    modules = {name: importlib.import_module(name) for name in names}
    paths = {name: Path(module.__file__).resolve() for name, module in modules.items()}
    require(all(path.is_relative_to(S) for path in paths.values()),
            'Native rotation/codec/loader imports escaped the recovered complete snapshot: ' + repr(paths))
    had, factor = modules['rotation.hadamard_utils'].get_hadK(4032)
    require(factor == 252 and tuple(had.shape) == (252, 252), 'Keep the original full-RHT4032/252 factor.')
    return modules, {name: bind(path) for name, path in paths.items()}


def inputs():
    import torch
    prepared, report, audit = read(OLD / 'prepared.json'), read(OLD / 'formal32/report.json'), read(AUDIT)
    require((OLD / 'formal32/run.exit').read_text().strip() == '0'
            and report['schema'] == 'G09_Muon_OPD_execution_v1'
            and report['experiment_schema'] == 'G09_Hotpot192_coverage_v1'
            and report['status'] == 'G09_Muon_nativegreedy_OPD_formal32_complete'
            and report['phase'] == 'formal32' and report['steps'] == 32
            and report['source_checkpoint_id'] == prepared['source_checkpoint_id'] == SOURCE_ID
            and report['prepared_sha256'] == sha(OLD / 'prepared.json')
            and report['contract'] == prepared['contract']
            and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
            and report['endpoint']['save_reload_install_exact'], 'Require the actual completed H19232 endpoint.')
    trajectory = report['training']
    require(trajectory['steps'] == 32 and trajectory['group_indices'] == list(range(32))
            and [row['group_index'] for row in trajectory['trajectory']] == list(range(32))
            and [row['step'] for row in trajectory['trajectory']] == list(range(1, 33))
            and trajectory['frozen_state_before'] == trajectory['frozen_state_after']
                == prepared['record']['validated_frozen_state']
            and trajectory['trainable_tensors'] == 1512 and trajectory['trainable_parameters'] == 4762368,
            'Changed full decoder trajectory or frozen complement.')
    patch = OLD / 'formal32/decoder_endpoint.pt'
    require(sha(patch) == report['endpoint']['sha256'] and patch.stat().st_size == report['endpoint']['bytes'],
            'Transferred endpoint differs from its completed report.')
    payload = torch.load(patch, map_location='cpu', weights_only=True)
    require(payload['schema'] == 'G09_Muon_OPD_decoder_endpoint_v1' and payload['steps'] == 32
            and payload['source_checkpoint_id'] == SOURCE_ID
            and payload['execution']['prepared_sha256'] == report['prepared_sha256']
            and payload['execution']['contract'] == prepared['contract']
            and payload['frozen_state'] == prepared['record']['validated_frozen_state'], 'Changed endpoint binding.')
    source = resolve_legacy_path(audit['checkpoint_dir'])
    meta = read(source / 'checkpoint_meta.json')
    require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id'] == SOURCE_ID
            and audit['fresh_strict_reload_state_exact'] and audit['saved_training_decoder_BF16_exact']
            and audit['selected_global_step'] == 1200 and audit['bit_budget']['budget_passed'],
            'Require the original verified G09 parent.')
    original = torch.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    decoder = payload['state']
    declared = prepared['decoder_parameters']
    require(len(original) == 3423 and len(decoder) == len(declared) == 1512 and set(decoder) == set(declared)
            and sum(t.numel() for t in decoder.values()) == 4762368, 'Changed actual3423/1512 decoder inventory.')
    for key, tensor in decoder.items():
        require(tensor.dtype == original[key].dtype == torch.bfloat16 and tensor.device.type == 'cpu'
                and tensor.shape == original[key].shape and tensor.is_contiguous() and torch.isfinite(tensor).all().item()
                and list(tensor.shape) == declared[key]['shape'], 'Invalid real BF16 decoder: ' + key)
    dataset = OLD / 'dataset.pt'
    require(sha(dataset) == prepared['dataset_binding']['sha256'], 'Changed original H192 train dataset.')
    return source, meta, original, decoder, prepared, report


def prepare():
    modules, imports = native()
    source, meta, original, decoder, historical, report = inputs()
    budget = modules['train_utils.checkpoint_bpw'].audit_state_dict(meta, original, max_bpw=2.6, max_lora_rank=0)
    expected_budget = modules['train_utils.checkpoint_bpw'].audit_state_dict(
        meta, {**original, **decoder}, max_bpw=2.6, max_lora_rank=0)
    require(budget['budget_passed'] and expected_budget['budget_passed']
            and all(budget[key] == expected_budget[key] == historical['source_bit_budget'][key] for key in FIELDS),
            'Endpoint changed actual native bit/storage/rank0 budget.')
    records = [OLD / 'prepared.json', OLD / 'formal32/report.json', OLD / 'formal32/run.exit',
               OLD / 'formal32/decoder_endpoint.pt', OLD / 'dataset.pt', AUDIT,
               source / 'checkpoint_meta.json', source / meta['state_dict_file'], Path(__file__),
               Path(__file__).with_name('cmcc_native_loader.py'), HF / 'config.json', HF / 'generation_config.json']
    result = dict(schema='cmcc_H192_completed_endpoint_export_binding_v1', status='CPU_verified_ready_for_native_export',
        source_checkpoint_id=SOURCE_ID, source_checkpoint=str(source), hf_path=str(HF), output_root=str(OUT),
        original_prepared_sha256=report['prepared_sha256'], historical_training_repeated=False,
        identity_rule='Historical path/inode/mtime retained as evidence; new files bound by actual path/SHA/stat.',
        input_bindings=[bind(path) for path in records], native_imports=imports,
        historical_frozen_state=historical['record']['validated_frozen_state'],
        decoder_tensors=1512, decoder_parameters=4762368, source_bit_budget=budget,
        original_export_failure='Mixed main-repo get_hadK with snapshot loader; full4032 requires existing252 factor.',
        constructor='Verified no_init/from_config Qwen3 BF16/FA2 architecture; canonical complete-v6 strict load; no HF shard reads.',
        prepared_at_unix=time.time(), python=sys.executable)
    require(not PREPARED.exists(), 'Never overwrite a continuation prepared identity.')
    write(PREPARED, result)
    print(json.dumps(dict(status=result['status'], source_checkpoint=str(source),
        logical_bpw=budget['user_linear_bpw'], packed_bpw=budget['packed_linear_bpw'])), flush=True)


def export():
    import torch
    modules, imports = native()
    prepared = read(PREPARED)
    require(prepared['native_imports'] == imports and prepared['python'] == sys.executable
            and prepared['output_root'] == str(OUT) and prepared['hf_path'] == str(HF), 'Changed cmcc export runtime binding.')
    for item in prepared['input_bindings']:
        require(bind(item['path']) == item, 'Changed export input: ' + item['path'])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Keep CUDA visible for canonical FA2 construction.')
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal
    from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
    from experiments.longbench_v1.common import file_stat
    from transformers import AutoTokenizer
    source, meta, original, decoder, historical, report = inputs()
    expected = {**original, **decoder}
    from cmcc_native_loader import load_complete_checkpoint
    loader = load_complete_checkpoint
    checkpoint = modules['train_utils.checkpoint_v6']
    output = OUT / 'native_final/final_model'
    require(not output.exists(), 'Never overwrite the new endpoint.')
    model, loaded, result = loader(source, base_model_path=HF)
    require(loaded == meta and not result.missing_keys and not result.unexpected_keys, 'Fresh source strict load failed.')
    live = model.state_dict()
    for key, tensor in decoder.items():
        live[key].copy_(tensor)
    model.eval().requires_grad_(False)
    checkpoint.refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected, 'Completed H192 all3423 native state')
    provenance = dict(historical_report=bind(OLD / 'formal32/report.json'), endpoint=bind(OLD / 'formal32/decoder_endpoint.pt'),
        continuation_prepared=bind(PREPARED), base_checkpoint_id=SOURCE_ID,
        actual_target_contract=historical['contract'], original_prepared_sha256=report['prepared_sha256'],
        source_bit_budget=prepared['source_bit_budget'], original_training_repeated=False)
    saved = checkpoint.save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(), train_mode='none',
        after_category_mode='none', lora_config=None, base_model_path=str(HF),
        tokenizer=AutoTokenizer.from_pretrained(str(HF), use_fast=True, local_files_only=True),
        extra_meta={'cmcc_H192_completed_endpoint': provenance})
    del live, model
    gc.collect()
    serialized = torch.load(output / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    compared = _assert_state_equal(serialized, expected, 'Serialized H192 all3423')
    fresh, loaded, result = loader(output)
    fresh_compared = _assert_state_equal(fresh.state_dict(), expected, 'Fresh strict H192 all3423')
    require(not result.missing_keys and not result.unexpected_keys and saved['checkpoint_id'] == loaded['checkpoint_id'] != SOURCE_ID
            and {_['name']: _inference_spec(_) for _ in loaded['converted_modules']}
                == {_['name']: _inference_spec(_) for _ in meta['converted_modules']}
            and loaded.get('immutable_resume_contract') is None and loaded['lora_config'] is None
            and type(fresh.lm_head) is torch.nn.Linear and not fresh.config.tie_word_embeddings,
            'Fresh endpoint identity/geometry/topology changed.')
    budget = modules['train_utils.checkpoint_bpw'].audit_state_dict(loaded, serialized, max_bpw=2.6, max_lora_rank=0)
    require(budget['budget_passed'] and all(budget[key] == prepared['source_bit_budget'][key] for key in FIELDS), 'Changed native storage.')
    frozen = {key: tensor for key, tensor in original.items() if key not in decoder}
    frozen_count = _assert_state_equal({key: serialized[key] for key in frozen}, frozen, 'H1921911 frozen complement')
    audit = dict(schema='cmcc_H192_completed_endpoint_native_audit_v1', status='verified', phase='formal32', profile='G',
        checkpoint_dir=str(output), checkpoint_id=loaded['checkpoint_id'], source_checkpoint_id=SOURCE_ID,
        serialized_tensors_compared=compared, fresh_strict_tensors_compared=fresh_compared,
        fresh_strict_reload_state_exact=True, frozen_tensors_compared=frozen_count, code_bank_count=504,
        source_geometry_unchanged=True, bit_budget=budget, source_bit_budget=prepared['source_bit_budget'],
        endpoint_report=bind(OLD / 'formal32/report.json'), prepared_binding=bind(PREPARED),
        checkpoint_weight_stat=file_stat(output / loaded['state_dict_file']),
        checkpoint_meta_sha256=sha(output / 'checkpoint_meta.json'), sources=provenance, native_imports=imports,
        Muon_fixed_kappa=report['training']['Muon_fixed_kappa'], completed_at_unix=time.time())
    write(OUT / 'native_audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    if args.prepare_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU prepare hides CUDA.')
        prepare()
    else:
        export()

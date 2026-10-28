"""Fixed decoder recovery contracts and CPU native state checks for this night."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = TOPIC / 'qwen_recompression_followup_20261026/recovery'
DATA = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/recovery')
PREPARED = OUTPUT / 'prepared.json'
HERE = Path(__file__).resolve().parent
SNAPSHOT = TOPIC / 'rotation_factor_probe/source_snapshot_20261005_01'
# Q/K full rotation needs the snapshot's exact extended Hadamard families.
sys.path.insert(0, str(SNAPSHOT))
HF = Path('/home/star/models/Qwen/Qwen3-8B')
SOURCES = {
    'C': Path('/data/star/VAELLM/qwen_recompression_20261026/C/native/home_star_models_Qwen_Qwen3-8B_20261026_172959/final_model'),
    'G': Path('/data/star/VAELLM/qwen_recompression_20261026/G/native/home_star_models_Qwen_Qwen3-8B_20261026_130001/final_model'),
}
SOURCE_IDS = {'C': '91360b50-b260-4136-bc8d-495aa96d6273', 'G': '956b5a70-f830-40cb-9d14-f155de316344'}
GPUS = {('C', 'ce10'): 0, ('C', 'ce50'): 1, ('G', 'ce10'): 2, ('G', 'ce50'): 3}
SCHEMA = 'qwen_CG_decoder_CE10_CE50_2000_followup_v1'
CORE_FILES = (
    'rotation/weight_preprocess.py', 'litebsq/vae_linear.py',
    'compressed_e2e_fintuning/runtime_v6.py', 'compressed_e2e_fintuning/runtime_v6_pipeline.py',
    'compressed_e2e_fintuning/args.py', 'train_utils/v6_model_loader.py',
    'train_utils/distill_precision.py', 'train_utils/checkpoint_v6.py',
    'experiments/qwen3_w25/reproduction_recipe.py', 'experiments/qwen3_w25/audit_e2e_decoder.py',
    'experiments/qwen3_w25/assemble_categories.py', 'experiments/qwen3_w25/assemble_runtime_categories.py',
    'experiments/cat_recompression/longbench.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stat(path):
    path = Path(path).resolve()
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, inode=value.st_ino,
                device=value.st_dev, mtime_ns=value.st_mtime_ns)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_sources():
    return sorted(HERE.glob('recovery*.py'))


def check_prepared(path=PREPARED):
    record = read(path)
    require(record['schema'] == SCHEMA and record['status'] == 'cpu_verified', 'Use the actual completed CPU preparation.')
    for filename, expected in record['source_sha256'].items():
        require(sha(filename) == expected, 'Frozen recovery source changed: ' + filename)
    for item in record['file_stats']:
        require(stat(item['path']) == item, 'Bound source/data stat changed: ' + item['path'])
    return record


def environment(gpu=''):
    from experiments.qwen3_w25.reproduce_from_bf16 import environment as original
    env = original(str(gpu))
    env.update(PYTHONDONTWRITEBYTECODE='1', MKL_NUM_THREADS='4',
               PYTHONPATH=str(SNAPSHOT) + os.pathsep + str(ROOT))
    return env


def check_gpu(gpu):
    value = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    require(not value.strip(), f'Assigned GPU{gpu} is occupied: {value.strip()}')


def checkpoint(directory):
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.checkpoint_bpw import audit_checkpoint
    # Its structural checks have no 2.5 rejection; replace only the old budget result.
    meta, _old_budget = _validate_checkpoint(Path(directory))
    budget = audit_checkpoint(directory, max_bpw=2.6, max_lora_rank=0)
    require(budget['budget_passed'] and not budget['contract_violations'], 'Native rank0/logical2.6 storage failed.')
    require(all(budget['components'].get(key, {}).get('storage_bits', 0) == 0
                for key in ('lora', 'residual_lora')), 'No deployed low-rank branches.')
    return meta, budget


def load_cpu(directory):
    helper = module_from(TOPIC / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py', '_night_recovery_cpu_loader')
    return helper.load_cpu_native(directory)


def cpu_source(profile):
    import torch
    from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
    source = SOURCES[profile]
    meta, budget = checkpoint(source)
    require(meta['checkpoint_id'] == SOURCE_IDS[profile], 'Wrong actual C/G source identity.')
    model, loaded_meta, loaded = load_cpu(source)
    require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta['checkpoint_id'] == meta['checkpoint_id'],
            'Source native strict load failed.')
    saved = _load_state(source)
    count = _assert_state_equal(model.state_dict(), saved, 'CPU source fresh strict')
    inventory, banks = persistent_inventory(model)
    require(count == 3423 and sum(map(len, inventory.values())) == 1512 and sum(map(len, banks.values())) == 504,
            'Expected the genuine rank0/S2 C/G decoder inventory.')
    result = dict(checkpoint=str(source), checkpoint_id=meta['checkpoint_id'], tensors=count,
                  decoder_tensors=1512, code_banks=504, fresh_strict_state_exact=True, bit_budget=budget,
                  metadata_sha256=sha(source / 'checkpoint_meta.json'), weight_stat=stat(source / 'pytorch_model.bin'))
    del model, saved
    gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU source checking must not initialize CUDA.')
    return result


def audit_final(source, final, training_checkpoint, output):
    import torch
    from experiments.qwen3_w25.audit_e2e_decoder import (
        _audit_decoder_export_state, _inference_spec, _validate_export_storage, persistent_inventory,
    )
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
    from experiments.qwen3_w25.export_e2e_selected import inspect_step
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    require(not Path(output).exists(), 'Do not overwrite a native endpoint audit.')
    start = time.monotonic()
    source, final = Path(source), Path(final)
    original_meta, original_budget = checkpoint(source)
    meta, budget = checkpoint(final)
    step_meta, state, _snapshot, bound_source, _ = inspect_step(training_checkpoint)
    require(state['global_step'] == state['max_steps'] == 2000 and bound_source == source.resolve()
            and step_meta['train_mode'] == 'decoder', 'Use the actual final decoder step from this source.')
    require(meta['checkpoint_id'] != original_meta['checkpoint_id'] and meta['lora_config'] is None
            and meta['immutable_resume_contract']['input_checkpoint_id'] == original_meta['checkpoint_id'],
            'Native final source identity differs.')
    require({x['name']: _inference_spec(x, export_decoder_dtype=True) for x in original_meta['converted_modules']}
            == {x['name']: _inference_spec(x) for x in meta['converted_modules']}, 'Native final geometry changed.')
    original, actual = _load_state(source), _load_state(final)
    model, loaded_meta, loaded = load_cpu(final)
    require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta['checkpoint_id'] == meta['checkpoint_id'],
            'Final native strict loading failed.')
    inventory, banks = persistent_inventory(model)
    compared = _assert_state_equal(model.state_dict(), actual, 'Independent native final strict load')
    report = _audit_decoder_export_state(original, actual, inventory, banks, train_mode='decoder')
    _validate_export_storage(original_budget, budget, report)
    mutable, manifest = load_v6_training_model_state(str(training_checkpoint), map_location='cpu')
    decoder_keys = {key for values in inventory.values() for key in values}
    require(set(mutable) == decoder_keys and len(mutable) == 1512
            and all(row['component_class'] == 'decoder' for row in manifest), 'Only all1512 decoder fields are mutable.')
    _assert_state_equal({key: actual[key] for key in decoder_keys},
                        {key: value.to(dtype=torch.bfloat16) for key, value in mutable.items()},
                        'Actual saved FP32 step to native BF16 final')
    require(read(final.parent / 'run_meta.json')['global_step'] == 2000, 'Native runtime did not finish2000.')
    report.update(status='verified', checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=original_meta['checkpoint_id'],
                  checkpoint_dir=str(final), training_checkpoint=str(training_checkpoint), bit_budget=budget,
                  source_bit_budget=original_budget, serialized_tensors_compared=len(actual),
                  fresh_strict_tensors_compared=compared, fresh_strict_reload_state_exact=True,
                  saved_training_decoder_BF16_exact=True, native_finalization=meta['finalized_status'],
                  export_used_cuda=False, elapsed_seconds=time.monotonic() - start)
    write(output, report)
    del model, original, actual, mutable
    gc.collect()
    require(not torch.cuda.is_initialized(), 'The independent audit must remain on CPU.')
    return report


def assemble(profile, output):
    from experiments.qwen3_w25.assemble_categories import (
        EXPECTED, _assert_state_equal, _expected_state, _load_state, _replace_modules,
    )
    from experiments.qwen3_w25.assemble_runtime_categories import _validate_sources
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint
    records = {recipe: read(OUTPUT / profile / recipe / 'pipeline.json') for recipe in ('ce10', 'ce50')}
    require(all(row['status'] == 'complete_donor' and row['profile'] == profile for row in records.values()),
            'Both same-source fixed2000 donors must finish native audit and full8.')
    base, donor = (Path(records[key]['final_model']) for key in ('ce50', 'ce10'))
    require(not output.exists(), 'Assembly output must be new.')
    categories = ['gate_proj', 'up_proj', 'down_proj']
    base_meta, donor_meta, names, original, tokenizer, runtime = _validate_sources(base, donor, categories)
    for key, directory in (('ce50', base), ('ce10', donor)):
        audit = read(OUTPUT / profile / key / 'native_audit.json')
        require(audit['status'] == 'verified' and audit['checkpoint_id'] == read(directory / 'checkpoint_meta.json')['checkpoint_id']
                and audit['source_checkpoint_id'] == SOURCE_IDS[profile], 'Donor does not descend from this profile source.')
    base_state, donor_state = _load_state(base), _load_state(donor)
    expected = _expected_state(base_state, donor_state, names, EXPECTED)
    model, _, _ = load_cpu(base)
    donor_model, _, _ = load_cpu(donor)
    _replace_modules(model, donor_model, names)
    _assert_state_equal(model.state_dict(), expected, 'Fixed same-source CE50 attention/CE10 MLP assembly')
    budget = audit_state_dict(base_meta, expected, max_bpw=2.6, max_lora_rank=0)
    require(budget['budget_passed'] and not budget['contract_violations'], 'Actual assembly storage exceeds2.6/rank0.')
    provenance = dict(profile=profile, source_checkpoint_id=SOURCE_IDS[profile], base_checkpoint_dir=str(base),
                      base_checkpoint_id=base_meta['checkpoint_id'], donor_checkpoint_dir=str(donor),
                      donor_checkpoint_id=donor_meta['checkpoint_id'], categories=categories,
                      source_config_runtime_fields=runtime, purpose='fixed inference assembly; no exact-resume claim')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=base_meta['compressed_targets'], pending_dense_targets=(), skip_targets=(),
        train_mode='none', after_category_mode='none', lora_config=None, base_model_path=str(original),
        tokenizer=tokenizer, extra_meta={'category_assembly': provenance})
    serialized = _load_state(output)
    compared = _assert_state_equal(serialized, expected, 'Saved fixed assembly')
    del model, donor_model
    gc.collect()
    fresh, actual_meta, loaded = load_cpu(output)
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'Assembly fresh strict load failed.')
    _assert_state_equal(fresh.state_dict(), expected, 'Independent assembly strict reload')
    require(actual_meta['checkpoint_id'] == saved['checkpoint_id']
            and saved['checkpoint_id'] not in (base_meta['checkpoint_id'], donor_meta['checkpoint_id'])
            and actual_meta.get('immutable_resume_contract') is None, 'Assembly inference identity is invalid.')
    _, actual_budget = checkpoint(output)
    require(actual_budget['components'] == budget['components'], 'Saved assembly storage differs from actual planned state.')
    audit = dict(status='verified', checkpoint_id=saved['checkpoint_id'], checkpoint_dir=str(output),
                 serialized_tensors_compared=compared, fresh_strict_tensors_compared=compared,
                 fresh_strict_reload_state_exact=True, sources=provenance, bit_budget=actual_budget, export_used_cuda=False)
    write(output / 'assembly_report.json', audit)
    del fresh, serialized, expected, base_state, donor_state
    gc.collect()
    return audit

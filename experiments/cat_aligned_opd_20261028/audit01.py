"""Independent terminal audit of the running, immutable aligned Q/K2000 stage."""
import gc
import os
from pathlib import Path
import sys
import time

import bindings as b
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.w25_eval.subset_audit import audit_subset_training_state
from experiments.w25_eval.subset_state import persistent_inventory
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_training_model_state, load_v6_training_step_meta
from train_utils.v6_model_loader import load_v6_model_checkpoint


@b.torch.no_grad()
def main():
    started = time.monotonic()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and not b.SOURCE_AUDIT.exists(),
              'Use bitvae and a new terminal audit; never overwrite an audit.')
    b.torch.set_num_threads(4)
    prepared = b.read(b.STAGE01 / 'prepared.json')
    training, execution = b.read(b.STAGE01 / 'training.json'), b.read(b.STAGE01 / 'execution.json')
    b.require(training['status'] == execution['status'] == 'completed_native_pending_audit'
              and execution['exit_code'] == 0 and (b.STAGE01 / 'run.exit').read_text().strip() == '0'
              and training['training_result']['global_step'] == 2000
              and training['completed_steps'] == [400, 800, 1200, 1600, 2000], '01 has not actually completed all2000 updates.')
    b.check_hashes(prepared['source_sha256']); b.check_hashes(prepared['input_evidence_sha256'])
    b.check_stat(prepared['source_weight_stat'])
    source = Path(prepared['source_checkpoint']).resolve()
    final = Path(training['training_result']['saved_model_dir']).resolve()
    run = Path(training['training_result']['run_output_dir']).resolve()
    step = run / 'trainer_state/checkpoint-2000'
    b.require(final.is_relative_to(b.STAGE01) and run.is_relative_to(b.STAGE01)
              and final.name == 'final_model' and not final.is_symlink(), 'Actual 01 final escaped its bound run.')
    source_meta, meta, step_meta = b.read(source / 'checkpoint_meta.json'), b.read(final / 'checkpoint_meta.json'), load_v6_training_step_meta(str(step))
    state = b.read(step / 'trainer_state.json')
    b.require(step_meta['round_base_checkpoint_id'] == source_meta['checkpoint_id'] == prepared['source_checkpoint_id']
              and meta['checkpoint_id'] == training['training_result']['checkpoint_id']
              and meta['checkpoint_id'] not in (source_meta['checkpoint_id'], step_meta['checkpoint_id'])
              and state['global_step'] == state['max_steps'] == 2000
              and meta['immutable_resume_contract'] == step_meta['immutable_resume_contract']
              and meta['target_layers'] == step_meta['target_layers'] == list(range(36))
              and set(meta['target_modules']) == set(step_meta['target_modules']) == {'q_proj', 'k_proj'}
              and meta['train_mode'] == step_meta['train_mode'] == 'decoder'
              and meta['norm_train_mode'] == meta['lm_head_train_mode'] == 'none' and meta['lora_config'] is None
              and not meta.get('post_norm_head_linear'), '01 source/scope/terminal training contract changed.')
    b.require({_['name']: _inference_spec(_) for _ in source_meta['converted_modules']}
              == {_['name']: _inference_spec(_) for _ in meta['converted_modules']}, '01 changed native geometry/rotation/protection.')
    original, actual = b.load_state(source), b.load_state(final)
    model, loaded, result = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == source_meta and not result.missing_keys and not result.unexpected_keys, '01 source strict loading failed.')
    _assert_state_equal(model.state_dict(), original, '01 round-base strict source')
    inventory, banks = persistent_inventory(model, prepared['architecture'])
    del model
    gc.collect()
    mutable, manifest = load_v6_training_model_state(str(step), map_location='cpu')
    expected = {key for name in prepared['selected_targets'] for key in inventory[name]}
    b.require(set(mutable) == expected and len(expected) == len(manifest) == 432
              and {row['name']: row['component_class'] for row in manifest} == {key: 'decoder' for key in expected},
              '01 FP32 endpoint is not exactly all72 Q/K decoders.')
    report = audit_subset_training_state(original, actual, inventory, banks,
        names=prepared['selected_targets'], mutable=mutable, packed_snapshot=None, train_mode='decoder')
    source_budget = audit_state_dict(source_meta, original, max_bpw=2.6, max_lora_rank=0)
    budget = audit_state_dict(meta, actual, max_bpw=2.6, max_lora_rank=0)
    b.require(len(original) == len(actual) == 3423 and report['frozen_tensors_compared'] == 2991
              and report['code_bank_count'] == 504 and report['packed_codes_unchanged']
              and source_budget['budget_passed'] and budget['budget_passed']
              and all(source_budget[k] == budget[k] for k in b.BUDGET_FIELDS), '01 frozen state or native storage budget changed.')
    fresh, loaded, result = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == meta and not result.missing_keys and not result.unexpected_keys, '01 final fresh strict loading failed.')
    compared = _assert_state_equal(fresh.state_dict(), actual, '01 final fresh strict3423')
    b.require(type(fresh.lm_head) is b.torch.nn.Linear and not fresh.config.tie_word_embeddings
              and persistent_inventory(fresh, prepared['architecture']) == (inventory, banks), '01 native topology changed.')
    evidence = [b.STAGE01 / 'prepared.json', b.STAGE01 / 'training.json', b.STAGE01 / 'execution.json',
        final / 'checkpoint_meta.json', source / 'checkpoint_meta.json', step / 'checkpoint_meta.json',
        step / 'trainer_state.json', step / 'training_model_state.pt']
    report.update(schema='aligned_QK2000_terminal_native_v1', status='verified', profile=b.PROFILE,
        checkpoint_dir=str(final), checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=source_meta['checkpoint_id'],
        selected_global_step=2000, selected_checkpoint_id=step_meta['checkpoint_id'], training_checkpoint_dir=str(step),
        serialized_tensors_compared=len(actual), fresh_strict_tensors_compared=compared,
        fresh_strict_reload_state_exact=True, source_bit_budget=source_budget, bit_budget=budget,
        checkpoint_weight_stat=b.file_stat(final / meta['state_dict_file']),
        input_evidence_sha256={str(path): b.sha256(path) for path in evidence},
        audit_code_sha256={str(path): b.sha256(path) for path in (Path(__file__).resolve(), b.HERE / 'bindings.py')},
        CUDA_initialized=b.torch.cuda.is_initialized(), scope='CPU state/cast/storage and independent native strict loading; no forward.',
        elapsed_seconds=time.monotonic()-started)
    b.core.write(b.SOURCE_AUDIT, report)
    print(dict(status=report['status'], profile=b.PROFILE, checkpoint_id=meta['checkpoint_id'],
        frozen=2991, fresh_strict=compared, logical_bpw=budget['user_linear_bpw']), flush=True)


if __name__ == '__main__':
    main()

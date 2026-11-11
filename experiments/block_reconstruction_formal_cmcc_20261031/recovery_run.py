"""Run the real BR endpoint through the unchanged original500/Muon32 prefix."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from transformers import AutoModelForCausalLM

from . import recovery_bindings as b
from experiments.cmcc_runtime_20261030 import sequence
from experiments.fresh_muon_cmcc_20261030.run import supervision
from experiments.muon_compact_cmcc_20261030.prepare import exact


@b.torch.no_grad()
def load_pair(prepared):
    record = prepared['record']
    teacher = AutoModelForCausalLM.from_pretrained(str(b.HF), local_files_only=True,
        trust_remote_code=False, torch_dtype=b.torch.bfloat16,
        attn_implementation='flash_attention_2').eval().requires_grad_(False)
    teacher.seqlen = 2048
    model, meta, loaded, budget = b.checkpoint.load(prepared['source_binding']['checkpoint_dir'])
    b.require(meta['checkpoint_id'] == record['source_checkpoint_id']
        and not loaded.missing_keys and not loaded.unexpected_keys
        and budget == prepared['source_budget']
        and b.core.native_policy(model) == b.core.native_policy(teacher),
        'Load the genuine BR student and the independently loaded original full-HF BF16/FA2 teacher.')
    model.to('cuda:0')
    return model, teacher, dict(source_BR_checkpoint_id=meta['checkpoint_id'],
        ancestor_C0_checkpoint_id=prepared['source_C0_binding']['checkpoint_id'],
        persistent_tensors=len(model.state_dict()), source_budget=budget,
        policy=b.core.native_policy(model),
        source_nonprojection_HF_exact=record['teacher_identity']['nonprojection_tensors_compared'])


def run(prepared_path, gpu, seconds):
    b.configure()
    prepared = b.read_prepared(prepared_path)
    record = prepared['record']
    b.require(type(gpu) is int and 0 <= gpu <= 7 and Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
        and seconds > 0,
        'Use caller bitvae and exactly one assigned physical GPU0..7, exposed as logical cuda0.')
    resources = sequence.check_resources([str(gpu)])
    b.require(b.torch.cuda.device_count() == 1, 'Expose the single assigned GPU as logical cuda0.')
    output = Path(prepared_path).resolve().parent / b.RUN_NAME
    b.require(not output.exists(), 'Never overwrite or automatically retry a BR->E execution.')
    output.mkdir()
    started = time.monotonic()
    identity = dict(schema='BR_E32_execution_v1', source_checkpoint_id=record['source_checkpoint_id'],
        source_C0_binding=prepared['source_C0_binding'], BR_record=prepared['BR_record'],
        prepared=b.binding(prepared_path), phase='formal32', group_indices=b.GROUPS, seed=137,
        physical_GPU=gpu, pid=os.getpid(), argv=sys.argv, python=sys.executable, cwd=str(Path.cwd()),
        contract=b.CONTRACT, implementation=b.code_bindings(), fresh_optimizer=True,
        original_schedule_steps=500, validation_only=False, resources=resources)
    b.core.write(output/'execution.json', identity)
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    observe = lambda event, values: print(json.dumps(dict(event=event,
        elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    try:
        dataset, recovery, cache, evidence = b.inputs.load_inputs()
        b.require(dataset['groups'] == record['groups'] and dataset['train_ids'] == record['train_ids']
            and evidence == prepared['M_target_evidence'], 'Prepared original H96 inputs or once-mixed M targets changed.')
        model, teacher, load_audit = load_pair(prepared)
        observe('real_BR_and_original_HF_loaded', load_audit)
        tokenizer = b.core.build_tokenizer(str(b.HF))
        training, initial = b.fit.fit(model, teacher, tokenizer, recovery, record,
            'formal32', observe, started+seconds)
        b.require(training['steps'] == 32 and training['group_indices'] == b.GROUPS,
            'Complete every original group0..31; a short endpoint is not eligible.')
        training['actual_OPD_supervision'] = supervision(training)
        del teacher, recovery
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
        with b.torch.no_grad():
            modules = b.mature.selected_modules(model)
            expected_decoder = {name:p.detach().cpu().bfloat16()
                for name,p in b.lifecycle.decoder_parameters(modules).items()}
            training['deployment'] = b.lifecycle.deployment_mode(model, modules)
            endpoint = b.lifecycle.copy_parameters(modules)
            exact(endpoint, expected_decoder, 'BR->E final FP32 masters to actual BF16 endpoint')
            b.require(b.lifecycle.frozen_proof(model, set(endpoint)) == record['validated_frozen_state'],
                'Deployment changed a frozen BR code, route, protection, rotation or nondecoder field.')
            model.config.use_cache = True
            fixture = dict(input_ids=dataset['training_cases'][0]['input_ids'].unsqueeze(0),
                source_case_id=dataset['training_cases'][0]['case_id'])
            values = model(input_ids=fixture['input_ids'].to('cuda:0'), use_cache=False, logits_to_keep=1).logits
            b.require(values.dtype == b.torch.bfloat16 and b.torch.isfinite(values).all().item(),
                'The real BR->E deployment must produce finite BF16 logits.')
            fixture['logits'] = values.cpu()
            b.torch.save(fixture, output/'same_input_output.pt')
            meta, budget, state = b.checkpoint.save(model, output/'final_model', provenance=dict(
                schema='BR_E32_provenance_v1', source_binding=prepared['source_binding'],
                source_C0_binding=prepared['source_C0_binding'], BR_record=prepared['BR_record'],
                source_checkpoint_id=record['source_checkpoint_id'], prepared=identity['prepared'],
                actual_updates=32, planned_schedule=500, group_indices=b.GROUPS,
                contract=b.CONTRACT, seed=137, phase='formal32', validation_only=False))
            b.require(all(budget[key] == prepared['source_budget'][key] for key in
                ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'full_model_tensor_bytes')),
                'Fixed-code BR->E must preserve the actual deployed tensor storage budget.')
            expected = {key:b.core.tensor_record(value) for key,value in state.items()}
            b.core.write(output/'expected_state.json', expected)
            exact(endpoint, {key:state[key] for key in endpoint}, 'serialized BR->E BF16 decoder endpoint')
            parameters = dict(model.named_parameters())
            for key,value in initial.items():
                parameters[key].data.copy_(value.to(parameters[key].device))
            b.lifecycle.clear(modules)
            exact(b.lifecycle.copy_parameters(modules), initial, 'genuine BR source restoration')
            b.require(b.lifecycle.frozen_proof(model, set(initial)) == record['validated_frozen_state'],
                'BR source restoration changed frozen persistent fields.')
        report = dict(**identity, status='BR_E32_training_export_complete_fresh_pending',
            steps=32, training=training, load_audit=load_audit, target_cache=cache,
            unchanged_M_target_evidence=evidence, source_restored_exact=True,
            saved_decoder_matches_final_FP32_cast_exact=True,
            export=dict(directory=str(output/'final_model'), checkpoint_id=meta['checkpoint_id'],
                bit_budget=budget, tensors=len(state), expected_state=b.binding(output/'expected_state.json'),
                fixture=b.binding(output/'same_input_output.pt')),
            elapsed_seconds=time.monotonic()-started)
        b.core.write(output/'report.json', report)
        (output/'run.exit').write_text('0\n', encoding='utf-8')
        observe('BR_E32_training_export_complete', dict(checkpoint_id=meta['checkpoint_id']))
    except BaseException:
        b.core.write(output/'failure.json', dict(**identity, status='failed', error=traceback.format_exc()))
        (output/'run.exit').write_text('1\n', encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=range(8))
    parser.add_argument('--timeout-seconds', type=int, default=43200)
    args = parser.parse_args()
    run(args.prepared, args.gpu, args.timeout_seconds)


if __name__ == '__main__':
    main()

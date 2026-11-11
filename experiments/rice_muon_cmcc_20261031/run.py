"""Independent original-C1 lifecycle2 or original500 schedule prefix32; shared Muon math."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from experiments.fresh_muon_cmcc_20261030.run import load_pair, supervision
from experiments.muon_compact_cmcc_20261030.prepare import exact


def run(prepared_path, gpu, phase, seconds):
    b.configure()
    prepared_path = Path(prepared_path).resolve()
    prepared = b.read_prepared(prepared_path)
    record, groups = prepared['record'], b.PHASES[phase]
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and seconds > 0,
        'Use caller bitvae and the single assigned physical GPU, exposed as logical cuda0.')
    b.require(b.torch.cuda.device_count() == 1, 'Expose exactly the assigned single GPU.')
    short = b.verified_Rice_short(prepared_path) if phase == 'formal32' else None
    b.require(short is None or short['source_checkpoint_id'] == record['source_checkpoint_id']
        and short['checkpoint_id'] != record['source_checkpoint_id'], 'Engineering2 must verify this same original C1 parent.')
    output = prepared_path.parent / phase
    b.require(not output.exists(), 'Never overwrite, resume or automatically retry a Rice Muon run.')
    output.mkdir()
    started = time.monotonic()
    identity = dict(schema='fresh_C1_Rice_Muon_execution_v1', source_checkpoint_id=record['source_checkpoint_id'],
        source_binding=prepared['source_binding'], prepared=b.binding(prepared_path), phase=phase,
        validation_only=phase == 'short2', group_indices=groups, seed=137, physical_GPU=gpu,
        pid=os.getpid(), argv=sys.argv, python=sys.executable, cwd=str(Path.cwd()),
        contract=b.CONTRACT, implementation=b.code_bindings(), fresh_optimizer=True,
        original_schedule_steps=500, Rice_lifecycle_evidence=short)
    b.core.write(output / 'execution.json', identity)
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    observe = lambda event, values: print(json.dumps(dict(event=event,
        elapsed_seconds=time.monotonic() - started, **values)), flush=True)
    try:
        dataset, recovery, cache, evidence = b.inputs.load_inputs()
        b.require(dataset['groups'] == record['groups'] and dataset['train_ids'] == record['train_ids']
            and evidence == prepared['M_target_evidence'], 'Prepared original H96 inputs or once-mixed M targets changed.')
        # The reused loader reads only prepared.source_binding, always original C1.
        # It has no short-checkpoint or optimizer-state loading path.
        model, teacher, load_audit = load_pair(prepared)
        b.require(b.rice_proof(model) == record['validated_Rice_state'], 'Original C1 Rice fields changed before training.')
        load_audit['student_source'] = 'fresh_C1_Rice'
        observe('real_fresh_C1_Rice_and_original_HF_loaded', load_audit)
        tokenizer = b.core.build_tokenizer(str(b.HF))
        training, initial = b.fit.fit(model, teacher, tokenizer, recovery, record, phase, observe, started + seconds)
        b.require(training['steps'] == len(groups) and training['group_indices'] == groups,
            'Complete exactly the declared original group sequence; never count engineering2 toward formal32.')
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
            exact(endpoint, expected_decoder, 'actual final FP32 masters to BF16')
            b.require(b.lifecycle.frozen_proof(model, set(endpoint)) == record['validated_frozen_state']
                and b.rice_proof(model) == record['validated_Rice_state'],
                'Deployment changed a frozen code, protection, sign, Rice or nondecoder field.')
            model.config.use_cache = True
            fixture = dict(input_ids=dataset['training_cases'][0]['input_ids'].unsqueeze(0),
                source_case_id=dataset['training_cases'][0]['case_id'])
            values = model(input_ids=fixture['input_ids'].to('cuda:0'), use_cache=False, logits_to_keep=1).logits
            b.require(values.dtype == b.torch.bfloat16 and b.torch.isfinite(values).all().item(),
                'The real exported deployment must produce finite BF16 logits.')
            fixture['logits'] = values.cpu()
            b.torch.save(fixture, output / 'same_input_output.pt')
            meta, budget, state = b.checkpoint.save(model, output / 'final_model', provenance=dict(
                schema='fresh_C1_Rice_Muon_provenance_v1', source_binding=prepared['source_binding'],
                source_checkpoint_id=record['source_checkpoint_id'], prepared=identity['prepared'],
                actual_updates=len(groups), planned_schedule=500, group_indices=groups, contract=b.CONTRACT,
                seed=137, phase=phase, validation_only=phase == 'short2', Rice_lifecycle_evidence=short))
            b.require(meta['checkpoint_id'] != record['source_checkpoint_id']
                and all(budget[key] == prepared['source_budget'][key] for key in
                    ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'full_model_tensor_bytes')),
                'The new decoder endpoint needs a new UUID and unchanged actual tensor budget.')
            expected = {key:b.core.tensor_record(value) for key,value in state.items()}
            b.core.write(output / 'expected_state.json', expected)
            exact(endpoint, {key:state[key] for key in endpoint}, 'serialized BF16 endpoint')
            parameters = dict(model.named_parameters())
            for key,value in initial.items():
                parameters[key].data.copy_(value.to(parameters[key].device))
            b.lifecycle.clear(modules)
            exact(b.lifecycle.copy_parameters(modules), initial, 'actual original C1 source restoration')
            b.require(b.lifecycle.frozen_proof(model, set(initial)) == record['validated_frozen_state']
                and b.rice_proof(model) == record['validated_Rice_state'], 'Original C1 restoration changed frozen/Rice fields.')
        report = dict(**identity, status='Rice_Muon_training_export_complete_fresh_pending', steps=len(groups),
            training=training, load_audit=load_audit, target_cache=cache, unchanged_M_target_evidence=evidence,
            source_restored_exact=True, saved_decoder_matches_final_FP32_cast_exact=True,
            Rice_frozen_exact=True, Rice_state=record['validated_Rice_state'],
            export=dict(directory=str(output / 'final_model'), checkpoint_id=meta['checkpoint_id'], bit_budget=budget,
                tensors=len(state), expected_state=b.binding(output / 'expected_state.json'),
                fixture=b.binding(output / 'same_input_output.pt')), elapsed_seconds=time.monotonic() - started)
        b.core.write(output / 'report.json', report)
        (output / 'run.exit').write_text('0\n', encoding='utf-8')
        observe('Rice_Muon_training_export_complete', dict(phase=phase, checkpoint_id=meta['checkpoint_id']))
    except BaseException:
        b.core.write(output / 'failure.json', dict(**identity, status='failed', error=traceback.format_exc()))
        (output / 'run.exit').write_text('1\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=range(8))
    parser.add_argument('--phase', required=True, choices=('short2', 'formal32'))
    parser.add_argument('--timeout-seconds', type=int, default=43200)
    args = parser.parse_args()
    run(args.prepared, args.gpu, args.phase, args.timeout_seconds)

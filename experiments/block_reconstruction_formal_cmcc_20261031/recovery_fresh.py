"""Independently verify the complete real BR->Muon32 deployment endpoint."""
import argparse
import json
import os
from pathlib import Path
import sys

from . import recovery_bindings as b
from experiments.cmcc_runtime_20261030 import sequence


def fresh(prepared_path, gpu):
    b.configure()
    prepared = b.read_prepared(prepared_path)
    root = Path(prepared_path).resolve().parent/b.RUN_NAME
    report = b.read(root/'report.json')
    b.require(type(gpu) is int and 0 <= gpu <= 7 and Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
        and (root/'run.exit').read_text().strip() == '0'
        and report['schema'] == 'BR_E32_execution_v1' and report['validation_only'] is False
        and report['status'] == 'BR_E32_training_export_complete_fresh_pending'
        and report['phase'] == 'formal32' and report['steps'] == 32 and report['group_indices'] == b.GROUPS
        and report['prepared'] == b.binding(prepared_path) and report['contract'] == b.CONTRACT
        and report['source_checkpoint_id'] == prepared['source_binding']['checkpoint_id']
        and report['source_C0_binding'] == prepared['source_C0_binding'] and report['BR_record'] == prepared['BR_record'],
        'Require the genuine completed BR->E32 export with its real BR/C0 lineage and one assigned GPU.')
    b.require(not (root/'fresh_audit.json').exists(), 'Never replace an independent BR->E audit.')
    resources = sequence.check_resources([str(gpu)])
    b.require(b.torch.cuda.device_count() == 1, 'Expose the single assigned GPU as logical cuda0.')
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    for key in ('expected_state', 'fixture'):
        item = report['export'][key]
        b.require(b.binding(item['path']) == item, 'Actual BR->E export evidence changed: '+key)
    model, meta, strict, budget = b.checkpoint.load(root/'final_model')
    expected = b.read(root/'expected_state.json')
    actual = {key:b.core.tensor_record(value) for key,value in model.state_dict().items()}
    provenance = meta['provenance']
    b.require(actual == expected and not strict.missing_keys and not strict.unexpected_keys
        and meta['checkpoint_id'] == report['export']['checkpoint_id']
        and budget == report['export']['bit_budget'] and provenance['schema'] == 'BR_E32_provenance_v1'
        and provenance['source_binding'] == prepared['source_binding']
        and provenance['source_C0_binding'] == prepared['source_C0_binding']
        and provenance['BR_record'] == prepared['BR_record']
        and provenance['actual_updates'] == 32 and provenance['planned_schedule'] == 500
        and provenance['group_indices'] == b.GROUPS and provenance['validation_only'] is False,
        'Fresh full state, actual UUID, storage ledger or BR->E provenance differs.')
    frozen = b.lifecycle.frozen_proof(model, set(prepared['record']['decoder_parameters']))
    b.require(frozen == prepared['record']['validated_frozen_state'], 'Fresh BR->E nondecoder fields changed.')
    model.to('cuda:0').eval().requires_grad_(False)
    warmup = b.lifecycle.prewarm(b.mature.selected_modules(model))
    fixture = b.torch.load(root/'same_input_output.pt', map_location='cpu', weights_only=True)
    with b.torch.no_grad():
        logits = model(input_ids=fixture['input_ids'].to('cuda:0'), use_cache=False, logits_to_keep=1).logits.cpu()
    b.require(logits.dtype == fixture['logits'].dtype == b.torch.bfloat16
        and b.torch.isfinite(logits).all().item() and b.torch.equal(logits, fixture['logits']),
        'Fresh independent same-input finite BF16 BR->E deployment differs.')
    groups = b.checkpoint.inventories(model)
    b.require([len(groups[key]) for key in ('native','compact','pending_dense')] == [216,36,0],
        'The actual final BR->E mixed topology changed.')
    audit = dict(schema='BR_E32_fresh_audit_v1', status='verified',
        source_checkpoint_id=prepared['source_binding']['checkpoint_id'],
        source_C0_binding=prepared['source_C0_binding'], BR_record=prepared['BR_record'],
        checkpoint_id=meta['checkpoint_id'], physical_GPU=gpu, resources=resources,
        full_state_exact=True, same_complete_input_BF16_logits_exact=True, finite=True,
        serialized_tensors_compared=len(expected), fresh_strict_tensors_compared=len(actual),
        frozen_tensors_compared=frozen['tensors'], frozen_state=frozen,
        code_bank_count=sum('vq_weight' in key or '.codes_' in key for key in actual),
        inventories=groups, warmup=warmup, bit_budget=budget, input_tokens=fixture['input_ids'].shape[-1],
        prepared=b.binding(prepared_path), training_report=b.binding(root/'report.json'),
        export_meta=b.binding(root/'final_model/adaptive_model_meta.json'),
        scope='Actual BR->E32 deployment integrity; same-parent complete full9 remains the quality criterion.')
    b.core.write(root/'fresh_audit.json', audit)
    print(json.dumps(dict(status=audit['status'], checkpoint_id=meta['checkpoint_id'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=range(8))
    args = parser.parse_args()
    fresh(args.prepared, args.gpu)


if __name__ == '__main__':
    main()

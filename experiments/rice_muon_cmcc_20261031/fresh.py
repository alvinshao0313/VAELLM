"""Independent complete Rice state/frozen/ledger and same-input BF16 audit."""
import argparse
import json
import os
from pathlib import Path
import sys

from . import bindings as b


def fresh(prepared_path, gpu, phase):
    b.configure()
    prepared_path = Path(prepared_path).resolve()
    prepared = b.read_prepared(prepared_path)
    root, groups = prepared_path.parent / phase, b.PHASES[phase]
    report = b.read(root / 'report.json')
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and b.torch.cuda.device_count() == 1
        and (root / 'run.exit').read_text().strip() == '0'
        and report['schema'] == 'fresh_C1_Rice_Muon_execution_v1'
        and report['status'] == 'Rice_Muon_training_export_complete_fresh_pending'
        and report['phase'] == phase and report['steps'] == len(groups) and report['group_indices'] == groups
        and report['validation_only'] is (phase == 'short2')
        and report['prepared'] == b.binding(prepared_path) and report['contract'] == b.CONTRACT
        and report['source_binding'] == prepared['source_binding']
        and report['source_checkpoint_id'] == prepared['source_binding']['checkpoint_id'],
        'Require the actual completed phase, same original C1 source, and its real saved export.')
    b.require(not (root / 'fresh_audit.json').exists(), 'Never replace a prior independent Rice audit.')
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    for key in ('expected_state', 'fixture'):
        item = report['export'][key]
        b.require(b.binding(item['path']) == item, 'Actual export evidence changed: ' + key)
    model, meta, strict, budget = b.checkpoint.load(root / 'final_model')
    expected = b.read(root / 'expected_state.json')
    actual = {key:b.core.tensor_record(value) for key,value in model.state_dict().items()}
    provenance = meta['provenance']
    b.require(actual == expected and not strict.missing_keys and not strict.unexpected_keys
        and meta['checkpoint_id'] == report['export']['checkpoint_id'] != prepared['source_binding']['checkpoint_id']
        and budget == report['export']['bit_budget'] and provenance['schema'] == 'fresh_C1_Rice_Muon_provenance_v1'
        and provenance['source_binding'] == prepared['source_binding'] and provenance['prepared'] == b.binding(prepared_path)
        and provenance['phase'] == phase and provenance['actual_updates'] == len(groups)
        and provenance['planned_schedule'] == 500 and provenance['group_indices'] == groups
        and provenance['validation_only'] is (phase == 'short2'), 'Fresh complete state, identity, provenance or ledger differs.')
    frozen, rice = b.lifecycle.frozen_proof(model, set(prepared['record']['decoder_parameters'])), b.rice_proof(model)
    b.require(frozen == prepared['record']['validated_frozen_state']
        and rice == prepared['record']['validated_Rice_state'] == report['Rice_state'], 'Fresh non-D/Rice persistent fields changed.')
    model.to('cuda:0').eval().requires_grad_(False)
    warmup = b.lifecycle.prewarm(b.mature.selected_modules(model))
    fixture = b.torch.load(root / 'same_input_output.pt', map_location='cpu', weights_only=True)
    with b.torch.no_grad():
        logits = model(input_ids=fixture['input_ids'].to('cuda:0'), use_cache=False, logits_to_keep=1).logits.cpu()
    b.require(logits.dtype == fixture['logits'].dtype == b.torch.bfloat16
        and b.torch.isfinite(logits).all().item() and b.torch.equal(logits, fixture['logits']),
        'Independent same-input finite BF16 output differs.')
    inventory = b.checkpoint.inventories(model)
    b.require([len(inventory[key]) for key in ('native', 'compact', 'pending_dense')] == [216, 36, 0], 'Wrong final Rice topology.')
    b.require(all(budget[key] == prepared['source_budget'][key] for key in
        ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'full_model_tensor_bytes')),
        'Rice Muon changed actual deployed tensor budget.')
    audit = dict(schema='fresh_C1_Rice_Muon_fresh_audit_v1', status='verified', phase=phase,
        steps=len(groups), validation_only=phase == 'short2', source_checkpoint_id=prepared['source_binding']['checkpoint_id'],
        checkpoint_id=meta['checkpoint_id'], full_state_exact=True, same_complete_input_BF16_logits_exact=True, finite=True,
        serialized_tensors_compared=len(expected), fresh_strict_tensors_compared=len(actual),
        frozen_tensors_compared=frozen['tensors'], frozen_state=frozen, Rice_state=rice, Rice_frozen_exact=True,
        inventories=inventory, warmup=warmup, bit_budget=budget, input_tokens=fixture['input_ids'].shape[-1],
        prepared=b.binding(prepared_path), training_report=b.binding(root / 'report.json'),
        export_meta=b.binding(root / 'final_model/adaptive_model_meta.json'),
        scope='Actual Rice lifecycle/export integrity only; short2 is never a formal parent or quality result.')
    b.core.write(root / 'fresh_audit.json', audit)
    print(json.dumps(dict(status=audit['status'], phase=phase, checkpoint_id=meta['checkpoint_id'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=range(8))
    parser.add_argument('--phase', required=True, choices=('short2', 'formal32'))
    args = parser.parse_args()
    fresh(args.prepared, args.gpu, args.phase)

"""Independent exported-model-only strict state, frozen, and BF16 output audit."""
import json
import os

from . import bindings as b
from . import lifecycle


def main():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6' and b.torch.cuda.device_count() == 1,
              'Independent fresh audit uses only assigned GPU6.')
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    run = b.OUTPUT/'short2'
    report = b.read(run/'report.json')
    prepared = b.read(b.OUTPUT/'prepared.json')
    b.require((run/'run.exit').read_text().strip() == '0'
        and report['status'] == 'training_and_BF16_export_complete_fresh_pending', 'Require actual successful two-update export.')
    for key in ('expected_state','fixture'):
        binding = report['export'][key]
        b.require(b.binding(binding['path']) == binding, 'Changed fresh audit fixture: '+key)
    model,meta,result,budget = b.checkpoint.load(run/'final_model')
    expected = b.read(run/'expected_state.json')
    state = model.state_dict()
    actual = {key:b.core.tensor_record(value) for key,value in state.items()}
    b.require(actual == expected and not result.missing_keys and not result.unexpected_keys,
              'Fresh independent full persistent state differs.')
    frozen = lifecycle.frozen_proof(model,set(prepared['record']['decoder_parameters']))
    b.require(frozen == prepared['record']['validated_frozen_state']
        and meta['checkpoint_id'] == report['export']['checkpoint_id']
        and budget == report['export']['bit_budget'], 'Fresh frozen complement, identity or storage changed.')
    model.to('cuda:0').eval().requires_grad_(False)
    modules = b.selected_modules(model)
    warm = lifecycle.prewarm(modules)
    fixture = b.torch.load(run/'same_input_output.pt',map_location='cpu',weights_only=True)
    with b.torch.no_grad():
        logits = model(input_ids=fixture['input_ids'].to('cuda:0'),use_cache=False,logits_to_keep=1).logits.cpu()
    b.require(logits.dtype == fixture['logits'].dtype == b.torch.bfloat16 and b.torch.isfinite(logits).all().item()
        and b.torch.equal(logits,fixture['logits']), 'Independent fresh BF16 same-input output is not exact.')
    groups = b.checkpoint.inventories(model)
    frozen_names = [key for key in state if key not in prepared['record']['decoder_parameters']]
    audit = dict(schema='mixed_compact_Muon_short2_fresh_audit_v1',status='verified',physical_GPU=6,
        source_checkpoint_id=report['source_checkpoint_id'],checkpoint_id=meta['checkpoint_id'],
        serialized_tensors_compared=len(expected),fresh_strict_tensors_compared=len(actual),
        frozen_tensors_compared=frozen['tensors'],frozen_state=frozen,code_bank_count=sum('vq_weight' in key or '.codes_' in key for key in frozen_names),
        native_modules=len(groups['native']),compact_modules=len(groups['compact']),pending_dense=len(groups['pending_dense']),
        trainable_tensors=prepared['record']['trainable_tensors'],trainable_elements=prepared['record']['trainable_elements'],
        full_state_exact=True,same_complete_input_BF16_logits_exact=True,finite=True,
        input_tokens=int(fixture['input_ids'].shape[-1]),output_sha=b.core.tensor_record(logits),
        frozen_contract='All actual nondecoder persistent fields including code/length/protection/rotation; no fixed old G09 count',
        warmup=warm,bit_budget=budget,training_report=b.binding(run/'report.json'),
        prepared=b.binding(b.OUTPUT/'prepared.json'),export_meta=b.binding(run/'final_model/adaptive_model_meta.json'),
        scope='Full8B group0/1 M32+QA8 short2 lifecycle only; not32-step quality, fresh BF16 base, or dynamic/Rice GPU proof')
    b.core.write(run/'fresh_audit.json',audit)
    print(json.dumps(audit),flush=True)


if __name__ == '__main__':
    main()

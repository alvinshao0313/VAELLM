"""Verify the two actual canonical saves and their shared non-gate complement."""
import argparse
import os
from pathlib import Path

from bindings import s, validate_raw2_meta


def execute(output):
    import torch
    from experiments.adaptive_vector_rvq.common import digest
    s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU boundary verification only.')
    request = s.read(output / 'request.json')
    after_o, event = s.own_after_o(request)
    gate = s.read(output / 'gate_raw2.json')
    s.require(gate['status'] == 'raw2_completed' and gate['exit_code'] == 0
        and gate['parent_event'] == event and gate['recoveries'] == []
        and gate['request_sha256'] == s.sha(output / 'request.json'), 'Raw2 is not this execution\'s direct after_o child.')
    raw_event = gate['saved_boundaries'][-1]
    raw = Path(raw_event['path']).resolve(strict=True)
    s.require(raw.is_relative_to(output / 'gate_raw2/native') and raw.name == 'round_base', 'Unexpected canonical raw2 path.')
    meta = s.read(raw / 'checkpoint_meta.json')
    validate_raw2_meta(meta, request['base_model_path'])
    s.require(s.sha(raw / 'checkpoint_meta.json') == raw_event['metadata_sha256']
        and s.sha(raw / 'cat_runtime_state.pt') == raw_event['runtime_sha256']
        and s.stat(raw / meta['state_dict_file']) == raw_event['weight_stat'], 'Raw2 changed after its real native save.')
    before = s.read(after_o / 'checkpoint_meta.json')
    left = torch.load(after_o / before['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    right = torch.load(raw / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    gate_names = tuple(name + '.' for name in s.projection_names(('gate_proj',)))
    left_non_gate = {name: value for name, value in left.items() if not name.startswith(gate_names)}
    right_non_gate = {name: value for name, value in right.items() if not name.startswith(gate_names)}
    s.require(left_non_gate.keys() == right_non_gate.keys() and all(left_non_gate[name].dtype == value.dtype
        and left_non_gate[name].shape == value.shape and torch.equal(left_non_gate[name], value)
        for name, value in right_non_gate.items()), 'Raw2 changed the common after_o non-gate complement.')
    runtime_left = torch.load(after_o / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    runtime_right = torch.load(raw / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    s.require(runtime_left['runtime_identity'] == runtime_right['runtime_identity'] == gate['runtime_identity'],
        'Cross-process runtime identity changed.')
    prefix = s.read(output / 'prefix.json')
    if request['validation_only']:
        s.require(all(item.get('validation_only') and item.get('actual_updates') == 2
            and item.get('schedule_max_steps') == 500 and item.get('formal_parent_eligible') is False
            for item in meta['extra_meta']['distill_stage_history']), 'Every short recovery must be marked in raw2 history.')
        s.require(len(prefix['recoveries']) == 4 and all(item['actual_steps'] == 2
            and item['updates'][-1]['changed_decoder_elements'] > 0 for item in prefix['recoveries'])
            and any(1024 in item['input_shapes'][index] for item in prefix['recoveries']
                for index in range(len(item['input_shapes']))), 'Missing real short recovery updates or the max1024 resource condition.')
    result = dict(status='verified', validation_only=request['validation_only'],
        after_o_checkpoint_id=before['checkpoint_id'], raw2_checkpoint_id=meta['checkpoint_id'],
        non_gate_after_o_complement_exact=True, non_gate_tensor_count=len(left_non_gate),
        non_gate_tensor_digest=digest(left_non_gate), runtime_identity_equal=True,
        prefix_process=prefix['pid'], raw2_process=gate['pid'], gate_recovery_updates=0,
        actual_raw2_tensor_count=len(right), cuda_initialized=torch.cuda.is_initialized(),
        verification='Actual canonical tensor files and strict second-process native after_o load; no C1 consumer or quality claim.')
    s.write(output / 'boundary_verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.resolve(strict=True))

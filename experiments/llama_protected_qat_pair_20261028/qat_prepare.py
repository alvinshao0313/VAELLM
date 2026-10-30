"""CPU-bind actual Llama geometry, fixed-scale STE and same-a1 cached teacher targets."""
import ast
import os
from pathlib import Path
import sys
import time
import qat_common as c
from qat_controller import QController, math_source


def mathematical_check():
    torch = c.torch
    from litebsq.protected_channel_quant import decode_protected_channel_weight
    q = torch.tensor([-129., -128., -127.5, -1.5, -.5, 0., .5, 1.5, 126.5, 127., 128.], requires_grad=True)
    clipped = q.clamp(-128., 127.)
    hard = clipped + (clipped.round() - clipped).detach()
    expected = torch.tensor([-128., -128., -128., -2., 0., 0., 0., 2., 126., 127., 127.])
    scale = torch.tensor([.0078125], dtype=torch.bfloat16)
    rows = (hard.reshape(1, -1) * scale.float().unsqueeze(1)).bfloat16()
    payload = hard.detach().to(torch.int8).view(torch.uint8).reshape(1, -1)
    native = decode_protected_channel_weight(payload, scale, quant_format='int8', dtype=torch.bfloat16, device=torch.device('cpu'))
    grad = torch.autograd.grad(rows.float().sum(), q)[0]
    c.require(torch.equal(hard.detach(), expected) and torch.equal(rows.detach(), native)
        and torch.equal(grad, torch.tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]) * float(scale[0])),
        'Preserve actual tie-even/clamp/signed-UINT8/native-BF16-row/STE derivative.')
    c.require(QController._coordinates is __import__('qat_controller').math.ProtectionController._coordinates,
        'Use the existing verified fixed-scale signed-q coordinate math.')
    return {'clamp_round_signed_uint8_native_rows_and_gradient_exact': True,
        'hard_changes_required_for_short2': False, 'math_source': c.binding(math_source), 'whole_model_FW_BW': 0}


def main():
    started = time.monotonic(); c.check_runtime()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not c.torch.cuda.is_initialized()
        and not (c.HERE / 'prepared.json').exists(), 'Fresh CPU-only matched64 preparation required.')
    c.HERE.mkdir(parents=True, exist_ok=True)
    import qat_longbench, qat_evaluation, qat_nine, qat_run, qat_pipeline
    c.require(qat_longbench.runtime.n is qat_longbench.n, 'Bind actual native2.6 runtime interface before GPU execution.')
    previous = c.previous.prepared()
    data, original = c.inputs()
    archive, retained = c.retention.load_retention()
    targets = c.retention.load_targets()
    c.require(len(targets) == 896 and data['groups'] == previous['groups'], 'Reuse exact existing train-only groups/R signatures.')
    del targets
    cache_path = c.PREVIOUS / 'shared/retention_cache.json'; cache = c.read(cache_path)
    meta = c.read(c.SOURCE / 'checkpoint_meta.json')
    state = c.torch.load(c.SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    proof = c.state_proof(state)
    c.require(meta['checkpoint_id'] == c.SOURCE_ID and len(state) == 3619
        and proof == cache['source_original_state']
        and proof['sha256'] == '504009544f0f9c57d03d9b5c8b586ef4d27fb3b6b71c14a3b58c7e101643d92d'
        and cache['native_policy'] == previous['native_policy'] == c.protocol.model_policy(c.HF),
        'Cached R896 must match the complete actual same-a1 source state and native tokenizer/model policy.')
    entries = {name: c.tensor_record(value) for name, value in state.items()}
    decoder = {name: value for name, value in entries.items() if '._parallel_stage_decoder.' in name}
    c.require(decoder == previous['decoder_parameters'] and len(decoder) == 1344
        and sum(state[name].numel() for name in decoder) == 4078592, 'Actual Llama D inventory changed.')
    qparams, geometry = {}, {}
    from litebsq.protected_channel_quant import decode_protected_channel_weight
    from train_utils.checkpoint_bpw import audit_state_dict, _code_specs
    for spec in meta['converted_modules']:
        name = spec['name']; key = name + '.protected_input_qvalues'
        q, scale, index = (state[name + '.' + suffix] for suffix in
            ('protected_input_qvalues', 'protected_input_scales', 'protected_input_indices'))
        c.require(spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
            and q.dtype == c.torch.uint8 and q.ndim == 2 and q.is_contiguous() and q.shape[1] == spec['out_features']
            and scale.dtype == c.torch.bfloat16 and scale.shape == index.shape == (q.shape[0],)
            and index.dtype == c.torch.int64 and c.torch.isfinite(scale).all().item() and (scale > 0).all().item()
            and c.torch.unique(index).numel() == index.numel() and index.min() >= 0 and index.max() < spec['in_features'],
            'Wrong actual original Llama protection topology: ' + name)
        rows = (q.view(c.torch.int8).float() * scale.float().unsqueeze(1)).bfloat16()
        deployed = decode_protected_channel_weight(q, scale, quant_format='int8', dtype=c.torch.bfloat16, device=c.torch.device('cpu'))
        c.require(c.torch.equal(rows, deployed), 'Actual source signed-q/BF16 native rows differ: ' + name)
        qparams[key] = {'shape': list(q.shape), 'elements': q.numel(), 'source_tensor': entries[key]}
        codes = list(_code_specs(spec))
        c.require(len(codes) == spec['residual_stages'] == (4 if name.endswith('.gate_proj') else 2),
            'Preserve real Llama S4/S2 decoder geometry.')
        geometry[name] = {'stages': spec['residual_stages'], 'codebook_dims': spec['stage_codebook_dims'],
            'logical_banks': [code for _, code in codes], 'q_shape': list(q.shape)}
    qkeys = set(qparams); dkeys = set(decoder)
    frozen = {arm: c.state_proof(state, dkeys | (qkeys if arm == 'DQ' else set())) for arm in c.ARMS}
    c.require(len(qkeys) == 224 and sum(row['elements'] for row in qparams.values()) == 113246208
        and frozen['D'] == previous['frozen_state'] and frozen['D']['tensors'] == 2275
        and frozen['DQ']['tensors'] == 2051 and sum('.sparse_residual_' in name for name in state) == 576,
        'Actual D1344/q224/frozen2051/sparse576 inventory differs.')
    budget = audit_state_dict(meta, state, max_bpw=c.MAX_BPW, max_lora_rank=0)
    c.require(budget == previous['bit_budget'] and budget['budget_passed']
        and budget['user_linear_bpw'] == budget['packed_linear_bpw'] == c.BPW, 'Preserve actual native2.548370/rank0 ledger.')
    geometry_r = c.retention.shapes(archive)
    c.require(geometry_r == previous['R_batch_shapes'], 'R896 branch/token/padding geometry changed.')
    paths = set(c.CODE.glob('*.py')) | set(c.CODE.glob('*.sh'))
    paths.update(Path(value.__file__).resolve() for value in sys.modules.values()
        if getattr(value, '__file__', None) and str(value.__file__).startswith(str(c.R)) and str(value.__file__).endswith('.py'))
    paths.update(Path(path) for path in previous['code_sha256'])
    paths.update((math_source, c.previous.SPARSE / 'native_export/runtime.py'))
    for path in paths:
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    input_paths = {Path(item['path']) for item in previous['bindings'].values()}
    input_paths.update((c.PREVIOUS / 'prepared.json', cache_path, c.PREVIOUS / 'shared/retention_prepared.json'))
    input_paths.update(Path(cache['tasks'][task]['payload']['path']) for task in c.TASKS)
    record = {'schema': 'llama_protected_qat_matched64_v1', 'status': 'CPU_prepared_real_QAT_short2_pending',
        'source_checkpoint_id': c.SOURCE_ID, 'source_checkpoint': str(c.SOURCE), 'steps': 64, 'seed': c.SEED,
        'group_indices': list(range(32)) * 2, 'short2_group_indices': c.SHORT_GROUPS, 'groups': data['groups'],
        'dataset': original['dataset'], 'train_ids': data['train_ids'], 'decoder_parameters': decoder,
        'q_parameters': qparams, 'frozen_state': frozen, 'geometry': geometry, 'source_state': proof,
        'frozen_sparse_tensors': previous['frozen_sparse_tensors'], 'bit_budget': budget,
        'native_policy': previous['native_policy'], 'R_dataset': retained['dataset_binding'],
        'R_teacher_cache': c.binding(cache_path), 'R_cache_reused_same_source_exact': True,
        'R_batch_shapes': geometry_r, 'extra_short_retention_batches': previous['extra_short_retention_batches'],
        'short2_longest_QA_prompt': previous['short2_longest_QA_prompt'],
        'objective': previous['objective'], 'OPD': previous['OPD'],
        'optimization': {'AdamW_betas': [.9, .95], 'eps': 1e-8, 'weight_decay': 0., 'joint_clip': 1.5,
            'D_lr': 1e-6, 'q_lr': .03, 'schedule': '1-(step-1)/500', 'continuous64_no_reset_at32': True,
            'fresh_source_and_optimizer_each_arm': True, 'q_source_uint8_view_signed_int8': True,
            'q_clamp': [-128, 127], 'round': 'PyTorch_ties_to_even_STE', 'fixed_source_BF16_scale': True},
        'only_arm_variable': 'Train original224 protected q jointly with D; evolving student OPD trajectories can differ endogenously.',
        'budgets': {'training_seconds_per_arm': 43200, 'full_chain_seconds_per_arm': 86400,
            'automatic_retry': False, 'automatic_resume': False, 'fixed_endpoint_steps': 64},
        'full9': {**previous['full9'], 'all_eight_and21_for_both': True}, 'CPU_math': mathematical_check(),
        'code_sha256': {str(path): c.sha(path) for path in sorted(paths)},
        'bindings': {str(path): c.binding(path) for path in sorted(input_paths)},
        'CPU_entry_imports': ['qat_longbench', 'qat_evaluation', 'qat_nine', 'qat_run', 'qat_pipeline'],
        'CUDA_initialized': c.torch.cuda.is_initialized(), 'elapsed_seconds': time.monotonic() - started}
    c.require(not c.torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    c.write(c.HERE / 'prepared.json', record)
    print({key: record[key] for key in ('status', 'source_state', 'frozen_state', 'R_cache_reused_same_source_exact', 'elapsed_seconds')}, flush=True)


if __name__ == '__main__': main()

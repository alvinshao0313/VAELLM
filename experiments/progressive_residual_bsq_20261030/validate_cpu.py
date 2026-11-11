"""CPU verification of real S4 state, observation RNG, fixed dependency and LB wiring."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from bindings import HERE, HF, PYTHON, boundary, s, source_pins
from fit import tensor_digest


def pins_digest(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def execute(short_report):
    import torch
    s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
        'CPU validation must not initialize CUDA.')
    proof = s.read(short_report)
    s.require(proof['status'] == 'failed' and proof['exit_code'] == 1,
        'Bind the actual failed original short; never rewrite it as completed.')
    output = short_report.parent
    fitted = s.read(output / 'fit.json')
    evaluated = s.read(output / 'evaluation.json')
    s.require(len(fitted['boundaries']) == 7 and len(fitted['groups']) == 7
        and all(row['status'] == 'fitted_and_full_encoded' and row['optimizer_initializations'] == 4
            and row['complete_noncurrent_complement_frozen_exact']
            and len(row['BF16_deployment_prefixes']) == 36 for row in fitted['groups'])
        and 'OutOfMemoryError' in fitted['error'] and fitted['recoveries'] == 0,
        'Require all genuine S4 payloads and their real post-save OOM; no completed inference is inferred.')
    actual = boundary(fitted['boundaries'][-1]['path'])
    s.require(actual['checkpoint_id'] == fitted['boundaries'][-1]['checkpoint_id'], 'Saved S4 identity changed.')
    from transformers import Qwen3ForCausalLM
    from rotation.model_utils import temporarily_disable_torch_weight_init
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
    with temporarily_disable_torch_weight_init():
        model = Qwen3ForCausalLM.from_pretrained(str(HF), torch_dtype='auto',
            attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=True)
    model.seqlen = 2048
    model, meta, loaded = load_v6_full_checkpoint_into_model(model, actual['path'], map_location='cpu', strict=True)
    state = torch.load(Path(actual['path']) / meta['state_dict_file'], map_location='cpu', mmap=True, weights_only=True)
    restored = model.state_dict()
    s.require(not loaded.missing_keys and not loaded.unexpected_keys and restored.keys() == state.keys()
        and all(value.dtype == state[name].dtype and value.shape == state[name].shape
            and torch.equal(value, state[name]) for name, value in restored.items()),
        'Fresh CPU S4 full persistent state differs from the actual saved tensor file.')
    fields = len(state)
    decoder = model.get_submodule('model.layers.0.self_attn.q_proj')
    before_persistent = tensor_digest(decoder.state_dict())
    scratch = decoder._get_parallel_stage_grouped_vq(dtype=torch.bfloat16, device=torch.device('cpu'))
    s.require(scratch.device.type == 'cpu' and scratch.dtype == torch.bfloat16, 'Exercise actual grouped stage scratch.')
    decoder._clear_parallel_stage_decode_runtime_cache()
    s.require(decoder._parallel_stage_grouped_vq_runtime is None
        and tensor_digest(decoder.state_dict()) == before_persistent, 'Scratch cleanup changed persistent S4 state.')
    del scratch
    from litebsq.autoencoder import MultiLayerVAE, pack_decoders
    import recipe
    _, _, _, vae, _, _, _, _ = recipe.parse(output / 'CPU_validation_unused_output')
    from train_utils.cat_train_pipeline import _clone_namespace
    from litebsq.vae_args import apply_autoencoder_arch_defaults
    stage_args = _clone_namespace(vae, parallel_layers=36, codebook_dim=32, codebook_bits=32,
        base_ch=128, num_res_blocks=0, norm_type='layer', activation_type='swish',
        decoder_type='symmetric', decoder_base_ch=None, decoder_num_res_blocks=None)
    apply_autoencoder_arch_defaults(stage_args)
    torch.manual_seed(31)
    original_rng = torch.get_rng_state().clone()
    reference = MultiLayerVAE(stage_args)
    expected_initial = tensor_digest(reference.state_dict())
    del reference
    torch.set_rng_state(original_rng)
    with torch.random.fork_rng(devices=[]):
        for depth in (1, 2, 3):
            stages = [decoder.get_stage_part_decoder(index) for index in range(depth)]
            packed = pack_decoders(stages)
            del packed, stages
        s.require(not torch.equal(torch.get_rng_state(), original_rng),
            'The real native prefix packing did not exercise its known random constructor path.')
    s.require(torch.equal(torch.get_rng_state(), original_rng), 'Observation CPU RNG was not restored.')
    next_vae = MultiLayerVAE(stage_args)
    s.require(tensor_digest(next_vae.state_dict()) == expected_initial,
        'The next genuine M36 VAE initialization changed after observation.')
    del next_vae, model, restored, state
    gc.collect()
    from run import bind_existing_control
    parent = bind_existing_control()
    from experiments.cmcc_runtime_20261030.sequence import proc_start, wait_identity
    exit_path = output / 'cpu_pidfd_child.exit'
    shell = 'exit_path=$1\npython_bin=$2\n"$python_bin" -c "import time; time.sleep(0.1)"\ncode=$?\nprintf "%s\\n" "$code" > "$exit_path"\nexit "$code"\n'
    child = subprocess.Popen(['/bin/bash', '-c', shell, 's4-real-pidfd-proof', str(exit_path), str(PYTHON)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ticks = proc_start(child.pid)
    s.require(ticks is not None, 'Actual short proof shell exited before binding.')
    observed = wait_identity(child.pid, ticks)
    s.require(child.wait() == 0 and exit_path.read_text().strip() == '0',
        'Real shell pidfd completion did not precede successful exit publication.')
    exit_path.unlink()
    current = source_pins()
    request = s.read(output / 'request.json')
    from evaluation import native_config
    request_now = {**request, 'source_sha256': current}
    config = native_config(request_now, actual['path'], short=False)
    from experiments.longbench_v1.models import check_config
    check_config(config)
    config_path = output / 'CPU_formal_LB_config.json'
    s.write(config_path, config)
    from longbench import HELPERS, PARTITIONS, TERMINAL, make_plan
    plan = make_plan(request_now, output / 'request.json', actual, config, config_path)
    assigned = [task for unit in plan['execution_units'] for task in unit['datasets']]
    from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS
    s.require(len(assigned) == len(set(assigned)) == 21 and set(assigned) == set(DATASETS)
        and sum(EXPECTED_COUNTS[name] for name in assigned) == 4750
        and [unit['gpu'] for unit in plan['execution_units']] == ['7', '6']
        and config['model']['checkpoint_id'] == actual['checkpoint_id'],
        'The unchanged canonical two-card LB assignment or shared UUID differs.')
    old = proof['source_sha256']
    delta = {path: dict(GPU_verified_sha256=old.get(path), current_sha256=current.get(path))
        for path in sorted(old.keys() | current.keys()) if old.get(path) != current.get(path)}
    allowed = {str(HERE / name) for name in ('bindings.py', 'evaluation.py', 'fit.py', 'run.py', 'longbench.py', 'prewarm.py', 'readout.py', 'validate_cpu.py')}
    allowed.update(map(str, (HELPERS, PARTITIONS, TERMINAL / 'bind_terminal.py', TERMINAL / 'merge_full21.py')))
    s.require(set(delta) <= allowed and str(HERE / 'recipe.py') not in delta
        and all(path in current for path in old if path not in allowed), 'A shared GPU runtime or recipe changed.')
    report = dict(status='verified_S4_CPU_fixes_pending_GPU_prewarm', exit_code=0,
        original_GPU_short_report=str(short_report), original_GPU_report_sha256=s.sha(short_report),
        original_GPU_request_sha256=s.sha(output / 'request.json'), current_source_pins_digest=pins_digest(current),
        source_delta=delta, unchanged_source_count=len(current.keys() & old.keys()) - len(set(delta) & current.keys() & old.keys()),
        full_native_S4_state_exact=True, persistent_fields=fields, checkpoint=actual,
        runtime_scratch_CPU_release_preserves_persistent_state_exact=True, original_short_exit_code=1,
        CPU_model_backend='Actual HF Qwen3 SDPA construction, native v6 strict tensor loading; no forward/numeric claim.',
        original_fitter_live_logits_available=False, GPU_prewarm_readout_validation_pending=True,
        observation_CPU_RNG_preserved=True, next_real_M36_VAE_initialization_exact=True,
        next_VAE_initial_digest=expected_initial, cuda_initialized=torch.cuda.is_initialized(),
        fixed_existing_dependency=parent, real_shell_pidfd_publication=observed,
        dependency_state='Existing controller/shell identity bound; their future success is not asserted.',
        LongBench=dict(shared_config=str(config_path), shared_config_sha256=s.sha(config_path),
            checkpoint_id=actual['checkpoint_id'], partition_tasks=[unit['datasets'] for unit in plan['execution_units']],
            partition_samples=[unit['expected_samples'] for unit in plan['execution_units']],
            complete_expected_tasks=21, complete_expected_samples=4750, generation_and_scoring_unchanged=True),
        reuse_boundary='Original GPU mathematical/consumer evidence retains original hashes. New RNG isolation, fixed CPU '
            'dependency wait, terminal wiring and same-runner partition branches have CPU evidence; no new full GPU claim.')
    s.write(output / 'formal_cpu_validation.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'persistent_fields', 'next_real_M36_VAE_initialization_exact', 'cuda_initialized')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-report', required=True, type=Path)
    args = parser.parse_args()
    execute(args.short_report.resolve(strict=True))


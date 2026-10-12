"""One explicitly bound run; no scheduler, retries, benchmark tuning or full checkpoint copy."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed
from .data import ARMS, assets, check, native_inputs, read
from .observe import evaluate, persist
from .training import fit_arm, initial_states


def run(path):
    record = read(path); output = Path(record['output_dir']); work = output / 'run'
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not work.exists()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and sys.executable == record['python_executable']
            and not record['queued'] and not record['automatic_retry'], 'Use one explicit bound launch from the prepared cwd with fresh outputs.')
    require(all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
            'Prepared native numerical/offline environment differs.')
    check(record); work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); code = 1
    try:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one GPU is required.')
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False
        deadline = record['started_unix'] + record['timeout_seconds']
        steps = record.get('steps_per_stage', 256)
        diagnostic_steps = tuple(record.get('diagnostic_steps', ()))
        data = assets(); material, args, argv = native_inputs(data, steps=steps); starts = initial_states(args)
        require(argv == record['native_argument_recipe'] and vars(args) == record['resolved_vae_args']
                and data['spec'] == record['native_spec'], 'Native recipe or actual geometry changed.')
        require([{k: tensor_record(v) for k, v in state.items()} for state in starts] == record['initial_states'], 'Fresh initial states differ.')
        require({d: {k: tensor_record(v) for k, v in b.items()} for d, b in data['batches'].items()}
                == record['token_batches'], 'Fixed data replay differs.')
        require({k: tensor_record(getattr(material.split_metas[0], k)) for k in record['common_fresh_hf_protection']}
                == record['common_fresh_hf_protection'], 'Common fresh protection payload differs.')
        endpoints, fits, saves = {}, {}, {}
        for arm in ARMS:
            torch.cuda.reset_peak_memory_stats()
            payload, fits[arm] = fit_arm(arm, data, material, args, starts, work, deadline=deadline,
                steps=steps, diagnostic_steps=diagnostic_steps)
            fits[arm]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            fits[arm]['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            endpoints[arm], saves[arm] = persist(payload, data, work / arm)
            write_json(work / (arm + '.json'), {'fit': fits[arm], 'save': saves[arm]})
            del payload; gc.collect(); torch.cuda.empty_cache()
        for stage in range(2):
            a, b = [fits[arm]['stages'][stage] for arm in ARMS]
            require(a['initial_state_sha256'] == b['initial_state_sha256'] and a['token_order'] == b['token_order'],
                    'Two arms did not receive identical fresh initialization and document order.')
        for key in record['common_fresh_hf_protection']:
            require(torch.equal(endpoints[ARMS[0]][key], endpoints[ARMS[1]][key]), 'Paired protection differs.')
        del material, starts; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        observation = evaluate(data, endpoints, record['source_native_policy'], deadline=deadline)
        expected, actual = record['source_bit_budget'], observation['budget']
        require(actual['user_linear_bits'] == expected['user_linear_bits'], 'Native union storage bits changed.')
        report = {'status': 'completed', 'protocol': record['protocol'], 'fits': fits, 'saves': saves,
            'observation': observation, 'elapsed_seconds': time.time() - record['started_unix'],
            'observation_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'observation_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'source_checkpoint_id': record['source_checkpoint_id']}
        if diagnostic_steps:
            report['deployment_sse_comparison'] = {arm: {domain: {
                'unfused_fp32_sse': values['sse'],
                'native_bf16_group8_sse': observation['arms'][arm]['local_fixed_prefix'][domain]['sse'],
                'native_minus_unfused_sse': observation['arms'][arm]['local_fixed_prefix'][domain]['sse'] - values['sse'],
            } for domain, values in fits[arm]['unfused_fp32_local_fixed_prefix'].items()} for arm in ARMS}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=repr(error)); traceback.print_exc(); raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    run(parser.parse_args().launch_record)

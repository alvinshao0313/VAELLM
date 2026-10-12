"""Run four native FineWeb updates; the dispatcher exports after this process exits."""
import argparse
from pathlib import Path

import torch

from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import write_json
from experiments.long_context_recovery.contract import require
from experiments.llama_s4_lm_validation.config import check_inputs, configuration, read
from experiments.llama_s4_lm_validation.observations import GateStageObservations
from train_utils.runtime_snapshot import to_runtime_jsonable


def train(path):
    record = read(path)
    output = Path(record['output_dir'])
    require(not (output / 'training.json').exists() and torch.cuda.device_count() == 1,
            'One bound GPU and a fresh output are required.')
    check_inputs(record)
    cfg, hf, args, names, architecture = configuration(record)
    observer = GateStageObservations(names, architecture)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    report = {'status': 'running', 'source_checkpoint_id': record['source_checkpoint_id'],
              'expected_inventory': record['expected_inventory'], 'diagnostic_not_quality': True}
    try:
        result = to_runtime_jsonable(run(cfg, hf, args, extra_callbacks=(observer,)))
        observed = observer.diagnostic
        require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and [x['step'] for x in observed['optimizer_steps']] == [1, 2, 3, 4]
                and [x['step'] for x in observed['native_saves']] == [2, 4]
                and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                'Real B4 update/save2/4 observations are incomplete.')
        inv = observed['trainable_inventory']
        require(inv['decoder_tensors'] == 1344 and inv['decoder_elements'] == 4078592
                and inv['score_elements'] == 0 and inv['selected_projection_code_banks'] == 512,
                'Only all1344 native decoder leaves may train.')
        for key in ('gate_gradients', 'gate_updates'):
            require([x['step'] for x in observed[key]] == [1, 2, 3, 4]
                    and all(len(x['stages']) == 128 for x in observed[key]), 'Incomplete S4 gate stage observations.')
        check_inputs(record)
        report.update(status='trained', training_result=result)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(observations=observer.diagnostic, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            resource_coverage='Four actual native dynamic B4/max1024 batches; not a full-shard worst-case bound.')
        write_json(output / 'training.json', report)
        observer.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    train(parser.parse_args().launch_record)

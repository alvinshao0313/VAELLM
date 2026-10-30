"""Run only the first fixed G09 alignment stage through the mature native runtime."""
import os
from pathlib import Path
import sys
import time
import traceback

import common as c
from callback import AlignedQKCallback, SAVE_STEPS
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed


def main():
    c.require(Path(sys.executable).resolve() == c.PYTHON.resolve() and Path.cwd() == c.SNAPSHOT
              and os.environ.get('CUDA_VISIBLE_DEVICES') == str(c.GPU), 'Use caller-activated bitvae and the assigned profile GPU.')
    record = c.read_prepared()
    c.require(all(os.environ.get(key) == value for key, value in record['environment'].items())
              and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
              'Keep the old native training environment.')
    path = c.OUTPUT / 'execution.json'
    c.require(not path.exists() and not list(c.OUTPUT.glob('home_*')), 'One fresh run; no automatic retry.')
    cfg, hf, args = c.configuration(record)
    execution = dict(status='running', argv=sys.argv, pid=os.getpid(), physical_gpu=c.GPU, profile=c.PROFILE,
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        prepared_sha256=c.sha256(c.OUTPUT / 'prepared.json'), source_checkpoint_id=c.SOURCE_ID,
        started_unix=time.time(), automatic_retry=False)
    c.write_json(path, execution)
    code = 1
    try:
        callback = AlignedQKCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=c.OUTPUT)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(args.seed)
        result = c.to_runtime_jsonable(run(cfg, hf, args, extra_callbacks=(callback,)))
        c.require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == c.SOURCE_ID
                  and callback.completed_steps == SAVE_STEPS and result['saved_model_dir'], 'Require actual2000 and every native save.')
        c.check_inputs(record)
        c.write_json(c.OUTPUT / 'training.json', dict(status='completed_native_pending_audit',
            training_result=result, source_checkpoint_id=c.SOURCE_ID, completed_steps=callback.completed_steps,
            actual_scope_sha256=c.sha256(c.OUTPUT / 'actual_scope.json'), next_stage_started=False,
            boundary='Native training complete; independent terminal frozen-state/native-source audit remains required.'))
        execution.update(status='completed_native_pending_audit', training_result=result)
        code = 0
    except BaseException as error:
        execution.update(status='failed', error=str(error), traceback=traceback.format_exc())
        print(execution['traceback'], flush=True)
    finally:
        execution.update(exit_code=code, finished_unix=time.time())
        c.write_json(path, execution)
        (c.OUTPUT / 'run.exit').write_text(str(code) + '\n', encoding='utf-8')
        if c.torch.distributed.is_available() and c.torch.distributed.is_initialized():
            c.torch.distributed.destroy_process_group()
    return code


if __name__ == '__main__':
    sys.exit(main())

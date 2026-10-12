"""Sequential same-GPU train, native step4 export, and existing frozen-state audit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.step_export_callback import run_logged_stage


def require_idle(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader'], text=True)
    require(all(line.split(',')[-1].strip() != uuid for line in active.splitlines()),
            'Assigned GPU still has a compute process; do not overlap training/export.')


def run(path):
    path = Path(path).resolve(); record = read(path); output = Path(record['output_dir']); code = 1
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path.cwd() == Path(record['cwd'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and not record['queued'] and not record['automatic_retry'], 'Use the explicitly bound GPU and bitvae.')
        require(all(os.environ.get(key) == value for key,value in record['environment'].items())
                and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC','NVIDIA_TF32_OVERRIDE')),
                'Keep the exact native training environment.')
        require(not list(output.glob('home_*')) and not (output/'training.json').exists()
                and not (output/'step_0004').exists(), 'Use a fresh run without overwrite or resume.')
        check_inputs(record); require_idle(record['gpu'])
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
            started_unix=time.time(), stages={})
        write_json(path,record)
        with (output/'training.log').open('x') as log:
            child = subprocess.Popen(record['argv'], cwd=record['cwd'], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record.update(training_pid=child.pid,
                training_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
            write_json(path,record); status=child.wait()
        (output/'training.exit').write_text(str(status)+'\n')
        record.update(training_exit_code=status,training_finished_unix=time.time())
        write_json(path,record)
        require(status == 0, 'Native full-KL short training failed; no automatic retry.')
        trained = read(output/'training.json'); result=trained['training_result']
        require(trained['status']=='trained' and result['global_step']==4
                and result['round_base_checkpoint_id']==record['source_checkpoint_id'], 'Wrong native completion.')
        require_idle(record['gpu'])
        checkpoint = Path(result['run_output_dir'])/'trainer_state/checkpoint-4'
        step=output/'step_0004'; step.mkdir()
        commands=[('export',[sys.executable,'-m','experiments.w25_eval.subset_export',
            '--checkpoint_dir',str(checkpoint),'--output_dir',str(step/'final_model')]),
            ('audit',[sys.executable,'-m','experiments.w25_eval.subset_audit',
            '--source_dir',record['source_checkpoint'],'--checkpoint_dir',str(step/'final_model'),
            '--output',str(step/'audit.json'),'--training_checkpoint_dir',str(checkpoint)])]
        for name,command in commands:
            record.update(status=name)
            record['stages'][name]={'argv':command,'started_unix':time.time()};write_json(path,record)
            status=run_logged_stage(command,name=name,output_dir=step,env=os.environ.copy(),allowed_exit_codes=(0,))
            record['stages'][name].update(exit_code=status,finished_unix=time.time());write_json(path,record)
        audit,exported=read(step/'audit.json'),read(step/'final_model/export_record.json')
        expected=record['expected_inventory']
        require(audit['status']=='verified' and audit['source_checkpoint_id']==record['source_checkpoint_id']
                and audit['selected_global_step']==4 and audit['selected_module_count']==expected['selected_projections']
                and audit['selected_decoder_tensor_count']==expected['selected_decoder_tensors']
                and audit['frozen_tensors_compared']==expected['frozen_tensors']
                and audit['frozen_code_bank_count']==audit['code_bank_count']==expected['all_code_banks']
                and audit['serialized_tensors_compared']==audit['reloaded_tensors_compared']==expected['serialized_tensors']
                and audit['changed_code_bits']==0 and audit['selected_decoder_matches_fp32_cast_exactly']
                and audit['fresh_strict_reload_state_exact'] and audit['rank_zero'], 'The native step4 audit failed.')
        require(exported['source_global_step']==4 and exported['source_max_steps']==2000
                and exported['restored_decoder_tensors']==expected['selected_decoder_tensors']
                and exported['optimizer_steps_executed']==0 and exported['sparse_bit_restore'] is None,
                'Independent export changed the native selected state or executed updates.')
        from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
        native,independent=_load_state(Path(result['saved_model_dir'])),_load_state(step/'final_model')
        compared=_assert_state_equal(native,independent,'Native final versus independently restored full-KL step4')
        require(compared==expected['serialized_tensors'], 'Native final tensor inventory differs.')
        del native,independent
        check_inputs(record)
        report={'status':'verified','source_checkpoint_id':record['source_checkpoint_id'],
            'checkpoint_id':audit['checkpoint_id'],'training_report_sha256':sha256(output/'training.json'),
            'audit_sha256':sha256(step/'audit.json'),'export_record_sha256':sha256(step/'final_model/export_record.json'),
            'source_pin_count':len(record['source_sha256']),'input_pin_count':len(record['input_evidence_sha256']),
            'native_vs_step4_tensors_compared':compared,'bit_budget':audit['bit_budget'],
            'training_peak_allocated_bytes':trained['peak_allocated_bytes'],
            'training_peak_reserved_bytes':trained['peak_reserved_bytes'],
            'export_peak_allocated_bytes':exported['peak_cuda_allocated_bytes'],
            'export_peak_reserved_bytes':exported['peak_cuda_reserved_bytes'],
            'diagnostic_not_quality':True,'full8_and_longbench':'not_evaluated',
            'resource_coverage':trained['resource_coverage'],'same_gpu_sequential_training_export':True}
        write_json(output/'verification.json',report)
        record.update(status='verified',verification_sha256=sha256(output/'verification.json'));code=0
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        print(record['traceback'],flush=True)
    finally:
        record.update(exit_code=code,finished_unix=time.time());write_json(path,record)
        (output/'run.exit').write_text(str(code)+'\n')
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record',type=Path,required=True)
    sys.exit(run(parser.parse_args().launch_record))

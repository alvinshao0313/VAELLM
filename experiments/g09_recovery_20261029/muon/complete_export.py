"""Close the actual short2 export continuation while preserving its initial failure."""
import json
import os
from pathlib import Path
import sys
import time

import bindings as b


def main():
    prepared,_ = b.read_prepared()
    root = b.OUTPUT
    report_path = root / 'short2/report.json'
    audit_path = root / 'short2/native_audit.json'
    report,audit = b.read(report_path),b.read(audit_path)
    b.require((root / 'short2_initial.exit').read_text().strip() == '1'
        and (root / 'short2/run.exit').read_text().strip() == '0'
        and (root / 'short2_export_completion.exit').read_text().strip() == '0',
        'Require the real initial failure, successful training, and successful resumed exporter.')
    b.require(report['status'] == 'G09_Muon_nativegreedy_OPD_short2_complete'
        and report['source_checkpoint_id'] == audit['source_checkpoint_id'] == b.SOURCE_ID
        and report['prepared_sha256'] == audit['prepared_sha256'] == b.sha256(root / 'prepared.json')
        and report['steps'] == 2 and report['source_restored_exact']
        and report['saved_decoder_matches_final_FP32_cast_exact']
        and b.sha256(report['endpoint']['path']) == report['endpoint']['sha256'],
        'Retain the exact original valid training endpoint and preparation.')
    b.require(audit['schema'] == 'G09_Muon_online_OPD_native_audit_v1' and audit['status'] == 'verified'
        and audit['phase'] == 'short2' and audit['fresh_strict_reload_state_exact']
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 1911 and audit['code_bank_count'] == 504
        and audit['endpoint_report'] == b.binding(report_path), 'Native short2 audit has not actually completed.')
    directory = Path(audit['checkpoint_dir'])
    b.require(audit['checkpoint_meta_sha256'] == b.sha256(directory / 'checkpoint_meta.json'),
        'Resumed export checkpoint metadata changed.')
    b.direct.check_stat(audit['checkpoint_weight_stat'])
    record = dict(schema='G09_Muon_short2_export_completion_v1',
        status='valid_training_plus_visible_CUDA_native_export_verified',
        original_pipeline_exit_code=1,original_pipeline_exit=b.binding(root / 'short2_initial.exit'),
        original_pipeline_command=['bash',str(b.HERE / 'run_short.sh')],
        original_tmux='G09_Muon_short2_GPU7_20261029',original_training_worker_pid=report['pid'],
        original_worker_argv=report['argv'],original_core_log=b.binding(root / 'short2.log'),
        training_exit_code=0,resumed_export_exit_code=0,
        resumed_export_log=b.binding(root / 'short2_export_completion.log'),
        resumed_export_exit=b.binding(root / 'short2_export_completion.exit'),
        resumed_export_command=['/home/star/miniconda3/envs/bitvae/bin/python','-B','-u',str(b.HERE / 'export.py'),
            '--report',str(report_path),'--output-dir',str(directory)],
        resumed_export_CUDA_VISIBLE_DEVICES='7',native_audit=b.binding(audit_path),
        checkpoint_id=audit['checkpoint_id'],prepared_sha256=report['prepared_sha256'],
        training_report=b.binding(report_path),endpoint_sha256=report['endpoint']['sha256'],
        source_and_prepared_unchanged=True,training_repeated=False,
        initial_failure='FA2 construction rejected hidden CUDA despite CPU map_location; training and endpoint were valid.',
        finalizer=dict(pid=os.getpid(),argv=sys.argv,python=sys.executable,source=b.binding(Path(__file__)),
            checked_at_unix=time.time(),exit_code=0),
        canonical_short2_exit_semantics='This validated continuation completion; original run_short.sh invocation still exited1.')
    b.core.write(root / 'short2/completion.json',record)
    (root / 'short2.exit').write_text('0\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],finalizer_pid=os.getpid(),
        native_audit_sha256=record['native_audit']['sha256'],canonical_completion_exit=0)),flush=True)


if __name__ == '__main__':
    main()

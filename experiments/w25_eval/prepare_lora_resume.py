"""Copy the preserved native1200 state into isolated diagnostic/formal segments."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

from compressed_e2e_fintuning.args import parse_args
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.lora_state import inspect_step
from experiments.w25_eval.resume_lora_only import validate_resume

REPO = Path(__file__).resolve().parents[2]
FILES = ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt',
         'optimizer.pt', 'scheduler.pt', 'rng_state.pth')


def prepare(original_root, diagnostic_root, formal_root):
    original_root, diagnostic_root, formal_root = (Path(path).resolve() for path in
        (original_root, diagnostic_root, formal_root))
    old = json.loads((original_root / 'launch.json').read_text())
    callback = json.loads((original_root / 'step_1200/step_export.json').read_text())
    _require(old['status'] == callback['status'] == 'failed' and callback['global_step'] == 1200,
             'Only recover the recorded interrupted native1200 trajectory.')
    checkpoint = Path(callback['source_checkpoint'])
    meta, state, snapshot, source, _base, _architecture = inspect_step(checkpoint)
    _require(state['global_step'] == 1200 and state['max_steps'] == 2000, 'Wrong resume boundary/horizon.')
    pins = {str((REPO / path).resolve()): value for path, value in old['source_sha256'].items()}
    for name in ('prepare_lora_resume.py', 'resume_lora_only.py', 'validate_lora_resume.py',
                 'resume_head_only.py', 'validate_head_resume.py'):
        path = Path(__file__).parent / name
        pins[str(path)] = sha256(path)
    check_hashes(pins)
    protected_stats = [{key: row[key] for key in ('path', 'size', 'mtime_ns', 'inode')}
        for row in [old['source_weight_stat'], *old['input_data_stat'], *old['hf_source']['weight_files']]]
    for row in protected_stats:
        check_stat(row)
    evidence = {str(checkpoint / name): sha256(checkpoint / name) for name in FILES}
    evidence[str(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')] = sha256(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    evidence[str(original_root / 'launch.json')] = sha256(original_root / 'launch.json')
    evidence[str(original_root / 'step_1200/step_export.json')] = sha256(original_root / 'step_1200/step_export.json')
    optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=True, map_location='cpu', mmap=True)
    scheduler = torch.load(checkpoint / 'scheduler.pt', weights_only=True, map_location='cpu')
    mutable = torch.load(checkpoint / 'training_model_state.pt', weights_only=True, map_location='cpu', mmap=True)
    _require(len(mutable) == len(optimizer['state']) == 144
             and all(value.dtype == torch.float32 for value in mutable.values())
             and all(int(value['step']) == 1200 and value['exp_avg'].dtype == value['exp_avg_sq'].dtype == torch.float32
                     for value in optimizer['state'].values())
             and scheduler['last_epoch'] == 1200 and scheduler['_step_count'] == 1201,
             'Missing exact native FP32 LoRA/Adam/scheduler state.')
    outputs = []
    for root, end_step, diagnostic in ((diagnostic_root, 1202, True), (formal_root, 2000, False)):
        _require(not root.exists(), 'Use a new independent continuation output root.')
        copied = root / f'segment_1200_{end_step}/trainer_state/checkpoint-1200'
        copied.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(checkpoint / name, copied / name)
            _require(sha256(copied / name) == evidence[str(checkpoint / name)], 'Checkpoint copy changed native bytes.')
        snap_path = copied.parent.parent / 'normalized_e2e_runtime_args.json'
        shutil.copy2(checkpoint.parent.parent / snap_path.name, snap_path)
        _require(sha256(snap_path) == evidence[str(checkpoint.parent.parent / snap_path.name)], 'Original normalized snapshot changed.')
        argv = list(old['training_argv'])
        argv[argv.index('--run_root_dir') + 1] = str(root)
        argv += ['--resume_from_checkpoint', str(copied)]
        cfg, hf, args = parse_args(argv)
        validate_resume(cfg, hf, args, end_step, diagnostic=diagnostic)
        full_argv = ['--end_step', str(end_step), '--export_cuda_visible_devices', '4', *argv]
        env = {**old['launch_environment'], 'CUDA_VISIBLE_DEVICES': '6', 'PYTHONPATH': str(REPO),
               'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        record = {'status': 'prepared_not_scheduled', 'diagnostic': diagnostic, 'gpu': 6, 'export_gpu': 4,
            'original_failed_root': str(original_root), 'output_dir': str(root), 'cwd': str(REPO),
            'python_executable': sys.executable, 'source_checkpoint': str(source),
            'source_checkpoint_id': meta['round_base_checkpoint_id'], 'resume_checkpoint_id': meta['checkpoint_id'],
            'original_resume_checkpoint': str(checkpoint), 'copied_resume_checkpoint': str(copied),
            'source_sha256': pins, 'protected_original_sha256': evidence,
            'copied_checkpoint_sha256': {str(copied / name): evidence[str(checkpoint / name)] for name in FILES},
            'copied_original_snapshot_sha256': sha256(snap_path),
            'original_input_evidence_sha256': old['input_evidence_sha256'],
            'protected_weight_data_stats': protected_stats,
            'hf_metadata_sha256': old['hf_source']['metadata_sha256'], 'training_argv': full_argv, 'environment': env,
            'data_reference': str(diagnostic_root / 'data_reference.json'),
            'start_step': 1200, 'end_step': end_step, 'max_steps': 2000, 'warmup_steps': 20, 'save_steps': 400,
            'saved_scheduler': scheduler, 'native_fp32_lora_tensor_count': 144,
            'original_worker_count': meta['immutable_resume_contract']['dataloader']['num_workers'],
            'source_configuration_identical': True, 'scientific_change': None,
            'changes': ['Native resume_from_checkpoint points to a byte-identical isolated copy',
                        'Output root and private absolute segment endpoint', 'Three offline flags for training and evaluation subprocesses'],
            'diagnostic_proof': str(diagnostic_root / 'verification.json'),
            'automatic_retry': False, 'queued': False, 'formal_approval_required': not diagnostic}
        write_json(root / 'preparation.json', record)
        outputs.append({'path': str(root / 'preparation.json'), 'sha256': sha256(root / 'preparation.json')})
    check_hashes(evidence)
    print(json.dumps({'status': 'prepared', 'native1200_identity': meta['checkpoint_id'], 'outputs': outputs}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original_root', 'diagnostic_root', 'formal_root'):
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(**vars(parser.parse_args()))
